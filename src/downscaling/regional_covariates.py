"""
regional_covariates.py
=======================
Stage C: Localized Generative Downscaling — "Local Adapters" (architecture
doc Sec. 3-C)

Purpose
-------
Regional covariates (topography/DEM, land use, urban canopy/impervious
surface, coastline mask) are static, high-resolution rasters that condition
the downscaling model so the *same* coarse global forecast produces
different, terrain-aware local output for a mountain valley vs. an
adjacent urban core.

This module builds a fixed, deterministic conditioning tensor stack for a
named region, resampled to the target high-resolution grid. In production
these would be loaded once from real DEM/land-use raster sources (e.g.
SRTM, NLCD) and cached; here they're generated deterministically from a
seed so the covariate stack is reproducible and the downscaler's
terrain-awareness can be unit-tested (e.g. "does the model behave
differently over water vs. urban core?").

# SWAP_POINT: replace `_synthesize_*` with real raster loads (rasterio /
# rioxarray) against actual DEM and land-use datasets, reprojected to the
# target grid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RegionalCovariates:
    """Stacked static covariates for a region, all on the same grid."""

    region_id: str
    shape: tuple  # (h, w)
    elevation_m: np.ndarray
    land_use_urban_fraction: np.ndarray  # 0..1
    impervious_fraction: np.ndarray      # 0..1
    coastline_mask: np.ndarray           # 1.0 = water, 0.0 = land

    def as_stack(self) -> np.ndarray:
        """Return a (4, H, W) array suitable for concatenation onto a
        model's conditioning channels."""
        return np.stack(
            [self.elevation_m, self.land_use_urban_fraction, self.impervious_fraction, self.coastline_mask],
            axis=0,
        )


def build_regional_covariates(region_id: str, shape: tuple = (256, 256)) -> RegionalCovariates:
    """
    Deterministically synthesize a covariate stack for `region_id`.

    The same region_id always yields the same covariates (seeded on the
    region name), which matters for reproducible tests and for verification
    that "terrain-aware" behavior is stable run over run.
    """
    seed = abs(hash(region_id)) % (2**32)
    rng = np.random.default_rng(seed)
    h, w = shape

    # Elevation: smooth ridge-and-valley structure, 0-2500m.
    y = np.linspace(0, 4 * np.pi, h)
    x = np.linspace(0, 4 * np.pi, w)
    xx, yy = np.meshgrid(x, y)
    elevation = (np.sin(xx * rng.uniform(0.3, 0.8)) * np.cos(yy * rng.uniform(0.3, 0.8)) + 1) / 2
    elevation = elevation * 2500.0

    # Urban core: a soft blob near a random center, representing a metro area.
    cy, cx = rng.uniform(0.3, 0.7) * h, rng.uniform(0.3, 0.7) * w
    yy_idx, xx_idx = np.mgrid[0:h, 0:w]
    dist = np.sqrt((yy_idx - cy) ** 2 + (xx_idx - cx) ** 2)
    urban_fraction = np.clip(1.0 - dist / (0.25 * max(h, w)), 0.0, 1.0)
    impervious_fraction = np.clip(urban_fraction * rng.uniform(0.8, 1.0), 0.0, 1.0)

    # Coastline: a water mask on one edge, tapering inland.
    edge = rng.choice(["left", "right", "top", "bottom"])
    coastline = np.zeros((h, w), dtype=np.float32)
    depth = int(0.15 * (w if edge in ("left", "right") else h))
    if edge == "left":
        coastline[:, :depth] = np.linspace(1, 0, depth)
    elif edge == "right":
        coastline[:, -depth:] = np.linspace(0, 1, depth)
    elif edge == "top":
        coastline[:depth, :] = np.linspace(1, 0, depth)[:, None]
    else:
        coastline[-depth:, :] = np.linspace(0, 1, depth)[:, None]

    return RegionalCovariates(
        region_id=region_id,
        shape=shape,
        elevation_m=elevation.astype(np.float32),
        land_use_urban_fraction=urban_fraction.astype(np.float32),
        impervious_fraction=impervious_fraction.astype(np.float32),
        coastline_mask=coastline.astype(np.float32),
    )
