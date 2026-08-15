"""
polyline_utils.py
==================
Pure, dependency-free helpers for working with Google's encoded polyline
format and for sampling evenly-spaced waypoints along a route. Kept
separate from the network-calling client (`google_directions.py`) so this
logic is fully unit-testable without hitting Google's API or needing
network access.

Google's polyline encoding: https://developers.google.com/maps/documentation/utilities/polylinealgorithm
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple


def decode_polyline(encoded: str) -> List[Tuple[float, float]]:
    """
    Decode a Google-encoded polyline string into a list of (lat, lon) pairs.
    Standard algorithm — precision 1e-5.
    """
    points: List[Tuple[float, float]] = []
    index = lat = lng = 0
    length = len(encoded)

    while index < length:
        result = shift = 0
        while True:
            b = ord(encoded[index]) - 63
            index += 1
            result |= (b & 0x1f) << shift
            shift += 5
            if b < 0x20:
                break
        dlat = ~(result >> 1) if (result & 1) else (result >> 1)
        lat += dlat

        result = shift = 0
        while True:
            b = ord(encoded[index]) - 63
            index += 1
            result |= (b & 0x1f) << shift
            shift += 5
            if b < 0x20:
                break
        dlng = ~(result >> 1) if (result & 1) else (result >> 1)
        lng += dlng

        points.append((lat / 1e5, lng / 1e5))

    return points


def haversine_km(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Great-circle distance between two (lat, lon) points, in kilometers."""
    lat1, lon1 = a
    lat2, lon2 = b
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    h = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(h)))


@dataclass(frozen=True)
class SampledPoint:
    lat: float
    lon: float
    distance_from_start_km: float


def sample_evenly(path: List[Tuple[float, float]], n_points: int) -> List[SampledPoint]:
    """
    Given a dense polyline path (many points), pick `n_points` points spaced
    evenly by cumulative distance along the route (not just evenly by index
    — a route with a dense cluster of points on a curve and sparse points
    on a straight stretch would otherwise be sampled unevenly in space).

    Always includes the first and last point of the path.
    """
    if not path:
        return []
    if len(path) == 1 or n_points <= 1:
        return [SampledPoint(lat=path[0][0], lon=path[0][1], distance_from_start_km=0.0)]

    # Cumulative distance along the path.
    cumulative = [0.0]
    for i in range(1, len(path)):
        cumulative.append(cumulative[-1] + haversine_km(path[i - 1], path[i]))
    total = cumulative[-1]

    if total == 0:
        return [SampledPoint(lat=path[0][0], lon=path[0][1], distance_from_start_km=0.0)]

    targets = [total * i / (n_points - 1) for i in range(n_points)]

    sampled: List[SampledPoint] = []
    j = 0
    for target in targets:
        while j < len(cumulative) - 1 and cumulative[j + 1] < target:
            j += 1
        # Linear interpolation between path[j] and path[j+1].
        if j >= len(path) - 1:
            lat, lon = path[-1]
            dist = cumulative[-1]
        else:
            seg_len = cumulative[j + 1] - cumulative[j]
            t = 0.0 if seg_len == 0 else (target - cumulative[j]) / seg_len
            lat = path[j][0] + t * (path[j + 1][0] - path[j][0])
            lon = path[j][1] + t * (path[j + 1][1] - path[j][1])
            dist = target
        sampled.append(SampledPoint(lat=lat, lon=lon, distance_from_start_km=dist))

    return sampled
