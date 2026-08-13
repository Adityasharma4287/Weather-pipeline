"""
run_pipeline.py
================
Standalone end-to-end demo: runs the full A -> D pipeline once (bypassing
the API layer) and prints a human-readable summary, then writes a JSON
report to disk. Useful for a quick sanity check without starting the
FastAPI server.

Usage:
    python run_pipeline.py
    python run_pipeline.py --region metro-demo --variable tp --lead-hours 48
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime

from src.pipeline.orchestrator import WeatherPipelineOrchestrator
from src.security.audit_log import AuditLog


def main():
    parser = argparse.ArgumentParser(description="Run the localized weather pipeline end to end.")
    parser.add_argument("--region", default="metro-demo-v1")
    parser.add_argument("--variable", default="t2m", choices=["t2m", "u10", "v10", "tp"])
    parser.add_argument("--lead-hours", type=int, default=24)
    parser.add_argument("--grid", type=int, default=16, help="coarse global-model grid size (NxN)")
    parser.add_argument("--out", default="pipeline_report.json")
    args = parser.parse_args()

    print(f"Running pipeline: region={args.region} variable={args.variable} lead_hours={args.lead_hours}")
    print("-" * 70)

    orchestrator = WeatherPipelineOrchestrator(region_grid_shape=(args.grid, args.grid))
    result = orchestrator.run(
        region_id=args.region,
        variable=args.variable,
        lead_hours=args.lead_hours,
        init_time=datetime(2026, 1, 1),
        requested_by="cli-demo",
    )

    print(f"Stage B (global forecast) model version : {result.model_version}")
    print(f"Stage C (downscaling) output grid        : {result.ensemble.mean_field.shape}")
    print(f"Stage C ensemble members                 : {result.ensemble.members.shape[0]}")
    print(f"Artifact signature verified               : {result.signature_verified}")
    print(f"Stage D RMSE  (vs. pseudo-truth)          : {result.rmse_vs_pseudo_truth:.4f}")
    print(f"Stage D CRPS  (vs. pseudo-truth)          : {result.crps_vs_pseudo_truth:.4f}")
    print(f"Stage D spread/skill ratio                 : {result.spread_skill:.4f}")

    # Fit + apply bias correction as a demonstration of that sub-stage.
    bc_pipeline = orchestrator.bias_correction_pipeline_for(result.ensemble, args.variable)
    corrected_mean = bc_pipeline.apply(result.ensemble.mean_field)
    print(f"Bias-corrected mean field shape            : {corrected_mean.shape}")

    report = {
        "region_id": args.region,
        "variable": args.variable,
        "lead_hours": args.lead_hours,
        "model_version": result.model_version,
        "signature_verified": result.signature_verified,
        "rmse": result.rmse_vs_pseudo_truth,
        "crps": result.crps_vs_pseudo_truth,
        "spread_skill_ratio": result.spread_skill,
        "ensemble_shape": list(result.ensemble.members.shape),
    }
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print("-" * 70)
    print(f"Report written to {args.out}")

    audit = AuditLog()
    intact = audit.verify_integrity()
    print(f"Audit log integrity check                  : {'OK' if intact else 'FAILED'}")


if __name__ == "__main__":
    main()
