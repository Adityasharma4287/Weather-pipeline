"""
interactive_cli.py
===================
Interactive REPL that runs the real pipeline (Stages A-D via the
orchestrator, optionally through the secured API layer) and returns actual
computed forecasts — not scripted/canned responses.

Two modes:
  1. Direct mode (default): calls WeatherPipelineOrchestrator in-process.
  2. API mode (--api): issues a JWT token then talks to a running
     `uvicorn src.api.main:app` instance over HTTP, exercising the full
     secured Stage E path (auth, rate limiting, caching, signature check).

Usage:
    python interactive_cli.py
    python interactive_cli.py --api --base-url http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import shlex
import sys
from datetime import datetime
from typing import Optional

import numpy as np

VALID_VARIABLES = {"t2m": "2m temperature (K)", "u10": "10m u-wind (m/s)",
                    "v10": "10m v-wind (m/s)", "tp": "total precipitation (mm/hr)"}

BANNER = """
==================================================================
 Localized Weather Intelligence Pipeline — Interactive CLI
 (runs the real Stage A->D pipeline; nothing here is scripted)
==================================================================
Commands:
  forecast <region> [variable] [lead_hours]   e.g. forecast metro-demo t2m 24
  history                                     show past queries this session
  vars                                        list available variables
  help                                        show this message
  quit / exit
"""


def render_field_ascii(field: np.ndarray, width: int = 32) -> str:
    """
    Render a 2D field as a coarse ASCII heatmap so the CLI shows something
    visual without needing a plotting library.
    """
    h, w = field.shape
    step_y = max(1, h // 16)
    step_x = max(1, w // width)
    chars = " .:-=+*#%@"
    lo, hi = float(field.min()), float(field.max())
    span = hi - lo if hi > lo else 1.0

    lines = []
    for y in range(0, h, step_y):
        row = []
        for x in range(0, w, step_x):
            val = field[y, x]
            idx = int((val - lo) / span * (len(chars) - 1))
            row.append(chars[idx])
        lines.append("".join(row))
    return "\n".join(lines) + f"\n(range: {lo:.2f} .. {hi:.2f})"


class DirectRunner:
    """In-process pipeline runner (Stages A-D, no HTTP layer)."""

    def __init__(self):
        from src.pipeline.orchestrator import WeatherPipelineOrchestrator
        self._orchestrator = WeatherPipelineOrchestrator(region_grid_shape=(16, 16))

    def forecast(self, region: str, variable: str, lead_hours: int) -> dict:
        result = self._orchestrator.run(
            region_id=region, variable=variable, lead_hours=lead_hours,
            init_time=datetime(2026, 1, 1), requested_by="cli-interactive",
        )
        return {
            "region": region,
            "variable": variable,
            "lead_hours": lead_hours,
            "model_version": result.model_version,
            "signature_verified": result.signature_verified,
            "rmse": result.rmse_vs_pseudo_truth,
            "crps": result.crps_vs_pseudo_truth,
            "spread_skill": result.spread_skill,
            "mean_field": result.ensemble.mean_field,
            "ensemble_members": result.ensemble.members.shape[0],
        }


class ApiRunner:
    """HTTP client against a running FastAPI instance (full Stage E path)."""

    def __init__(self, base_url: str):
        import httpx
        self._base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=30.0)
        self._token = self._issue_token()

    def _issue_token(self) -> str:
        resp = self._client.post(
            f"{self._base_url}/v1/auth/token",
            json={"sub": "cli-user", "tenant": "cli-tenant", "scopes": ["forecast:read"]},
        )
        resp.raise_for_status()
        return resp.json()["access_token"]

    def forecast(self, region: str, variable: str, lead_hours: int) -> dict:
        resp = self._client.get(
            f"{self._base_url}/v1/forecast/{region}",
            params={"variable": variable, "lead_hours": lead_hours},
            headers={"Authorization": f"Bearer {self._token}"},
        )
        resp.raise_for_status()
        body = resp.json()
        mean_field = np.array(body["ensemble_summary"]["mean"])
        return {
            "region": region,
            "variable": variable,
            "lead_hours": lead_hours,
            "model_version": body["model_version"],
            "signature_verified": body["signature_verified"],
            "rmse": body["verification"]["rmse"] if body["verification"] else None,
            "crps": body["verification"]["crps"] if body["verification"] else None,
            "spread_skill": body["verification"]["spread_skill_ratio"] if body["verification"] else None,
            "mean_field": mean_field,
            "ensemble_members": None,
        }


def print_result(res: dict) -> None:
    print("-" * 66)
    print(f"Region            : {res['region']}")
    print(f"Variable          : {res['variable']} ({VALID_VARIABLES.get(res['variable'], '?')})")
    print(f"Lead time         : {res['lead_hours']}h")
    print(f"Model version     : {res['model_version']}")
    print(f"Signature verified: {res['signature_verified']}")
    if res["ensemble_members"] is not None:
        print(f"Ensemble members  : {res['ensemble_members']}")
    if res["rmse"] is not None:
        print(f"RMSE (vs pseudo-truth)  : {res['rmse']:.4f}")
        print(f"CRPS (vs pseudo-truth)  : {res['crps']:.4f}")
        print(f"Spread/skill ratio      : {res['spread_skill']:.4f}")
    print("\nDownscaled mean field (ASCII preview):")
    print(render_field_ascii(res["mean_field"]))
    print("-" * 66)


def run_repl(runner) -> None:
    print(BANNER)
    history = []
    while True:
        try:
            raw = input("weather> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if not raw:
            continue

        parts = shlex.split(raw)
        cmd = parts[0].lower()

        if cmd in ("quit", "exit"):
            print("Goodbye.")
            break
        elif cmd == "help":
            print(BANNER)
        elif cmd == "vars":
            for k, v in VALID_VARIABLES.items():
                print(f"  {k:5s} - {v}")
        elif cmd == "history":
            if not history:
                print("(no queries yet)")
            for i, h in enumerate(history, 1):
                print(f"{i}. {h}")
        elif cmd == "forecast":
            if len(parts) < 2:
                print("Usage: forecast <region> [variable] [lead_hours]")
                continue
            region = parts[1]
            variable = parts[2] if len(parts) > 2 else "t2m"
            lead_hours = int(parts[3]) if len(parts) > 3 else 24

            if variable not in VALID_VARIABLES:
                print(f"Unknown variable '{variable}'. Try: {', '.join(VALID_VARIABLES)}")
                continue
            if not (6 <= lead_hours <= 240):
                print("lead_hours must be between 6 and 240.")
                continue

            try:
                res = runner.forecast(region, variable, lead_hours)
            except Exception as exc:  # noqa: BLE001
                print(f"Forecast failed: {exc}")
                continue

            print_result(res)
            history.append(f"forecast {region} {variable} {lead_hours}h")
        else:
            print(f"Unknown command '{cmd}'. Type 'help' for a list of commands.")


def main():
    parser = argparse.ArgumentParser(description="Interactive CLI for the weather pipeline.")
    parser.add_argument("--api", action="store_true", help="Talk to a running FastAPI server instead of running in-process.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="Base URL when --api is used.")
    args = parser.parse_args()

    if args.api:
        print(f"Connecting to API at {args.base_url} ...")
        runner = ApiRunner(args.base_url)
        print("Connected and authenticated.")
    else:
        print("Running in direct (in-process) mode — no server needed.")
        runner = DirectRunner()

    run_repl(runner)


if __name__ == "__main__":
    main()
