"""
google_directions.py
=====================
Stage: Routing / Navigation (new — extends the architecture doc's Stage E
"Visualization" with real turn-by-turn navigation via the Google Directions
API, overlaid with the pipeline's weather output).

Purpose
-------
Wraps calls to Google's Directions API
(https://maps.googleapis.com/maps/api/directions/json) behind the same
secured pattern used for weather data sources: credential resolved from the
SecretsManager (never hardcoded), audit-logged, and with a clean parsing
boundary so the HTTP-calling code and the response-parsing code are
separately testable.

Getting your API key wired in
------------------------------
Set the `GOOGLE_MAPS_API_KEY` environment variable (or, in Docker, put it
in `.env`) before starting the API. It is resolved through
`SecretsManager` under the reference `secretsmanager://routing/google-maps-api-key`,
exactly like the weather data-source credentials.

Enable the "Directions API" for this key in Google Cloud Console, and if
you also use the same key client-side for map rendering, restrict that
client-side use with an HTTP referrer restriction (see README).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import httpx

from src.routing.polyline_utils import decode_polyline
from src.security.audit_log import AuditLog
from src.security.secrets_manager import SecretsManager, get_default_secrets_manager

DIRECTIONS_API_URL = "https://maps.googleapis.com/maps/api/directions/json"


class DirectionsRequestError(Exception):
    """Raised when the Directions API call fails or returns a non-OK status."""


@dataclass
class RouteResult:
    origin: str
    destination: str
    distance_km: float
    duration_min: float
    path: List[tuple]  # dense list of (lat, lon) decoded from the route polyline
    steps_summary: List[str]  # human-readable turn-by-turn steps (HTML tags stripped)


def _strip_html(text: str) -> str:
    out, in_tag = [], False
    for ch in text:
        if ch == "<":
            in_tag = True
        elif ch == ">":
            in_tag = False
        elif not in_tag:
            out.append(ch)
    return "".join(out)


def parse_directions_response(payload: dict, origin: str, destination: str) -> RouteResult:
    """
    Pure parsing function — takes a Google Directions API JSON payload
    (already fetched) and turns it into a RouteResult. Separated from the
    network call so this can be unit-tested against a fixture payload
    without any network access.
    """
    status = payload.get("status")
    if status != "OK":
        raise DirectionsRequestError(
            f"Directions API returned status '{status}': {payload.get('error_message', 'no message')}"
        )

    routes = payload.get("routes", [])
    if not routes:
        raise DirectionsRequestError("Directions API returned no routes.")

    route = routes[0]
    legs = route.get("legs", [])
    if not legs:
        raise DirectionsRequestError("Directions API route has no legs.")

    total_distance_m = sum(leg["distance"]["value"] for leg in legs)
    total_duration_s = sum(leg["duration"]["value"] for leg in legs)

    steps_summary: List[str] = []
    for leg in legs:
        for step in leg.get("steps", []):
            instruction = _strip_html(step.get("html_instructions", ""))
            distance_text = step.get("distance", {}).get("text", "")
            steps_summary.append(f"{instruction} ({distance_text})")

    overview_polyline = route.get("overview_polyline", {}).get("points", "")
    path = decode_polyline(overview_polyline) if overview_polyline else []

    return RouteResult(
        origin=origin,
        destination=destination,
        distance_km=total_distance_m / 1000.0,
        duration_min=total_duration_s / 60.0,
        path=path,
        steps_summary=steps_summary,
    )


class GoogleDirectionsClient:
    """Secured client for the Google Directions API."""

    def __init__(self, secrets: SecretsManager = None, audit: AuditLog = None, timeout_seconds: float = 15.0):
        self._secrets = secrets or get_default_secrets_manager()
        self._audit = audit or AuditLog()
        self._timeout = timeout_seconds

    def _api_key(self) -> str:
        key = self._secrets.get_secret(
            "secretsmanager://routing/google-maps-api-key", requested_by="GoogleDirectionsClient"
        )
        if not key:
            raise DirectionsRequestError(
                "GOOGLE_MAPS_API_KEY is not set. Add it to your .env (or export it before running "
                "uvicorn) — see README 'Map navigation setup'."
            )
        return key

    def get_route(self, origin: str, destination: str, mode: str = "driving",
                   requested_by: str = "unknown") -> RouteResult:
        """
        Fetch a route between `origin` and `destination` (either free-text
        addresses or "lat,lon" strings — both are accepted by Google's API).
        """
        api_key = self._api_key()
        params = {"origin": origin, "destination": destination, "mode": mode, "key": api_key}

        try:
            resp = httpx.get(DIRECTIONS_API_URL, params=params, timeout=self._timeout)
            resp.raise_for_status()
            payload = resp.json()
        except httpx.HTTPError as exc:
            self._audit.record(actor=requested_by, action="directions_request_failed",
                                resource=f"{origin}->{destination}", metadata={"error": str(exc)})
            raise DirectionsRequestError(f"Directions API request failed: {exc}") from exc

        result = parse_directions_response(payload, origin, destination)

        self._audit.record(
            actor=requested_by,
            action="directions_request",
            resource=f"{origin}->{destination}",
            metadata={"distance_km": result.distance_km, "duration_min": result.duration_min},
        )
        return result
