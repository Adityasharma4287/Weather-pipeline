"""
route_weather_service.py
=========================
Ties `GoogleDirectionsClient` (real navigation) to
`WeatherPipelineOrchestrator` (Stages A-D) so a route between two places
comes back with weather sampled at points along the way — "map pe route +
us route ke weather conditions".

Flow:
    origin, destination
        -> Google Directions API (real route, real distance/duration)
        -> sample N evenly-spaced waypoints along the route
        -> for each waypoint: run the weather pipeline at that (lat, lon)
        -> return route + per-waypoint weather summary together
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

from src.pipeline.orchestrator import WeatherPipelineOrchestrator
from src.routing.google_directions import GoogleDirectionsClient, RouteResult
from src.routing.polyline_utils import SampledPoint, sample_evenly

MAX_WEATHER_WAYPOINTS = 10  # bounds pipeline compute cost per route request


@dataclass
class WaypointWeather:
    lat: float
    lon: float
    distance_from_start_km: float
    variable: str
    value: float             # central-pixel mean value at this waypoint
    p10: float
    p90: float
    signature_verified: bool


@dataclass
class RouteWeatherReport:
    origin: str
    destination: str
    distance_km: float
    duration_min: float
    path: List[tuple]                  # dense polyline for drawing the route on a map
    steps_summary: List[str]           # turn-by-turn instructions
    waypoints: List[WaypointWeather]   # sparse weather samples along the route
    model_version: str


class RouteWeatherService:
    def __init__(self, directions_client: Optional[GoogleDirectionsClient] = None,
                 orchestrator: Optional[WeatherPipelineOrchestrator] = None):
        self._directions = directions_client or GoogleDirectionsClient()
        self._orchestrator = orchestrator or WeatherPipelineOrchestrator(region_grid_shape=(12, 12))

    def get_route_with_weather(self, origin: str, destination: str, variable: str = "t2m",
                                lead_hours: int = 24, num_waypoints: int = 6,
                                requested_by: str = "unknown") -> RouteWeatherReport:
        num_waypoints = max(2, min(num_waypoints, MAX_WEATHER_WAYPOINTS))

        route: RouteResult = self._directions.get_route(origin, destination, requested_by=requested_by)
        sampled: List[SampledPoint] = sample_evenly(route.path, num_waypoints)

        init_time = datetime(2026, 1, 1)
        waypoints: List[WaypointWeather] = []
        model_version = "unknown"

        for point in sampled:
            result = self._orchestrator.run_for_coordinates(
                lat=point.lat, lon=point.lon, variable=variable, lead_hours=lead_hours,
                init_time=init_time, requested_by=requested_by,
            )
            model_version = result.model_version
            mean_field = result.ensemble.mean_field
            members = result.ensemble.members
            cy, cx = mean_field.shape[0] // 2, mean_field.shape[1] // 2

            import numpy as np  # local import keeps this module's top-level dependency surface minimal
            waypoints.append(WaypointWeather(
                lat=point.lat,
                lon=point.lon,
                distance_from_start_km=point.distance_from_start_km,
                variable=variable,
                value=float(mean_field[cy, cx]),
                p10=float(np.percentile(members[:, cy, cx], 10)),
                p90=float(np.percentile(members[:, cy, cx], 90)),
                signature_verified=result.signature_verified,
            ))

        return RouteWeatherReport(
            origin=origin,
            destination=destination,
            distance_km=route.distance_km,
            duration_min=route.duration_min,
            path=route.path,
            steps_summary=route.steps_summary,
            waypoints=waypoints,
            model_version=model_version,
        )
