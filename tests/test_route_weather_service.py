from datetime import datetime

from src.pipeline.orchestrator import WeatherPipelineOrchestrator
from src.routing.google_directions import RouteResult
from src.routing.route_weather_service import RouteWeatherService


def test_run_for_coordinates_different_points_differ():
    orch = WeatherPipelineOrchestrator(region_grid_shape=(8, 8))
    a = orch.run_for_coordinates(lat=22.72, lon=75.86, variable="t2m", lead_hours=24,
                                  init_time=datetime(2026, 1, 1), requested_by="test")
    b = orch.run_for_coordinates(lat=23.26, lon=77.41, variable="t2m", lead_hours=24,
                                  init_time=datetime(2026, 1, 1), requested_by="test")
    assert not (a.ensemble.mean_field == b.ensemble.mean_field).all()


def test_run_for_coordinates_reproducible_for_same_point():
    orch = WeatherPipelineOrchestrator(region_grid_shape=(8, 8))
    a = orch.run_for_coordinates(lat=22.72, lon=75.86, variable="t2m", lead_hours=24,
                                  init_time=datetime(2026, 1, 1), requested_by="test")
    b = orch.run_for_coordinates(lat=22.72, lon=75.86, variable="t2m", lead_hours=24,
                                  init_time=datetime(2026, 1, 1), requested_by="test")
    assert (a.ensemble.mean_field == b.ensemble.mean_field).all()


class _FakeDirectionsClient:
    """Stand-in for GoogleDirectionsClient — returns a fixed route without any network call."""

    def get_route(self, origin, destination, mode="driving", requested_by="unknown"):
        # A simple straight-line path from Indore towards Bhopal, dense enough to sample.
        path = [(22.72 + i * 0.05, 75.86 + i * 0.15) for i in range(12)]
        return RouteResult(
            origin=origin, destination=destination,
            distance_km=185.0, duration_min=210.0,
            path=path,
            steps_summary=["Head northeast (10 km)", "Continue on NH46 (175 km)"],
        )


def test_route_weather_service_end_to_end():
    orch = WeatherPipelineOrchestrator(region_grid_shape=(8, 8))
    service = RouteWeatherService(directions_client=_FakeDirectionsClient(), orchestrator=orch)

    report = service.get_route_with_weather(
        origin="Indore, MP", destination="Bhopal, MP", variable="t2m",
        lead_hours=24, num_waypoints=4, requested_by="test",
    )

    assert report.distance_km == 185.0
    assert report.duration_min == 210.0
    assert len(report.waypoints) == 4
    assert report.waypoints[0].distance_from_start_km == 0.0
    # waypoints should be ordered by increasing distance from start
    dists = [w.distance_from_start_km for w in report.waypoints]
    assert dists == sorted(dists)
    # different points along the route should generally not all be identical
    values = [w.value for w in report.waypoints]
    assert len(set(round(v, 3) for v in values)) > 1


def test_route_weather_service_precip_nonnegative():
    orch = WeatherPipelineOrchestrator(region_grid_shape=(8, 8))
    service = RouteWeatherService(directions_client=_FakeDirectionsClient(), orchestrator=orch)
    report = service.get_route_with_weather(
        origin="A", destination="B", variable="tp", lead_hours=12, num_waypoints=3, requested_by="test",
    )
    for w in report.waypoints:
        assert w.value >= 0.0
        assert w.p10 >= 0.0
