import pytest

from src.routing.google_directions import (
    DirectionsRequestError,
    parse_directions_response,
)
from src.routing.polyline_utils import decode_polyline, haversine_km, sample_evenly


def test_decode_polyline_known_example():
    # Official Google example: encodes to these three points.
    encoded = "_p~iF~ps|U_ulLnnqC_mqNvxq`@"
    points = decode_polyline(encoded)
    assert len(points) == 3
    assert points[0] == pytest.approx((38.5, -120.2), abs=1e-4)
    assert points[1] == pytest.approx((40.7, -120.95), abs=1e-4)
    assert points[2] == pytest.approx((43.252, -126.453), abs=1e-4)


def test_haversine_known_distance():
    # Roughly Indore to Bhopal, ~185km straight-line.
    indore = (22.7196, 75.8577)
    bhopal = (23.2599, 77.4126)
    dist = haversine_km(indore, bhopal)
    assert 150 < dist < 200


def test_haversine_zero_for_same_point():
    p = (12.34, 56.78)
    assert haversine_km(p, p) == pytest.approx(0.0, abs=1e-9)


def test_sample_evenly_includes_endpoints():
    path = [(0.0, 0.0), (0.0, 1.0), (0.0, 2.0), (0.0, 3.0)]
    sampled = sample_evenly(path, 4)
    assert sampled[0].lat == pytest.approx(0.0)
    assert sampled[0].lon == pytest.approx(0.0)
    assert sampled[-1].lon == pytest.approx(3.0, abs=1e-6)


def test_sample_evenly_spacing_is_roughly_uniform():
    path = [(0.0, float(i)) for i in range(11)]  # straight line, 0..10 degrees lon
    sampled = sample_evenly(path, 5)
    dists = [p.distance_from_start_km for p in sampled]
    gaps = [dists[i + 1] - dists[i] for i in range(len(dists) - 1)]
    # all gaps should be roughly equal (within 5%)
    avg_gap = sum(gaps) / len(gaps)
    for g in gaps:
        assert abs(g - avg_gap) / avg_gap < 0.05


def test_sample_evenly_single_point_path():
    sampled = sample_evenly([(1.0, 2.0)], 5)
    assert len(sampled) == 1
    assert sampled[0].lat == 1.0


def test_parse_directions_response_success():
    payload = {
        "status": "OK",
        "routes": [{
            "legs": [{
                "distance": {"value": 12000, "text": "12 km"},
                "duration": {"value": 900, "text": "15 mins"},
                "steps": [
                    {"html_instructions": "Head <b>north</b>", "distance": {"text": "1 km"}},
                    {"html_instructions": "Turn <b>right</b> onto Main St", "distance": {"text": "11 km"}},
                ],
            }],
            "overview_polyline": {"points": "_p~iF~ps|U_ulLnnqC_mqNvxq`@"},
        }],
    }
    result = parse_directions_response(payload, "A", "B")
    assert result.distance_km == pytest.approx(12.0)
    assert result.duration_min == pytest.approx(15.0)
    assert result.steps_summary == ["Head north (1 km)", "Turn right onto Main St (11 km)"]
    assert len(result.path) == 3


def test_parse_directions_response_error_status():
    payload = {"status": "ZERO_RESULTS"}
    with pytest.raises(DirectionsRequestError):
        parse_directions_response(payload, "A", "B")


def test_parse_directions_response_no_routes():
    payload = {"status": "OK", "routes": []}
    with pytest.raises(DirectionsRequestError):
        parse_directions_response(payload, "A", "B")


def test_parse_directions_response_multi_leg_sums_distance():
    payload = {
        "status": "OK",
        "routes": [{
            "legs": [
                {"distance": {"value": 5000}, "duration": {"value": 300}, "steps": []},
                {"distance": {"value": 7000}, "duration": {"value": 420}, "steps": []},
            ],
            "overview_polyline": {"points": ""},
        }],
    }
    result = parse_directions_response(payload, "A", "C")
    assert result.distance_km == pytest.approx(12.0)
    assert result.duration_min == pytest.approx(12.0)
    assert result.path == []
