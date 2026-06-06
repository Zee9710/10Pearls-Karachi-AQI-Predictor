import pytest
from src.aqi import compute_aqi, aqi_category, _sub_index, PM25_BREAKPOINTS


def test_zero_concentrations_give_zero():
    assert compute_aqi(0.0, 0.0, 0.0, 0.0, 0.0, 0.0) == 0


def test_pm25_boundary_gives_50():
    # PM2.5 = 12.0 µg/m³ → sub-index exactly 50; all others near zero
    result = compute_aqi(12.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert result == 50


def test_pm25_unhealthy_range():
    # PM2.5 = 100 µg/m³ → sub-index in 151-200 range
    result = compute_aqi(100.0, 10.0, 20.0, 10.0, 5.0, 500.0)
    assert 151 <= result <= 200


def test_aqi_takes_max_sub_index():
    # High PM10 should dominate low PM2.5
    result_high_pm10 = compute_aqi(5.0, 300.0, 20.0, 10.0, 5.0, 300.0)
    result_low_pm10 = compute_aqi(5.0, 50.0, 20.0, 10.0, 5.0, 300.0)
    assert result_high_pm10 > result_low_pm10


def test_sub_index_interpolation():
    # PM2.5 midpoint of first band: 6.0 → AQI should be ~25
    result = _sub_index(6.0, PM25_BREAKPOINTS)
    assert 20 <= result <= 30


def test_sub_index_below_range_gives_zero():
    assert _sub_index(-1.0, PM25_BREAKPOINTS) == 0


def test_sub_index_above_range_gives_500():
    assert _sub_index(9999.0, PM25_BREAKPOINTS) == 500


def test_aqi_category_good():
    assert aqi_category(25) == "Good"


def test_aqi_category_moderate():
    assert aqi_category(75) == "Moderate"


def test_aqi_category_unhealthy_sensitive():
    assert aqi_category(120) == "Unhealthy for Sensitive Groups"


def test_aqi_category_unhealthy():
    assert aqi_category(175) == "Unhealthy"


def test_aqi_category_very_unhealthy():
    assert aqi_category(250) == "Very Unhealthy"


def test_aqi_category_hazardous():
    assert aqi_category(350) == "Hazardous"


def test_unit_conversion_co():
    # CO at 4.5 ppm = 4.5 * 1145.6 µg/m³ ≈ 5155 → should give AQI ~51
    co_ugm3 = 4.5 * 1145.6
    result = compute_aqi(0.0, 0.0, 0.0, 0.0, 0.0, co_ugm3)
    assert 50 <= result <= 55
