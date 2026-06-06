PM25_BREAKPOINTS = [
    (0.0, 12.0, 0, 50),
    (12.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200),
    (150.5, 250.4, 201, 300),
    (250.5, 350.4, 301, 400),
    (350.5, 500.4, 401, 500),
]
PM10_BREAKPOINTS = [
    (0, 54, 0, 50),
    (55, 154, 51, 100),
    (155, 254, 101, 150),
    (255, 354, 151, 200),
    (355, 424, 201, 300),
    (425, 504, 301, 400),
    (505, 604, 401, 500),
]
O3_PPB_BREAKPOINTS = [
    (0, 54, 0, 50),
    (55, 70, 51, 100),
    (71, 85, 101, 150),
    (86, 105, 151, 200),
    (106, 200, 201, 300),
]
NO2_PPB_BREAKPOINTS = [
    (0, 53, 0, 50),
    (54, 100, 51, 100),
    (101, 360, 101, 150),
    (361, 649, 151, 200),
    (650, 1249, 201, 300),
    (1250, 1649, 301, 400),
    (1650, 2049, 401, 500),
]
SO2_PPB_BREAKPOINTS = [
    (0, 35, 0, 50),
    (36, 75, 51, 100),
    (76, 185, 101, 150),
    (186, 304, 151, 200),
    (305, 604, 201, 300),
    (605, 804, 301, 400),
    (805, 1004, 401, 500),
]
CO_PPM_BREAKPOINTS = [
    (0.0, 4.4, 0, 50),
    (4.5, 9.4, 51, 100),
    (9.5, 12.4, 101, 150),
    (12.5, 15.4, 151, 200),
    (15.5, 30.4, 201, 300),
    (30.5, 40.4, 301, 400),
    (40.5, 50.4, 401, 500),
]

# µg/m³ → ppb conversion factors at 25°C, 1 atm
_O3_UGM3_PER_PPB = 1.9957
_NO2_UGM3_PER_PPB = 1.9125
_SO2_UGM3_PER_PPB = 2.6196
_CO_UGM3_PER_PPM = 1145.6


def _sub_index(concentration: float, breakpoints: list) -> int:
    for c_lo, c_hi, aqi_lo, aqi_hi in breakpoints:
        if c_lo <= concentration <= c_hi:
            return round(
                (aqi_hi - aqi_lo) / (c_hi - c_lo) * (concentration - c_lo) + aqi_lo
            )
    if concentration < breakpoints[0][0]:
        return 0
    return 500


def compute_aqi(
    pm2_5: float,
    pm10: float,
    ozone_ugm3: float,
    no2_ugm3: float,
    so2_ugm3: float,
    co_ugm3: float,
) -> int:
    o3_ppb = ozone_ugm3 / _O3_UGM3_PER_PPB
    no2_ppb = no2_ugm3 / _NO2_UGM3_PER_PPB
    so2_ppb = so2_ugm3 / _SO2_UGM3_PER_PPB
    co_ppm = co_ugm3 / _CO_UGM3_PER_PPM
    sub_indices = [
        _sub_index(pm2_5, PM25_BREAKPOINTS),
        _sub_index(pm10, PM10_BREAKPOINTS),
        _sub_index(o3_ppb, O3_PPB_BREAKPOINTS),
        _sub_index(no2_ppb, NO2_PPB_BREAKPOINTS),
        _sub_index(so2_ppb, SO2_PPB_BREAKPOINTS),
        _sub_index(co_ppm, CO_PPM_BREAKPOINTS),
    ]
    return max(sub_indices)


def aqi_category(aqi: int) -> str:
    if aqi <= 50:
        return "Good"
    if aqi <= 100:
        return "Moderate"
    if aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    if aqi <= 200:
        return "Unhealthy"
    if aqi <= 300:
        return "Very Unhealthy"
    return "Hazardous"
