-- Feature store schema for the Pearls AQI Predictor.
-- Run once in the Supabase SQL Editor to create the feature table.
-- The service (secret) API key bypasses row-level security, so no policies
-- are required for the server-side feature/training/serving jobs.

create table if not exists aqi_features (
  datetime timestamptz primary key,
  pm2_5 double precision,
  pm10 double precision,
  ozone double precision,
  nitrogen_dioxide double precision,
  sulphur_dioxide double precision,
  carbon_monoxide double precision,
  temperature_2m double precision,
  relative_humidity_2m double precision,
  wind_speed_10m double precision,
  precipitation double precision,
  surface_pressure double precision,
  aqi double precision,
  hour_sin double precision,
  hour_cos double precision,
  dow_sin double precision,
  dow_cos double precision,
  month_sin double precision,
  month_cos double precision,
  is_weekend double precision,
  aqi_lag_1h double precision,
  aqi_lag_24h double precision,
  aqi_lag_48h double precision,
  aqi_lag_72h double precision,
  aqi_roll24_mean double precision,
  aqi_roll24_std double precision,
  aqi_roll72_mean double precision,
  aqi_roll72_max double precision,
  aqi_change_24h double precision,
  temp_humidity double precision,
  wind_pollution double precision
);
