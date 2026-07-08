"""Shared fixtures for the test suite."""

import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_weather_df():
    return pd.DataFrame({
        'datetime': ['2024-01-15 10:00:00', '2024-01-15 11:00:00'],
        'day': ['2024-01-15', '2024-01-15'],
        'temp': [35.0, 36.0],
        'humidity': [65.0, 62.0],
        'precip': [0.0, 0.1],
        'windspeed': [12.0, 14.0],
        'feelslike': [28.0, 30.0],
        'visibility': [10.0, 9.0],
        'dew': [25.0, 24.0],
        'snow': [0.0, 0.0],
        'snowdepth': [0.0, 0.0],
    })


@pytest.fixture
def sample_taxi_df():
    data = {
        'PULocationID': [1, 1, 2, 2],
        'bin': pd.to_datetime(['2024-01-15 10:00:00', '2024-01-15 11:00:00',
                               '2024-01-15 10:00:00', '2024-01-15 11:00:00']),
        'target_yellow': [100, 120, 80, 90],
        'target_green': [30, 40, 20, 25],
        'target_hvfhv': [50, 60, 40, 45],
        'MPH': [25.0, 22.0, 30.0, 28.0],
        'city_avg_speed': [27.5, 25.0, 27.5, 25.0],
        'city_congestion_index': [0.036, 0.040, 0.036, 0.040],
        'zone_avg_speed': [25.0, 22.0, 30.0, 28.0],
        'zone_congestion_index': [0.040, 0.045, 0.033, 0.036],
        'trip_miles': [2.5, 3.0, 1.5, 2.0],
        'trip_duration_hr': [0.5, 0.6, 0.3, 0.4],
        'service_type': ['yellow', 'yellow', 'green', 'green'],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_merged_df(sample_taxi_df, sample_weather_df):
    weather = sample_weather_df.copy()
    weather['bin'] = pd.to_datetime(
        weather['day'].astype(str) + ' ' + weather['datetime'].astype(str)
    )
    weather = weather.drop(columns='day')
    return sample_taxi_df.merge(weather, on='bin', how='left')
