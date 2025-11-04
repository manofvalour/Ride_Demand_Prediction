import os, sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pydantic import BaseModel, field_validator
import requests


from src.DynamicPricingEngine.logger.logger import logger
from src.DynamicPricingEngine.exception.customexception import RideDemandException
from src.DynamicPricingEngine.inference.schema import PredictionRequest

"""" Things to Do
- input from the users (datetime (time rounded to hour) and PULocation ID(between 1 and 263))
- engineer the needed features (hour, day_of_the_month, month, is_holiday, is_special_event, 
                                season of the year, day_of_the_week, pickup_lag1, is_night_hour, 
                                is_rush_hour, pickup_lag24, pickup_year, 
                                pickup_roll_mean24, pickup_roll_std_24, city_avg_speed, is_payday,
                                zone_avg_speed, zone_congestion_index, city_congestion_index,
                                city_pickup_lag1, city_pickup_lag24, city_pickups, neighbor_pickups_sum,
                                neighbor_pickup_lag1, neighbor_pickup_lag24, is_rush_hour, is_night_hour,
                                temp, dew, humidity, precip, snow, windspeed, feelslike, snow_depth,
                                visibility)
- push_transformed_data_to_feature_store
- ingest_the_data_and model_from_feature_store_and_model_store
- predict and return the output

"""





# src/inference/feature_builder.py
import pandas as pd
import numpy as np
from meteostat import Point, Hourl
from datetime import timedelta
import holidays
from joblib import Parallel, delayed


def is_holiday(dt: pd.Timestamp) -> int:
    us_holidays = holidays.US(years=dt.year)
    return int(dt.date() in us_holidays)

def build_features(dt: pd.Timestamp, pulocationid: int, historical_df: pd.DataFrame) -> pd.DataFrame:
    """Build one row for inference."""
    # Base
    row = {
        'pickup_year': dt.year,
        'pickup_month': dt.month,
        'pulocationid': pulocationid,
        'datetime': dt,
        'day_of_month': dt.day,
        'pickup_hour': dt.hour,
        'day_of_week': dt.weekday(),
        'is_weekend': int(dt.weekday() >= 5),
        'is_rush_hour': int(dt.hour in [7,8,9,17,18,19]),
        'is_night_hour': int(dt.hour in [0,1,2,3,4,5,22,23]),
        'season_of_year': (dt.month % 12 + 3) // 3,  # 1=winter, 2=spring, etc.
        'is_holiday': is_holiday(dt),
        'is_special_event': 0,  # extend later
        'is_payday': int(dt.day in [1, 15]),
    }

    # Weather
    weather = get_weather(dt)
    row.update(weather)

    # Lags & Rolling (from historical data)
    zone_hist = historical_df[historical_df['pulocationid'] == pulocationid].copy()
    zone_hist['datetime'] = pd.to_datetime(zone_hist['datetime'])
    zone_hist = zone_hist.set_index('datetime').sort_index()

    target_time = dt.floor('H')
    if len(zone_hist) > 0:
        # Lag 1h, 24h
        lag_1h = zone_hist.asof(target_time - timedelta(hours=1))
        lag_24h = zone_hist.asof(target_time - timedelta(hours=24))

        row['pickups_lag_1h'] = lag_1h['pickups'] if lag_1h is not None else 0
        row['pickups_lag_24h'] = lag_24h['pickups'] if lag_24h is not None else 0

        # Rolling 24h
        window = zone_hist.loc[target_time - timedelta(hours=23): target_time]
        row['pickups_roll_mean_24h'] = window['pickups'].mean() if len(window) > 0 else 0
        row['pickups_roll_std_24h'] = window['pickups'].std() if len(window) > 0 else 0
    else:
        row.update({
            'pickups_lag_1h': 0, 'pickups_lag_24h': 0,
            'pickups_roll_mean_24h': 0, 'pickups_roll_std_24h': 0
        })

    # City & Neighbor aggregates (simplified)
    city_hist = historical_df.copy()
    city_hist['datetime'] = pd.to_datetime(city_hist['datetime'])
    city_hist = city_hist.set_index('datetime').sort_index()

    city_lag_1h = city_hist.asof(target_time - timedelta(hours=1))
    city_lag_24h = city_hist.asof(target_time - timedelta(hours=24))

    row['city_pickups'] = city_hist.asof(target_time)['pickups'] if city_lag_1h is not None else 0
    row['city_pickups_lag_1h'] = city_lag_1h['pickups'] if city_lag_1h is not None else 0
    row['city_pickups_lag_24h'] = city_lag_24h['pickups'] if city_lag_24h is not None else 0

    # Neighbor sum (example: zones within 1km — you need a mapping)
    # row['neighbor_pickups_sum'] = ...

    # Speed & congestion (mock or from external API)
    row['city_avg_speed'] = 25.0
    row['city_congestion_index'] = 1.2
    row['zone_avg_speed'] = 22.0
    row['zone_congestion_index'] = 1.3

    # Bin string (from pulocationid + hour)
    row['bin_str'] = f"{pulocationid}_{dt.hour}"

    return pd.DataFrame([row])