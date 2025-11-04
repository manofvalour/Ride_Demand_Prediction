import os, sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pydantic import BaseModel, field_validator
import requests
from dotenv import load_dotenv

load_dotenv()

from src.DynamicPricingEngine.logger.logger import logger
from src.DynamicPricingEngine.exception.customexception import RideDemandException
from src.DynamicPricingEngine.entity.config_entity import InferenceConfig

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

class PredictionRequest(BaseModel):
    datetime: str
    pulocationid: int

    @field_validator('datetime', pre=True)
    def parse_datetime(self, v):
        return pd.to_datetime(v)


    @field_validator('pulocationid')
    def check_pulocationid(self, v):
        if not (1<=v<=263):
            raise ValueError("PULocationID must be between 1-263")
        
        return v

class InferencePipeline:
    def __init__(self, config: InferenceConfig, start_date_time: str, 
                 end_date_time:str, api_key: str, pulocationid: int):
        try:
            self.config = config
            if not isinstance(start_date_time, str):
                raise ValueError("start_date_time must be a string in 'YYYY-MM-DD' format")
            self.start_date = start_date_time

            if not isinstance(end_date_time, str):
                raise ValueError("end_date_time must be a string in 'YYYY-MM-DD' format")
            self.end_date = end_date_time

            self.api_key = api_key

            if not (1<=pulocationid<=263):
                raise ValueError("PULocationID must be between 1-263")
            self.pulocationid = pulocationid

        except Exception as e:
            logger.error("Error initializing InferencePipeline", e)
            raise RideDemandException(e, sys)

    def get_nyc_prediction_weather_data(self)-> pd.DataFrame:

        try:
            base_url = self.config.weather_data_url
            location: str = "New York, NY, United States"
            start_of_month_str = self.start_date
            end_of_month_str = self.end_date
            api_key = self.api_key

            params = {
                "unitGroup": "us",
                "key": api_key,
                "include": "days,hours,current,alerts,stations",
                "contentType": "json"
            }

            url = f"{base_url}/{location}/{start_of_month_str}/{end_of_month_str}"
            logger.info(f"Fetching data from {start_of_month_str} to {end_of_month_str}...")

            try:
                response = requests.get(url, params=params)
                response.raise_for_status()

            except requests.RequestException as e:
                print(f"Request failed: {e}")
                return None

            data = response.json()
            days = data.get("days", [])
            if not days:
                print("No 'days' data found in response.")
                return None

            hourly_records = []
            fields = ['datetime', 'temp', 'dew', 'humidity', 'precip',
                      'snow', 'windspeed', 'feelslike', 'snowdepth', 'visibility']


            for day in days:
                for hour in day.get("hours", []):
                    filtered_hour = {key: hour.get(key) for key in fields}
                    filtered_hour["day"] = day.get("datetime")
                    hourly_records.append(filtered_hour)

            df_hours = pd.DataFrame(hourly_records)
            logger.info(f"Retrieved {len(days)} days ({df_hours.shape[0]} hourly records).")

            return df_hours

        except Exception as e:
            logger.error(f"failed to extract weather data {e}")
            raise RideDemandException(e,sys)

    def engineer_temporal_prediction_features(self, weather_df: pd.DataFrame)-> pd.DataFrame:
        try:
            pulocationid = self.pulocationid
            weather_df['datetime']= pd.to_datetime(weather_df['day'] + ' ' + weather_df['datetime'])
            weather_df.drop(columns=['day'], inplace=True)

            # Feature engineering logic goes here
            # Example: Extract hour, day_of_week, is_weekend, etc. from datetime
            weather_df['hour'] = weather_df['datetime'].dt.hour
            weather_df['day_of_week'] = weather_df['datetime'].dt.dayofweek
            weather_df['is_weekend'] = weather_df['day_of_week'].isin([5, 6]).astype(int)
            weather_df['month'] = weather_df['datetime'].dt.month
            weather_df['is_night_hour'] = weather_df['hour'].apply(lambda x: 1 if x < 6 or x > 20 else 0)
            weather_df['is_rush_hour'] = weather_df['hour'].apply(lambda x: 1 if 7 <= x <= 9 or 16 <= x <= 19 else 0)
            weather_df['season_of_year'] = weather_df['month'].apply(
                lambda x: (x%12 + 3)//3
            )  # 1: Winter, 2: Spring, 3: Summer, 4: Fall
            weather_df['is_holiday'] = 0  # Placeholder, implement holiday logic as needed
            weather_df['is_special_event'] = 0  # Placeholder, implement special event logic as needed
            weather_df['is_payday'] = 0  # Placeholder, implement payday logic as needed
            weather_df['pickup_year'] = weather_df['datetime'].dt.year
            weather_df['day_of_month'] = weather_df['datetime'].dt.day

            weather_df['pulocationid'] = pulocationid
         

            logger.info("Temporal Feature for prediction generated successfully.")
            return weather_df

        except Exception as e:
            logger.error(f"failed to engineer temporal features for prediction data {e}")
            raise RideDemandException(e,sys)

    def generate_lag_features_for_prediction(self, hist_df:pd.DataFrame,
                                              pred_df: pd.DataFrame)-> pd.DataFrame:
        try:
            pulocationid = self.pulocationid
            df = df.sort_values(by='datetime')
            df.set_index('datetime', inplace=True)

            # Generating lag features
            df['pickup_lag1'] = df['pulocationid'].shift(1)
            df['pickup_lag24'] = df['pulocationid'].shift(24)
            df['pickup_roll_mean24'] = df['pulocationid'].rolling(window=24).mean()
            df['pickup_roll_std_24'] = df['pulocationid'].rolling(window=24).std()

            df.reset_index(inplace=True)

            logger.info("Lag features for prediction generated successfully.")
            return df

        except Exception as e:
            logger.error(f"failed to engineer lag features for prediction data {e}")
            raise RideDemandException(e,sys)
        

    def push_engineered_features_to_feature_store(self):
        pass

    def make_predictions(self):
        pass

    def initiate_prediction_pipeline(self):
        pass