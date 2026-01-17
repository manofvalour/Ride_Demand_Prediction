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

       final_features = [
                'temp',
                'humidity',
                'pickup_hour',
                'is_rush_hour',
                'city_avg_speed',
                'zone_avg_speed',
                'zone_congestion_index',
                'pickups_lag_1h',
                'pulocationid',
                'pickups_lag_24h',
                'city_pickups_lag_1h',
                'neighbor_pickups_lag_1h'
            ]

            # 2. Adding the target variable to the list for the final dataframe
            target_column = 'pickups'

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
    def __init__(self, config: InferenceConfig):
        try:
            self.config = config
            self.api_key = os.getenv('API_KEY')


        except Exception as e:
            logger.error("Error initializing Inference Pipeline", e)
            raise RideDemandException(e, sys)

    def get_nyc_prediction_weather_data(self, config:InferenceConfig)-> pd.DataFrame:

        try:
            # Define your API configuration
            api_key = self.api_key
            location = "New York, NY, United States"
            base_url = self.config.weather_data_url
        
            params = {
                "unitGroup": "us",
                "key": api_key,
                "include": "current",
                "contentType": "json"
            }

            url = f"{base_url}/{location}/today"
            logger.info(f"Fetching data for current timestamp{datetime.now().strftime('%Y-%m-%d, %H')}:00:00 ...")

            try:
                response = requests.get(url, params=params)
                response.raise_for_status()

            except requests.RequestException as e:
                print(f"Request failed: {e}")
                return None

            # Access specific data (e.g., current conditions)
            response = response.json()
            current = response.get("currentConditions")
            hourly_records = []
            hourly_records.append(current)

            df_hours = pd.DataFrame(hourly_records)
            df_hours = df_hours[['datetime', 'temp', 'humidity']]
            df_hours['datetime'] = pd.to_datetime(df_hours['datetime'])
            
            logger.info(f"Successfully Retrieved the weather data for {datetime.now().strftime('%Y-%m-%d, %H')}:00:00.")

            return df_hours

        except Exception as e:
            logger.error(f"failed to extract weather data {e}")
            raise RideDemandException(e,sys)
        
    def engineer_temporal_prediction_features(self, df: pd.DataFrame)-> pd.DataFrame:
        try:
            pulocationid = self.pulocationid

            # Feature engineering logic goes here
            df['pickup_hour'] = df['datetime'].dt.hour
            df['is_rush_hour'] = df['pickup_hour'].apply(lambda x: 1 if 7 <= x <= 9 or 16 <= x <= 19 else 0)
        
            logger.info("Temporal Feature for prediction generated successfully.")
            return df

        except Exception as e:
            logger.error(f"failed to engineer temporal features for prediction data {e}")
            raise RideDemandException(e,sys)

    def extract_historical_pickup_data(self)-> pd.DataFrame:
        try:
            self.cat_cols = ['pickup_hour','is_rush_hour']

            # Define the time range for historical data extraction
            now = datetime.now()
            end_date = datetime.now() - timedelta(minutes=now.minute, 
                                                  seconds=now.second, 
                                                  microseconds=now.microsecond)
            start_date = end_date - timedelta(hours=24)   

            logger.info('Retrieving the dataset from hopsworks feature store')

            ## login to feature store
            fs = self.project.get_feature_store()

            # Get the feature group
            fg = fs.get_feature_group(name="ridedemandprediction", version=1)
            query=fg.select_all()

            # creating a feature view
            # create a new feature view from the feature group
            #feature_view = fs.create_feature_view(name="ride_demand_prediction_fv",
                                                   # version=1,
                                                  #  description="ride demand historical data for prediction",
                                                 #   query=query)

            logger.info('hopsworks feature view created successfully')

            feature_view = fs.get_feature_view(name='ride_demand_prediction_fv', version= 1)

            # Materialize training dataset using Spark job
            version, jobs = feature_view.create_training_data(start_time = start_date,
                                                                end_time = end_date,
                                                                description="ride demand training data",
                                                                data_format="parquet",
                                                                write_options = {'use_spark': True}
                                                                )

            logger.info('Training data created successfully and materialized in hopsworks')
            logger.info(f"Data from {start_date} to {end_date} created and materialized Successfully")

            feature_view = fs.get_feature_view(name='ride_demand_fv', version= 1)

            df, _ = feature_view.get_training_data(training_dataset_version=1,
                                                read_options={"use_hive":False})
            
            logger.info('Data successfully retrieved from the feature store')
            
            df.set_index(['bin'], inplace=True)

            return df
        
        except Exception as e:
            logger.error(f"failed to extract historical pickup data {e}")
            raise RideDemandException(e, sys)

    def generate_lag_features_for_prediction(self, hist_df:pd.DataFrame,
                                              pred_df: pd.DataFrame)-> pd.DataFrame:
        try:
            #merging historical and prediction data
            df = pd.concat([hist_df, pred_df], ignore_index=True)
            df = df.sort_values(by='datetime')
            df.set_index('datetime', inplace=True)

            # getting only the last 24 hours of historical data
            df = df.last('24H')

            ## creating citywide pickup counts
            df_city = df.groupby('datetime').agg({'pickups':'sum'}).rename(columns={'pickups':'city_pickups'})
            df = df.merge(df_city, on='datetime', how='left')

            # city avg speed
            df['city_avg_speed'] = df['city_pickups'] / df['city_trip_distance']

            ## creating neighbor pickup counts
            df_neigh = df.groupby('pulocationid').agg({'pickups':'sum'}).rename(columns={'pickups':'neighbor_pickups_sum'})
            df = df.merge(df_neigh, on='pulocationid', how='left')

            ##  Final features needed for prediction
               # 'city_avg_speed',
              #  'zone_avg_speed',
             #   'zone_congestion_index',
            #    'pickups_lag_1h',
           #     'pulocationid',
          #      'pickups_lag_24h',
         #       'city_pickups_lag_1h',
           #     'neighbor_pickups_lag_1h'



            # Generating lag features
            df['pickup_lag1'] = df['pickups'].shift(1)
            df['pickup_lag24'] = df['pickups'].shift(24)
            
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