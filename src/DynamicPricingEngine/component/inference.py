import os, sys
import numpy as np
import pandas as pd
from pydantic import BaseModel, field_validator
import requests
from dotenv import load_dotenv
import hopsworks
from dateutil.relativedelta import relativedelta
from datetime import timedelta, datetime
from zoneinfo import ZoneInfo
from hsfs.feature import Feature
import zipfile
import io
import geopandas as gpd
import pickle
from sodapy import Socrata


load_dotenv()

from src.DynamicPricingEngine.logger.logger import logger
from src.DynamicPricingEngine.exception.customexception import RideDemandException
from src.DynamicPricingEngine.entity.config_entity import InferenceConfig
from src.DynamicPricingEngine.utils.common_utils import load_shapefile_from_zipfile, download_csv_from_web

"""" Things to Do
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

#class PredictionRequest(BaseModel):
 #   datetime: str
  #  pulocationid: int

   # @field_validator('datetime', pre=True)
 #   def parse_datetime(self, v):
  #      return pd.to_datetime(v)

#    @field_validator('pulocationid')
 #   def check_pulocationid(self, v):
  #      if not (1<=v<=263):
   #         raise ValueError("PULocationID must be between 1-263")
        
    #    return v

class Inference:
    def __init__(self, config: InferenceConfig):
        try:
            self.config = config
            self.api_key = os.getenv('API_KEY')
            self.hopsworks_api = os.getenv('HOPSWORKS_API_KEY')
            self.ny_tz = ZoneInfo("America/New_York")

        #Cache neighbor dictionary
            self._neighbor_dict = None
            self._neighbor_cache_path = os.path.join(self.config.shapefile_dir, "neighbors.pkl")

        except Exception as e:
            logger.error("Error initializing Inference Pipeline", e)
            raise RideDemandException(e, sys)

    def _get_neighbor_dict(self) -> dict:
        if self._neighbor_dict is not None:
            return self._neighbor_dict

        if os.path.exists(self._neighbor_cache_path):
            try:
                with open(self._neighbor_cache_path, "rb") as f:
                    _neighbor_dict = pickle.load(f)
                logger.info("Loaded neighbor dictionary from cache")
                return _neighbor_dict
            except Exception as e:
                logger.info(f"Failed to load neighbor cache: {e}")

        logger.info('loading neighbor feature in _get_neighbor dict')
        zones_gdf = load_shapefile_from_zipfile(self.config.taxi_zone_shapefile_url,
                                            self.config.shapefile_dir)
        
        if zones_gdf is None:
            logger.error("Failed to acquire shapefile after all retries.")
            return {}

        # Spatial Join Logic
        logger.info("Calculating adjacency (touches)...")
        
        zones_gdf_left = zones_gdf.rename(columns={"LocationID": "LocationID_left"})
        zones_gdf_right = zones_gdf.rename(columns={"LocationID": "LocationID_right"})
        neighbors_df = gpd.sjoin(zones_gdf_left, zones_gdf_right, how="left", predicate="touches")
        neighbors_df = neighbors_df[neighbors_df['LocationID_left'] != neighbors_df['LocationID_right']]
        _neighbor_dict = (neighbors_df.groupby('LocationID_left')['LocationID_right']
                                .apply(lambda s: sorted(list(set(s))))
                                .to_dict())
        try:
            os.makedirs(self.config.shapefile_dir, exist_ok=True)
            with open(self._neighbor_cache_path, "wb") as f:
                pickle.dump(_neighbor_dict, f)

        except Exception as e:
            logger.error(f"Failed to persist neighbor cache: {e}")

        logger.info("neighbor dict retrieved successfully")
        return _neighbor_dict

    def get_nyc_prediction_weather_data(self) -> pd.DataFrame:
        try:
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
            print(f"Fetching data for: {datetime.now(self.ny_tz).strftime('%Y-%m-%d %H')}:00:00")

            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            # Extract current conditions
            current = data.get("currentConditions")
            if not current:
                logger.info("No current conditions found in response.")
                return None

            # Convert to DataFrame
            df_hours = pd.DataFrame([current])
            cols = ['datetime', 'temp', 'humidity']
            df_hours = df_hours[cols]

            #Ensure datetime includes today's date so localization works correctly
            today_date = datetime.now().strftime('%Y-%m-%d')

            # Localize to UTC before flooring to make it timezone-aware
            df_hours['datetime'] = pd.to_datetime(today_date + ' ' + df_hours['datetime']).dt.tz_localize('UTC').dt.floor('H') + timedelta(hours=1)

            logger.info(f"Successfully retrieved weather data.")
            return df_hours

        except Exception as e:
            logger.error(f"Failed to extract weather data: {e}")
            raise RideDemandException(e,sys)    
        

    def engineer_temporal_prediction_features(self, weather_df: pd.DataFrame)-> pd.DataFrame:
        try:
            # Extract temporal features
            weather_df['pickup_hour'] = weather_df['datetime'].dt.hour
            weather_df['is_rush_hour'] = weather_df['pickup_hour'].apply(lambda x: 1 if 7 <= x <= 9 or 16 <= x <= 19 else 0)
            weather_df.rename(columns={'datetime':'bin'}, inplace=True)

            logger.info("Temporal Feature for prediction generated successfully.")
            return weather_df

        except Exception as e:
            logger.error(f"failed to engineer temporal features for prediction data {e}")
            raise RideDemandException(e,sys)
            

    def extract_historical_pickup_data(self, start_date=None, end_date=None) -> pd.DataFrame:
        try:
            project = hopsworks.login(project='RideDemandPrediction', api_key_value=self.hopsworks_api)
            fs = project.get_feature_store()

            end_date_utc = datetime.now(self.ny_tz).replace(minute=0, second=0, microsecond=0)

            start_date_utc = (end_date_utc - relativedelta(hours=24))
           
            fh_start = start_date_utc.strftime('%Y-%m-%d %H:%M:%S')
            fh_end = end_date_utc.strftime('%Y-%m-%d %H:%M:%S')

            prediction_fg = fs.get_or_create_feature_group(
                name="demand_predictions",
                version=1,
                primary_key=['pulocationid', 'bin_str'],
                event_time='bin',
                description="Logs of model predictions for evaluation"
            )
            if start_date is None or end_date is None:
                start_date =fh_start
                end_date = fh_end

            final_features = ['temp', 'humidity', 'pickup_hour', 'is_rush_hour', 'pulocationid', 
                            'pickups', 'bin', 'city_avg_speed', 'zone_avg_speed', 'zone_congestion_index',]

            # Use the explicit query format
            query = prediction_fg.select(final_features).filter(
                (prediction_fg.bin >= start_date) & (prediction_fg.bin <= end_date)
            )

            historic_df = query.read()

            logger.info(f"Successfully retrieved {len(historic_df)} rows for window: {start_date} to {end_date}")

            return historic_df

        except Exception as e:
            logger.error(f"Failed to extract historical pickup data: {e}")
            raise RideDemandException(e,sys)
        
    def citywide_hourly_demand(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            # Aggregate citywide pickups per hour
            city_demand = (
                df.groupby('bin')['pickups']
                .sum()
                .reset_index()
                .rename(columns={'pickups': 'city_pickups'})
            )

            # Merge back into main df
            df = df.merge(city_demand, on='bin', how='left')

            return df

        except Exception as e:
            logger.error("Unable to engineer citywide hourly demand features")
            raise RideDemandException(e,sys)

    def generate_neighbor_features(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            print("generating neighbor features")
            neighbor_dict = self._get_neighbor_dict()

            print("generated features")
            #
            # Build neighbor pairs in Pandas, then convert to Dask
            neighbor_pdf = pd.DataFrame(
                [(zone, n) for zone, neighs in neighbor_dict.items() for n in neighs],
                columns=['pulocationid', 'neighbor_id']
            ).fillna(-1)

            neighbor_pdf['pulocationid'] = neighbor_pdf['pulocationid'].astype(
                df['pulocationid'].dtype
                )
            neighbor_pdf['neighbor_id'] = neighbor_pdf['neighbor_id'].astype(
                df['pulocationid'].dtype
                )

            # Prepare neighbor pickups
            df_neighbors = df[['pulocationid', 'bin', 'pickups']].rename(
                columns={'pulocationid': 'neighbor_id', 'pickups': 'neighbor_pickups'}
            )

            merged = neighbor_pdf.merge(df_neighbors, on='neighbor_id', how='left')

            neighbor_demand_df = (
                merged.groupby(['pulocationid', 'bin'])['neighbor_pickups']
                .sum()
                .reset_index()
                .rename(columns={'neighbor_pickups': 'neighbor_pickups_sum'})
            )

            df = df.merge(neighbor_demand_df, on=['pulocationid', 'bin'], how='left')
            df = df.rename(columns={'neighbor_pickups_sum_y':'neighbor_pickups_sum'})

            df['neighbor_pickups_sum'] = df['neighbor_pickups_sum'].fillna(0)

            return df

        except Exception as e:
            print("Unable to generate neighbor features")
            raise e


    def get_zone_speeds(self, df):
        try:    
            # Load the official NYC Taxi Zone lookup table
            app_token = os.getenv('NYC_OPEN_DATA_APP_TOKEN')
            client = Socrata("data.cityofnewyork.us", app_token=app_token) # App Token is better if you have one
            
            # Get the most recent 2,000 speed records in one go
            logger.info('loading the dataset')
            results = client.get("i4gi-tjb9", limit=2000, order="data_as_of DESC")
            logger.info('data successfully downloaded from socatrated')
            speed_data = pd.DataFrame.from_records(results)
            speed_data['speed'] = pd.to_numeric(speed_data['speed'])
            
            #Create a dictionary of Borough Averages as a backup
            borough_map = speed_data.groupby('borough')['speed'].mean().to_dict()

            # 4. MAP DATA: Match Zone IDs to speeds in memory (No API calls inside this loop!)
            def fast_map(row):
                try:
                    # Get the zone name for the ID
                    zone_df = download_csv_from_web(self.config.zone_lookup_table_url)
                    z_info = zone_df[zone_df['LocationID'] == row['pulocationid']]
                    if z_info.empty: return None
                    
                    z_name_raw = z_info.iloc[0]['Zone']
                    bor = z_info.iloc[0]['Borough']
                    
                    # Ensure z_name is a string before using it in str.contains
                    if pd.isna(z_name_raw):
                        z_name = "" # Use empty string for pattern if NaN
                    else:
                        z_name = str(z_name_raw)

                    # Check if any live segment name contains the zone name
                    matched_speeds = speed_data[speed_data['link_name'].str.contains(z_name, case=False, na=False)]
                    
                    if not matched_speeds.empty:
                        return matched_speeds['speed'].mean()
                    return borough_map.get(bor, None) # Use borough average if specific zone isn't found
                
                except Exception as e:
                    raise RideDemandException(e,sys)
                
            # Apply the logic locally
            df['zone_avg_speed'] = df.apply(fast_map, axis=1)
            return df
        
        except Exception as e:
            raise RideDemandException(e,sys)
        
    def congestion_features(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            # zone Congestion index
            df['zone_congestion_index'] = np.where(df['zone_avg_speed'] 
                                                   > 0, 1.0 / df['zone_avg_speed'], np.nan)

            # Compute citywide average speed per hour            
            city_avg_speed = (
                df.groupby('bin')['zone_avg_speed']
                .mean()
                .reset_index()
                .rename(columns={'zone_avg_speed': 'city_avg_speed'})
            )

            df = df.merge(city_avg_speed, on=['bin'], how='left')

            return df

        except Exception as e:
            logger.error("Unable to generate zone-level features")
            raise e

    def generate_lag_features_for_prediction(hist_df:pd.DataFrame,
                                              pred_df: pd.DataFrame)-> pd.DataFrame:
        try:
            if 'pickups' not in pred_df.columns:
                pred_df['pickups'] = np.nan

            if 'city_pickups' not in pred_df.columns:
                pred_df['city_pickups'] = np.nan
            if 'neighbor_pickups_sum' not in pred_df.columns:
                pred_df['neighbor_pickups_sum'] = np.nan

            # Ensure 'bin' columns are proper datetime objects for sorting
            hist_df['bin'] = pd.to_datetime(hist_df['bin'])
            pred_df['bin'] = pd.to_datetime(pred_df['bin'])

            # Concatenate historical and prediction data
            df_final = pd.concat([hist_df, pred_df], axis = 0, ignore_index=True)

            #Sort the combined DataFrame for accurate lag calculation
            df_final = df_final.sort_values(['pulocationid', 'bin'])

            #Generate lag features per pulocationid group
            def make_lags_per_zone(group):
                group['pickups_lag_1h'] = group['pickups'].shift(1)
                group['pickups_lag_24h'] = group['pickups'].shift(24)
                group['city_pickups_lag_1h'] = group['city_pickups'].shift(1)
                group['neighbor_pickups_lag_1h'] = group['neighbor_pickups_sum'].shift(1)
                return group

            df_final = df_final.groupby('pulocationid', group_keys=False).apply(make_lags_per_zone)
            df_final.fillna(0, inplace=True)

            logger.info("Lag features for prediction generated successfully.")
            return df_final

        except Exception as e:
            logger.error(f"failed to engineer lag features for prediction data {e}")
            raise RideDemandException(e,sys)
        
    def final_data(df):
        try:
            df = df.sort_values(by=['pulocationid', 'bin'])
            df['bin'] = pd.to_datetime(df['bin'], utc=True) #Ensure it's datetime
            if df['bin'].dt.tz is not None: # If it's timezone-aware, convert to naive
                df['bin'] = df['bin'].dt.tz_convert(None)
            
            df = df.set_index('bin')
            df = df.drop(['city_pickups', 'neighbor_pickups_sum', 'pickups'], axis=1, errors='ignore')

            ny_tz = ZoneInfo("America/New_York")    
            target_time = datetime.now(ny_tz).replace(tzinfo=None) 
            
            # Round to the top of the hour (e.g., 10:45 -> 10:00)
            target_hour = target_time.replace(minute=0, second=0, microsecond=0)
            print(target_hour)

            # Filtering for the current hour and forward
            df_filtered = df.loc[df.index >= target_hour]

            return df_filtered
        
        except Exception as e:
            logger.error()
            raise RideDemandException(e,sys)

    
    def push_engineered_features_to_feature_store(self):
        pass

    def make_predictions(self):
        pass

    def initiate_prediction_pipeline(self):
        pass