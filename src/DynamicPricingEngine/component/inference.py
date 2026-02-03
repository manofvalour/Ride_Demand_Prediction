import os, sys
import numpy as np
import pandas as pd
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
import joblib
import time
import json
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential, retry_if_exception_message

load_dotenv()

from src.DynamicPricingEngine.logger.logger import logger
from src.DynamicPricingEngine.exception.customexception import RideDemandException
from src.DynamicPricingEngine.entity.config_entity import InferenceConfig
from src.DynamicPricingEngine.utils.common_utils import load_shapefile_from_zipfile, download_csv_from_web


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

load_dotenv()

class Inference:
    def __init__(self, config:InferenceConfig):
        try:
            self.config = config
            self.weather_api_key = os.getenv('API_KEY')
            hopsworks_api = os.getenv('HOPSWORKS_API_KEY')
            self.ny_tz = ZoneInfo("America/New_York")
            self.project = hopsworks.login(project='RideDemandPrediction', api_key_value=hopsworks_api)

        #Cache neighbor dictionary
            self._neighbor_dict = None
            self._neighbor_cache_path = os.path.join(self.config.shapefile_dir, "neighbors.pkl")

        except Exception as e:
            print("Error initializing Inference Pipeline", e)
            raise e

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
                logger.error(f"Failed to load neighbor cache: {e}")

        print('loading neighbor feature in _get_neighbor dict')
        zones_gdf = load_shapefile_from_zipfile(self.config.taxi_zone_shapefile_url,
                                                self.config.shapefile_dir)
        
        zones_gdf.to_file(self.config.geojson_output_path, driver="GeoJSON")
        
        if zones_gdf is None:
            print("Failed to acquire shapefile after all retries.")
            return e

        # Spatial Join Logic
        print("Calculating adjacency (touches)...")
        
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
            api_key = self.weather_api_key
            location = "New York, NY, United States"
            base_url = self.config.weather_data_url

            params = {
                "unitGroup": "us",
                "key": api_key,
                "include": "current",
                "contentType": "json"
            }

            url = f"{base_url}/{location}/today"
            target_time = datetime.now(self.ny_tz)

            if target_time.minute < 35:
               logger.info(f"Fetching data for: {(datetime.now(self.ny_tz)).strftime('%Y-%m-%d %H')}:00:00")
          
            else:
              logger.info(f"Fetching data for: {(datetime.now(self.ny_tz)+timedelta(hours=1)).strftime('%Y-%m-%d %H')}:00:00")
            

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
            cols = ['datetime', 'humidity', 'precip','windspeed', 'feelslike', 'visibility']
            df_hours = df_hours[cols]

            #Ensure datetime includes today's date so localization works correctly
            today_date = datetime.now(self.ny_tz).strftime('%Y-%m-%d')

            # Localizing to UTC before flooring to make it timezone-aware
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
            weather_df['pickup_month'] = weather_df['datetime'].dt.month
            weather_df['day_of_week'] = weather_df['datetime'].dt.dayofweek
            weather_df['is_night_hour'] = (~weather_df['pickup_hour'].between(7, 20)).astype(int)
            weather_df.rename(columns={'datetime':'bin'}, inplace=True)
            
            logger.error("Temporal Feature for prediction generated successfully.")
            return weather_df

        except Exception as e:
            logger.error(f"failed to engineer temporal features for prediction data {e}")
            raise RideDemandException(e,sys)
    
    def extract_historical_pickup_data(self) -> pd.DataFrame:
        try:
            final_features = ['pickup_month', 'humidity', 'precip','windspeed', 'feelslike', 'visibility',
                                'pickup_hour', 'day_of_week','is_rush_hour', 'is_night_hour',
                                'pulocationid', 'bin', 'target_yellow', 'target_green', 'target_hvfhv',
                              'zone_congestion_index', "city_congestion_index"]

            fs = self.project.get_feature_store()

            target_time = datetime.now(self.ny_tz)
            target_time = target_time.replace(tzinfo=None)

            if target_time.minute < 35:
              end_date_utc = target_time.replace(minute=0, second=0, microsecond=0)
            else:
              end_date_utc = target_time.replace(minute=0, second=0, microsecond=0) + relativedelta(hours=1)

            start_date_utc = (end_date_utc - relativedelta(hours=24))

            fh_start = start_date_utc.strftime('%Y-%m-%d %H:%M:%S')
            fh_end = end_date_utc.strftime('%Y-%m-%d %H:%M:%S')

            try:
              prediction_fg = fs.get_feature_group(
                  name="demandpred",
                  version=1)

              if prediction_fg is None:
                raise ValueError("Feature Group is empty, falling back to 1-year history.")

              query = prediction_fg.select(final_features).filter(
                  (prediction_fg.get_feature("bin") >= fh_start) & (prediction_fg.get_feature("bin") < fh_end))

              historical_df = query.read()
              if historical_df.empty:
                raise ValueError("Recent data empty, falling back to 1-year history.")

            except Exception as e:
                logger.info(f"Switching to fallback logic: {e}")

                fg = fs.get_feature_group(
                    name = 'nycdemandprediction',
                    version=1,
                )
                fh_start = start_date_utc - relativedelta(months=12)
                fh_end = fh_start + relativedelta(hours=24)

                query = fg.select(final_features).filter(
                    (fg.get_feature('bin') >=fh_start) & (fg.get_feature('bin') < fh_end))
                historical_df = query.read()

            historical_df['bin'] = pd.to_datetime(historical_df['bin']) + pd.DateOffset(years=1)
            logger.info(f"Successfully retrieved {len(historical_df)} rows for window: {fh_start} to {fh_end}")

            return historical_df

        except Exception as e:
            logger.error(f"Failed to extract historical pickup data: {e}")
            raise RideDemandException(e,sys)


    def citywide_hourly_demand(self, df: pd.DataFrame) -> pd.DataFrame:
      try:
          services = ['target_yellow', 'target_green', 'target_hvfhv']
              
          #Aggregate citywide pickups per hour
          for s in services:
              city_demand = (
                  df.groupby('bin')[s]
                  .sum()
                  .reset_index()
                  .rename(columns={s: f'{s}_city_hour_pickups'})
              )
              # Merge back into main df
              df = df.merge(city_demand, on='bin', how='left')
              
          return df

      except Exception as e:
          logger.error("Unable to engineer citywide hourly demand features")
          raise RideDemandException(e,sys)

    def generate_neighbor_features(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            neighbor_dict = self._get_neighbor_dict()

            # Build neighbor pairs in Pandas, then convert to Dask
            neighbor_pdf = pd.DataFrame(
                [(zone, n) for zone, neighs in neighbor_dict.items() for n in neighs],
                columns=['pulocationid', 'neighbor_id']
            ).fillna(-1)

            rename_map = {
                'pulocationid': 'neighbor_id',
                'target_yellow': 'yellow_neighbor_pickups',
                'target_green': 'green_neighbor_pickups',
                'target_hvfhv': 'hvfhv_neighbor_pickups'
            }
            
            # Select original columns from df for df_neighbors
            df_neighbors_orig_cols = ['pulocationid', 'bin', 'target_yellow', 
                                      'target_green', 'target_hvfhv']

            df_neighbors = df[df_neighbors_orig_cols].rename(columns=rename_map)
            merged = neighbor_pdf.merge(df_neighbors, on='neighbor_id', how='left')

            # List of the *actual* column names in 'merged' that represent neighbor pickups
            neighbor_pickup_cols_in_merged = [
                'yellow_neighbor_pickups','green_neighbor_pickups',
                'hvfhv_neighbor_pickups']

            # Corresponding desired final column names in the main df
            final_output_col_names = [
                'neighbor_pickups_target_yellow',
                'neighbor_pickups_target_green',
                'neighbor_pickups_target_hvfhv'
            ]

            for i, merged_col_name in enumerate(neighbor_pickup_cols_in_merged):
                output_col_name = final_output_col_names[i]

                neighbor_demand_df = (
                    merged.groupby(['pulocationid', 'bin'])[merged_col_name]
                    .sum()
                    .reset_index()
                    .rename(columns={merged_col_name: output_col_name})
                )

                df = df.merge(neighbor_demand_df, on=['pulocationid', 'bin'], 
                              how='left', suffixes=("", '_y'))
                df[output_col_name] = df[output_col_name].fillna(0)

            return df
        except Exception as e:
            logger.error("Unable to generate neighbor features", e)
            raise RideDemandException(e,sys)
        

    def get_zone_speeds(self, df):
        try:
            app_token = os.getenv('NYC_OPEN_DATA_APP_TOKEN')

            # Using a context manager ensures the session closes properly
            with Socrata("data.cityofnewyork.us", app_token) as client:
                logger.info('Loading speed dataset...')
                results = client.get("i4gi-tjb9", limit=2000, order="data_as_of DESC")
                speed_data = pd.DataFrame.from_records(results)
            
            # Pre-process speed data once
            speed_data['speed'] = pd.to_numeric(speed_data['speed'], errors='coerce')
            borough_map = speed_data.groupby('borough')['speed'].mean().to_dict()

            logger.info('Downloading zone lookup table...')
            zone_df = download_csv_from_web(self.config.zone_lookup_table_url)
            
            # First, attach Zone/Borough info to your main dataframe
            df = df.merge(
                zone_df[['LocationID', 'Zone', 'Borough']], 
                left_on='pulocationid', 
                right_on='LocationID', 
                how='left'
            )

            # Defining the localized mapping logic
            def calculate_speed(row):
                z_name = str(row['Zone']) if pd.notna(row['Zone']) else ""
                bor = row['Borough']

                if z_name:
                    # Filter speed_data for matching link names
                    matched = speed_data[speed_data['link_name'].str.contains(z_name, case=False, na=False)]
                    if not matched.empty:
                        return matched['speed'].mean()
                
                # Fallback to borough average if no specific zone match
                return borough_map.get(bor)

            logger.info('Calculating zone speeds...')
            df['zone_avg_speed'] = df.apply(calculate_speed, axis=1)

            # Cleanup extra columns from merge if necessary
            df.drop(columns=['LocationID', 'Zone', 'Borough'], inplace=True)
            
            return df

        except Exception as e:
            logger.error(f"Error in get_zone_speeds: {e}")
            raise RideDemandException(e,sys)
        

    def congestion_features(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            # Compute citywide average speed per hour            
            city_avg_speed = (
                df.groupby('bin')['zone_avg_speed']
                .mean()
                .reset_index()
                .rename(columns={'zone_avg_speed': 'city_avg_speed'})
            )


            df = df.merge(city_avg_speed, on=['bin'], how='left')
            global_avg = df['city_avg_speed'].mean()
            df['city_avg_speed'] = df['city_avg_speed'].fillna(global_avg)

            fallback_city_congestion = 1.0 / global_avg
            df['city_congestion_index'] = np.where(df['city_avg_speed'] 
                                                   > 0, 1.0 / df['city_avg_speed'], fallback_city_congestion)
            
            df['zone_avg_speed'] = df['zone_avg_speed'].combine_first(df['city_avg_speed'])
            
            # zone Congestion index
            global_zone_avg = df['zone_avg_speed'].mean()
            fallback_zone_congestion = 1.0 /global_zone_avg
            df['zone_congestion_index'] = np.where(df['zone_avg_speed'] 
                                                   > 0, 1.0 / df['zone_avg_speed'], fallback_zone_congestion)
            
            df.drop(columns=['city_avg_speed', 'zone_avg_speed'], inplace=True)

            return df

        except Exception as e:
            logger.error("Unable to generate zone-level features")
            raise RideDemandException(e,sys)
    
    def engineer_autoregressive_signals(self, hist_df: pd.DataFrame, 
                                        pred_df:pd.DataFrame) -> pd.DataFrame:
        try:

            # Define the three targets we are tracking
            services = ['target_yellow', 'target_green', 'target_hvfhv']
            neighbors = [
                'neighbor_pickups_target_yellow',
                'neighbor_pickups_target_green',
                'neighbor_pickups_target_hvfhv'
            ]
            city = ['target_yellow_city_hour_pickups', 'target_green_city_hour_pickups', 
                    'target_hvfhv_city_hour_pickups',]

            if not all(service_col in pred_df.columns for service_col in services):
                pred_df['target_yellow'] = np.nan
                pred_df['target_green'] = np.nan
                pred_df['target_hvfhv'] = np.nan

            if not all(city_col in pred_df.columns for city_col in city):
                pred_df['target_yellow_city_hour_pickups'] = np.nan
                pred_df['target_green_city_hour_pickups'] = np.nan
                pred_df['target_hvfhv_city_hour_pickups'] = np.nan
                
            if not all(neighbor_col in pred_df.columns for neighbor_col in neighbors):
                pred_df['neighbor_pickups_target_yellow'] = np.nan
                pred_df['neighbor_pickups_target_green'] = np.nan
                pred_df['neighbor_pickups_target_hvfhv'] = np.nan

            print('hist_df: ', hist_df.shape)
            print('pred_df: ', pred_df.shape)

            # Ensure 'bin' columns are proper datetime objects for sorting
            hist_df['bin'] = pd.to_datetime(hist_df['bin'])
            pred_df['bin'] = pd.to_datetime(pred_df['bin'])

            # Concatenate historical and prediction data
            df_final = pd.concat([hist_df, pred_df], axis = 0, ignore_index=True)

            df_final.drop_duplicates(['bin',"pulocationid"], inplace=True)

            #Sort the combined DataFrame for accurate lag calculation
            pdf = df_final.sort_values(['pulocationid', 'bin'])

            for s in services:
                for l in [1, 24]:
                    pdf[f'{s}_lag_{l}h'] = pdf.groupby('pulocationid')[s].shift(l)

            for c in city:
                pdf[f'{c}_lag_1h'] = pdf.groupby('pulocationid')[c].shift(1)

            for n in neighbors:
                pdf[f'{n}_lag_1h'] = pdf.groupby('pulocationid')[n].shift(1)

            pdf.fillna(0, inplace=True)
            df = df_final.merge(pdf, on=['pulocationid', 'bin'], how='left', suffixes=('', '_y'))
            
            # Dropping redundant columns from the merge
            cols_to_drop = [c for c in df.columns if c.endswith('_y')]
            df = df.drop(cols_to_drop, axis=1)

            return df

        except Exception as e:
            logger.error("Failed to generate multi-output autoregressive features")
            raise RideDemandException(e,sys)
            
    def final_data(self, df):
        try:
            df = df.sort_values(by=['pulocationid', 'bin'])
            df['bin'] = pd.to_datetime(df['bin'], utc=True) #Ensure it's datetime
            if df['bin'].dt.tz is not None: # If it's timezone-aware, convert to naive
                df['bin'] = df['bin'].dt.tz_convert(None)
            
            df = df.set_index('bin')

            targets=['target_yellow', 'target_green', 'target_hvfhv']
            to_drop = targets + ['pickup_datetime', 'bin','target_yellow_city_hour_pickups',
                                 'target_green_city_hour_pickups', 'target_hvfhv_city_hour_pickups',
                                 'neighbor_pickups_target_yellow', 'neighbor_pickups_target_green',
                                 'neighbor_pickups_target_hvfhv', 'zone_avg_speed','city_avg_speed']

            df = df.drop(columns=to_drop, axis=1, errors='ignore')
    
            target_time = datetime.now(self.ny_tz).replace(tzinfo=None)
            
            if target_time.minute < 40:
              target_time = target_time.replace(hour=target_time.hour)
            else:
              target_time = target_time.replace(hour=target_time.hour + 1)
            
            # Round to the top of the hour (e.g., 10:45 -> 10:00)
            target_hour = target_time.replace(minute=0, second=0, microsecond=0)
           # print(target_hour)

            # Filtering for the current hour and forward
            df_filtered = df.loc[df.index == target_hour]
            df_filtered = df_filtered

            return df_filtered
        
        except Exception as e:
            logger.error("failed to process data")
            raise RideDemandException(e,sys)

    def download_model_and_load(self):
    # Attempt the connection/download up to 3 times
      for attempt in range(3):
          try:
              mr = self.project.get_model_registry()

              #Get metadata and download files
              model_meta = mr.get_best_model("ride_demand_prediction_model", metric = 'rmse', direction='min')

              print(f"Attempt {attempt + 1}: Downloading model...")
              model_dir = model_meta.download() # This is the root directory of the downloaded artifacts

              # Identify and Load the actual model file
              possible_model_names = ["model.pkl", "model.joblib", "ride_demand_prediction_model.pkl"]
              found_model_file = None

              # Check the root of the downloaded directory first
              for name in possible_model_names:
                  potential_path = os.path.join(model_dir, name)
                  if os.path.exists(potential_path):
                      found_model_file = potential_path
                      break

              # checking a common 'model' subdirectory (e.g., for MLflow-saved models)
              if found_model_file is None:
                  model_subdir = os.path.join(model_dir, "model") # Common MLflow subdirectory
                  for name in possible_model_names:
                      potential_path = os.path.join(model_subdir, name)
                      if os.path.exists(potential_path):
                          found_model_file = potential_path
                          break

              if found_model_file is None:
                  raise FileNotFoundError(f"No model file found in '{model_dir}' or its 'model' subdirectory with expected names.")

              loaded_model = joblib.load(found_model_file)
              logger.info(f"Model loaded successfully from: {found_model_file}")

              return loaded_model

          except (ConnectionError, Exception) as e:
              logger.error(f"Attempt {attempt + 1} failed: {e}")
              if attempt < 2:
                  time.sleep(5) # Wait 5 seconds before retrying
              else:
                  logger.error(f"unable to load the model from hopsworks, {e}")
                  raise RideDemandException(e, sys)

    def prepare_and_predict(self, model, final_df):
        try:
            #Definining the EXACT feature list used during training
            final_features = ['pickup_month', 'city_congestion_index', 'zone_congestion_index', 
                              'humidity', 'precip','windspeed', 'feelslike', 'visibility', 
                              'pickup_hour', 'day_of_week','is_rush_hour', 'is_night_hour', 
                              'target_yellow_lag_1h','target_yellow_lag_24h', 
                              'target_green_lag_1h', 'target_green_lag_24h',
                              'target_hvfhv_lag_1h', 'target_hvfhv_lag_24h',
                              'target_yellow_city_hour_pickups_lag_1h',
                              'target_green_city_hour_pickups_lag_1h', 'pulocationid',
                              'target_hvfhv_city_hour_pickups_lag_1h',
                              'neighbor_pickups_target_yellow_lag_1h',
                              'neighbor_pickups_target_green_lag_1h',
                              'neighbor_pickups_target_hvfhv_lag_1h',
                            ]

            #Reordering columns and handle missing data
            X = final_df[final_features].copy() # Added .copy() to prevent SettingWithCopyWarning
            X.reset_index(drop=True, inplace=True)

            #Convert categorical features to 'category' dtype as they were during training
            categorical_features_for_prediction = ['day_of_week','is_night_hour','pickup_hour',
                                                'is_rush_hour','pulocationid', "pickup_month"]
            for col in categorical_features_for_prediction:
                if col in X.columns:
                    X[col] = X[col].astype('category')

            #Generate Predictions
            predictions = model.predict(X)

            targets=['target_yellow', 'target_green', 'target_hvfhv']
            pred = pd.DataFrame(predictions, columns=targets)
            
            # Rounding the prediction
            pred = pred.clip(lower=0).round().astype(int)

            features_to_save = ['pickup_month', 'humidity', 'precip','windspeed', 'feelslike', 'visibility', 
                                'pickup_hour', 'day_of_week','is_rush_hour', 'is_night_hour', 
                                'pulocationid', 'zone_congestion_index', "city_congestion_index"]
            

            #Attach predictions back to a readable dataframe
            results = final_df.copy()

            results = results[features_to_save]
            results= results.reset_index(drop=False)
            results = pd.concat([results, pred], axis=1)
            results['bin'] = results['bin'].astype(str)
            predictions_dict = results.set_index('pulocationid').to_dict(orient='index')

            final_output = {
                "metadata": {
                    "generated_at": datetime.now(self.ny_tz).isoformat(),
                    "total_zones": len(results),
                    "prediction_window": results['bin'].iloc[0] if 'bin' in results.columns else "Unknown"
                },
                "predictions": predictions_dict
            }

            # Saving the JSON file
            with open(self.config.predictions_output_path, 'w') as f:
                json.dump(final_output, f, indent=4)
            
            logger.info(f"Successfully saved to {self.config.predictions_output_path}")

            logger.info('Prediction completed!')

            return results
        
        except Exception as e:
            logger.error(f"Error in prepare_and_predict: {e}")
            raise RideDemandException(e,sys)
      
      
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=5, max=60),
        retry = retry_if_exception_type(Exception),
        before_sleep=lambda retry_state: print(f"Retrying Hopsworks push... Attempt {retry_state.attempt_number}"),
        reraise=True)
      
    def push_prediction_to_feature_store(self, pred, hist_data)-> None:
        try:
            type_mapping = {
                'bin': 'datetime64[ns]', 'is_rush_hour': 'int8',
                'is_night_hour': 'int8', 'day_of_week': 'int16',
                'pickup_month': 'int16', 'pickup_hour': 'int16',
                'pulocationid': 'int16', 'zone_congestion_index': 'float32',
                'city_congestion_index': 'float32', 
                'humidity': 'float64', 'precip': 'float64',
                'windspeed': 'float64', 'feelslike': 'float64', 
                'visibility': 'float64', 'target_yellow': 'float64', 
                'target_green': 'float64', 'target_hvfhv': 'float64',
            }

            for col, dtype in type_mapping.items():
                if col in pred.columns:
                    pred[col] = pred[col].astype(dtype)

            # Clean up potentially redundant index columns if they exist
            if 'level_0' in pred.columns:
                pred.drop(columns=['level_0'], inplace=True)
            if 'index' in pred.columns:
                pred.drop(columns=['index'], inplace=True)

            # Ensure 'bin' is a column and create 'bin_str' from it.
            if 'bin' in hist_data.columns:
                hist_data['bin_str'] = hist_data['bin'].astype(str)
                
            if 'bin' in pred.columns:
                pred['bin_str'] = pred['bin'].astype(str)
            else:
                # Fallback if 'bin' somehow became the index again
                pred['bin_str'] = pred.index.astype(str)
                pred.reset_index(inplace=True)

            ## retrieving the feature group
            ##initializing and login to hopswork feature store
            fs = self.project.get_feature_store()
            prediction_fg = fs.get_feature_group(
                    name="demandpred",
                    version=1,
            )

            if prediction_fg is not None:
              ## inserting new data in the feature group created above
              prediction_fg.insert(pred, write_options = {'wait_for_job': False, 'use_spark':True})

            else:
              prediction_fg = fs.create_feature_group(
                    name="demandpred",
                    version=1,
                    primary_key=['pulocationid', 'bin_str'],
                    event_time='bin',
                    description="Logs of model predictions for evaluation"
                )
              
              data = pd.concat([pred, hist_data], axis=0)
  
              ## inserting new data in the feature group created above
              prediction_fg.insert(data, write_options = {'wait_for_job': False, 'use_spark':True})
    
            logger.info('data successfully added to hopsworks feature group')

        except Exception as e:
            logger.error('unable to store the dataset to feature store')
            raise  RideDemandException(e,sys)


    def initiate_inference(self)-> pd.DataFrame:
        try:

            logger.info('Extracting the prediction Data...')

            weather_df = self.get_nyc_prediction_weather_data()
            pred_df = self.engineer_temporal_prediction_features(weather_df)
            hist_df = self.extract_historical_pickup_data()

            unique_pulocationids = pd.DataFrame({'pulocationid': hist_df.pulocationid.unique()})
            pred_df = pd.merge(unique_pulocationids, pred_df, how='cross')
            pred_df = self.get_zone_speeds(pred_df)
            pred_df = self.congestion_features(pred_df)

            hist_dfs = self.citywide_hourly_demand(hist_df)
            hist_dfs = self.generate_neighbor_features(hist_dfs)

            df = self.engineer_autoregressive_signals(hist_dfs, pred_df)
            df2 = self.final_data(df)

            logger.info('Inference data created Successfully')

            model = self.download_model_and_load()
            prediction = self.prepare_and_predict(model, df2)
            self.push_prediction_to_feature_store(prediction, hist_df)

        except Exception as e:
            logger.error(f'Unable to initiate model training, {e}')
            raise RideDemandException(e,sys)