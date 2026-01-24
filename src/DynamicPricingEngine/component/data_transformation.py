import os, sys, pickle
import pandas as pd
import numpy as np
import geopandas as gpd
import dask.dataframe as dd
from datetime import datetime, timedelta
import hopsworks
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv
import time
from pathlib import Path
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential, retry_if_exception_message


load_dotenv()

from src.DynamicPricingEngine.logger.logger import logger
from src.DynamicPricingEngine.exception.customexception import RideDemandException
from src.DynamicPricingEngine.entity.config_entity import DataTransformationConfig
from src.DynamicPricingEngine.utils.common_utils import load_shapefile_from_zipfile
from src.DynamicPricingEngine.utils.data_ingestion_utils import time_subtract

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        
        self.config = config
        
        #Read both datasets with Dask
        self.taxi_df = dd.read_parquet(config.taxi_data_local_file_path)
        self.weather_df = dd.read_csv(config.weather_data_local_file_path)

        #Ensure datetime types
        for col in ['tpep_pickup_datetime', 'tpep_dropoff_datetime']:
            if col in self.taxi_df.columns:
                self.taxi_df[col] = dd.to_datetime(self.taxi_df[col], errors='coerce')

         #Cache neighbor dictionary
        self._neighbor_dict = None
        self._neighbor_cache_path = os.path.join(self.config.shapefile_dir, "neighbors.pkl")

    def _get_neighbor_dict(self) -> dict:
        if self._neighbor_dict is not None:
            return self._neighbor_dict

        if os.path.exists(self._neighbor_cache_path):
            try:
                with open(self._neighbor_cache_path, "rb") as f:
                    self._neighbor_dict = pickle.load(f)
                logger.info("Loaded neighbor dictionary from cache")
                return self._neighbor_dict
            except Exception as e:
                logger.warning(f"Failed to load neighbor cache: {e}")

        zones_gdf = load_shapefile_from_zipfile(self.config.taxi_zone_shapefile_url, 
                                            self.config.shapefile_dir)
        zones_gdf_left = zones_gdf.rename(columns={"LocationID": "LocationID_left"})
        zones_gdf_right = zones_gdf.rename(columns={"LocationID": "LocationID_right"})
        neighbors_df = gpd.sjoin(zones_gdf_left, zones_gdf_right, how="left", predicate="touches")
        neighbors_df = neighbors_df[neighbors_df['LocationID_left'] != neighbors_df['LocationID_right']]
        self._neighbor_dict = (neighbors_df.groupby('LocationID_left')['LocationID_right']
                               .apply(lambda s: sorted(list(set(s))))
                               .to_dict())
        try:
            os.makedirs(self.config.shapefile_dir, exist_ok=True)
            with open(self._neighbor_cache_path, "wb") as f:
                pickle.dump(self._neighbor_dict, f)
        except Exception as e:
            logger.warning(f"Failed to persist neighbor cache: {e}")
        return self._neighbor_dict


    def merge_weather_features(self) -> dd.DataFrame:
        try:
            
            # Weather alignment
            weather_df = self.weather_df
            taxi_df = self.taxi_df
            weather_df['bin'] = dd.to_datetime(
                weather_df['day'].astype(str) + ' ' + 
                weather_df['datetime'].astype(str),
                errors='coerce'
            )
            
            #Dropping the Day column
            weather_df = weather_df.drop(columns='day')

            ## making the dtype the same
            weather_df['bin'] = weather_df['bin'].astype('datetime64[us]')

            # Merge target + weather and sort by PUlocationID and bin
            df = taxi_df.merge(weather_df, on='bin', how='left').map_partitions(
                lambda pdf: pdf.sort_values(['PULocationID', 'bin'])
            )

            return (df)

        except Exception as e:
            logger.error("Failed to generate the target feature", exc_info=True)
            raise RideDemandException(e, sys)
        
    def engineer_temporal_feature(self, df: dd.DataFrame) -> dd.DataFrame:
        """
        Deriving the temporal feature
        
        :type df: dd.DataFrame
        :return: Description
        :rtype: DataFrame
        """
        try:
            # Temporal from 'bin'
            df['pickup_year'] = df['bin'].dt.year
            df['pickup_month'] = df['bin'].dt.month
            df['day_of_month'] = df['bin'].dt.day
            df['Pickup_hour'] = df['bin'].dt.hour
            df['day_of_week'] = df['bin'].dt.dayofweek
            df["bin_str"] = df["bin"].astype('str')

            # Vectorized flags
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype('int8')
            df['is_rush_hour'] = df['Pickup_hour'].isin([7, 8, 9, 16, 17, 18, 19]).astype('int8')
            df['is_night_hour'] = df['Pickup_hour'].isin([0,1,2,3,4,5,6,20,21,22,23]).astype('int8')

            # Season mapping
            season_map = {
                12: 'winter', 1: 'winter', 2: 'winter',
                3: 'spring', 4: 'spring', 5: 'spring',
                6: 'summer', 7: 'summer', 8: 'summer',
                9: 'autumn', 10: 'autumn', 11: 'autumn'
            }
            df['season_of_year'] = df['pickup_month'].map(season_map)

            # Fixed holidays and specials
            fixed_holidays = {(1, 1), (7, 4), (11, 11), (6, 19), (12, 25)}
            fixed_specials = {(3, 17), (7, 4), (6, 4), (12, 31),
                            (6, 5), (6, 6), (6, 7), (6, 8), (6, 9), (6, 10),
                            (6, 11), (6, 12), (6, 13), (6, 14), (6, 15)}

            df['is_holiday'] = df[['pickup_month', 'day_of_month']].map_partitions(
                lambda pdf: pdf.apply(lambda r: int((r['pickup_month'], 
                                                     r['day_of_month']) in fixed_holidays), axis=1),
                meta=('is_holiday', 'int32')
            )

            df['Is_special_event'] = df[['pickup_month', 'day_of_month']].map_partitions(
                lambda pdf: pdf.apply(lambda r: int((r['pickup_month'], 
                                                     r['day_of_month']) in fixed_specials), axis=1),
                meta=('Is_special_event', 'int32')
            )

            # Movable holidays and specials (row-wise logic via map_partitions)
            def add_movable_flags(pdf: dd.DataFrame) -> dd.DataFrame:
                def _movable_holiday(row):
                    y, m, d = row['pickup_year'], row['pickup_month'], row['day_of_month']
                    date = pd.Timestamp(y, m, d)
                    if m == 11 and date.weekday() == 3 and date + pd.Timedelta(days=7) > pd.Timestamp(y, 11, 30):
                        return 1
                    if m == 5 and date.weekday() == 0 and date + pd.Timedelta(days=7) > pd.Timestamp(y, 5, 31):
                        return 1
                    if m == 1 and date.weekday() == 0 and 15 <= d <= 21:
                        return 1
                    if m == 9 and date.weekday() == 0 and 1 <= d <= 7:
                        return 1
                    if m == 11 and date.weekday() == 1 and 1 <= d <= 7:
                        return 1
                    if m == 10 and date.weekday() == 0 and 8 <= d <= 14:
                        return 1
                    return 0

                def _movable_special(row):
                    y, m, d = row['pickup_year'], row['pickup_month'], row['day_of_month']
                    date = pd.Timestamp(y, m, d)
                    if m == 11 and date.weekday() == 3 and date + pd.Timedelta(days=7) > pd.Timestamp(y, 11, 30):
                        return 1
                    if m == 6 and date.weekday() == 6 and date + pd.Timedelta(days=7) > pd.Timestamp(y, 6, 30):
                        return 1
                    return 0

                pdf['is_holiday'] = pdf['is_holiday'].where(pdf['is_holiday'] == 1,
                                                        pdf.apply(_movable_holiday, axis=1))
                pdf.loc[pdf['Is_special_event'] == 0, 'Is_special_event'] = pdf.loc[pdf['Is_special_event'] == 0].apply(_movable_special, axis=1)
                
                return pdf

            df = df.map_partitions(add_movable_flags) #meta=df._meta)

            ## creating a column for Payday Indicator
            def is_payday(data):

                date = pd.Timestamp(year=data['pickup_year'], 
                                    month=data['pickup_month'], 
                                    day=data['day_of_month'])

                if date.is_month_end:
                    return 1
                
                if date.day ==(15 or 16 or 17) and date.isoweekday!=(6 or 7):
                    return 1

                return 0

            #pdf = df.compute()

            pdf= df[['PULocationID', 'bin', 'pickup_year', 
                     'pickup_month', 'day_of_month']].compute()

            pdf['is_payday'] = pdf.apply(is_payday, axis=1) ##deriving the payday indicator

            df = df.merge(dd.from_pandas(pdf, npartitions=4), 
                          on=['PULocationID', "bin"], how='left')
            df= df.rename(columns={'pickup_year_x':'pickup_year', 
                                   'pickup_month_x':'pickup_month', 
                                   'day_of_month_x':'day_of_month'})
            
            df = df.drop(['pickup_year_y', 'pickup_month_y', 
                          'day_of_month_y'], axis=1)

            return df

        except Exception as e:
            logger.error("Failed to engineer temporal features", exc_info=True)
            raise RideDemandException(e, sys)


    def citywide_hourly_demand(self, df: dd.DataFrame) -> dd.DataFrame:
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


    def generate_neighbor_features(self, df: dd.DataFrame) -> dd.DataFrame:
        try:
            neighbor_dict = self._get_neighbor_dict()

            # Build neighbor pairs in Pandas, then convert to Dask
            neighbor_pdf = pd.DataFrame(
                [(zone, n) for zone, neighs in neighbor_dict.items() for n in neighs],
                columns=['PULocationID', 'neighbor_id']
            ).fillna(-1)

            neighbor_pdf['PULocationID'] = neighbor_pdf['PULocationID'].astype(
                df['PULocationID'].dtype
                )
            neighbor_pdf['neighbor_id'] = neighbor_pdf['neighbor_id'].astype(
                df['PULocationID'].dtype
                )

            neighbor_ddf = dd.from_pandas(neighbor_pdf, npartitions=1)

            # Prepare neighbor pickups
            # Create a map for the new column names in df_neighbors
            rename_map = {
                'PULocationID': 'neighbor_id', # This is needed for the merge key
                'target_yellow': 'yellow_neighbor_pickups',
                'target_green': 'green_neighbor_pickups',
                'target_hvfhv': 'hvfhv_neighbor_pickups'
            }
            
            # Select original columns from df for df_neighbors
            df_neighbors_orig_cols = ['PULocationID', 'bin', 'target_yellow', 
                                      'target_green', 'target_hvfhv']
            # Apply select and rename in sequence to get df_neighbors
            df_neighbors = df[df_neighbors_orig_cols].rename(columns=rename_map)

            merged = neighbor_ddf.merge(df_neighbors, on='neighbor_id', how='left')
            # List of the *actual* column names in 'merged' that represent neighbor pickups
            neighbor_pickup_cols_in_merged = [
                'yellow_neighbor_pickups',
                'green_neighbor_pickups',
                'hvfhv_neighbor_pickups'
            ]

            # Corresponding desired final column names in the main df
            final_output_col_names = [
                'neighbor_pickups_target_yellow',
                'neighbor_pickups_target_green',
                'neighbor_pickups_target_hvfhv'
            ]

            for i, merged_col_name in enumerate(neighbor_pickup_cols_in_merged):
                output_col_name = final_output_col_names[i]

                neighbor_demand_df = (
                    merged.groupby(['PULocationID', 'bin'])[merged_col_name]
                    .sum()
                    .reset_index()
                    .rename(columns={merged_col_name: output_col_name})
                )

                df = df.merge(neighbor_demand_df, on=['PULocationID', 'bin'], 
                              how='left', suffixes=("", '_y'))
                df[output_col_name] = df[output_col_name].fillna(0)

            logger.info(df.columns)

            return df
        except Exception as e:
            logger.error("Unable to generate neighbor features", e)
            raise RideDemandException(e,sys)
        
    def engineer_autoregressive_signals(self, df: dd.DataFrame) -> dd.DataFrame:
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

            # Select all necessary columns for computation
            cols_to_compute = ['PULocationID', 'bin'] + services + city + neighbors
            pdf = df[cols_to_compute].compute()
            
            pdf = pdf.sort_values(['PULocationID', 'bin'])

            def make_lags(group):
                for s in services:
                    #Zone-level Lags (1h and 24h)
                    for l in [1, 24]:
                        group[f'{s}_lag_{l}h'] = group[s].shift(l)

                for c in city:
                    #city-level Lags (1h and 24h)
                    for l in [1, 24]:
                        group[f'{c}_lag_{l}h'] = group[c].shift(l)
                
                for n in neighbors:
                    #Neighbor Lags (1h and 24h)
                    for l in [1, 24]:
                        group[f'{n}_lag_{l}h'] = group[n].shift(l)
                
                return group

            # Apply lags per Zone
            pdf = pdf.groupby('PULocationID', group_keys=False).apply(make_lags)

            pdf.fillna(0, inplace=True)

            # Merge back to Dask
            new_ddf = dd.from_pandas(pdf, npartitions=df.npartitions)
            df = df.merge(new_ddf, on=['PULocationID', 'bin'], how='left', suffixes=('', '_y'))
            
            # Dropping redundant columns from the merge
            cols_to_drop = [c for c in df.columns if c.endswith('_y')]
            df = df.drop(cols_to_drop, axis=1)

            return df

        except Exception as e:
            logger.error("Failed to generate multi-output autoregressive features")
            raise RideDemandException(e,sys)
            
        
    def save_data_to_feature_store(self, df):
        try:
            df = self.generate_neighbor_features(df)
            transformed_data_store = self.config.transformed_data_file_path

            logger.info("Saving the transformed dataset to the feature store")
            os.makedirs(os.path.dirname(transformed_data_store), exist_ok=True)
            df.to_parquet(transformed_data_store)

            logger.info(f"Transformed data saved to path: {transformed_data_store}")
            print(f"Size of transformed data: {df.shape}")

        except Exception as e:
            logger.error("Unable to save the file", exc_info=True)
            raise RideDemandException(e, sys)

    @retry(
        stop=stop_after_attempt(5), 
        wait=wait_exponential(multiplier=2, min=5, max=60),
        retry = retry_if_exception_type(RideDemandException),
        before_sleep=lambda retry_state: logger.warning(f"Retrying Hopsworks push... Attempt {retry_state.attempt_number}"),
        reraise=True
    )
    def push_transformed_data_to_feature_store(self, data)-> None:
        try:
            api = os.getenv('HOPSWORKS_API_KEY')
            
            ##initializing and login to hopswork feature store
            project = hopsworks.login(project='RideDemandPrediction', api_key_value=api)
            fs = project.get_feature_store()

            ##converting dask dataframe to pandas dataframe
            data = data.compute()

            # 1. Define the desired data types
            #type_mapping = {
             #   'pickups': 'int64',
              #  'city_pickups': 'int64',
               # 'neighbor_pickups_sum': 'int64',
                #'is_holiday': 'int32',
           #     'Is_special_event': 'int32'
           # }

            # 2. Apply the casting safely
         #   for col, dtype in type_mapping.items():
          #      if col in data.columns:
           #         data[col] = data[col].astype(dtype)

            ## creating a new feature group
            fg = fs.get_or_create_feature_group(
                name = 'nycdemandprediction',
                version = 1,
                primary_key = ['PULocationID', 'bin_str'],
                event_time = 'bin',
                description = 'NYC yellow taxi pickup demands per hour per zone',
                online_enabled = False,
                partition_key = ['pickup_year','pickup_month']
            )

            ## inserting new data in the feature group created above
            fg.insert(data, storage = 'offline', write_options = {'wait_for_job': True, 'use_spark':True})

            logger.info('data successfully added to hopsworks feature group')

        except Exception as e:
            raise  RideDemandException(e,sys)
         
    def initiate_feature_engineering(self):
        try:
            df = self.merge_weather_features()

            ## Temporal feature
            df = self.engineer_temporal_feature(df)
            
            ## citywide hourly demand
            df = self.citywide_hourly_demand(df)

            ## generate neighbor features
            df = self.generate_neighbor_features(df)

             ## Autoregressive feature
            df = self.engineer_autoregressive_signals(df)

            logger.info(len(df.columns))
            logger.info(df.columns)

            ## saving the data
            #self.save_data_to_feature_store(df)

            ## pushing data to feature store
            self.push_transformed_data_to_feature_store(df)
        
        except Exception as e:
            logger.error(f"Unable to complete feature engineering process", e)
            raise RideDemandException(e,sys)
