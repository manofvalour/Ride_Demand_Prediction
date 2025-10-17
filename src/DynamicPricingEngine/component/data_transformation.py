import os, sys, pickle
import pandas as pd
import numpy as np
from datetime import datetime
import geopandas as gpd
import time
from memory_profiler import profile
import dask.dataframe as dd

from src.DynamicPricingEngine.logger.logger import logger
from src.DynamicPricingEngine.exception.customexception import RideDemandException
from src.DynamicPricingEngine.entity.config_entity import DataTransformationConfig
from src.DynamicPricingEngine.utils.common_utils import create_dir, load_shapefile_from_zip


class DataTransformation:
    def __init__(self, config: DataTransformationConfig, 
                 nyc_taxi_data: str, 
                 nyc_weather_data: str):
        
        self.config = config
        
        #Read both datasets with Dask
        self.taxi_df = dd.read_parquet(nyc_taxi_data)
        self.weather_df = dd.read_csv(nyc_weather_data)

        #Ensure datetime types
        for col in ['tpep_pickup_datetime', 'tpep_dropoff_datetime']:
            if col in self.taxi_df.columns:
                self.taxi_df[col] = dd.to_datetime(self.taxi_df[col], errors='coerce')

        # Precompute bin
        if 'tpep_pickup_datetime' in self.taxi_df.columns:
            self.taxi_df['bin'] = self.taxi_df['tpep_pickup_datetime'].dt.floor('60min')

        # Cache neighbor dictionary
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

        zones_gdf = load_shapefile_from_zip(self.config.taxi_zone_shapefile_url, self.config.shapefile_dir)
        zones_gdf_left = zones_gdf.rename(columns={"LocationID": "LocationID_left"})
        zones_gdf_right = zones_gdf.rename(columns={"LocationID": "LocationID_right"})
        neighbors_df = gpd.sjoin(zones_gdf_left, zones_gdf_right, how="left", predicate="touches")
        neighbors_df = neighbors_df[neighbors_df['LocationID_left'] != neighbors_df['LocationID_right']]
        self._neighbor_dict = (neighbors_df.groupby('LocationID_left')['LocationID_right']
                               .apply(lambda s: list(sorted(set(s))))
                               .to_dict())
        try:
            os.makedirs(self.config.shapefile_dir, exist_ok=True)
            with open(self._neighbor_cache_path, "wb") as f:
                pickle.dump(self._neighbor_dict, f)
        except Exception as e:
            logger.warning(f"Failed to persist neighbor cache: {e}")
        return self._neighbor_dict


    @profile
    def derive_target_and_join_to_weather_feature(self) -> dd.DataFrame:
        try:
            taxi_df = self.taxi_df[['PULocationID', 'bin']]

            # Aggregate pickups per zone-hour
            y = (taxi_df
                 .groupby(['PULocationID', 'bin'])
                 .size().rename('pickups')
                 .reset_index())
            
            y['bin']= y['bin'].astype('datetime64[ns]')
            y['PULocationID']= y['PULocationID'].astype('int32')

            # Build full grid (requires materialization in Pandas, then back to Dask)
            zones = y['PULocationID'].unique().compute()

            time_index = pd.date_range(y['bin'].min().compute(), y['bin'].max().compute(), freq='60min')
            grid = pd.MultiIndex.from_product([zones, time_index], names=['PULocationID', 'bin']).to_frame(index=False)

            y['PULocationID'] = y['PULocationID'].astype('int32')
            grid['PULocationID'] = grid['PULocationID'].astype('int32')

            # Align datetime precision
            y['bin'] = y['bin'].astype('datetime64[ns]')
            grid['bin'] = grid['bin'].astype('datetime64[ns]')

            y = dd.from_pandas(grid, npartitions=4).merge(y, how='left', on=['PULocationID', 'bin'])
            y = y.fillna({'pickups': 0})

            # Weather alignment
            weather_df = self.weather_df
            weather_df['bin'] = dd.to_datetime(
                weather_df['day'].astype(str) + ' ' + weather_df['datetime'].astype(str),
                errors='coerce'
            )

            weather_df = weather_df.drop(columns='day')

            # Merge target + weather
            df = y.merge(weather_df, on='bin', how='left').map_partitions(
                lambda pdf: pdf.sort_values(['PULocationID', 'bin'])
            )

            return (df)

        except Exception as e:
            logger.error("Failed to generate the target feature", exc_info=True)
            raise RideDemandException(e, sys)
        
    @profile
    def engineer_temporal_feature(self, df: dd.DataFrame) -> dd.DataFrame:
        try:
            # Temporal from 'bin'
            df['pickup_year'] = df['bin'].dt.year
            df['pickup_month'] = df['bin'].dt.month
            df['day_of_month'] = df['bin'].dt.day
            df['Pickup_hour'] = df['bin'].dt.hour
            df['day_of_week'] = df['bin'].dt.dayofweek

            # Vectorized flags
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype('int8')
            df['is_rush_hour'] = df['Pickup_hour'].isin([7, 8, 9, 16, 17, 18, 19]).astype('int8')
            df['is_night_hour'] = df['Pickup_hour'].isin([0,1,2,3,4,5,6,20,21,22,23]).astype('int8')
            #df['is_payday']= df['day_of_month'].isin([1,15]).astype('int8')

            # Season mapping (works with map in Dask)
            season_map = {
                12: 'winter', 1: 'winter', 2: 'winter',
                3: 'spring', 4: 'spring', 5: 'spring',
                6: 'summer', 7: 'summer', 8: 'summer',
                9: 'autumn', 10: 'autumn', 11: 'autumn'
            }
            df['season_of_year'] = df['pickup_month'].map(season_map)

            # Fixed holidays and specials (vectorized)
            fixed_holidays = {(1, 1), (7, 4), (11, 11), (6, 19), (12, 25)}
            fixed_specials = {(3, 17), (7, 4), (6, 4), (12, 31),
                            (6, 5), (6, 6), (6, 7), (6, 8), (6, 9), (6, 10),
                            (6, 11), (6, 12), (6, 13), (6, 14), (6, 15)}

            df['is_holiday'] = df[['pickup_month', 'day_of_month']].map_partitions(
                lambda pdf: pdf.apply(lambda r: int((r['pickup_month'], r['day_of_month']) in fixed_holidays), axis=1),
                meta=('is_holiday', 'int8')
            )

            df['Is_special_event'] = df[['pickup_month', 'day_of_month']].map_partitions(
                lambda pdf: pdf.apply(lambda r: int((r['pickup_month'], r['day_of_month']) in fixed_specials), axis=1),
                meta=('Is_special_event', 'int8')
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

                                
                def _is_pay_day(row):
                    y, m, d = row['pickup_year'], row['pickup_month'], row['day_of_month']
                    date = pd.Timestamp(y,m,d)
                                       
                    if date.is_month_end:
                        return 1
                    return 0

                pdf['is_holiday'] = pdf['is_holiday'].where(pdf['is_holiday'] == 1,
                                                        pdf.apply(_movable_holiday, axis=1))
                pdf.loc[pdf['Is_special_event'] == 0, 'Is_special_event'] = pdf.loc[pdf['Is_special_event'] == 0].apply(_movable_special, axis=1)
                
                #pdf['is_payday'] = pdf['is_pay_day'].where(pdf['is_payday']==0, 
                 #                                                   pdf.apply(_is_pay_day, axis=1))
                return pdf

            df = df.map_partitions(add_movable_flags, meta=df._meta)

            return df

        except Exception as e:
            logger.error("Failed to engineer temporal features", exc_info=True)
            raise RideDemandException(e, sys)

    @profile
    def engineer_autoregressive_signals(self, df: dd.DataFrame) -> dd.DataFrame:
        try:
            ## Define a Pandas function to apply per-partition
            df = df.sort_values(['PULocationID', 'bin'])
            g = df.groupby('PULocationID')['pickups']
            df['pickups_lag_1h'] = g.apply(lambda x: x.shift(1), meta=('pickups_lag_1h', 'f8'))
            df['pickups_lag_24h'] = g.apply(lambda x: x.shift(24), meta=('pickups_lag_24h', 'f8'))

            df['pickups_roll_mean_24h']=g.apply(lambda x: x.rolling('1D').mean(), meta=('pickups_roll_mean_24h', 'f8'))
            df['pickups_roll_std_24h']=g.apply(lambda x: x.rolling('1D').mean(), meta=('pickups_roll_std_24h', 'f8'))            

            lag_cols = [
                'pickups_lag_1h', 'pickups_lag_24h',
                'pickups_roll_mean_24h', 'pickups_roll_std_24h'
            ]
            existing_cols = [c for c in lag_cols if c in df.columns]
            df[existing_cols] = df[existing_cols].fillna(0)

            return df

        except Exception as e:
            logger.error("Failed to generate autoregressive features", exc_info=True)
            raise RideDemandException(e, sys)

    @profile
    def city_wide_congestion_features(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            # Select needed columns
            taxi_df = self.taxi_df[['bin', 'tpep_pickup_datetime', 'tpep_dropoff_datetime', 'trip_distance']]

            # Compute trip duration in hours
            taxi_df['trip_duration_hr'] = (
                (taxi_df['tpep_dropoff_datetime'] - taxi_df['tpep_pickup_datetime']).dt.total_seconds() / 3600
            )

            # Filter invalid trips
            taxi_df = taxi_df[(taxi_df['trip_duration_hr'] > 0) & (taxi_df['trip_distance'] >= 0)]

            # Compute speed
            taxi_df['MPH'] = taxi_df['trip_distance'] / taxi_df['trip_duration_hr']
            taxi_df = taxi_df[taxi_df['MPH'].between(1, 60)]

            # Compute citywide average speed per hour
            city_speed = (
                taxi_df.groupby('bin')['MPH']
                .mean()
                .reset_index()
                .rename(columns={'MPH': 'city_avg_speed'})
            )

            # Congestion index
            city_speed['city_congestion_index'] = city_speed['city_avg_speed'].map_partitions(
                lambda pdf: np.where(pdf > 0, 1.0 / pdf, np.nan),
                meta=('city_congestion_index', 'f8')
            )

            # Merge back into main df
            df = df.merge(city_speed, on='bin', how='left')
            return df

        except Exception as e:
            logger.error("Unable to generate city-wide features", exc_info=True)
            raise RideDemandException(e, sys)
        
    @profile
    def zone_level_congestion_features(self, df: dd.DataFrame) -> dd.DataFrame:
        try:
            # Select needed columns
            taxi_df = self.taxi_df[['PULocationID', 'bin', 'tpep_pickup_datetime',
                                    'tpep_dropoff_datetime', 'trip_distance']]

            # Compute trip duration in hours
            taxi_df['trip_duration_hr'] = (
                (taxi_df['tpep_dropoff_datetime'] - taxi_df['tpep_pickup_datetime']).dt.total_seconds() / 3600
            )

            # Filter invalid trips
            taxi_df = taxi_df[(taxi_df['trip_duration_hr'] > 0) & (taxi_df['trip_distance'] >= 0)]

            # Compute speed
            taxi_df['MPH'] = taxi_df['trip_distance'] / taxi_df['trip_duration_hr']
            taxi_df = taxi_df[taxi_df['MPH'].between(1, 60)]

            # Compute zone-level average speed per hour
            zone_speed = (
                taxi_df.groupby(['PULocationID', 'bin'])['MPH']
                .mean()
                .reset_index()
                .rename(columns={'MPH': 'zone_avg_speed'})
            )

            # Congestion index
            zone_speed['zone_congestion_index'] = zone_speed['zone_avg_speed'].map_partitions(
                lambda pdf: np.where(pdf > 0, 1.0 / pdf, np.nan),
                meta=('zone_congestion_index', 'f8')
            )

            # Build full grid (zones × time) in Pandas, then convert back to Dask
            zones = df['PULocationID'].unique().compute()
            time_index = pd.date_range(df['bin'].min().compute(), df['bin'].max().compute(), freq='60min')
            grid = pd.MultiIndex.from_product([zones, time_index], names=['PULocationID', 'bin']).to_frame(index=False)
            grid_dd = dd.from_pandas(grid, npartitions=4)

            zone_speed['PULocationID']= zone_speed['PULocationID'].astype('int32')

            # Merge grid with zone_speed to ensure full coverage
            zone_speed = grid_dd.merge(zone_speed, how='left', on=['PULocationID', 'bin'])
            zone_speed[['zone_avg_speed', 'zone_congestion_index']] = zone_speed[
                ['zone_avg_speed', 'zone_congestion_index']
            ].fillna(0)

            #zone_speed['PULocationID']= zone_speed['PULocationID'].astype('int32')
            # Merge back into main df
            df = df.merge(zone_speed, on=['PULocationID', 'bin'], how='left')
            return df

        except Exception as e:
            logger.error("Unable to generate zone-level features", exc_info=True)
            raise RideDemandException(e, sys)
        
    @profile
    def citywide_hourly_demand(self, df: dd.DataFrame) -> dd.DataFrame:
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

            # Define a Pandas function to add lags
            df = df.sort_values('bin')
            df['city_pickups_lag_1h'] = df['city_pickups'].apply(lambda x: x.shift(1), meta=('city_pickups_lag_1h', 'f8'))
            #df['city_pickups'].shift(1)
            df['city_pickups_lag_24h'] = df['city_pickups'].apply(lambda x: x.shift(1), meta=('city_pickups_lag_24h', 'f8'))
            #df['city_pickups'].shift(24)
            #df = df.map_partitions(add_city_lags, meta=df._meta)

            lag_cols = [
                'city_pickups_lag_1h', 'city_pickups_lag_24h'
                ]
            
            existing_cols = [c for c in lag_cols if c in df.columns]
            df[existing_cols] = df[existing_cols].fillna(0)

            return df

        except Exception as e:
            logger.error("Unable to engineer citywide hourly demand features", exc_info=True)
            raise RideDemandException(e, sys)
        
    @profile
    def generate_neighbor_features(self, df: dd.DataFrame) -> dd.DataFrame:
        try:
            start = time.time()
            neighbor_dict = self._get_neighbor_dict()
            print(f" get_neighbor_dict function took {time.time()- start: .2f} seconds")

            # Build neighbor pairs in Pandas, then convert to Dask
            neighbor_pdf = pd.DataFrame(
                [(zone, n) for zone, neighs in neighbor_dict.items() for n in neighs],
                columns=['PULocationID', 'neighbor_id']
            ).fillna(-1)

            neighbor_pdf['PULocationID'] = neighbor_pdf['PULocationID'].astype(df['PULocationID'].dtype)
            neighbor_pdf['neighbor_id'] = neighbor_pdf['neighbor_id'].astype(df['PULocationID'].dtype)

            neighbor_ddf = dd.from_pandas(neighbor_pdf, npartitions=1)

            # Prepare neighbor pickups
            df_neighbors = df[['PULocationID', 'bin', 'pickups']].rename(
                columns={'PULocationID': 'neighbor_id', 'pickups': 'neighbor_pickups'}
            )

            merged = neighbor_ddf.merge(df_neighbors, on='neighbor_id', how='left')

            neighbor_demand_df = (
                merged.groupby(['PULocationID', 'bin'])['neighbor_pickups']
                .sum()
                .reset_index()
                .rename(columns={'neighbor_pickups': 'neighbor_pickups_sum'})
            )

            neighbor_demand_df['neighbor_pickups_sum'] = neighbor_demand_df['neighbor_pickups_sum'].fillna(-1)

            df = df.merge(neighbor_demand_df, on=['PULocationID', 'bin'], how='left')

            # Add lag features per partition
            #def add_neighbor_lags(pdf: pd.DataFrame) -> pd.DataFrame:
            df = df.sort_values(['PULocationID', 'bin'])
            g = df.groupby('PULocationID')['neighbor_pickups_sum']
            df['neighbor_pickups_lag_1h'] = g.apply(lambda x: x.shift(1), meta=('neighbor_pickups_lag_1h', 'f8'))
            df['neighbor_pickups_lag_24h'] = g.apply(lambda x: x.shift(24), meta=('neighbor_pickups_lag_24h', 'f8'))
            #    return pdf

            #df = df.map_partitions(add_neighbor_lags, meta=df._meta)

            # Fill NaNs in lag/rolling features
            lag_cols = [
                'neighbor_pickups_lag_1h', 'neighbor_pickups_lag_24h'
            ]
            existing_cols = [c for c in lag_cols if c in df.columns]
            df[existing_cols] = df[existing_cols].fillna(0)

            return df

        except Exception as e:
            logger.error("Unable to generate neighbor features", exc_info=True)
            raise RideDemandException(e, sys)
        
    @profile
    def save_data_to_feature_store(self, df):
        try:
            #df = self.generate_neighbor_features()
            transformed_data_store = self.config.transformed_data_file_path
            logger.info("Saving the transformed dataset to the feature store")
            os.makedirs(os.path.dirname(transformed_data_store), exist_ok=True)
            df.to_parquet(transformed_data_store, index=False)
            logger.info(f"Transformed data saved to path: {transformed_data_store}")
            print(f"Size of transformed data: {df.shape}")

        except Exception as e:
            logger.error("Unable to save the file", exc_info=True)
            raise RideDemandException(e, sys)
        
    def initiate_feature_engineering(self):
        try:
    
            start= time.time()
            df = self.derive_target_and_join_to_weather_feature()
            print(f" derive_target_and_join_to_weather function took {time.time()- start: .2f} seconds")

            ## Temporal feature
            start=time.time()
            df = self.engineer_temporal_feature(df)
            print(f" Engineer_temporal_feature function took {time.time()- start: .2f} seconds")

            ## Autoregressive feature
            start = time.time()
            df = self.engineer_autoregressive_signals(df)
            print(f" Autoregressive function took {time.time()- start: .2f} seconds")

            ## Citywide congestion features
            start = time.time()
            df = self.city_wide_congestion_features(df)
            print(f"city_wide_congestion function took {time.time()- start: .2f} seconds")

            ## zone level congestion features
            start = time.time()
            df = self.zone_level_congestion_features(df)
            print(f"Zone level congestion function took {time.time()- start: .2f} seconds")

            ## citywide hourly demand
            start = time.time()
            df = self.citywide_hourly_demand(df)
            print(f"citywide hourly demand function took {time.time()- start: .2f} seconds")

            ## generate neighbor features
            start = time.time()
            df = self.generate_neighbor_features(df)
            print(f" generate Neighbor feature function took {time.time()- start: .2f} seconds")

            ## saving the data
            start = time.time()
            self.save_data_to_feature_store(df)
            print(f" save data to feature store function took {time.time()- start: .2f} seconds")
        
        except Exception as e:
            raise RideDemandException(e,sys)