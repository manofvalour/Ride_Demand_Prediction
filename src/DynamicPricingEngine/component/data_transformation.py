import os, sys, pickle
import pandas as pd
import numpy as np
from datetime import datetime
import geopandas as gpd
import dask.dataframe as dd

from src.DynamicPricingEngine.logger.logger import logger
from src.DynamicPricingEngine.exception.customexception import RideDemandException
from src.DynamicPricingEngine.entity.config_entity import DataTransformationConfig
from src.DynamicPricingEngine.utils.common_utils import create_dir, load_shapefile_from_zip
from src.DynamicPricingEngine.config.configuration import ConfigurationManager


class DataTransformation:
    def __init__(self, config: DataTransformationConfig, 
                 nyc_taxi_data: str, 
                 nyc_weather_data: str):
        
        self.config = config
        
        #Read both datasets with Dask
        self.taxi_df = dd.read_parquet(nyc_taxi_data)
        self.weather_df = dd.read_csv(nyc_weather_data)

        self.taxi_df.index = self.taxi_df.index.astype('int64')
        self.weather_df.index = self.weather_df.index.astype('int64')

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

            #y['PULocationID'] = y['PULocationID'].astype('int32')
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
            y.index = y.index.astype('int64')

            # Merge target + weather
            df = y.merge(weather_df, on='bin', how='left').map_partitions(
                lambda pdf: pdf.sort_values(['PULocationID', 'bin'])
            )

            return (df)

        except Exception as e:
            logger.error("Failed to generate the target feature", exc_info=True)
            raise RideDemandException(e, sys)
        

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
                
                return pdf

            df = df.map_partitions(add_movable_flags) #meta=df._meta)

            ## creating a column for Payday Indicator
            def is_payday(data):

                date = pd.Timestamp(year=data['pickup_year'], month=data['pickup_month'], day=data['day_of_month'])

                if date.is_month_start or data['day_of_month']==15 or date.is_month_end:
                    return 1

                return 0

            #pdf = df.compute()

            pdf= df[['PULocationID', 'bin', 'pickup_year', 'pickup_month', 'day_of_month']].compute()

            pdf['is_payday'] = pdf.apply(is_payday, axis=1) ##deriving the payday indicator

            df = df.merge(dd.from_pandas(pdf, npartitions=4), on=['PULocationID', "bin"], how='left')
            df= df.rename(columns={'pickup_year_x':'pickup_year', 'pickup_month_x':'pickup_month', 
                       'day_of_month_x':'day_of_month'})
            df = df.drop(['pickup_year_y', 'pickup_month_y', 'day_of_month_y'], axis=1)

            return df

        except Exception as e:
            logger.error("Failed to engineer temporal features", exc_info=True)
            raise RideDemandException(e, sys)


    def city_wide_congestion_features(self, df: pd.DataFrame) -> dd.DataFrame:
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

            return df

        except Exception as e:
            logger.error("Unable to engineer citywide hourly demand features", exc_info=True)
            raise RideDemandException(e, sys)


    def generate_neighbor_features(self, df: dd.DataFrame) -> dd.DataFrame:
        try:
            neighbor_dict = self._get_neighbor_dict()

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

            neighbor_demand_df['neighbor_pickups_sum'] = neighbor_demand_df['neighbor_pickups_sum']#.fillna(-1)

            df = df.merge(neighbor_demand_df, on=['PULocationID', 'bin'], how='left')
            df['neighbor_pickups_sum'] = df['neighbor_pickups_sum'].fillna(0)

            return df

        except Exception as e:
            logger.error("Unable to generate neighbor features", exc_info=True)
            raise RideDemandException(e, sys)
        
    
    def engineer_autoregressive_signals(self, df: dd.DataFrame) -> dd.DataFrame:
        try:
            ## Define a Pandas function to apply per-partition
            pdf= df[['PULocationID', 'bin', 'pickups', 'city_pickups', 'neighbor_pickups_sum']].compute()

            def make_lags(group, col='pickups'):

                ## for lag features
                for l in [1,24]:
                    group[f'{col}_lag_{l}h'] = group[col].shift(l)

                ## for rolling mean and std for zonelevel/bin
                for w in [24]:
                    group[f'{col}_roll_mean_{w}h'] = group[col].shift(1).rolling(w).mean()
                    group[f'{col}_roll_std_{w}h'] = group[col].shift(1).rolling(w).std()
                return group
            
            pdf.reset_index()
            pdf = pdf.sort_values(['PULocationID','bin'])
            pdf = pdf.groupby('PULocationID', group_keys=False).apply(make_lags) ##generating the autoregressive feature


            # Create lag features for city pickups(1h, 24h)
            for lag in [1, 24]:
                pdf[f'city_pickups_lag_{lag}h'] = pdf['city_pickups'].shift(lag)

            ## computing the Lagged neighbor demand
            for lag in [1, 24]:
                pdf[f'neighbor_pickups_lag_{lag}h'] = pdf.groupby('PULocationID')['neighbor_pickups_sum'].shift(lag)

            pdf.fillna(0, inplace=True)

            df = df.merge(dd.from_pandas(pdf, npartitions=4), on=['PULocationID', "bin"], how='left')
            df= df.rename(columns={'pickups_x':'pickups', 'city_pickups_x':'city_pickups', 
                       'neighbor_pickups_sum_x':'neighbor_pickups_sum'})
            df = df.drop(['pickups_y', 'city_pickups_y', 'neighbor_pickups_sum_y'], axis=1)

            return df

        except Exception as e:
            logger.error("Failed to generate autoregressive features", exc_info=True)
            raise RideDemandException(e, sys)

        
    def save_data_to_feature_store(self, df):
        try:
            #df = self.generate_neighbor_features()
            transformed_data_store = self.config.transformed_data_file_path
            logger.info("Saving the transformed dataset to the feature store")
            os.makedirs(os.path.dirname(transformed_data_store), exist_ok=True)
            df.to_parquet(transformed_data_store)
            logger.info(f"Transformed data saved to path: {transformed_data_store}")
            print(f"Size of transformed data: {df.shape}")

        except Exception as e:
            logger.error("Unable to save the file", exc_info=True)
            raise RideDemandException(e, sys)
        
    def initiate_feature_engineering(self):
        try:
    
            df = self.derive_target_and_join_to_weather_feature()

            ## Temporal feature
            df = self.engineer_temporal_feature(df)
            
            ## Citywide congestion features
            df = self.city_wide_congestion_features(df)

            ## zone level congestion features
            df = self.zone_level_congestion_features(df)

            ## citywide hourly demand
            df = self.citywide_hourly_demand(df)

            ## generate neighbor features
            df = self.generate_neighbor_features(df)

             ## Autoregressive feature
            df = self.engineer_autoregressive_signals(df)

            ## saving the data
            self.save_data_to_feature_store(df)
        
        except Exception as e:
            logger.error(f"Unable to complete feature engineering process", e)
            raise RideDemandException(e,sys)