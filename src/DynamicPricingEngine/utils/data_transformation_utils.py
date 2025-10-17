import os, sys
from pathlib import Path
import pandas as pd

from src.DynamicPricingEngine.logger.logger import logger
from src.DynamicPricingEngine.exception.customexception import RideDemandException
from ensure import ensure_annotations

@ensure_annotations
def is_holiday(df):
    # Weekends
    if df['day_of_week'] in [5, 6]:  # Saturday=5, Sunday=6
        return 1

    # Fixed-date holidays
    fixed_holidays = [(1, 1), (7, 4), (11, 11), (6, 19), (12, 25)]
    if (df['pickup_month'], df['day_of_month']) in fixed_holidays:
        return 1

    # Movable holidays
    date = pd.Timestamp(year=df['pickup_year'], month=df['pickup_month'], day=df['day_of_month'])

    # Last Thursday in November
    if df['pickup_month'] == 11 and date.weekday() == 3 and date + pd.Timedelta(
        days=7) > pd.Timestamp(year=df['pickup_year'], month=11, day=30):
        return 1

    # Last Monday in May
    if df['pickup_month'] == 5 and date.weekday() == 0 and date + pd.Timedelta(days=7) > pd.Timestamp(
        year=df['pickup_year'], month=5, day=31):
        return 1

    # Third Monday in January
    if df['pickup_month'] == 1 and date.weekday() == 0 and 15 <= df['day_of_month'] <= 21:
        return 1

    # First Monday in September
    if df['pickup_month'] == 9 and date.weekday() == 0 and 1 <= df['day_of_month'] <= 7:
        return 1

    # First Tuesday in November
    if df['pickup_month'] == 11 and date.weekday() == 1 and 1 <= df['day_of_month'] <= 7:
        return 1

    # Second Monday in October
    if df['pickup_month'] == 10 and date.weekday() == 0 and 8 <= df['day_of_month'] <= 14:
        return 1

    return 0




import os,sys
import pandas as pd
import numpy as np
from datetime import datetime
import geopandas as gdp

from src.DynamicPricingEngine.logger.logger import logger
from src.DynamicPricingEngine.exception.customexception import RideDemandException
from src.DynamicPricingEngine.entity.config_entity import DataTransformationConfig
from src.DynamicPricingEngine.utils.common_utils import create_dir, load_shapefile_from_zip

class DataTransformation:
    def __init__(self, config:DataTransformationConfig, 
                 nyc_taxi_data:str,
                  nyc_weather_data:str):
        
        self.config = config
        taxi_data = pd.read_parquet(nyc_taxi_data)
        weather_data = pd.read_csv(nyc_weather_data)

        self.taxi_df = taxi_data
        self.weather_df = weather_data

    def derive_target_and_join_to_weather_feature(self)->pd.DataFrame:
        try:

            taxi_df = self.taxi_df
            weather_df = self.weather_df

            ## deriving the target feature
            taxi_df['bin'] = taxi_df['tpep_pickup_datetime'].dt.floor('60min')  ## bining into hour

            y = (taxi_df
                .groupby(['PULocationID','bin'])
                .size()
                .rename('pickups')
                .reset_index())

            # Build full grid to include zeros
            zones = y['PULocationID'].unique()
            time_index = pd.date_range(y['bin'].min(), y['bin'].max(), freq='60min')
            grid = pd.MultiIndex.from_product([zones, time_index], names=['PULocationID','bin']).to_frame(index=False)

            y = grid.merge(y, how='left', on=['PULocationID','bin']).fillna({'pickups':0})
            
            logger.info("target feature derived successfully")

            ## Adding the weather feature created to the already created pickups columns
            weather_df['bin'] = pd.to_datetime(weather_df['day'].astype(str) + ' ' + weather_df['datetime'].astype(str))

            #weather_hourly = (weather_df
             #               .groupby('bin')
              #              .agg({'temp':'mean','dew':'mean','snow':'sum',
               #                     'snowdepth':'sum',
                #                    'precip':'sum','windspeed':'mean',
                 #                   'visibility':'mean','humidity':'mean'})
                  #          .reset_index())

            df = y.merge(weather_df, on='bin', how='left').ffill()
            df.set_index('bin', inplace=True)
            logger.info("The data joined successfully")

            return df
        
        except Exception as e:
            logger.error("Failed to generate the target feature", e)
            raise RideDemandException(e,sys)
    
    def engineer_temporal_feature(self, df:pd.DataFrame)->pd.DataFrame:
        
        try:
            ##seperating tpep_pickup_datetime
            df['pickup_year']= df.index.year
            df['pickup_month']= df.index.month
            df['day_of_month']= df.index.day
            df['Pickup_hour']= df.index.hour
            df['day_of_week']= df.index.dayofweek

            ## creating the is_weekend, is_rush_hour, is_night_hour, is_holiday, season of the year, and special event data
            df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x in [5, 6] else 0)
            df['is_rush_hour'] = df['Pickup_hour'].apply(lambda x: 1 if x in [7, 8, 9, 16, 17,18,19] else 0)
            df['is_night_hour'] = df['Pickup_hour'].apply(lambda x: 1 if x in [0,1,2,3,4,5,6,20,21,22,23] else 0)

            ## creating the season of the year
            df['season_of_year'] = df['pickup_month'].apply(lambda x: 'winter' if x in [12, 1, 2]
                                                            else 'spring' if x in [3,4,5] else
                                                            'summer' if x in [6,7,8] else 'autumn')


            ## Deriving the Holdiay feature
            def is_holiday(data:pd.DataFrame):
                # Weekends
                if data['day_of_week'] in [5, 6]:  # Saturday=5, Sunday=6
                    return 1

                # Fixed-date holidays
                fixed_holidays = [(1, 1), (7, 4), (11, 11), (6, 19), (12, 25)]
                if (data['pickup_month'], data['day_of_month']) in fixed_holidays:
                    return 1

                # Movable holidays
                date = pd.Timestamp(year=data['pickup_year'], month=data['pickup_month'], day=data['day_of_month'])

                # Last Thursday in November
                if data['pickup_month'] == 11 and date.weekday() == 3 and date + pd.Timedelta(
                    days=7) > pd.Timestamp(year=data['pickup_year'], month=11, day=30):
                    return 1

                # Last Monday in May
                if data['pickup_month'] == 5 and date.weekday() == 0 and date + pd.Timedelta(days=7) > pd.Timestamp(
                    year=data['pickup_year'], month=5, day=31):
                    return 1

                # Third Monday in January
                if data['pickup_month'] == 1 and date.weekday() == 0 and 15 <= data['day_of_month'] <= 21:
                    return 1

                # First Monday in September
                if data['pickup_month'] == 9 and date.weekday() == 0 and 1 <= data['day_of_month'] <= 7:
                    return 1

                # First Tuesday in November
                if data['pickup_month'] == 11 and date.weekday() == 1 and 1 <= data['day_of_month'] <= 7:
                    return 1

                # Second Monday in October
                if data['pickup_month'] == 10 and date.weekday() == 0 and 8 <= data['day_of_month'] <= 14:
                    return 1

                return 0
            
            df['is_holiday'] = df.apply(is_holiday, axis=1)

            ## creating a column for special event
            def is_special_event(data):
                # Fixed-date holidays
                fixed_holidays = [(3, 17), (7, 4), (6, 4), (12,31), (6, 5), (6, 6), (6,7),(6,8),
                                (6,9),(6,10), (6,11), (6,12), (6,13),(6,14),(6,15)]
                if (data['pickup_month'], data['day_of_month']) in fixed_holidays:
                    return 1

                # Movable holidays
                date = pd.Timestamp(year=data['pickup_year'], month=data['pickup_month'], day=data['day_of_month'])

                # Macy's Thanksgiving Parade
                if data['pickup_month'] == 11 and date.weekday() == 3 and date + pd.Timedelta(
                    days=7) > pd.Timestamp(year=data['pickup_year'], month=11, day=30):
                    return 1

                # Last Sunday in June (Pride Month)
                if data['pickup_month'] == 6 and date.weekday() == 6 and date + pd.Timedelta(days=7) > pd.Timestamp(
                    year=data['pickup_year'], month=6, day=30):
                    return 1

                return 0

            df['Is_special_event'] = df.apply(is_special_event, axis=1) ## creating the feature

            ## creating a column for Payday Indicator
            def is_payday(data):

                date = pd.Timestamp(year=data['pickup_year'], month=data['pickup_month'], day=data['day_of_month'])

                if date.is_month_start or data['day_of_month']==15 or date.is_month_end:
                    return 1

                return 0

            df['is_payday'] = df.apply(is_payday, axis=1) ##deriving the payday indicator

            return df
        
        except Exception as e:
            raise RideDemandException(e,sys)
        

    def engineer_autoregressive_signals(self, df:pd.DataFrame)->pd.DataFrame:

        try:
            def make_lags(group, col='pickups'):

                ## for lag features
                for l in [1,24]:
                    group[f'{col}_lag_{l}'] = group[col].shift(l)

                ## for rolling mean and std
                for w in [24]:
                    group[f'{col}_roll_mean_{w}'] = group[col].shift(1).rolling(w).mean()
                    group[f'{col}_roll_std_{w}'] = group[col].shift(1).rolling(w).std()
                return group
            
            df.reset_index()
            df = df.sort_values(['PULocationID','bin'])
            df = df.groupby('PULocationID', group_keys=False).apply(make_lags) ##generating the autoregressive feature

            return df

        except Exception as e:
            logger.error('Failed to generate the features', e)
            raise RideDemandException(e,sys)
        
    def city_wide_congestion_features(self, df:pd.DataFrame)-> pd.DataFrame:
        try:
            taxi_df = self.taxi_df

            ## computing the trip distance
            taxi_df['trip_duration_hr'] = (
                (taxi_df['tpep_dropoff_datetime'] - taxi_df['tpep_pickup_datetime'])
                .dt.total_seconds() / 3600
            )

            ## computing the MPH
            taxi_df['MPH']= taxi_df['trip_distance']/taxi_df['trip_duration_hr']

            ## filtering out invalid mph
            ## based on research, traffic in new york city rarely goes above 60mph
            taxi_df = taxi_df[
                taxi_df['MPH'].between(1, 60)  # min 1 mph, max 60 mph
            ]

            ## creating the city-wide congestion
            city_speed = (taxi_df
                .groupby('bin')['MPH']
                .mean()
                .rename('city_avg_speed')
                .reset_index()
            )

            ## computing the congestion index by inverting the city_avg_speed
            city_speed['city_congestion_index'] = 1 / city_speed['city_avg_speed']

            # Merge City speed into the main table
            df = df.merge(city_speed, on=['bin'], how='left')

            return df

        except Exception as e:
            logger.error("Unable to generate the city wide features",e)
            raise RideDemandException(e,sys)
        

    def zone_level_congestion_features(self, df:pd.DataFrame)->pd.DataFrame:
        try:
            taxi_df = self.taxi_df

            ## computing the zone level congestion
            zone_speed = (taxi_df
                .groupby(['PULocationID', 'bin'])['MPH']
                .mean()
                .rename('zone_avg_speed')
                .reset_index()
            )

            # Inverse speed as congestion index
            zone_speed['zone_congestion_index'] = 1 / zone_speed['zone_avg_speed']

            # Build full grid to include zeros
            zones = df['PULocationID'].unique()
            time_index = pd.date_range(df['bin'].min(), df['bin'].max(), freq='60min')
            grid = pd.MultiIndex.from_product([zones, time_index], names=['PULocationID','bin']).to_frame(index=False)

            zone_speed = grid.merge(zone_speed, how='left', on=['PULocationID','bin']).fillna({'zone_avg_speed':0, 'zone_congestion_index':0})

            # Merge the Zone_speed into the main table
            df = df.merge(zone_speed, on=['PULocationID','bin'], how='left')

            return df

        except Exception as e:
            logger.error('Unable to generate the zone level features', e)
            raise RideDemandException(e,sys)
        
    def citywide_hourly_demand(self, df:pd.DataFrame)->pd.DataFrame:
        try:
            # Citywide hourly demand
            city_demand = (df.groupby('bin')['pickups']
                            .sum()
                            .rename('city_pickups')
                            .reset_index())

            # Merge into zone-hour table
            df = df.merge(city_demand, on='bin', how='left')

            # Create lag features (1h, 24h)
            for lag in [1, 24]:
                df[f'city_pickups_lag_{lag}h'] = df['city_pickups'].shift(lag)

            return df

        except Exception as e:
            logger.error("Unable to engineer the citywide hourly demand features", e)
            raise RideDemandException(e,sys)
        
    def generate_neighbor_features(self, df:pd.DataFrame)->pd.DataFrame:
        try:
            zones_gdf = load_shapefile_from_zip(self.config.taxi_zone_shapefile_url,
                                               self.config.shapefile_dir)
            logger.info(f'loaded the shapefile successfully to {self.config.shapefile_dir}')

            # Spatial join: each zone with all zones it touches
            neighbors_df = gdp.sjoin(zones_gdf, zones_gdf, how="left", predicate="touches")

            # Remove self-joins
            neighbors_df = neighbors_df[neighbors_df['LocationID_left'] != neighbors_df['LocationID_right']]

            # Group into a dictionary: zone -> list of neighbor zones
            neighbor_dict = (neighbors_df.groupby('LocationID_left')['LocationID_right']
                            .apply(list)
                            .to_dict())
            
            # df: your zone-hour pickup table
            # Columns: ['PULocationID', 'bin', 'pickups']
            # neighbor_dict: {zone_id: [neighbor_zone_ids, ...]}

            # Step 1: Create a DataFrame mapping each zone to its neighbors
            neighbor_pairs = []
            for zone, neighs in neighbor_dict.items():
                for n in neighs:
                    neighbor_pairs.append((zone, n))

            neighbor_df = pd.DataFrame(neighbor_pairs, columns=['PULocationID', 'neighbor_id'])

            # Step 2: Join neighbor_df with df to get neighbor pickups
            # Rename df columns for clarity before merge
            df_neighbors = df.rename(columns={'PULocationID': 'neighbor_id', 'pickups': 'neighbor_pickups'})
            df_neighbors.reset_index(inplace=True)

            # Merge: for each (zone, neighbor), bring in neighbor's pickups for each hour
            merged = neighbor_df.merge(df_neighbors, on='neighbor_id', how='left')


            # Step 3: Group by zone and hour to sum neighbor pickups
            neighbor_demand_df = (merged
                .groupby(['PULocationID', 'bin'])['neighbor_pickups']
                .sum()
                .reset_index()
                .rename(columns={'neighbor_pickups': 'neighbor_pickups_sum'})
            )

            # Step 4: Merge back into your main df
            df = df.reset_index()
            df = df.merge(neighbor_demand_df, on=['PULocationID', 'bin'], how='left')
            df['neighbor_pickups_sum'] = df['neighbor_pickups_sum'].fillna(0)

            ## computing the Lagged neighbor demand
            for lag in [1, 24]:
                df[f'neighbor_pickups_lag_{lag}h'] = df.groupby('PULocationID')['neighbor_pickups_sum'].shift(lag)

            # Function to sum neighbor demand
            #def neighbor_demand(row):
             #   neighbors = neighbor_dict
              #  neigh_ids = neighbors.get(row['PULocationID'], [])
               # mask = (df['PULocationID'].isin(neigh_ids)) & (df['bin'] == row['bin'])
                #return df.loc[mask, 'pickups'].sum()

            ## handling the missing data caused by lag, rolling mean/std
            df = df.fillna(method='bfill')

        except Exception as e:
            logger.error("Unable to generate the Neighbor features", e)
            raise RideDemandException(e,sys)
        
    def save_data_to_feature_store(self):
        try:
            df = self.generate_neighbor_features()
            transformed_data_store = self.config.transformed_data_file_path
            logger.info(f"Saving the transformed dataset to the feature store")

            df.to_csv(transformed_data_store, index=False)
            logger.info(f'Transformed data saved to path, {transformed_data_store}')

            print(f"size of Transformed data: {df.shape}")

        except Exception as e:
            logger.error("Unable to save the file",e)
            raise RideDemandException(e,sys)
    
    #def feature_engineering(self):


