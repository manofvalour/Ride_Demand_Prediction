"""Feature engineering and transformation utilities.

This module exposes the `DataTransformation` class which merges weather
and target data, engineers temporal, neighbor and autoregressive
features, and pushes the transformed dataset to a feature store.
"""

import os
import sys
import pickle
import pandas as pd
import geopandas as gpd
import hopsworks
from dotenv import load_dotenv
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential


load_dotenv()

from src.DynamicPricingEngine.logger.logger import logger
from src.DynamicPricingEngine.exception.customexception import RideDemandException
from src.DynamicPricingEngine.entity.config_entity import DataTransformationConfig
from src.DynamicPricingEngine.utils.common_utils import load_shapefile_from_zipfile

class DataTransformation:
    """Handles feature engineering for model training.

    Uses pandas for scalable IO and GeoPandas for spatial neighbor calculations.

    Args:
        config (DataTransformationConfig): Configuration with paths and URLs.
    """
    def __init__(self, config: DataTransformationConfig):

        self.config = config

        self.taxi_df = pd.read_parquet(config.taxi_data_local_file_path)
        self.weather_df = pd.read_csv(config.weather_data_local_file_path)

        for col in ['tpep_pickup_datetime', 'tpep_dropoff_datetime']:
            if col in self.taxi_df.columns:
                self.taxi_df[col] = pd.to_datetime(self.taxi_df[col], errors='coerce')

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

    def merge_weather_features(self) -> pd.DataFrame:
        """Join hourly weather features onto the target taxi dataframe."""
        try:
            weather_df = self.weather_df.copy()
            taxi_df = self.taxi_df.copy()
            weather_df['bin'] = pd.to_datetime(
                weather_df['day'].astype(str) + ' ' +
                weather_df['datetime'].astype(str),
                errors='coerce'
            )
            weather_df = weather_df.drop(columns='day')
            weather_df['bin'] = weather_df['bin'].astype('datetime64[us]')

            df = taxi_df.merge(weather_df, on='bin', how='left').sort_values(['PULocationID', 'bin'])
            return df

        except Exception as e:
            logger.error("Failed to generate the target feature", exc_info=True)
            raise RideDemandException(e, sys)

    def engineer_temporal_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derive temporal features from the bin column."""
        try:
            df = df.copy()
            df['pickup_year'] = df['bin'].dt.year
            df['pickup_month'] = df['bin'].dt.month
            df['day_of_month'] = df['bin'].dt.day
            df['Pickup_hour'] = df['bin'].dt.hour
            df['day_of_week'] = df['bin'].dt.dayofweek
            df["bin_str"] = df["bin"].astype('str')

            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype('int8')
            df['is_rush_hour'] = df['Pickup_hour'].isin([7, 8, 9, 16, 17, 18, 19]).astype('int8')
            df['is_night_hour'] = df['Pickup_hour'].isin([0, 1, 2, 3, 4, 5, 6, 20, 21, 22, 23]).astype('int8')

            season_map = {
                12: 'winter', 1: 'winter', 2: 'winter',
                3: 'spring', 4: 'spring', 5: 'spring',
                6: 'summer', 7: 'summer', 8: 'summer',
                9: 'autumn', 10: 'autumn', 11: 'autumn'
            }
            df['season_of_year'] = df['pickup_month'].map(season_map)

            fixed_holidays = {(1, 1), (7, 4), (11, 11), (6, 19), (12, 25)}
            fixed_specials = {(3, 17), (7, 4), (6, 4), (12, 31),
                            (6, 5), (6, 6), (6, 7), (6, 8), (6, 9), (6, 10),
                            (6, 11), (6, 12), (6, 13), (6, 14), (6, 15)}

            df['is_holiday'] = df.apply(
                lambda r: int((r['pickup_month'], r['day_of_month']) in fixed_holidays), axis=1
            )
            df['Is_special_event'] = df.apply(
                lambda r: int((r['pickup_month'], r['day_of_month']) in fixed_specials), axis=1
            )

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

            df['is_holiday'] = df.apply(
                lambda r: _movable_holiday(r) if r['is_holiday'] == 0 else r['is_holiday'], axis=1
            )
            df['Is_special_event'] = df.apply(
                lambda r: _movable_special(r) if r['Is_special_event'] == 0 else r['Is_special_event'], axis=1
            )

            def is_payday(data):
                date = pd.Timestamp(year=data['pickup_year'],
                                    month=data['pickup_month'],
                                    day=data['day_of_month'])
                if date.is_month_end:
                    return 1
                if date.day in (15, 16, 17) and date.isoweekday() not in (6, 7):
                    return 1
                return 0

            df['is_payday'] = df.apply(is_payday, axis=1)
            return df

        except Exception as e:
            logger.error("Failed to engineer temporal features", exc_info=True)
            raise RideDemandException(e, sys)


    def citywide_hourly_demand(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute citywide hourly pickup aggregates in a single pass."""
        try:
            services = ['target_yellow', 'target_green', 'target_hvfhv']
            city_demand = df.groupby('bin')[services].sum().reset_index()
            rename = {s: f'{s}_city_hour_pickups' for s in services}
            city_demand = city_demand.rename(columns=rename)
            df = df.merge(city_demand, on='bin', how='left')
            return df

        except Exception as e:
            logger.error("Unable to engineer citywide hourly demand features")
            raise RideDemandException(e, sys)


    def generate_neighbor_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Construct neighbor pickup aggregates for each zone and hour."""
        try:
            neighbor_dict = self._get_neighbor_dict()

            neighbor_pdf = pd.DataFrame(
                [(zone, n) for zone, neighs in neighbor_dict.items() for n in neighs],
                columns=['PULocationID', 'neighbor_id']
            ).fillna(-1)

            neighbor_pdf['PULocationID'] = neighbor_pdf['PULocationID'].astype(df['PULocationID'].dtype)
            neighbor_pdf['neighbor_id'] = neighbor_pdf['neighbor_id'].astype(df['PULocationID'].dtype)

            rename_map = {
                'PULocationID': 'neighbor_id',
                'target_yellow': 'yellow_neighbor_pickups',
                'target_green': 'green_neighbor_pickups',
                'target_hvfhv': 'hvfhv_neighbor_pickups'
            }

            df_neighbors_orig_cols = ['PULocationID', 'bin', 'target_yellow',
                                      'target_green', 'target_hvfhv']
            df_neighbors = df[df_neighbors_orig_cols].rename(columns=rename_map)
            merged = neighbor_pdf.merge(df_neighbors, on='neighbor_id', how='left')

            neighbor_pickup_cols_in_merged = [
                'yellow_neighbor_pickups', 'green_neighbor_pickups', 'hvfhv_neighbor_pickups'
            ]
            final_output_col_names = [
                'neighbor_pickups_target_yellow', 'neighbor_pickups_target_green', 'neighbor_pickups_target_hvfhv'
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

            cols_to_drop = [c for c in df.columns if c.endswith('_y')]
            df = df.drop(columns=cols_to_drop)

            return df
        except Exception as e:
            logger.error("Unable to generate neighbor features", e)
            raise RideDemandException(e, sys)

    def engineer_autoregressive_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lagged autoregressive features for targets and aggregates.

        Produces 1-hour and 24-hour lags for zone-level targets, and 1-hour
        lags for city-level and neighbor-level aggregates (24h variants are
        not used by the model).
        """
        try:
            services = ['target_yellow', 'target_green', 'target_hvfhv']
            neighbors = [
                'neighbor_pickups_target_yellow', 'neighbor_pickups_target_green', 'neighbor_pickups_target_hvfhv'
            ]
            city = ['target_yellow_city_hour_pickups', 'target_green_city_hour_pickups',
                    'target_hvfhv_city_hour_pickups']

            df = df.sort_values(['PULocationID', 'bin'])

            for s in services:
                for l in [1, 24]:
                    df[f'{s}_lag_{l}h'] = df.groupby('PULocationID')[s].shift(l)

            for c in city:
                df[f'{c}_lag_1h'] = df.groupby('PULocationID')[c].shift(1)

            for n in neighbors:
                df[f'{n}_lag_1h'] = df.groupby('PULocationID')[n].shift(1)

            df = df.fillna(0)
            return df

        except Exception as e:
            logger.error("Failed to generate multi-output autoregressive features")
            raise RideDemandException(e, sys)


    def save_data_to_feature_store(self, df):
        """Persist transformed dataframe to a local parquet."""
        try:
            df = self.generate_neighbor_features(df)
            transformed_data_store = self.config.transformed_data_file_path
            logger.info("Saving the transformed dataset to the feature store")
            os.makedirs(os.path.dirname(transformed_data_store), exist_ok=True)
            df.to_parquet(transformed_data_store)
            logger.info(f"Transformed data saved to path: {transformed_data_store}")
            logger.info(f"Size of transformed data: {df.shape}")

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
    def push_transformed_data_to_feature_store(self, data) -> None:
        """Push the transformed dataset into Hopsworks Feature Store."""
        try:
            api = os.getenv('HOPSWORKS_API_KEY')

            project = hopsworks.login(project='RideDemandPrediction',
                                      host= 'https://eu-west.cloud.hopsworks.ai',
                                      api_key_value=api)
            fs = project.get_feature_store()

            fg = fs.get_or_create_feature_group(
                name = 'nycdemandprediction',
                version = 1,
                primary_key = ['PULocationID', 'bin_str'],
                event_time = 'bin',
                description = 'NYC yellow taxi pickup demands per hour per zone',
                online_enabled = False,
                partition_key = ['pickup_year','pickup_month']
            )

            fg.insert(data, storage = 'offline', write_options = {'wait_for_job': False, 'use_spark':True})
            logger.info('data successfully added to hopsworks feature group')

        except Exception as e:
            raise  RideDemandException(e, sys)

    def initiate_feature_engineering(self):
        """Run the end-to-end transformation pipeline and push to FS."""
        try:
            df = self.merge_weather_features()
            df = self.engineer_temporal_feature(df)
            df = self.citywide_hourly_demand(df)
            df = self.generate_neighbor_features(df)
            df = self.engineer_autoregressive_signals(df)
            self.push_transformed_data_to_feature_store(df)

        except Exception as e:
            logger.error("Unable to complete feature engineering process", e)
            raise RideDemandException(e, sys)
