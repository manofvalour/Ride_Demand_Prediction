"""Data ingestion helpers for pulling NYC taxi and weather datasets.

This module provides the `DataIngestion` class which downloads taxi trip
files from the NYC TLC site, fetches weather data from VisualCrossing,
and aggregates trip-level data into hourly demand targets.
"""

import sys
import os
import pandas as pd
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import requests
import re
from dotenv import load_dotenv
from dateutil.relativedelta import relativedelta
import numpy as np

from src.DynamicPricingEngine.logger.logger import logger
from src.DynamicPricingEngine.exception.customexception import RideDemandException
from src.DynamicPricingEngine.entity.config_entity import DataIngestionConfig
from src.DynamicPricingEngine.utils.data_ingestion_utils import time_subtract, dtype_downcast

load_dotenv()

class DataIngestion:
    """Class responsible for downloading and preparing raw input data.

    Args:
        config (DataIngestionConfig): Configuration with URLs and file paths.
    """
    def __init__(self, config: DataIngestionConfig):
        try:
            ## two months into the past
            now = (datetime.today()-relativedelta(months=11))
            end_date = now - timedelta(days=now.day) ## retrieving the last day of the previous month

            ## accessing the previous month
            days_to_subtract = time_subtract(end_date.strftime('%Y-%m-%d'))
            end_date = (end_date - timedelta(days=days_to_subtract))

            ## start of the month
            days= time_subtract(end_date.strftime('%Y-%m-%d'))
            start_date= end_date - timedelta(days=days-1)

            self.config = config
           # start_date = start_date - relativedelta(months=1)  #1
            #end_date = end_date - relativedelta(months=1) #2

            self.start_date = start_date.strftime('%Y-%m-%d')
            self.end_date = end_date.strftime('%Y-%m-%d')

            self.api_key = os.getenv('API_KEY')

        except Exception as e:
            logger.error(f"unable to calculate the end_date and start_date, {e}")
            raise RideDemandException(e,sys)

    ## things to do: (ingesting the data (weather and nyc_yellow_taxi_data)) -> nyc_tlc_url, 

    def get_NYC_ride_data(self, taxi_type: str) -> pd.DataFrame:
      """Download and normalize trip data for a given taxi type.

      The function scrapes the NYC TLC trip data page, locates the
      parquet file for the configured month, reads only required
      columns, normalizes column names across taxi types and performs
      basic filtering and binning.

      Args:
          taxi_type (str): One of 'yellow', 'green', or 'hvfhv'.

      Returns:
          pd.DataFrame: A dataframe with standardized columns including
          `pickup_datetime`, `dropoff_datetime`, `PULocationID`,
          `DOLocationID`, `trip_miles`, `trip_duration_hr`, `MPH`,
          `service_type`, and `bin`.
      """

      taxi_data_url = self.config.taxi_data_url
      taxi_data_date = datetime.strptime(self.start_date, "%Y-%m-%d")
      taxi_end_date = datetime.strptime(self.end_date, "%Y-%m-%d")
      taxi_data_end_date = taxi_end_date + timedelta(days=1)
      try:
        
        headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124 Safari/537.36'
                        }

        # Send GET request
        response = requests.get(taxi_data_url, headers = headers,timeout = 30) ## retrieve the content of the url
        soup = BeautifulSoup(response.content, "html.parser")

        # Regex pattern to match Yellow, Green, and FHvHV Trip files with date
        if taxi_type == 'yellow':
          pattern = re.compile(r"(yellow_tripdata_)(\d{4}-\d{2})\.parquet", re.IGNORECASE)
        elif taxi_type == 'green':
          pattern = re.compile(r"(green_tripdata_)(\d{4}-\d{2})\.parquet", re.IGNORECASE)
        elif taxi_type == 'hvfhv':
          pattern = re.compile(r"(fhvhv_tripdata_)(\d{4}-\d{2})\.parquet", re.IGNORECASE)

        else:
          raise ValueError("Invalid type. Must be 'yellow', 'green', or 'hvfhv'.")

        # Loop through all links
        data= None
        for link in soup.find_all("a", href=True):
            href = link["href"]
            match = pattern.search(href)
            if match:
                date_str = match.group(2)
                file_date = datetime.strptime(date_str, "%Y-%m")
                if file_date == taxi_data_date:
                    full_url = href if href.startswith("http") else f"https://www.nyc.gov{href}"
                    logger.info(f"Downloading data {date_str} from {full_url}")

                    if taxi_type == 'yellow':
                      cols = ["tpep_pickup_datetime","tpep_dropoff_datetime",
                              "PULocationID", "DOLocationID","trip_distance"]
                    elif taxi_type == 'green':
                      cols = ["lpep_pickup_datetime","lpep_dropoff_datetime", ## Corrected column name here
                              "PULocationID", "DOLocationID","trip_distance"]
                    elif taxi_type == 'hvfhv':
                      cols = ["pickup_datetime", "dropoff_datetime", "PULocationID",
                              "DOLocationID", "trip_miles", "trip_time"]

                    data = pd.read_parquet(full_url, columns=cols)

                    if taxi_type == 'yellow':
                      data = data.rename(columns={
                          'trip_distance': 'trip_miles', 
                          'tpep_pickup_datetime': 'pickup_datetime',
                          'tpep_dropoff_datetime': 'dropoff_datetime'
                      })
                    elif taxi_type == 'green':
                        data = data.rename(columns={
                            'trip_distance': 'trip_miles', 
                            'lpep_pickup_datetime': 'pickup_datetime',
                            'lpep_dropoff_datetime': 'dropoff_datetime'
                        })

                    # Date Filtering
                    data['pickup_datetime'] = pd.to_datetime(data['pickup_datetime'])
                    date_mask = (data['pickup_datetime'] >= self.start_date) & (data['pickup_datetime'] < taxi_data_end_date)
                    data = data.loc[date_mask].copy() 

                    # 3. Compute Duration (Standardizing the different logic)
                    if taxi_type == 'hvfhv' and 'trip_time' in data.columns:
                        # HVFHV provides 'trip_time' in seconds directly
                        data['trip_duration_hr'] = data['trip_time'] / 3600
                        data = data.drop(columns=['trip_time'])
                        data['service_type'] = 'hvfhv'
                    else:
                        # Yellow/Green require calculation from timestamps
                        data['dropoff_datetime'] = pd.to_datetime(data['dropoff_datetime'])
                        data['trip_duration_hr'] = (data['dropoff_datetime'] - data['pickup_datetime']).dt.total_seconds() / 3600
                        if taxi_type == 'green':
                          data['service_type'] = 'green'
                        elif taxi_type == 'yellow':
                          data['service_type'] = 'yellow'

                    #Filter and Calculate Speed
                    data = data[
                        (data['trip_duration_hr'] > 0) & 
                        (data['trip_miles'] >= 0)
                    ].copy()

                    data['MPH'] = data['trip_miles'] / data['trip_duration_hr']
                    
                    # Standardize filtering for all types
                    data = data[data['MPH'].between(1, 60)].copy()

                    # 5. Create Time Bins
                    data['bin'] = data['pickup_datetime'].dt.floor('60min')
                    data = dtype_downcast(data)

                    logger.info(f"data for {date_str} successfully downloaded")

        if data is None:
          logger.info(f"No data found for the {date_str}.")
        else:
          return data

      except Exception as e:
        logger.error(f"Error retrieving the data for {date_str}")
        raise RideDemandException(e,sys)
      
    def derive_targets(self, yellow_df, green_df, hvfhv_df):
        """Aggregate trip-level rows into hourly zone-level demand targets.

        Args:
            yellow_df (pd.DataFrame): Yellow taxi trips for the timeframe.
            green_df (pd.DataFrame): Green taxi trips for the timeframe.
            hvfhv_df (pd.DataFrame): HVFHV trips for the timeframe.

        Returns:
            pd.DataFrame: A dataframe indexed by `bin` and `PULocationID`
            containing `target_yellow`, `target_green`, `target_hvfhv`,
            and congestion/speed features.
        """
        try:
            ## concatinating the dataset
            df = pd.concat([yellow_df, green_df, hvfhv_df], axis=0)
            df['service_type'] = df['service_type'].astype('category')

            logger.info('dataframe concatenated sucessfully!')
            
            # Group by Hour (bin), Zone (pulocationid), and Service Type
            counts = df.groupby(['bin', 'PULocationID', 'service_type']).size().reset_index(name='trip_count')

            #Pivot the service_type so each service gets its own column
            target_df = counts.pivot_table(
                index=['bin', 'PULocationID'], 
                columns='service_type', 
                values='trip_count', 
                fill_value=0
            ).reset_index()

            #Ensure all three target columns exist (even if one service had zero trips total)
            for service in ['yellow', 'green', 'hvfhv']:
                if service not in target_df.columns:
                    target_df[service] = 0
                    
            #Rename columns clearly for the model
            target_df = target_df.rename(columns={
                'yellow': 'target_yellow',
                'green': 'target_green',
                'hvfhv': 'target_hvfhv'
            })

            # Create a skeleton of all possible combinations
            all_bins = df['bin'].unique()
            all_zones = df['PULocationID'].unique()
            grid = pd.MultiIndex.from_product([all_bins, all_zones], names=['bin', 'PULocationID']).to_frame(index=False)

            # Merge your targets onto this grid
            data = grid.merge(target_df, on=['bin', 'PULocationID'], how='left').fillna(0)

            # Compute citywide average speed per hour
            city_speed = (
                df.groupby('bin')['MPH']
                .mean()
                .reset_index()
                .rename(columns={'MPH': 'city_avg_speed'})
            )

            #Compute Congestion Index
            city_speed['city_congestion_index'] = np.where(
                city_speed['city_avg_speed'] > 0, 
                1.0 / city_speed['city_avg_speed'], 
                np.nan
            )

            # Merge back into the main dataframe
            data = data.merge(city_speed, on='bin', how='left')
            logger.info('city_wide congestion index and speed successfully derived')

            #compute Zone-Level Speed
            zone_speed = (
                df.groupby(['PULocationID', 'bin'])['MPH']
                .mean()
                .reset_index()
                .rename(columns={'MPH': 'zone_avg_speed'})
            )

            #Create the Full Grid (Ensure no missing hours/zones)
            zones = df['PULocationID'].unique()
            time_index = pd.date_range(df['bin'].min(), df['bin'].max(), freq='60min')
            grid = pd.MultiIndex.from_product(
                [zones, time_index], 
                names=['PULocationID', 'bin']
            ).to_frame(index=False)

            #Merge Zone Data onto Grid, then Merge City Fallback
            combined_speed = grid.merge(zone_speed, on=['PULocationID', 'bin'], how='left')
            combined_speed = combined_speed.merge(city_speed, on='bin', how='left')
            combined_speed['zone_avg_speed'] = combined_speed['zone_avg_speed'].fillna(combined_speed['city_avg_speed'])
            
            #Handle cases where both might be NaN
            global_mean = combined_speed['city_avg_speed'].mean()
            combined_speed['zone_avg_speed'] = combined_speed['zone_avg_speed'].replace(0, np.nan).fillna(global_mean)

            #Compute Congestion Index (1/Speed)
            combined_speed['zone_congestion_index'] = np.where(
                combined_speed['zone_avg_speed'] > 0, 
                1.0 / combined_speed['zone_avg_speed'], 
                0
            )

            #Merge back into main df
            data = data.merge(
                combined_speed[['PULocationID', 'bin', 'zone_avg_speed', 
                                'zone_congestion_index',]], 
                on=['PULocationID', 'bin'], 
                how='left'
            )

            logger.info('city_wide congestion index and speed feature derived successfully')

            ## round data to 3 decimal
            data['city_avg_speed'] = data['city_avg_speed'].round(2)
            data['city_congestion_index'] = data['city_congestion_index'].round(3)
            data['zone_avg_speed'] = data['zone_avg_speed'].round(2)
            data['zone_congestion_index'] = data['zone_congestion_index'].round(3)

            logger.info('Target features has been derived')

            return data
        
        except Exception as e:
           logger.error('Unable to create the target features')
           raise RideDemandException(e,sys)

    def extract_nyc_weather_data(self) -> pd.DataFrame:
        """Fetch hourly weather data from the configured weather API.

        The method requests the VisualCrossing timeline API for the
        date range defined on the ingestion object and flattens the
        returned JSON into an hourly dataframe.

        Returns:
            pd.DataFrame: Hourly weather records with fields like
            `datetime`, `temp`, `humidity`, `precip`, `windspeed`,
            `feelslike`, and `visibility`.
        """

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
                response = requests.get(url, params=params, timeout=30) # Always set a timeout
                response.raise_for_status()

            except requests.exceptions.HTTPError:
                logger.error(f"CRITICAL: HTTP error. Status: {response.status_code} | Response: {response.text[:200]}")
                # Raising the error stops the pipeline immediately
                raise RideDemandException(f"API failed with status {response.status_code}", sys)

            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as network_err:
                logger.error(f"CRITICAL: Network/Timeout error: {network_err}")
                raise RideDemandException("Weather API is unreachable. Check internet or API status.", sys)

            except requests.RequestException as e:
                logger.error(f"CRITICAL: Unexpected API error: {e}")
                raise RideDemandException(e, sys)

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

            df_hours = (pd.DataFrame(hourly_records))#.pipe(dtype_downcast)
            logger.info(f"Retrieved {len(days)} days ({df_hours.shape[0]} hourly records).")

            return df_hours

        except Exception as e:
            logger.error(f"failed to extract weather data {e}")
            raise RideDemandException(e,sys)
        
    
    def save_data_to_artifact(self, nyc_taxi_data, nyc_weather_data) -> None:
        """Persist downloaded taxi and weather datasets to artifact paths.

        Args:
            nyc_taxi_data (pd.DataFrame): Trip-level taxi dataframe.
            nyc_weather_data (pd.DataFrame): Hourly weather dataframe.
        """
        try:
            taxi_file_path = self.config.taxi_data_local_file_path
            weather_file_path = self.config.weather_data_local_file_path

            ## saving the datasets
            logger.info(f'Saving the NYC_taxi datasets to {taxi_file_path}')
            nyc_taxi_data.to_parquet(taxi_file_path, index=False)

            logger.info(f"saving the NYC_weather dataset to {weather_file_path}")
            nyc_weather_data.to_csv(weather_file_path, index= False)

            logger.info(f'Successfully saved the datasets to artifacts paths: {taxi_file_path},{weather_file_path}')
       
        except Exception as e:
            logger.error(f"Unable to save the datasets to artifact: {e}")
            raise RideDemandException(e,sys)
        
    def initiate_data_ingestion(self):
        """Run the full ingestion flow.

        This convenience method downloads taxi and weather data, derives
        hourly targets and returns a tuple of `(taxi_df, weather_df)`
        ready for downstream transformation.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: `(taxi_df, weather_df)`
        """
        try:
            yellow_df = self.get_NYC_ride_data('yellow')
            green_df = self.get_NYC_ride_data('green')
            hvfhv_df = self.get_NYC_ride_data('hvfhv')

            logger.info(f'Taxi_data downloaded. yellow_taxi_data_size: {yellow_df.shape}')
            logger.info(f'Taxi_data downloaded. Green_taxi_data_size: {green_df.shape}')
            logger.info(f'Taxi_data downloaded. HVFHV_data_size: {hvfhv_df.shape}')

            taxi_df = self.derive_targets(yellow_df, green_df, hvfhv_df)
            logger.info("Target feature derived successfully!")

            weather_df = self.extract_nyc_weather_data()
            logger.info(f'Weather_data downloaded. Weather_data_size: {weather_df.shape}')

            return taxi_df, weather_df

        except Exception as e:
            raise RideDemandException(e,sys)
