"""Utilities to extract and persist historical training data month-by-month.

This module provides `ExtractTrainingData` which downloads monthly taxi
and weather data for a specified date range and saves each month's data
as separate artifacts for use in training.
"""
import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime, timedelta
import time
import pandas as pd
import os
import sys

from dotenv import load_dotenv
load_dotenv()

from src.DynamicPricingEngine.logger.logger import logger
from src.DynamicPricingEngine.exception.customexception import RideDemandException
from src.DynamicPricingEngine.entity.config_entity import DataIngestionConfig

class ExtractTrainingData:
  """Download and save monthly slices of taxi and weather data.

  Args:
    config (DataIngestionConfig): Paths and directories to save files.
    start_date (str): ISO date string for the start (YYYY-MM-DD).
    end_date (str): ISO date string for the end (YYYY-MM-DD).
  """
  def __init__(self, config: DataIngestionConfig,
         start_date: str, end_date: str):
      
      self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
      self.end_date = datetime.strptime(end_date, "%Y-%m-%d")
      self.taxi_path = config.taxi_data_dir
      self.weather_path = config.weather_data_dir

      month_end = self.start_date + timedelta(days=32)

      self.month_end = month_end.replace(day=1) - timedelta(days=1)
      self.taxi_data_url = "https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page"
      self.weather_data_url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
      self.api_key = os.getenv('API_KEY')

  def extract_nyc_yellow_taxi_data(self):
    """Iterate month-by-month and download yellow taxi data files.

      Saves each month's parquet slice under `self.taxi_path` with a
      descriptive filename. Uses the public NYC TLC data page to find
      the parquet URLs.
      """
    taxi_data_url = self.taxi_data_url
    taxi_data_date = self.start_date
    end_date =self.end_date

    try:

      # Send GET request
      while taxi_data_date <= end_date:
        month_end = taxi_data_date + timedelta(days=32)
        taxi_data_month_end = month_end.replace(day=1) - timedelta(days=1)
        print(f"Downloading data from {taxi_data_date} to {taxi_data_month_end}")

        taxi_data_next_month = month_end.replace(day=1)

        response = requests.get(taxi_data_url)
        soup = BeautifulSoup(response.content, "html.parser")

        # Regex pattern to match Yellow Taxi files with date
        pattern = re.compile(r"(yellow_tripdata_)(\d{4}-\d{2})\.parquet", re.IGNORECASE)

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
                    print(f"Downloading data {date_str} from {full_url}")
                    cols = ["tpep_pickup_datetime","tpep_dropoff_datetime",
                            "PULocationID","DOLocationID","trip_distance"]

                    data = pd.read_parquet(full_url, columns=cols)
                    data =data[data['tpep_pickup_datetime']>= taxi_data_date]
                    data =data[data['tpep_pickup_datetime']< taxi_data_next_month]

                    logger.info(f"data for {date_str} successfully downloaded")

        if data is None:
          logger.info(f"No data found for the {date_str}.")
        else:
          ## saving the datasets
          taxi_path_dir = f"{self.taxi_path}/taxi_data_{taxi_data_date}.parquet"
          data.to_parquet(taxi_path_dir, index=False)
          logger.info(f'Saving the NYC_taxi datasets to {taxi_path_dir}')

        taxi_data_date = taxi_data_next_month
      
      print(f"Successfully saved data from {taxi_data_date}_to_{taxi_data_month_end} to {self.weather_path}")

    except Exception as e:
      logger.info(f"Error retrieving the data for {date_str}")
      raise RideDemandException(e,sys)

    def extract_nyc_weather_data(self):
      """Download monthly weather data from VisualCrossing and save CSVs.

      Iterates over months between `self.start_date` and `self.end_date`
      and persists hourly weather records per month.
      """
      try:

        start_date = self.start_date.replace(day=1)
        month_end = self.month_end
        end_date = self.end_date
        api_keys = self.api_key
        location = "New York NY United States"
        base_url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"

        current_date = start_date

        while current_date <= end_date:
          next_month = (current_date.replace(day=28) + timedelta(days=4)).replace(day=1)
          month_end = min(next_month - timedelta(days=1), end_date)
          start_of_month_str = current_date.strftime("%Y-%m-%d")
          end_of_month_str = month_end.strftime("%Y-%m-%d")

          params = {
              "unitGroup": "us",
              "key": api_keys,
              "include": "days,hours,current,alerts,stations",
              "contentType": "json"
          }

          url = f"{base_url}/{location}/{start_of_month_str}/{end_of_month_str}"
          print(f"Fetching data from {start_of_month_str} to {end_of_month_str}...")

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
          print(f"Retrieved {len(days)} days ({df_hours.shape[0]} hourly records).")

          ## saving the file to artifacts
          taxi_path_dir = f"{self.weather_path}/weather_data_{start_of_month_str}_to_{end_of_month_str}.csv"
          df_hours.to_csv(taxi_path_dir, index=False)
          logger.info(f'Saving the NYC_weather datasets to {taxi_path_dir}')


          #return df_hours

          current_date = next_month
          time.sleep(3)
        
        print(f"Successfully saved data from {start_date}_to_{end_date} to {self.weather_path}")

      except Exception as e:
        print(f"failed to extract weather data {e}")
        raise e