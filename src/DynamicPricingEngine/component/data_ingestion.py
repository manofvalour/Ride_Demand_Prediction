import sys, os
from pathlib import Path
from typing import List
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import requests
import re
import io
import time
from dotenv import load_dotenv

from src.DynamicPricingEngine.logger.logger import logger
from src.DynamicPricingEngine.exception.customexception import RideDemandException
from src.DynamicPricingEngine.utils.common_utils import create_dir, read_yaml
from src.DynamicPricingEngine.entity.config_entity import DataIngestionConfig
from src.DynamicPricingEngine.utils.data_ingestion_utils import time_subtract

load_dotenv()

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        try:
            
            ## two months into the past
            now = datetime.now().strftime("%Y-%m-%d")
            end_date = datetime.strptime(now, "%Y-%m-%d") ## converting to datetime
            end_date = end_date - timedelta(days=end_date.day) ## retrieving the last day of the previos month

            ## accessing the previous month
            days_to_subtract = time_subtract(end_date.strftime('%Y-%m-%d'))
            end_date = (end_date- timedelta(days=days_to_subtract))

            ## start of the month
            days= time_subtract(end_date.strftime('%Y-%m-%d'))
            start_date= end_date - timedelta(days=days-1)

            self.config = config
            self.start_date = start_date.strftime('%Y-%m-%d')
            self.end_date = end_date.strftime('%Y-%m-%d')
            self.api_key = os.getenv('API_KEY')

        except Exception as e:
            logger.info(f"unable to calculate the end_date and start_date, {e}")
            raise RideDemandException(e,sys)

    ## things to do: (ingesting the data (weather and nyc_yellow_taxi_data)) -> nyc_tlc_url, 

    def get_NYC_yellow_taxi_data(self)->pd.DataFrame:
      
      taxi_data_url = self.config.taxi_data_url
      taxi_data_date = datetime.strptime(self.start_date, "%Y-%m-%d")
      taxi_data_end_date = datetime.strptime(self.end_date, "%Y-%m-%d")
      try:

        # Send GET request
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
                    #cols = ["tpep_pickup_datetime","tpep_dropoff_datetime","PULocationID","DOLocationID","passenger_count",
                           # "trip_distance","fare_amount", "tip_amount", "total_amount"] columns=cols

                    data = pd.read_parquet(full_url)
                    data =data[data['tpep_pickup_datetime']>= taxi_data_date]
                    data =data[data['tpep_pickup_datetime']<= taxi_data_end_date]

                    logger.info(f"data for {date_str} successfully downloaded")


        if data is None:
          logger.info(f"No data found for the {date_str}.")
        else:
          return data

      except Exception as e:
        logger.error(f"Error retrieving the data for {date_str}")
        raise RideDemandException(e,sys)
      

    def extract_nyc_weather_data(self)->pd.DataFrame:

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
            logger.info(f"Retrieved {len(days)} days ({df_hours.shape[0]} hourly records).")

            return df_hours

        except Exception as e:
            logger.error(f"failed to extract weather data {e}")
            raise RideDemandException(e,sys)
    
    def save_data_to_artifact(self, nyc_taxi_data, nyc_weather_data)->None:
        try:
            taxi_file_path = self.config.taxi_data_local_file_path
            weather_file_path = self.config.weather_data_local_file_path

            ## saving the datasets
            logger.info(f'Saving the NYC_taxi datasets to {taxi_file_path}')
            nyc_taxi_data.to_parquet(taxi_file_path, index=False)

            logger.info(f"saving the NYC_weather dataset to {weather_file_path}")
            nyc_weather_data.to_csv(weather_file_path, index= False)

            logger.info('Successfully saved the datasets to artifacts paths: {taxi_file_path},{weather_file_path}')
       
        except Exception as e:
            raise RideDemandException(e,sys)
    