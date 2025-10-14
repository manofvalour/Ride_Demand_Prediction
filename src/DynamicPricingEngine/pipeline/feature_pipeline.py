import os, sys

from src.DynamicPricingEngine.component.data_ingestion import DataIngestion
from src.DynamicPricingEngine.exception.customexception import RideDemandException
from src.DynamicPricingEngine.logger.logger import logger
from src.DynamicPricingEngine.config.configuration import ConfigurationManager

class FeaturePipeline:
    def __init__(self):
        pass

    def initiate_data_ingestion(self):
        """ Function to Initiate the data ingestion class """
        try:

            logger.info('initiating data ingestion')
            config = ConfigurationManager()
            data_ingestion_config = config.get_data_ingestion_config()
            logger.info('Data Ingestion configuration successfully loaded')

            data_ingestion = DataIngestion(config = data_ingestion_config)
            nyc_taxi_data = data_ingestion.get_NYC_yellow_taxi_data()
            nyc_weather_data = data_ingestion.extract_nyc_weather_data()
            data_ingestion.save_data_to_artifact(nyc_taxi_data, nyc_weather_data)
            
            logger.info('Data Ingestion pipeline initiated successfully')

        except Exception as e:
            logger.error(f"Failed to initiate data ingestion, {e}")
            raise RideDemandException(e,sys)