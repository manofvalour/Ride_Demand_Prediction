import sys

from src.DynamicPricingEngine.exception.customexception import RideDemandException
from src.DynamicPricingEngine.logger.logger import logger
from src.DynamicPricingEngine.config.configuration import ConfigurationManager
from src.DynamicPricingEngine.component.ingest_training_data import ExtractTrainingData

class FeaturePipeline:
    def __init__(self):
        pass    

    def initiate_training_data(self):
        try:
            logger.info('Extracting the Training Data...')
            config = ConfigurationManager()
            data_ingestion_config = config.get_data_ingestion_config()
            logger.info('Data Ingestion Configuration successfully loaded')

            training_data = ExtractTrainingData(data_ingestion_config, '2025-07-01', '2025-07-31')
            training_data.extract_nyc_yellow_taxi_data()
            training_data.extract_nyc_weather_data()
            
        except Exception as e:
            logger.error(f"Failed to initiate training data, {e}")
            raise RideDemandException(e,sys)