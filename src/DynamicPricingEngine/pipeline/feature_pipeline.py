import os, sys

from src.DynamicPricingEngine.component.data_ingestion import DataIngestion
from src.DynamicPricingEngine.component.data_transformation import DataTransformation
from src.DynamicPricingEngine.exception.customexception import RideDemandException
from src.DynamicPricingEngine.logger.logger import logger
from src.DynamicPricingEngine.config.configuration import ConfigurationManager
from src.DynamicPricingEngine.component.ingest_training_data import ExtractTrainingData

class FeaturePipeline:
    def __init__(self):
        pass

    def initiate_data_ingestion_and_transformation(self):
        """ Function to Initiate the data ingestion and transformation class """
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

            logger.info('initiating data transformation')

            data_transformation_config = config.get_data_transformation_config()
            logger.info('Data Transformation configuration loaded successfully')
            
            data_transformation = DataTransformation(data_transformation_config,
                                                     nyc_taxi_data,
                                                     nyc_weather_data)
            data_transformation.initiate_feature_engineering()

            logger.info(f"Data Transformation pipeline completed")

        except Exception as e:
            logger.error(f"Failed to initiate data transformation, {e}")
            raise RideDemandException(e,sys)