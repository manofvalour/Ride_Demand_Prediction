from src.DynamicPricingEngine.utils.common_utils import create_dir, read_yaml
from src.DynamicPricingEngine.logger.logger import logger
from src.DynamicPricingEngine.exception.customexception import RideDemandException
from src.DynamicPricingEngine.entity.config_entity import (DataIngestionConfig,
                                                           DataTransformationConfig,
                                                           ModelTrainerConfig,
                                                           DataValidationConfig)
from src.DynamicPricingEngine.constants import *

from pathlib import Path
import os, sys

## Defining the configuration manager

class ConfigurationManager:
    def __init__(self, config_path:Path = CONFIG_FILE_PATH,
                 params_path:Path = PARAMS_FILE_PATH):
        try:
            self.config = read_yaml(config_path)
            self.params = read_yaml(params_path)

            artifact_root = self.config.artifacts_root

            ## creating the "artifacts" root directory
            create_dir([artifact_root])
            logger.info(f"Artifacts root directory successfully created: {artifact_root}")            

        except Exception as e:
            logger.error(f"failed to create the artifacts root directory")
            raise RideDemandException(e,sys)
        

    def get_data_ingestion_config(self)-> DataIngestionConfig:
        config = self.config.data_ingestion

        try:
            ##creating the data ingestion root directory
            create_dir([config.root_dir, config.taxi_data_dir, config.weather_data_dir])

            data_ingestion_config = DataIngestionConfig(
                root_dir=config.root_dir,
                taxi_data_url=config.taxi_data_url,
                weather_data_url=config.weather_data_url,
                taxi_data_local_file_path=config.taxi_data_local_file_path,
                weather_data_local_file_path=config.weather_data_local_file_path,
                taxi_data_dir=config.taxi_data_dir,
                weather_data_dir= config.weather_data_dir
            )

            return data_ingestion_config

        except Exception as e:
            logger.error("Cannot get the data ingestion config", e)
            raise RideDemandException(e,sys)
        
    def get_data_transformation_config(self)-> DataTransformationConfig:
        config = self.config.data_transformation

        try:
            ## creating the data transformation root directory
            create_dir([config.root_dir])

            data_transformation_config = DataTransformationConfig(
                root_dir = config.root_dir,
                shapefile_dir = config.shapfile_dir,
                feature_store_url_path= config.feature_store_url_path,
                taxi_zone_shapefile_url= config.taxi_zone_shapefile_url,
                transformed_data_file_path= config.transformed_data_file_path
            )

            return data_transformation_config

        except Exception as e:
            logger.error("Cannot load the data ingestion config", e)
            raise RideDemandException(e,sys)

    def get_model_trainer_config(self)-> ModelTrainerConfig:
        try:
            config = self.config.model_training
            params = self.params.ml_params

            ## creating the model training root directory
            create_dir([config.root_dir])
            logger.info(f"Model Training root directory created successfully: {config.root_dir}")

            model_training_config = ModelTrainerConfig(
                root_dir = config.root_dir,
                train_data_path = config.train_data_path,
                val_data_path = config.val_data_path,
                test_data_path = config.test_data_path,
                trained_model_path=config.trained_model_path,
                split_ratio = params.split_ratio
            )

            return model_training_config
        except Exception as e:
            logger.error("Failed to load the Model Trainer Configuration", e)
            raise RideDemandException(e,sys)