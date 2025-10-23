import os, sys

from src.DynamicPricingEngine.exception.customexception import RideDemandException
from src.DynamicPricingEngine.logger.logger import logger
from src.DynamicPricingEngine.config.configuration import ConfigurationManager
from src.DynamicPricingEngine.component.model_training import ModelTrainer

class TrainingPipeline:
    def __init__(self):
        pass    

    def initiate_model_training(self):
        try:
            logger.info('Extracting the Training Data...')
            config = ConfigurationManager()
            model_trainer_config = config.get_model_trainer_config
            logger.info('Data Ingestion Configuration successfully loaded')

            model_trainer = ModelTrainer(model_trainer_config)
            data = model_trainer.retrieve_engineered_feature()

            logger.info('successfully Loaded the data')

        except Exception as e:
            raise RideDemandException(e,sys)

