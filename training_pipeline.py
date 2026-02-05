"""Entrypoint script to run the model training pipeline.

Loads training configuration, constructs a `ModelTrainer` and runs
the end-to-end training and registration process.
"""

import os, sys
import pandas as pd

from src.DynamicPricingEngine.exception.customexception import RideDemandException
from src.DynamicPricingEngine.logger.logger import logger
from src.DynamicPricingEngine.config.configuration import ConfigurationManager
from src.DynamicPricingEngine.component.model_training import ModelTrainer
import sys
from time import time

if __name__=="__main__":

    STAGE_NAME = 'Model Training Stage'   
    try:
        t0= time()
        logger.info(f"{STAGE_NAME} initiated")
        config = ConfigurationManager()
        inference_config = config.get_inference_config()
        training_pipeline = ModelTrainer(config= inference_config)
        training_pipeline.initiate_model_training()

        t1 = time()
        dt = (t1 - t0)
        logger.info(f"{STAGE_NAME} completed in {dt:.2f}secs")

    except Exception as e:
        raise RideDemandException(e,sys)
    