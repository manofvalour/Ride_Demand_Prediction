import sys
from src.DynamicPricingEngine.exception.customexception import RideDemandException
from src.DynamicPricingEngine.logger.logger import logger
from src.DynamicPricingEngine.config.configuration import ConfigurationManager
from src.DynamicPricingEngine.component.inference import Inference
from time import time



def predict():

    try:
        t0= time()
        logger.info(f"Prediction initiated")
        config = ConfigurationManager()
        inference_config = config.get_inference_config()
        pred_pipeline = Inference(inference_config)
        pred_pipeline.initiate_inference()

        t1 = time()
        dt = (t1 - t0)
        logger.info(f"Prediction completed in {dt:.2f}secs")

    except Exception as e:
        raise RideDemandException(e,sys)