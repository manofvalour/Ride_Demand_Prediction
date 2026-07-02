from src.DynamicPricingEngine.logger.logger import logger
from src.DynamicPricingEngine.exception.customexception import RideDemandException
from src.DynamicPricingEngine.pipeline.feature_pipeline import FeaturePipeline

import sys
from time import time

if __name__=="__main__":

    STAGE_NAME = 'Data Ingestion and Transformation Stage'
    
    try:
        t0 = time()
        logger.info(f"{STAGE_NAME} initiated")
        feature_pipeline = FeaturePipeline()
        feature_pipeline.initiate_data_ingestion_and_transformation()

        t1 = time()
        dt = (t1 - t0)
        logger.info(f"{STAGE_NAME} completed in {dt:.2f}secs")

    except Exception as e:
        raise RideDemandException(e,sys)