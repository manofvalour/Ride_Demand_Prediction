from src.DynamicPricingEngine.logger.logger import logger
from src.DynamicPricingEngine.exception.customexception import RideDemandException
from src.DynamicPricingEngine.pipeline.feature_pipeline import FeaturePipeline
import sys

if __name__=="__main__":

    STAGE_NAME = 'Data Ingestion Stage'
    
   # try:
       # logger.info(f"{STAGE_NAME} initiated")
        #feature_pipeline = FeaturePipeline()
        #feature_pipeline.initiate_data_ingestion()

        #logger.info(f"{STAGE_NAME} completed")

    #except Exception as e:
     #   raise RideDemandException(e,sys)



    STAGE_NAME = 'Training Data Ingestion Stage'
    
    try:
        logger.info(f"{STAGE_NAME} initiated")
        feature_pipeline = FeaturePipeline()
        feature_pipeline.initiate_training_data()

        logger.info(f"{STAGE_NAME} completed")

    except Exception as e:
        raise RideDemandException(e,sys)


    
    STAGE_NAME = 'Data Transformation Stage'
    
    #try:
     #   logger.info(f"{STAGE_NAME} initiated")
      #  feature_pipeline = FeaturePipeline()
       # feature_pipeline.initiate_data_transformation()

        #logger.info(f"{STAGE_NAME} completed")

    #except Exception as e:
     #   raise RideDemandException(e,sys)
    
