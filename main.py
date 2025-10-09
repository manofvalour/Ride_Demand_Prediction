from src.DynamicPricingEngine.logger.logger import logger
from src.DynamicPricingEngine.exception.customexception import DynamicPricingException
import sys

if __name__=="__main__()":
    
    try:
        logger.info("Testing the file system")
        1/0
    except Exception as e:
        raise DynamicPricingException(e,sys)