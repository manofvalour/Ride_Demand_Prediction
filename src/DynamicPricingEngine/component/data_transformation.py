import os,sys
import pandas as pd
import numpy as np
from datetime import datetime

from src.DynamicPricingEngine.logger.logger import logger
from src.DynamicPricingEngine.exception.customexception import RideDemandException
from src.DynamicPricingEngine.config.configuration import ConfigurationManager
from src.DynamicPricingEngine.entity.config_entity import DataTransformationConfig
from src.DynamicPricingEngine.utils.common_utils import create_dir

class DataTransformation:
    def __init__(self, config:DataTransformationConfig):
        pass

    def feature_engineering(self):
        
        """
        Things to do here:
        1. generating target features
        2. generating the weather feature and joining with target feature
        3. engineering the temporal features
        4. engineering the spatial features
        5. the traffic and congestion features
        6. lag and rolling demand features
        7. derived and interaction features
        """