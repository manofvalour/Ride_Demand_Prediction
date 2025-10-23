import os, sys
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.DynamicPricingEngine.logger.logger import logger
from src.DynamicPricingEngine.exception.customexception import RideDemandException
from src.DynamicPricingEngine.config.configuration import ModelTrainerConfig
import hopsworks

from dotenv import load_dotenv
load_dotenv()

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        try:
            self.config= config

        except Exception as e:
            raise RideDemandException(e,sys)
        
    def retrieve_engineered_feature(self):
        try:
            ## login to feature store
            api_key = os.getenv('HOPSWORKS_API_KEY')
            project = hopsworks.login(project='RideDemandPrediction', api_key_value=api_key)
            fs = project.get_feature_store()
            
            feature_view = fs.get_feature_view(name='ride_demand_fv', version= 1)

            df, _ = feature_view.get_training_data(training_dataset_version=1)
            
            return df
               
        except Exception as e:
            raise RideDemandException(e,sys)

    def data_preprocessing(self):
        pass

    def model_training_and_evaluation(self):
        pass

    def save_model_in_model_store(self):
        pass

    def initiate_model_training(self):
        pass

    