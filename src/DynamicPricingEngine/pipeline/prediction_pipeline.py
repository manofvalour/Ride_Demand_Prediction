import os, sys
import pandas as pd

from src.DynamicPricingEngine.exception.customexception import RideDemandException
from src.DynamicPricingEngine.logger.logger import logger
from src.DynamicPricingEngine.config.configuration import ConfigurationManager
from src.DynamicPricingEngine.component.inference import Inference
from src.DynamicPricingEngine.utils.data_ingestion_utils import time_subtract
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

class InferencePipeline:
    def __init__(self):
        # Defining the time range for historical data extraction
        now = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = now - timedelta(days=now.day) ## retrieving the last day of the previous month

        ## accessing the previous month
        days_to_subtract = time_subtract(end_date.strftime('%Y-%m-%d'))
        end_date = (end_date- timedelta(days=days_to_subtract)+ timedelta(days=1))

        ## a year back from end date
        start_date = end_date - relativedelta(days= 1)

        self.start_date = start_date.strftime('%Y-%m-%d %H:%M:%S')
        self.end_date = end_date.strftime('%Y-%m-%d %H:%M:%S')
        print(start_date, end_date)

    def initiate_inference(self, start_date=None, end_date=None)-> pd.DataFrame:
        try:

            logger.info('Extracting the prediction Data...')

            config = ConfigurationManager()
            inference_config = config.get_inference_config()
            logger.info('Inferencing Configuration successfully loaded')

            pred_pipe = Inference(inference_config)

            hist_data = pred_pipe.extract_historical_pickup_data(self.start_date, self.end_date)

            hist_data = pred_pipe.citywide_hourly_demand(hist_data)
            hist_data = pred_pipe.generate_neighbor_features(hist_data)

            df = pred_pipe.get_nyc_prediction_weather_data()
            df = pred_pipe.engineer_temporal_prediction_features(df)

            unique_pulocationids = pd.DataFrame({'pulocationid': hist_data.pulocationid.unique()})
            df = pd.merge(unique_pulocationids, df, how='cross')

            df = pred_pipe.get_zone_speeds(df)
            df = pred_pipe.congestion_features(df)
            final_df = pred_pipe.generate_lag_features_for_prediction(hist_data, df)
            final_df = pred_pipe.final_data(final_df)

            logger.info('Inference data created Successfully')

            model= pred_pipe.deploy_model_and_load()
            prediction = pred_pipe.prepare_and_predict(model, final_df)
            pred_pipe.push_predition_to_feature_store(prediction)
        
        except Exception as e:
            logger.error(f'Unable to initiate model training, {e}')
            raise RideDemandException(e,sys)
