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

    def initiate_inference(self)-> pd.DataFrame:
        try:

            logger.info('Extracting the prediction Data...')

            config = ConfigurationManager()
            inference_config = config.get_inference_config()
            logger.info('Inferencing Configuration successfully loaded')

            pred_pipe = Inference(inference_config)

            weather_df = pred_pipe.get_nyc_prediction_weather_data()
            pred_df = pred_pipe.engineer_temporal_prediction_features(weather_df)
            hist_df = pred_pipe.extract_historical_pickup_data()

            unique_pulocationids = pd.DataFrame({'pulocationid': hist_df.pulocationid.unique()})
            pred_df = pd.merge(unique_pulocationids, pred_df, how='cross')
            pred_df = pred_pipe.get_zone_speeds(pred_df)
            pred_df = pred_pipe.congestion_features(pred_df)

            hist_dfs = pred_pipe.citywide_hourly_demand(hist_df)
            hist_dfs = pred_pipe.generate_neighbor_features(hist_dfs)

            df = pred_pipe.engineer_autoregressive_signals(hist_dfs, pred_df)
            df2 = pred_pipe.final_data(df)

            logger.info('Inference data created Successfully')

            model = pred_pipe.download_model_and_load()
            prediction = pred_pipe.prepare_and_predict(model, df2)
            pred_pipe.push_prediction_to_feature_store(prediction, hist_df)

        except Exception as e:
            logger.error(f'Unable to initiate model training, {e}')
            raise RideDemandException(e,sys)