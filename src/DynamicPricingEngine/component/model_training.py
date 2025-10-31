import os, sys
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
from datetime import timedelta, datetime
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import dagshub

import optuna
import mlflow
from pathlib import Path
import lightgbm as lgbm

from src.DynamicPricingEngine.logger.logger import logger
from src.DynamicPricingEngine.exception.customexception import RideDemandException
from src.DynamicPricingEngine.config.configuration import ModelTrainerConfig
from src.DynamicPricingEngine.utils.data_ingestion_utils import time_subtract
from src.DynamicPricingEngine.utils.ml_utils import evaluate_model
import hopsworks
from hsfs.feature_view import FeatureView

from dotenv import load_dotenv
load_dotenv()

#dagshub.init(repo_owner='manofvalour',
 #            repo_name='Dynamic-Pricing-Engine',
  #           mlflow=True)

#mlflow.set_tracking_uri("https://dagshub.com/manofvalour/Dynamic-Pricing-Engine.mlflow")
#mlflow.set_experiment(experiment_name="Ride Demand Prediction")

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        try:
            self.config= config
            self.cat_cols = ['pulocationid', 'pickup_hour', 'day_of_week', 'season_of_year',
                        'is_weekend', 'is_rush_hour', 'is_night_hour', 'is_holiday', 
                        'is_special_event', 'is_payday']
     

        except Exception as e:
            raise RideDemandException(e,sys)
        
    def retrieve_engineered_feature(self):
        try:
            ## login to feature store
            api_key = os.getenv('HOPSWORKS_API_KEY')
            
            now = datetime.today().strftime("%Y-%m-%d")
            end_date = datetime.strptime(now, "%Y-%m-%d") ## converting to datetime
            end_date = end_date - timedelta(days=end_date.day) ## retrieving the last day of the previous month

            ## accessing the previous month
            days_to_subtract = time_subtract(end_date.strftime('%Y-%m-%d'))
            end_date = (end_date- timedelta(days=days_to_subtract)+ timedelta(days=1))

            ## a year back from end date 
            start_date = end_date - relativedelta(months= 12)


            ## login to feature store
            project = hopsworks.login(project='RideDemandPrediction', api_key_value=api_key)
            fs = project.get_feature_store()

            # Get the feature group
            fg = fs.get_feature_group(name="ridedemandprediction", version=1)
            query=fg.select_all()

            # creating a feature view

            # delete the previous month feature view data
            FeatureView.clean(feature_store_id=fs._id, 
                                feature_view_name='ride_demand_fv',
                                feature_view_version=1)

            # create a new feature view from the feature group
            feature_view = fs.create_feature_view(name="ride_demand_fv",
                                                    version=1,
                                                    description="Features for ride demand prediction",
                                                    query=query)

            logger.info('hopsworks feature view created successfully')

            feature_view = fs.get_feature_view(name='ride_demand_fv', version= 1)

            # Materialize training dataset using Spark job
            version, jobs = feature_view.create_training_data(start_time = start_date,
                                                                end_time = end_date,
                                                                description="365 days ride demand training data",
                                                                data_format="parquet",
                                                                write_options = {'use_spark': True}
                                                                )

            logger.info('Training data created successfully and materialized in hopsworks')
            logger.info(f"Data from {start_date} to {end_date} created and materialized Successfully")

            feature_view = fs.get_feature_view(name='ride_demand_fv', version= 1)

            df, _ = feature_view.get_training_data(training_dataset_version=1,
                                                read_options={"use_hive":False})
            
            logger.info('Data successfully retrieved from the feature store')

            ## splitting the dataset into train and test split
            df.sort_values(by=['pulocationid', 'bin'], inplace =True).reset_index(drop=True)
            df.set_index(['bin'], inplace=True)

            # Split per zone
            train_list = []
            val_list = []
            test_list = []

            for zone_id, group in df.groupby('pulocationid'):
                train_end = int(len(group) * self.config.train_split_ratio)
                val_end   = int(len(group) * self.config.val_split_ratio)

                train_list.append(group.iloc[:train_end])
                val_list.append(group.iloc[train_end:val_end])
                test_list.append(group.iloc[val_end:])

            # Concatenate all zones
            train_df = pd.concat(train_list)
            val_df = pd.concat(val_list)
            test_df = pd.concat(test_list)

            return train_df, val_df, test_df
                        
        except Exception as e:
            logger.error(f"Error retrieving the dataset, {e}")
            raise RideDemandException(e,sys)


    def prepare_features(self, df:pd.DataFrame, target:str):
        try:
            X = df.drop(columns=[target, "bin_str", 'datetime'], errors='ignore')
            y = df[target]

            for col in self.cat_cols:
                if col in X.columns:
                    X[col] = X[col].astype('category')

            return X, y
        
        except Exception as e:
            logger.error(f"feature preparation failed, {e}")
            raise RideDemandException(e,sys)

    def model_training_and_evaluation(self, 
    
                                    train_df:pd.DataFrame, 
                                    val_df:pd.DataFrame, 
                                    test_df:pd.DataFrame):
        try:
            target = self.config.target_col

            ## models for training data
            models = {"lgbm": LGBMRegressor,
                    #"xgboost": XGBRegressor,
                    #"catboost": CatBoostRegressor,
                    }

            X_train, y_train = self.prepare_features(train_df, target)
            X_val, y_val = self.prepare_features(val_df, target)
            X_test, y_test = self.prepare_features(test_df, target)
           
            # Model Training and Hyperparameter Tuning
            model_report, trained_models = evaluate_model(x_train=X_train, y_train=y_train,
                                            x_test=X_val, y_test=y_val, models=models,
                                            param_spaces=self.config.optuna_param_spaces,
                                            n_trials=5)

            ## selecting and saving the best model
            result_df = pd.DataFrame(model_report).T.sort_values(by='rmse', ascending=True) ## converting report to dataframe
            best_model_name =result_df.index[0] ## selecting the top model (model with the smallest 'RMSE')
            best_model = trained_models[best_model_name]

            # Pretty print
            logger.info("\nFINAL RESULTS")
            for name, metrics in model_report.items():
                logger.info(f"{name:8} → MAE: {metrics['mae']:.3f} | RMSE: {metrics['rmse']:.3f} | R²: {metrics['R2_score']:.3f}")

            return best_model

        except Exception as e:
            logger.error(f"Model training and evaluation failed, {e}")
            raise RideDemandException(e,sys)


    def save_model_in_model_store(self):
        """ saving model to the artifact store """
        pass

    def initiate_model_training(self):
        pass

    