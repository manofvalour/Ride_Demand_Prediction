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
from sklearn.ensemble import RandomForestRegressor
import dagshub
import dill
from sklearn.feature_selection import mutual_info_regression
import mlflow
from pathlib import Path
from hsml.schema import Schema
from hsml.model_schema import ModelSchema

from src.DynamicPricingEngine.logger.logger import logger
from src.DynamicPricingEngine.exception.customexception import RideDemandException
from src.DynamicPricingEngine.config.configuration import ModelTrainerConfig
from src.DynamicPricingEngine.utils.data_ingestion_utils import time_subtract
from src.DynamicPricingEngine.utils.ml_utils import evaluate_model
from src.DynamicPricingEngine.utils.common_utils import save_pickle
import hopsworks
from hsfs.feature_view import FeatureView

from dotenv import load_dotenv
load_dotenv()

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        try:
            dagshub.init(repo_owner='manofvalour', 
                         repo_name='Dynamic-Pricing-Engine',
                         mlflow=True)

            mlflow.set_tracking_uri("https://dagshub.com/manofvalour/Dynamic-Pricing-Engine.mlflow")
            mlflow.set_experiment(experiment_name="Ride Demand Prediction")

            self.api_key = os.getenv('HOPSWORKS_API_KEY')
            self.config= config
            self.cat_cols = ['day_of_week','is_night_hour','pickup_hour',
                             'is_rush_hour','pulocationid', 
                             "pickup_month"]
            self.project = hopsworks.login(project='RideDemandPrediction',
                                      api_key_value=self.api_key)


        except Exception as e:
            raise RideDemandException(e,sys)
        
    def retrieve_engineered_feature(self) -> pd.DataFrame:
        try:

            now = datetime.today()
            end_date = now - timedelta(days=now.day) ## retrieving the last day of the previous month

            ## accessing the previous month
            days_to_subtract = time_subtract(end_date.strftime('%Y-%m-%d'))
            end_date = (end_date- timedelta(days=days_to_subtract)+ timedelta(days=1))
            start_date = end_date - relativedelta(months=7)  #a year back from end date

            start_date = start_date.strftime('%Y-%m-%d')
            end_date = end_date.strftime('%Y-%m-%d')

            logger.info('Retrieving the dataset from hopsworks feature store')

            fs = self.project.get_feature_store()
            
            fg = fs.get_feature_group(
                name = 'nycdemandprediction',
                version = 1)

            final_features = ['pickup_month', 'city_congestion_index', 'zone_congestion_index', 
                              'humidity', 'precip','windspeed', 'feelslike', 'visibility', 
                              'pickup_hour', 'day_of_week','is_rush_hour', 'is_night_hour', 
                              'target_yellow_lag_1h','target_yellow_lag_24h', 
                              'target_green_lag_1h', 'target_green_lag_24h',
                              'target_hvfhv_lag_1h', 'target_hvfhv_lag_24h',
                              'target_yellow_city_hour_pickups_lag_1h',
                              'target_green_city_hour_pickups_lag_1h', 'pulocationid',
                              'target_hvfhv_city_hour_pickups_lag_1h',
                              'neighbor_pickups_target_yellow_lag_1h',
                              'neighbor_pickups_target_green_lag_1h',
                              'neighbor_pickups_target_hvfhv_lag_1h','bin',
                              'target_yellow', 'target_green', 'target_hvfhv']
            
            query = fg.select(final_features).filter(
                (fg.bin >= start_date) & (fg.bin <= end_date))

            df = query.read()
            logger.info(f"Successfully retrieved {len(df)} rows for window: {start_date} to {end_date}")

            df.columns = df.columns.str.replace('nycdemandprediction_', '', regex=False)
            df.set_index(['bin'], inplace=True)

            return df

        except Exception as e:
            logger.error(f"Failed to extract historical pickup data: {e}")
            raise RideDemandException(e,sys)
       
    def split_data(self, df):
        try:
            # Split per zone
            train_list = []
            val_list = []
            test_list = []

            for zone_id, group in df.groupby('pulocationid'):
                n =len(group)
                train_end = int(n * self.config.train_split_ratio)
                val_end   = int(n * self.config.val_split_ratio)

                train_list.append(group.iloc[:train_end])
                val_list.append(group.iloc[train_end:val_end])
                test_list.append(group.iloc[val_end:])

            # Concatenate all zones
            train_df = pd.concat(train_list).sort_index()
            val_df = pd.concat(val_list).sort_index()
            test_df = pd.concat(test_list).sort_index()

            logger.info(f"Data split successfully!")
            logger.info(f"Train split: {train_df.shape}")
            logger.info(f"Val Split: {val_df.shape}")
            logger.info(f"Test split: {test_df.shape}")

            return train_df, val_df, test_df
        
        except Exception as e:
            logger.error(f"Unable to split the dataset")
            raise RideDemandException(e, sys)

    def _prepare_features(self, df: pd.DataFrame, 
                      targets: list[str]=['target_yellow', 'target_green', 'target_hvfhv']):
        """
        Separates the dataframe into features (X) and targets (y), 
        converts categorical types, and ensures only valid features are kept.
        """
        try:
            #Separate targets and features
            y = df[targets]
            
            # Define non-feature columns that should be dropped 
            to_drop = targets + ['pickup_datetime', 'bin']
            df.reset_index(inplace=True)
            X = df.drop(columns=to_drop, errors='ignore')

            #Handle Categorical Columns
            for col in self.cat_cols:
                if col in X.columns:
                    X[col] = X[col].astype('category')

            logger.info(f"Data Split into {X.shape}, {y.shape}")
            return X, y

        except Exception as e:
            logger.error(f"Feature preparation failed: {e}")
            raise RideDemandException(e,sys)

    def model_training_and_evaluation(self, train_df:pd.DataFrame,
                                    val_df:pd.DataFrame,
                                    test_df:pd.DataFrame):
        try:
            target = ['target_yellow', 'target_green', 'target_hvfhv']

            ## models for training data
            models = {"lgbm": LGBMRegressor,
                    #"xgboost": XGBRegressor,
                    #"random_forest": RandomForestRegressor,
                    #'catboost': CatBoostRegressor
                    }

            X_train, y_train = self._prepare_features(train_df, target)
            X_val, y_val = self._prepare_features(val_df, target)
            X_test, y_test = self._prepare_features(test_df, target)

            # Model Training and Hyperparameter Tuning
            model_report, trained_models = evaluate_model(x_train=X_train, y_train=y_train,
                                            x_test=X_val, y_test=y_val, models=models,
                                            param_spaces=self.config.optuna_param_spaces,
                                            n_trials=30)

            ## selecting and saving the best model
            result_df = pd.DataFrame(model_report).T.sort_values(by='rmse', ascending=True) ## converting report to dataframe
            best_model_name =result_df.index[0] ## selecting the top model (model with the lowest 'RMSE')
            best_model = trained_models[best_model_name]

            logger.info("\nFINAL RESULTS")
            for name, metrics in model_report.items():
                logger.info(f"{name:8} → MAE: {metrics['mae']:.3f} | RMSE: {metrics['rmse']:.3f} | R²: {metrics['R2_score']:.3f}")
                best_model_metrics = model_report[best_model_name]

            logger.info(f"{best_model_name} is the model with the best metrics")
            return best_model, best_model_metrics, X_test, y_test

        except Exception as e:
            logger.error(f"Model training and evaluation failed, {e}")
            raise RideDemandException(e,sys)
        
    def save_model_to_model_store(self, model, model_metrics, 
                                  x_test, y_test):
        """ saving model to the artifact store """
        try:
            # saving model 
            model_dir = self.config.trained_model_path ## model path
            os.makedirs(os.path.dirname(model_dir), exist_ok=True)

            with open(model_dir, 'wb') as file_path:
                dill.dump(model,file_path)     

            ## saving the model to hopsworks model store
            logger.info('Saving the model to hopsworks model registry')
            
            # Define the input and output schema from your training data
            input_schema = Schema(x_test)
            output_schema = Schema(y_test)

            # Create the Model Schema object
            model_schema = ModelSchema(input_schema, output_schema)
            model_registry = self.project.get_model_registry()

            model_hopsworks = model_registry.python.create_model(
                name="ride_demand_prediction_model",
                metrics= model_metrics,
                model_schema = model_schema,
                description="Model to predict ride demand based on historical features."
            )
            model_hopsworks.save(model_dir)

            logger.info(f"Trained model saved to {model_dir}")

        except Exception as e:
            logger.error(f'Failed to save the Best model to the model artifact store')
            raise RideDemandException(e,sys)
