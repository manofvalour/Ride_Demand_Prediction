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

from src.DynamicPricingEngine.logger.logger import logger
from src.DynamicPricingEngine.exception.customexception import RideDemandException
from src.DynamicPricingEngine.config.configuration import ModelTrainerConfig
from src.DynamicPricingEngine.utils.data_ingestion_utils import time_subtract
import hopsworks
from hsfs.feature_view import FeatureView

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
                train_end = int(len(group) * 0.7)
                val_end   = int(len(group) * 0.85)

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


    def data_preprocessing(self, train_df:pd.DataFrame, test_df:pd.DataFrame):
        
        try:
            ## spliting the feature and label
            X_train = train_df.drop(columns=['pickups'])
            y_train = train_df['pickups']

            ##test
            X_test = test_df.drop(columns=['pickups'])
            y_test = test_df['pickups']

            ## dropping columns
            columns_to_drop = ['bin_str', 'datetime']
            X_train.drop(columns=columns_to_drop, inplace = True)
            X_test.drop(columns=columns_to_drop, inplace = True)

            logger.info(f"Columns are successfully dropped: {columns_to_drop}")

            ## splitting numerical and categorical
            num_cols = X_train.dtypes[X_train.dtypes != 'object'].index
            cat_cols =  X_train.dtypes[X_train.dtypes == 'object'].index

            ## Setting up the preprocessing pipeline
            numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median'))
            ])

            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])

            preprocess = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, num_cols),
                    ('cat', categorical_transformer, cat_cols)
                ]
            )

            ##Applying the preprocessing
            data_pipeline = Pipeline(steps=[
                ('preprocess', preprocess),
                ('scaler', StandardScaler())
            ])

            X_train_transformed = data_pipeline.fit_transform(X_train)
            X_test_transformed = data_pipeline.transform(X_test)

            logger.info("train and test data preprocessed")

            ## saving the dataset as an numpy array using np.C_
            train_arr = np.c_[X_train_transformed, y_train]
            test_arr = np.c_[X_test_transformed, y_test]

            return train_arr, test_arr 

        except Exception as e:
            logger.error(f"Error preprocessing data, {e}")
            raise RideDemandException(e,sys)

    def model_training_and_evaluation(self, train_arr, test_arr):
        pass

    def save_model_in_model_store(self):
        pass

    def initiate_model_training(self):
        pass

    