# optuna_evaluate_with_metrics.py
import sys
from typing import Dict, Any, Tuple
import optuna
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import numpy as np
import os, sys
import mlflow
from sklearn.multioutput import MultiOutputRegressor

from src.DynamicPricingEngine.exception.customexception import RideDemandException
from src.DynamicPricingEngine.logger.logger import logger

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

class _ModelTrial:
    """Wraps one model + Optuna search space."""
    def __init__(self, name: str, model_cls, param_space: Dict[str, Any]):
        self.name = name
        self.model_cls = model_cls
        self.param_space = param_space
        self.cat_cols = ['day_of_week','is_night_hour','pickup_hour',
                             'is_rush_hour','pulocationid',
                             "pickup_month"]

    def _objective(self, trial, X_train, y_train, X_val, y_val):
        try:
            params = {}
            for pname, pdef in self.param_space.items():
                if pdef['type'] == "int":
                    params[pname] = trial.suggest_int(pname, int(pdef["low"]), int(pdef["high"]))
                elif pdef["type"] == "float":
                    params[pname] = trial.suggest_float(pname, float(pdef["low"]), 
                                                        float(pdef["high"]), 
                                                        log=bool(pdef.get("log", False)))
                elif pdef["type"] == "categorical":
                    params[pname] = trial.suggest_categorical(pname, pdef["choices"])

            # Logic for Categorical Encoding
            if self.name in ['random_forest', 'decision_tree']:
                from sklearn.pipeline import Pipeline
                from sklearn.preprocessing import OneHotEncoder
                from sklearn.compose import ColumnTransformer

                preprocessor = ColumnTransformer(
                    transformers=[('cat', OneHotEncoder(handle_unknown='ignore', 
                                                        sparse_output=False), self.cat_cols)],
                    remainder='passthrough'
                )
                
                base_model = Pipeline([
                    ('prep', preprocessor),
                    ('reg', self.model_cls(**params, n_jobs=-1))
                ])
            
            elif self.name == "lgbm":
                base_model = self.model_cls(**params, device='cpu', n_jobs=-1, verbosity=-1)
            
            elif self.name == 'catboost':
                cat_idx = [X_train.columns.get_loc(c) for c in self.cat_cols if c in X_train.columns]
                # CatBoost handles categories internally
                base_model = self.model_cls(**params, loss_function="MultiRMSE", task_type="CPU", thread_count=-1, 
                                            cat_features=cat_idx, verbose=False)
                
            elif self.name == 'xgboost':
                # XGBoost handles categories internally if enable_categorical=True
                base_model = self.model_cls(enable_categorical=True, **params, n_jobs=-1)

            # Wrap and Fit
            model = MultiOutputRegressor(base_model)
            model.fit(X_train, y_train)

            val_pred = model.predict(X_val)
            return np.sqrt(mean_squared_error(y_val, val_pred))

        except Exception as e:
            logger.error(f"Failed during Optuna trial: {e}")
            raise RideDemandException(e,sys)

    def tune(self, X_train, y_train, X_val, 
             y_val, n_trials: int = 30):
        """ Optuna Tuning"""

        try:
            study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=42),
                pruner=optuna.pruners.MedianPruner())

            study.optimize(
                lambda t: self._objective(t, X_train, y_train, X_val, y_val),
                n_trials=n_trials,
                show_progress_bar=True,
            )

            logger.info('Hyperparameter Tuning completed successfully')
            return study.best_params

        except Exception as e:
            logger.error(f"Hyperparameter Tuning Failed, {e}")
            raise RideDemandException(e,sys)



 ## For tracking experiments
def log_model(name, model, params, X_val, y_val):
    try:
        with mlflow.start_run():

            if y_val is not None:
                y_pred = model.predict(X_val)

                rmse_values = []
                mae_values = []
                r2_values = []
                target_names = ['target_yellow', 'target_green', 'target_hvfhv']
                for i, target in enumerate(target_names):

                    rmse = np.sqrt(mean_squared_error(y_val.iloc[:,i], y_pred[:,i]))
                    mae = mean_absolute_error(y_val.iloc[:,i], y_pred[:,i])
                    r2 = r2_score(y_val.iloc[:,i], y_pred[:,i])

                    mlflow.log_metric(f"{target}_val_rmse", rmse)
                    mlflow.log_metric(f"{target}_val_mae", mae)
                    mlflow.log_metric(f"{target}_val_r2_score", r2)
                    mlflow.log_params(model.get_params())
                    logger.info(f"{name} model parameter and evaluation metrics tracked with MLflow")
                
                    rmse_values.append(rmse)
                    mae_values.append(mae)
                    r2_values.append(r2)
                
                # Calculate overall metrics as the mean of individual target metrics
                overall_rmse = np.mean(rmse_values)
                overall_mae = np.mean(mae_values)
                overall_r2 = np.mean(r2_values)

            return overall_rmse, overall_mae, overall_r2 # Return single values for summary

    except Exception as e:
        logger.error(f'failed to log parameter and metrics to MLFlow, {e}')
        raise RideDemandException(e,sys)


## Evaluating models
def evaluate_model(x_train, y_train, x_test, y_test, 
                   models, param_spaces, n_trials=30):
    try:
        report = {}
        trained_models = {}
        cat_cols = ['day_of_week','is_night_hour','pickup_hour',
                    'is_rush_hour','pulocationid',"pickup_month"]

        for name, model_cls in models.items():
            print(f"\n TUNING {name.upper()}")

            trial = _ModelTrial(name, model_cls, param_spaces[name])
            best_params = trial.tune(x_train, y_train, x_test, y_test, n_trials=n_trials)

            # Final Model
            if name == 'random_forest':
                preprocessor = ColumnTransformer(
                    transformers=[('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)],
                    remainder='passthrough'
                )
                base_est = Pipeline([('prep', preprocessor), ('reg', model_cls(**best_params, n_jobs=-1))])
            
            elif name == 'catboost':
                cat_idx = [x_train.columns.get_loc(c) for c in cat_cols if c in x_train.columns]
                base_est = model_cls(**best_params, thread_count=-1, cat_features=cat_idx, verbose=False)
            
            elif name == 'xgboost':
                base_est = model_cls(enable_categorical=True, **best_params, n_jobs=-1)
            
            else: # Default
                base_est = model_cls(**best_params, n_jobs=-1, verbosity=-1)

            # Wrap in MultiOutput and Fit
            final_model = MultiOutputRegressor(base_est)
            final_model.fit(x_train, y_train)

            # Log to MLflow and Evaluate
            rmse, mae, r2 = log_model(name, final_model, best_params, x_test, y_test)

            report[name] = {"rmse": rmse, "mae": mae, "R2_score": r2}
            trained_models[name] = final_model
            logger.info(f"{name.upper()} → MAE: {mae:.3f} | R2: {r2:.3f}")

        return report, trained_models

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise RideDemandException(e,sys)