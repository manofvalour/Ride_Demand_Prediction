# optuna_evaluate_with_metrics.py
import sys
from typing import Dict, Any, Tuple
import optuna
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import numpy as np
import os, sys
import mlflow

from src.DynamicPricingEngine.exception.customexception import RideDemandException
from src.DynamicPricingEngine.logger.logger import logger


class _ModelTrial:
    """Wraps one model + Optuna search space."""
    def __init__(self, name: str, model_cls, param_space: Dict[str, Any]):
        self.name = name
        self.model_cls = model_cls
        self.param_space = param_space
        self.cat_cols = ['pulocationid', 'pickup_hour', 'day_of_week', 'season_of_year',
                         'is_weekend', 'is_rush_hour', 'is_night_hour', 'is_holiday',
                         'is_special_event', 'is_payday']

    def _objective(self, trial, X_train, y_train, X_val, y_val):

        try:

            params = {}
            for pname, pdef in self.param_space.items():
                if pdef["type"] == "int":
                    params[pname] = trial.suggest_int(pname, pdef["low"], pdef["high"])

                elif pdef["type"] == "float":
                    params[pname] = trial.suggest_float(
                        pname, pdef["low"], pdef["high"], log=pdef.get("log", False)
                    )

                elif pdef["type"] == "categorical":
                    params[pname] = trial.suggest_categorical(pname, pdef["choices"])

            # Early stopping per model
            if self.name == "lgbm":
                model = self.model_cls(**params)
                model.fit(
                    X_train, y_train,
                    #eval_set=[(X_val, y_val)],
                # callbacks=[optuna.integration.LightGBMPruningCallback(trial, "rmse")],
                # verbose=False,
                )

            elif self.name == "random_forest":
                # One-hot encode categoricals

                cat_cols_present = [col for col in self.cat_cols if col in X_train.columns]
                X_train_encoded = pd.get_dummies(X_train, columns=cat_cols_present, drop_first=True)
                #X_val = pd.get_dummies(X_val, columns=cat_cols_present, drop_first=True)

                #X_train_enc, X_val_enc = X_train_enc.align(X_val_enc, join='left', axis=1, fill_value=0)

                model = self.model_cls(**params)
                model.fit(
                    X_train_encoded, y_train,
                    #eval_set=[(X_val_encoded, y_val)],
                    #early_stopping_rounds=50,
                    #verbose=False,
                )

            elif self.name == 'catboost':
            cat_idx = [X_train.columns.get_loc(c) for c in self.cat_cols if c in X_train.columns]
            model = self.model_cls(**params, task_type='GPU'if torch.cuda.is_available() else 'CPU')
            model.fit(X_train, y_train, cat_features=cat_idx,
                        #eval_set=[(X_val, y_val)],
                        verbose=False)

            elif self.name == 'xgboost':
            model = self.model_cls(enable_categorical=True, **params)#, early_stopping_rounds=10)
            model.fit(X_train, y_train,
                        #eval_set=[(X_val, y_val)],
                        verbose=False)

            else:
                model = self.model_cls(**params)
                model.fit(X_train, y_train)

            val_pred = model.predict(X_val)
            return np.sqrt(mean_squared_error(y_val, val_pred))  # Optuna minimizes

        except Exception as e:
            logger.error(f"failed to load parameters and train model")
            raise RideDemandException(e,sys)


    def tune(self, X_train, y_train, X_val, y_val, n_trials: int = 30):
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
                if name == 'Temporal Fusion Transformer':
                    y_pred = model.predict(X_val).numpy().flatten()
                else:
                    y_pred = model.predict(X_val)

                rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                mae = mean_absolute_error(y_val, y_pred)
                r2 = r2_score(y_true, y_pred)

            #  rmse, mae, r2 = regression_error_score(y_val, pred)

                mlflow.log_metric("val_rmse", rmse)
                mlflow.log_metric("val_mae", mae)
                mlflow.log_metric("val_r2_score", r2)
                mlflow.log_params(model.get_params())
                logger.info(f"{name} model parameter and evaluation metrics tracked with MLflow")

            return rmse, mae, r2
        
    except Exception as e:
        logger.error(f'failed to log parameter and metrics to MLFlow, {e}')
        raise RideDemandException(e,sys)


#def regression_error_score(y_true, y_pred):
 #   rmse = np.sqrt(mean_squared_error(y_true, y_pred))
  #  mae = mean_absolute_error(y_true, y_pred)
   # r2 = r2_score(y_true, y_pred)

    #return rmse, mae, r2



def evaluate_model(x_train: pd.DataFrame, y_train: pd.Series | np.ndarray, x_test: pd.DataFrame, 
                   y_test: pd.Series | np.ndarray, models: Dict[str, Any], 
                   param_spaces: Dict[str, Dict[str, Any]], n_trials: int = 30,
                   random_state: int = 42, epoch:int=5) -> Dict[str, Dict[str, float]]:
    """
    Returns: {"model_name": {"mae": ..., "rmse": ..., "r2": ...}}
    """
    try:
        report: Dict[str, Dict[str, float]] = {}
        trained_models = {}
        cat_cols = ['pulocationid', 'pickup_hour', 'day_of_week', 'season_of_year',
                         'is_weekend', 'is_rush_hour', 'is_night_hour', 'is_holiday',
                         'is_special_event', 'is_payday']

        for name, model_cls in models.items():
            print(f"\nTUNING {name.upper()}")

            trial = _ModelTrial(name, model_cls, param_spaces[name])
            best_params = trial.tune(x_train, y_train, x_test, y_test, n_trials=n_trials)
            print(f"Best params for {name}: {best_params}")

            # Retrain on FULL training data
            if name == "xgboost":
                final_model = model_cls(enable_categorical=True, **best_params)
                final_model.fit(x_train, y_train)


            elif name == 'catboost':
                cat_idx = [X_train.columns.get_loc(c) for c in cat_cols if c in X_train.columns]
                final_model = model_cls(**best_params, task_type='GPU')
                final_model.fit(X_train, y_train, cat_features=cat_idx,
                          #eval_set=[(X_val, y_val)],
                          verbose=False)
                
            elif name == 'random_forest':
                cat_cols_present = [col for col in cat_cols if col in X_train.columns]
                X_train_encoded = pd.get_dummies(X_train, columns=cat_cols_present, drop_first=True)
                x_test = pd.get_dummies(x_test, columns=cat_cols_present, drop_first=True)
                final_model = model_cls(**best_params)
                final_model.fit(X_train_encoded, y_train)

            else:
              final_model = model_cls(**best_params)
              final_model.fit(x_train, y_train)

            rmse, mae, r2= log_model(name, final_model, best_params, x_test, y_test)

            report[name] = {"rmse": rmse, "mae": mae, "R2_score": r2}
            trained_models[name]= final_model
            logger.info(f"{name.upper()} → MAE: {mae:.3f} | RMSE: {rmse:.3f} | R2_score: {r2:.3f}")

        return report, trained_models

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise RideDemandException (e,sys)