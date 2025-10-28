import os, sys
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.DynamicPricingEngine.logger.logger import logger
from src.DynamicPricingEngine.exception.customexception import RideDemandException


def compute_metrics(y_true, y_pred)->dict:

    mae  = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    return {'mae': mae, 'mse': mse, 
            'r2_score':r2, 'rmse': rmse}


