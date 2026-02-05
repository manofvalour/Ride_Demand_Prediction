"""Model evaluation helpers (metrics) used by training and tests."""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(y_true, y_pred) -> dict:
    """Compute regression metrics between `y_true` and `y_pred`.

    Returns a dictionary containing MAE, MSE, R² and RMSE.
    """

    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    return {'mae': mae, 'mse': mse,
            'r2_score': r2, 'rmse': rmse}


