import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def xgb_train(
    X_train,
    y_train,
    **params
):
    """
    Train XGBoost regressor.

    Returns
    -------
    model : trained XGBRegressor
    """

    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
  
    return model

