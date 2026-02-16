import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def train_xgb_model(
    X_train,
    y_train,
    random_state=42
):
    """
    Train XGBoost regressor.

    Returns
    -------
    model : trained XGBRegressor
    """

    if params is None:
        params = {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": random_state,
        }

    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
  
    return model
