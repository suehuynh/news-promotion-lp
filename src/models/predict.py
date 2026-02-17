import pandas as pd

def xgb_predict(model, X, prediction_column="pred_shares"):
    """
    Generate predictions.

    Returns
    -------
    preds : array-like
        Predicted values.
    """

    preds = model.predict(X)

    return preds
