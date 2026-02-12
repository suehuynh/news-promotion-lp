import pandas as pd

def xgb_predict(model, X, prediction_column="pred_shares"):
    """
    Generate predictions and append to dataframe.

    Returns
    -------
    df_with_predictions : pd.DataFrame
    """

    preds = model.predict(X)

    df_output = X.copy()
    df_output[prediction_column] = preds

    return df_output
