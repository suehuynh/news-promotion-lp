"""
Data preprocessing module for the News Linear Programming project.

This module:
- Loads raw dataset
- Separates features and target
- Applies standard scaling and log-transforming
- Selects topic indicator columns
- Simplifies predictors
- Returns structured outputs for modeling
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
!pip install ucimlrepo
from ucimlrepo import fetch_ucirepo


def load_data() -> pd.DataFrame:
    """
    Load raw dataset.

    Parameters
    ----------
    filepath : str
        Path to dataset CSV file.

    Returns
    -------
    df : pd.DataFrame
    """
    # fetch dataset
    online_news_popularity = fetch_ucirepo(id=332)
    # data
    X = online_news_popularity.data.features
    y = online_news_popularity.data.targets
    # dataframe
    X = X.reset_index()
    y = y.reset_index()
    df = pd.merge(X,y, on='index', how='outer')

    return df


def categories_features(df: pd.DataFrame) -> list:
    """
    Identify binary category indicator columns.

    Returns
    -------
    list of topic column names
    """
    categories_features = [c for c in df.columns if c.startswith(" data_channel_is")]
    return categories_features


def preprocess_features(
    df: pd.DataFrame,
    target_column: str = " log_shares",
    scale_numeric: bool = True,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Full preprocessing pipeline.

    Steps:
    - Separate features and target
    - Scale numeric features
    - Add category indicator columns
    - Preserve time and attribute columns
    - Train/test split

    Parameters
    ----------
    df : pd.DataFrame
    target_column : str
    scale_numeric : bool
    test_size : float
    random_state : int

    Returns
    -------
    X_train, X_test, y_train, y_test
    categories_features : list
    scaler : fitted scaler (or None)
    """

    df = df.copy()
   
    # Add other column
    df[' data_channel_is_other'] = df[' data_channel_is_lifestyle'] + df[' data_channel_is_entertainment'] + df[' data_channel_is_bus'] + df[' data_channel_is_socmed'] + df[' data_channel_is_tech'] + df[' data_channel_is_world']
    df[' data_channel_is_other'] = 1 - df[' data_channel_is_other'] 

    # Identify features 
    categories_features = identify_topic_columns(df)
    time_features = [c for c in df.columns if c.startswith(" weekday_")]
    content_features = [' n_unique_tokens',' num_hrefs',' num_imgs', ' num_videos', 
                    ' average_token_length',' num_keywords', 
                    ' kw_avg_min', ' kw_avg_max', ' kw_min_avg',' kw_avg_avg', 
                    ' self_reference_avg_sharess',
                    ' LDA_00', ' LDA_01', ' LDA_02',
                    ' LDA_03', ' LDA_04', 
                    ' global_subjectivity',' global_sentiment_polarity', 
                    ' global_rate_positive_words',
                    ' avg_positive_polarity',
                    ' title_subjectivity', ' title_sentiment_polarity']

    feature_cols = categories_features + time_features + content_features
    
    # Log-transform target column
    df[" log_shares"] = np.log1p(df[" shares"])

    # Separate features and target
    y = df[target_column]
    X = df[feature_cols]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )
    
    # Scaling
    scaler = StandardScaler()
    
    X_train[content_features] = scaler.fit_transform(X_train[content_features])
    X_test[content_features] = scaler.transform(X_test[content_features])
    
    return X_train, X_test, y_train, y_test, topic_columns, scaler


def prepare_lp_dataframe(df: pd.DataFrame, prediction_column: str = "pred_shares"):
    """
    Prepare dataframe for LP solver.

    Assumes predictions already exist.

    Returns
    -------
    shares : np.array
    topic_indicators : dict of np.arrays
    """

    df = df.copy()

    if prediction_column not in df.columns:
        raise ValueError(f"{prediction_column} not found in dataframe")

    topic_columns = identify_topic_columns(df)

    shares = df[prediction_column].to_numpy()

    topic_dict = {
        col: df[col].to_numpy()
        for col in topic_columns
    }

    return shares, topic_dict
