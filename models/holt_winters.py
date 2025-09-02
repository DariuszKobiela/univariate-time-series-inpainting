import pandas as pd
from statsmodels.tsa.api import ExponentialSmoothing

def train_holt_winters(train_series: pd.Series, horizon: int, random_state: int = None):
    """
    Trains a Holt-Winters Exponential Smoothing model and returns a forecast.
    Note: random_state is ignored as this model is deterministic.
    """
    # 1. Create a modern "dummy" DatetimeIndex to satisfy the model.
    series = train_series.copy()
    series.index = pd.date_range(start='2000-01-01', periods=len(series), freq='D')
        
    # 2. Train the model
    model = ExponentialSmoothing(
        series,
        seasonal_periods=7,  # Weekly seasonality for daily data
        trend="add",
        seasonal="add",
        initialization_method="estimated",
    ).fit()
    
    # 3. Generate the forecast and set the correct integer year index.
    forecast = model.forecast(steps=horizon)
    last_original_year = train_series.index[-1]
    forecast.index = range(last_original_year + 1, last_original_year + 1 + horizon)
    return forecast 