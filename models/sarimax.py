import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

def train_sarimax(train_series: pd.Series, horizon: int, random_state: int = None):
    """
    Trains a SARIMAX model and returns a forecast.
    Note: random_state is ignored as SARIMAX is deterministic.
    """
    # 1. Create a modern "dummy" DatetimeIndex with an hourly frequency
    #    to satisfy the model and avoid out-of-bounds errors.
    series = train_series.copy()
    series.index = pd.date_range(start='2000-01-01', periods=len(series), freq='H')

    # 2. Train the model
    model = SARIMAX(
        series,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False,
        suppress_warnings=True
    ).fit(disp=False)
    
    # 3. Generate the forecast
    forecast = model.forecast(steps=horizon)

    # 4. Convert the forecast's index back to the original integer years.
    last_original_year = train_series.index[-1]
    forecast.index = range(last_original_year + 1, last_original_year + 1 + horizon)
    return forecast