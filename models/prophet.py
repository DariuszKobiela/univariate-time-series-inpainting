import pandas as pd
from prophet import Prophet

def train_prophet(train_series: pd.Series, horizon: int, random_state: int = None):
    """
    Prepares data for Prophet, trains a model, and returns a forecast.
    Note: random_state is ignored as Prophet's point forecast is deterministic.
    """
    # 1. Convert the input series to the required format ['ds', 'y'].
    df = pd.DataFrame({'y': train_series.values})

    # 2. Create a "dummy" date column ('ds') using a daily frequency to
    #    ensure the date range stays within pandas' valid timestamp bounds.
    df['ds'] = pd.date_range(start='2000-01-01', periods=len(train_series), freq='D')

    # 3. Train the model
    model = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
    model.fit(df)

    # 4. Make future dataframe and forecast
    future = model.make_future_dataframe(periods=horizon, freq='D')
    forecast = model.predict(future)

    # 5. Return only the forecasted values with the correct integer year index.
    forecast_values = forecast['yhat'].iloc[-horizon:].values
    last_original_year = train_series.index[-1]
    forecast_index = range(last_original_year + 1, last_original_year + 1 + horizon)
    
    return pd.Series(forecast_values, index=forecast_index, name="yhat")