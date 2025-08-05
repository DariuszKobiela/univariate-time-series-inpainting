from statsmodels.tsa.holtwinters import ExponentialSmoothing

def train_holt_winters(train_series, seasonal_periods=12, trend='add', seasonal='add'):
    model = ExponentialSmoothing(train_series, trend=trend, seasonal=seasonal, seasonal_periods=seasonal_periods)
    model_fit = model.fit()
    return model_fit

def forecast_holt_winters(model_fit, steps):
    return model_fit.forecast(steps)