import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

def train_sarimax(train_series: pd.Series, horizon: int, random_state: int = None):
    """
    Trains a SARIMAX model and returns a forecast.
    Note: random_state is ignored as SARIMAX is deterministic.
    
    Handles extreme values that can cause convergence issues by:
    1. Clipping extreme outliers
    2. Using simpler model if complex one fails
    3. Fallback to simple forecasting if all else fails
    """
    # 1. Create a modern "dummy" DatetimeIndex with an hourly frequency
    #    to satisfy the model and avoid out-of-bounds errors.
    series = train_series.copy()
    series.index = pd.date_range(start='2000-01-01', periods=len(series), freq='H')
    
    # 2. Check for and handle extreme values that can cause SARIMAX to fail
    # Calculate reasonable bounds based on the data
    mean_val = series.mean()
    std_val = series.std()
    
    # Clip extreme outliers (beyond 5 standard deviations)
    # This prevents polynomial interpolation artifacts from breaking SARIMAX
    if std_val > 0:
        lower_bound = mean_val - 5 * std_val
        upper_bound = mean_val + 5 * std_val
        series = series.clip(lower=lower_bound, upper=upper_bound)

    try:
        # 3. Train the model with the original parameters
        model = SARIMAX(
            series,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 12),
            enforce_stationarity=False,
            enforce_invertibility=False,
            suppress_warnings=True
        ).fit(disp=False)
        
        # 4. Generate the forecast
        forecast = model.forecast(steps=horizon)
        
        # Check if forecast contains inf or nan
        if not np.isfinite(forecast.values).all():
            raise ValueError("Forecast contains inf or nan values")
            
    except Exception as e:
        # If complex model fails, try simpler ARIMA model
        try:
            model = SARIMAX(
                series,
                order=(1, 1, 1),
                seasonal_order=(0, 0, 0, 0),  # No seasonal component
                enforce_stationarity=True,    # Enforce stability
                enforce_invertibility=True,
                suppress_warnings=True
            ).fit(disp=False, maxiter=50)
            
            forecast = model.forecast(steps=horizon)
            
            if not np.isfinite(forecast.values).all():
                raise ValueError("Forecast contains inf or nan values")
                
        except Exception as e2:
            # Last resort: simple trend extrapolation
            clean_series = series.dropna()
            if len(clean_series) >= 2:
                trend = np.mean(np.diff(clean_series.tail(min(10, len(clean_series)))))
                last_value = clean_series.iloc[-1]
                forecast_values = [last_value + trend * (i + 1) for i in range(horizon)]
            else:
                # Just repeat last value
                last_value = clean_series.iloc[-1] if len(clean_series) > 0 else 0
                forecast_values = [last_value] * horizon
            
            forecast = pd.Series(forecast_values)

    # 5. Convert the forecast's index back to the original integer years.
    last_original_year = train_series.index[-1]
    forecast.index = range(last_original_year + 1, last_original_year + 1 + horizon)
    return forecast