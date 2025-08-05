import pandas as pd
import numpy as np
import xgboost as xgb

def create_lag_features(series: pd.Series, lag: int):
    df = pd.DataFrame(series.values, columns=['y'])
    for i in range(1, lag + 1):
        df[f'lag_{i}'] = df['y'].shift(i)
    df.dropna(inplace=True)
    return df

def train_xgboost(train_series: pd.Series, horizon: int, lag: int = 10, random_state: int = None):
    """
    Creates lag features, trains an XGBoost model, and performs a recursive
    forecast for the specified horizon.
    """
    # Create lag features from the training data
    df = create_lag_features(train_series, lag)
    X, y = df.drop('y', axis=1), df['y']

    # Train the model
    model = xgb.XGBRegressor(
        objective='reg:squarederror', 
        n_estimators=100,
        random_state=random_state
    )
    model.fit(X, y)

    # Recursive forecasting
    history = list(train_series.values)
    predictions = []
    for _ in range(horizon):
        # Create input features from the most recent 'lag' values
        input_features = np.array(history[-lag:]).reshape(1, -1)
        
        # Predict the next step
        pred = model.predict(input_features)[0]
        predictions.append(pred)
        
        # Add the prediction to history for the next step's features
        history.append(pred)
        
    # Return the forecast with a proper integer year index
    last_original_year = train_series.index[-1]
    forecast_index = range(last_original_year + 1, last_original_year + 1 + horizon)
    return pd.Series(predictions, index=forecast_index)

def forecast_xgboost(model, series, lags, steps):
    predictions = []
    last_values = list(series[-lags:])
    for _ in range(steps):
        X = [last_values[-lag] for lag in range(1, lags+1)]
        pred = model.predict([X])[0]
        predictions.append(pred)
        last_values.append(pred)
    return predictions