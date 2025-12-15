import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.models import RNNModel
from darts.utils.likelihood_models import GaussianLikelihood, QuantileRegression
from pytorch_lightning.callbacks import EarlyStopping

# Model-specific parameters
DEEPAR_INPUT_LEN = 100
DEEPAR_HIDDEN_DIM = 40
DEEPAR_LAYERS = 2

def train_deepar(train_series: pd.Series, horizon: int,
                 input_chunk_length: int = DEEPAR_INPUT_LEN,
                 hidden_dim: int = DEEPAR_HIDDEN_DIM,
                 n_rnn_layers: int = DEEPAR_LAYERS,
                 epochs: int = 100,
                 random_state: int = None):
    """
    Trains a DeepAR-like model using Darts RNNModel with probabilistic likelihood.
    
    DeepAR is a probabilistic forecasting method based on autoregressive RNNs,
    developed by Amazon. This implementation uses:
    - LSTM architecture for capturing temporal dependencies
    - Gaussian likelihood for probabilistic forecasts
    - Autoregressive approach for multi-step predictions
    
    The model predicts a probability distribution over future values rather than
    point estimates, providing uncertainty quantification.
    
    Parameters
    ----------
    train_series : pd.Series
        Training time series data
    horizon : int
        Number of steps to forecast
    input_chunk_length : int
        Number of past time steps to use as input (context length)
    hidden_dim : int
        Size of LSTM hidden state
    n_rnn_layers : int
        Number of LSTM layers
    epochs : int
        Number of training epochs
    random_state : int
        Random seed for reproducibility
        
    Returns
    -------
    pd.Series
        Forecasted values (mean of the distribution) with integer index
        
    Notes
    -----
    This implementation uses the mean of the predicted Gaussian distribution.
    For full probabilistic forecasts with prediction intervals, use the model's
    predict() method with num_samples parameter.
    """
    
    try:
        # 1. Create TimeSeries with datetime index
        # Using hourly frequency to avoid overflow
        date_index = pd.date_range(start='2000-01-01', periods=len(train_series), freq='H')
        full_ts = TimeSeries.from_times_and_values(times=date_index, values=train_series.values, freq='H')
        
        # 2. Split into training and validation sets
        train_split_point = int(len(full_ts) * 0.8)
        ts, val_ts = full_ts[:train_split_point], full_ts[train_split_point:]
        
        # Adjust input_chunk_length if series is too short
        if len(ts) < input_chunk_length:
            input_chunk_length = max(10, len(ts) // 2)
        
        # 3. Define EarlyStopping callback
        early_stopper = EarlyStopping(
            "val_loss", patience=5, min_delta=0.005, verbose=False
        )
        
        # 4. Initialize DeepAR-like model
        # Key difference from standard RNN: uses GaussianLikelihood for probabilistic forecasts
        model = RNNModel(
            model="LSTM",  # DeepAR uses LSTM
            input_chunk_length=input_chunk_length,
            training_length=min(24, input_chunk_length),
            hidden_dim=hidden_dim,
            n_rnn_layers=n_rnn_layers,
            dropout=0.1,
            batch_size=32,
            n_epochs=epochs,
            likelihood=GaussianLikelihood(),  # Probabilistic output
            random_state=random_state,
            pl_trainer_kwargs={
                "callbacks": [early_stopper],
                "accelerator": "auto",
                "enable_progress_bar": False,
                "enable_model_summary": False
            },
            force_reset=True,
            save_checkpoints=False
        )
        
        # 5. Train the model
        model.fit(ts, val_series=val_ts, verbose=False)
        
        # 6. Generate forecast
        # For point forecasts, we use num_samples=1 (returns the mean)
        # For probabilistic forecasts with uncertainty, increase num_samples
        prediction = model.predict(n=horizon, num_samples=1)
        
        # 7. Convert back to pd.Series with integer index
        forecast_values = prediction.values().flatten()
        last_original_year = train_series.index[-1]
        forecast_index = range(last_original_year + 1, last_original_year + 1 + horizon)
        
        return pd.Series(forecast_values, index=forecast_index, name='predicted')
        
    except Exception as e:
        print(f"Warning: DeepAR forecasting failed: {e}")
        # Fallback: simple trend extrapolation
        clean_data = train_series.dropna()
        if len(clean_data) >= 2:
            trend = np.mean(np.diff(clean_data.tail(min(10, len(clean_data)))))
            last_value = clean_data.iloc[-1]
            forecast_values = [last_value + trend * (i + 1) for i in range(horizon)]
        else:
            last_value = clean_data.iloc[-1] if len(clean_data) > 0 else 0
            forecast_values = [last_value] * horizon
        
        last_original_year = train_series.index[-1]
        forecast_index = range(last_original_year + 1, last_original_year + 1 + horizon)
        return pd.Series(forecast_values, index=forecast_index, name='predicted')


def train_deepar_quantile(train_series: pd.Series, horizon: int, 
                          quantiles: list = None, random_state: int = None):
    """
    Trains a DeepAR model with quantile regression for probabilistic forecasts.
    
    This variant uses quantile regression instead of Gaussian likelihood,
    allowing for non-Gaussian distributions and direct prediction of quantiles.
    
    Parameters
    ----------
    train_series : pd.Series
        Training time series data
    horizon : int
        Number of steps to forecast
    quantiles : list
        List of quantiles to predict (default: [0.1, 0.5, 0.9])
    random_state : int
        Random seed for reproducibility
        
    Returns
    -------
    pd.Series
        Forecasted values (median) with integer index
    """
    if quantiles is None:
        quantiles = [0.1, 0.5, 0.9]  # Lower bound, median, upper bound
    
    try:
        # Create TimeSeries
        date_index = pd.date_range(start='2000-01-01', periods=len(train_series), freq='H')
        full_ts = TimeSeries.from_times_and_values(times=date_index, values=train_series.values, freq='H')
        
        # Split
        train_split_point = int(len(full_ts) * 0.8)
        ts, val_ts = full_ts[:train_split_point], full_ts[train_split_point:]
        
        input_chunk_length = min(DEEPAR_INPUT_LEN, len(ts) // 2)
        
        # EarlyStopping
        early_stopper = EarlyStopping("val_loss", patience=5, min_delta=0.005, verbose=False)
        
        # Model with Quantile Regression
        model = RNNModel(
            model="LSTM",
            input_chunk_length=input_chunk_length,
            training_length=min(24, input_chunk_length),
            hidden_dim=DEEPAR_HIDDEN_DIM,
            n_rnn_layers=DEEPAR_LAYERS,
            dropout=0.1,
            batch_size=32,
            n_epochs=100,
            likelihood=QuantileRegression(quantiles=quantiles),
            random_state=random_state,
            pl_trainer_kwargs={
                "callbacks": [early_stopper],
                "accelerator": "auto",
                "enable_progress_bar": False,
                "enable_model_summary": False
            },
            force_reset=True,
            save_checkpoints=False
        )
        
        model.fit(ts, val_series=val_ts, verbose=False)
        prediction = model.predict(n=horizon, num_samples=1)
        
        forecast_values = prediction.values().flatten()
        last_original_year = train_series.index[-1]
        forecast_index = range(last_original_year + 1, last_original_year + 1 + horizon)
        
        return pd.Series(forecast_values, index=forecast_index, name='predicted')
        
    except Exception as e:
        print(f"Warning: DeepAR quantile forecasting failed: {e}")
        # Fallback
        clean_data = train_series.dropna()
        if len(clean_data) >= 2:
            trend = np.mean(np.diff(clean_data.tail(min(10, len(clean_data)))))
            last_value = clean_data.iloc[-1]
            forecast_values = [last_value + trend * (i + 1) for i in range(horizon)]
        else:
            last_value = clean_data.iloc[-1] if len(clean_data) > 0 else 0
            forecast_values = [last_value] * horizon
        
        last_original_year = train_series.index[-1]
        forecast_index = range(last_original_year + 1, last_original_year + 1 + horizon)
        return pd.Series(forecast_values, index=forecast_index, name='predicted')

