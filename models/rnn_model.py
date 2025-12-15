import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.models import RNNModel
from pytorch_lightning.callbacks import EarlyStopping

# Model-specific parameters
RNN_INPUT_LEN = 100
RNN_HIDDEN_DIM = 32
RNN_LAYERS = 2

def train_rnn(train_series: pd.Series, horizon: int, model_type: str = "LSTM", 
              input_chunk_length: int = RNN_INPUT_LEN, hidden_dim: int = RNN_HIDDEN_DIM,
              n_rnn_layers: int = RNN_LAYERS, epochs: int = 100, random_state: int = None):
    """
    Trains a Recurrent Neural Network (RNN) model using Darts.
    Supports LSTM, GRU, and vanilla RNN architectures.
    
    Parameters
    ----------
    train_series : pd.Series
        Training time series data
    horizon : int
        Number of steps to forecast
    model_type : str
        Type of RNN: "LSTM", "GRU", or "RNN" (default: "LSTM")
    input_chunk_length : int
        Number of past time steps to use as input
    hidden_dim : int
        Size of hidden layer
    n_rnn_layers : int
        Number of RNN layers
    epochs : int
        Number of training epochs
    random_state : int
        Random seed for reproducibility
        
    Returns
    -------
    pd.Series
        Forecasted values with integer index
    """
    
    try:
        # 1. Darts requires a TimeSeries object with datetime index
        # Using hourly frequency to avoid overflow for long series
        date_index = pd.date_range(start='2000-01-01', periods=len(train_series), freq='H')
        full_ts = TimeSeries.from_times_and_values(times=date_index, values=train_series.values, freq='H')
        
        # 2. Split into training and validation sets
        # Last 20% for validation
        train_split_point = int(len(full_ts) * 0.8)
        ts, val_ts = full_ts[:train_split_point], full_ts[train_split_point:]
        
        # Ensure we have enough data for the model
        if len(ts) < input_chunk_length:
            # Fallback: use smaller input chunk length
            input_chunk_length = max(10, len(ts) // 2)
        
        # 3. Define EarlyStopping callback
        early_stopper = EarlyStopping(
            "val_loss", patience=5, min_delta=0.005, verbose=False
        )
        
        # 4. Initialize RNN model (LSTM, GRU, or vanilla RNN)
        model = RNNModel(
            model=model_type,
            input_chunk_length=input_chunk_length,
            training_length=min(24, input_chunk_length),
            hidden_dim=hidden_dim,
            n_rnn_layers=n_rnn_layers,
            dropout=0.1,
            batch_size=32,
            n_epochs=epochs,
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
        prediction = model.predict(n=horizon)
        
        # 7. Convert back to pd.Series with integer index
        forecast_values = prediction.values().flatten()
        last_original_year = train_series.index[-1]
        forecast_index = range(last_original_year + 1, last_original_year + 1 + horizon)
        
        return pd.Series(forecast_values, index=forecast_index, name='predicted')
        
    except Exception as e:
        print(f"Warning: RNN forecasting failed: {e}")
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


def train_lstm(train_series: pd.Series, horizon: int, random_state: int = None):
    """
    Trains an LSTM model (wrapper for train_rnn with model_type='LSTM').
    
    Parameters
    ----------
    train_series : pd.Series
        Training time series data
    horizon : int
        Number of steps to forecast
    random_state : int
        Random seed for reproducibility
        
    Returns
    -------
    pd.Series
        Forecasted values with integer index
    """
    return train_rnn(train_series, horizon, model_type="LSTM", random_state=random_state)


def train_gru(train_series: pd.Series, horizon: int, random_state: int = None):
    """
    Trains a GRU model (wrapper for train_rnn with model_type='GRU').
    
    Parameters
    ----------
    train_series : pd.Series
        Training time series data
    horizon : int
        Number of steps to forecast
    random_state : int
        Random seed for reproducibility
        
    Returns
    -------
    pd.Series
        Forecasted values with integer index
    """
    return train_rnn(train_series, horizon, model_type="GRU", random_state=random_state)

