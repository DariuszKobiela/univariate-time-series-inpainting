import pandas as pd
from darts import TimeSeries
from darts.models import TCNModel
from pytorch_lightning.callbacks import EarlyStopping

# A model-specific parameter defining the minimum input length for training
TCN_INPUT_LEN = 100

def train_tcn(train_series: pd.Series, horizon: int, input_chunk_length: int = TCN_INPUT_LEN, output_chunk_length: int = 10, epochs: int = 100, random_state: int = None):
    """
    Trains a Temporal Convolutional Network (TCN) model using Darts and
    returns a forecast for the specified horizon.
    """
    
    # 1. Darts requires a TimeSeries object. We create a dummy index with
    #    a monthly frequency to stay within pandas' valid date bounds.
    date_index = pd.date_range(start='2000', periods=len(train_series), freq='M')
    full_ts = TimeSeries.from_times_and_values(times=date_index, values=train_series.values, freq='M')
    
    # Split the series into training and validation sets.
    # The last 20% of the data will be used for validation.
    train_split_point = int(len(full_ts) * 0.8)
    ts, val_ts = full_ts[:train_split_point], full_ts[train_split_point:]

    # Define the TCN model architecture and training parameters.
    # EarlyStopping helps prevent overfitting and reduces unnecessary training time.
    early_stopper = EarlyStopping(
        "val_loss", patience=5, min_delta=0.005, verbose=False
    )
    
    model = TCNModel(
        input_chunk_length=input_chunk_length,
        output_chunk_length=output_chunk_length,
        n_epochs=epochs,
        random_state=random_state,
        pl_trainer_kwargs={"callbacks": [early_stopper], "accelerator": "auto"}
    )
    
    # Train the model on the provided series.
    model.fit(ts, val_series=val_ts, verbose=False)
    
    # Generate the forecast.
    prediction = model.predict(n=horizon)
    
    # Manually construct the forecast Series to avoid potential method name issues.
    forecast_values = prediction.values().flatten()
    last_original_year = train_series.index[-1]
    forecast_index = range(last_original_year + 1, last_original_year + 1 + horizon)
    return pd.Series(forecast_values, index=forecast_index, name='predicted')
