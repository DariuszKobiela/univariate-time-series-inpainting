# New Deep Learning Forecasting Models

This document describes the new deep learning models added to the time series inpainting project.

---

## 📋 Overview

Four new state-of-the-art deep learning architectures have been added for time series forecasting:

| Model | Architecture | Key Features | Best For |
|-------|--------------|--------------|----------|
| **LSTM/GRU** | Recurrent Neural Network | Sequential processing, memory cells | Short-to-medium sequences, temporal dependencies |
| **N-BEATS** | Neural Basis Expansion | Stacked blocks, doubly residual, interpretable | General forecasting, trend/seasonality decomposition |
| **Transformer (TFT)** | Attention-based | Multi-head attention, variable selection | Long-range dependencies, multiple time scales |
| **DeepAR** | Probabilistic RNN | Gaussian/Quantile likelihood, uncertainty | Probabilistic forecasts, risk assessment |

---

## 🚀 Usage

### In Experiments

To use these models in your experiments, simply add their names to the `forecasting_models` list:

```python
from iterative_experiment import IterativeExperiment

# Initialize experiment with new models
experiment = IterativeExperiment(
    data_paths=["data/0_source_data/boiler.csv"],
    forecasting_models=[
        # Traditional models
        "XGBoost",
        "Prophet",
        "SARIMAX",
        "HoltWinters",
        "TCN",
        
        # New deep learning models
        "LSTM",          # Long Short-Term Memory
        "GRU",           # Gated Recurrent Unit
        "NBEATS",        # Generic N-BEATS
        "NBEATS-I",      # Interpretable N-BEATS
        "TFT",           # Temporal Fusion Transformer
        "Transformer",   # Alias for TFT
        "DeepAR",        # DeepAR with Gaussian likelihood
        "DeepAR-Q"       # DeepAR with Quantile Regression
    ],
    n_iterations=5,
    test_size=10
)

# Run experiment
experiment.run_experiment()
```

### Standalone Usage

Each model can also be used independently:

#### LSTM/GRU

```python
from models.rnn_model import train_lstm, train_gru
import pandas as pd

# Your time series data
train_series = pd.Series([...], index=range(1000))

# Forecast next 10 points with LSTM
forecast_lstm = train_lstm(train_series, horizon=10, random_state=42)

# Forecast with GRU
forecast_gru = train_gru(train_series, horizon=10, random_state=42)
```

#### N-BEATS

```python
from models.nbeats_model import train_nbeats, train_nbeats_interpretable

# Generic N-BEATS
forecast = train_nbeats(train_series, horizon=10, random_state=42)

# Interpretable N-BEATS (separate trend and seasonality)
forecast_interpretable = train_nbeats_interpretable(
    train_series, 
    horizon=10, 
    random_state=42
)
```

#### Transformer (TFT)

```python
from models.transformer_model import train_tft, train_transformer

# Temporal Fusion Transformer
forecast = train_tft(train_series, horizon=10, random_state=42)

# Or use the alias
forecast = train_transformer(train_series, horizon=10, random_state=42)
```

#### DeepAR

```python
from models.deepar_model import train_deepar, train_deepar_quantile

# DeepAR with Gaussian likelihood
forecast = train_deepar(train_series, horizon=10, random_state=42)

# DeepAR with Quantile Regression (for prediction intervals)
forecast_quantile = train_deepar_quantile(
    train_series, 
    horizon=10,
    quantiles=[0.1, 0.5, 0.9],  # Lower, median, upper
    random_state=42
)
```

---

## 🔧 Advanced Configuration

### RNN Models (LSTM/GRU)

```python
from models.rnn_model import train_rnn

forecast = train_rnn(
    train_series,
    horizon=10,
    model_type="LSTM",           # "LSTM", "GRU", or "RNN"
    input_chunk_length=100,      # Lookback window
    hidden_dim=32,               # Hidden layer size
    n_rnn_layers=2,              # Number of RNN layers
    epochs=100,                  # Training epochs
    random_state=42
)
```

### N-BEATS

```python
from models.nbeats_model import train_nbeats

forecast = train_nbeats(
    train_series,
    horizon=10,
    input_chunk_length=24,       # Lookback window
    output_chunk_length=12,      # Forecast chunk size
    generic_architecture=True,   # False for interpretable
    num_stacks=30,               # Number of stacks
    num_blocks=1,                # Blocks per stack
    num_layers=4,                # Layers per block
    layer_widths=256,            # Layer width
    epochs=100,
    random_state=42
)
```

### TFT (Transformer)

```python
from models.transformer_model import train_tft

forecast = train_tft(
    train_series,
    horizon=10,
    input_chunk_length=24,       # Encoder length
    output_chunk_length=12,      # Decoder length
    hidden_size=64,              # Hidden state size
    lstm_layers=1,               # LSTM layers in encoder
    num_attention_heads=4,       # Attention heads
    epochs=100,
    random_state=42
)
```

### DeepAR

```python
from models.deepar_model import train_deepar

forecast = train_deepar(
    train_series,
    horizon=10,
    input_chunk_length=100,      # Context length
    hidden_dim=40,               # LSTM hidden size
    n_rnn_layers=2,              # LSTM layers
    epochs=100,
    random_state=42
)
```

---

## 📊 Model Characteristics

### LSTM/GRU

**Pros:**
- Handles sequential data naturally
- Good for capturing temporal patterns
- Flexible architecture
- Fast training for moderate sequence lengths

**Cons:**
- May struggle with very long sequences
- Requires sufficient training data
- Prone to overfitting on small datasets

**When to use:**
- Short-to-medium length sequences (< 1000 steps)
- Clear temporal dependencies
- When interpretability is not critical

---

### N-BEATS

**Pros:**
- State-of-the-art performance
- Interpretable variant available
- No need for feature engineering
- Handles trend and seasonality well

**Cons:**
- Computationally intensive
- Requires longer training time
- Many hyperparameters to tune

**When to use:**
- High accuracy is critical
- Trend/seasonality decomposition needed
- Sufficient computational resources available

---

### Transformer (TFT)

**Pros:**
- Captures long-range dependencies
- Multi-head attention for different patterns
- Variable selection for interpretability
- Excellent for complex patterns

**Cons:**
- Computationally expensive
- Requires more data than RNNs
- Longer training time

**When to use:**
- Long sequences with complex patterns
- Multiple time scales in data
- Attention mechanism beneficial

---

### DeepAR

**Pros:**
- Probabilistic forecasts
- Uncertainty quantification
- Good for risk assessment
- Handles missing values well

**Cons:**
- Point forecasts may be less accurate
- Requires understanding of probabilistic outputs
- Slower than deterministic models

**When to use:**
- Uncertainty quantification needed
- Risk assessment important
- Prediction intervals required

---

## 💾 Model Files

All new models are located in the `models/` directory:

```
models/
├── rnn_model.py           # LSTM, GRU
├── nbeats_model.py        # N-BEATS (generic and interpretable)
├── transformer_model.py   # TFT (Temporal Fusion Transformer)
├── deepar_model.py        # DeepAR (Gaussian and Quantile)
└── NEW_MODELS_README.md   # This file
```

---

## 🐛 Troubleshooting

### "Out of memory" errors

**Solution:** Reduce batch size or model complexity
```python
# Example for RNN
forecast = train_lstm(
    train_series, 
    horizon=10,
    hidden_dim=16,  # Reduced from 32
    n_rnn_layers=1  # Reduced from 2
)
```

### "Training too slow"

**Solution:** Reduce epochs or use simpler model
```python
forecast = train_nbeats(
    train_series,
    horizon=10,
    epochs=50,      # Reduced from 100
    num_stacks=10   # Reduced from 30
)
```

### "Model diverges / NaN loss"

**Solution:** Check for extreme values or scale data
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_data = scaler.fit_transform(train_series.values.reshape(-1, 1)).flatten()
scaled_series = pd.Series(scaled_data, index=train_series.index)

forecast_scaled = train_tft(scaled_series, horizon=10)

# Inverse transform
forecast = scaler.inverse_transform(forecast_scaled.values.reshape(-1, 1)).flatten()
```

---

## 📚 References

1. **LSTM/GRU:**
   - Hochreiter & Schmidhuber (1997). "Long Short-Term Memory"
   - Cho et al. (2014). "Learning Phrase Representations using RNN Encoder-Decoder"

2. **N-BEATS:**
   - Oreshkin et al. (2020). "N-BEATS: Neural basis expansion analysis for interpretable time series forecasting"

3. **TFT:**
   - Lim et al. (2021). "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting"

4. **DeepAR:**
   - Salinas et al. (2020). "DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks"

---

## 🤝 Integration with Existing Pipeline

All new models are fully integrated with the existing experimental pipeline:

1. **Data Loading:** Uses same data format as existing models
2. **Preprocessing:** Compatible with missingness injection
3. **Evaluation:** Same metrics (MAPE, SMAPE, MAE, RMSE)
4. **Output:** Consistent CSV format in `results/`

No changes to existing workflows are needed!

---

## 🎯 Recommended Model Selection

| Use Case | Recommended Model | Alternative |
|----------|-------------------|-------------|
| **Fast experimentation** | LSTM | GRU |
| **Best accuracy** | N-BEATS | TFT |
| **Interpretability** | NBEATS-I | Prophet |
| **Long sequences** | TFT | Transformer |
| **Uncertainty quantification** | DeepAR | DeepAR-Q |
| **Limited data** | LSTM | XGBoost |
| **Limited compute** | GRU | SARIMAX |

---

**Last Updated:** December 15, 2025  
**Author:** Darek (with Cursor AI)  
**Library:** Darts (https://unit8co.github.io/darts/)

