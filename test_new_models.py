"""
Quick test script for new deep learning forecasting models.

This script demonstrates how to use the new models (LSTM, GRU, N-BEATS, TFT, DeepAR)
and compares their performance on a sample time series.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from models.rnn_model import train_lstm, train_gru
from models.nbeats_model import train_nbeats
from models.transformer_model import train_tft
from models.deepar_model import train_deepar

def generate_sample_series(length=1000):
    """Generate a sample time series with trend and seasonality"""
    t = np.arange(length)
    trend = 0.01 * t
    seasonality = 10 * np.sin(2 * np.pi * t / 50)
    noise = np.random.normal(0, 1, length)
    series = 100 + trend + seasonality + noise
    return pd.Series(series, index=range(length))

def test_single_model(model_name, train_func, train_series, horizon=20):
    """Test a single model and measure time"""
    import time
    
    print(f"\n{'='*60}")
    print(f"Testing {model_name}")
    print(f"{'='*60}")
    
    try:
        start_time = time.time()
        forecast = train_func(train_series, horizon, random_state=42)
        elapsed_time = time.time() - start_time
        
        print(f"✅ {model_name} completed successfully!")
        print(f"   Training time: {elapsed_time:.2f} seconds")
        print(f"   Forecast shape: {forecast.shape}")
        print(f"   Forecast range: [{forecast.min():.2f}, {forecast.max():.2f}]")
        print(f"   First 5 predictions: {forecast.head().values}")
        
        return forecast, elapsed_time, True
        
    except Exception as e:
        print(f"❌ {model_name} failed: {e}")
        return None, None, False

def main():
    """Main test function"""
    print("\n" + "="*60)
    print("New Deep Learning Models Test")
    print("="*60)
    
    # Generate sample data
    print("\n📊 Generating sample time series...")
    full_series = generate_sample_series(length=1000)
    train_series = full_series[:800]
    test_series = full_series[800:820]
    horizon = len(test_series)
    
    print(f"   Training data: {len(train_series)} points")
    print(f"   Test data: {len(test_series)} points")
    print(f"   Forecast horizon: {horizon}")
    
    # Test models
    models_to_test = [
        ("LSTM", train_lstm),
        ("GRU", train_gru),
        ("N-BEATS", train_nbeats),
        ("TFT", train_tft),
        ("DeepAR", train_deepar)
    ]
    
    results = {}
    
    for model_name, train_func in models_to_test:
        forecast, elapsed_time, success = test_single_model(
            model_name, train_func, train_series, horizon
        )
        if success:
            results[model_name] = {
                'forecast': forecast,
                'time': elapsed_time,
                'mae': np.mean(np.abs(forecast.values - test_series.values))
            }
    
    # Summary
    print(f"\n{'='*60}")
    print("📈 Results Summary")
    print(f"{'='*60}")
    
    if results:
        print(f"\n{'Model':<15} {'Time (s)':<12} {'MAE':<12} {'Status'}")
        print("-" * 60)
        
        for model_name in results:
            time_str = f"{results[model_name]['time']:.2f}"
            mae_str = f"{results[model_name]['mae']:.4f}"
            print(f"{model_name:<15} {time_str:<12} {mae_str:<12} ✅")
        
        # Find best model
        best_model = min(results.items(), key=lambda x: x[1]['mae'])
        print(f"\n🏆 Best model (lowest MAE): {best_model[0]}")
        
        # Plot results
        print(f"\n📊 Generating comparison plot...")
        try:
            plt.figure(figsize=(15, 8))
            
            # Plot training data (last 200 points)
            plt.plot(train_series.index[-200:], train_series.values[-200:], 
                    'k-', label='Training data', alpha=0.5)
            
            # Plot test data
            plt.plot(test_series.index, test_series.values, 
                    'ko-', label='Actual', linewidth=2, markersize=6)
            
            # Plot forecasts
            colors = ['blue', 'green', 'red', 'purple', 'orange']
            for (model_name, result), color in zip(results.items(), colors):
                plt.plot(result['forecast'].index, result['forecast'].values,
                        marker='o', label=f'{model_name} (MAE: {result["mae"]:.2f})',
                        color=color, alpha=0.7)
            
            plt.axvline(x=800, color='gray', linestyle='--', alpha=0.5, 
                       label='Train/Test split')
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.title('Deep Learning Models Comparison')
            plt.legend(loc='best')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            output_path = 'models_comparison.png'
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"   Plot saved to: {output_path}")
            
        except Exception as e:
            print(f"   Warning: Could not create plot: {e}")
    
    else:
        print("\n❌ No models completed successfully")
    
    print(f"\n{'='*60}")
    print("✅ Test completed!")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()

