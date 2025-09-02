#!/usr/bin/env python3
"""
Quick test script for forecasting models
Tests XGBoost, HoltWinters, and Prophet to ensure they work before running the full experiment.
"""

import pandas as pd
import numpy as np
import sys
import os

# Add current directory to path to import models
sys.path.append('.')

# Import the forecasting models
available_models = {}

try:
    from models.xgboost import train_xgboost
    available_models['XGBoost'] = train_xgboost
    print("✅ XGBoost model imported successfully")
except ImportError as e:
    print(f"❌ Failed to import XGBoost model: {e}")
    
try:
    from models.holt_winters import train_holt_winters
    available_models['HoltWinters'] = train_holt_winters
    print("✅ HoltWinters model imported successfully")
except ImportError as e:
    print(f"❌ Failed to import HoltWinters model: {e}")
    
try:
    from models.prophet import train_prophet
    available_models['Prophet'] = train_prophet
    print("✅ Prophet model imported successfully")
except ImportError as e:
    print(f"❌ Failed to import Prophet model: {e}")
    print(f"   💡 Try installing: pip install prophet")

def test_forecasting_model(model_func, model_name, train_data, n_points=10):
    """Test a single forecasting model"""
    print(f"\n🔮 Testing {model_name}...")
    try:
        # Call the model
        forecast = model_func(train_data, n_points)
        
        # Check the output
        if isinstance(forecast, pd.Series):
            forecast_array = forecast.values
        else:
            forecast_array = np.array(forecast)
        
        # Validate output
        if len(forecast_array) == n_points:
            print(f"✅ {model_name}: Success! Generated {len(forecast_array)} predictions")
            print(f"   📊 Sample predictions: {forecast_array[:3]}")
            print(f"   📈 Prediction range: [{forecast_array.min():.4f}, {forecast_array.max():.4f}]")
            return True
        else:
            print(f"❌ {model_name}: Wrong output length! Expected {n_points}, got {len(forecast_array)}")
            return False
            
    except Exception as e:
        print(f"❌ {model_name}: FAILED with error: {e}")
        import traceback
        print(f"   📋 Full traceback:")
        traceback.print_exc()
        return False

def main():
    print("🧪 FORECASTING MODELS TEST")
    print("=" * 50)
    
    # Test data paths
    test_data_paths = [
        "data/0_source_data/boiler_outlet_temp_univ.csv",
        "data/0_source_data/pump_sensor_28_univ.csv",
        "data/0_source_data/vibration_sensor_S1.csv"
    ]
    
    # Find available test data
    available_data = []
    for path in test_data_paths:
        if os.path.exists(path):
            available_data.append(path)
            print(f"✅ Found test data: {path}")
        else:
            print(f"⚠️ Missing test data: {path}")
    
    if not available_data:
        print("❌ No test data available! Please ensure data files exist.")
        print("Expected files:")
        for path in test_data_paths:
            print(f"  - {path}")
        return
    
    # Load first available dataset
    data_path = available_data[0]
    print(f"\n📂 Loading dataset: {data_path}")
    
    try:
        df = pd.read_csv(data_path, index_col=0)
        series_data = df.iloc[:, 0]  # First column
        print(f"✅ Data loaded: {len(series_data)} points")
        print(f"   📊 Data range: [{series_data.min():.4f}, {series_data.max():.4f}]")
        print(f"   📈 Sample values: {series_data.head(3).values}")
        
        # Reset index to numeric (as done in the experiment)
        series_data = series_data.reset_index(drop=True)
        
        # Split into train/test (same as experiment)
        test_size = 10
        train_data = series_data[:-test_size]
        test_data = series_data[-test_size:]
        
        print(f"📊 Train set: {len(train_data)} points")
        print(f"📊 Test set: {len(test_data)} points")
        
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return
    
    # Test all forecasting models
    print(f"\n🎯 Testing forecasting models (predicting {test_size} points)...")
    if not available_models:
        print("❌ No models available to test!")
        return
    
    models_to_test = [(func, name) for name, func in available_models.items()]
    print(f"📋 Will test {len(models_to_test)} available models: {list(available_models.keys())}")
    
    results = {}
    for model_func, model_name in models_to_test:
        success = test_forecasting_model(model_func, model_name, train_data, test_size)
        results[model_name] = success
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 TEST RESULTS SUMMARY")
    print("=" * 50)
    
    successful_models = [name for name, success in results.items() if success]
    failed_models = [name for name, success in results.items() if not success]
    
    if successful_models:
        print(f"✅ WORKING MODELS ({len(successful_models)}/{len(results)}):")
        for model in successful_models:
            print(f"   ✓ {model}")
    
    if failed_models:
        print(f"\n❌ FAILED MODELS ({len(failed_models)}/{len(results)}):")
        for model in failed_models:
            print(f"   ✗ {model}")
        print(f"\n⚠️ WARNING: These models will fail in the full experiment!")
        print(f"   Please fix the issues above before running the full experiment.")
    else:
        print(f"\n🎉 ALL MODELS WORKING!")
        print(f"   Safe to run the full experiment with all 3 forecasting models.")
    
    print(f"\n💡 NEXT STEPS:")
    if failed_models:
        print(f"   1. Fix the failed models above")
        print(f"   2. Re-run this test script")
        print(f"   3. Once all pass, run the full experiment")
    else:
        print(f"   1. Run: python run_improved_experiment.py --quick")
        print(f"   2. All forecasting models should work properly!")

if __name__ == "__main__":
    main()
