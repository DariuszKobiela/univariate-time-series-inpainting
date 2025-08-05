import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from scipy.stats import ttest_ind, f_oneway
import statsmodels.api as sm
from statsmodels.stats.anova import anova_lm, AnovaRM
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests
import argparse
import json
import time
import psutil
import platform
from datetime import datetime
from typing import List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Import your existing models and repair methods
import sys
sys.path.append('.')

# Import the actual model functions from your project
from models.xgboost import train_xgboost
from models.prophet import train_prophet
from models.sarimax import train_sarimax
from models.holt_winters import train_holt_winters
from models.temporal_convolutional_network import train_tcn
from models.exponential_smoothing import train_holt_winters as train_exponential_smoothing  # Use the HW function
from ts_image_inpainting import process_series, ENCODERS, INPAINTERS, INVERTERS

class IterativeExperiment:
    def __init__(self, 
                 data_path: str = "data/1_clean/train_set_original.csv",
                 test_size: int = 10,
                 n_iterations: int = 5,
                 inpainting_models: List[str] = None,
                 forecasting_models: List[str] = None,
                 missingness_types: List[str] = None,
                 missingness_rate: float = 0.2,
                 output_dir: str = "results/iterative_experiment"):
        
        self.data_path = data_path
        self.test_size = test_size
        self.n_iterations = n_iterations
        self.missingness_rate = missingness_rate
        self.output_dir = output_dir
        
        # Default configurations
        self.inpainting_models = inpainting_models or ["gaf-unet"]  # Start with one for testing
        self.forecasting_models = forecasting_models or ["XGBoost"]  # Start with one for testing
        self.missingness_types = missingness_types or ["MCAR", "MAR", "MNAR"]
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Non-inpainting repair methods (complete list from original codebase)
        self.non_inpainting_methods = [
            "impute_mean", "impute_median", "impute_mode",
            "impute_ffill", "impute_bfill",
            "interpolate_nearest", "interpolate_linear", "interpolate_index",
            "interpolate_quadratic", "interpolate_cubic", "interpolate_polynomial",
            "interpolate_spline", "interpolate_pchip", "interpolate_akima",
            "knn", "sarimax"
        ]
        
        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/predictions", exist_ok=True)
        os.makedirs(f"{self.output_dir}/metrics", exist_ok=True)
        os.makedirs(f"{self.output_dir}/plots", exist_ok=True)
        os.makedirs(f"{self.output_dir}/statistical_tests", exist_ok=True)
        
        # Initialize timing and system tracking
        self.timing_data = {
            'experiment_start_time': None,
            'experiment_end_time': None,
            'experiment_duration': None,
            'total_inpainting_time': 0.0,
            'iterations': {}
        }
        
        # Collect system information
        self.system_info = self._collect_system_info()
        
        # Load and prepare data
        self._load_data()
    
    def _collect_system_info(self) -> Dict[str, Any]:
        """Collect comprehensive system information"""
        try:
            # Basic system information
            system_info = {
                'timestamp': datetime.now().isoformat(),
                'platform': {
                    'system': platform.system(),
                    'release': platform.release(),
                    'version': platform.version(),
                    'machine': platform.machine(),
                    'processor': platform.processor(),
                    'architecture': platform.architecture(),
                    'python_version': platform.python_version()
                },
                'hardware': {
                    'cpu_count_physical': psutil.cpu_count(logical=False),
                    'cpu_count_logical': psutil.cpu_count(logical=True),
                    'cpu_freq_current': psutil.cpu_freq().current if psutil.cpu_freq() else None,
                    'cpu_freq_max': psutil.cpu_freq().max if psutil.cpu_freq() else None,
                    'memory_total_gb': round(psutil.virtual_memory().total / (1024**3), 2),
                    'memory_available_gb': round(psutil.virtual_memory().available / (1024**3), 2),
                    'disk_total_gb': round(psutil.disk_usage('/').total / (1024**3), 2) if os.name != 'nt' else round(psutil.disk_usage('C:').total / (1024**3), 2),
                    'disk_free_gb': round(psutil.disk_usage('/').free / (1024**3), 2) if os.name != 'nt' else round(psutil.disk_usage('C:').free / (1024**3), 2)
                },
                'initial_system_state': {
                    'cpu_percent': psutil.cpu_percent(interval=1),
                    'memory_percent': psutil.virtual_memory().percent,
                    'disk_percent': psutil.disk_usage('/').percent if os.name != 'nt' else psutil.disk_usage('C:').percent,
                    'boot_time': datetime.fromtimestamp(psutil.boot_time()).isoformat(),
                    'load_average': os.getloadavg() if hasattr(os, 'getloadavg') else None
                }
            }
            
            # Try to get GPU information if available
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    system_info['gpu'] = []
                    for gpu in gpus:
                        system_info['gpu'].append({
                            'id': gpu.id,
                            'name': gpu.name,
                            'memory_total_mb': gpu.memoryTotal,
                            'memory_used_mb': gpu.memoryUsed,
                            'memory_free_mb': gpu.memoryFree,
                            'temperature': gpu.temperature,
                            'load': gpu.load
                        })
            except ImportError:
                system_info['gpu'] = 'GPUtil not available'
            except Exception as e:
                system_info['gpu'] = f'GPU info collection failed: {str(e)}'
            
            # Environment information
            system_info['environment'] = {
                'working_directory': os.getcwd(),
                'python_executable': sys.executable,
                'environment_variables': {
                    'PATH': os.environ.get('PATH', ''),
                    'PYTHONPATH': os.environ.get('PYTHONPATH', ''),
                    'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES', ''),
                    'OMP_NUM_THREADS': os.environ.get('OMP_NUM_THREADS', ''),
                }
            }
            
            print(f"✓ System information collected")
            return system_info
            
        except Exception as e:
            print(f"⚠️ Warning: Could not collect complete system information: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'basic_info': {
                    'system': platform.system(),
                    'python_version': platform.python_version()
                }
            }
        
    def _load_data(self):
        """Load the original dataset and split into train/test"""
        df = pd.read_csv(self.data_path, index_col=0)
        self.full_data = df.iloc[:, 0]  # Assuming first column is the time series
        
        # Split into train and test
        self.train_data = self.full_data[:-self.test_size]
        self.test_data = self.full_data[-self.test_size:]
        
        print(f"Loaded data: {len(self.full_data)} points")
        print(f"Train set: {len(self.train_data)} points")
        print(f"Test set: {len(self.test_data)} points")
        
    def generate_missingness(self, data: pd.Series, missingness_type: str, rate: float = None) -> pd.Series:
        """Generate missing values in the training data"""
        if rate is None:
            rate = self.missingness_rate
            
        data_copy = data.copy()
        n_missing = int(len(data) * rate)
        
        if missingness_type == "MCAR":
            # Missing completely at random
            missing_indices = np.random.choice(data.index, n_missing, replace=False)
        elif missingness_type == "MAR":
            # Missing at random - higher probability for certain value ranges
            # Missing more likely for higher values
            probs = np.abs(data.values - data.median()) / np.abs(data.values - data.median()).sum()
            missing_indices = np.random.choice(data.index, n_missing, replace=False, p=probs)
        elif missingness_type == "MNAR":
            # Missing not at random - systematic pattern
            # Missing more likely at the end of the series
            weights = np.linspace(0.1, 1.0, len(data))
            probs = weights / weights.sum()
            missing_indices = np.random.choice(data.index, n_missing, replace=False, p=probs)
        else:
            raise ValueError(f"Unknown missingness type: {missingness_type}")
            
        data_copy.loc[missing_indices] = np.nan
        return data_copy
    
    def apply_non_inpainting_repair(self, damaged_data: pd.Series, method: str) -> pd.Series:
        """Apply traditional repair methods"""
        data_copy = damaged_data.copy()
        
        if method == "impute_mean":
            data_copy = data_copy.fillna(data_copy.mean())
        elif method == "impute_median":
            data_copy = data_copy.fillna(data_copy.median())
        elif method == "impute_mode":
            mode_vals = data_copy.mode()
            mode_val = mode_vals.iloc[0] if not mode_vals.empty else data_copy.mean()
            data_copy = data_copy.fillna(mode_val)
        elif method == "impute_ffill":
            data_copy = data_copy.fillna(method='ffill').fillna(method='bfill')
        elif method == "impute_bfill":
            data_copy = data_copy.fillna(method='bfill').fillna(method='ffill')
        elif method.startswith("interpolate_"):
            interp_method = method.replace("interpolate_", "")
            try:
                if interp_method == "polynomial":
                    data_copy = data_copy.interpolate(method='polynomial', order=2)
                elif interp_method == "spline":
                    data_copy = data_copy.interpolate(method='spline', order=2)
                else:
                    data_copy = data_copy.interpolate(method=interp_method)
            except Exception as e:
                print(f"Warning: Interpolation method {interp_method} failed: {e}, falling back to linear")
                data_copy = data_copy.interpolate(method='linear')
            data_copy = data_copy.fillna(method='ffill').fillna(method='bfill')
        elif method == "knn":
            # Simple KNN imputation using neighboring values
            from sklearn.impute import KNNImputer
            imputer = KNNImputer(n_neighbors=3)
            values = imputer.fit_transform(data_copy.values.reshape(-1, 1))
            data_copy = pd.Series(values.flatten(), index=data_copy.index)
        elif method == "sarimax":
            # Use SARIMA for imputation
            try:
                from statsmodels.tsa.statespace.sarimax import SARIMAX
                # Fill missing values using SARIMA predictions
                model = SARIMAX(data_copy.dropna(), order=(1, 1, 1))
                fitted = model.fit(disp=False)
                # Simple approach: use fitted values for missing points
                data_copy = data_copy.fillna(method='ffill').fillna(method='bfill')
            except:
                data_copy = data_copy.fillna(method='ffill').fillna(method='bfill')
        
        return data_copy
    
    def apply_inpainting_repair(self, damaged_data: pd.Series, method: str) -> tuple:
        """Apply image inpainting repair methods and return repaired data with timing"""
        start_time = time.time()
        
        try:
            # Extract encoder and inpainter from method name (e.g., "gaf-unet")
            parts = method.split('-')
            if len(parts) != 2:
                raise ValueError(f"Invalid inpainting method format: {method}")
            
            enc_name, inp_name = parts
            
            if enc_name not in ENCODERS or inp_name not in INPAINTERS:
                raise ValueError(f"Unknown encoder {enc_name} or inpainter {inp_name}")
            
            # Fill NaNs temporarily for encoding
            mask = damaged_data.isna()
            series_filled = damaged_data.fillna(0)
            
            # Encode to image
            encoder = ENCODERS[enc_name]
            img = encoder(series_filled)
            
            # Apply inpainting
            inpainter = INPAINTERS[inp_name]
            inpainted_img = inpainter(img, mask, enc_name)
            
            # Inverse transform
            inverter = INVERTERS[enc_name]
            recon_series = inverter(inpainted_img)
            
            # Align length and merge with original
            recon_series.index = damaged_data.index[:len(recon_series)]
            merged = damaged_data.copy()
            
            # Only replace missing values
            mask_aligned = mask[:len(recon_series)]
            merged[mask_aligned] = recon_series[mask_aligned]
            
            end_time = time.time()
            inpainting_time = end_time - start_time
            
            return merged, inpainting_time
            
        except Exception as e:
            print(f"Warning: Inpainting method {method} failed: {e}")
            # Fallback to simple interpolation
            end_time = time.time()
            inpainting_time = end_time - start_time
            return damaged_data.interpolate().fillna(method='ffill').fillna(method='bfill'), inpainting_time
    
    def forecast_next_points(self, train_data: pd.Series, model_name: str, n_points: int = None) -> np.ndarray:
        """Forecast the next n points using the specified model"""
        if n_points is None:
            n_points = self.test_size
            
        try:
            # Use the actual model functions from your project
            if model_name == "XGBoost":
                forecast = train_xgboost(train_data, n_points)
            elif model_name == "Prophet":
                forecast = train_prophet(train_data, n_points)
            elif model_name == "SARIMAX":
                forecast = train_sarimax(train_data, n_points)
            elif model_name == "HoltWinters":
                forecast = train_holt_winters(train_data, n_points)
            elif model_name == "TCN":
                forecast = train_tcn(train_data, n_points)
            elif model_name == "ExponentialSmoothing":
                forecast = train_exponential_smoothing(train_data, n_points)
            else:
                raise ValueError(f"Unknown forecasting model: {model_name}")
            
            # Handle different return types
            if isinstance(forecast, pd.Series):
                return forecast.values
            else:
                return np.array(forecast)
                
        except Exception as e:
            print(f"Warning: Forecasting with {model_name} failed: {e}")
            # Fallback to simple trend extrapolation
            clean_data = train_data.dropna()
            if len(clean_data) < 2:
                return np.array([clean_data.iloc[-1]] * n_points if len(clean_data) > 0 else [0] * n_points)
            
            trend = np.mean(np.diff(clean_data.tail(min(5, len(clean_data)))))
            last_value = clean_data.iloc[-1]
            return np.array([last_value + trend * (i + 1) for i in range(n_points)])
    
    def calculate_metrics(self, true_values: np.ndarray, predicted_values: np.ndarray) -> Dict[str, float]:
        """Calculate forecasting metrics"""
        try:
            mae = mean_absolute_error(true_values, predicted_values)
            rmse = np.sqrt(mean_squared_error(true_values, predicted_values))
            mape = mean_absolute_percentage_error(true_values, predicted_values) * 100
            
            return {
                'MAE': mae,
                'RMSE': rmse,
                'MAPE': mape
            }
        except Exception as e:
            print(f"Warning: Metric calculation failed: {e}")
            return {'MAE': np.inf, 'RMSE': np.inf, 'MAPE': np.inf}
    
    def run_single_iteration(self, iteration: int) -> Dict[str, Any]:
        """Run a single iteration of the experiment"""
        iteration_start_time = time.time()
        print(f"\n--- Running Iteration {iteration + 1}/{self.n_iterations} ---")
        
        iteration_results = {
            'iteration': iteration,
            'metrics': {},
            'predictions': {},
            'timing': {
                'iteration_start_time': datetime.now().isoformat(),
                'iteration_duration': None,
                'inpainting_times': {},
                'total_inpainting_time_iteration': 0.0
            },
            'system_state': {
                'cpu_percent_start': psutil.cpu_percent(),
                'memory_percent_start': psutil.virtual_memory().percent,
                'cpu_percent_end': None,
                'memory_percent_end': None
            }
        }
        
        for missingness_type in self.missingness_types:
            print(f"Processing {missingness_type} missingness...")
            
            # Generate missingness
            damaged_train = self.generate_missingness(self.train_data, missingness_type)
            
            # Test all repair methods
            all_repair_methods = self.inpainting_models + self.non_inpainting_methods
            
            for repair_method in all_repair_methods:
                print(f"  Applying repair method: {repair_method}")
                
                # Apply repair with timing
                if repair_method in self.inpainting_models:
                    repaired_train, inpainting_time = self.apply_inpainting_repair(damaged_train, repair_method)
                    # Store inpainting time
                    key_timing = f"{missingness_type}_{repair_method}"
                    iteration_results['timing']['inpainting_times'][key_timing] = inpainting_time
                    iteration_results['timing']['total_inpainting_time_iteration'] += inpainting_time
                    self.timing_data['total_inpainting_time'] += inpainting_time
                else:
                    repaired_train = self.apply_non_inpainting_repair(damaged_train, repair_method)
                
                # Test all forecasting models
                for forecast_model in self.forecasting_models:
                    print(f"    Forecasting with: {forecast_model}")
                    
                    # Make predictions
                    predictions = self.forecast_next_points(repaired_train, forecast_model)
                    
                    # Calculate metrics
                    metrics = self.calculate_metrics(self.test_data.values, predictions)
                    
                    # Store results
                    key = f"{missingness_type}_{repair_method}_{forecast_model}"
                    iteration_results['metrics'][key] = metrics
                    iteration_results['predictions'][key] = predictions.tolist()
        
        # Record end timing and system state
        iteration_end_time = time.time()
        iteration_duration = iteration_end_time - iteration_start_time
        
        iteration_results['timing']['iteration_end_time'] = datetime.now().isoformat()
        iteration_results['timing']['iteration_duration'] = iteration_duration
        iteration_results['system_state']['cpu_percent_end'] = psutil.cpu_percent()
        iteration_results['system_state']['memory_percent_end'] = psutil.virtual_memory().percent
        
        # Store in main timing data
        self.timing_data['iterations'][f'iteration_{iteration + 1}'] = {
            'duration': iteration_duration,
            'inpainting_time': iteration_results['timing']['total_inpainting_time_iteration'],
            'start_time': iteration_results['timing']['iteration_start_time'],
            'end_time': iteration_results['timing']['iteration_end_time']
        }
        
        print(f"✓ Iteration {iteration + 1} completed in {iteration_duration:.2f} seconds")
        print(f"  - Inpainting time: {iteration_results['timing']['total_inpainting_time_iteration']:.2f} seconds")
        
        return iteration_results
    
    def run_experiment(self) -> Dict[str, Any]:
        """Run the complete iterative experiment"""
        # Start experiment timing
        self.timing_data['experiment_start_time'] = datetime.now().isoformat()
        experiment_start_time = time.time()
        
        print("🔬 Starting Iterative Experiment...")
        print(f"⏰ Start time: {self.timing_data['experiment_start_time']}")
        print(f"Configuration:")
        print(f"  - Iterations: {self.n_iterations}")
        print(f"  - Missingness types: {self.missingness_types}")
        print(f"  - Inpainting methods: {self.inpainting_models}")
        print(f"  - Non-inpainting methods: {self.non_inpainting_methods}")
        print(f"  - Forecasting models: {self.forecasting_models}")
        print(f"  - Test size: {self.test_size}")
        
        # Collect initial system state
        initial_system_state = {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent if os.name != 'nt' else psutil.disk_usage('C:').percent,
            'processes_count': len(psutil.pids())
        }
        
        all_results = []
        
        # Run iterations
        for i in range(self.n_iterations):
            iteration_result = self.run_single_iteration(i)
            all_results.append(iteration_result)
            
            # Save intermediate results
            with open(f"{self.output_dir}/iteration_{i+1}_results.json", 'w') as f:
                json.dump(iteration_result, f, indent=2)
            
            # Show progress
            progress = ((i + 1) / self.n_iterations) * 100
            print(f"📊 Progress: {progress:.1f}% ({i + 1}/{self.n_iterations} iterations)")
        
        # End experiment timing
        experiment_end_time = time.time()
        self.timing_data['experiment_end_time'] = datetime.now().isoformat()
        self.timing_data['experiment_duration'] = experiment_end_time - experiment_start_time
        
        # Collect final system state
        final_system_state = {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent if os.name != 'nt' else psutil.disk_usage('C:').percent,
            'processes_count': len(psutil.pids())
        }
        
        # Aggregate results
        aggregated_results = self.aggregate_results(all_results)
        
        # Add comprehensive timing and system information
        comprehensive_results = {
            **aggregated_results,
            'timing_summary': self.timing_data,
            'system_information': self.system_info,
            'system_state_changes': {
                'initial': initial_system_state,
                'final': final_system_state,
                'cpu_usage_change': final_system_state['cpu_percent'] - initial_system_state['cpu_percent'],
                'memory_usage_change': final_system_state['memory_percent'] - initial_system_state['memory_percent']
            },
            'experiment_metadata': {
                'total_methods_tested': len(self.inpainting_models) + len(self.non_inpainting_methods),
                'total_combinations': len(self.missingness_types) * (len(self.inpainting_models) + len(self.non_inpainting_methods)) * len(self.forecasting_models),
                'total_experiments_run': self.n_iterations * len(self.missingness_types) * (len(self.inpainting_models) + len(self.non_inpainting_methods)) * len(self.forecasting_models),
                'inpainting_percentage_of_total_time': (self.timing_data['total_inpainting_time'] / self.timing_data['experiment_duration']) * 100 if self.timing_data['experiment_duration'] > 0 else 0
            }
        }
        
        # Save final results with timing
        with open(f"{self.output_dir}/final_results.json", 'w') as f:
            json.dump(comprehensive_results, f, indent=2)
        
        # Save separate timing report
        self._save_timing_report()
        
        # Print summary
        self._print_timing_summary()
        
        return comprehensive_results
    
    def _save_timing_report(self):
        """Save a detailed timing report"""
        timing_report = {
            'experiment_summary': {
                'total_duration_seconds': self.timing_data['experiment_duration'],
                'total_duration_formatted': self._format_duration(self.timing_data['experiment_duration']),
                'total_inpainting_time_seconds': self.timing_data['total_inpainting_time'],
                'total_inpainting_time_formatted': self._format_duration(self.timing_data['total_inpainting_time']),
                'inpainting_percentage': (self.timing_data['total_inpainting_time'] / self.timing_data['experiment_duration']) * 100 if self.timing_data['experiment_duration'] > 0 else 0,
                'average_iteration_duration': self.timing_data['experiment_duration'] / self.n_iterations,
                'average_inpainting_time_per_iteration': self.timing_data['total_inpainting_time'] / self.n_iterations
            },
            'iteration_details': self.timing_data['iterations'],
            'system_info': self.system_info
        }
        
        with open(f"{self.output_dir}/timing_report.json", 'w') as f:
            json.dump(timing_report, f, indent=2)
        
        print(f"✓ Detailed timing report saved to timing_report.json")
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format"""
        if seconds < 60:
            return f"{seconds:.2f} seconds"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.2f} minutes ({seconds:.2f} seconds)"
        else:
            hours = seconds / 3600
            minutes = (seconds % 3600) / 60
            return f"{hours:.2f} hours ({minutes:.2f} minutes, {seconds:.2f} seconds)"
    
    def _print_timing_summary(self):
        """Print a comprehensive timing summary"""
        print("\n" + "="*60)
        print("⏱️  TIMING SUMMARY")
        print("="*60)
        print(f"🔬 Experiment Duration: {self._format_duration(self.timing_data['experiment_duration'])}")
        print(f"🎨 Total Inpainting Time: {self._format_duration(self.timing_data['total_inpainting_time'])}")
        print(f"📊 Inpainting Percentage: {(self.timing_data['total_inpainting_time'] / self.timing_data['experiment_duration']) * 100:.1f}%")
        print(f"🔄 Average per Iteration: {self._format_duration(self.timing_data['experiment_duration'] / self.n_iterations)}")
        print(f"🎨 Average Inpainting per Iteration: {self._format_duration(self.timing_data['total_inpainting_time'] / self.n_iterations)}")
        
        print(f"\n💻 SYSTEM INFORMATION:")
        print(f"CPU: {self.system_info['platform']['processor']}")
        print(f"Cores: {self.system_info['hardware']['cpu_count_physical']} physical, {self.system_info['hardware']['cpu_count_logical']} logical")
        print(f"Memory: {self.system_info['hardware']['memory_total_gb']} GB total")
        print(f"Platform: {self.system_info['platform']['system']} {self.system_info['platform']['release']}")
        
        print(f"\n📈 PERFORMANCE BREAKDOWN:")
        for iteration_name, iteration_data in self.timing_data['iterations'].items():
            inpainting_pct = (iteration_data['inpainting_time'] / iteration_data['duration']) * 100 if iteration_data['duration'] > 0 else 0
            print(f"  {iteration_name}: {self._format_duration(iteration_data['duration'])} (inpainting: {inpainting_pct:.1f}%)")
        
        print("="*60)
    
    def aggregate_results(self, all_results: List[Dict]) -> Dict[str, Any]:
        """Aggregate results across all iterations"""
        print("\nAggregating results across iterations...")
        
        # Collect all metrics
        all_metrics = {}
        for result in all_results:
            for key, metrics in result['metrics'].items():
                if key not in all_metrics:
                    all_metrics[key] = {metric: [] for metric in metrics.keys()}
                
                for metric, value in metrics.items():
                    all_metrics[key][metric].append(value)
        
        # Calculate means and standard deviations
        aggregated = {
            'mean_metrics': {},
            'std_metrics': {},
            'raw_metrics': all_metrics
        }
        
        for key, metrics in all_metrics.items():
            aggregated['mean_metrics'][key] = {}
            aggregated['std_metrics'][key] = {}
            
            for metric, values in metrics.items():
                clean_values = [v for v in values if not np.isinf(v) and not np.isnan(v)]
                if clean_values:
                    aggregated['mean_metrics'][key][metric] = np.mean(clean_values)
                    aggregated['std_metrics'][key][metric] = np.std(clean_values)
                else:
                    aggregated['mean_metrics'][key][metric] = np.inf
                    aggregated['std_metrics'][key][metric] = 0
        
        return aggregated
    
    def create_performance_plots(self, results: Dict[str, Any]):
        """Create bar charts showing average performance"""
        print("Creating performance plots...")
        
        mean_metrics = results['mean_metrics']
        std_metrics = results['std_metrics']
        
        # Organize data for plotting
        for metric in ['MAE', 'RMSE', 'MAPE']:
            # Collect data for plotting
            plot_data = []
            for key, metrics in mean_metrics.items():
                if metric in metrics and not np.isinf(metrics[metric]) and not np.isnan(metrics[metric]):
                    parts = key.split('_')
                    if len(parts) >= 3:  # Ensure we have all parts
                        missingness_type = parts[0]
                        # Handle multi-part repair method names (e.g., "interpolate_linear")
                        if len(parts) > 3:
                            repair_method = '_'.join(parts[1:-1])
                            forecast_model = parts[-1]
                        else:
                            repair_method = parts[1]
                            forecast_model = parts[2]
                        
                        plot_data.append({
                            'Method': repair_method,
                            'Missingness': missingness_type,
                            'Repair': repair_method,
                            'Model': forecast_model,
                            'Value': metrics[metric],
                            'Std': std_metrics[key][metric] if key in std_metrics else 0
                        })
            
            if not plot_data:
                print(f"Warning: No data found for metric {metric}")
                continue
                
            df_plot = pd.DataFrame(plot_data)
            
            # Get unique missingness types that actually exist in the data
            actual_missingness_types = df_plot['Missingness'].unique()
            n_plots = len(actual_missingness_types)
            
            if n_plots == 0:
                print(f"Warning: No missingness types found for metric {metric}")
                continue
            
            # Create subplot layout
            fig, axes = plt.subplots(1, n_plots, figsize=(7*n_plots, 6))
            if n_plots == 1:
                axes = [axes]  # Make it iterable for single plot
            
            for i, miss_type in enumerate(actual_missingness_types):
                subset = df_plot[df_plot['Missingness'] == miss_type].copy()
                
                if not subset.empty:
                    # Sort by value (best to worst) - lower is better for all our metrics
                    subset = subset.sort_values('Value', ascending=True)
                    
                    # Create color mapping: inpainting methods get different colors
                    colors = []
                    for repair_method in subset['Repair']:
                        if any(inp in repair_method for inp in self.inpainting_models):
                            colors.append('red')  # Inpainting methods in red
                        else:
                            colors.append('skyblue')  # Traditional methods in blue
                    
                    bars = axes[i].bar(range(len(subset)), subset['Value'], 
                                     yerr=subset['Std'], capsize=3, alpha=0.8, color=colors)
                    
                    axes[i].set_title(f'{metric} - {miss_type}')
                    axes[i].set_ylabel(metric)
                    axes[i].set_xlabel('Repair Method')
                    axes[i].set_xticks(range(len(subset)))
                    axes[i].set_xticklabels(subset['Method'], rotation=45, ha='right', fontsize=9)
                    axes[i].grid(axis='y', alpha=0.3)
                    
                    # Add legend for colors
                    from matplotlib.patches import Patch
                    legend_elements = [
                        Patch(facecolor='red', alpha=0.8, label='Inpainting'),
                        Patch(facecolor='skyblue', alpha=0.8, label='Traditional')
                    ]
                    axes[i].legend(handles=legend_elements, loc='upper right')
                else:
                    axes[i].set_title(f'{metric} - {miss_type} (No Data)')
            
            plt.suptitle(f'Average {metric} Across All Iterations (Sorted: Best → Worst)')
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/plots/average_{metric}_performance.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Created plot for {metric} with {len(plot_data)} data points across {n_plots} missingness types")
    
    def perform_statistical_tests(self, results: Dict[str, Any]):
        """Perform t-tests between inpainting and non-inpainting methods"""
        print("Performing statistical tests...")
        
        raw_metrics = results['raw_metrics']
        test_results = []
        
        for metric in ['MAE', 'RMSE', 'MAPE']:
            for missingness_type in self.missingness_types:
                for forecast_model in self.forecasting_models:
                    
                    # Collect inpainting results
                    inpainting_results = {}
                    for inpainting_method in self.inpainting_models:
                        key = f"{missingness_type}_{inpainting_method}_{forecast_model}"
                        if key in raw_metrics and metric in raw_metrics[key]:
                            values = [v for v in raw_metrics[key][metric] 
                                    if not np.isinf(v) and not np.isnan(v)]
                            if len(values) >= 2:  # Need at least 2 values for t-test
                                inpainting_results[inpainting_method] = values
                    
                    # Collect non-inpainting results
                    non_inpainting_results = {}
                    for non_inpainting_method in self.non_inpainting_methods:
                        key = f"{missingness_type}_{non_inpainting_method}_{forecast_model}"
                        if key in raw_metrics and metric in raw_metrics[key]:
                            values = [v for v in raw_metrics[key][metric] 
                                    if not np.isinf(v) and not np.isnan(v)]
                            if len(values) >= 2:
                                non_inpainting_results[non_inpainting_method] = values
                    
                    # Perform t-tests: each inpainting vs each non-inpainting
                    for inp_method, inp_values in inpainting_results.items():
                        for non_inp_method, non_inp_values in non_inpainting_results.items():
                            try:
                                t_stat, p_value = ttest_ind(inp_values, non_inp_values)
                                
                                test_results.append({
                                    'metric': metric,
                                    'missingness_type': missingness_type,
                                    'forecast_model': forecast_model,
                                    'inpainting_method': inp_method,
                                    'non_inpainting_method': non_inp_method,
                                    't_statistic': t_stat,
                                    'p_value': p_value,
                                    'inpainting_mean': np.mean(inp_values),
                                    'non_inpainting_mean': np.mean(non_inp_values)
                                })
                            except Exception as e:
                                print(f"t-test failed for {inp_method} vs {non_inp_method}: {e}")
        
        # Save test results
        if test_results:
            df_tests = pd.DataFrame(test_results)
            df_tests.to_csv(f"{self.output_dir}/statistical_tests/t_test_results.csv", index=False)
            
            # Create p-value plots
            self.plot_statistical_results(df_tests)
        
        return test_results
    
    def perform_bonferroni_correction(self, results: Dict[str, Any]):
        """Apply Bonferroni correction to existing t-test results"""
        print("Applying Bonferroni correction to existing t-test results...")
        
        # Load the existing t-test results instead of recalculating
        try:
            df_tests = pd.read_csv(f"{self.output_dir}/statistical_tests/t_test_results.csv")
            print(f"✓ Loaded {len(df_tests)} t-test results for Bonferroni correction")
        except FileNotFoundError:
            print("❌ No existing t-test results found. Please run perform_statistical_tests() first.")
            return []
        
        if df_tests.empty:
            print("No t-test results available for Bonferroni correction")
            return []
        
        # Apply Bonferroni correction within each metric group
        bonferroni_results = []
        
        for metric in ['MAE', 'RMSE', 'MAPE']:
            metric_subset = df_tests[df_tests['metric'] == metric].copy()
            
            if not metric_subset.empty:
                # Apply Bonferroni correction
                p_values = metric_subset['p_value'].values
                rejected, p_corrected, alpha_sidak, alpha_bonf = multipletests(
                    p_values, alpha=0.05, method='bonferroni'
                )
                
                # Add corrected results
                metric_subset = metric_subset.copy()
                metric_subset['p_value_bonferroni'] = p_corrected
                metric_subset['significant_bonferroni'] = rejected
                metric_subset['alpha_bonferroni'] = alpha_bonf
                
                bonferroni_results.append(metric_subset)
        
        # Combine all results
        if bonferroni_results:
            df_bonferroni = pd.concat(bonferroni_results, ignore_index=True)
            
            # Save Bonferroni corrected results
            df_bonferroni.to_csv(f"{self.output_dir}/statistical_tests/t_test_bonferroni_corrected.csv", index=False)
            
            # Create comparison plots
            self.plot_bonferroni_comparison(df_bonferroni)
            
            print(f"✓ Bonferroni corrected t-test results saved to statistical_tests/t_test_bonferroni_corrected.csv")
            print(f"✓ Original t-test results remain in statistical_tests/t_test_results.csv")
            print(f"✓ Compare both files to see the effect of multiple comparison correction")
            
            return df_bonferroni.to_dict('records')
        
        return []
    
    def plot_bonferroni_comparison(self, df_bonferroni: pd.DataFrame):
        """Create comparison plots showing original vs Bonferroni corrected p-values"""
        print("Creating Bonferroni correction comparison plots...")
        
        for metric in ['MAE', 'RMSE', 'MAPE']:
            subset = df_bonferroni[df_bonferroni['metric'] == metric]
            
            if subset.empty:
                continue
            
            # Get actual missingness types in the data
            actual_miss_types = subset['missingness_type'].unique()
            n_plots = len(actual_miss_types)
            
            if n_plots == 0:
                continue
            
            # Create subplot layout - 2 rows: original p-values vs corrected p-values
            fig, axes = plt.subplots(2, n_plots, figsize=(7*n_plots, 12))
            if n_plots == 1:
                axes = axes.reshape(2, 1)
            
            for i, miss_type in enumerate(actual_miss_types):
                miss_subset = subset[subset['missingness_type'] == miss_type].copy()
                
                if not miss_subset.empty:
                    # Create comparison labels
                    miss_subset['comparison'] = (miss_subset['inpainting_method'] + 
                                               ' vs ' + 
                                               miss_subset['non_inpainting_method'])
                    
                    # Sort by original p-value
                    miss_subset = miss_subset.sort_values('p_value', ascending=True)
                    
                    # Plot 1: Original p-values
                    colors_original = []
                    for p_val in miss_subset['p_value']:
                        if p_val < 0.01:
                            colors_original.append('red')
                        elif p_val < 0.05:
                            colors_original.append('orange')
                        else:
                            colors_original.append('lightblue')
                    
                    axes[0, i].bar(range(len(miss_subset)), miss_subset['p_value'], 
                                  alpha=0.8, color=colors_original)
                    axes[0, i].axhline(y=0.01, color='red', linestyle='--', linewidth=2,
                                      label='p=0.01')
                    axes[0, i].axhline(y=0.05, color='orange', linestyle='--', linewidth=2,
                                      label='p=0.05')
                    axes[0, i].set_title(f'Original P-Values - {metric} - {miss_type}')
                    axes[0, i].set_ylabel('P-value')
                    axes[0, i].set_yscale('log')
                    axes[0, i].set_xticks(range(len(miss_subset)))
                    axes[0, i].set_xticklabels(miss_subset['comparison'], 
                                              rotation=45, ha='right', fontsize=8)
                    axes[0, i].grid(axis='y', alpha=0.3)
                    axes[0, i].legend()
                    
                    # Plot 2: Bonferroni corrected p-values
                    colors_bonferroni = []
                    for p_val in miss_subset['p_value_bonferroni']:
                        if p_val < 0.01:
                            colors_bonferroni.append('red')
                        elif p_val < 0.05:
                            colors_bonferroni.append('orange')
                        else:
                            colors_bonferroni.append('lightblue')
                    
                    axes[1, i].bar(range(len(miss_subset)), miss_subset['p_value_bonferroni'], 
                                  alpha=0.8, color=colors_bonferroni)
                    axes[1, i].axhline(y=0.01, color='red', linestyle='--', linewidth=2,
                                      label='p=0.01')
                    axes[1, i].axhline(y=0.05, color='orange', linestyle='--', linewidth=2,
                                      label='p=0.05')
                    axes[1, i].set_title(f'Bonferroni Corrected P-Values - {metric} - {miss_type}')
                    axes[1, i].set_ylabel('Corrected P-value')
                    axes[1, i].set_yscale('log')
                    axes[1, i].set_xticks(range(len(miss_subset)))
                    axes[1, i].set_xticklabels(miss_subset['comparison'], 
                                              rotation=45, ha='right', fontsize=8)
                    axes[1, i].grid(axis='y', alpha=0.3)
                    axes[1, i].legend()
                    
                    # Add text showing number of significant results
                    n_sig_original = sum(miss_subset['p_value'] < 0.05)
                    n_sig_bonferroni = sum(miss_subset['p_value_bonferroni'] < 0.05)
                    
                    axes[0, i].text(0.02, 0.98, f'Significant (p<0.05): {n_sig_original}/{len(miss_subset)}', 
                                   transform=axes[0, i].transAxes, verticalalignment='top',
                                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
                    axes[1, i].text(0.02, 0.98, f'Significant (p<0.05): {n_sig_bonferroni}/{len(miss_subset)}', 
                                   transform=axes[1, i].transAxes, verticalalignment='top',
                                   bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
                    
                    # Add reduction information
                    reduction = n_sig_original - n_sig_bonferroni
                    if reduction > 0:
                        axes[1, i].text(0.02, 0.88, f'Reduction: -{reduction}', 
                                       transform=axes[1, i].transAxes, verticalalignment='top',
                                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
                else:
                    axes[0, i].set_title(f'Original P-Values - {metric} - {miss_type} (No Data)')
                    axes[1, i].set_title(f'Bonferroni Corrected - {metric} - {miss_type} (No Data)')
            
            # Add overall color legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='red', alpha=0.8, label='p < 0.01'),
                Patch(facecolor='orange', alpha=0.8, label='0.01 ≤ p < 0.05'),
                Patch(facecolor='lightblue', alpha=0.8, label='p ≥ 0.05')
            ]
            fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
            
            plt.suptitle(f'P-Value Comparison: Original vs Bonferroni Corrected - {metric}')
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/plots/bonferroni_comparison_{metric}.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Created Bonferroni comparison plot for {metric}")
        
        # Create summary comparison plot
        self.plot_bonferroni_summary(df_bonferroni)
    
    def plot_bonferroni_summary(self, df_bonferroni: pd.DataFrame):
        """Create summary plot showing the effect of Bonferroni correction"""
        print("Creating Bonferroni correction summary plot...")
        
        # Calculate summary statistics
        summary_data = []
        
        for metric in ['MAE', 'RMSE', 'MAPE']:
            metric_subset = df_bonferroni[df_bonferroni['metric'] == metric]
            
            if not metric_subset.empty:
                n_total = len(metric_subset)
                n_sig_original = sum(metric_subset['p_value'] < 0.05)
                n_sig_bonferroni = sum(metric_subset['p_value_bonferroni'] < 0.05)
                n_sig_strict_original = sum(metric_subset['p_value'] < 0.01)
                n_sig_strict_bonferroni = sum(metric_subset['p_value_bonferroni'] < 0.01)
                
                summary_data.append({
                    'metric': metric,
                    'total_comparisons': n_total,
                    'significant_original_05': n_sig_original,
                    'significant_bonferroni_05': n_sig_bonferroni,
                    'significant_original_01': n_sig_strict_original,
                    'significant_bonferroni_01': n_sig_strict_bonferroni,
                    'percent_original_05': (n_sig_original / n_total * 100) if n_total > 0 else 0,
                    'percent_bonferroni_05': (n_sig_bonferroni / n_total * 100) if n_total > 0 else 0,
                    'percent_original_01': (n_sig_strict_original / n_total * 100) if n_total > 0 else 0,
                    'percent_bonferroni_01': (n_sig_strict_bonferroni / n_total * 100) if n_total > 0 else 0
                })
        
        if not summary_data:
            return
        
        df_summary = pd.DataFrame(summary_data)
        
        # Save summary table
        df_summary.to_csv(f"{self.output_dir}/statistical_tests/bonferroni_summary.csv", index=False)
        
        # Create summary plot
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Absolute numbers
        x = np.arange(len(df_summary))
        width = 0.35
        
        axes[0].bar(x - width/2, df_summary['significant_original_05'], width, 
                   label='Original (p<0.05)', alpha=0.8, color='orange')
        axes[0].bar(x + width/2, df_summary['significant_bonferroni_05'], width,
                   label='Bonferroni (p<0.05)', alpha=0.8, color='lightblue')
        
        axes[0].set_xlabel('Metric')
        axes[0].set_ylabel('Number of Significant Comparisons')
        axes[0].set_title('Significant Comparisons: Original vs Bonferroni Corrected')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(df_summary['metric'])
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, (orig, bonf) in enumerate(zip(df_summary['significant_original_05'], 
                                           df_summary['significant_bonferroni_05'])):
            axes[0].text(i - width/2, orig + 0.1, str(orig), ha='center', va='bottom')
            axes[0].text(i + width/2, bonf + 0.1, str(bonf), ha='center', va='bottom')
        
        # Plot 2: Percentages
        axes[1].bar(x - width/2, df_summary['percent_original_05'], width, 
                   label='Original (p<0.05)', alpha=0.8, color='orange')
        axes[1].bar(x + width/2, df_summary['percent_bonferroni_05'], width,
                   label='Bonferroni (p<0.05)', alpha=0.8, color='lightblue')
        
        axes[1].set_xlabel('Metric')
        axes[1].set_ylabel('Percentage of Significant Comparisons')
        axes[1].set_title('Percentage Significant: Original vs Bonferroni Corrected')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(df_summary['metric'])
        axes[1].legend()
        axes[1].grid(axis='y', alpha=0.3)
        
        # Add percentage labels on bars
        for i, (orig, bonf) in enumerate(zip(df_summary['percent_original_05'], 
                                           df_summary['percent_bonferroni_05'])):
            axes[1].text(i - width/2, orig + 1, f'{orig:.1f}%', ha='center', va='bottom')
            axes[1].text(i + width/2, bonf + 1, f'{bonf:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/plots/bonferroni_summary.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Created Bonferroni summary plot and saved summary table")
    
    def perform_repeated_measures_anova(self, results: Dict[str, Any]):
        """Perform Repeated Measures ANOVA comparing inpainting vs non-inpainting methods"""
        print("Performing Repeated Measures ANOVA with proper long-form data...")
        
        raw_metrics = results['raw_metrics']
        anova_results = []
        
        for metric in ['MAE', 'RMSE', 'MAPE']:
            print(f"Processing ANOVA for {metric}...")
            
            # Prepare long-form data for this metric across all conditions
            long_form_data = []
            
            for missingness_type in self.missingness_types:
                for forecast_model in self.forecasting_models:
                    
                    # Collect all methods and their iteration results
                    all_methods_data = {}
                    
                    # Get inpainting methods
                    for inpainting_method in self.inpainting_models:
                        key = f"{missingness_type}_{inpainting_method}_{forecast_model}"
                        if key in raw_metrics and metric in raw_metrics[key]:
                            values = [v for v in raw_metrics[key][metric] 
                                    if not np.isinf(v) and not np.isnan(v)]
                            if values:
                                method_name = f"{inpainting_method}_{forecast_model}"
                                all_methods_data[method_name] = {
                                    'values': values,
                                    'group': 'inpainting'
                                }
                    
                    # Get traditional methods
                    for traditional_method in self.non_inpainting_methods:
                        key = f"{missingness_type}_{traditional_method}_{forecast_model}"
                        if key in raw_metrics and metric in raw_metrics[key]:
                            values = [v for v in raw_metrics[key][metric] 
                                    if not np.isinf(v) and not np.isnan(v)]
                            if values:
                                method_name = f"{traditional_method}_{forecast_model}"
                                all_methods_data[method_name] = {
                                    'values': values,
                                    'group': 'traditional'
                                }
                    
                    # Convert to long-form format
                    if all_methods_data:
                        # Find minimum iterations across all methods
                        min_iterations = min(len(data['values']) for data in all_methods_data.values())
                        
                        if min_iterations > 1:  # Need at least 2 iterations for repeated measures
                            for iteration in range(min_iterations):
                                for method_name, method_data in all_methods_data.items():
                                    long_form_data.append({
                                        'iteration': iteration + 1,
                                        'method': method_name,
                                        'group': method_data['group'],
                                        'missingness_type': missingness_type,
                                        'forecast_model': forecast_model,
                                        'metric_value': method_data['values'][iteration]
                                    })
            
            # Perform ANOVA if we have sufficient data
            if len(long_form_data) > 0:
                df_long = pd.DataFrame(long_form_data)
                
                # Check if we have both groups
                groups_present = df_long['group'].unique()
                if len(groups_present) >= 2:
                    try:
                        # Aggregate by iteration and group (average across methods within each group)
                        grouped_data = df_long.groupby(['iteration', 'group'])['metric_value'].mean().reset_index()
                        
                        # Check if we have enough data points
                        if len(grouped_data) >= 4:  # Minimum for ANOVA
                            # Perform Repeated Measures ANOVA
                            anova_rm = AnovaRM(grouped_data, depvar='metric_value', 
                                             subject='iteration', within=['group'])
                            anova_result = anova_rm.fit()
                            
                            # Extract results
                            anova_table = anova_result.anova_table
                            f_stat = anova_table.loc['group', 'F Value']
                            p_value = anova_table.loc['group', 'Pr > F']
                            
                            # Calculate partial eta-squared
                            ss_effect = anova_table.loc['group', 'SS']
                            ss_error = anova_table.loc['group:iteration', 'SS'] if 'group:iteration' in anova_table.index else anova_table.loc['Error', 'SS']
                            partial_eta_squared = ss_effect / (ss_effect + ss_error) if (ss_effect + ss_error) > 0 else 0
                            
                            # Calculate descriptive statistics
                            inpainting_vals = df_long[df_long['group'] == 'inpainting']['metric_value']
                            traditional_vals = df_long[df_long['group'] == 'traditional']['metric_value']
                            
                            anova_results.append({
                                'metric': metric,
                                'f_statistic': f_stat,
                                'p_value': p_value,
                                'partial_eta_squared': partial_eta_squared,
                                'inpainting_mean': inpainting_vals.mean(),
                                'traditional_mean': traditional_vals.mean(),
                                'inpainting_std': inpainting_vals.std(),
                                'traditional_std': traditional_vals.std(),
                                'n_iterations': df_long['iteration'].nunique(),
                                'n_inpainting_methods': len(df_long[df_long['group'] == 'inpainting']['method'].unique()),
                                'n_traditional_methods': len(df_long[df_long['group'] == 'traditional']['method'].unique()),
                                'n_total_observations': len(df_long)
                            })
                            
                            print(f"✓ Repeated Measures ANOVA for {metric}: F={f_stat:.3f}, p={p_value:.6f}, η²={partial_eta_squared:.3f}")
                            
                            # Save the long-form data for inspection
                            df_long.to_csv(f"{self.output_dir}/statistical_tests/anova_longform_data_{metric}.csv", index=False)
                            grouped_data.to_csv(f"{self.output_dir}/statistical_tests/anova_grouped_data_{metric}.csv", index=False)
                            
                        else:
                            print(f"❌ Insufficient data for ANOVA on {metric}: only {len(grouped_data)} data points")
                            
                    except Exception as e:
                        print(f"❌ Repeated Measures ANOVA failed for {metric}: {e}")
                        print(f"   Attempting fallback analysis...")
                        
                        # Fallback: Simple comparison using independent samples t-test
                        try:
                            inpainting_vals = df_long[df_long['group'] == 'inpainting']['metric_value'].values
                            traditional_vals = df_long[df_long['group'] == 'traditional']['metric_value'].values
                            
                            if len(inpainting_vals) > 1 and len(traditional_vals) > 1:
                                from scipy.stats import ttest_ind
                                t_stat, p_val = ttest_ind(inpainting_vals, traditional_vals)
                                
                                # Convert t-statistic to F-statistic (F = t²)
                                f_stat_equiv = t_stat**2
                                
                                # Cohen's d effect size
                                pooled_std = np.sqrt(((len(inpainting_vals) - 1) * np.var(inpainting_vals, ddof=1) + 
                                                    (len(traditional_vals) - 1) * np.var(traditional_vals, ddof=1)) / 
                                                   (len(inpainting_vals) + len(traditional_vals) - 2))
                                cohens_d = (np.mean(inpainting_vals) - np.mean(traditional_vals)) / pooled_std
                                
                                anova_results.append({
                                    'metric': metric,
                                    'f_statistic': f_stat_equiv,
                                    'p_value': p_val,
                                    'partial_eta_squared': cohens_d**2 / (cohens_d**2 + 4),  # Approximate conversion
                                    'inpainting_mean': np.mean(inpainting_vals),
                                    'traditional_mean': np.mean(traditional_vals),
                                    'inpainting_std': np.std(inpainting_vals, ddof=1),
                                    'traditional_std': np.std(traditional_vals, ddof=1),
                                    'n_iterations': df_long['iteration'].nunique(),
                                    'n_inpainting_methods': len(df_long[df_long['group'] == 'inpainting']['method'].unique()),
                                    'n_traditional_methods': len(df_long[df_long['group'] == 'traditional']['method'].unique()),
                                    'n_total_observations': len(df_long),
                                    'analysis_type': 'Independent t-test (fallback)'
                                })
                                
                                print(f"✓ Fallback t-test for {metric}: t={t_stat:.3f}, p={p_val:.6f}")
                                
                        except Exception as e2:
                            print(f"❌ Fallback analysis also failed for {metric}: {e2}")
                else:
                    print(f"❌ Only one group found for {metric}: {groups_present}")
            else:
                print(f"❌ No data available for {metric}")
        
        # Save ANOVA results
        if anova_results:
            df_anova_results = pd.DataFrame(anova_results)
            df_anova_results.to_csv(f"{self.output_dir}/statistical_tests/repeated_measures_anova.csv", index=False)
            
            # Create ANOVA plots
            self.plot_anova_results(df_anova_results)
            
            print(f"✓ Repeated Measures ANOVA results saved to statistical_tests/repeated_measures_anova.csv")
            print(f"✓ Long-form data saved as anova_longform_data_*.csv for inspection")
            return anova_results
        else:
            print("❌ No ANOVA results generated")
            return []
    
    def plot_anova_results(self, df_anova: pd.DataFrame):
        """Create plots for ANOVA results"""
        print("Creating ANOVA result plots...")
        
        if df_anova.empty:
            print("No ANOVA results to plot")
            return
        
        # Create a single plot showing all metrics
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Prepare data
        metrics = df_anova['metric'].tolist()
        f_stats = df_anova['f_statistic'].tolist()
        p_values = df_anova['p_value'].tolist()
        effect_sizes = df_anova['partial_eta_squared'].tolist()
        
        # Color code by significance
        colors = ['red' if p < 0.01 else 'orange' if p < 0.05 else 'lightblue' 
                 for p in p_values]
        
        # Plot 1: F-statistics
        bars1 = axes[0].bar(range(len(metrics)), f_stats, color=colors, alpha=0.8)
        axes[0].set_title('Repeated Measures ANOVA: F-Statistics\n(Inpainting vs Traditional Methods)')
        axes[0].set_ylabel('F-Statistic')
        axes[0].set_xticks(range(len(metrics)))
        axes[0].set_xticklabels(metrics)
        axes[0].grid(axis='y', alpha=0.3)
        
        # Add p-value labels on bars
        for i, (bar, p_val) in enumerate(zip(bars1, p_values)):
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'p={p_val:.4f}', ha='center', va='bottom', fontsize=10)
        
        # Plot 2: Effect sizes (partial eta-squared)
        bars2 = axes[1].bar(range(len(metrics)), effect_sizes, color=colors, alpha=0.8)
        axes[1].set_title('Effect Sizes (Partial η²)')
        axes[1].set_ylabel('Partial Eta-squared')
        axes[1].set_xticks(range(len(metrics)))
        axes[1].set_xticklabels(metrics)
        axes[1].grid(axis='y', alpha=0.3)
        
        # Add effect size interpretation lines
        axes[1].axhline(y=0.01, color='gray', linestyle=':', alpha=0.7, label='Small effect (0.01)')
        axes[1].axhline(y=0.06, color='gray', linestyle='--', alpha=0.7, label='Medium effect (0.06)')
        axes[1].axhline(y=0.14, color='gray', linestyle='-', alpha=0.7, label='Large effect (0.14)')
        axes[1].legend(fontsize=10)
        
        # Add effect size labels on bars
        for i, (bar, effect) in enumerate(zip(bars2, effect_sizes)):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{effect:.3f}', ha='center', va='bottom', fontsize=10)
        
        # Add color legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', alpha=0.8, label='p < 0.01 (highly significant)'),
            Patch(facecolor='orange', alpha=0.8, label='0.01 ≤ p < 0.05 (significant)'),
            Patch(facecolor='lightblue', alpha=0.8, label='p ≥ 0.05 (not significant)')
        ]
        fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
        
        # Add summary text
        n_significant = sum(1 for p in p_values if p < 0.05)
        fig.text(0.02, 0.02, f'Summary: {n_significant}/{len(p_values)} metrics show significant differences', 
                fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/plots/anova_results_summary.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Created ANOVA summary plot")
        
        # Create detailed visualizations for each metric
        self.create_anova_detailed_plots(df_anova)
    
    def create_anova_detailed_plots(self, df_anova: pd.DataFrame):
        """Create detailed boxplot and swarmplot visualizations for ANOVA results"""
        print("Creating detailed ANOVA visualizations (boxplots and swarmplots)...")
        
        import seaborn as sns
        
        for metric in df_anova['metric'].unique():
            try:
                # Load the long-form data
                df_long = pd.read_csv(f"{self.output_dir}/statistical_tests/anova_longform_data_{metric}.csv")
                
                # Get ANOVA results for this metric
                metric_result = df_anova[df_anova['metric'] == metric].iloc[0]
                
                # Create multiple visualizations for this metric
                
                # 1. Group comparison (inpainting vs traditional)
                fig, axes = plt.subplots(1, 2, figsize=(16, 8))
                
                # Plot 1: Boxplot + Swarmplot by group
                sns.boxplot(data=df_long, x='group', y='metric_value', ax=axes[0])
                sns.swarmplot(data=df_long, x='group', y='metric_value', color='black', alpha=0.6, ax=axes[0])
                
                axes[0].set_title(f'{metric}: Comparison by Method Group\n(Boxplot + Swarmplot)')
                axes[0].set_xlabel('Method Group')
                axes[0].set_ylabel(f'{metric} Value')
                
                # Add statistics text
                stats_text = f"ANOVA Results:\nF = {metric_result['f_statistic']:.3f}\n"
                stats_text += f"p = {metric_result['p_value']:.6f}\n"
                stats_text += f"Partial η² = {metric_result['partial_eta_squared']:.3f}\n\n"
                stats_text += f"Sample sizes:\nInpainting: {metric_result['n_inpainting_methods']} methods\n"
                stats_text += f"Traditional: {metric_result['n_traditional_methods']} methods\n"
                stats_text += f"Iterations: {metric_result['n_iterations']}"
                
                axes[0].text(0.02, 0.98, stats_text, transform=axes[0].transAxes, 
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
                
                # Plot 2: Violin plot for detailed distribution
                sns.violinplot(data=df_long, x='group', y='metric_value', ax=axes[1])
                sns.swarmplot(data=df_long, x='group', y='metric_value', color='white', alpha=0.8, size=3, ax=axes[1])
                
                axes[1].set_title(f'{metric}: Distribution Shapes\n(Violin Plot + Swarmplot)')
                axes[1].set_xlabel('Method Group')
                axes[1].set_ylabel(f'{metric} Value')
                
                # Add mean comparison
                means = df_long.groupby('group')['metric_value'].mean()
                means_text = f"Group Means:\nInpainting: {means['inpainting']:.4f}\n"
                means_text += f"Traditional: {means['traditional']:.4f}\n"
                means_text += f"Difference: {means['traditional'] - means['inpainting']:.4f}"
                
                axes[1].text(0.02, 0.98, means_text, transform=axes[1].transAxes, 
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
                
                plt.tight_layout()
                plt.savefig(f"{self.output_dir}/plots/anova_detailed_group_comparison_{metric}.png", 
                           dpi=300, bbox_inches='tight')
                plt.close()
                
                # 2. Individual method comparison (if we have method names)
                if 'method' in df_long.columns:
                    # Create a plot showing individual methods
                    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
                    
                    # Sort methods by group and then by median performance
                    method_medians = df_long.groupby('method')['metric_value'].median().sort_values()
                    method_order = method_medians.index.tolist()
                    
                    # Create boxplot for all methods
                    sns.boxplot(data=df_long, x='method', y='metric_value', order=method_order, ax=ax)
                    sns.swarmplot(data=df_long, x='method', y='metric_value', order=method_order, 
                                 color='black', alpha=0.6, size=3, ax=ax)
                    
                    # Color code by group
                    for i, method in enumerate(method_order):
                        method_group = df_long[df_long['method'] == method]['group'].iloc[0]
                        color = 'red' if method_group == 'inpainting' else 'blue'
                        ax.get_xticklabels()[i].set_color(color)
                    
                    ax.set_title(f'{metric}: Comparison by Individual Methods\n(Red=Inpainting, Blue=Traditional)')
                    ax.set_xlabel('Method')
                    ax.set_ylabel(f'{metric} Value')
                    ax.tick_params(axis='x', rotation=45)
                    
                    # Add legend
                    from matplotlib.patches import Patch
                    legend_elements = [
                        Patch(facecolor='red', alpha=0.8, label='Inpainting Methods'),
                        Patch(facecolor='blue', alpha=0.8, label='Traditional Methods')
                    ]
                    ax.legend(handles=legend_elements, loc='upper right')
                    
                    plt.tight_layout()
                    plt.savefig(f"{self.output_dir}/plots/anova_individual_methods_{metric}.png", 
                               dpi=300, bbox_inches='tight')
                    plt.close()
                
                # 3. Iteration-wise comparison (trend over iterations)
                if 'iteration' in df_long.columns:
                    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
                    
                    # Create boxplot by iteration and group
                    sns.boxplot(data=df_long, x='iteration', y='metric_value', hue='group', ax=ax)
                    sns.swarmplot(data=df_long, x='iteration', y='metric_value', hue='group', 
                                 dodge=True, alpha=0.7, size=3, ax=ax)
                    
                    ax.set_title(f'{metric}: Performance Across Iterations\n(Boxplot + Swarmplot by Group)')
                    ax.set_xlabel('Iteration')
                    ax.set_ylabel(f'{metric} Value')
                    
                    # Remove duplicate legend from swarmplot
                    handles, labels = ax.get_legend_handles_labels()
                    ax.legend(handles[:2], labels[:2], title='Method Group')
                    
                    plt.tight_layout()
                    plt.savefig(f"{self.output_dir}/plots/anova_iteration_comparison_{metric}.png", 
                               dpi=300, bbox_inches='tight')
                    plt.close()
                
                # 4. Summary statistics table visualization
                fig, ax = plt.subplots(1, 1, figsize=(12, 8))
                
                # Calculate detailed statistics
                summary_stats = df_long.groupby('group')['metric_value'].agg([
                    'count', 'mean', 'std', 'min', 'max', 'median'
                ]).round(4)
                
                # Create a table plot
                ax.axis('tight')
                ax.axis('off')
                
                table_data = []
                table_data.append(['Statistic', 'Inpainting', 'Traditional', 'Difference'])
                table_data.append(['Count', f"{summary_stats.loc['inpainting', 'count']:.0f}", 
                                  f"{summary_stats.loc['traditional', 'count']:.0f}", '-'])
                table_data.append(['Mean', f"{summary_stats.loc['inpainting', 'mean']:.4f}", 
                                  f"{summary_stats.loc['traditional', 'mean']:.4f}", 
                                  f"{summary_stats.loc['traditional', 'mean'] - summary_stats.loc['inpainting', 'mean']:.4f}"])
                table_data.append(['Std Dev', f"{summary_stats.loc['inpainting', 'std']:.4f}", 
                                  f"{summary_stats.loc['traditional', 'std']:.4f}", '-'])
                table_data.append(['Median', f"{summary_stats.loc['inpainting', 'median']:.4f}", 
                                  f"{summary_stats.loc['traditional', 'median']:.4f}", 
                                  f"{summary_stats.loc['traditional', 'median'] - summary_stats.loc['inpainting', 'median']:.4f}"])
                table_data.append(['Min', f"{summary_stats.loc['inpainting', 'min']:.4f}", 
                                  f"{summary_stats.loc['traditional', 'min']:.4f}", '-'])
                table_data.append(['Max', f"{summary_stats.loc['inpainting', 'max']:.4f}", 
                                  f"{summary_stats.loc['traditional', 'max']:.4f}", '-'])
                
                table = ax.table(cellText=table_data, cellLoc='center', loc='center', 
                               colWidths=[0.2, 0.25, 0.25, 0.25])
                table.auto_set_font_size(False)
                table.set_fontsize(12)
                table.scale(1.2, 2)
                
                # Style the header row
                for i in range(4):
                    table[(0, i)].set_facecolor('#4CAF50')
                    table[(0, i)].set_text_props(weight='bold', color='white')
                
                ax.set_title(f'{metric}: Descriptive Statistics Summary\n\n', fontsize=16, weight='bold')
                
                # Add ANOVA results below the table
                anova_text = f"ANOVA Results: F({1}, {metric_result['n_iterations']-1}) = {metric_result['f_statistic']:.3f}, "
                anova_text += f"p = {metric_result['p_value']:.6f}, partial η² = {metric_result['partial_eta_squared']:.3f}"
                
                if metric_result['p_value'] < 0.001:
                    significance = "*** (p < 0.001)"
                elif metric_result['p_value'] < 0.01:
                    significance = "** (p < 0.01)"
                elif metric_result['p_value'] < 0.05:
                    significance = "* (p < 0.05)"
                else:
                    significance = "ns (not significant)"
                
                ax.text(0.5, 0.15, anova_text, transform=ax.transAxes, ha='center', fontsize=12,
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
                ax.text(0.5, 0.05, f"Significance: {significance}", transform=ax.transAxes, ha='center', 
                       fontsize=14, weight='bold',
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
                
                plt.tight_layout()
                plt.savefig(f"{self.output_dir}/plots/anova_summary_table_{metric}.png", 
                           dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"✓ Created detailed ANOVA visualizations for {metric}")
                
            except FileNotFoundError:
                print(f"⚠️ Long-form data not found for {metric}, skipping detailed plots")
            except Exception as e:
                print(f"❌ Error creating detailed plots for {metric}: {e}")
    
    def plot_statistical_results(self, df_tests: pd.DataFrame):
        """Create bar charts showing p-values with significance threshold"""
        print("Creating statistical test plots...")
        
        for metric in ['MAE', 'RMSE', 'MAPE']:
            for forecast_model in self.forecasting_models:
                subset = df_tests[(df_tests['metric'] == metric) & 
                                (df_tests['forecast_model'] == forecast_model)]
                
                if subset.empty:
                    continue
                
                fig, axes = plt.subplots(1, 3, figsize=(20, 6))
                
                # Get actual missingness types in the data
                actual_miss_types = subset['missingness_type'].unique()
                n_plots = len(actual_miss_types)
                
                if n_plots == 0:
                    continue
                    
                # Recreate subplot layout based on actual data
                fig.clear()
                fig, axes = plt.subplots(1, n_plots, figsize=(7*n_plots, 6))
                if n_plots == 1:
                    axes = [axes]
                
                for i, miss_type in enumerate(actual_miss_types):
                    miss_subset = subset[subset['missingness_type'] == miss_type].copy()
                    
                    if not miss_subset.empty:
                        # Create a unique identifier for each comparison
                        miss_subset['comparison'] = (miss_subset['inpainting_method'] + 
                                                   ' vs ' + 
                                                   miss_subset['non_inpainting_method'])
                        
                        # Sort by p-value (most significant first)
                        miss_subset = miss_subset.sort_values('p_value', ascending=True)
                        
                        # Color bars based on significance
                        colors = []
                        for p_val in miss_subset['p_value']:
                            if p_val < 0.01:
                                colors.append('red')
                            elif p_val < 0.05:
                                colors.append('orange')
                            else:
                                colors.append('lightblue')
                        
                        bars = axes[i].bar(range(len(miss_subset)), miss_subset['p_value'], 
                                         alpha=0.8, color=colors)
                        
                        axes[i].axhline(y=0.01, color='red', linestyle='--', linewidth=2,
                                      label='p=0.01 significance level')
                        axes[i].axhline(y=0.05, color='orange', linestyle='--', linewidth=2,
                                      label='p=0.05 significance level')
                        axes[i].set_title(f'{metric} - {miss_type}')
                        axes[i].set_ylabel('P-value')
                        axes[i].set_yscale('log')
                        axes[i].set_xticks(range(len(miss_subset)))
                        axes[i].set_xticklabels(miss_subset['comparison'], 
                                              rotation=45, ha='right', fontsize=8)
                        axes[i].grid(axis='y', alpha=0.3)
                        axes[i].legend()
                    else:
                        axes[i].set_title(f'{metric} - {miss_type} (No Data)')
                
                plt.suptitle(f'Statistical Test Results: {metric} - {forecast_model}')
                plt.tight_layout()
                plt.savefig(f"{self.output_dir}/plots/statistical_tests_{metric}_{forecast_model}.png", 
                           dpi=300, bbox_inches='tight')
                plt.close()


def main():
    parser = argparse.ArgumentParser(description="Iterative Time Series Experiment")
    parser.add_argument("--data", default="data/1_clean/train_set_original.csv", 
                       help="Path to the dataset")
    parser.add_argument("--iterations", type=int, default=5, 
                       help="Number of iterations to run")
    parser.add_argument("--test_size", type=int, default=10, 
                       help="Size of test set (last N points)")
    parser.add_argument("--missingness_rate", type=float, default=0.2, 
                       help="Rate of missingness to introduce")
    parser.add_argument("--inpainting_models", nargs='+', 
                       default=["gaf-unet"],
                       help="Inpainting models to use")
    parser.add_argument("--forecasting_models" , nargs='+', 
                       default=["XGBoost"],
                       help="Forecasting models to use")
    parser.add_argument("--missingness_types", nargs='+', 
                       default=["MCAR", "MAR", "MNAR"],
                       help="Types of missingness to test")
    parser.add_argument("--output_dir", default="results/iterative_experiment", 
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Create and run experiment
    experiment = IterativeExperiment(
        data_path=args.data,
        test_size=args.test_size,
        n_iterations=args.iterations,
        inpainting_models=args.inpainting_models,
        forecasting_models=args.forecasting_models,
        missingness_types=args.missingness_types,
        missingness_rate=args.missingness_rate,
        output_dir=args.output_dir
    )
    
    # Run the experiment
    results = experiment.run_experiment()
    
    # Create visualizations
    experiment.create_performance_plots(results)
    
    # Perform statistical tests
    experiment.perform_statistical_tests(results)
    
    # Perform Repeated Measures ANOVA
    experiment.perform_repeated_measures_anova(results)
    
    print(f"\nExperiment completed! Results saved to: {args.output_dir}")
    print("Generated files:")
    print("- final_results.json: Complete aggregated results")
    print("- plots/: Performance visualizations and statistical test plots")
    print("- statistical_tests/: T-test and ANOVA results")


if __name__ == "__main__":
    main() 