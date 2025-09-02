#!/usr/bin/env python3
"""
Improved Time Series Experiment Runner

This script implements the new iterative methodology for evaluating time series 
repair and forecasting methods:

1. Split dataset into train/test (test = last 10 points)
2. For each iteration:
   - Generate missingness (MCAR, MAR, MNAR) in training data
   - Apply all repair techniques (inpainting + traditional)
   - Forecast next 10 points using repaired data
   - Calculate metrics (MAE, RMSE, MAPE)
3. Aggregate results across iterations
4. Generate performance visualizations
5. Perform statistical significance tests

Usage examples:
  # Quick test with minimal models (recommended for first run)
  python run_improved_experiment.py --quick

  # Full experiment with all models (takes longer)
  python run_improved_experiment.py --full

  # Custom configuration
  python run_improved_experiment.py --iterations 10 --inpainting_models gaf-unet mtf-unet --forecasting_models XGBoost Prophet
"""

import sys
import argparse
import pandas as pd
from iterative_experiment import IterativeExperiment

def run_quick_experiment():
    """Run a quick experiment for testing and demonstration"""
    print("="*60)
    print("RUNNING QUICK EXPERIMENT")
    print("="*60)
    print("Configuration: 10 iterations, MCAR/MAR/MNAR, 4 inpainting vs 16 traditional methods, 3 forecasting models")
    print("This should take about 20-30 minutes...")
    print()
    
    experiment = IterativeExperiment(
        data_paths=["data/0_source_data/boiler_outlet_temp_univ.csv",
                   "data/0_source_data/pump_sensor_28_univ.csv",
                   "data/0_source_data/vibration_sensor_S1.csv"],
        n_iterations=10,
        inpainting_models=["gaf-unet", "mtf-unet", "rp-unet", "spec-unet"],
        forecasting_models=["XGBoost", "HoltWinters", "Prophet"],
        missingness_types=["MCAR", "MAR", "MNAR"],
        test_size=10,
        missingness_rates=[0.005, 0.02, 0.05, 0.10, 0.20, 0.30],  # 2%, 5%, 10%
        output_dir="results/quick_experiment"
    )
    
    results = experiment.run_experiment()
    
    # Get the final dataframe
    df_final = experiment.get_final_dataframe()
    
    experiment.create_performance_plots(results)
    experiment.perform_statistical_tests(results)
    experiment.perform_bonferroni_correction(results)
    experiment.perform_repeated_measures_anova(results)
    
    print("\n" + "="*60)
    print("QUICK EXPERIMENT COMPLETED!")
    print("="*60)
    print(f"Results saved to: results/quick_experiment")
    print(f"Final dataframe shape: {df_final.shape}")
    print("Check the plots/ directory for visualizations")
    print("Check statistical_tests/ for t-test, Bonferroni, and ANOVA results")
    
    return df_final

def run_medium_experiment():
    """Run a medium-sized experiment with multiple models"""
    print("="*60)
    print("RUNNING MEDIUM EXPERIMENT")
    print("="*60)
    print("Configuration: 5 iterations, all missingness types, selected models")
    print("This should take about 30-45 minutes...")
    print()
    
    experiment = IterativeExperiment(
        data_paths=["data/0_source_data/boiler_outlet_temp_univ.csv",
                   "data/0_source_data/pump_sensor_28_univ.csv",
                   "data/0_source_data/vibration_sensor_S1.csv"],
        n_iterations=5,
        inpainting_models=["gaf-unet", "mtf-unet"],
        forecasting_models=["XGBoost", "Prophet"],
        missingness_types=["MCAR", "MAR", "MNAR"],
        test_size=10,
        missingness_rates=[0.02, 0.05, 0.10],  # 2%, 5%, 10%
        output_dir="results/medium_experiment"
    )
    
    results = experiment.run_experiment()
    
    # Get the final dataframe
    df_final = experiment.get_final_dataframe()
    
    experiment.create_performance_plots(results)
    experiment.perform_statistical_tests(results)
    experiment.perform_bonferroni_correction(results)
    experiment.perform_repeated_measures_anova(results)
    
    print("\n" + "="*60)
    print("MEDIUM EXPERIMENT COMPLETED!")
    print("="*60)
    print(f"Results saved to: results/medium_experiment")
    print(f"Final dataframe shape: {df_final.shape}")
    
    return df_final

def run_full_experiment():
    """Run the complete experiment with all available models"""
    print("="*60)
    print("RUNNING FULL EXPERIMENT")
    print("="*60)
    print("Configuration: 10 iterations, all missingness types, all models")
    print("This will take 2-3 hours. Consider running overnight.")
    print()
    
    response = input("This is a long experiment. Continue? (y/n): ")
    if response.lower() != 'y':
        print("Experiment cancelled.")
        return
    
    experiment = IterativeExperiment(
        data_paths=["data/0_source_data/boiler_outlet_temp_univ.csv",
                   "data/0_source_data/pump_sensor_28_univ.csv",
                   "data/0_source_data/vibration_sensor_S1.csv"],
        n_iterations=10,
        inpainting_models=["gaf-unet", "mtf-unet", "rp-unet", "spec-unet"],
        forecasting_models=["XGBoost", "Prophet", "SARIMAX", "HoltWinters"],
        missingness_types=["MCAR", "MAR", "MNAR"],
        test_size=10,
        missingness_rates=[0.02, 0.05, 0.10],  # 2%, 5%, 10%
        output_dir="results/full_experiment"
    )
    
    results = experiment.run_experiment()
    
    # Get the final dataframe
    df_final = experiment.get_final_dataframe()
    
    experiment.create_performance_plots(results)
    experiment.perform_statistical_tests(results)
    experiment.perform_bonferroni_correction(results)
    experiment.perform_repeated_measures_anova(results)
    
    print("\n" + "="*60)
    print("FULL EXPERIMENT COMPLETED!")
    print("="*60)
    print(f"Results saved to: results/full_experiment")
    print(f"Final dataframe shape: {df_final.shape}")
    
    return df_final

def print_results_summary(results_dir):
    """Print a summary of what to look for in the results"""
    print("\n" + "="*60)
    print("RESULTS GUIDE")
    print("="*60)
    print(f"Your results are in: {results_dir}")
    print()
    print("KEY FILES TO EXAMINE:")
    print("📊 plots/average_MAE_performance.png     - Overall MAE comparison")
    print("📊 plots/average_RMSE_performance.png    - Overall RMSE comparison") 
    print("📊 plots/average_MAPE_performance.png    - Overall MAPE comparison")
    print("📈 plots/statistical_tests_*.png         - P-value significance tests")
    print("📈 plots/bonferroni_comparison_*.png     - Original vs Bonferroni corrected p-values")
    print("📈 plots/bonferroni_summary.png          - Summary of Bonferroni correction effects")
    print("📈 plots/anova_results_summary.png       - ANOVA F-statistics and effect sizes overview")
    print("📈 plots/anova_detailed_group_comparison_*.png - Boxplot + swarmplot by method group")
    print("📈 plots/anova_individual_methods_*.png  - Individual method performance (boxplot + swarmplot)")
    print("📈 plots/anova_iteration_comparison_*.png - Performance trends across iterations")
    print("📈 plots/anova_summary_table_*.png       - Descriptive statistics tables")
    print("📋 statistical_tests/t_test_results.csv  - ORIGINAL t-test results (uncorrected)")
    print("📋 statistical_tests/t_test_bonferroni_corrected.csv - CORRECTED t-test results")
    print("📋 statistical_tests/bonferroni_summary.csv - Summary comparing both approaches")
    print("📋 statistical_tests/repeated_measures_anova.csv - ANOVA results table")
    print("📄 final_results.json                    - Complete numerical results with timing")
    print("⏱️ timing_report.json                     - Detailed timing and system information")
    print("💻 Individual iteration_*.json files      - Per-iteration results with timing")
    print()
    print("WHAT TO LOOK FOR:")
    print("• Lower values = better performance in bar charts")
    print("• P-values < 0.01 = statistically significant differences")
    print("• Red bars in statistical plots = significant improvements")
    print("• ORIGINAL t-tests: May have inflated significance due to multiple comparisons")
    print("• BONFERRONI corrected: More conservative, controls family-wise error rate")
    print("• Compare the two CSV files to see which results survive correction")
    print("• Bonferroni plots show side-by-side comparison with reduction counts")
    print("• Higher F-statistics = stronger evidence of differences")
    print("• Effect sizes (η²): 0.01=small, 0.06=medium, 0.14=large effects")
    print("• DETAILED ANOVA PLOTS: Boxplots + swarmplots show data distributions")
    print("• Individual method plots: See which specific methods perform best")
    print("• Iteration trends: Check consistency across experimental iterations")
    print("• TIMING ANALYSIS: Check timing_report.json for performance bottlenecks")
    print("• System info: Hardware specs and resource usage during experiments")
    print("• Compare inpainting methods vs traditional interpolation/imputation")

def main():
    parser = argparse.ArgumentParser(
        description="Run improved time series repair and forecasting experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Preset experiment types
    parser.add_argument("--quick", action="store_true", 
                       help="Run quick experiment (3 iterations, minimal models)")
    parser.add_argument("--medium", action="store_true",
                       help="Run medium experiment (5 iterations, selected models)")
    parser.add_argument("--full", action="store_true",
                       help="Run full experiment (10 iterations, all models)")
    
    # Custom configuration options
    parser.add_argument("--data", nargs='+', 
                       default=["data/0_source_data/boiler_outlet_temp_univ.csv",
                               "data/0_source_data/pump_sensor_28_univ.csv",
                               "data/0_source_data/vibration_sensor_S1.csv"],
                       help="Paths to the datasets (can specify multiple)")
    parser.add_argument("--iterations", type=int, default=5,
                       help="Number of iterations to run")
    parser.add_argument("--test_size", type=int, default=10,
                       help="Size of test set (last N points)")
    parser.add_argument("--missingness_rates", nargs='+', type=float, default=[0.02, 0.05, 0.10],
                       help="Rates of missingness to introduce (e.g., 0.02 0.05 0.10)")
    parser.add_argument("--inpainting_models", nargs='+',
                       default=["gaf-unet"],
                       help="Inpainting models to use (gaf-unet, mtf-unet, rp-unet, spec-unet)")
    parser.add_argument("--forecasting_models", nargs='+',
                       default=["XGBoost"],
                       help="Forecasting models to use (XGBoost, Prophet, SARIMAX, HoltWinters)")
    parser.add_argument("--missingness_types", nargs='+',
                       default=["MCAR", "MAR", "MNAR"],
                       help="Types of missingness to test")
    parser.add_argument("--output_dir", default="results/custom_experiment",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Print welcome message
    print("🔬 IMPROVED TIME SERIES EXPERIMENT FRAMEWORK")
    print("=" * 60)
    print("This implements the new iterative methodology:")
    print("1. Train/test split (test = last 10 points)")
    print("2. Multiple iterations with random missingness")
    print("3. Comparison of inpainting vs traditional methods")
    print("4. Statistical significance testing")
    print("5. Comprehensive visualizations")
    print()
    
    # Check if all data files exist
    import os
    for data_path in args.data:
        if not os.path.exists(data_path):
            print(f"❌ Error: Data file not found: {data_path}")
            print("Please ensure you have run the data preparation steps first.")
            sys.exit(1)
    
    # Run the appropriate experiment
    if args.quick:
        run_quick_experiment()
        print_results_summary("results/quick_experiment")
    elif args.medium:
        run_medium_experiment()
        print_results_summary("results/medium_experiment")
    elif args.full:
        run_full_experiment()
        print_results_summary("results/full_experiment")
    else:
        # Custom experiment
        print("RUNNING CUSTOM EXPERIMENT")
        print("=" * 60)
        print(f"Configuration:")
        print(f"  - Iterations: {args.iterations}")
        print(f"  - Inpainting models: {args.inpainting_models}")
        print(f"  - Forecasting models: {args.forecasting_models}")
        print(f"  - Missingness types: {args.missingness_types}")
        print(f"  - Test size: {args.test_size}")
        print()
        
        experiment = IterativeExperiment(
            data_paths=args.data,
            n_iterations=args.iterations,
            inpainting_models=args.inpainting_models,
            forecasting_models=args.forecasting_models,
            missingness_types=args.missingness_types,
            test_size=args.test_size,
            missingness_rates=args.missingness_rates,
            output_dir=args.output_dir
        )
        
        results = experiment.run_experiment()
        
        # Get the final dataframe
        df_final = experiment.get_final_dataframe()
        
        experiment.create_performance_plots(results)
        experiment.perform_statistical_tests(results)
        experiment.perform_bonferroni_correction(results)
        experiment.perform_repeated_measures_anova(results)
        
        print(f"Final dataframe shape: {df_final.shape}")
        print_results_summary(args.output_dir)

if __name__ == "__main__":
    main() 