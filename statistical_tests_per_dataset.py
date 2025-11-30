#!/usr/bin/env python3
"""
Statistical tests comparing SD methods vs other methods PER DATASET
Using Wilcoxon signed-rank test with Bonferroni correction
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def wilcoxon_test_with_stats(method_a_values, method_b_values):
    """
    Performs Wilcoxon signed-rank test and calculates additional statistics
    
    Returns:
        dict with statistic, p_value, mean_diff, median_diff, effect_size
    """
    # Calculate differences
    differences = method_a_values - method_b_values
    mean_diff = np.mean(differences)
    median_diff = np.median(differences)
    
    # Wilcoxon test
    try:
        statistic, p_value = stats.wilcoxon(method_a_values, method_b_values, alternative='two-sided')
    except Exception as e:
        print(f"Warning: Wilcoxon test failed - {e}")
        statistic, p_value = np.nan, np.nan
    
    # Effect size (r = Z / sqrt(N))
    n = len(differences)
    if not np.isnan(p_value) and p_value > 0:
        z_score = stats.norm.ppf(1 - p_value/2)  # approximate Z from p-value
        effect_size = z_score / np.sqrt(n)
    else:
        effect_size = np.nan
    
    return {
        'statistic': statistic,
        'p_value': p_value,
        'mean_diff': mean_diff,
        'median_diff': median_diff,
        'effect_size': effect_size,
        'n_comparisons': n
    }

def perform_pairwise_tests_for_dataset(df, dataset_name, sd_method, exclude_methods):
    """
    Performs pairwise Wilcoxon tests between sd_method and all other methods
    for a specific dataset
    
    Args:
        df: DataFrame with comprehensive summary
        dataset_name: Name of the dataset to filter
        sd_method: The SD method to test (e.g., 'rpsd2all4')
        exclude_methods: List of methods to exclude from comparison
    
    Returns:
        DataFrame with test results
    """
    print(f"\n{'='*80}")
    print(f"Testing: {sd_method} vs all other methods for {dataset_name}")
    print(f"{'='*80}")
    
    # Filter for specific dataset
    df_dataset = df[df['Dataset'] == dataset_name].copy()
    
    # Pivot data to have methods as columns, experiments as rows
    pivot_df = df_dataset.pivot_table(
        index=['Missingness_Type', 'Missingness_Rate'],
        columns='Fixing_Method',
        values='Sum_of_Absolute_Differences'
    ).reset_index()
    
    # Check if SD method exists
    if sd_method not in pivot_df.columns:
        print(f"ERROR: {sd_method} not found in data!")
        return pd.DataFrame()
    
    # Get all methods except the SD method and excluded methods
    all_methods = [col for col in pivot_df.columns 
                   if col not in ['Missingness_Type', 'Missingness_Rate']]
    
    comparison_methods = [m for m in all_methods 
                          if m != sd_method and m not in exclude_methods]
    
    print(f"SD Method: {sd_method}")
    print(f"Dataset: {dataset_name}")
    print(f"Comparing against {len(comparison_methods)} methods")
    print(f"Number of experiments: {len(pivot_df)}")
    
    # Perform tests
    results = []
    sd_values = pivot_df[sd_method].values
    
    for other_method in comparison_methods:
        other_values = pivot_df[other_method].values
        
        # Remove NaN pairs
        valid_mask = ~(np.isnan(sd_values) | np.isnan(other_values))
        sd_valid = sd_values[valid_mask]
        other_valid = other_values[valid_mask]
        
        if len(sd_valid) < 5:  # Need at least 5 pairs
            print(f"  Skipping {other_method} - insufficient data")
            continue
        
        # Perform test
        test_result = wilcoxon_test_with_stats(sd_valid, other_valid)
        
        results.append({
            'dataset': dataset_name,
            'sd_method': sd_method,
            'comparison_method': other_method,
            'n_experiments': test_result['n_comparisons'],
            'mean_diff': test_result['mean_diff'],
            'median_diff': test_result['median_diff'],
            'wilcoxon_statistic': test_result['statistic'],
            'p_value': test_result['p_value'],
            'effect_size': test_result['effect_size']
        })
    
    results_df = pd.DataFrame(results)
    
    if results_df.empty:
        return results_df
    
    # Apply Bonferroni correction
    n_tests = len(results_df)
    results_df['p_value_bonferroni'] = results_df['p_value'] * n_tests
    results_df['p_value_bonferroni'] = results_df['p_value_bonferroni'].clip(upper=1.0)
    
    # Add significance markers
    results_df['significant_0.05'] = results_df['p_value_bonferroni'] < 0.05
    results_df['significant_0.01'] = results_df['p_value_bonferroni'] < 0.01
    results_df['significant_0.001'] = results_df['p_value_bonferroni'] < 0.001
    
    # Add interpretation
    def interpret_diff(row):
        if row['mean_diff'] < 0:
            return f"{row['sd_method']} BETTER (lower difference)"
        elif row['mean_diff'] > 0:
            return f"{row['comparison_method']} BETTER (lower difference)"
        else:
            return "No difference"
    
    results_df['interpretation'] = results_df.apply(interpret_diff, axis=1)
    
    # Sort by p-value
    results_df = results_df.sort_values('p_value_bonferroni')
    
    return results_df

def main():
    print("="*80)
    print("STATISTICAL TESTS: SD METHODS vs OTHER METHODS PER DATASET")
    print("Using Wilcoxon signed-rank test with Bonferroni correction")
    print("="*80)
    
    # Load filtered data
    csv_path = 'reports/differences_summary/comprehensive_summary_filtered.csv'
    print(f"\nLoading data from: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
        print(f"✅ Loaded {len(df)} rows")
    except FileNotFoundError:
        print(f"❌ Error: File not found - {csv_path}")
        print("Run generate_differences_report.py first!")
        return
    
    # Get datasets
    datasets = sorted(df['Dataset'].unique())
    print(f"📊 Datasets: {datasets}")
    
    # Define SD methods and exclusions
    sd_methods = {
        'rpsd2all4': ['mtfsd2all4', 'gafsd2all4', 'specsd2all4'],
        'mtfsd2all4': ['rpsd2all4', 'gafsd2all4', 'specsd2all4'],
        'gafsd2all4': ['rpsd2all4', 'mtfsd2all4', 'specsd2all4'],
        'specsd2all4': ['rpsd2all4', 'mtfsd2all4', 'gafsd2all4']
    }
    
    # Perform tests for each dataset
    for dataset in datasets:
        print(f"\n\n{'#'*80}")
        print(f"# PROCESSING DATASET: {dataset}")
        print(f"{'#'*80}")
        
        # Create output directory
        output_dir = Path(f'reports/statistical_tests_sd_{dataset}')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        all_results = {}
        
        for sd_method, exclude_list in sd_methods.items():
            results_df = perform_pairwise_tests_for_dataset(df, dataset, sd_method, exclude_list)
            
            if not results_df.empty:
                # Save results
                output_file = output_dir / f'{sd_method}_vs_others.csv'
                results_df.to_csv(output_file, index=False)
                print(f"\n✅ Saved results to: {output_file}")
                
                all_results[sd_method] = results_df
            else:
                print(f"\n❌ No results for {sd_method}")
        
        # Create combined summary for this dataset
        if all_results:
            print(f"\n{'='*80}")
            print(f"CREATING COMBINED SUMMARY FOR {dataset}")
            print(f"{'='*80}")
            
            combined_summary = []
            for sd_method, results_df in all_results.items():
                summary = {
                    'Dataset': dataset,
                    'SD_Method': sd_method,
                    'Total_Comparisons': len(results_df),
                    'Significant_0.05': results_df['significant_0.05'].sum(),
                    'Significant_0.01': results_df['significant_0.01'].sum(),
                    'Significant_0.001': results_df['significant_0.001'].sum(),
                    'Better_Than': (results_df['mean_diff'] < 0).sum(),
                    'Worse_Than': (results_df['mean_diff'] > 0).sum(),
                    'Mean_Diff_Avg': results_df['mean_diff'].mean(),
                    'Median_Diff_Avg': results_df['median_diff'].mean()
                }
                combined_summary.append(summary)
            
            combined_df = pd.DataFrame(combined_summary)
            combined_file = output_dir / 'combined_summary.csv'
            combined_df.to_csv(combined_file, index=False)
            
            print(f"\n✅ Saved combined summary to: {combined_file}")
            print(f"\n{combined_df.to_string(index=False)}")
    
    print(f"\n{'='*80}")
    print("✅ ALL TESTS COMPLETED FOR ALL DATASETS!")
    print(f"{'='*80}")
    print(f"\n📁 Results saved in: reports/statistical_tests_sd_{{dataset}}/")

if __name__ == "__main__":
    main()

