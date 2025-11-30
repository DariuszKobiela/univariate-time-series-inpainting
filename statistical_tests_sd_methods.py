#!/usr/bin/env python3
"""
Statistical tests comparing Stable Diffusion methods vs other methods
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

def perform_pairwise_tests(df, sd_method, exclude_methods):
    """
    Performs pairwise Wilcoxon tests between sd_method and all other methods
    
    Args:
        df: DataFrame with comprehensive summary
        sd_method: The SD method to test (e.g., 'rpsd2all4')
        exclude_methods: List of methods to exclude from comparison
    
    Returns:
        DataFrame with test results
    """
    print(f"\n{'='*80}")
    print(f"Testing: {sd_method} vs all other methods")
    print(f"{'='*80}")
    
    # Pivot data to have methods as columns, experiments as rows
    pivot_df = df.pivot_table(
        index=['Dataset', 'Missingness_Type', 'Missingness_Rate'],
        columns='Fixing_Method',
        values='Sum_of_Absolute_Differences'
    ).reset_index()
    
    # Check if SD method exists
    if sd_method not in pivot_df.columns:
        print(f"ERROR: {sd_method} not found in data!")
        return pd.DataFrame()
    
    # Get all methods except the SD method and excluded methods
    all_methods = [col for col in pivot_df.columns 
                   if col not in ['Dataset', 'Missingness_Type', 'Missingness_Rate']]
    
    comparison_methods = [m for m in all_methods 
                          if m != sd_method and m not in exclude_methods]
    
    print(f"SD Method: {sd_method}")
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

def print_summary(results_df, sd_method):
    """Print summary statistics"""
    print(f"\n{'='*80}")
    print(f"SUMMARY: {sd_method}")
    print(f"{'='*80}")
    
    total_comparisons = len(results_df)
    sig_05 = results_df['significant_0.05'].sum()
    sig_01 = results_df['significant_0.01'].sum()
    sig_001 = results_df['significant_0.001'].sum()
    
    better_than = (results_df['mean_diff'] < 0).sum()
    worse_than = (results_df['mean_diff'] > 0).sum()
    
    print(f"\nTotal comparisons: {total_comparisons}")
    print(f"Significant at α=0.05: {sig_05} ({sig_05/total_comparisons*100:.1f}%)")
    print(f"Significant at α=0.01: {sig_01} ({sig_01/total_comparisons*100:.1f}%)")
    print(f"Significant at α=0.001: {sig_001} ({sig_001/total_comparisons*100:.1f}%)")
    
    print(f"\nPerformance comparison:")
    print(f"  {sd_method} performs BETTER than: {better_than} methods")
    print(f"  {sd_method} performs WORSE than: {worse_than} methods")
    
    # Top 5 methods that are significantly better/worse
    print(f"\n🏆 TOP 5 methods significantly BETTER than {sd_method} (p<0.05):")
    better = results_df[
        (results_df['mean_diff'] > 0) & 
        (results_df['significant_0.05'])
    ].head(5)
    
    if len(better) > 0:
        for idx, row in better.iterrows():
            print(f"  • {row['comparison_method']}")
            print(f"    Mean diff: {row['mean_diff']:.0f}, p-value: {row['p_value_bonferroni']:.6f}")
    else:
        print("  None")
    
    print(f"\n⚠️ TOP 5 methods significantly WORSE than {sd_method} (p<0.05):")
    worse = results_df[
        (results_df['mean_diff'] < 0) & 
        (results_df['significant_0.05'])
    ].head(5)
    
    if len(worse) > 0:
        for idx, row in worse.iterrows():
            print(f"  • {row['comparison_method']}")
            print(f"    Mean diff: {row['mean_diff']:.0f}, p-value: {row['p_value_bonferroni']:.6f}")
    else:
        print("  None")

def main():
    print("="*80)
    print("STATISTICAL TESTS: SD METHODS vs OTHER METHODS")
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
    
    # Create output directory
    output_dir = Path('reports/statistical_tests_sd')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define SD methods and exclusions
    sd_methods = {
        'rpsd2all4': ['mtfsd2all4', 'gafsd2all4', 'specsd2all4'],
        'mtfsd2all4': ['rpsd2all4', 'gafsd2all4', 'specsd2all4'],
        'gafsd2all4': ['rpsd2all4', 'mtfsd2all4', 'specsd2all4'],
        'specsd2all4': ['rpsd2all4', 'mtfsd2all4', 'gafsd2all4']
    }
    
    # Perform tests for each SD method
    all_results = {}
    
    for sd_method, exclude_list in sd_methods.items():
        results_df = perform_pairwise_tests(df, sd_method, exclude_list)
        
        if not results_df.empty:
            # Save results
            output_file = output_dir / f'{sd_method}_vs_others.csv'
            results_df.to_csv(output_file, index=False)
            print(f"\n✅ Saved results to: {output_file}")
            
            # Print summary
            print_summary(results_df, sd_method)
            
            all_results[sd_method] = results_df
        else:
            print(f"\n❌ No results for {sd_method}")
    
    # Create combined summary
    print(f"\n{'='*80}")
    print("CREATING COMBINED SUMMARY")
    print(f"{'='*80}")
    
    combined_summary = []
    for sd_method, results_df in all_results.items():
        summary = {
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
    print("\nCombined Summary:")
    print(combined_df.to_string(index=False))
    
    print(f"\n{'='*80}")
    print("✅ ALL TESTS COMPLETED!")
    print(f"{'='*80}")
    print(f"\n📁 All results saved in: {output_dir}/")
    print("\nGenerated files:")
    print("  1. rpsd2all4_vs_others.csv")
    print("  2. mtfsd2all4_vs_others.csv")
    print("  3. gafsd2all4_vs_others.csv")
    print("  4. specsd2all4_vs_others.csv")
    print("  5. combined_summary.csv")

if __name__ == "__main__":
    main()



