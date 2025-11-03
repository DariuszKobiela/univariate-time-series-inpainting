#!/usr/bin/env python3
"""
Statistical tests comparing SD methods vs other methods for filtered data
Using Wilcoxon signed-rank test with Bonferroni correction
Based on MAPE metric from df_filtered_without_lakes_unet.csv
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle

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
        df: DataFrame with MAPE values
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
        index=['dataset', 'missing_data_type', 'missing_rate', 'iteration_nr'],
        columns='fixing_method',
        values='MAPE'
    ).reset_index()
    
    # Check if SD method exists
    if sd_method not in pivot_df.columns:
        print(f"ERROR: {sd_method} not found in data!")
        return pd.DataFrame()
    
    # Get all methods except the SD method and excluded methods
    all_methods = [col for col in pivot_df.columns 
                   if col not in ['dataset', 'missing_data_type', 'missing_rate', 'iteration_nr']]
    
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
    
    # Add interpretation (for MAPE, lower is better)
    def interpret_diff(row):
        if row['mean_diff'] < 0:
            return f"{row['sd_method']} BETTER (lower MAPE)"
        elif row['mean_diff'] > 0:
            return f"{row['comparison_method']} BETTER (lower MAPE)"
        else:
            return "No difference"
    
    results_df['interpretation'] = results_df.apply(interpret_diff, axis=1)
    
    # Sort by p-value
    results_df = results_df.sort_values('p_value_bonferroni')
    
    return results_df

def print_summary(results_df, sd_method):
    """Print summary of test results"""
    print(f"\n{'='*80}")
    print(f"SUMMARY: {sd_method}")
    print(f"{'='*80}")
    
    total_tests = len(results_df)
    sig_001 = (results_df['significant_0.001']).sum()
    sig_01 = (results_df['significant_0.01']).sum()
    sig_05 = (results_df['significant_0.05']).sum()
    
    print(f"\nTotal tests: {total_tests}")
    print(f"Significant at p < 0.001: {sig_001} ({sig_001/total_tests*100:.1f}%)")
    print(f"Significant at p < 0.01: {sig_01} ({sig_01/total_tests*100:.1f}%)")
    print(f"Significant at p < 0.05: {sig_05} ({sig_05/total_tests*100:.1f}%)")
    
    # Show top significant results
    print(f"\nTop 5 most significant differences:")
    top_5 = results_df.nsmallest(5, 'p_value_bonferroni')
    for _, row in top_5.iterrows():
        print(f"  {row['comparison_method']:30s} | p={row['p_value_bonferroni']:.6f} | "
              f"mean_diff={row['mean_diff']:.6f} | {row['interpretation']}")

def create_discrete_pvalue_heatmap(all_data, output_dir):
    """Create heatmap with discrete color categories for p-values"""
    
    # Get all comparison methods
    all_methods = set()
    for df in all_data.values():
        all_methods.update(df['comparison_method'].tolist())
    all_methods = sorted(list(all_methods))
    
    # Create matrix
    sd_methods = ['rpsd2all4', 'mtfsd2all4', 'gafsd2all4', 'specsd2all4']
    matrix = np.zeros((len(all_methods), len(sd_methods)))  # Transposed!
    
    for j, sd_method in enumerate(sd_methods):
        if sd_method not in all_data:
            continue
        df = all_data[sd_method]
        for i, comp_method in enumerate(all_methods):
            row = df[df['comparison_method'] == comp_method]
            if not row.empty:
                p_val = row['p_value_bonferroni'].values[0]
                matrix[i, j] = p_val
            else:
                matrix[i, j] = np.nan
    
    # Convert to discrete categories
    # 0 = p < 0.01 (highly significant)
    # 1 = 0.01 <= p < 0.05 (significant)
    # 2 = p >= 0.05 (not significant)
    discrete_matrix = np.zeros_like(matrix)
    discrete_matrix[matrix < 0.01] = 0
    discrete_matrix[(matrix >= 0.01) & (matrix < 0.05)] = 1
    discrete_matrix[matrix >= 0.05] = 2
    
    # Create custom colormap
    colors = ['#d73027', '#fee090', '#91cf60']  # Red, Yellow, Green
    cmap = ListedColormap(colors)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 10))
    
    # Create heatmap
    im = ax.imshow(discrete_matrix, cmap=cmap, aspect='auto', vmin=0, vmax=2)
    
    # Set ticks
    ax.set_xticks(np.arange(len(sd_methods)))
    ax.set_yticks(np.arange(len(all_methods)))
    ax.set_xticklabels(sd_methods, fontsize=11, fontweight='bold')
    ax.set_yticklabels(all_methods, fontsize=10)
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    
    # Add grid
    ax.set_xticks(np.arange(len(sd_methods)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(all_methods)) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=2)
    
    # Add text annotations with actual p-values
    for i in range(len(all_methods)):
        for j in range(len(sd_methods)):
            p_val = matrix[i, j]
            if not np.isnan(p_val):
                if p_val < 0.001:
                    text = '***'
                    color = 'white'
                elif p_val < 0.01:
                    text = '**'
                    color = 'white'
                elif p_val < 0.05:
                    text = '*'
                    color = 'black'
                else:
                    text = 'ns'
                    color = 'black'
                
                ax.text(j, i, text, ha='center', va='center', 
                       color=color, fontsize=12, fontweight='bold')
    
    # Labels
    ax.set_xlabel('SD Methods', fontsize=13, fontweight='bold', labelpad=10)
    ax.set_ylabel('Comparison Methods', fontsize=13, fontweight='bold', labelpad=10)
    
    # Title
    plt.title('Statistical Significance: Inpainting Methods vs Other Methods\n(Bonferroni-corrected p-values)', 
              fontsize=14, fontweight='bold', pad=20)
    
    # Create custom legend
    legend_elements = [
        Rectangle((0, 0), 1, 1, fc='#d73027', edgecolor='black', label='p < 0.01 (highly significant)'),
        Rectangle((0, 0), 1, 1, fc='#fee090', edgecolor='black', label='0.01 ≤ p < 0.05 (significant)'),
        Rectangle((0, 0), 1, 1, fc='#91cf60', edgecolor='black', label='p ≥ 0.05 (not significant)')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1), 
             frameon=True, fontsize=10)
    
    plt.tight_layout()
    
    output_file = output_dir / 'heatmap_pvalues_discrete_filtered.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    
    plt.close()
    
    return output_file

def create_pvalue_heatmap_log_vertical(all_data, output_dir):
    """Create heatmap of p-values in log scale (vertical version)"""
    
    # Get all comparison methods
    all_methods = set()
    for df in all_data.values():
        all_methods.update(df['comparison_method'].tolist())
    all_methods = sorted(list(all_methods))
    
    # Create matrix (transposed!)
    sd_methods = ['rpsd2all4', 'mtfsd2all4', 'gafsd2all4', 'specsd2all4']
    matrix = np.zeros((len(all_methods), len(sd_methods)))  # Transposed dimensions
    
    for j, sd_method in enumerate(sd_methods):
        if sd_method not in all_data:
            continue
        df = all_data[sd_method]
        for i, comp_method in enumerate(all_methods):
            row = df[df['comparison_method'] == comp_method]
            if not row.empty:
                matrix[i, j] = row['p_value_bonferroni'].values[0]
            else:
                matrix[i, j] = np.nan
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(8, 10))
    
    # Use log scale for better visualization
    matrix_log = np.where(matrix == 0, 1e-10, matrix)
    matrix_log = -np.log10(matrix_log)
    
    sns.heatmap(matrix_log, 
                xticklabels=sd_methods,
                yticklabels=all_methods,
                cmap='RdYlGn',
                center=1.301,
                vmin=0,
                vmax=6,
                annot=False,
                cbar_kws={'label': '-log10(p-value Bonferroni)'},
                linewidths=0.5,
                linecolor='white')
    
    plt.title('Statistical Significance: Inpainting Methods vs Other Methods \n(-log10 of Bonferroni-corrected p-values)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('SD Methods', fontsize=12, fontweight='bold')
    plt.ylabel('Comparison Methods', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    
    # Add colorbar labels
    cbar = ax.collections[0].colorbar
    cbar.ax.axhline(y=1.301, color='blue', linestyle='--', linewidth=2, alpha=0.7)
    cbar.ax.axhline(y=2, color='blue', linestyle='--', linewidth=2, alpha=0.7)
    cbar.ax.axhline(y=3, color='blue', linestyle='--', linewidth=2, alpha=0.7)
    cbar.ax.text(3.5, 1.301, 'p=0.05', va='center', fontsize=9, color='blue')
    cbar.ax.text(3.5, 2, 'p=0.01', va='center', fontsize=9, color='blue')
    cbar.ax.text(3.5, 3, 'p=0.001', va='center', fontsize=9, color='blue')
    
    plt.tight_layout()
    
    output_file = output_dir / 'heatmap_pvalues_log_vertical_filtered.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    
    plt.close()

def create_mean_difference_heatmap(all_data, output_dir):
    """Create heatmap of mean MAPE differences"""
    
    # Get all comparison methods
    all_methods = set()
    for df in all_data.values():
        all_methods.update(df['comparison_method'].tolist())
    all_methods = sorted(list(all_methods))
    
    # Create matrix (transposed!)
    sd_methods = ['rpsd2all4', 'mtfsd2all4', 'gafsd2all4', 'specsd2all4']
    matrix = np.zeros((len(all_methods), len(sd_methods)))
    
    for j, sd_method in enumerate(sd_methods):
        if sd_method not in all_data:
            continue
        df = all_data[sd_method]
        for i, comp_method in enumerate(all_methods):
            row = df[df['comparison_method'] == comp_method]
            if not row.empty:
                matrix[i, j] = row['mean_diff'].values[0]
            else:
                matrix[i, j] = np.nan
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(8, 10))
    
    sns.heatmap(matrix, 
                xticklabels=sd_methods,
                yticklabels=all_methods,
                cmap='RdYlGn_r',
                center=0,
                annot=False,
                cbar_kws={'label': 'Mean MAPE Difference\nPositive = SD method is worse'},
                linewidths=0.5,
                linecolor='white',
                fmt='.4f')
    
    plt.title('Mean MAPE Differences: Inpainting Methods vs Other Methods \n(Positive = SD method has higher MAPE = worse)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('SD Methods', fontsize=12, fontweight='bold')
    plt.ylabel('Comparison Methods', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    output_file = output_dir / 'heatmap_mean_differences_filtered.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    
    plt.close()

def main():
    print("="*80)
    print("STATISTICAL TESTS: SD METHODS vs OTHER METHODS (FILTERED DATA)")
    print("Using Wilcoxon signed-rank test with Bonferroni correction")
    print("Metric: MAPE (Mean Absolute Percentage Error)")
    print("="*80)
    
    # Load filtered data
    csv_path = 'results/quick_experiment/df_filtered_without_lakes_unet.csv'
    print(f"\nLoading data from: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
        print(f"✅ Loaded {len(df)} rows")
        print(f"   Datasets: {sorted(df['dataset'].unique())}")
        print(f"   Methods: {sorted(df['fixing_method'].unique())}")
    except FileNotFoundError:
        print(f"❌ Error: File not found - {csv_path}")
        return
    
    # Create output directory
    output_dir = Path('reports/statistical_tests_filtered')
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
    
    # Create visualizations
    if all_results:
        print(f"\n{'='*80}")
        print("CREATING VISUALIZATIONS")
        print(f"{'='*80}")
        
        print("\n1. Discrete p-value heatmap...")
        create_discrete_pvalue_heatmap(all_results, output_dir)
        
        print("\n2. P-value heatmap (log scale - vertical)...")
        create_pvalue_heatmap_log_vertical(all_results, output_dir)
        
        print("\n3. Mean difference heatmap...")
        create_mean_difference_heatmap(all_results, output_dir)
    
    print("\n" + "="*80)
    print("✅ ALL TESTS AND VISUALIZATIONS COMPLETED!")
    print("="*80)
    print(f"\n📁 All saved in: {output_dir}/")
    print("\nGenerated files:")
    print("  CSV files:")
    for sd_method in sd_methods.keys():
        print(f"    - {sd_method}_vs_others.csv")
    print("  Visualization files:")
    print("    - heatmap_pvalues_discrete_filtered.png")
    print("    - heatmap_pvalues_log_vertical_filtered.png")
    print("    - heatmap_mean_differences_filtered.png")

if __name__ == "__main__":
    main()

