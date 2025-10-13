#!/usr/bin/env python3
"""
Script to calculate statistical significance of different fixing methods
compared to the original data for each dataset.
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Configuration
INPUT_FILE = 'results/quick_experiment/df_final.csv'
OUTPUT_DIR = 'importance_results'
OUTPUT_CSV = 'importance.csv'

# Value columns to compare
VALUE_COLS = [f'val_{i}' for i in range(1, 11)]


def read_data():
    """Read the df_final.csv file."""
    print(f"Reading data from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} rows")
    return df


def get_unique_datasets(df):
    """Get list of unique datasets."""
    datasets = df['dataset'].unique()
    print(f"Found {len(datasets)} datasets: {', '.join(datasets)}")
    return datasets


def get_fixing_methods(df, dataset):
    """Get all fixing methods for a dataset (excluding 'original')."""
    methods = df[(df['dataset'] == dataset) & (df['fixing_method'] != 'original')]['fixing_method'].unique()
    return sorted(methods)


def get_original_values(df, dataset):
    """Get the values from the original row for a dataset."""
    original_rows = df[(df['dataset'] == dataset) & (df['fixing_method'] == 'original')]
    
    if len(original_rows) == 0:
        raise ValueError(f"No original data found for dataset: {dataset}")
    
    # Take the first row (they should all be the same)
    original_row = original_rows.iloc[0]
    original_values = original_row[VALUE_COLS].values.astype(float)
    
    return original_values


def get_method_values(df, dataset, fixing_method):
    """Get all values for a specific fixing method and dataset."""
    method_rows = df[(df['dataset'] == dataset) & (df['fixing_method'] == fixing_method)]
    
    if len(method_rows) == 0:
        raise ValueError(f"No data found for dataset: {dataset}, method: {fixing_method}")
    
    # Collect all values from all rows
    all_values = []
    for _, row in method_rows.iterrows():
        values = row[VALUE_COLS].values.astype(float)
        all_values.extend(values)
    
    return np.array(all_values)


def perform_ttest(original_values, method_values):
    """
    Perform independent t-test comparing original values to method values.
    Returns the p-value.
    """
    # original_values: 10 values from one row
    # method_values: all values from multiple rows (n_rows * 10)
    
    # Perform independent samples t-test
    t_stat, p_value = stats.ttest_ind(original_values, method_values)
    
    return p_value


def calculate_significance(df):
    """Calculate statistical significance for all methods across all datasets."""
    datasets = get_unique_datasets(df)
    
    results = []
    
    for dataset in datasets:
        print(f"\nProcessing dataset: {dataset}")
        
        # Get original values for this dataset
        original_values = get_original_values(df, dataset)
        print(f"  Original values shape: {original_values.shape}")
        
        # Get all fixing methods for this dataset
        fixing_methods = get_fixing_methods(df, dataset)
        print(f"  Found {len(fixing_methods)} fixing methods")
        
        for method in fixing_methods:
            # Get all values for this method
            method_values = get_method_values(df, dataset, method)
            
            # Perform t-test
            p_value = perform_ttest(original_values, method_values)
            
            # Store result
            results.append({
                'dataset': dataset,
                'fixing_method': method,
                'p_value': p_value,
                'n_original': len(original_values),
                'n_method': len(method_values)
            })
            
            print(f"  {method}: p-value = {p_value:.6f}")
    
    return pd.DataFrame(results)


def create_plots(results_df, output_dir):
    """Create bar plots for each dataset showing p-values."""
    datasets = results_df['dataset'].unique()
    
    for dataset in datasets:
        print(f"Creating plot for dataset: {dataset}")
        
        # Filter data for this dataset
        dataset_data = results_df[results_df['dataset'] == dataset].copy()
        
        # Sort by p-value for better visualization
        dataset_data = dataset_data.sort_values('p_value')
        
        # Create figure
        plt.figure(figsize=(14, 8))
        
        # Create bar plot
        x_pos = np.arange(len(dataset_data))
        bars = plt.bar(x_pos, dataset_data['p_value'], alpha=0.7, color='steelblue')
        
        # Add horizontal lines for significance levels
        plt.axhline(y=0.05, color='orange', linestyle='--', linewidth=2, label='α = 0.05')
        plt.axhline(y=0.01, color='red', linestyle='--', linewidth=2, label='α = 0.01')
        
        # Color bars based on significance
        for i, (idx, row) in enumerate(dataset_data.iterrows()):
            if row['p_value'] < 0.01:
                bars[i].set_color('darkgreen')
            elif row['p_value'] < 0.05:
                bars[i].set_color('lightgreen')
        
        # Customize plot
        plt.xlabel('Fixing Method', fontsize=12, fontweight='bold')
        plt.ylabel('p-value', fontsize=12, fontweight='bold')
        plt.title(f'Statistical Significance of Fixing Methods for Dataset: {dataset}', 
                  fontsize=14, fontweight='bold')
        plt.xticks(x_pos, dataset_data['fixing_method'], rotation=45, ha='right')
        plt.legend(fontsize=10)
        plt.grid(axis='y', alpha=0.3, linestyle=':')
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(output_dir, f'{dataset}_significance.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved plot to: {plot_path}")


def main():
    """Main function."""
    print("=" * 70)
    print("Statistical Significance Analysis")
    print("=" * 70)
    
    # Read data
    df = read_data()
    
    # Calculate significance
    print("\nCalculating statistical significance...")
    results_df = calculate_significance(df)
    
    # Create output directory
    output_dir = OUTPUT_DIR
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"\nCreated output directory: {output_dir}")
    
    # Save results to CSV
    csv_path = OUTPUT_CSV
    results_df.to_csv(csv_path, index=False)
    print(f"Saved results to: {csv_path}")
    
    # Create plots
    print("\nCreating plots...")
    create_plots(results_df, output_dir)
    
    # Print summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Total comparisons: {len(results_df)}")
    print(f"Significant at α=0.05: {len(results_df[results_df['p_value'] < 0.05])}")
    print(f"Significant at α=0.01: {len(results_df[results_df['p_value'] < 0.01])}")
    print("\nAnalysis complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()

