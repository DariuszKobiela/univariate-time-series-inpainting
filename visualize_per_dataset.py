#!/usr/bin/env python3
"""
Visualization of statistical test results for SD methods PER DATASET
Creates heatmaps for each dataset separately
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_test_results_for_dataset(dataset_name):
    """Load all statistical test results for a specific dataset"""
    results_dir = Path(f'reports/statistical_tests_sd_{dataset_name}')
    
    sd_methods = ['rpsd2all4', 'mtfsd2all4', 'gafsd2all4', 'specsd2all4']
    all_data = {}
    
    for sd_method in sd_methods:
        file_path = results_dir / f'{sd_method}_vs_others.csv'
        if file_path.exists():
            all_data[sd_method] = pd.read_csv(file_path)
        else:
            print(f"Warning: {file_path} not found")
    
    return all_data

def create_combined_plot_for_dataset(all_data, dataset_name, output_dir):
    """Create combined plot with both p-values and mean differences for a specific dataset"""
    
    # Get all comparison methods
    all_methods = set()
    for df in all_data.values():
        all_methods.update(df['comparison_method'].tolist())
    all_methods = sorted(list(all_methods))
    
    sd_methods = ['rpsd2all4', 'mtfsd2all4', 'gafsd2all4', 'specsd2all4']
    sd_methods_display = ['rpsd', 'mtfsd', 'gafsd', 'specsd']  # Short names for display
    
    # Create matrices
    pvalue_matrix = np.zeros((len(sd_methods), len(all_methods)))
    diff_matrix = np.zeros((len(sd_methods), len(all_methods)))
    
    for i, sd_method in enumerate(sd_methods):
        if sd_method not in all_data:
            continue
        df = all_data[sd_method]
        for j, comp_method in enumerate(all_methods):
            row = df[df['comparison_method'] == comp_method]
            if not row.empty:
                pvalue_matrix[i, j] = row['p_value_bonferroni'].values[0]
                diff_matrix[i, j] = row['mean_diff'].values[0]
            else:
                pvalue_matrix[i, j] = np.nan
                diff_matrix[i, j] = np.nan
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: P-values
    pvalue_matrix_log = np.where(pvalue_matrix == 0, 1e-10, pvalue_matrix)
    pvalue_matrix_log = -np.log10(pvalue_matrix_log)
    
    im1 = sns.heatmap(pvalue_matrix_log, 
                xticklabels=all_methods,
                yticklabels=sd_methods_display,
                cmap='RdYlGn',
                center=1.301,
                vmin=0,
                vmax=6,
                annot=False,
                cbar_kws={'label': '-log10(p-value)'},
                linewidths=0.5,
                linecolor='white',
                ax=ax1)
    
    # Add reference lines to colorbar for p-value thresholds
    cbar1 = ax1.collections[0].colorbar
    cbar1.ax.axhline(y=1.301, color='blue', linestyle='--', linewidth=2, alpha=0.7)
    cbar1.ax.axhline(y=2, color='blue', linestyle='--', linewidth=2, alpha=0.7)
    cbar1.ax.axhline(y=3, color='blue', linestyle='--', linewidth=2, alpha=0.7)
    # Position text to the left of the colorbar to avoid overlap
    cbar1.ax.text(-0.5, 1.301, 'p=0.05', va='center', ha='right', fontsize=8, color='blue', fontweight='bold')
    cbar1.ax.text(-0.5, 2, 'p=0.01', va='center', ha='right', fontsize=8, color='blue', fontweight='bold')
    cbar1.ax.text(-0.5, 3, 'p=0.001', va='center', ha='right', fontsize=8, color='blue', fontweight='bold')
    
    # Add annotations to Plot 1: significance + direction
    for i in range(len(sd_methods)):
        for j in range(len(all_methods)):
            p_val = pvalue_matrix[i, j]
            mean_diff = diff_matrix[i, j]
            
            if not np.isnan(p_val) and not np.isnan(mean_diff):
                # Determine direction
                if mean_diff > 0:
                    direction = '▲'  # classical method better
                    color = 'white' if p_val < 0.01 else 'black'
                elif mean_diff < 0:
                    direction = '▼'  # inpainting method better
                    color = 'white' if p_val < 0.01 else 'black'
                else:
                    direction = '='
                    color = 'black'
                
                # Significance level
                if p_val < 0.001:
                    sig = '***'
                elif p_val < 0.01:
                    sig = '**'
                elif p_val < 0.05:
                    sig = '*'
                else:
                    sig = 'ns'
                
                text = f'{sig}\n{direction}'
                ax1.text(j + 0.5, i + 0.5, text, ha='center', va='center',
                        color=color, fontsize=9, fontweight='bold')
    
    ax1.set_title('A) Statistical Significance (-log10 of p-values)', fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlabel('')
    ax1.set_ylabel('Inpainting Methods', fontsize=11, fontweight='bold')
    ax1.set_xticklabels([])
    
    # Plot 2: Mean differences
    diff_matrix_millions = diff_matrix / 1_000_000
    
    im2 = sns.heatmap(diff_matrix_millions, 
                xticklabels=all_methods,
                yticklabels=sd_methods_display,
                cmap='RdYlGn_r',
                center=0,
                annot=False,
                cbar_kws={'label': 'Mean Abs Diff (M)'},
                linewidths=0.5,
                linecolor='white',
                ax=ax2)
    
    # Add annotations to Plot 2: values + direction
    for i in range(len(sd_methods)):
        for j in range(len(all_methods)):
            diff_val = diff_matrix_millions[i, j]
            
            if not np.isnan(diff_val):
                # Determine direction
                if diff_val > 0:
                    direction = '▲'  # classical method better
                elif diff_val < 0:
                    direction = '▼'  # inpainting method better
                else:
                    direction = '='
                
                # Format the value
                if abs(diff_val) >= 100:
                    text = f'{diff_val:.0f}M\n{direction}'
                elif abs(diff_val) >= 10:
                    text = f'{diff_val:.1f}M\n{direction}'
                elif abs(diff_val) >= 0.01:
                    text = f'{diff_val:.2f}M\n{direction}'
                else:
                    text = f'{diff_val:.3f}M\n{direction}'
                
                # Color based on value
                if abs(diff_val) < 50:
                    color = 'black'
                else:
                    color = 'white'
                
                ax2.text(j + 0.5, i + 0.5, text, ha='center', va='center',
                        color=color, fontsize=8, fontweight='bold')
    
    ax2.set_title('B) Mean Absolute Differences (Millions)', fontsize=12, fontweight='bold', pad=10)
    ax2.set_xlabel('Classical Methods', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Inpainting Methods', fontsize=11, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    
    # Add title
    plt.suptitle(f'Statistical Comparison: Inpainting vs Classical Methods - {dataset_name.upper()}', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Add legend at the top, below the title - one clean box
    legend_text = ('Statistical significance: *** p<0.001   ** p<0.01   * p<0.05   ns = not significant\n'
                   'Performance indicators: ▲ = Classical method better   ▼ = Inpainting method better')
    
    fig.text(0.5, 0.935, legend_text, ha='center', fontsize=10, 
             bbox=dict(boxstyle='round,pad=0.8', facecolor='lightyellow', 
                      alpha=0.9, edgecolor='darkgray', linewidth=1.5),
             verticalalignment='top')
    
    plt.tight_layout(rect=[0, 0, 1, 0.91])  # Make room for title and legend at top
    
    output_file = output_dir / f'heatmap_combined_{dataset_name}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    
    plt.close()

def main():
    print("="*80)
    print("VISUALIZING STATISTICAL TEST RESULTS PER DATASET")
    print("="*80)
    
    datasets = ['boiler', 'pump', 'vibr']
    
    for dataset in datasets:
        print(f"\n{'#'*80}")
        print(f"# Processing dataset: {dataset}")
        print(f"{'#'*80}")
        
        # Load test results
        print(f"\nLoading test results for {dataset}...")
        all_data = load_test_results_for_dataset(dataset)
        
        if not all_data:
            print(f"❌ No test results found for {dataset}")
            continue
        
        print(f"✅ Loaded results for {len(all_data)} SD methods")
        
        # Create output directory
        output_dir = Path(f'reports/statistical_tests_sd_{dataset}')
        
        # Create combined plot
        print(f"\n📊 Creating combined heatmap...")
        create_combined_plot_for_dataset(all_data, dataset, output_dir)
    
    print("\n" + "="*80)
    print("✅ ALL VISUALIZATIONS COMPLETED!")
    print("="*80)
    print("\n📁 Generated files:")
    for dataset in datasets:
        print(f"  - reports/statistical_tests_sd_{dataset}/heatmap_combined_{dataset}.png")

if __name__ == "__main__":
    main()

