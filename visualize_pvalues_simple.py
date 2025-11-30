#!/usr/bin/env python3
"""
Simple visualization of p-values with discrete color categories
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.colors import ListedColormap

def load_test_results():
    """Load all statistical test results"""
    results_dir = Path('reports/statistical_tests_sd')
    
    sd_methods = ['rpsd2all4', 'mtfsd2all4', 'gafsd2all4', 'specsd2all4']
    all_data = {}
    
    for sd_method in sd_methods:
        file_path = results_dir / f'{sd_method}_vs_others.csv'
        if file_path.exists():
            all_data[sd_method] = pd.read_csv(file_path)
    
    return all_data

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
    
    # Add text annotations with p-values and direction
    for i in range(len(all_methods)):
        for j in range(len(sd_methods)):
            p_val = matrix[i, j]
            if not np.isnan(p_val):
                # Get mean_diff to determine direction
                sd_method = sd_methods[j]
                comp_method = all_methods[i]
                df = all_data[sd_method]
                row = df[df['comparison_method'] == comp_method]
                
                if not row.empty:
                    mean_diff = row['mean_diff'].values[0]
                    
                    # Determine direction
                    if mean_diff > 0:
                        direction = '▲'  # comparison method better (SD worse)
                    elif mean_diff < 0:
                        direction = '▼'  # SD method better
                    else:
                        direction = '='
                    
                    # Create text with significance
                    if p_val < 0.001:
                        sig_text = '***'
                        color = 'white'
                    elif p_val < 0.01:
                        sig_text = '**'
                        color = 'white'
                    elif p_val < 0.05:
                        sig_text = '*'
                        color = 'black'
                    else:
                        sig_text = 'ns'
                        color = 'black'
                    
                    text = f'{sig_text}\n{direction}'
                    
                    ax.text(j, i, text, ha='center', va='center', 
                           color=color, fontsize=10, fontweight='bold')
    
    # Labels
    ax.set_xlabel('SD Methods', fontsize=13, fontweight='bold', labelpad=10)
    ax.set_ylabel('Comparison Methods', fontsize=13, fontweight='bold', labelpad=10)
    
    # Title
    plt.title('Statistical Significance: Inpainting Methods vs Other Methods\n(Bonferroni-corrected p-values)\n▲ = Comparison method better | ▼ = SD method better', 
              fontsize=13, fontweight='bold', pad=20)
    
    # Create custom legend
    from matplotlib.patches import Rectangle
    legend_elements = [
        Rectangle((0, 0), 1, 1, fc='#d73027', edgecolor='black', label='p < 0.01 (highly significant)'),
        Rectangle((0, 0), 1, 1, fc='#fee090', edgecolor='black', label='0.01 ≤ p < 0.05 (significant)'),
        Rectangle((0, 0), 1, 1, fc='#91cf60', edgecolor='black', label='p ≥ 0.05 (not significant)')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1), 
             frameon=True, fontsize=10)
    
    plt.tight_layout()
    
    output_file = output_dir / 'heatmap_pvalues_discrete.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    
    plt.close()
    
    return output_file

def create_pvalue_heatmap_log(all_data, output_dir):
    """Create heatmap of p-values in log scale (horizontal version)"""
    
    # Get all comparison methods
    all_methods = set()
    for df in all_data.values():
        all_methods.update(df['comparison_method'].tolist())
    all_methods = sorted(list(all_methods))
    
    # Create matrix
    sd_methods = ['rpsd2all4', 'mtfsd2all4', 'gafsd2all4', 'specsd2all4']
    matrix = np.zeros((len(sd_methods), len(all_methods)))
    
    for i, sd_method in enumerate(sd_methods):
        df = all_data[sd_method]
        for j, comp_method in enumerate(all_methods):
            row = df[df['comparison_method'] == comp_method]
            if not row.empty:
                matrix[i, j] = row['p_value_bonferroni'].values[0]
            else:
                matrix[i, j] = np.nan
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Use log scale for better visualization
    matrix_log = np.where(matrix == 0, 1e-10, matrix)
    matrix_log = -np.log10(matrix_log)
    
    sns.heatmap(matrix_log, 
                xticklabels=all_methods,
                yticklabels=sd_methods,
                cmap='RdYlGn',
                center=1.301,
                vmin=0,
                vmax=6,
                annot=False,
                cbar_kws={'label': '-log10(p-value Bonferroni)'},
                linewidths=0.5,
                linecolor='white')
    
    plt.title('Statistical Significance: Inpainting Methods vs Other Methods\n(-log10 of Bonferroni-corrected p-values)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Comparison Methods', fontsize=12, fontweight='bold')
    plt.ylabel('Inpainting Methods', fontsize=12, fontweight='bold')
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
    
    output_file = output_dir / 'heatmap_pvalues_log.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    
    plt.close()

def create_pvalue_heatmap_log_vertical(all_data, output_dir):
    """Create heatmap of p-values in log scale (vertical version - rotated 90 degrees)"""
    
    # Get all comparison methods
    all_methods = set()
    for df in all_data.values():
        all_methods.update(df['comparison_method'].tolist())
    all_methods = sorted(list(all_methods))
    
    # Create matrix (transposed!)
    sd_methods = ['rpsd2all4', 'mtfsd2all4', 'gafsd2all4', 'specsd2all4']
    matrix = np.zeros((len(all_methods), len(sd_methods)))  # Transposed dimensions
    
    for j, sd_method in enumerate(sd_methods):
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
    
    plt.title('Statistical Significance: Inpainting Methods vs Other Methods\n(-log10 of Bonferroni-corrected p-values)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Inpainting Methods', fontsize=12, fontweight='bold')
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
    
    output_file = output_dir / 'heatmap_pvalues_log_vertical.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    
    plt.close()

def create_pvalue_heatmap_annotated(all_data, output_dir):
    """Create heatmap with actual p-values annotated"""
    
    # Get all comparison methods
    all_methods = set()
    for df in all_data.values():
        all_methods.update(df['comparison_method'].tolist())
    all_methods = sorted(list(all_methods))
    
    # Create matrix
    sd_methods = ['rpsd2all4', 'mtfsd2all4', 'gafsd2all4', 'specsd2all4']
    matrix = np.zeros((len(sd_methods), len(all_methods)))
    annotations = []
    
    for i, sd_method in enumerate(sd_methods):
        df = all_data[sd_method]
        row_annotations = []
        for j, comp_method in enumerate(all_methods):
            row = df[df['comparison_method'] == comp_method]
            if not row.empty:
                p_val = row['p_value_bonferroni'].values[0]
                matrix[i, j] = p_val
                
                # Create annotation
                if p_val < 0.001:
                    row_annotations.append('***')
                elif p_val < 0.01:
                    row_annotations.append('**')
                elif p_val < 0.05:
                    row_annotations.append('*')
                else:
                    row_annotations.append('ns')
            else:
                matrix[i, j] = np.nan
                row_annotations.append('')
        annotations.append(row_annotations)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Use log scale for better visualization
    matrix_log = np.where(matrix == 0, 1e-10, matrix)
    matrix_log = -np.log10(matrix_log)
    
    sns.heatmap(matrix_log, 
                xticklabels=all_methods,
                yticklabels=sd_methods,
                cmap='RdYlGn',
                center=1.301,
                vmin=0,
                vmax=6,
                annot=np.array(annotations),
                fmt='s',
                cbar_kws={'label': '-log10(p-value Bonferroni)'},
                linewidths=0.5,
                linecolor='white',
                annot_kws={'fontsize': 10, 'weight': 'bold'})
    
    plt.title('Statistical Significance with Annotations\n(*** p<0.001, ** p<0.01, * p<0.05, ns = not significant)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Comparison Methods', fontsize=12, fontweight='bold')
    plt.ylabel('Inpainting Methods', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    output_file = output_dir / 'heatmap_pvalues_annotated.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    
    plt.close()

def create_mean_difference_heatmap(all_data, output_dir):
    """Create heatmap of mean differences"""
    
    # Get all comparison methods
    all_methods = set()
    for df in all_data.values():
        all_methods.update(df['comparison_method'].tolist())
    all_methods = sorted(list(all_methods))
    
    # Create matrix
    sd_methods = ['rpsd2all4', 'mtfsd2all4', 'gafsd2all4', 'specsd2all4']
    matrix = np.zeros((len(sd_methods), len(all_methods)))
    
    for i, sd_method in enumerate(sd_methods):
        df = all_data[sd_method]
        for j, comp_method in enumerate(all_methods):
            row = df[df['comparison_method'] == comp_method]
            if not row.empty:
                matrix[i, j] = row['mean_diff'].values[0]
            else:
                matrix[i, j] = np.nan
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Convert to millions for better readability
    matrix_millions = matrix / 1_000_000
    
    sns.heatmap(matrix_millions, 
                xticklabels=all_methods,
                yticklabels=sd_methods,
                cmap='RdYlGn_r',
                center=0,
                annot=False,
                cbar_kws={'label': 'Mean Difference (Millions)\nPositive = Inpainting method is worse'},
                linewidths=0.5,
                linecolor='white')
    
    plt.title('Mean Differences: Inpainting Methods vs Other Methods\n(Positive = Inpainting method has higher difference = worse)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Comparison Methods', fontsize=12, fontweight='bold')
    plt.ylabel('Inpainting Methods', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    output_file = output_dir / 'heatmap_mean_differences.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    
    plt.close()

def create_combined_plot(all_data, output_dir):
    """Create combined plot with both p-values and mean differences"""
    
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
                else:
                    text = f'{diff_val:.2f}M\n{direction}'
                
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
    plt.suptitle('Statistical Comparison: Inpainting Methods vs Classical Methods', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Add legend at the top, below the title - one clean box
    legend_text = ('Statistical significance: *** p<0.001   ** p<0.01   * p<0.05   ns = not significant\n'
                   'Performance indicators: ▲ = Classical method better   ▼ = Inpainting method better')
    
    fig.text(0.5, 0.935, legend_text, ha='center', fontsize=10, 
             bbox=dict(boxstyle='round,pad=0.8', facecolor='lightyellow', 
                      alpha=0.9, edgecolor='darkgray', linewidth=1.5),
             verticalalignment='top')
    
    plt.tight_layout(rect=[0, 0, 1, 0.91])  # Make room for title and legend at top
    
    output_file = output_dir / 'heatmap_combined.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    
    plt.close()

def main():
    print("="*80)
    print("CREATING ALL HEATMAP VISUALIZATIONS")
    print("="*80)
    
    # Load data
    print("\nLoading test results...")
    all_data = load_test_results()
    
    if not all_data:
        print("❌ No test results found!")
        return
    
    print(f"✅ Loaded results for {len(all_data)} SD methods")
    
    # Create output directory
    output_dir = Path('reports/statistical_tests_sd')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create all visualizations
    print("\n📊 Creating heatmaps...")
    
    print("\n1. Discrete p-value heatmap (simplified)...")
    create_discrete_pvalue_heatmap(all_data, output_dir)
    
    print("\n2. P-value heatmap (log scale - horizontal)...")
    create_pvalue_heatmap_log(all_data, output_dir)
    
    print("\n3. P-value heatmap (log scale - vertical)...")
    create_pvalue_heatmap_log_vertical(all_data, output_dir)
    
    print("\n4. P-value heatmap (annotated)...")
    create_pvalue_heatmap_annotated(all_data, output_dir)
    
    print("\n5. Mean difference heatmap...")
    create_mean_difference_heatmap(all_data, output_dir)
    
    print("\n6. Combined plot...")
    create_combined_plot(all_data, output_dir)
    
    print("\n" + "="*80)
    print("✅ ALL VISUALIZATIONS COMPLETED!")
    print("="*80)
    print(f"\n📁 All saved in: {output_dir}/")
    print("\nGenerated files:")
    print("  1. heatmap_pvalues_discrete.png - Simple 3-color version")
    print("  2. heatmap_pvalues_log.png - Log scale p-values (horizontal)")
    print("  3. heatmap_pvalues_log_vertical.png - Log scale p-values (vertical)")
    print("  4. heatmap_pvalues_annotated.png - With significance markers")
    print("  5. heatmap_mean_differences.png - Effect sizes")
    print("  6. heatmap_combined.png - Combined view")

if __name__ == "__main__":
    main()

