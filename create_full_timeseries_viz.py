#!/usr/bin/env python3
"""
Create full time series visualization for best inpainting case
Shows: original, disrupted, fixed with inpainting, fixed with traditional method
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_time_series_data(dataset, miss_type, miss_rate, iteration):
    """Load actual time series data from files"""
    
    # Map dataset names to file names
    dataset_file_mapping = {
        'boiler': 'boiler_outlet_temp_univ.csv',
        'lake1': 'water_level_sensors_2010_L300.csv',
        'lake2': 'water_level_sensors_2010_L308.csv', 
        'lake3': 'water_level_sensors_2010_L311.csv',
        'pump': 'pump_sensor_28_univ.csv',
        'vibr': 'vibration_sensor_S1.csv'
    }
    
    # Load original data
    original_file = Path('data/0_source_data') / dataset_file_mapping[dataset]
    df_original = pd.read_csv(original_file, index_col=0)
    series_original = df_original.iloc[:, 0]
    
    # Load disrupted data (with missing values)
    disrupted_file = Path('data/1_missing_data') / f'{dataset}_{miss_type}_{miss_rate}p_{iteration}.csv'
    
    if disrupted_file.exists():
        df_disrupted = pd.read_csv(disrupted_file, index_col=0)
        series_disrupted = df_disrupted.iloc[:, 0]
    else:
        print(f"⚠️  Disrupted file not found: {disrupted_file}")
        series_disrupted = None
    
    return series_original, series_disrupted

def load_fixed_series(dataset, miss_type, miss_rate, iteration, method):
    """Load fixed series from 2_fixed_data"""
    
    # Transform method name (e.g., 'rp-unet' -> 'rpunet')
    method_suffix = method.replace('-', '')
    
    fixed_file = Path('data/2_fixed_data') / f'{dataset}_{miss_type}_{miss_rate}p_{iteration}_{method_suffix}.csv'
    
    if fixed_file.exists():
        df_fixed = pd.read_csv(fixed_file, index_col=0)
        series_fixed = df_fixed.iloc[:, 0]
        return series_fixed
    else:
        print(f"⚠️  Fixed file not found: {fixed_file}")
        return None

def apply_traditional_method(series_disrupted, method_name):
    """Apply traditional fixing method"""
    series_fixed = series_disrupted.copy()
    
    if method_name.startswith('interpolate_'):
        method = method_name.replace('interpolate_', '')
        if method in ['polynomial', 'quadratic']:
            temp_series = series_disrupted.copy()
            temp_series.index = range(len(temp_series))
            temp_series = temp_series.interpolate(method=method, order=2)
            series_fixed.iloc[:] = temp_series.values
        else:
            series_fixed = series_disrupted.interpolate(method=method)
    elif method_name.startswith('impute_'):
        method = method_name.replace('impute_', '')
        if method == 'mean':
            series_fixed = series_disrupted.fillna(series_disrupted.mean())
        elif method == 'median':
            series_fixed = series_disrupted.fillna(series_disrupted.median())
        elif method == 'mode':
            series_fixed = series_disrupted.fillna(series_disrupted.mode()[0])
        elif method == 'ffill':
            series_fixed = series_disrupted.fillna(method='ffill')
        elif method == 'bfill':
            series_fixed = series_disrupted.fillna(method='bfill')
    
    return series_fixed

def create_full_visualization(case_info, output_dir):
    """Create comprehensive time series visualization"""
    
    dataset = case_info['dataset']
    miss_type = case_info['missing_data_type']
    miss_rate = int(case_info['missing_rate'])
    iteration = int(case_info['iteration_nr'])
    inpainting_method = case_info['best_inpainting_method']
    traditional_method = case_info['best_traditional_method']
    
    print(f"\n{'='*80}")
    print(f"Creating visualization for:")
    print(f"  Dataset: {dataset}")
    print(f"  Missing type: {miss_type}, Rate: {miss_rate}%, Iteration: {iteration}")
    print(f"  Inpainting: {inpainting_method} (MAPE: {case_info['best_inpainting_MAPE']:.6f})")
    print(f"  Traditional: {traditional_method} (MAPE: {case_info['best_traditional_MAPE']:.6f})")
    print(f"  Improvement: {case_info['improvement_pct']:.2f}%")
    print(f"{'='*80}")
    
    # Load data
    series_original, series_disrupted = load_time_series_data(dataset, miss_type, miss_rate, iteration)
    
    if series_original is None or series_disrupted is None:
        print("❌ Could not load time series data")
        return None
    
    # Load or create fixed series
    series_fixed_inpainting = load_fixed_series(dataset, miss_type, miss_rate, iteration, inpainting_method)
    
    if series_fixed_inpainting is None:
        print(f"⚠️  Using interpolation as fallback for {inpainting_method}")
        series_fixed_inpainting = series_disrupted.interpolate(method='linear')
    
    series_fixed_traditional = apply_traditional_method(series_disrupted, traditional_method)
    
    # Create output directory
    case_name = f"{dataset}_{miss_type}_{miss_rate}_iter{iteration}"
    case_dir = output_dir / case_name
    case_dir.mkdir(parents=True, exist_ok=True)
    
    # Create main visualization
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    # Plot 1: Full comparison
    ax1 = axes[0]
    
    ax1.plot(series_original.index, series_original.values, 
            'k-', label='Original (Source)', linewidth=2.5, alpha=0.9, zorder=1)
    ax1.plot(series_disrupted.index, series_disrupted.values, 
            'r:', label=f'Disrupted ({miss_type}, {miss_rate}%)', 
            linewidth=2, alpha=0.8, markersize=4, zorder=2)
    ax1.plot(series_fixed_inpainting.index, series_fixed_inpainting.values, 
            'g-', label=f'Fixed - {inpainting_method} (MAPE: {case_info["best_inpainting_MAPE"]:.4f})', 
            linewidth=1.8, alpha=0.8, zorder=3)
    ax1.plot(series_fixed_traditional.index, series_fixed_traditional.values, 
            'b--', label=f'Fixed - {traditional_method} (MAPE: {case_info["best_traditional_MAPE"]:.4f})', 
            linewidth=1.8, alpha=0.8, zorder=4)
    
    # Highlight missing regions
    missing_mask = series_disrupted.isna()
    if missing_mask.any():
        # Find continuous missing regions
        missing_indices = np.where(missing_mask)[0]
        if len(missing_indices) > 0:
            # Group consecutive indices
            groups = np.split(missing_indices, np.where(np.diff(missing_indices) != 1)[0] + 1)
            for group in groups:
                if len(group) > 0:
                    start_idx = series_disrupted.index[group[0]]
                    end_idx = series_disrupted.index[group[-1]]
                    ax1.axvspan(start_idx, end_idx, alpha=0.15, color='red', zorder=0)
    
    ax1.set_xlabel('Time', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Value', fontsize=13, fontweight='bold')
    ax1.set_title(f'Time Series Comparison: {dataset.upper()} ({miss_type}, {miss_rate}% missing)\n'
                 f'Inpainting Improvement: {case_info["improvement_pct"]:.2f}%', 
                 fontsize=15, fontweight='bold', pad=15)
    ax1.legend(loc='best', fontsize=11, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Plot 2: Zoomed view on missing region (if exists)
    ax2 = axes[1]
    
    if missing_mask.any():
        # Find the largest missing region
        groups = np.split(missing_indices, np.where(np.diff(missing_indices) != 1)[0] + 1)
        largest_group = max(groups, key=len)
        
        # Expand view to show context
        margin = max(10, len(largest_group) // 2)
        start_zoom = max(0, largest_group[0] - margin)
        end_zoom = min(len(series_original) - 1, largest_group[-1] + margin)
        
        zoom_indices = series_original.index[start_zoom:end_zoom+1]
        
        ax2.plot(zoom_indices, series_original.iloc[start_zoom:end_zoom+1].values,
                'k-', label='Original', linewidth=3, alpha=0.9, marker='o', markersize=4)
        ax2.plot(zoom_indices, series_disrupted.iloc[start_zoom:end_zoom+1].values,
                'r:', label='Disrupted', linewidth=2.5, alpha=0.8, marker='x', markersize=6)
        ax2.plot(zoom_indices, series_fixed_inpainting.iloc[start_zoom:end_zoom+1].values,
                'g-', label=f'{inpainting_method}', linewidth=2.5, alpha=0.8, marker='s', markersize=5)
        ax2.plot(zoom_indices, series_fixed_traditional.iloc[start_zoom:end_zoom+1].values,
                'b--', label=f'{traditional_method}', linewidth=2.5, alpha=0.8, marker='^', markersize=5)
        
        # Highlight missing region in zoom
        zoom_start_idx = series_original.index[largest_group[0]]
        zoom_end_idx = series_original.index[largest_group[-1]]
        ax2.axvspan(zoom_start_idx, zoom_end_idx, alpha=0.2, color='red', label='Missing region')
        
        ax2.set_xlabel('Time', fontsize=13, fontweight='bold')
        ax2.set_ylabel('Value', fontsize=13, fontweight='bold')
        ax2.set_title('Zoomed View: Missing Data Region', fontsize=14, fontweight='bold', pad=10)
        ax2.legend(loc='best', fontsize=11, framealpha=0.9)
        ax2.grid(True, alpha=0.4, linestyle='--')
    else:
        ax2.text(0.5, 0.5, 'No missing data to zoom', 
                ha='center', va='center', fontsize=14, transform=ax2.transAxes)
        ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(case_dir / 'full_time_series_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {case_dir / 'full_time_series_comparison.png'}")
    plt.close()
    
    # Create error comparison plot
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Calculate errors (differences from original)
    error_inpainting = np.abs(series_fixed_inpainting - series_original)
    error_traditional = np.abs(series_fixed_traditional - series_original)
    
    ax.plot(series_original.index, error_inpainting.values, 
           'g-', label=f'{inpainting_method} error', linewidth=1.5, alpha=0.7)
    ax.plot(series_original.index, error_traditional.values, 
           'b-', label=f'{traditional_method} error', linewidth=1.5, alpha=0.7)
    
    ax.fill_between(series_original.index, error_inpainting.values, alpha=0.3, color='green')
    ax.fill_between(series_original.index, error_traditional.values, alpha=0.3, color='blue')
    
    ax.set_xlabel('Time', fontsize=13, fontweight='bold')
    ax.set_ylabel('Absolute Error', fontsize=13, fontweight='bold')
    ax.set_title(f'Reconstruction Error Comparison: {dataset.upper()}\n'
                f'{inpainting_method} vs {traditional_method}', 
                fontsize=15, fontweight='bold', pad=15)
    ax.legend(loc='best', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    stats_text = (f'Mean Error:\n'
                 f'{inpainting_method}: {error_inpainting.mean():.4f}\n'
                 f'{traditional_method}: {error_traditional.mean():.4f}')
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(case_dir / 'error_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {case_dir / 'error_comparison.png'}")
    plt.close()
    
    return case_dir

def main():
    print("="*80)
    print("CREATING FULL TIME SERIES VISUALIZATIONS")
    print("="*80)
    
    # Load best cases
    best_cases_file = Path('reports/best_inpainting_cases/best_inpainting_cases.csv')
    df = pd.read_csv(best_cases_file)
    
    print(f"✅ Loaded {len(df)} best cases")
    
    # Create output directory
    output_dir = Path('reports/best_inpainting_cases')
    
    # Create visualization for top case
    print("\n📊 Creating visualization for TOP case...")
    top_case = df.iloc[0]
    case_dir = create_full_visualization(top_case, output_dir)
    
    # Ask if user wants more
    print(f"\n{'='*80}")
    print("✅ VISUALIZATION COMPLETED!")
    print(f"{'='*80}")
    print(f"\n📁 Files saved in: {case_dir}/")
    print("\nGenerated files:")
    print("  - full_time_series_comparison.png - Complete time series with zoom")
    print("  - error_comparison.png - Reconstruction errors")

if __name__ == "__main__":
    main()

