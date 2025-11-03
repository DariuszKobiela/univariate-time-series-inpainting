#!/usr/bin/env python3
"""
Simple visualization of best inpainting cases - time series only
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def find_best_inpainting_cases(df):
    """Find cases where inpainting methods performed better"""
    
    inpainting_methods = ['gaf-unet', 'mtf-unet', 'rp-unet', 'spec-unet']
    interpolation_methods = ['interpolate_linear', 'interpolate_cubic', 'interpolate_akima', 
                            'interpolate_pchip', 'interpolate_polynomial', 'interpolate_quadratic',
                            'interpolate_nearest', 'interpolate_index']
    imputation_methods = ['impute_mean', 'impute_median', 'impute_mode', 'impute_ffill', 'impute_bfill']
    
    traditional_methods = interpolation_methods + imputation_methods
    
    groups = df.groupby(['dataset', 'missing_data_type', 'missing_rate', 'iteration_nr'])
    
    best_cases = []
    
    for group_key, group_data in groups:
        dataset, miss_type, miss_rate, iteration = group_key
        
        inpainting_data = group_data[group_data['fixing_method'].isin(inpainting_methods)]
        traditional_data = group_data[group_data['fixing_method'].isin(traditional_methods)]
        
        if len(inpainting_data) == 0 or len(traditional_data) == 0:
            continue
        
        best_inpainting = inpainting_data.loc[inpainting_data['MAPE'].idxmin()]
        best_traditional = traditional_data.loc[traditional_data['MAPE'].idxmin()]
        
        improvement = best_traditional['MAPE'] - best_inpainting['MAPE']
        improvement_pct = (improvement / best_traditional['MAPE']) * 100
        
        if improvement > 0:
            best_cases.append({
                'dataset': dataset,
                'missing_data_type': miss_type,
                'missing_rate': miss_rate,
                'iteration_nr': iteration,
                'best_inpainting_method': best_inpainting['fixing_method'],
                'best_inpainting_MAPE': best_inpainting['MAPE'],
                'best_traditional_method': best_traditional['fixing_method'],
                'best_traditional_MAPE': best_traditional['MAPE'],
                'improvement': improvement,
                'improvement_pct': improvement_pct,
                'inpainting_predictions': [best_inpainting[f'val_{i}'] for i in range(1, 11)],
                'traditional_predictions': [best_traditional[f'val_{i}'] for i in range(1, 11)]
            })
    
    best_cases_df = pd.DataFrame(best_cases)
    if len(best_cases_df) > 0:
        best_cases_df = best_cases_df.sort_values('improvement_pct', ascending=False)
    
    return best_cases_df

def create_simple_visualization(case, output_dir):
    """Create simple prediction comparison visualization"""
    
    dataset = case['dataset']
    miss_type = case['missing_data_type']
    miss_rate = case['missing_rate']
    
    case_dir = output_dir / f"{dataset}_{miss_type}_{miss_rate}_iter{case['iteration_nr']}"
    case_dir.mkdir(parents=True, exist_ok=True)
    
    # Prediction comparison plot
    fig, ax = plt.subplots(figsize=(14, 7))
    
    x = np.arange(1, 11)
    width = 0.35
    
    bars1 = ax.bar(x - width/2, case['inpainting_predictions'], width, 
                   label=f'{case["best_inpainting_method"]} (MAPE: {case["best_inpainting_MAPE"]:.6f})',
                   color='green', alpha=0.7)
    bars2 = ax.bar(x + width/2, case['traditional_predictions'], width,
                   label=f'{case["best_traditional_method"]} (MAPE: {case["best_traditional_MAPE"]:.6f})',
                   color='blue', alpha=0.7)
    
    ax.set_xlabel('Prediction Step', fontsize=14, fontweight='bold')
    ax.set_ylabel('Predicted Value', fontsize=14, fontweight='bold')
    ax.set_title(f'Forecast Comparison: {dataset.upper()} ({miss_type}, {miss_rate}% missing)\n'
                f'Inpainting Improvement: {case["improvement_pct"]:.2f}%', 
                fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticks(x)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(case_dir / 'prediction_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {case_dir / 'prediction_comparison.png'}")
    plt.close()
    
    return case_dir

def main():
    print("="*80)
    print("FINDING BEST INPAINTING CASES - SIMPLE VERSION")
    print("="*80)
    
    # Load data
    df = pd.read_csv('results/quick_experiment/df_final.csv')
    print(f"✅ Loaded {len(df)} rows from df_final.csv")
    
    # Find best cases
    print("\n🔍 Searching for cases where inpainting performed better...")
    best_cases = find_best_inpainting_cases(df)
    
    if len(best_cases) == 0:
        print("❌ No cases found where inpainting performed better")
        return
    
    print(f"\n✅ Found {len(best_cases)} cases where inpainting performed better!")
    
    # Create output directory
    output_dir = Path('reports/best_inpainting_cases')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results
    best_cases.to_csv(output_dir / 'best_inpainting_cases.csv', index=False)
    print(f"✅ Saved results to: {output_dir / 'best_inpainting_cases.csv'}")
    
    # Show top 10
    print("\nTop 10 cases:")
    print(best_cases[['dataset', 'missing_data_type', 'missing_rate', 'best_inpainting_method', 
                      'best_inpainting_MAPE', 'best_traditional_method', 'best_traditional_MAPE',
                      'improvement_pct']].head(10).to_string(index=False))
    
    # Create visualizations for top 5 cases
    print(f"\n{'='*80}")
    print("CREATING VISUALIZATIONS FOR TOP 5 CASES")
    print(f"{'='*80}")
    
    for i, (idx, case) in enumerate(best_cases.head(5).iterrows(), 1):
        print(f"\n{i}. {case['dataset']} - {case['missing_data_type']} - {case['missing_rate']}% - "
              f"Improvement: {case['improvement_pct']:.2f}%")
        case_dir = create_simple_visualization(case, output_dir)
    
    print(f"\n{'='*80}")
    print("✅ ALL VISUALIZATIONS COMPLETED!")
    print(f"{'='*80}")
    print(f"\n📁 All files saved in: {output_dir}/")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"\nTotal cases where inpainting was better: {len(best_cases)}")
    print(f"Average improvement: {best_cases['improvement_pct'].mean():.2f}%")
    print(f"Median improvement: {best_cases['improvement_pct'].median():.2f}%")
    print(f"Max improvement: {best_cases['improvement_pct'].max():.2f}%")
    
    print("\nBreakdown by dataset:")
    print(best_cases.groupby('dataset').size().to_string())
    
    print("\nBreakdown by inpainting method:")
    print(best_cases.groupby('best_inpainting_method').size().to_string())
    
    print("\nBreakdown by missing type:")
    print(best_cases.groupby('missing_data_type').size().to_string())

if __name__ == "__main__":
    main()


