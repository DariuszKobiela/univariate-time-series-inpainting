#!/usr/bin/env python3
"""
Skrypt do generowania zbiorczego raportu z df_differences.csv
Tworzy agregowane raporty pokazujące najlepsze metody dla różnych kombinacji parametrów
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

def generate_summary_report():
    """Generuje zbiorczy raport z sum różnic bezwzględnych"""
    
    print("📊 Generating Summary Report from df_differences.csv")
    print("=" * 80)
    
    # Wczytaj dane
    try:
        df = pd.read_csv('df_differences.csv')
        print(f"✅ Loaded {len(df)} records from df_differences.csv")
    except FileNotFoundError:
        print("❌ Error: df_differences.csv not found!")
        sys.exit(1)
    
    # Utworz folder na raporty
    output_dir = Path('reports/differences_summary')
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")
    
    # ============================================
    # 1. RAPORT ZBIORCZY - wszystkie kombinacje
    # ============================================
    print("\n📋 Generating comprehensive summary table...")
    
    # Agreguj po wszystkich parametrach
    comprehensive_summary = df.groupby([
        'dataset_name', 
        'missing_data_type', 
        'missing_rate', 
        'fixing_method'
    ])['difference'].sum().reset_index()
    
    comprehensive_summary.columns = [
        'Dataset', 
        'Missingness_Type', 
        'Missingness_Rate', 
        'Fixing_Method', 
        'Sum_of_Absolute_Differences'
    ]
    
    comprehensive_summary = comprehensive_summary.sort_values([
        'Dataset', 
        'Missingness_Type', 
        'Missingness_Rate', 
        'Sum_of_Absolute_Differences'
    ])
    
    comprehensive_summary.to_csv(output_dir / 'comprehensive_summary.csv', index=False)
    print(f"   ✅ Saved: comprehensive_summary.csv ({len(comprehensive_summary)} rows)")
    
    # Wersja przefiltrowana (bez lake1, lake2, lake3 i bez gaf-unet, mtf-unet, rp-unet, spec-unet)
    datasets_to_exclude = ['lake1', 'lake2', 'lake3']
    methods_to_exclude = ['gafunet', 'mtfunet', 'rpunet', 'specunet']
    
    filtered_summary = comprehensive_summary[
        (~comprehensive_summary['Dataset'].isin(datasets_to_exclude)) &
        (~comprehensive_summary['Fixing_Method'].isin(methods_to_exclude))
    ]
    
    filtered_summary.to_csv(output_dir / 'comprehensive_summary_filtered.csv', index=False)
    print(f"   ✅ Saved: comprehensive_summary_filtered.csv ({len(filtered_summary)} rows)")
    print(f"      Excluded datasets: {datasets_to_exclude}")
    print(f"      Excluded methods: {methods_to_exclude}")
    
    # ============================================
    # 2. RAPORT OGÓLNY - agregacja po fixing_method
    # ============================================
    print("\n📋 Generating overall summary by fixing_method...")
    
    overall_summary = df.groupby('fixing_method')['difference'].agg([
        ('total_difference', 'sum'),
        ('mean_difference', 'mean'),
        ('median_difference', 'median'),
        ('std_difference', 'std'),
        ('min_difference', 'min'),
        ('max_difference', 'max'),
        ('count', 'count')
    ]).round(4)
    
    overall_summary = overall_summary.sort_values('total_difference')
    overall_summary['rank'] = range(1, len(overall_summary) + 1)
    overall_summary = overall_summary[['rank', 'total_difference', 'mean_difference', 'median_difference', 
                                       'std_difference', 'min_difference', 'max_difference', 'count']]
    
    overall_summary.to_csv(output_dir / 'overall_summary_by_method.csv')
    print(f"   ✅ Saved: overall_summary_by_method.csv")
    
    # ============================================
    # 2. RAPORT PO DATASET
    # ============================================
    print("\n📋 Generating summary by dataset...")
    
    dataset_summary = df.groupby(['dataset_name', 'fixing_method'])['difference'].agg([
        ('total_difference', 'sum'),
        ('mean_difference', 'mean'),
        ('count', 'count')
    ]).round(4).reset_index()
    
    dataset_summary = dataset_summary.sort_values(['dataset_name', 'total_difference'])
    dataset_summary.to_csv(output_dir / 'summary_by_dataset.csv', index=False)
    print(f"   ✅ Saved: summary_by_dataset.csv")
    
    # ============================================
    # 3. RAPORT PO MISSING_DATA_TYPE
    # ============================================
    print("\n📋 Generating summary by missing_data_type...")
    
    missing_type_summary = df.groupby(['missing_data_type', 'fixing_method'])['difference'].agg([
        ('total_difference', 'sum'),
        ('mean_difference', 'mean'),
        ('count', 'count')
    ]).round(4).reset_index()
    
    missing_type_summary = missing_type_summary.sort_values(['missing_data_type', 'total_difference'])
    missing_type_summary.to_csv(output_dir / 'summary_by_missing_type.csv', index=False)
    print(f"   ✅ Saved: summary_by_missing_type.csv")
    
    # ============================================
    # 4. RAPORT PO MISSING_RATE
    # ============================================
    print("\n📋 Generating summary by missing_rate...")
    
    missing_rate_summary = df.groupby(['missing_rate', 'fixing_method'])['difference'].agg([
        ('total_difference', 'sum'),
        ('mean_difference', 'mean'),
        ('count', 'count')
    ]).round(4).reset_index()
    
    missing_rate_summary = missing_rate_summary.sort_values(['missing_rate', 'total_difference'])
    missing_rate_summary.to_csv(output_dir / 'summary_by_missing_rate.csv', index=False)
    print(f"   ✅ Saved: summary_by_missing_rate.csv")
    
    # ============================================
    # 5. MACIERZ - Best methods dla każdej kombinacji dataset x missing_type
    # ============================================
    print("\n📋 Generating best methods matrix (dataset x missing_type)...")
    
    # Dla każdej kombinacji znajdź najlepszą metodę
    best_methods_matrix = []
    
    for dataset in df['dataset_name'].unique():
        for missing_type in df['missing_data_type'].unique():
            subset = df[(df['dataset_name'] == dataset) & 
                       (df['missing_data_type'] == missing_type)]
            
            if not subset.empty:
                method_totals = subset.groupby('fixing_method')['difference'].sum()
                best_method = method_totals.idxmin()
                best_value = method_totals.min()
                worst_method = method_totals.idxmax()
                worst_value = method_totals.max()
                
                best_methods_matrix.append({
                    'dataset': dataset,
                    'missing_type': missing_type,
                    'best_method': best_method,
                    'best_value': best_value,
                    'worst_method': worst_method,
                    'worst_value': worst_value,
                    'improvement': ((worst_value - best_value) / worst_value * 100) if worst_value > 0 else 0
                })
    
    best_methods_df = pd.DataFrame(best_methods_matrix)
    best_methods_df = best_methods_df.sort_values(['dataset', 'missing_type'])
    best_methods_df.to_csv(output_dir / 'best_methods_matrix_dataset_x_type.csv', index=False)
    print(f"   ✅ Saved: best_methods_matrix_dataset_x_type.csv")
    
    # ============================================
    # 6. MACIERZ - Best methods dla każdej kombinacji missing_type x missing_rate
    # ============================================
    print("\n📋 Generating best methods matrix (missing_type x missing_rate)...")
    
    best_methods_matrix_2 = []
    
    for missing_type in df['missing_data_type'].unique():
        for missing_rate in df['missing_rate'].unique():
            subset = df[(df['missing_data_type'] == missing_type) & 
                       (df['missing_rate'] == missing_rate)]
            
            if not subset.empty:
                method_totals = subset.groupby('fixing_method')['difference'].sum()
                best_method = method_totals.idxmin()
                best_value = method_totals.min()
                worst_method = method_totals.idxmax()
                worst_value = method_totals.max()
                
                best_methods_matrix_2.append({
                    'missing_type': missing_type,
                    'missing_rate': missing_rate,
                    'best_method': best_method,
                    'best_value': best_value,
                    'worst_method': worst_method,
                    'worst_value': worst_value,
                    'improvement': ((worst_value - best_value) / worst_value * 100) if worst_value > 0 else 0
                })
    
    best_methods_df_2 = pd.DataFrame(best_methods_matrix_2)
    best_methods_df_2 = best_methods_df_2.sort_values(['missing_type', 'missing_rate'])
    best_methods_df_2.to_csv(output_dir / 'best_methods_matrix_type_x_rate.csv', index=False)
    print(f"   ✅ Saved: best_methods_matrix_type_x_rate.csv")
    
    # ============================================
    # 7. TOP 10 NAJLEPSZYCH I NAJGORSZYCH KOMBINACJI
    # ============================================
    print("\n📋 Generating top 10 best and worst combinations...")
    
    # Grupuj po wszystkich parametrach
    full_combinations = df.groupby(['dataset_name', 'missing_data_type', 'missing_rate', 'fixing_method'])['difference'].sum().reset_index()
    full_combinations = full_combinations.sort_values('difference')
    
    # Top 10 najlepszych
    top_10_best = full_combinations.head(10).copy()
    top_10_best['rank'] = range(1, 11)
    top_10_best.to_csv(output_dir / 'top_10_best_combinations.csv', index=False)
    print(f"   ✅ Saved: top_10_best_combinations.csv")
    
    # Top 10 najgorszych
    top_10_worst = full_combinations.tail(10).copy()
    top_10_worst = top_10_worst.sort_values('difference', ascending=False)
    top_10_worst['rank'] = range(1, 11)
    top_10_worst.to_csv(output_dir / 'top_10_worst_combinations.csv', index=False)
    print(f"   ✅ Saved: top_10_worst_combinations.csv")
    
    # ============================================
    # 8. STATYSTYKI OGÓLNE
    # ============================================
    print("\n📋 Generating overall statistics...")
    
    stats = {
        'total_records': len(df),
        'unique_datasets': df['dataset_name'].nunique(),
        'unique_missing_types': df['missing_data_type'].nunique(),
        'unique_missing_rates': df['missing_rate'].nunique(),
        'unique_fixing_methods': df['fixing_method'].nunique(),
        'total_iterations': df['iteration_nr'].nunique(),
        'total_difference_sum': df['difference'].sum(),
        'mean_difference': df['difference'].mean(),
        'median_difference': df['difference'].median(),
        'std_difference': df['difference'].std(),
        'min_difference': df['difference'].min(),
        'max_difference': df['difference'].max()
    }
    
    stats_df = pd.DataFrame([stats]).T
    stats_df.columns = ['value']
    stats_df.to_csv(output_dir / 'overall_statistics.csv')
    print(f"   ✅ Saved: overall_statistics.csv")
    
    # ============================================
    # PODSUMOWANIE
    # ============================================
    print("\n" + "=" * 80)
    print("✅ REPORT GENERATION COMPLETE!")
    print("=" * 80)
    print(f"\n📁 All reports saved to: {output_dir}/")
    print("\nGenerated files:")
    print("  1. comprehensive_summary.csv - MAIN TABLE with all combinations")
    print("  1b. comprehensive_summary_filtered.csv - FILTERED TABLE (no lake1/2/3, no gaf/mtf/rp/spec-unet)")
    print("  2. overall_summary_by_method.csv - Overall ranking of all methods")
    print("  3. summary_by_dataset.csv - Summary by dataset")
    print("  4. summary_by_missing_type.csv - Summary by missing data type")
    print("  5. summary_by_missing_rate.csv - Summary by missing rate")
    print("  6. best_methods_matrix_dataset_x_type.csv - Best methods for dataset x type")
    print("  7. best_methods_matrix_type_x_rate.csv - Best methods for type x rate")
    print("  8. top_10_best_combinations.csv - Top 10 best combinations")
    print("  9. top_10_worst_combinations.csv - Top 10 worst combinations")
    print("  10. overall_statistics.csv - Overall statistics")
    
    # Wyświetl top 5 metod
    print("\n🏆 TOP 5 BEST METHODS (by total difference):")
    print("-" * 80)
    print(overall_summary.head(5)[['rank', 'total_difference', 'mean_difference', 'count']].to_string())
    
    print("\n🥇 OVERALL BEST METHOD:")
    best = overall_summary.iloc[0]
    print(f"   Method: {best.name}")
    print(f"   Total Difference: {best['total_difference']:.2f}")
    print(f"   Mean Difference: {best['mean_difference']:.2f}")
    print(f"   Records: {int(best['count'])}")


if __name__ == "__main__":
    generate_summary_report()

