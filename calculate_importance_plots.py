import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Wczytaj dane
df = pd.read_csv('importance.csv')

# Utwórz folder na wykresy
output_dir = Path('importance_results')
output_dir.mkdir(exist_ok=True)

print(f"Wczytano {len(df)} wierszy danych")
print(f"Generowanie wykresów do folderu: {output_dir}")

# Pobierz unikalne kombinacje
datasets = sorted(df['dataset_name'].unique())
missingness_types = sorted(df['missingness_type'].unique())
missing_percentages = sorted(df['missing_percentage'].unique(), 
                            key=lambda x: int(x.replace('p', '')))
fixing_methods = sorted(df['fixing_method'].unique())

print(f"\nDatasets: {datasets}")
print(f"Missingness types: {missingness_types}")
print(f"Missing percentages: {missing_percentages}")
print(f"Fixing methods: {len(fixing_methods)} metod")

# Licznik wykresów
plot_count = 0
total_plots = len(datasets) * len(missingness_types) * len(missing_percentages)

# Generuj wykresy
for dataset in datasets:
    for miss_type in missingness_types:
        for miss_pct in missing_percentages:
            # Filtruj dane dla tej kombinacji
            subset = df[
                (df['dataset_name'] == dataset) & 
                (df['missingness_type'] == miss_type) & 
                (df['missing_percentage'] == miss_pct)
            ]
            
            if len(subset) == 0:
                print(f"Brak danych dla: {dataset}, {miss_type}, {miss_pct}")
                continue
            
            plot_count += 1
            print(f"[{plot_count}/{total_plots}] Generowanie: {dataset}_{miss_type}_{miss_pct}")
            
            # Przygotuj dane do wykresu
            # Dla każdej fixing_method mamy 10 iteracji
            fig, ax = plt.subplots(figsize=(20, 8))
            
            # Szerokość pojedynczego słupka
            bar_width = 0.08
            
            # Pozycje dla każdej fixing_method (grupy po 10 słupków)
            method_positions = np.arange(len(fixing_methods))
            
            # Kolory dla iteracji (gradient)
            colors = plt.cm.viridis(np.linspace(0, 1, 10))
            
            # Dla każdej fixing_method rysuj 10 słupków (po 1 dla każdej iteracji)
            for method_idx, method in enumerate(fixing_methods):
                method_data = subset[subset['fixing_method'] == method].sort_values('iteration_nr')
                
                # Pozycja bazowa dla tej metody
                base_pos = method_positions[method_idx]
                
                # Rysuj 10 wąskich słupków dla iteracji 1-10
                for iter_idx, (_, row) in enumerate(method_data.iterrows()):
                    iteration_nr = row['iteration_nr']
                    p_value = row['p_value']
                    
                    # Pozycja tego konkretnego słupka
                    # Centrujemy grupę 10 słupków wokół pozycji bazowej
                    offset = (iter_idx - 4.5) * bar_width
                    pos = base_pos + offset
                    
                    # Rysuj słupek
                    ax.bar(pos, p_value, width=bar_width, 
                          color=colors[iteration_nr - 1],
                          edgecolor='black', linewidth=0.3,
                          label=f'Iter {iteration_nr}' if method_idx == 0 else '')
            
            # Ustawienia wykresu
            ax.set_xlabel('Fixing Method', fontsize=12, fontweight='bold')
            ax.set_ylabel('P-value (lower = more different from original)', fontsize=12, fontweight='bold')
            ax.set_title(f'{dataset.upper()} | {miss_type} | {miss_pct} missing', 
                        fontsize=14, fontweight='bold')
            
            # Etykiety na osi X
            ax.set_xticks(method_positions)
            ax.set_xticklabels(fixing_methods, rotation=45, ha='right')
            
            # Siatka
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.set_axisbelow(True)
            
            # Legenda (tylko dla iteracji)
            handles, labels = ax.get_legend_handles_labels()
            # Usuń duplikaty z legendy
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), 
                     title='Iterations', loc='upper right', ncol=2, fontsize=9)
            
            # Dodaj linię poziomą przy p=0.05 (próg istotności statystycznej)
            ax.axhline(y=0.05, color='red', linestyle='--', linewidth=1.5, 
                      alpha=0.7, label='p=0.05 threshold')
            
            # Dostosuj layout
            plt.tight_layout()
            
            # Zapisz wykres
            filename = f"{dataset}_{miss_type}_{miss_pct}.png"
            filepath = output_dir / filename
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()

print(f"\n✓ Wygenerowano {plot_count} wykresów w folderze '{output_dir}'")

# Wygeneruj również wykres podsumowujący - średnie p-value dla każdej metody
print("\nGenerowanie wykresu podsumowującego...")

fig, ax = plt.subplots(figsize=(16, 8))

# Oblicz średnie p-value dla każdej fixing_method
mean_pvalues = df.groupby('fixing_method')['p_value'].mean().sort_values()

# Bar plot
bars = ax.bar(range(len(mean_pvalues)), mean_pvalues.values, 
              color='steelblue', edgecolor='black', linewidth=0.5)

# Koloruj słupki w zależności od wartości (czerwony = niskie p-value, zielony = wysokie)
norm = plt.Normalize(vmin=mean_pvalues.min(), vmax=mean_pvalues.max())
colors_map = plt.cm.RdYlGn(norm(mean_pvalues.values))
for bar, color in zip(bars, colors_map):
    bar.set_color(color)

ax.set_xlabel('Fixing Method', fontsize=12, fontweight='bold')
ax.set_ylabel('Mean P-value (across all datasets/conditions)', fontsize=12, fontweight='bold')
ax.set_title('Average Performance of Fixing Methods\n(Higher p-value = closer to original values)', 
            fontsize=14, fontweight='bold')
ax.set_xticks(range(len(mean_pvalues)))
ax.set_xticklabels(mean_pvalues.index, rotation=45, ha='right')
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

# Dodaj wartości na słupkach
for i, (method, value) in enumerate(mean_pvalues.items()):
    ax.text(i, value + max(mean_pvalues) * 0.01, f'{value:.3f}', 
           ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(output_dir / 'summary_mean_pvalues.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"✓ Zapisano wykres podsumowujący: {output_dir / 'summary_mean_pvalues.png'}")

# Wygeneruj heatmapę
print("\nGenerowanie heatmapy...")

import seaborn as sns

# Pivot table: fixing_method vs kombinacje
df['condition'] = df['dataset_name'] + '_' + df['missingness_type'] + '_' + df['missing_percentage']
pivot_data = df.groupby(['fixing_method', 'condition'])['p_value'].mean().reset_index()
pivot_table = pivot_data.pivot(index='fixing_method', columns='condition', values='p_value')

# Sortuj według średniej
pivot_table = pivot_table.loc[pivot_table.mean(axis=1).sort_values().index]

fig, ax = plt.subplots(figsize=(24, 12))
sns.heatmap(pivot_table, annot=False, cmap='RdYlGn', center=0.5, 
           cbar_kws={'label': 'P-value'}, linewidths=0.5, ax=ax)
ax.set_xlabel('Dataset_MissingnessType_Percentage', fontsize=11, fontweight='bold')
ax.set_ylabel('Fixing Method', fontsize=11, fontweight='bold')
ax.set_title('Heatmap of P-values: Fixing Methods vs Conditions\n(Green = high p-value = good, Red = low p-value = bad)', 
            fontsize=13, fontweight='bold')
plt.xticks(rotation=90, ha='right', fontsize=8)
plt.yticks(fontsize=9)
plt.tight_layout()
plt.savefig(output_dir / 'heatmap_pvalues.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"✓ Zapisano heatmapę: {output_dir / 'heatmap_pvalues.png'}")

print("\n" + "="*60)
print("GOTOWE! Wszystkie wykresy zostały wygenerowane.")
print("="*60)

