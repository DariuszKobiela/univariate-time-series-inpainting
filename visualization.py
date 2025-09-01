import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Wczytanie danych
df = pd.read_csv('results/quick_experiment/df_final.csv')

# Obliczenie średnich dla każdej kombinacji fixing_method i missing_data_type
grouped_data = df.groupby(['fixing_method', 'missing_data_type'])[['MAE', 'MAPE', 'RMSE']].mean().reset_index()

# Ustawienia stylu wykresów
plt.style.use('default')
sns.set_palette("husl")

# Utworzenie figury z 9 wykresami (3x3)
fig, axes = plt.subplots(3, 3, figsize=(20, 16))
fig.suptitle('Średnie wartości metryk dla różnych metod uzupełniania danych\n(pogrupowane według typów brakujących danych)', 
             fontsize=18, fontweight='bold', y=0.98)

# Lista typów brakujących danych
missing_types = ['MAR', 'MCAR', 'MNAR']
metrics = ['MAE', 'MAPE', 'RMSE']
colors = [['skyblue', 'lightsteelblue', 'cornflowerblue'],
          ['lightcoral', 'salmon', 'indianred'],
          ['lightgreen', 'mediumseagreen', 'forestgreen']]

# Tworzenie wykresów
for i, metric in enumerate(metrics):
    for j, missing_type in enumerate(missing_types):
        # Filtrowanie danych dla konkretnego typu brakujących danych
        data_subset = grouped_data[grouped_data['missing_data_type'] == missing_type]
        
        # Sortowanie według wartości metryki
        data_sorted = data_subset.sort_values(metric)
        
        # Tworzenie wykresu
        axes[i, j].bar(data_sorted['fixing_method'], data_sorted[metric], 
                      alpha=0.7, color=colors[i][j])
        
        # Tytuł wykresu
        axes[i, j].set_title(f'{metric} - {missing_type}', fontsize=14, fontweight='bold')
        
        # Etykiety osi
        axes[i, j].set_xlabel('Fixing Method', fontsize=10)
        if metric == 'MAPE':
            axes[i, j].set_ylabel(f'{metric} (%)', fontsize=10)
        else:
            axes[i, j].set_ylabel(metric, fontsize=10)
        
        # Obrócenie etykiet na osi X
        axes[i, j].tick_params(axis='x', rotation=45)
        axes[i, j].grid(True, alpha=0.3)
        
        # Dodanie wartości na słupkach
        for k, v in enumerate(data_sorted[metric]):
            axes[i, j].text(k, v + max(data_sorted[metric]) * 0.01, f'{v:.3f}', 
                           ha='center', va='bottom', fontsize=8)

# Ustawienie layout
plt.tight_layout()

# Zapisanie wykresu
plt.savefig('results/quick_experiment/plots/metrics_comparison_detailed.png', dpi=300, bbox_inches='tight')
plt.show()

# Wyświetlenie szczegółowych tabel dla każdego typu brakujących danych
print("=" * 100)
print("SZCZEGÓŁOWE WYNIKI DLA KAŻDEGO TYPU BRAKUJĄCYCH DANYCH")
print("=" * 100)

for missing_type in missing_types:
    data_subset = grouped_data[grouped_data['missing_data_type'] == missing_type]
    print(f"\n{missing_type} (Missing At Random/Completely At Random/Not At Random):")
    print("-" * 80)
    
    # Sortowanie według MAE i wyświetlenie
    data_sorted = data_subset.sort_values('MAE')
    print(data_sorted[['fixing_method', 'MAE', 'MAPE', 'RMSE']].to_string(index=False, float_format='%.4f'))
    
    # Najlepsze metody dla danego typu
    print(f"\nNAJLEPSZE METODY dla {missing_type}:")
    print(f"  MAE:  {data_subset.loc[data_subset['MAE'].idxmin(), 'fixing_method']} ({data_subset['MAE'].min():.4f})")
    print(f"  MAPE: {data_subset.loc[data_subset['MAPE'].idxmin(), 'fixing_method']} ({data_subset['MAPE'].min():.4f})")
    print(f"  RMSE: {data_subset.loc[data_subset['RMSE'].idxmin(), 'fixing_method']} ({data_subset['RMSE'].min():.4f})")

# Podsumowanie ogólne
print("\n" + "=" * 100)
print("PODSUMOWANIE OGÓLNE - NAJLEPSZE METODY ACROSS ALL MISSING TYPES")
print("=" * 100)

overall_grouped = df.groupby('fixing_method')[['MAE', 'MAPE', 'RMSE']].mean().reset_index()
overall_sorted = overall_grouped.sort_values('MAE')
print("\nŚrednie wartości dla wszystkich typów brakujących danych:")
print(overall_sorted.to_string(index=False, float_format='%.4f'))

print(f"\nNAJLEPSZE METODY OGÓŁEM:")
print(f"  MAE:  {overall_grouped.loc[overall_grouped['MAE'].idxmin(), 'fixing_method']} ({overall_grouped['MAE'].min():.4f})")
print(f"  MAPE: {overall_grouped.loc[overall_grouped['MAPE'].idxmin(), 'fixing_method']} ({overall_grouped['MAPE'].min():.4f})")
print(f"  RMSE: {overall_grouped.loc[overall_grouped['RMSE'].idxmin(), 'fixing_method']} ({overall_grouped['RMSE'].min():.4f})")

# =====================================================================
# INTERAKTYWNA CZĘŚĆ Z STREAMLIT
# =====================================================================

def create_interactive_app():
    """
    Funkcja tworząca interaktywną aplikację Streamlit
    Uruchom przez: streamlit run visualization.py
    """
    import streamlit as st
    import plotly.express as px
    import plotly.graph_objects as go
    from scipy import stats
    
    st.set_page_config(page_title="Interactive Time Series Analysis", layout="wide")
    
    st.title("🔧 Interaktywna Analiza Metod Uzupełniania Danych Czasowych")
    st.markdown("---")
    
    # Wczytanie danych
    @st.cache_data
    def load_data():
        return pd.read_csv('results/quick_experiment/df_final.csv')
    
    df = load_data()
    
    # SIDEBAR - KONTROLKI
    st.sidebar.header("⚙️ Ustawienia Wykresu")
    
    # Wybór metryki
    metric = st.sidebar.selectbox(
        "📊 Wybierz metrykę:",
        options=['MAE', 'MAPE', 'RMSE'],
        index=0
    )
    
    # Wybór agregacji
    aggregation = st.sidebar.selectbox(
        "🧮 Rodzaj agregacji:",
        options=['mean', 'median', 'mode'],
        index=0
    )
    
    # FILTRY
    st.sidebar.markdown("### 🔍 Filtry")
    
    # Filtr missing_rate
    missing_rates = st.sidebar.multiselect(
        "Missing Rate (%):",
        options=sorted(df['missing_rate'].unique()),
        default=sorted(df['missing_rate'].unique())
    )
    
    # Filtr missing_data_type
    missing_types = st.sidebar.multiselect(
        "Missing Data Type:",
        options=sorted(df['missing_data_type'].unique()),
        default=sorted(df['missing_data_type'].unique())
    )
    
    # Filtr dataset
    datasets = st.sidebar.multiselect(
        "Dataset:",
        options=sorted(df['dataset'].unique()),
        default=sorted(df['dataset'].unique())
    )
    
    # Filtr fixing_method (opcjonalnie ograniczyć)
    fixing_methods = st.sidebar.multiselect(
        "Fixing Methods (optional filter):",
        options=sorted(df['fixing_method'].unique()),
        default=sorted(df['fixing_method'].unique())
    )
    
    # FILTROWANIE DANYCH
    filtered_df = df[
        (df['missing_rate'].isin(missing_rates)) &
        (df['missing_data_type'].isin(missing_types)) &
        (df['dataset'].isin(datasets)) &
        (df['fixing_method'].isin(fixing_methods))
    ]
    
    if filtered_df.empty:
        st.error("❌ Brak danych dla wybranych filtrów!")
        return
    
    # AGREGACJA DANYCH
    def aggregate_data(df, metric, agg_method):
        if agg_method == 'mode':
            # Mode może zwracać kilka wartości, bierzemy pierwszą
            return df.groupby('fixing_method')[metric].agg(lambda x: stats.mode(x)[0])
        else:
            return df.groupby('fixing_method')[metric].agg(agg_method)
    
    aggregated_data = aggregate_data(filtered_df, metric, aggregation).reset_index()
    aggregated_data = aggregated_data.sort_values(metric)
    
    # GŁÓWNY WYKRES
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader(f"📈 {metric} według Fixing Method ({aggregation})")
        
        # Plotly bar chart
        fig = px.bar(
            aggregated_data,
            x='fixing_method',
            y=metric,
            title=f'{metric} - {aggregation.title()} (Filtered Data)',
            color=metric,
            color_continuous_scale='viridis',
            text=metric
        )
        
        # Formatowanie wykresu
        fig.update_traces(
            texttemplate='%{text:.3f}',
            textposition='outside'
        )
        
        fig.update_layout(
            xaxis_tickangle=-45,
            height=600,
            showlegend=False,
            xaxis_title="Fixing Method",
            yaxis_title=f"{metric} ({'%' if metric == 'MAPE' else ''})"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Statystyki")
        best_method = aggregated_data.loc[aggregated_data[metric].idxmin(), 'fixing_method']
        best_value = aggregated_data[metric].min()
        worst_method = aggregated_data.loc[aggregated_data[metric].idxmax(), 'fixing_method']
        worst_value = aggregated_data[metric].max()
        
        st.metric("🥇 Najlepsza metoda", best_method, f"{best_value:.4f}")
        st.metric("🥉 Najgorsza metoda", worst_method, f"{worst_value:.4f}")
        st.metric("📈 Improvement", "", f"{((worst_value-best_value)/worst_value*100):.1f}%")
        
        # Statystyki filtrów
        st.markdown("### 🔢 Aktywne filtry:")
        st.write(f"• Missing rates: {missing_rates}")
        st.write(f"• Data types: {missing_types}")  
        st.write(f"• Datasets: {datasets}")
        st.write(f"• Records: {len(filtered_df)}")
    
    # TABELA SZCZEGÓŁÓW
    st.subheader("📋 Szczegółowe Wyniki")
    
    # Formatowanie tabeli
    display_df = aggregated_data.copy()
    display_df[metric] = display_df[metric].round(4)
    display_df['rank'] = range(1, len(display_df) + 1)
    display_df = display_df[['rank', 'fixing_method', metric]]
    
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )
    
    # PORÓWNANIE METRYK
    if st.checkbox("📊 Pokaż porównanie wszystkich metryk"):
        st.subheader("🔄 Porównanie MAE, MAPE, RMSE")
        
        # Agregacja dla wszystkich metryk
        all_metrics_data = filtered_df.groupby('fixing_method')[['MAE', 'MAPE', 'RMSE']].agg(aggregation).reset_index()
        
        # Normalizacja dla lepszego porównania (0-1 scale)
        metrics_normalized = all_metrics_data.copy()
        for col in ['MAE', 'MAPE', 'RMSE']:
            metrics_normalized[f'{col}_norm'] = (metrics_normalized[col] - metrics_normalized[col].min()) / (metrics_normalized[col].max() - metrics_normalized[col].min())
        
        # Heatmap
        fig_heatmap = px.imshow(
            metrics_normalized[['MAE_norm', 'MAPE_norm', 'RMSE_norm']].T,
            x=metrics_normalized['fixing_method'],
            y=['MAE (norm)', 'MAPE (norm)', 'RMSE (norm)'],
            color_continuous_scale='RdYlBu_r',
            title="Znormalizowane Metryki (ciemniejsze = gorsze)"
        )
        
        fig_heatmap.update_layout(height=400)
        st.plotly_chart(fig_heatmap, use_container_width=True)

# Uruchamianie interaktywnej części tylko gdy jest uruchamiana jako Streamlit app
if __name__ == "__main__":
    # Sprawdź czy to Streamlit
    try:
        import streamlit as st
        # Jeśli streamlit jest dostępny i uruchamiamy jako app
        if hasattr(st, 'get_option'):
            create_interactive_app()
    except ImportError:
        print("Streamlit nie jest zainstalowany. Zainstaluj przez: pip install streamlit plotly")
        print("Uruchom interaktywną aplikację przez: streamlit run visualization.py")