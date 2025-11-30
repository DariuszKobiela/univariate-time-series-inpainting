import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from scipy import stats

# Wczytanie danych
df = pd.read_csv('results/quick_experiment/df_final.csv')

# Obliczenie średnich dla każdej kombinacji fixing_method i missing_data_type
grouped_data = df.groupby(['fixing_method', 'missing_data_type'])[['MAE', 'MAPE', 'RMSE']].mean().reset_index()

# Ustawienia stylu wykresów
plt.style.use('default')
sns.set_palette("husl")

# Utworzenie figury z 9 wykresami (3x3)
fig, axes = plt.subplots(3, 3, figsize=(20, 16))
fig.suptitle('Average Metric Values for Different Data Imputation Methods\n(grouped by missing data types)', 
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
print("DETAILED RESULTS FOR EACH MISSING DATA TYPE")
print("=" * 100)

for missing_type in missing_types:
    data_subset = grouped_data[grouped_data['missing_data_type'] == missing_type]
    print(f"\n{missing_type} (Missing At Random/Completely At Random/Not At Random):")
    print("-" * 80)
    
    # Sortowanie według MAE i wyświetlenie
    data_sorted = data_subset.sort_values('MAE')
    print(data_sorted[['fixing_method', 'MAE', 'MAPE', 'RMSE']].to_string(index=False, float_format='%.4f'))
    
    # Najlepsze metody dla danego typu
    print(f"\nBEST METHODS for {missing_type}:")
    print(f"  MAE:  {data_subset.loc[data_subset['MAE'].idxmin(), 'fixing_method']} ({data_subset['MAE'].min():.4f})")
    print(f"  MAPE: {data_subset.loc[data_subset['MAPE'].idxmin(), 'fixing_method']} ({data_subset['MAPE'].min():.4f})")
    print(f"  RMSE: {data_subset.loc[data_subset['RMSE'].idxmin(), 'fixing_method']} ({data_subset['RMSE'].min():.4f})")

# Podsumowanie ogólne
print("\n" + "=" * 100)
print("OVERALL SUMMARY - BEST METHODS ACROSS ALL MISSING TYPES")
print("=" * 100)

overall_grouped = df.groupby('fixing_method')[['MAE', 'MAPE', 'RMSE']].mean().reset_index()
overall_sorted = overall_grouped.sort_values('MAE')
print("\nAverage values for all missing data types:")
print(overall_sorted.to_string(index=False, float_format='%.4f'))

print(f"\nBEST METHODS OVERALL:")
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
    import os
    from PIL import Image
    
    st.set_page_config(page_title="Interactive Time Series Analysis", layout="wide")
    
    st.title("🔧 Interactive Analysis of Time Series Imputation Methods")
    st.markdown("---")
    
    # Wczytanie danych
    @st.cache_data
    def load_data():
        return pd.read_csv('results/quick_experiment/df_final.csv')
    
    df = load_data()
    
    # SIDEBAR - KONTROLKI
    st.sidebar.header("⚙️ Chart Settings")
    
    # Debug mode
    debug_mode = st.sidebar.checkbox("🐛 Debug mode for absolute differences", value=False)
    if debug_mode:
        st.session_state['debug_differences'] = True
    else:
        st.session_state['debug_differences'] = False
    
    # Wybór metryki
    metric = st.sidebar.selectbox(
        "📊 Select metric:",
        options=['MAE', 'MAPE', 'RMSE'],
        index=1
    )
    
    # Wybór agregacji
    aggregation = st.sidebar.selectbox(
        "🧮 Aggregation type:",
        options=['mean', 'median', 'mode'],
        index=0
    )
    
    # FILTRY
    st.sidebar.markdown("### 🔍 Filters")
    
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
    # Usuń niechciane datasety z listy opcji
    all_datasets = df['dataset'].unique()
    datasets_to_exclude = ['lake1', 'lake2', 'lake3']  # Datasety do usunięcia
    available_datasets = [d for d in all_datasets if d not in datasets_to_exclude]
    
    datasets = st.sidebar.multiselect(
        "Dataset:",
        options=available_datasets,
        default=available_datasets
    )
    
    # Filtr fixing_method (opcjonalnie ograniczyć)
    # Usuń niechciane metody z listy opcji
    all_methods = sorted(df['fixing_method'].unique())
    methods_to_exclude = ['gaf-unet', 'mtf-unet', 'rp-unet', 'spec-unet']  # Metody do usunięcia
    available_methods = [m for m in all_methods if m not in methods_to_exclude]
    
    fixing_methods = st.sidebar.multiselect(
        "Fixing Methods (optional filter):",
        options=available_methods,
        default=available_methods
    )
    
    # Filtr prediction_method (model forecasting)
    prediction_methods = st.sidebar.multiselect(
        "Prediction Methods (Forecasting Model):",
        options=sorted(df['prediction_method'].unique()),
        default=sorted(df['prediction_method'].unique())
    )
    
    
    # FILTROWANIE DANYCH
    filtered_df = df[
        (df['missing_rate'].isin(missing_rates)) &
        (df['missing_data_type'].isin(missing_types)) &
        (df['dataset'].isin(datasets)) &
        (df['fixing_method'].isin(fixing_methods)) &
        (df['prediction_method'].isin(prediction_methods))
    ]
    
    if filtered_df.empty:
        st.error("❌ No data for selected filters!")
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
        st.subheader(f"📈 {metric} by Fixing Method ({aggregation})")
        
        # Plotly bar chart
        # Formatowanie missing_rates do wyświetlenia w tytule
        if len(missing_rates) == len(sorted(df['missing_rate'].unique())):
            missing_rate_str = "all"
        else:
            missing_rate_str = ", ".join([f"{int(r)}%" for r in sorted(missing_rates)])
        
        # Formatowanie missing_types do wyświetlenia w tytule
        if len(missing_types) == len(sorted(df['missing_data_type'].unique())):
            missing_type_str = "all"
        else:
            missing_type_str = ", ".join(sorted(missing_types))
        
        # Formatowanie datasets do wyświetlenia w tytule
        if len(datasets) == len(available_datasets):
            dataset_str = "all"
        else:
            dataset_str = ", ".join(sorted(datasets))
        
        fig = px.bar(
            aggregated_data,
            x='fixing_method',
            y=metric,
            title=f'{aggregation} {metric} for each fixing method - missing rate: {missing_rate_str}, missing type: {missing_type_str}, dataset: {dataset_str}',
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
        
        # =====================================
        # NOWY WYKRES - SUMA RÓŻNIC BEZWZGLĘDNYCH
        # =====================================
        st.subheader("📊 Sum of Absolute Differences by Fixing Method")
        
        # Funkcja do wczytywania i agregowania danych różnic z pliku CSV
        @st.cache_data
        def load_and_aggregate_differences(filtered_main_df):
            """Wczytuje dane różnic z df_differences.csv i agreguje je według filtrów (suma)"""
            
            try:
                # Wczytaj dane różnic - sprawdź różne ścieżki
                possible_paths = [
                    'df_differences.csv',
                    './df_differences.csv', 
                    '../df_differences.csv',
                    '/home/darek/univariate-time-series-inpainting/df_differences.csv'
                ]
                
                differences_df = None
                for path in possible_paths:
                    try:
                        if os.path.exists(path):
                            differences_df = pd.read_csv(path)
                            if st.session_state.get('debug_differences', False):
                                st.write(f"🔍 Debug - data loaded from: {path}")
                            break
                    except:
                        continue
                
                if differences_df is None:
                    raise FileNotFoundError("File df_differences.csv not found in any of the paths")
                
                # Debug info
                if st.session_state.get('debug_differences', False):
                    st.write(f"🔍 Debug - differences_df shape: {differences_df.shape}")
                    st.write(f"🔍 Debug - filtered_main_df shape: {filtered_main_df.shape}")
                
                # Wyciągnij unikalne kombinacje parametrów z odfiltrowanych danych głównych
                main_combinations = filtered_main_df[['dataset', 'missing_data_type', 'missing_rate', 'fixing_method']].drop_duplicates()
                
                # Konwersja formatów dla kompatybilności
                # 1. missing_rate: liczby -> stringi z 'p' (np. 2 -> '2p')
                missing_rate_converted = [f"{int(rate)}p" for rate in main_combinations['missing_rate'].unique()]
                
                # 2. fixing_method: konwersja nazw metod
                # df_final używa nazw z myślnikami, df_differences bez myślników i podkreślników
                fixing_method_mapping = {
                    # Mapowanie z df_final do df_differences
                    'gaf-unet': 'gafunet',
                    'mtf-unet': 'mtfunet', 
                    'rp-unet': 'rpunet',
                    'spec-unet': 'specunet',
                    'impute_bfill': 'imputebfill',
                    'impute_ffill': 'imputeffill',
                    'impute_mean': 'imputemean',
                    'impute_median': 'imputemedian',
                    'impute_mode': 'imputemode',
                    'interpolate_linear': 'interpolatelinear',
                    'interpolate_nearest': 'interpolatenearest',
                    'interpolate_cubic': 'interpolatecubic',
                    'interpolate_quadratic': 'interpolatequadratic',
                    'interpolate_polynomial': 'interpolatepolynomial',
                    'interpolate_pchip': 'interpolatepchip',
                    'interpolate_akima': 'interpolateakima',
                    'interpolate_index': 'interpolateindex'
                }
                
                # Stwórz listę metod do wyszukania (oryginalne + zmapowane)
                fixing_methods_to_search = set(main_combinations['fixing_method'].unique())
                for orig_method in main_combinations['fixing_method'].unique():
                    if orig_method in fixing_method_mapping:
                        fixing_methods_to_search.add(fixing_method_mapping[orig_method])
                
                # Filtruj dane różnic według aktualnych filtrów
                filtered_differences = differences_df[
                    (differences_df['dataset_name'].isin(main_combinations['dataset'].unique())) &
                    (differences_df['missing_data_type'].isin(main_combinations['missing_data_type'].unique())) &
                    (differences_df['missing_rate'].isin(missing_rate_converted)) &
                    (differences_df['fixing_method'].isin(fixing_methods_to_search))
                ]
                
                # Debug info
                if st.session_state.get('debug_differences', False):
                    st.write(f"🔍 Debug - filtered_differences shape: {filtered_differences.shape}")
                    if filtered_differences.empty:
                        st.write("🔍 Debug - available dataset_name in differences:", differences_df['dataset_name'].unique())
                        st.write("🔍 Debug - searched datasets:", main_combinations['dataset'].unique())
                        st.write("🔍 Debug - available missing_rate in differences:", differences_df['missing_rate'].unique())
                        st.write("🔍 Debug - searched missing_rate:", missing_rate_converted)
                        st.write("🔍 Debug - available fixing_method in differences:", differences_df['fixing_method'].unique()[:10])
                        st.write("🔍 Debug - searched fixing_method:", list(fixing_methods_to_search))
                
                if filtered_differences.empty:
                    return pd.DataFrame()
                
                # Agregacja według fixing_method - zawsze używamy sumy dla różnic bezwzględnych
                aggregated = filtered_differences.groupby('fixing_method')['difference'].agg('sum')
                
                return aggregated.reset_index()
                
            except Exception as e:
                st.error(f"Error loading df_differences.csv file: {str(e)}")
                import traceback
                st.error(f"Stack trace: {traceback.format_exc()}")
                return pd.DataFrame()
        
        # Oblicz dane dla nowego wykresu
        abs_diff_data = load_and_aggregate_differences(filtered_df)
        
        if not abs_diff_data.empty:
            # Sortuj dane
            abs_diff_sorted = abs_diff_data.sort_values('difference')
            
            # Tworzenie wykresu różnic bezwzględnych
            fig_abs = px.bar(
                abs_diff_sorted,
                x='fixing_method',
                y='difference',
                title='Sum of Absolute Differences',
                color='difference',
                color_continuous_scale='plasma'
            )
            
            fig_abs.update_layout(
                xaxis_tickangle=-45,
                height=600,
                showlegend=False,
                xaxis_title="Fixing Method",
                yaxis_title="Sum of Absolute Differences"
            )
            
            st.plotly_chart(fig_abs, use_container_width=True)
            
        else:
            st.warning("⚠️ Cannot calculate absolute differences for selected filters - check data files availability.")
    
    with col2:
        st.subheader("📊 Statistics")
        best_method = aggregated_data.loc[aggregated_data[metric].idxmin(), 'fixing_method']
        best_value = aggregated_data[metric].min()
        worst_method = aggregated_data.loc[aggregated_data[metric].idxmax(), 'fixing_method']
        worst_value = aggregated_data[metric].max()
        
        st.metric("🥇 Best method", best_method, f"{best_value:.4f}")
        st.metric("🥉 Worst method", worst_method, f"{worst_value:.4f}")
        st.metric("📈 Improvement", "", f"{((worst_value-best_value)/worst_value*100):.1f}%")
        
        # Statystyki filtrów
        st.markdown("### 🔢 Active filters:")
        st.write(f"• Missing rates: {missing_rates}")
        st.write(f"• Data types: {missing_types}")  
        st.write(f"• Datasets: {datasets}")
        st.write(f"• Prediction methods: {prediction_methods}")
        st.write(f"• Records: {len(filtered_df)}")
    
    # TABELA SZCZEGÓŁÓW
    st.subheader("📋 Detailed Results")
    
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
    
    # =====================================
    # COMPREHENSIVE SUMMARY TABLE - ABSOLUTE DIFFERENCES
    # =====================================
    st.markdown("---")
    st.subheader("📊 Comprehensive Summary - Absolute Differences")
    
    # Funkcja do ładowania comprehensive summary
    @st.cache_data
    def load_comprehensive_summary(filtered=True):
        """Ładuje comprehensive summary z pliku CSV"""
        try:
            filename = 'comprehensive_summary_filtered.csv' if filtered else 'comprehensive_summary.csv'
            possible_paths = [
                f'reports/differences_summary/{filename}',
                f'./reports/differences_summary/{filename}',
                f'/home/darek/univariate-time-series-inpainting/reports/differences_summary/{filename}'
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    return pd.read_csv(path)
            
            return None
        except Exception as e:
            st.error(f"Error loading comprehensive summary: {str(e)}")
            return None
    
    # Opcja wyboru wersji tabeli
    use_filtered = st.checkbox(
        "📌 Use filtered version (excludes lake1/lake2/lake3 and gaf/mtf/rp/spec-unet)", 
        value=True,
        key="use_filtered_comprehensive"
    )
    
    # Wczytaj comprehensive summary
    comprehensive_df = load_comprehensive_summary(filtered=use_filtered)
    
    if comprehensive_df is not None:
        version_text = "filtered (no lake datasets, no unet methods)" if use_filtered else "complete"
        st.markdown(f"**Complete table with all combinations ({version_text}):**")
        st.markdown(f"*Dataset × Missingness Type × Missingness Rate × Fixing Method*")
        
        # Dodaj filtry dla tabeli
        col1, col2, col3 = st.columns(3)
        
        with col1:
            filter_datasets = st.multiselect(
                "Filter by Dataset:",
                options=sorted(comprehensive_df['Dataset'].unique()),
                default=sorted(comprehensive_df['Dataset'].unique()),
                key="comp_dataset_filter"
            )
        
        with col2:
            filter_miss_types = st.multiselect(
                "Filter by Missingness Type:",
                options=sorted(comprehensive_df['Missingness_Type'].unique()),
                default=sorted(comprehensive_df['Missingness_Type'].unique()),
                key="comp_misstype_filter"
            )
        
        with col3:
            filter_miss_rates = st.multiselect(
                "Filter by Missingness Rate:",
                options=sorted(comprehensive_df['Missingness_Rate'].unique()),
                default=sorted(comprehensive_df['Missingness_Rate'].unique()),
                key="comp_missrate_filter"
            )
        
        # Filtruj dane
        filtered_comprehensive = comprehensive_df[
            (comprehensive_df['Dataset'].isin(filter_datasets)) &
            (comprehensive_df['Missingness_Type'].isin(filter_miss_types)) &
            (comprehensive_df['Missingness_Rate'].isin(filter_miss_rates))
        ]
        
        # Statystyki
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rows", len(filtered_comprehensive))
        with col2:
            st.metric("Unique Methods", filtered_comprehensive['Fixing_Method'].nunique())
        with col3:
            best_combo = filtered_comprehensive.nsmallest(1, 'Sum_of_Absolute_Differences')
            if not best_combo.empty:
                st.metric("Best Method", best_combo.iloc[0]['Fixing_Method'])
        with col4:
            if not filtered_comprehensive.empty:
                st.metric("Best Score", f"{filtered_comprehensive['Sum_of_Absolute_Differences'].min():.2f}")
        
        # Wyświetl tabelę
        st.dataframe(
            filtered_comprehensive,
            use_container_width=True,
            hide_index=True,
            height=400
        )
        
        # Download button
        csv = filtered_comprehensive.to_csv(index=False)
        st.download_button(
            label="📥 Download filtered table as CSV",
            data=csv,
            file_name="filtered_comprehensive_summary.csv",
            mime="text/csv"
        )
        
    else:
        st.warning("⚠️ Comprehensive summary file not found. Run generate_differences_report.py first.")
    
    # =====================================
    # MACIERZ NAJLEPSZYCH METOD - ABSOLUTE DIFFERENCES
    # =====================================
    st.markdown("---")
    st.subheader("🏆 Best Methods Matrix - Absolute Differences")
    
    # Funkcja do generowania macierzy najlepszych metod
    @st.cache_data
    def generate_best_methods_matrix(filtered_main_df):
        """Generuje macierz pokazującą najlepsze metody dla różnych kombinacji"""
        try:
            # Wczytaj dane różnic
            possible_paths = [
                'df_differences.csv',
                './df_differences.csv', 
                '../df_differences.csv',
                '/home/darek/univariate-time-series-inpainting/df_differences.csv'
            ]
            
            differences_df = None
            for path in possible_paths:
                try:
                    if os.path.exists(path):
                        differences_df = pd.read_csv(path)
                        break
                except:
                    continue
            
            if differences_df is None:
                return None
            
            # Konwersja formatów (podobnie jak wcześniej)
            main_combinations = filtered_main_df[['dataset', 'missing_data_type', 'missing_rate']].drop_duplicates()
            missing_rate_converted = [f"{int(rate)}p" for rate in main_combinations['missing_rate'].unique()]
            
            # Filtruj dane różnic według aktualnych filtrów
            filtered_differences = differences_df[
                (differences_df['dataset_name'].isin(main_combinations['dataset'].unique())) &
                (differences_df['missing_data_type'].isin(main_combinations['missing_data_type'].unique())) &
                (differences_df['missing_rate'].isin(missing_rate_converted))
            ]
            
            if filtered_differences.empty:
                return None
            
            # Generuj macierz najlepszych metod dla dataset x missing_type
            matrix_data = []
            
            for dataset in filtered_differences['dataset_name'].unique():
                row_data = {'Dataset': dataset}
                
                for missing_type in sorted(filtered_differences['missing_data_type'].unique()):
                    subset = filtered_differences[
                        (filtered_differences['dataset_name'] == dataset) & 
                        (filtered_differences['missing_data_type'] == missing_type)
                    ]
                    
                    if not subset.empty:
                        method_totals = subset.groupby('fixing_method')['difference'].sum()
                        best_method = method_totals.idxmin()
                        best_value = method_totals.min()
                        
                        row_data[missing_type] = f"{best_method}\n({best_value:.0f})"
                    else:
                        row_data[missing_type] = "N/A"
                
                matrix_data.append(row_data)
            
            return pd.DataFrame(matrix_data)
            
        except Exception as e:
            st.error(f"Error generating best methods matrix: {str(e)}")
            return None
    
    # Generuj macierz
    best_methods_df = generate_best_methods_matrix(filtered_df)
    
    if best_methods_df is not None and not best_methods_df.empty:
        st.markdown("**Best method (and total difference) for each Dataset × Missing Type combination:**")
        st.dataframe(
            best_methods_df,
            use_container_width=True,
            hide_index=True
        )
        
        # Dodatkowa tabela - ranking metod według częstości wygranych
        st.markdown("---")
        st.markdown("**🥇 Methods Ranking by Number of Wins:**")
        
        # Policz ile razy każda metoda była najlepsza
        wins_count = {}
        for col in best_methods_df.columns:
            if col != 'Dataset':
                for val in best_methods_df[col]:
                    if val != "N/A" and isinstance(val, str):
                        method = val.split('\n')[0]  # Wyciągnij nazwę metody
                        wins_count[method] = wins_count.get(method, 0) + 1
        
        if wins_count:
            wins_df = pd.DataFrame(list(wins_count.items()), columns=['Method', 'Wins'])
            wins_df = wins_df.sort_values('Wins', ascending=False)
            wins_df['Rank'] = range(1, len(wins_df) + 1)
            wins_df = wins_df[['Rank', 'Method', 'Wins']]
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.dataframe(wins_df, use_container_width=True, hide_index=True)
            
            with col2:
                st.metric("🏆 Most Wins", wins_df.iloc[0]['Method'], f"{int(wins_df.iloc[0]['Wins'])} wins")
                if len(wins_df) > 1:
                    st.metric("🥈 Second Place", wins_df.iloc[1]['Method'], f"{int(wins_df.iloc[1]['Wins'])} wins")
    else:
        st.warning("⚠️ Cannot generate best methods matrix - check data availability.")
    
    # PORÓWNANIE METRYK
    if st.checkbox("📊 Show comparison of all metrics"):
        st.subheader("🔄 Comparison of MAE, MAPE, RMSE")
        
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
            title="Normalized Metrics (darker = worse)"
        )
        
        fig_heatmap.update_layout(height=400)
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # =====================================
    # SEKCJA WYŚWIETLANIA OBRAZÓW
    # =====================================
    st.markdown("---")
    st.subheader("🖼️ Image Visualization Comparison")
    
    # KONTROLKI OBRAZÓW - w głównej części strony
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        image_dataset = st.selectbox(
            "Dataset:",
            options=['boiler', 'lake1', 'lake2', 'lake3', 'pump', 'vibr'],
            index=0,
            key="image_dataset"
        )
    
    with col2:
        image_degradation = st.selectbox(
            "Degradation Method:",
            options=['MAR', 'MCAR', 'MNAR'],
            index=1,  # MCAR jako domyślny
            key="image_degradation"
        )
    
    with col3:
        image_percentage = st.selectbox(
            "Degradation Percentage:",
            options=['2p', '5p', '20p'],
            index=0,
            key="image_percentage"
        )
    
    with col4:
        image_iteration = st.selectbox(
            "Iteration:",
            options=[1, 2, 3],
            index=0,
            key="image_iteration"
        )
    
    st.markdown("")  # Dodanie odstępu
    
    # Funkcja do budowania nazw plików
    def build_image_paths(dataset, degradation, percentage, iteration):
        """Buduje ścieżki do plików obrazów dla wszystkich typów i folderów"""
        base_path = "data/images_inpainting"
        image_types = ['gaf', 'spec', 'rp', 'mtf']
        
        paths = {}
        for img_type in image_types:
            paths[img_type] = {
                'original': f"{base_path}/0_original_images/{dataset}_image_{img_type}.png",
                'missing': f"{base_path}/1_missing_images/{dataset}_{degradation}_{percentage}_{iteration}_imagemissing_{img_type}.png",
                'fixed': f"{base_path}/2_fixed_images/{dataset}_{degradation}_{percentage}_{iteration}_imagefixed_{img_type}_unet.png",
                'difference': f"{base_path}/3_difference_images/{dataset}_{degradation}_{percentage}_{iteration}_imagedifference_{img_type}_unet.png"
            }
        return paths
    
    # Funkcja do wyświetlania obrazu z obsługą błędów
    def display_image_safe(image_path, caption, width=150):
        """Bezpiecznie wyświetla obraz z obsługą błędów"""
        if os.path.exists(image_path):
            try:
                image = Image.open(image_path)
                st.image(image, caption=caption, width=width)
            except Exception as e:
                st.error(f"Loading error: {caption}")
        else:
            st.warning(f"File not found: {caption}")
    
    # Budowanie ścieżek dla wybranych parametrów
    image_paths = build_image_paths(image_dataset, image_degradation, image_percentage, image_iteration)
    
    # Tworzenie tabeli obrazów 5x4 (5 kolumn: typ + 4 obrazy, 4 wiersze: gaf/spec/rp/mtf)
    st.markdown("### Image Comparison Table")
    
    # Nagłówki kolumn
    col_header1, col_header2, col_header3, col_header4, col_header5 = st.columns([1, 2, 2, 2, 2])
    with col_header1:
        st.markdown("**Type**")
    with col_header2:
        st.markdown("**Original**")
    with col_header3:
        st.markdown("**Missing**")
    with col_header4:
        st.markdown("**Fixed**")
    with col_header5:
        st.markdown("**Difference**")
    
    # Wyświetlanie obrazów dla każdego typu
    image_types = ['gaf', 'spec', 'rp', 'mtf']
    type_names = {
        'gaf': 'GAF',
        'spec': 'Spectrogram', 
        'rp': 'Recurrence Plot',
        'mtf': 'MTF'
    }
    
    for img_type in image_types:
        col1, col2, col3, col4, col5 = st.columns([1, 2, 2, 2, 2])
        
        with col1:
            st.markdown(f"**{type_names[img_type]}**")
        
        with col2:
            display_image_safe(
                image_paths[img_type]['original'], 
                f"{type_names[img_type]} - Original"
            )
        
        with col3:
            display_image_safe(
                image_paths[img_type]['missing'], 
                f"{type_names[img_type]} - Missing"
            )
        
        with col4:
            display_image_safe(
                image_paths[img_type]['fixed'], 
                f"{type_names[img_type]} - Fixed"
            )
        
        with col5:
            display_image_safe(
                image_paths[img_type]['difference'], 
                f"{type_names[img_type]} - Difference"
            )
    
    # =====================================
    # SEKCJA WYKRESU LINIOWEGO PORÓWNAWCZEGO
    # =====================================
    st.markdown("---")
    st.subheader("📈 Time Series Comparison Chart")
    
    # Filtr metody uzupełniania dla trzeciej linii
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    
    with col1:
        fixing_method = st.selectbox(
            "Fixing method:",
            options=['gaf-unet', 'mtf-unet', 'rp-unet', 'spec-unet'],
            index=0,
            key="chart_fixing_method"
        )
    
    # Mapowanie nazw datasetu na nazwy plików
    dataset_file_mapping = {
        'boiler': 'boiler_outlet_temp_univ.csv',
        'lake1': 'water_level_sensors_2010_L300.csv',
        'lake2': 'water_level_sensors_2010_L308.csv', 
        'lake3': 'water_level_sensors_2010_L311.csv',
        'pump': 'pump_sensor_28_univ.csv', 
        'vibr': 'vibration_sensor_S1.csv'
    }
    
    # Funkcja do wczytywania danych szeregów czasowych
    @st.cache_data
    def load_time_series_data(file_path):
        """Wczytuje dane szeregu czasowego z obsługą błędów"""
        try:
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                # Sprawdź czy jest kolumna z czasem/datą
                if len(df.columns) >= 2:
                    df.columns = ['timestamp', 'value'] if len(df.columns) == 2 else list(df.columns)
                    # Spróbuj przekonwertować pierwszą kolumnę na datetime jeśli to możliwe
                    try:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                    except:
                        # Jeśli nie można, użyj indeksu jako timestamp
                        df['timestamp'] = pd.to_datetime(df.index, unit='D', origin='2020-01-01')
                    return df
                else:
                    return None
            else:
                return None
        except Exception as e:
            st.error(f"Error loading file {file_path}: {str(e)}")
            return None
    
    # Budowanie ścieżek do plików danych
    def build_data_paths(dataset, degradation, percentage, iteration, fixing_method):
        """Buduje ścieżki do plików danych czasowych"""
        base_path = "data"
        
        # Oryginalne dane
        original_file = dataset_file_mapping.get(dataset, f"{dataset}_univ.csv")
        original_path = f"{base_path}/0_source_data/{original_file}"
        
        # Dane z brakującymi wartościami
        missing_path = f"{base_path}/1_missing_data/{dataset}_{degradation}_{percentage}_{iteration}.csv"
        
        # Dane po uzupełnieniu - przekształć nazwę metody
        method_suffix = fixing_method.replace('-', '')  # gaf-unet -> gafunet
        fixed_path = f"{base_path}/2_fixed_data/{dataset}_{degradation}_{percentage}_{iteration}_{method_suffix}.csv"
        
        return {
            'original': original_path,
            'missing': missing_path, 
            'fixed': fixed_path
        }
    
    # Wczytanie danych na podstawie aktualnych filtrów
    data_paths = build_data_paths(
        image_dataset, 
        image_degradation, 
        image_percentage, 
        image_iteration,
        fixing_method
    )
    
    # Wczytanie wszystkich trzech szeregów
    original_data = load_time_series_data(data_paths['original'])
    missing_data = load_time_series_data(data_paths['missing'])
    fixed_data = load_time_series_data(data_paths['fixed'])
    
    # Sprawdzenie czy dane zostały wczytane
    if original_data is not None and missing_data is not None and fixed_data is not None:
        
        # Wyrównanie długości szeregów (wszystkie do najmniejszego)
        min_length = min(len(original_data), len(missing_data), len(fixed_data))
        
        original_trimmed = original_data.head(min_length).copy()
        missing_trimmed = missing_data.head(min_length).copy()
        fixed_trimmed = fixed_data.head(min_length).copy()
        
        # Utworzenie wykresu plotly
        fig = go.Figure()
        
        # Dodanie linii oryginalnych danych
        fig.add_trace(go.Scatter(
            x=original_trimmed['timestamp'],
            y=original_trimmed['value'],
            mode='lines',
            name='Original data',
            line=dict(color='blue', width=1.5)
        ))
        
        # Dodanie linii z brakującymi danymi
        fig.add_trace(go.Scatter(
            x=missing_trimmed['timestamp'],
            y=missing_trimmed['value'],
            mode='lines',
            name='Data with missing values',
            line=dict(color='red', width=1.5)
        ))
        
        # Dodanie linii uzupełnionych danych
        fig.add_trace(go.Scatter(
            x=fixed_trimmed['timestamp'],
            y=fixed_trimmed['value'],
            mode='lines',
            name=f'Imputed ({fixing_method})',
            line=dict(color='green', width=1.5)
        ))
        
        # Konfiguracja wykresu
        fig.update_layout(
            title=f'Time Series Comparison - {image_dataset.upper()} ({image_degradation}, {image_percentage}, iter: {image_iteration})',
            xaxis_title='Time',
            yaxis_title='Value',
            height=500,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Wyświetlenie wykresu
        st.plotly_chart(fig, use_container_width=True)
        
        # Wyświetlenie informacji o plikach
        with st.expander("📁 Data file information"):
            st.write("**File paths:**")
            for data_type, path in data_paths.items():
                status = "✅" if os.path.exists(path) else "❌"
                st.write(f"• {data_type.title()}: `{path}` {status}")
        
        # Statystyki porównawcze
        with st.expander("📊 Comparison statistics"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Original mean",
                    f"{original_trimmed['value'].mean():.3f}",
                    f"σ: {original_trimmed['value'].std():.3f}"
                )
            
            with col2:
                # Policz ile jest wartości NaN w danych z brakami
                missing_count = missing_trimmed['value'].isna().sum()
                missing_percent = (missing_count / len(missing_trimmed)) * 100
                st.metric(
                    "Data with missing values",
                    f"{missing_trimmed['value'].mean():.3f}",
                    f"Missing: {missing_percent:.1f}%"
                )
            
            with col3:
                # Oblicz MAE między oryginalnymi a uzupełnionymi
                mae = np.mean(np.abs(original_trimmed['value'] - fixed_trimmed['value']))
                st.metric(
                    f"Imputed ({fixing_method})",
                    f"{fixed_trimmed['value'].mean():.3f}",
                    f"MAE: {mae:.3f}"
                )
    
    else:
        # Komunikat o błędzie wczytywania danych
        st.error("❌ Cannot load all required data files!")
        
        with st.expander("🔍 Check file availability"):
            for data_type, path in data_paths.items():
                status = "✅ Exists" if os.path.exists(path) else "❌ Missing"
                st.write(f"• **{data_type.title()}**: `{path}` - {status}")
    
    # =====================================
    # SEKCJA HISTOGRAMU MAPE
    # =====================================
    st.markdown("---")
    st.subheader("📊 MAPE Distribution Histogram")
    
    # Wczytanie danych df_final
    @st.cache_data
    def load_final_data():
        """Wczytuje dane z df_final.csv"""
        try:
            return pd.read_csv('results/quick_experiment/df_final.csv')
        except Exception as e:
            st.error(f"Error loading df_final.csv: {str(e)}")
            return None
    
    df_final = load_final_data()
    
    if df_final is not None:
        # Filtry dla histogramu
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            # Filtr dataset (pojedynczy wybór)
            hist_dataset = st.selectbox(
                "Dataset:",
                options=sorted(df_final['dataset'].unique()),
                index=0,
                key="hist_dataset"
            )
        
        with col2:
            # Filtr fixing_method (wielokrotny wybór)
            hist_fixing_methods = st.multiselect(
                "Fixing Methods:",
                options=sorted(df_final['fixing_method'].unique()),
                default=sorted(df_final['fixing_method'].unique())[:5],  # Pierwsze 5 jako domyślne
                key="hist_fixing_methods"
            )
        
        with col3:
            # Wielkość bina
            bin_size = st.slider(
                "Bin size:",
                min_value=0.001,
                max_value=1.0,
                value=0.01,
                step=0.001,
                format="%.3f",
                key="hist_bin_size"
            )
        
        if hist_fixing_methods:  # Sprawdź czy wybrano jakieś metody
            # Filtrowanie danych
            filtered_final_df = df_final[
                (df_final['dataset'] == hist_dataset) &
                (df_final['fixing_method'].isin(hist_fixing_methods))
            ]
            
            if not filtered_final_df.empty:
                # Tworzenie histogramu
                fig_hist = go.Figure()
                
                # Palette kolorów
                colors = [
                    'rgba(31, 119, 180, 0.6)',   # niebieski
                    'rgba(255, 127, 14, 0.6)',   # pomarańczowy  
                    'rgba(44, 160, 44, 0.6)',    # zielony
                    'rgba(214, 39, 40, 0.6)',    # czerwony
                    'rgba(148, 103, 189, 0.6)',  # fioletowy
                    'rgba(140, 86, 75, 0.6)',    # brązowy
                    'rgba(227, 119, 194, 0.6)',  # różowy
                    'rgba(127, 127, 127, 0.6)',  # szary
                    'rgba(188, 189, 34, 0.6)',   # oliwkowy
                    'rgba(23, 190, 207, 0.6)'    # cyjan
                ]
                
                # Obliczenie zakresów binów
                mape_min = filtered_final_df['MAPE'].min()
                mape_max = filtered_final_df['MAPE'].max()
                
                # Dodanie histogramu dla każdej fixing_method
                for i, method in enumerate(hist_fixing_methods):
                    method_data = filtered_final_df[filtered_final_df['fixing_method'] == method]
                    
                    if not method_data.empty:
                        fig_hist.add_trace(go.Histogram(
                            x=method_data['MAPE'],
                            name=method,
                            opacity=0.6,
                            marker_color=colors[i % len(colors)],
                            xbins=dict(
                                start=mape_min,
                                end=mape_max + bin_size,
                                size=bin_size
                            )
                        ))
                
                # Konfiguracja wykresu
                fig_hist.update_layout(
                    title=f'MAPE Distribution for Dataset: {hist_dataset} (bin size: {bin_size})',
                    xaxis_title='MAPE',
                    yaxis_title='Count',
                    height=500,
                    barmode='overlay',  # Histogramy nachodzące na siebie
                    hovermode='x unified',
                    legend=dict(
                        orientation="v",
                        yanchor="top",
                        y=1,
                        xanchor="left",
                        x=1.02
                    )
                )
                
                # Wyświetlenie histogramu
                st.plotly_chart(fig_hist, use_container_width=True)
                
                # Statystyki szczegółowe
                with st.expander("📈 MAPE statistics for selected methods"):
                    stats_df = filtered_final_df.groupby('fixing_method')['MAPE'].agg([
                        'count', 'mean', 'median', 'std', 'min', 'max'
                    ]).round(4)
                    stats_df.columns = ['Count', 'Mean', 'Median', 'Std. dev.', 'Min', 'Max']
                    st.dataframe(stats_df, use_container_width=True)
                
                # Informacje o filtracji
                st.info(f"📊 Displayed {len(filtered_final_df)} records for dataset: **{hist_dataset}** "
                       f"and {len(hist_fixing_methods)} selected methods.")
            
            else:
                st.warning("⚠️ No data for selected filters!")
        
        else:
            st.warning("⚠️ Select at least one fixing_method!")
    
    else:
        st.error("❌ Cannot load df_final.csv file!")

# Uruchamianie interaktywnej części tylko gdy jest uruchamiana jako Streamlit app
if __name__ == "__main__":
    # Sprawdź czy to Streamlit
    try:
        import streamlit as st
        # Jeśli streamlit jest dostępny i uruchamiamy jako app
        if hasattr(st, 'get_option'):
            create_interactive_app()
    except ImportError:
        print("Streamlit is not installed. Install via: pip install streamlit plotly")
        print("Run interactive application via: streamlit run visualization.py")