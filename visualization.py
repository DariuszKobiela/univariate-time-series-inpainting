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
    import os
    from PIL import Image
    
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
    
    # Debug mode
    debug_mode = st.sidebar.checkbox("🐛 Tryb debug dla różnic bezwzględnych", value=False)
    if debug_mode:
        st.session_state['debug_differences'] = True
    else:
        st.session_state['debug_differences'] = False
    
    # Wybór metryki
    metric = st.sidebar.selectbox(
        "📊 Wybierz metrykę:",
        options=['MAE', 'MAPE', 'RMSE'],
        index=1
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
        
        # =====================================
        # NOWY WYKRES - SUMA RÓŻNIC BEZWZGLĘDNYCH
        # =====================================
        st.subheader("📊 Suma Różnic Bezwzględnych według Fixing Method")
        
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
                                st.write(f"🔍 Debug - wczytano dane z: {path}")
                            break
                    except:
                        continue
                
                if differences_df is None:
                    raise FileNotFoundError("Nie znaleziono pliku df_differences.csv w żadnej z ścieżek")
                
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
                        st.write("🔍 Debug - dostępne dataset_name w differences:", differences_df['dataset_name'].unique())
                        st.write("🔍 Debug - szukane datasety:", main_combinations['dataset'].unique())
                        st.write("🔍 Debug - dostępne missing_rate w differences:", differences_df['missing_rate'].unique())
                        st.write("🔍 Debug - szukane missing_rate:", missing_rate_converted)
                        st.write("🔍 Debug - dostępne fixing_method w differences:", differences_df['fixing_method'].unique()[:10])
                        st.write("🔍 Debug - szukane fixing_method:", list(fixing_methods_to_search))
                
                if filtered_differences.empty:
                    return pd.DataFrame()
                
                # Agregacja według fixing_method - zawsze używamy sumy dla różnic bezwzględnych
                aggregated = filtered_differences.groupby('fixing_method')['difference'].agg('sum')
                
                return aggregated.reset_index()
                
            except Exception as e:
                st.error(f"Błąd wczytywania pliku df_differences.csv: {str(e)}")
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
                title='Suma Różnic Bezwzględnych (Filtered Data)',
                color='difference',
                color_continuous_scale='plasma',
                text='difference'
            )
            
            # Formatowanie wykresu
            fig_abs.update_traces(
                texttemplate='%{text:.0f}',
                textposition='outside'
            )
            
            fig_abs.update_layout(
                xaxis_tickangle=-45,
                height=600,
                showlegend=False,
                xaxis_title="Fixing Method",
                yaxis_title="Suma Różnic Bezwzględnych"
            )
            
            st.plotly_chart(fig_abs, use_container_width=True)
            
        else:
            st.warning("⚠️ Nie można obliczyć różnic bezwzględnych dla wybranych filtrów - sprawdź dostępność plików danych.")
    
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
        st.write(f"• Prediction methods: {prediction_methods}")
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
    
    # =====================================
    # SEKCJA WYŚWIETLANIA OBRAZÓW
    # =====================================
    st.markdown("---")
    st.subheader("🖼️ Porównanie Wizualizacji Obrazowych")
    
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
            "Metoda Degradacji:",
            options=['MAR', 'MCAR', 'MNAR'],
            index=1,  # MCAR jako domyślny
            key="image_degradation"
        )
    
    with col3:
        image_percentage = st.selectbox(
            "Procent Degradacji:",
            options=['2p', '5p', '20p'],
            index=0,
            key="image_percentage"
        )
    
    with col4:
        image_iteration = st.selectbox(
            "Iteracja:",
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
                st.error(f"Błąd wczytywania: {caption}")
        else:
            st.warning(f"Brak pliku: {caption}")
    
    # Budowanie ścieżek dla wybranych parametrów
    image_paths = build_image_paths(image_dataset, image_degradation, image_percentage, image_iteration)
    
    # Tworzenie tabeli obrazów 5x4 (5 kolumn: typ + 4 obrazy, 4 wiersze: gaf/spec/rp/mtf)
    st.markdown("### Tabela Porównawcza Obrazów")
    
    # Nagłówki kolumn
    col_header1, col_header2, col_header3, col_header4, col_header5 = st.columns([1, 2, 2, 2, 2])
    with col_header1:
        st.markdown("**Typ**")
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
    st.subheader("📈 Wykres Porównawczy Szeregów Czasowych")
    
    # Filtr metody uzupełniania dla trzeciej linii
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    
    with col1:
        fixing_method = st.selectbox(
            "Metoda uzupełniania:",
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
            st.error(f"Błąd wczytywania pliku {file_path}: {str(e)}")
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
            name='Oryginalne dane',
            line=dict(color='blue', width=1.5)
        ))
        
        # Dodanie linii z brakującymi danymi
        fig.add_trace(go.Scatter(
            x=missing_trimmed['timestamp'],
            y=missing_trimmed['value'],
            mode='lines',
            name='Dane z brakami',
            line=dict(color='red', width=1.5)
        ))
        
        # Dodanie linii uzupełnionych danych
        fig.add_trace(go.Scatter(
            x=fixed_trimmed['timestamp'],
            y=fixed_trimmed['value'],
            mode='lines',
            name=f'Uzupełnione ({fixing_method})',
            line=dict(color='green', width=1.5)
        ))
        
        # Konfiguracja wykresu
        fig.update_layout(
            title=f'Porównanie Szeregów Czasowych - {image_dataset.upper()} ({image_degradation}, {image_percentage}, iter: {image_iteration})',
            xaxis_title='Czas',
            yaxis_title='Wartość',
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
        with st.expander("📁 Informacje o plikach danych"):
            st.write("**Ścieżki do plików:**")
            for data_type, path in data_paths.items():
                status = "✅" if os.path.exists(path) else "❌"
                st.write(f"• {data_type.title()}: `{path}` {status}")
        
        # Statystyki porównawcze
        with st.expander("📊 Statystyki porównawcze"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Średnia oryginalna",
                    f"{original_trimmed['value'].mean():.3f}",
                    f"σ: {original_trimmed['value'].std():.3f}"
                )
            
            with col2:
                # Policz ile jest wartości NaN w danych z brakami
                missing_count = missing_trimmed['value'].isna().sum()
                missing_percent = (missing_count / len(missing_trimmed)) * 100
                st.metric(
                    "Dane z brakami",
                    f"{missing_trimmed['value'].mean():.3f}",
                    f"Brakuje: {missing_percent:.1f}%"
                )
            
            with col3:
                # Oblicz MAE między oryginalnymi a uzupełnionymi
                mae = np.mean(np.abs(original_trimmed['value'] - fixed_trimmed['value']))
                st.metric(
                    f"Uzupełnione ({fixing_method})",
                    f"{fixed_trimmed['value'].mean():.3f}",
                    f"MAE: {mae:.3f}"
                )
    
    else:
        # Komunikat o błędzie wczytywania danych
        st.error("❌ Nie można wczytać wszystkich wymaganych plików danych!")
        
        with st.expander("🔍 Sprawdź dostępność plików"):
            for data_type, path in data_paths.items():
                status = "✅ Istnieje" if os.path.exists(path) else "❌ Brak"
                st.write(f"• **{data_type.title()}**: `{path}` - {status}")
    
    # =====================================
    # SEKCJA HISTOGRAMU MAPE
    # =====================================
    st.markdown("---")
    st.subheader("📊 Histogram Rozkładu MAPE")
    
    # Wczytanie danych df_final
    @st.cache_data
    def load_final_data():
        """Wczytuje dane z df_final.csv"""
        try:
            return pd.read_csv('results/quick_experiment/df_final.csv')
        except Exception as e:
            st.error(f"Błąd wczytywania df_final.csv: {str(e)}")
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
                "Rozmiar bina:",
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
                    title=f'Rozkład MAPE dla Dataset: {hist_dataset} (bin size: {bin_size})',
                    xaxis_title='MAPE',
                    yaxis_title='Liczba wystąpień',
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
                with st.expander("📈 Statystyki MAPE dla wybranych metod"):
                    stats_df = filtered_final_df.groupby('fixing_method')['MAPE'].agg([
                        'count', 'mean', 'median', 'std', 'min', 'max'
                    ]).round(4)
                    stats_df.columns = ['Liczba', 'Średnia', 'Mediana', 'Odch. std.', 'Min', 'Max']
                    st.dataframe(stats_df, use_container_width=True)
                
                # Informacje o filtracji
                st.info(f"📊 Wyświetlono {len(filtered_final_df)} rekordów dla dataset: **{hist_dataset}** "
                       f"i {len(hist_fixing_methods)} wybranych metod.")
            
            else:
                st.warning("⚠️ Brak danych dla wybranych filtrów!")
        
        else:
            st.warning("⚠️ Wybierz przynajmniej jedną metodę fixing_method!")
    
    else:
        st.error("❌ Nie można wczytać pliku df_final.csv!")

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