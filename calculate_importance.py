import os
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
from tqdm import tqdm

# Mapowanie nazw datasetów
DATASET_MAPPING = {
    'boiler': 'boiler_outlet_temp_univ.csv',
    'pump': 'pump_sensor_28_univ.csv',
    'vibr': 'vibration_sensor_S1.csv',
    'lake1': 'water_level_sensors_2010_L300.csv',
    'lake2': 'water_level_sensors_2010_L308.csv',
    'lake3': 'water_level_sensors_2010_L311.csv'
}

def parse_fixed_filename(filename):
    """
    Parsuje nazwę pliku z 2_fixed_data.
    Format: datasetname_missingnessType_missingPercentage_iterationNr_fixingMethod.csv
    Przykład: boiler_MAR_20p_1_gafsd2all4.csv
    """
    parts = filename.replace('.csv', '').split('_')
    
    # Dataset name (może być wieloczęściowy, np. lake1)
    dataset_name = parts[0]
    
    # Missingness type (MAR, MCAR, MNAR)
    missingness_type = parts[1]
    
    # Missing percentage (np. 20p, 5p)
    missing_percentage = parts[2]
    
    # Iteration number
    iteration_nr = parts[3]
    
    # Fixing method (reszta nazwy po ostatnim podkreślniku)
    fixing_method = '_'.join(parts[4:])
    
    return {
        'dataset_name': dataset_name,
        'missingness_type': missingness_type,
        'missing_percentage': missing_percentage,
        'iteration_nr': iteration_nr,
        'fixing_method': fixing_method
    }

def get_corresponding_files(parsed_info, base_dir):
    """
    Zwraca ścieżki do odpowiadających plików w 0_source_data i 1_missing_data.
    """
    dataset_name = parsed_info['dataset_name']
    missingness_type = parsed_info['missingness_type']
    missing_percentage = parsed_info['missing_percentage']
    iteration_nr = parsed_info['iteration_nr']
    
    # Plik źródłowy
    source_filename = DATASET_MAPPING.get(dataset_name)
    if source_filename is None:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
    
    source_path = os.path.join(base_dir, '0_source_data', source_filename)
    
    # Plik z brakującymi danymi
    missing_filename = f"{dataset_name}_{missingness_type}_{missing_percentage}_{iteration_nr}.csv"
    missing_path = os.path.join(base_dir, '1_missing_data', missing_filename)
    
    return source_path, missing_path

def calculate_ttest_for_file(fixed_path, missing_path, source_path):
    """
    Oblicza t-test dla jednego zestawu plików.
    
    Dla każdej brakującej wartości w missing_data:
    - Pobiera oryginalną wartość z source_data
    - Pobiera naprawioną wartość z fixed_data
    - Porównuje je za pomocą paired t-test
    """
    try:
        # Wczytaj dane
        df_source = pd.read_csv(source_path)
        df_missing = pd.read_csv(missing_path)
        df_fixed = pd.read_csv(fixed_path)
        
        # Znajdź indeksy z brakującymi wartościami
        # Zakładam, że wartości są w drugiej kolumnie (kolumna 1)
        value_column = df_missing.columns[1]
        missing_mask = df_missing[value_column].isna()
        missing_indices = missing_mask[missing_mask].index
        
        if len(missing_indices) == 0:
            return None, 0, "No missing values found"
        
        # Zbierz oryginalne i naprawione wartości dla brakujących pozycji
        original_values = df_source.loc[missing_indices, value_column].values
        fixed_values = df_fixed.loc[missing_indices, value_column].values
        
        # Usuń NaN jeśli jakieś występują w wartościach naprawionych lub oryginalnych
        valid_mask = ~(np.isnan(original_values) | np.isnan(fixed_values))
        original_values = original_values[valid_mask]
        fixed_values = fixed_values[valid_mask]
        
        if len(original_values) < 2:
            return None, len(missing_indices), "Not enough valid values for t-test"
        
        # Wykonaj paired t-test
        # Testujemy czy fixed_values różnią się od original_values
        t_statistic, p_value = stats.ttest_rel(original_values, fixed_values)
        
        return p_value, len(missing_indices), None
        
    except Exception as e:
        return None, 0, str(e)

def main():
    # Ścieżki
    base_dir = '/home/darek/univariate-time-series-inpainting/data'
    fixed_data_dir = os.path.join(base_dir, '2_fixed_data')
    output_file = '/home/darek/univariate-time-series-inpainting/importance.csv'
    
    # Lista wszystkich plików w 2_fixed_data
    fixed_files = sorted([f for f in os.listdir(fixed_data_dir) if f.endswith('.csv')])
    
    print(f"Znaleziono {len(fixed_files)} plików do przetworzenia")
    
    # Lista wyników
    results = []
    errors = []
    
    # Przetwarzaj każdy plik
    for fixed_filename in tqdm(fixed_files, desc="Processing files"):
        try:
            # Parsuj nazwę pliku
            parsed_info = parse_fixed_filename(fixed_filename)
            
            # Znajdź odpowiednie pliki
            source_path, missing_path = get_corresponding_files(parsed_info, base_dir)
            fixed_path = os.path.join(fixed_data_dir, fixed_filename)
            
            # Sprawdź czy pliki istnieją
            if not os.path.exists(source_path):
                errors.append(f"Source file not found: {source_path}")
                continue
            if not os.path.exists(missing_path):
                errors.append(f"Missing file not found: {missing_path}")
                continue
            
            # Oblicz t-test
            p_value, n_missing, error = calculate_ttest_for_file(
                fixed_path, missing_path, source_path
            )
            
            if error:
                errors.append(f"{fixed_filename}: {error}")
                continue
            
            # Dodaj wynik
            results.append({
                'dataset_name': parsed_info['dataset_name'],
                'missingness_type': parsed_info['missingness_type'],
                'missing_percentage': parsed_info['missing_percentage'],
                'iteration_nr': parsed_info['iteration_nr'],
                'fixing_method': parsed_info['fixing_method'],
                'p_value': p_value,
                't_test_result': p_value,
                'n_missing_values': n_missing
            })
            
        except Exception as e:
            errors.append(f"{fixed_filename}: {str(e)}")
    
    # Zapisz wyniki do CSV
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_file, index=False)
    
    print(f"\nPrzetworzono {len(results)} plików pomyślnie")
    print(f"Wyniki zapisano do: {output_file}")
    
    if errors:
        print(f"\nWystąpiło {len(errors)} błędów:")
        for error in errors[:10]:  # Pokaż tylko pierwsze 10 błędów
            print(f"  - {error}")
        if len(errors) > 10:
            print(f"  ... i {len(errors) - 10} więcej")
    
    # Pokaż statystyki
    if len(results) > 0:
        print("\nStatystyki:")
        print(f"Średnia p-value: {df_results['p_value'].mean():.6f}")
        print(f"Mediana p-value: {df_results['p_value'].median():.6f}")
        print(f"Liczba wyników z p < 0.05: {(df_results['p_value'] < 0.05).sum()}")

if __name__ == "__main__":
    main()

