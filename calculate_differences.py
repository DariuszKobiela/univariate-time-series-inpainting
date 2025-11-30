#!/usr/bin/env python3
"""
Skrypt do porównywania plików z data/2_fixed_data z odpowiadającymi plikami z data/0_source_data
i obliczania sum różnic bezwzględnych między wartościami.
"""

import os
import csv
import pandas as pd
from pathlib import Path
import sys


def get_dataset_mapping():
    """Mapowanie nazw dataset'ów na pliki źródłowe"""
    return {
        "boiler": "data/0_source_data/boiler_outlet_temp_univ.csv",
        "pump": "data/0_source_data/pump_sensor_28_univ.csv",
        "vibr": "data/0_source_data/vibration_sensor_S1.csv",
        "lake1": "data/0_source_data/water_level_sensors_2010_L300.csv",
        "lake2": "data/0_source_data/water_level_sensors_2010_L308.csv",
        "lake3": "data/0_source_data/water_level_sensors_2010_L311.csv"
    }


def parse_filename(filename):
    """
    Parsuje nazwę pliku z data/2_fixed_data do składowych.
    Format: datasetName_missingDataType_missingRate_iterationNr_fixingMethod.csv
    
    Returns:
        tuple: (dataset_name, missing_data_type, missing_rate, iteration_nr, fixing_method)
    """
    # Usuń rozszerzenie .csv
    name_without_ext = filename.replace('.csv', '')
    
    # Podziel po '_'
    parts = name_without_ext.split('_')
    
    if len(parts) < 5:
        raise ValueError(f"Nieprawidłowy format nazwy pliku: {filename}")
    
    dataset_name = parts[0]
    missing_data_type = parts[1]
    missing_rate = parts[2]
    iteration_nr = parts[3]
    fixing_method = '_'.join(parts[4:])  # fixing_method może zawierać '_'
    
    return dataset_name, missing_data_type, missing_rate, iteration_nr, fixing_method


def compare_files(source_file_path, fixed_file_path):
    """
    Porównuje dwa pliki CSV linijka po linijce i oblicza sumę różnic bezwzględnych.
    
    Args:
        source_file_path: ścieżka do pliku źródłowego
        fixed_file_path: ścieżka do pliku naprawionego
        
    Returns:
        float: suma różnic bezwzględnych
    """
    try:
        # Wczytaj plik źródłowy - sprawdź czy zawiera lake (używa ; i przecinki dziesiętne)
        if 'water_level_sensors' in source_file_path:
            source_df = pd.read_csv(source_file_path, sep=';', decimal=',')
        else:
            source_df = pd.read_csv(source_file_path)
        
        # Wczytaj plik fixed (zawsze używa , jako separator i . jako przecinek dziesiętny)
        fixed_df = pd.read_csv(fixed_file_path)
        
        # Sprawdź czy pliki mają taką samą liczbę wierszy
        if len(source_df) != len(fixed_df):
            print(f"UWAGA: Różna liczba wierszy - source: {len(source_df)}, fixed: {len(fixed_df)}")
            # Weź minimum z obu
            min_len = min(len(source_df), len(fixed_df))
            source_df = source_df.head(min_len)
            fixed_df = fixed_df.head(min_len)
        
        # Sprawdź czy są jakieś dane
        if len(source_df) == 0 or len(fixed_df) == 0:
            print(f"UWAGA: Jeden z plików jest pusty - source: {len(source_df)}, fixed: {len(fixed_df)}")
            return None
        
        # Przyjmij, że pierwsza kolumna to timestamp, druga to wartość
        source_values = pd.to_numeric(source_df.iloc[:, 1], errors='coerce')
        fixed_values = pd.to_numeric(fixed_df.iloc[:, 1], errors='coerce')
        
        # Usuń wartości NaN
        valid_mask = ~(source_values.isna() | fixed_values.isna())
        source_values = source_values[valid_mask]
        fixed_values = fixed_values[valid_mask]
        
        if len(source_values) == 0:
            print(f"UWAGA: Brak poprawnych wartości liczbowych w plikach")
            return None
        
        # Oblicz różnice bezwzględne
        absolute_differences = (source_values - fixed_values).abs()
        
        # Sumuj różnice
        total_difference = absolute_differences.sum()
        
        return total_difference
        
    except Exception as e:
        print(f"Błąd podczas porównywania plików {source_file_path} i {fixed_file_path}: {e}")
        return None


def load_existing_results(output_file):
    """
    Wczytuje istniejące wyniki z pliku CSV.
    
    Returns:
        tuple: (existing_results list, existing_combinations set)
    """
    existing_results = []
    existing_combinations = set()
    
    if Path(output_file).exists():
        try:
            df = pd.read_csv(output_file)
            existing_results = df.to_dict('records')
            
            # Stwórz zbiór kombinacji już przetworzonych
            for row in existing_results:
                key = (
                    row['dataset_name'],
                    row['missing_data_type'],
                    row['missing_rate'],
                    row['iteration_nr'],
                    row['fixing_method']
                )
                existing_combinations.add(key)
            
            print(f"✓ Wczytano {len(existing_results)} istniejących wyników z {output_file}")
        except Exception as e:
            print(f"Uwaga: Nie można wczytać istniejących wyników: {e}")
    else:
        print(f"Plik {output_file} nie istnieje - zostaną stworzone nowe wyniki")
    
    return existing_results, existing_combinations


def main():
    """Główna funkcja skryptu"""
    
    # Ścieżki do katalogów
    fixed_data_dir = Path("data/2_fixed_data")
    output_file = "df_differences.csv"
    
    # Mapowanie dataset'ów
    dataset_mapping = get_dataset_mapping()
    
    # Wczytaj istniejące wyniki
    existing_results, existing_combinations = load_existing_results(output_file)
    
    # Lista do przechowywania NOWYCH wyników
    new_results = []
    skipped_count = 0
    processed_count = 0
    
    print("\nRozpoczynam porównywanie plików...")
    
    # Przejdź przez wszystkie pliki w data/2_fixed_data
    for fixed_file in fixed_data_dir.glob("*.csv"):
        filename = fixed_file.name
        
        try:
            # Parsuj nazwę pliku
            dataset_name, missing_data_type, missing_rate, iteration_nr, fixing_method = parse_filename(filename)
            
            # Sprawdź czy ta kombinacja już została przetworzona
            key = (dataset_name, missing_data_type, missing_rate, iteration_nr, fixing_method)
            if key in existing_combinations:
                skipped_count += 1
                continue
            
            # Znajdź odpowiadający plik źródłowy
            if dataset_name not in dataset_mapping:
                print(f"Nieznany dataset: {dataset_name} w pliku {filename}")
                continue
                
            source_file_path = dataset_mapping[dataset_name]
            
            # Sprawdź czy plik źródłowy istnieje
            if not Path(source_file_path).exists():
                print(f"Plik źródłowy nie istnieje: {source_file_path}")
                continue
            
            # Porównaj pliki
            difference = compare_files(source_file_path, str(fixed_file))
            
            if difference is not None:
                # Dodaj wynik do listy NOWYCH wyników
                new_results.append({
                    'dataset_name': dataset_name,
                    'missing_data_type': missing_data_type,
                    'missing_rate': missing_rate,
                    'iteration_nr': iteration_nr,
                    'fixing_method': fixing_method,
                    'difference': difference
                })
                
                processed_count += 1
                print(f"[{processed_count}] ✓ {filename} -> {difference:.2f}")
            
        except Exception as e:
            print(f"Błąd podczas przetwarzania pliku {filename}: {e}")
            continue
    
    # Połącz stare i nowe wyniki
    all_results = existing_results + new_results
    
    # Zapisz WSZYSTKIE wyniki do CSV
    if all_results:
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['dataset_name', 'missing_data_type', 'missing_rate', 'iteration_nr', 'fixing_method', 'difference']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in all_results:
                writer.writerow(result)
        
        print(f"\n✓ Wyniki zapisane do pliku: {output_file}")
        print(f"  - Istniejących wyników: {len(existing_results)}")
        print(f"  - Nowych wyników: {len(new_results)}")
        print(f"  - Pominiętych (już policzonych): {skipped_count}")
        print(f"  - Łącznie w pliku: {len(all_results)}")
    else:
        print("Brak wyników do zapisania.")


if __name__ == "__main__":
    main()
