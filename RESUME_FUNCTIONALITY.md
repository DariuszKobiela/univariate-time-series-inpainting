# Funkcjonalność wznowienia eksperymentu

## Jak to działa?

Dodałem mechanizm **automatycznego pomijania już obliczonych kombinacji**. Teraz możesz bezpiecznie uruchomić eksperyment ponownie - system automatycznie:

### 1. Przy starcie Phase 3 (Forecasting):
- Wczytuje istniejący plik `df_final.csv` (jeśli istnieje)
- Tworzy zestaw wszystkich już obliczonych kombinacji
- Format kombinacji: `(dataset, missing_data_type, missing_rate, iteration_nr, fixing_method, prediction_method)`

### 2. Podczas obliczeń:
- **Phase 1 & 2**: Już miały sprawdzanie czy pliki istnieją - nie zmieniłem tego
- **Phase 3**: Przed każdym forecasting sprawdza czy kombinacja już istnieje
  - Jeśli TAK → ⏭️ Pomija (wyświetla "Skipping... (already computed)")
  - Jeśli NIE → 🔮 Oblicza nowe wyniki
- **Phase 4**: Sprawdza czy wszystkie kombinacje "original" dla danego datasetu+modelu istnieją
  - Jeśli TAK → Pomija cały dataset+model
  - Jeśli NIE → Oblicza i dodaje tylko brakujące kombinacje

### 3. Na końcu:
- Łączy nowe wyniki z istniejącym `df_final.csv`
- Usuwa duplikaty (zachowuje najnowsze obliczenia)
- Zapisuje połączony plik

## Przykładowe wyjście:

```
🔬 PHASE 3: Running forecasting on all repaired datasets...
✓ Loaded existing results from df_final.csv (15842 rows)
✓ Found 8520 existing combinations that will be skipped

📁 Processing file: boiler_MCAR_2p_1_gafsd2all4.csv
    ⏭️  Skipping XGBoost (already computed)
    ⏭️  Skipping HoltWinters (already computed)
    🔮 Forecasting with: SARIMAX
      ✓ Completed (45.2% overall progress)

✅ Forecasting phase completed!
   - Total time: 1234.56 seconds
   - Processed: 1000/1000 forecasting tasks
   - Skipped (already computed): 852
   - Newly computed: 148
   - New results: 148 rows added to dataframe

📊 Merging new results with existing df_final.csv...
   - Existing results: 15842 rows
   - New results: 148 rows
   - Combined results: 15990 rows (after removing duplicates)
```

## Zalety:

✅ **Bezpieczne** - Nie nadpisuje istniejących wyników  
✅ **Wydajne** - Pomija już obliczone kombinacje  
✅ **Elastyczne** - Możesz dodać nowe modele/parametry bez ponownego liczenia wszystkiego  
✅ **Przejrzyste** - Pokazuje ile kombinacji pominięto vs. ile obliczono  
✅ **Odporne na błędy** - Jeśli eksperyment się zatrzyma, możesz go wznowić  

## Uwaga:

- Jeśli chcesz przeliczyć istniejące wyniki od nowa, usuń lub zmień nazwę `df_final.csv`
- System zachowuje najnowsze obliczenia w przypadku duplikatów (parametr `keep='last'`)

