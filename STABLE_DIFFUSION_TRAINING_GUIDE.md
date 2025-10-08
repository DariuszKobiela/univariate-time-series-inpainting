# 🎯 Przewodnik Treningu Custom Stable Diffusion 2 dla Obrazów Matematycznych

Ten przewodnik pokazuje jak wytrenować własny model Stable Diffusion 2 specjalnie na obrazach GAF, MTF, RP i Spec do lepszego inpainting'u szeregów czasowych.

## 📋 Przegląd Procesu

1. **Generowanie Datasetu** - Tworzenie syntetycznych szeregów czasowych i obrazów
2. **Fine-tuning Modelu** - Trenowanie Stable Diffusion 2 na matematycznych obrazach  
3. **Integracja** - Dodanie modelu do systemu eksperymentów
4. **Testowanie** - Porównanie z innymi metodami inpainting

## 🚀 Krok 1: Generowanie Datasetu Treningowego

### Instalacja zależności
```bash
conda activate timeseries
pip install diffusers transformers accelerate xformers
```

### Generowanie danych
```bash
# Wygeneruj 1000 par treningowych (default w folderze stdiff_training_data)
python generate_training_dataset.py --samples 1000

# Więcej próbek dla lepszego modelu
python generate_training_dataset.py --samples 5000 --output stdiff_training_data_large

# Custom konfiguracja
python generate_training_dataset.py \
    --samples 2000 \
    --output stdiff_custom_dataset \
    --seed 123
```

**Wygenerowane pliki:**
- `stdiff_training_data/original/` - Obrazy oryginalne (GAF, MTF, RP, Spec)  
- `stdiff_training_data/missing/` - Obrazy z brakującymi danymi
- `stdiff_training_data/masks/` - Metadane i maski
- `stdiff_training_data/dataset_summary.json` - Podsumowanie datasetu

## 🏋️ Krok 2: Fine-tuning Stable Diffusion 2

### Trenowanie modelu dla obrazów GAF
```bash
# Basic training (szybkie, 50 epok) - używa domyślnego folderu stdiff_training_data
python finetune_stable_diffusion.py \
    --image_type gaf \
    --output_dir models/stable_diffusion_2_gaf \
    --epochs 50

# Advanced training (więcej epok, większy batch)
python finetune_stable_diffusion.py \
    --dataset_dir stdiff_training_data_large \
    --image_type gaf \
    --output_dir models/stable_diffusion_2_gaf_advanced \
    --epochs 100 \
    --batch_size 2 \
    --learning_rate 5e-6 \
    --save_steps 250
```

### Trenowanie dla innych typów obrazów
```bash
# MTF images
python finetune_stable_diffusion.py \
    --image_type mtf \
    --output_dir models/stable_diffusion_2_mtf \
    --epochs 50

# RP images  
python finetune_stable_diffusion.py \
    --image_type rp \
    --output_dir models/stable_diffusion_2_rp \
    --epochs 50

# Spec images
python finetune_stable_diffusion.py \
    --image_type spec \
    --output_dir models/stable_diffusion_2_spec \
    --epochs 50
```

**Podczas treningu:**
- Model zapisuje checkpointy co 500 kroków
- Ewaluacja co 250 kroków w folderze `evaluation/`
- Tensorboard logi w folderze modelu
- Final model w `checkpoint-final/`

## 🔗 Krok 3: Integracja z Systemem

### Dodanie modelu do eksperymentów
```bash
# Integruj wytrenowany model GAF
python integrate_custom_model.py \
    --model_path models/stable_diffusion_2_gaf/checkpoint-final \
    --model_name stdiff-gaf \
    --test

# Integruj wszystkie modele
python integrate_custom_model.py \
    --model_path models/stable_diffusion_2_mtf/checkpoint-final \
    --model_name stdiff-mtf

python integrate_custom_model.py \
    --model_path models/stable_diffusion_2_rp/checkpoint-final \
    --model_name stdiff-rp

python integrate_custom_model.py \
    --model_path models/stable_diffusion_2_spec/checkpoint-final \
    --model_name stdiff-spec
```

## 🧪 Krok 4: Testowanie w Eksperymentach

### Test pojedynczego modelu
```bash
# Test tylko z custom GAF model
python run_improved_experiment.py \
    --quick \
    --inpainting_models stdiff-gaf

# Porównanie z istniejącymi modelami
python run_improved_experiment.py \
    --quick \
    --inpainting_models gaf-unet stdiff-gaf
```

### Test wszystkich custom modeli
```bash
# Wszystkie custom Stable Diffusion modele
python run_improved_experiment.py \
    --medium \
    --inpainting_models stdiff-gaf stdiff-mtf stdiff-rp stdiff-spec

# Porównanie: tradycyjne vs custom
python run_improved_experiment.py \
    --full \
    --inpainting_models gaf-unet mtf-unet rp-unet spec-unet stdiff-gaf stdiff-mtf stdiff-rp stdiff-spec
```

## 📊 Struktura Plików

```
univariate-time-series-inpainting/
├── generate_training_dataset.py     # Generator datasetu
├── finetune_stable_diffusion.py     # Skrypt fine-tuningu  
├── integrate_custom_model.py        # Integracja z systemem
├── models/stdiff.py                 # Custom inpainter class
├── stdiff_training_data/             # Wygenerowany dataset
│   ├── original/                    # Obrazy oryginalne  
│   ├── missing/                     # Obrazy z brakami
│   ├── masks/                       # Metadane
│   └── dataset_summary.json         # Podsumowanie
└── models/
    ├── stable_diffusion_2_gaf/      # Wytrenowany model GAF
    ├── stable_diffusion_2_mtf/      # Wytrenowany model MTF
    ├── stable_diffusion_2_rp/       # Wytrenowany model RP
    └── stable_diffusion_2_spec/     # Wytrenowany model Spec
```

## 🎯 Parametry Treningu

### Generowanie Datasetu
- **samples**: 1000-5000 (więcej = lepszy model, ale dłuższy trening)
- **pattern_distribution**: domyślnie zrównoważony mix wzorców
- **missing_rates**: 5-30% brakujących danych
- **missing_types**: random, block, periodic, edge

### Fine-tuning
- **epochs**: 50-100 (więcej dla lepszych wyników)
- **batch_size**: 1-2 (ograniczone pamięcią GPU)
- **learning_rate**: 1e-5 do 5e-6 (niższe = stabilniejsze)
- **mixed_precision**: fp16 (oszczędza pamięć GPU)

## 🔧 Troubleshooting

### Błędy pamięci GPU
```bash
# Zmniejsz batch size
--batch_size 1

# Użyj CPU (wolniejsze)
--mixed_precision no

# Mniejszy dataset
--samples 500
```

### Słabe wyniki modelu
```bash
# Więcej epok
--epochs 100

# Więcej danych treningowych  
--samples 5000

# Niższa learning rate
--learning_rate 5e-6
```

### Model nie ładuje się
```bash
# Sprawdź ścieżkę
ls models/stable_diffusion_2_gaf/checkpoint-final/

# Test integracji
python integrate_custom_model.py --test --model_name stdiff-gaf
```

## 📈 Oczekiwane Wyniki

Po wytrenowaniu custom model powinien:
- ✅ Lepiej rozumieć struktury matematyczne GAF/MTF/RP/Spec
- ✅ Generować bardziej realistyczne wzorce w obszarach inpainting
- ✅ Zachowywać ciągłość i spójność obrazów matematycznych
- ✅ Osiągać lepsze metryki MAE/MAPE/RMSE w eksperymentach

Porównaj wyniki w `results/` z innymi metodami inpainting!

## 🎉 Gratulacje!

Masz teraz własny model Stable Diffusion 2 wytrenowany specjalnie na obrazach szeregów czasowych! 🚀




python finetune_stable_diffusion.py --image_type gaf mtf rp spec --output_dir models/stable_diffusion_2_all --epochs 10 --save_steps 1000 --keep_checkpoints 3

# Podstawowe trenowanie z CV
python finetune_stable_diffusion.py --output_dir models/stable_diffusion_2_all --image_type gaf mtf rp spec --epochs 300 --cv_folds 10 --early_stopping_patience 5