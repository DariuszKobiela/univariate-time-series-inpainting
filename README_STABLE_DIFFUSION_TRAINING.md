# Stable Diffusion 2 Fine-tuning dla obrazów matematycznych

## Przegląd

System do fine-tuningu Stable Diffusion 2 na obrazach matematycznych (GAF, MTF, RP, SPEC) z cross-validation i early stopping.

## Struktura

```
finetune_stable_diffusion.py    # Główny skrypt treningu z CV
models/stdiff.py                # Klasa inpainter kompatybilna z systemem
integrate_custom_model.py       # Integracja z ts_image_inpainting.py
stdiff_training_data/           # Dataset (2000 próbek × 4 typy = 8000 par)
models/stable_diffusion_2_all/  # Wyjście treningu
```

## Użycie

### 1. Trening modelu

```bash
# Podstawowy trening (domyślne parametry)
python finetune_stable_diffusion.py

# Trening z custom parametrami
python finetune_stable_diffusion.py \
    --max_samples 4000 \
    --batch_size 2 \
    --learning_rate 2e-5 \
    --max_epochs 50 \
    --n_folds 5

# Zaawansowane opcje pamięci
python finetune_stable_diffusion.py \
    --mixed_precision fp16 \
    --gradient_accumulation_steps 8 \
    --batch_size 1
```

### 2. Integracja z systemem

```bash
# Auto-integracja najlepszego modelu
python integrate_custom_model.py \
    --model_path models/stable_diffusion_2_all/best_model \
    --model_name sd2-all \
    --test

# Sprawdź integrację
python -c "import ts_image_inpainting; print(list(ts_image_inpainting.INPAINTERS.keys()))"
```

### 3. Użycie w eksperymentach

```bash
# Użycie nowego modelu
python run_improved_experiment.py --inpainting_models sd2-all

# Porównanie z innymi modelami
python run_improved_experiment.py --inpainting_models unet sd2-all
```

## Parametry treningu

### Pamięć GPU (22GB)
- **batch_size**: 1-2 (bezpieczny start)
- **mixed_precision**: "fp16" (oszczędność pamięci)
- **gradient_accumulation_steps**: 4-8
- **image_size**: 512 (Stable Diffusion 2)

### Cross-validation
- **n_folds**: 10 (domyślnie)
- **early_stop_patience**: 5 epochs
- **max_epochs**: 100

### Optymalizacje
- Attention slicing
- Gradient checkpointing
- XFormers memory efficient attention
- Trenowany tylko UNet (VAE, Text Encoder zamrożone)

## Struktura wyjścia

```
models/stable_diffusion_2_all/
├── fold_1/
│   ├── checkpoint-fold_1_final/     # Finalny model fold 1
│   └── checkpoint-epoch-X/          # Checkpointy w trakcie
├── fold_2/
│   └── ...
├── best_model -> fold_X/checkpoint-fold_X_final/  # Link do najlepszego
└── cross_validation_results.json    # Wyniki wszystkich folds
```

## Detekacja typu modelu

System automatycznie wykrywa multi-type modele:
- Ścieżka zawiera "stable_diffusion_2_all" 
- Nazwa modelu zawiera "sd2-all"
- Ścieżka zawiera "multi"

## Przykład kompletnego workflow

```bash
# 1. Trening
python finetune_stable_diffusion.py --max_samples 4000

# 2. Integracja
python integrate_custom_model.py \
    --model_path models/stable_diffusion_2_all/best_model \
    --model_name sd2-all

# 3. Test
python run_improved_experiment.py --quick --inpainting_models sd2-all

# 4. Pełny eksperyment
python run_improved_experiment.py --inpainting_models sd2-all --datasets all
```

## Troubleshooting

### Błędy pamięci GPU
```bash
# Zmniejsz batch size
--batch_size 1

# Zwiększ gradient accumulation
--gradient_accumulation_steps 8

# Użyj CPU dla walidacji (dodaj w kodzie)
```

### Błędy modelu
```bash
# Sprawdź czy model istnieje
ls -la models/stable_diffusion_2_all/best_model

# Test ładowania
python -c "from models.stdiff import StableDiffusion2MathInpainter; m = StableDiffusion2MathInpainter('models/stable_diffusion_2_all/best_model')"
```

### Błędy zależności
```bash
pip install diffusers transformers accelerate torch torchvision
pip install sklearn tqdm matplotlib
```

## Monitoring treningu

Logi zawierają:
- Loss per epoch (train/validation)
- Memory usage
- Early stopping status
- Best model selection
- Cross-validation results

## Rozszerzenia

- Dodatkowe typy obrazów: edytuj `image_types` w dataset
- Custom prompty: modyfikuj `prompts` w `MathImageInpaintingDataset`
- Inne schedulery: zmień `DDPMScheduler` w trainerze
- Zaawansowane augmentacje: dodaj w `__getitem__`
