# finetune_stable_diffusion.py - Documentation

Fine-tuning script for Stable Diffusion 2 specialized for time series image inpainting.

---

## 📋 Overview

`finetune_stable_diffusion.py` is a comprehensive training script that fine-tunes Stable Diffusion 2 on mathematical time series visualizations (GAF, MTF, RP, Spectrogram). The script implements:

✅ Cross-validation training  
✅ Early stopping  
✅ Memory optimizations  
✅ Mixed precision training  
✅ Automatic checkpoint management  
✅ Progress tracking and logging  

---

## 🎯 Purpose

This script fine-tunes the **UNet component** of Stable Diffusion 2 Inpainting model to specialize in reconstructing missing regions in time series image representations. The VAE and text encoder remain frozen to preserve the base model's generative capabilities.

---

## 🚀 Quick Start

### Basic Usage

```bash
python finetune_stable_diffusion.py \
    --data_dir stdiff_training_data \
    --output_dir models/my_model \
    --max_samples 4000 \
    --batch_size 4 \
    --max_epochs 300
```

### With All Options

```bash
python finetune_stable_diffusion.py \
    --data_dir stdiff_training_data \
    --output_dir models/stable_diffusion_2_all_4 \
    --max_samples 4000 \
    --batch_size 4 \
    --learning_rate 1e-5 \
    --max_epochs 300 \
    --n_folds 2 \
    --train_ratio 0.75 \
    --early_stop_patience 5 \
    --mixed_precision fp16 \
    --gradient_accumulation_steps 4
```

---

## ⚙️ Command-Line Arguments

### Required Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data_dir` | str | `stdiff_training_data` | Path to training dataset directory |
| `--output_dir` | str | `models/stable_diffusion_2_all_4` | Output directory for trained models |

### Training Configuration

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--max_samples` | int | 4000 | Number of training image pairs **per fold** |
| `--batch_size` | int | 4 | Training batch size (reduce if OOM) |
| `--learning_rate` | float | 1e-5 | Learning rate for AdamW optimizer |
| `--max_epochs` | int | 300 | Maximum number of epochs per fold |
| `--n_folds` | int | 2 | Number of training runs (cross-validation) |
| `--train_ratio` | float | 0.75 | Train/validation split ratio (0.75 = 75% train, 25% val) |

### Optimization Settings

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--mixed_precision` | str | `no` | Mixed precision: `no`, `fp16`, or `bf16` |
| `--gradient_accumulation_steps` | int | 4 | Gradient accumulation steps |
| `--early_stop_patience` | int | 5 | Early stopping patience (epochs) |

### Advanced Options

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--resume_fold` | int | `None` | Resume training from specific fold |

---

## 📊 Training Process

### Stage 1: Initialization

1. **Load base model** from HuggingFace (`stabilityai/stable-diffusion-2-inpainting`)
2. **Freeze components:** VAE and text encoder (only UNet is trained)
3. **Apply memory optimizations:**
   - Attention slicing
   - Gradient checkpointing
   - XFormers (if available)

### Stage 2: Data Preparation

1. **Load dataset metadata** from `dataset_summary.json`
2. **Split data** into train/validation sets (configurable ratio)
3. **Create DataLoaders** with specified batch size
4. **Calculate actual training pairs:**
   - Each sample has 4 image types (GAF, MTF, RP, SPEC)
   - Total pairs = base_samples × 4

### Stage 3: Training Loop (Per Fold)

```
For each fold (1 to n_folds):
    1. Split data with different random seed
    2. Create train/val datasets
    3. Initialize optimizer (AdamW)
    4. For each epoch:
        a. Train on training set
        b. Validate on validation set
        c. Save checkpoint if val loss improved
        d. Check early stopping
    5. Save final model
    6. Clear GPU memory
```

### Stage 4: Model Selection

1. **Compare all folds** by validation loss
2. **Select best fold** (lowest validation loss)
3. **Create symbolic link** to best model
4. **Save results** to `cross_validation_results.json`

---

## 🏗️ Architecture

### Classes

#### **MathImageInpaintingDataset**

PyTorch Dataset for loading image pairs.

**Features:**
- Loads original and corrupted images
- Automatically creates inpainting masks
- Resizes to 512×512 (SD2 requirement)
- Normalizes to [-1, 1] range
- Provides image type-specific prompts

**Example:**
```python
dataset = MathImageInpaintingDataset(
    data_dir="stdiff_training_data",
    indices=[0, 1, 2, ...],  # Sample indices to use
    image_types=["gaf", "mtf", "rp", "spec"]
)
```

#### **StableDiffusionTrainer**

Main trainer class handling model operations.

**Key Methods:**
- `_load_model_components()` - Loads SD2 from HuggingFace
- `_setup_memory_optimizations()` - Configures memory optimizations
- `encode_prompt()` - Encodes text prompts using CLIP
- `compute_loss()` - Computes MSE loss in latent space
- `train_epoch()` - Trains one epoch
- `validate_epoch()` - Validates on validation set
- `save_checkpoint()` - Saves model checkpoint
- `save_final_model()` - Saves complete pipeline

**Example:**
```python
trainer = StableDiffusionTrainer(
    output_dir="models/my_model",
    mixed_precision="fp16"
)
```

#### **EarlyStopping**

Implements early stopping mechanism.

**Parameters:**
- `patience`: Number of epochs to wait for improvement
- `min_delta`: Minimum change to qualify as improvement

**Example:**
```python
early_stopping = EarlyStopping(patience=5, min_delta=0.001)

for epoch in range(max_epochs):
    val_loss = train_and_validate()
    if early_stopping(val_loss):
        break
```

---

## 💾 Output Structure

After training, the output directory contains:

```
models/stable_diffusion_2_all_4/
├── best_model -> fold_2/checkpoint-fold_2_final  # Symbolic link to best
├── cross_validation_results.json                  # Training metrics
│
├── fold_1/
│   ├── checkpoint-epoch-1/
│   ├── checkpoint-epoch-2/
│   ├── ...
│   └── checkpoint-fold_1_final/                   # Final model fold 1
│       ├── model_index.json
│       ├── scheduler/
│       ├── text_encoder/
│       ├── tokenizer/
│       ├── unet/                                  # Fine-tuned weights
│       └── vae/
│
└── fold_2/
    ├── checkpoint-epoch-1/
    └── checkpoint-fold_2_final/                   # Final model fold 2
```

### Files Generated

**`cross_validation_results.json`:**
```json
[
  {
    "fold": 1,
    "best_val_loss": 0.03802,
    "train_losses": [...],
    "val_losses": [...],
    "model_path": "models/.../fold_1/..."
  },
  {
    "fold": 2,
    "best_val_loss": 0.03623,
    "train_losses": [...],
    "val_losses": [...],
    "model_path": "models/.../fold_2/..."
  }
]
```

**Checkpoint metadata** (`training_metadata.json`):
```json
{
  "fold": 1,
  "epoch": 5,
  "train_loss": 0.0394,
  "val_loss": 0.0404,
  "model_id": "stabilityai/stable-diffusion-2-inpainting"
}
```

---

## 🔧 Memory Optimizations

The script implements several memory-saving techniques:

### 1. Component Freezing
- ✅ **VAE frozen** (83M params saved)
- ✅ **Text encoder frozen** (354M params saved)
- ✅ **Only UNet trained** (859M params)

### 2. Attention Optimizations
```python
# Attention slicing (reduces memory)
unet.set_attention_slice("auto")

# XFormers (if available)
unet.enable_xformers_memory_efficient_attention()
```

### 3. Gradient Checkpointing
```python
# Trade compute for memory
unet.enable_gradient_checkpointing()
```

### 4. Mixed Precision
```python
# FP16 training (50% memory reduction)
--mixed_precision fp16
```

### Memory Usage Comparison

| Configuration | VRAM Usage | Training Speed |
|---------------|------------|----------------|
| **Full FP32** | ~22 GB | 1.0x (baseline) |
| **+ Attention Slicing** | ~18 GB | 0.95x |
| **+ Grad Checkpoint** | ~14 GB | 0.85x |
| **+ FP16** | ~11 GB | 1.1x (faster!) |

---

## 📈 Training Metrics

### Loss Function

The model minimizes **MSE loss** in latent space:

```
L = E[||ε - ε_θ(z_t, t, c)||²]
```

Where:
- `ε`: True noise added to latents
- `ε_θ`: Model's noise prediction
- `z_t`: Noisy latents at timestep t
- `t`: Diffusion timestep
- `c`: Text conditioning (prompt embeddings)

### Monitored Metrics

**Per Epoch:**
- Training loss (averaged over batches)
- Validation loss (averaged over batches)

**Per Fold:**
- Best validation loss
- Training loss history
- Validation loss history

**Overall:**
- Mean validation loss across folds
- Standard deviation of validation losses

---

## 🎓 Training Tips

### Batch Size Selection

| GPU VRAM | Recommended Batch Size | Notes |
|----------|----------------------|-------|
| **8 GB** | 1 | Use gradient accumulation |
| **11 GB** | 1-2 | FP16 recommended |
| **16 GB** | 2-4 | Good balance |
| **24 GB** | 4-8 | Optimal for speed |

### Learning Rate

- **Default (1e-5):** Safe, stable training
- **Higher (5e-5):** Faster convergence, risk of instability
- **Lower (1e-6):** Very stable, slower convergence

### Early Stopping Patience

- **Patience 3:** Aggressive, faster training
- **Patience 5:** Balanced (recommended)
- **Patience 10:** Conservative, longer training

### Train/Val Split

- **75/25:** Standard, recommended
- **80/20:** More training data, less validation
- **70/30:** More validation, better generalization estimate

---

## 🐛 Troubleshooting

### Out of Memory (OOM)

**Solution 1:** Reduce batch size
```bash
--batch_size 2  # or even 1
```

**Solution 2:** Enable mixed precision
```bash
--mixed_precision fp16
```

**Solution 3:** Increase gradient accumulation
```bash
--gradient_accumulation_steps 8
```

### Training Too Slow

**Solution 1:** Increase batch size (if memory allows)
```bash
--batch_size 8
```

**Solution 2:** Use mixed precision
```bash
--mixed_precision fp16
```

**Solution 3:** Reduce validation frequency
```python
# Modify code to validate every N epochs instead of every epoch
```

### Model Not Improving

**Solution 1:** Check learning rate
```bash
--learning_rate 5e-5  # Try slightly higher
```

**Solution 2:** Increase training data
```bash
--max_samples 8000  # More samples per fold
```

**Solution 3:** Train longer
```bash
--max_epochs 500
--early_stop_patience 10
```

### CUDA Out of Memory

```python
RuntimeError: CUDA out of memory
```

**Solutions:**
1. Restart Python kernel
2. Clear CUDA cache: `torch.cuda.empty_cache()`
3. Reduce batch size
4. Enable all memory optimizations

---

## 📊 Example Training Session

### Command

```bash
python finetune_stable_diffusion.py \
    --data_dir stdiff_training_data \
    --output_dir models/sd2_custom \
    --max_samples 4000 \
    --batch_size 4 \
    --learning_rate 1e-5 \
    --max_epochs 300 \
    --n_folds 2 \
    --train_ratio 0.75 \
    --early_stop_patience 5
```

### Expected Output

```
🚀 STARTING STABLE DIFFUSION 2 FINE-TUNING
============================================================
📁 Data directory: stdiff_training_data
📁 Output directory: models/sd2_custom
🎯 Max samples: 4000
📊 Batch size: 4
🎓 Learning rate: 1e-05
📈 Training runs: 2
📊 Train/Val split: 75%/25%

🔧 GPU: NVIDIA TITAN RTX
💾 GPU Memory: 24.0 GB

📚 Total samples in dataset: 2000
🎯 Target: 4000 training pairs per fold
🎯 Using 1333 base samples = 5332 total image pairs
📊 Split ratio: 75% train / 25% validation
📊 Actual per fold: 3999 training pairs + 1333 validation pairs

Loading model components from stabilityai/stable-diffusion-2-inpainting
Model components loaded successfully
✅ Attention slicing enabled
✅ Gradient checkpointing enabled
✅ Memory optimizations setup completed

🔄 TRAINING RUN 1/2
========================================
📊 Train samples: 999 (3996 pairs)
📊 Val samples: 334 (1336 pairs)

Dataset created with 3996 image pairs
Dataset created with 1336 image pairs

🏃 Starting training for fold 1

📅 FOLD 1/2 - EPOCH 1/300
------------------------------------------------------------
🏃 Training epoch 1...
Training: 100%|████████| 999/999 [08:45<00:00, 1.90it/s, loss=0.047266]
🔍 Validating epoch 1...
Validation: 100%|████████| 334/334 [01:23<00:00, 4.01it/s]

📊 RESULTS - Epoch 1:
   📈 Train Loss: 0.047266
   📉 Val Loss: 0.047544
✅ New best validation loss: 0.047544

... [continues for more epochs] ...

✅ Fold 1 completed. Best val loss: 0.038021

🔄 TRAINING RUN 2/2
========================================
... [similar output] ...

✅ Fold 2 completed. Best val loss: 0.036234

🎉 CROSS-VALIDATION COMPLETED!
==================================================
📊 Mean validation loss: 0.037128 ± 0.000894
📁 Results saved to: models/sd2_custom/cross_validation_results.json
📁 Models saved in: models/sd2_custom
🏆 Best fold: 2 (val_loss: 0.036234)
🏆 Best model: models/sd2_custom/fold_2/checkpoint-fold_2_final
🔗 Best model linked as: models/sd2_custom/best_model

🎯 NEXT STEPS:
📁 Models saved in: models/sd2_custom
🏆 Best model: models/sd2_custom/best_model
```

### Training Time Estimate

| Configuration | Time per Epoch | Total Time (2 folds, ~10 epochs each) |
|---------------|----------------|--------------------------------------|
| **Batch 1, FP32** | ~25 min | ~8.3 hours |
| **Batch 4, FP32** | ~10 min | ~3.3 hours |
| **Batch 4, FP16** | ~7 min | ~2.3 hours |
| **Batch 8, FP16** | ~4 min | ~1.3 hours |

*Times measured on NVIDIA TITAN RTX with 4,000 training pairs per fold*

---

## 🔬 Technical Details

### Model Components

**UNet2DConditionModel (Trainable):**
- Input channels: 9 (noisy latent + mask + masked latent)
- Output channels: 4 (predicted noise)
- Architecture: U-Net with attention
- Parameters: ~859M

**AutoencoderKL (Frozen):**
- Encoder: 512×512 → 64×64 (compression factor: 8)
- Decoder: 64×64 → 512×512
- Latent channels: 4
- Parameters: ~83M

**CLIPTextModel (Frozen):**
- Model: OpenCLIP-ViT-H/14
- Max tokens: 77
- Embedding dim: 1024
- Parameters: ~354M

### Training Algorithm

```python
for epoch in range(max_epochs):
    # Training
    for batch in train_loader:
        # 1. Encode images to latents
        latents = vae.encode(images)
        
        # 2. Add noise
        noisy_latents = add_noise(latents, timesteps)
        
        # 3. Prepare input (concatenate with mask and masked latents)
        model_input = cat([noisy_latents, mask, masked_latents])
        
        # 4. Encode prompt
        prompt_embeds = text_encoder(prompts)
        
        # 5. Predict noise
        pred_noise = unet(model_input, timesteps, prompt_embeds)
        
        # 6. Compute loss
        loss = mse_loss(pred_noise, true_noise)
        
        # 7. Backprop and update
        loss.backward()
        optimizer.step()
    
    # Validation
    val_loss = validate(val_loader)
    
    # Early stopping check
    if early_stopping(val_loss):
        break
```

---

## 📦 Dependencies

### Required Packages

```txt
torch>=2.0.0
torchvision>=0.15.0
diffusers>=0.21.0
transformers>=4.30.0
accelerate>=0.20.0
pillow>=9.0.0
numpy>=1.21.0
tqdm>=4.64.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
```

### Installation

```bash
# PyTorch (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Diffusers and dependencies
pip install diffusers transformers accelerate

# Other requirements
pip install pillow numpy tqdm scikit-learn matplotlib
```

### Optional (for speed)

```bash
# XFormers (memory efficient attention)
pip install xformers

# Triton (kernel optimizations)
pip install triton
```

---

## 🔗 Related Files

### Dataset
- **Dataset README:** `stdiff_training_data/README.md`
- **Generation Script:** `generate_training_dataset.py`
- **Dataset Index:** `DATASET_DOCUMENTATION_INDEX.md`

### Model
- **Model README:** `models/stable_diffusion_2_all_4/best_model/README.md`
- **Integration:** `integrate_custom_model.py`
- **Inference:** `models/stdiff.py`

### Experiment
- **Main Experiment:** `iterative_experiment.py`
- **Image Encoders:** `ts_image_inpainting.py`

---

## 📖 References

### Papers

**Stable Diffusion:**
```bibtex
@article{rombach2022high,
  title={High-Resolution Image Synthesis with Latent Diffusion Models},
  author={Rombach, Robin and Blattmann, Andreas and Lorenz, Dominik and Esser, Patrick and Ommer, Bj{\"o}rn},
  journal={CVPR},
  year={2022}
}
```

**DDPM:**
```bibtex
@article{ho2020denoising,
  title={Denoising Diffusion Probabilistic Models},
  author={Ho, Jonathan and Jain, Ajay and Abbeel, Pieter},
  journal={NeurIPS},
  year={2020}
}
```

### Links

- **HuggingFace Model:** https://huggingface.co/stabilityai/stable-diffusion-2-inpainting
- **Diffusers Docs:** https://huggingface.co/docs/diffusers/
- **Stable Diffusion:** https://stability.ai/

---

## 📧 Support

**Common Issues:**
1. Check GPU memory with `nvidia-smi`
2. Verify dataset exists and has correct structure
3. Ensure all dependencies are installed
4. Review training logs for errors

**For Help:**
- Check troubleshooting section
- Review example training session
- Consult HuggingFace Diffusers documentation

---

**Script Version:** 1.0  
**Last Updated:** 1.12.2025  
**Status:** Production Ready ✅

