# Fine-tuned Stable Diffusion 2 Model for Time Series Image Inpainting

This directory contains a fine-tuned Stable Diffusion 2 inpainting model specialized for reconstructing mathematical time series visualizations (GAF, MTF, RP, Spectrogram).

---

## 📋 Model Overview

| Property | Value |
|----------|-------|
| **Base Model** | `stabilityai/stable-diffusion-2-inpainting` |
| **Specialized For** | Mathematical time series image reconstruction |
| **Image Types** | GAF, MTF, RP, Spectrogram |
| **Input Size** | 512×512 pixels |
| **Architecture** | UNet2DConditionModel (fine-tuned) |
| **Training Method** | Cross-validation with early stopping |
| **Best Validation Loss** | 0.03623 |
| **Training Date** | September 2024 |

---

## 🎯 Purpose

This model was fine-tuned to perform **inpainting** on time series visualizations. It can:

1. **Reconstruct missing regions** in GAF (Gramian Angular Field) images
2. **Fill gaps** in MTF (Markov Transition Field) images
3. **Complete** RP (Recurrence Plot) images
4. **Restore** Spectrogram images

The model is specifically trained on synthetic time series data with controlled missing patterns (random, block, periodic, edge).

---

## 📁 Model Structure

```
best_model/
├── README.md                 # This file
├── model_index.json          # Model configuration
├── scheduler/                # DDPM Scheduler
├── text_encoder/            # CLIP Text Encoder (frozen)
├── tokenizer/               # CLIP Tokenizer
├── unet/                    # Fine-tuned UNet (main trainable component)
└── vae/                     # VAE Encoder/Decoder (frozen)
```

**Note:** Only the **UNet** component was fine-tuned. All other components (VAE, text encoder) remain frozen from the base Stable Diffusion 2 model.

---

## 🚀 Quick Start

### Installation

```bash
pip install torch torchvision
pip install diffusers transformers accelerate
pip install pillow numpy
```

### Basic Usage

```python
from diffusers import StableDiffusionInpaintPipeline
import torch
from PIL import Image

# Load the fine-tuned model
model_path = "models/stable_diffusion_2_all_4/best_model"
pipeline = StableDiffusionInpaintPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.float16
).to("cuda")

# Load images
image = Image.open("missing_image.png").convert("RGB").resize((512, 512))
mask = Image.open("mask.png").convert("L").resize((512, 512))

# Inpaint with appropriate prompt
prompt = "high quality gramian angular field mathematical visualization"
result = pipeline(
    prompt=prompt,
    image=image,
    mask_image=mask,
    num_inference_steps=50
).images[0]

result.save("reconstructed_image.png")
```

### Image Type-Specific Prompts

Use these prompts for best results:

```python
prompts = {
    "gaf": "high quality gramian angular field mathematical visualization",
    "mtf": "high quality markov transition field mathematical visualization",
    "rp": "high quality recurrence plot mathematical visualization",
    "spec": "high quality spectrogram mathematical visualization"
}
```

---

## 🏋️ Training Details

### Training Configuration

| Parameter | Value |
|-----------|-------|
| **Training Samples** | ~3,000 per fold |
| **Validation Samples** | ~1,000 per fold |
| **Batch Size** | 4 |
| **Learning Rate** | 1e-5 |
| **Optimizer** | AdamW (weight_decay=0.01) |
| **Training Runs** | 2 folds |
| **Early Stopping** | Patience = 5 epochs |
| **Mixed Precision** | No (FP32) |
| **Epochs Trained** | 9-13 per fold |

### Training Results

**Fold 1:**
- Best Validation Loss: **0.03802**
- Epochs: 13
- Final Training Loss: 0.03784

**Fold 2** (Best Model):
- Best Validation Loss: **0.03623** ⭐
- Epochs: 9
- Final Training Loss: 0.03872

**Overall Performance:**
- Mean Validation Loss: 0.03713 ± 0.00089
- This model represents Fold 2 (best performer)

### Memory Optimizations Used

- Attention slicing enabled
- Gradient checkpointing
- Only UNet fine-tuned (VAE and text encoder frozen)

---

## 📊 Training Dataset

The model was trained on synthetically generated time series with:

- **2,000 base samples** = 8,000 images (4 types each)
- **Pattern types:** sine, cosine, trend, seasonal, noise, spikes, mixed
- **Missing data types:** random, block, periodic, edge
- **Missing rates:** 5% - 30%
- **Time series lengths:** 100 - 1,000 points

Dataset generation script: `generate_training_dataset.py`

---

## 💻 System Requirements

### Minimum Requirements

- **GPU:** NVIDIA GPU with 8+ GB VRAM
- **CUDA:** 11.0+
- **RAM:** 16 GB
- **Storage:** ~20 GB for model

### Recommended Requirements

- **GPU:** NVIDIA GPU with 16+ GB VRAM (RTX 3090, A5000, or better)
- **CUDA:** 11.8+
- **RAM:** 32 GB
- **Storage:** 50 GB (for model + datasets)

### Tested On

- **GPU:** NVIDIA TITAN RTX (24 GB GDDR6)
- **Driver:** 525.147.05
- **CUDA:** 12.0
- **OS:** Ubuntu 18.04.1 LTS

---

## 🎨 Example Use Cases

### 1. Recovering Missing Time Series Data

```python
# Load corrupted time series image
corrupted_img = load_corrupted_timeseries_image()
mask = create_missing_data_mask()

# Reconstruct using fine-tuned model
reconstructed = pipeline(
    prompt="high quality gramian angular field mathematical visualization",
    image=corrupted_img,
    mask_image=mask,
    guidance_scale=7.5,
    num_inference_steps=50
).images[0]
```

### 2. Batch Processing Multiple Images

```python
import glob
from pathlib import Path

# Process all GAF images
for img_path in glob.glob("data/missing/*.png"):
    image = Image.open(img_path).resize((512, 512))
    mask = generate_mask_from_image(image)
    
    result = pipeline(
        prompt="high quality gramian angular field mathematical visualization",
        image=image,
        mask_image=mask
    ).images[0]
    
    output_path = Path("data/reconstructed") / Path(img_path).name
    result.save(output_path)
```

### 3. Integration with Forecasting Pipeline

```python
# 1. Time series → Image (GAF/MTF/RP/SPEC)
image = time_series_to_gaf(corrupted_series)

# 2. Create mask for missing regions
mask = create_mask_from_nan(corrupted_series)

# 3. Inpaint image
reconstructed_image = pipeline(
    prompt="high quality gramian angular field mathematical visualization",
    image=image,
    mask_image=mask
).images[0]

# 4. Image → Time series
reconstructed_series = gaf_to_time_series(reconstructed_image)

# 5. Use for forecasting
forecast = xgboost_model.predict(reconstructed_series)
```

---

## 🔧 Advanced Configuration

### Inference Parameters

```python
result = pipeline(
    prompt=prompt,
    image=image,
    mask_image=mask,
    
    # Quality settings
    num_inference_steps=50,        # 20-100, higher = better quality
    guidance_scale=7.5,            # 1-20, higher = closer to prompt
    
    # Generation settings
    num_images_per_prompt=1,
    generator=torch.manual_seed(42),  # For reproducibility
    
    # Advanced
    eta=0.0,                       # DDIM eta parameter
    output_type="pil"              # "pil" or "np"
)
```

### Memory Management

```python
# For limited VRAM
pipeline.enable_attention_slicing()
pipeline.enable_vae_slicing()

# Use CPU offloading
pipeline.enable_sequential_cpu_offload()

# Lower precision
pipeline = pipeline.to(torch_dtype=torch.float16)
```

---

## 📈 Performance Metrics

### Validation Performance

| Metric | Value |
|--------|-------|
| **Best Val Loss** | 0.03623 |
| **Mean Val Loss** | 0.03713 ± 0.00089 |
| **Training Stability** | High (early stopping at epoch 9) |

### Inference Speed (TITAN RTX)

| Inference Steps | Time per Image |
|----------------|----------------|
| 20 steps | ~2.5 seconds |
| 50 steps | ~5.5 seconds |
| 100 steps | ~11 seconds |

---

## 🔬 Technical Details

### Model Architecture

**Base:** Stable Diffusion 2 Inpainting
- **UNet:** 859M parameters (fine-tuned)
  - Channels: 320, 640, 1280, 1280
  - Attention layers: Self-attention + Cross-attention
  - Conditioning: Text embeddings (CLIP)
  
- **VAE:** 83M parameters (frozen)
  - Encoder: 512×512 → 64×64 latents
  - Decoder: 64×64 latents → 512×512
  - Latent channels: 4
  
- **Text Encoder:** CLIP (frozen)
  - OpenCLIP-ViT-H/14
  - 354M parameters

### Training Objective

The model minimizes MSE loss in latent space:

```
L = E[||ε - ε_θ(z_t, t, c)||²]
```

Where:
- ε: True noise
- ε_θ: Model prediction
- z_t: Noisy latents at timestep t
- c: Text conditioning (prompt)

---

## 📝 How to Retrain or Fine-tune Further

### Re-training from Scratch

```bash
python finetune_stable_diffusion.py \
    --data_dir stdiff_training_data \
    --output_dir models/my_new_model \
    --max_samples 4000 \
    --batch_size 4 \
    --learning_rate 1e-5 \
    --max_epochs 300 \
    --n_folds 2 \
    --train_ratio 0.75 \
    --early_stop_patience 5
```

### Continuing Training from This Model

```bash
# Load this model and continue training
python finetune_stable_diffusion.py \
    --data_dir stdiff_training_data \
    --output_dir models/stable_diffusion_2_all_4_continued \
    --resume_fold 2 \
    --max_epochs 50
```

### Training on Custom Dataset

1. Generate your dataset:
```bash
python generate_training_dataset.py \
    --samples 5000 \
    --output my_custom_data
```

2. Train model:
```bash
python finetune_stable_diffusion.py \
    --data_dir my_custom_data \
    --output_dir models/custom_model
```

---

## 🐛 Troubleshooting

### Out of Memory (OOM)

```python
# Reduce batch size, use attention slicing
pipeline.enable_attention_slicing()
pipeline.enable_vae_slicing()

# Or use float16
pipeline = pipeline.to(torch_dtype=torch.float16)
```

### Poor Reconstruction Quality

1. **Increase inference steps:** 20 → 50 or 100
2. **Adjust guidance scale:** Try 5.0 - 10.0
3. **Check prompt:** Use correct image type prompt
4. **Mask quality:** Ensure mask accurately covers missing regions

### Slow Inference

```python
# Use fewer inference steps (trade quality for speed)
result = pipeline(..., num_inference_steps=20)

# Use DPM++ Sampler (faster)
from diffusers import DPMSolverMultistepScheduler
pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
    pipeline.scheduler.config
)
```

---

## 📚 Related Files

- **Training Script:** `finetune_stable_diffusion.py`
- **Dataset Generator:** `generate_training_dataset.py`
- **Integration Script:** `integrate_custom_model.py`
- **Main Experiment:** `iterative_experiment.py`
- **Image Encoders:** `ts_image_inpainting.py`

---

## 📖 References

### Base Model

```bibtex
@article{rombach2022high,
  title={High-resolution image synthesis with latent diffusion models},
  author={Rombach, Robin and Blattmann, Andreas and Lorenz, Dominik and Esser, Patrick and Ommer, Bj{\"o}rn},
  journal={CVPR},
  year={2022}
}
```

### Stable Diffusion 2

- HuggingFace: https://huggingface.co/stabilityai/stable-diffusion-2-inpainting
- Paper: https://arxiv.org/abs/2112.10752

---

## 🔗 Integration with Experimental Pipeline

This model can be used in the main experimental pipeline:

```python
# In iterative_experiment.py or ts_image_inpainting.py
from models.stdiff import StableDiffusion2MathInpainter

# Initialize inpainter with this model
inpainter = StableDiffusion2MathInpainter(
    model_path="models/stable_diffusion_2_all_4/best_model"
)

# Use for inpainting
reconstructed = inpainter.inpaint(
    image=corrupted_gaf_image,
    mask=missing_mask,
    enc_name="gaf"
)
```

---

## ⚖️ License

This model is a fine-tuned version of Stable Diffusion 2, which is released under the CreativeML OpenRAIL-M license. 

**Base Model License:** CreativeML OpenRAIL-M  
**Fine-tuned Weights:** Same as base model

See: https://huggingface.co/stabilityai/stable-diffusion-2-inpainting/blob/main/LICENSE

---

## 🤝 Citation

If you use this model in your research, please cite:

```bibtex
@misc{ts_sd2_finetuned_2024,
  title={Fine-tuned Stable Diffusion 2 for Time Series Image Inpainting},
  author={[Dariusz Kobiela, Jarosław Kobiela, Adam Kurowski, Agnieszka Landowska]},
  year={2025},
  howpublished={Trained on synthetic time series dataset},
  note={Fine-tuned from stabilityai/stable-diffusion-2-inpainting}
}
```

---

## 📧 Support

For questions or issues:

1. Check the troubleshooting section above
2. Review training logs in `../cross_validation_results.json`
3. Consult `finetune_stable_diffusion.py` documentation
4. Check original Stable Diffusion 2 documentation

---

**Model Version:** 1.0  
**Last Updated:** 1.12.2025  
**Status:** Production Ready ✅

