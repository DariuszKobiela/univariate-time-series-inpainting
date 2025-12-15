# 📚 Stable Diffusion 2 Fine-tuning - Complete Documentation Index

Complete documentation for fine-tuning and using Stable Diffusion 2 for time series image inpainting.

---

## 📄 Documentation Files

### 1. **Trained Model README**
📁 Location: `models/stable_diffusion_2_all_4/best_model/README.md`

**For:** Users who want to USE the trained model

**Contents:**
- ✅ Model overview and specifications
- ✅ Quick start guide with code examples
- ✅ Training details and performance metrics
- ✅ Use cases and integration examples
- ✅ Troubleshooting and optimization tips
- ✅ Technical architecture details

**When to read:** You have a trained model and want to use it for inference

---

### 2. **Training Script Documentation**
📁 Location: `FINETUNE_STABLE_DIFFUSION_README.md`

**For:** Users who want to TRAIN or fine-tune the model

**Contents:**
- ✅ Complete CLI reference
- ✅ Training process explained
- ✅ Architecture and class documentation
- ✅ Memory optimization strategies
- ✅ Troubleshooting training issues
- ✅ Example training sessions with expected output

**When to read:** You want to train your own model or understand the training process

---

## 🎯 Quick Navigation

### Want to Use the Model?

**→ Read:** `models/stable_diffusion_2_all_4/best_model/README.md`

**Quick Start:**
```python
from diffusers import StableDiffusionInpaintPipeline

pipeline = StableDiffusionInpaintPipeline.from_pretrained(
    "models/stable_diffusion_2_all_4/best_model"
).to("cuda")

result = pipeline(
    prompt="high quality gramian angular field mathematical visualization",
    image=your_image,
    mask_image=your_mask
).images[0]
```

---

### Want to Train a Model?

**→ Read:** `FINETUNE_STABLE_DIFFUSION_README.md`

**Quick Start:**
```bash
python finetune_stable_diffusion.py \
    --data_dir stdiff_training_data \
    --output_dir models/my_model \
    --max_samples 4000 \
    --batch_size 4
```

---

### Want to Generate Training Data?

**→ Read:** `DATASET_DOCUMENTATION_INDEX.md`

**Quick Start:**
```bash
python generate_training_dataset.py --samples 2000
```

---

## 📊 Complete Workflow

### End-to-End Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│ 1. GENERATE DATASET                                         │
│    python generate_training_dataset.py --samples 2000       │
│    → Output: stdiff_training_data/                          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. TRAIN MODEL                                              │
│    python finetune_stable_diffusion.py                      │
│    → Output: models/stable_diffusion_2_all_4/best_model/    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. USE MODEL                                                │
│    - Load with StableDiffusionInpaintPipeline              │
│    - Integrate with ts_image_inpainting.py                 │
│    - Run experiments with iterative_experiment.py          │
└─────────────────────────────────────────────────────────────┘
```

---

## 📂 File Structure

```
project/
├── STABLE_DIFFUSION_DOCUMENTATION_INDEX.md    # This file
├── FINETUNE_STABLE_DIFFUSION_README.md       # Training script docs
├── DATASET_DOCUMENTATION_INDEX.md            # Dataset docs
│
├── finetune_stable_diffusion.py              # Training script
├── generate_training_dataset.py              # Dataset generator
│
├── stdiff_training_data/                     # Training dataset
│   ├── README.md                             # Dataset documentation
│   ├── original/                             # Original images
│   ├── missing/                              # Corrupted images
│   └── masks/                                # Metadata
│
└── models/
    └── stable_diffusion_2_all_4/
        ├── best_model/                       # Best trained model
        │   ├── README.md                     # Model documentation
        │   ├── unet/                         # Fine-tuned UNet
        │   ├── vae/                          # VAE (frozen)
        │   └── ...
        └── cross_validation_results.json     # Training metrics
```

---

## 🎓 Learning Path

### For Beginners

1. **Start here:** `models/stable_diffusion_2_all_4/best_model/README.md`
   - Learn what the model does
   - Try basic inference examples
   
2. **Then read:** `stdiff_training_data/README.md`
   - Understand the training data
   - See what patterns were used

3. **Finally:** `FINETUNE_STABLE_DIFFUSION_README.md`
   - Learn how training works
   - Understand parameters and optimization

### For Advanced Users

1. **Training:** `FINETUNE_STABLE_DIFFUSION_README.md`
   - Detailed architecture
   - Optimization strategies
   - Troubleshooting

2. **Custom datasets:** `DATASET_DOCUMENTATION_INDEX.md`
   - Generate custom data
   - Modify patterns
   - Create specialized datasets

3. **Production deployment:** `best_model/README.md`
   - Integration examples
   - Performance optimization
   - Best practices

---

## 🔍 Find Information By Topic

### Model Usage
→ `models/stable_diffusion_2_all_4/best_model/README.md`
- Quick start (Python API)
- Example use cases
- Inference optimization
- Integration with pipeline

### Training Process
→ `FINETUNE_STABLE_DIFFUSION_README.md`
- CLI arguments
- Training loop
- Early stopping
- Checkpoint management

### Dataset Creation
→ `DATASET_DOCUMENTATION_INDEX.md`
- Pattern types
- Missing data types
- Image transformations
- PyTorch DataLoader

### Memory Issues
→ `FINETUNE_STABLE_DIFFUSION_README.md` (Troubleshooting)
- Batch size recommendations
- Mixed precision training
- Attention slicing
- Gradient checkpointing

### Performance Tuning
→ `best_model/README.md` (Advanced Configuration)
- Inference steps
- Guidance scale
- Sampling methods
- Speed optimizations

---

## 📈 Key Metrics Reference

### Model Performance

| Metric | Value | Source |
|--------|-------|--------|
| **Best Val Loss** | 0.03623 | `cross_validation_results.json` |
| **Training Samples** | ~3,000/fold | Training logs |
| **Model Size** | ~1 GB | Disk space |
| **Inference Speed** | ~5.5s @ 50 steps | TITAN RTX |

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU VRAM** | 8 GB | 16+ GB |
| **System RAM** | 16 GB | 32 GB |
| **Storage** | 20 GB | 50 GB |
| **CUDA** | 11.0+ | 11.8+ |

---

## 🔗 Related Documentation

### Time Series Processing
- `ts_image_inpainting.py` - Image encoders (GAF, MTF, RP, SPEC)
- `iterative_experiment.py` - Main experiment pipeline
- `EXPERIMENT_1_DESCRIPTION.md` - Experiment methodology

### Integration
- `integrate_custom_model.py` - Model integration script
- `models/stdiff.py` - SD2 inference wrapper

---

## 📝 Quick Reference Cards

### Training Quick Reference

```bash
# Basic training
python finetune_stable_diffusion.py

# Custom configuration
python finetune_stable_diffusion.py \
    --data_dir my_data \
    --output_dir my_model \
    --max_samples 8000 \
    --batch_size 4 \
    --learning_rate 1e-5 \
    --n_folds 2 \
    --train_ratio 0.75

# Low memory (8GB GPU)
python finetune_stable_diffusion.py \
    --batch_size 1 \
    --mixed_precision fp16 \
    --gradient_accumulation_steps 8
```

### Inference Quick Reference

```python
# Load model
from diffusers import StableDiffusionInpaintPipeline
pipeline = StableDiffusionInpaintPipeline.from_pretrained(
    "models/stable_diffusion_2_all_4/best_model"
).to("cuda")

# Inpaint
result = pipeline(
    prompt="high quality gramian angular field mathematical visualization",
    image=image,              # PIL Image 512x512
    mask_image=mask,          # PIL Image 512x512 (grayscale)
    num_inference_steps=50,
    guidance_scale=7.5
).images[0]

# Save
result.save("output.png")
```

### Dataset Generation Quick Reference

```bash
# Generate dataset
python generate_training_dataset.py --samples 2000

# Custom configuration
python generate_training_dataset.py \
    --samples 5000 \
    --output my_dataset \
    --seed 42
```

---

## ❓ FAQ

**Q: Where do I start if I just want to use the model?**  
A: Read `models/stable_diffusion_2_all_4/best_model/README.md` → Quick Start section

**Q: How do I train my own model?**  
A: Read `FINETUNE_STABLE_DIFFUSION_README.md` → Quick Start section

**Q: What if I get out of memory errors?**  
A: Check troubleshooting sections in both READMEs, reduce batch size, enable FP16

**Q: Can I use this on CPU?**  
A: No, GPU is required. Minimum 8GB VRAM.

**Q: How long does training take?**  
A: ~2-8 hours depending on GPU and configuration (see training README)

**Q: Can I fine-tune the model further?**  
A: Yes! Use `--resume_fold` to continue from existing checkpoint

**Q: What's the difference between the two READMEs?**  
A: Model README = how to USE, Training README = how to TRAIN

---

## 🎯 Checklist: Before You Start

### To Use the Model
- [ ] GPU with 8+ GB VRAM
- [ ] CUDA installed
- [ ] Python packages installed (`pip install diffusers transformers torch`)
- [ ] Model downloaded or trained
- [ ] Read: `best_model/README.md`

### To Train a Model
- [ ] GPU with 16+ GB VRAM (recommended)
- [ ] Training dataset generated
- [ ] ~50 GB free disk space
- [ ] Python packages installed
- [ ] Read: `FINETUNE_STABLE_DIFFUSION_README.md`

### To Generate Data
- [ ] Python packages installed (`pip install numpy pandas matplotlib pyts`)
- [ ] `ts_image_inpainting.py` available
- [ ] Read: `DATASET_DOCUMENTATION_INDEX.md`

---

## 📧 Support

For issues or questions:

1. **Model usage issues** → Check `best_model/README.md` troubleshooting
2. **Training issues** → Check `FINETUNE_STABLE_DIFFUSION_README.md` troubleshooting
3. **Dataset issues** → Check dataset documentation
4. **General questions** → Review this index for relevant documentation

---

**Last Updated:** December 2024  
**Documentation Version:** 1.0
