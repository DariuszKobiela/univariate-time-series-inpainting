# 📚 Training Dataset Documentation Index

Complete documentation for the Stable Diffusion 2 training dataset generation and usage.

---

## 📄 Documentation Files

### 1. **Dataset README** 
📁 Location: `stdiff_training_data/README.md`

**Contents:**
- Dataset structure and organization
- Statistics (2,000 samples, 16,000 images)
- Image types (GAF, MTF, RP, Spectrogram)
- Missing data types and patterns
- Metadata format
- Usage examples in Python
- Quality assurance guidelines

**Target Audience:** Users who want to USE the dataset

**Key Sections:**
- 📁 Directory Structure
- 📊 Dataset Statistics  
- 🎨 Image Types
- 🕳️ Missing Data Types
- 🚀 Usage Example
- 📋 Metadata Format

---

### 2. **Generation Script Documentation**
📁 Location: `GENERATE_TRAINING_DATASET_README.md`

**Contents:**
- Script architecture and design
- Command-line arguments
- Class documentation
- Advanced customization
- Integration examples
- Performance tips
- Troubleshooting

**Target Audience:** Users who want to GENERATE new datasets

**Key Sections:**
- 🚀 Usage & CLI Arguments
- 🏗️ Architecture (3 main classes)
- 📊 Output Structure
- 🔧 Advanced Usage
- 🐛 Error Handling
- ⚡ Performance Tips

---

## 🎯 Quick Start Guide

### I Want To: Use the Existing Dataset

1. Read: `stdiff_training_data/README.md`
2. Load the dataset in Python:

```python
from pathlib import Path
import json

# Load dataset summary
with open('stdiff_training_data/dataset_summary.json', 'r') as f:
    dataset = json.load(f)

print(f"Total samples: {dataset['total_samples']}")
```

3. See usage examples in: `stdiff_training_data/README.md` → Usage Example

---

### I Want To: Generate a New Dataset

1. Read: `GENERATE_TRAINING_DATASET_README.md`
2. Run the generation script:

```bash
python generate_training_dataset.py --samples 2000 --output my_dataset
```

3. Customize parameters as needed (see documentation)

---

### I Want To: Train a Model on This Dataset

1. Read: `stdiff_training_data/README.md` → Usage Example
2. Use the provided training script:

```bash
python finetune_stable_diffusion.py \
    --data_dir stdiff_training_data \
    --max_samples 4000 \
    --batch_size 4 \
    --max_epochs 300
```

---

## 📊 Dataset Overview

| Property | Value |
|----------|-------|
| **Location** | `stdiff_training_data/` |
| **Total Samples** | 2,000 time series |
| **Total Images** | 16,000 (8K original + 8K corrupted) |
| **Image Types** | GAF, MTF, RP, Spectrogram |
| **Image Size** | 64×64 RGB |
| **Disk Size** | ~1.2 GB |
| **Archive** | `stdiff_training_data.tar.gz` (1.2 GB) |

---

## 🔗 Related Files

### Core Files
- `generate_training_dataset.py` - Generation script
- `ts_image_inpainting.py` - Image encoding functions
- `finetune_stable_diffusion.py` - Training script

### Documentation
- `stdiff_training_data/README.md` - Dataset documentation
- `GENERATE_TRAINING_DATASET_README.md` - Generator documentation
- `DATASET_DOCUMENTATION_INDEX.md` - This file

### Data Files
- `stdiff_training_data.tar.gz` - Compressed dataset archive

---

## 🎓 Tutorials

### Tutorial 1: Loading and Visualizing Data

```python
import json
from PIL import Image
import matplotlib.pyplot as plt

# Load a sample
sample_id = 0
image_types = ['gaf', 'mtf', 'rp', 'spec']

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

for i, img_type in enumerate(image_types):
    # Original
    orig = Image.open(f'stdiff_training_data/original/{sample_id:06d}_{img_type}.png')
    axes[0, i].imshow(orig)
    axes[0, i].set_title(f'{img_type.upper()} - Original')
    
    # Corrupted
    miss = Image.open(f'stdiff_training_data/missing/{sample_id:06d}_{img_type}.png')
    axes[1, i].imshow(miss)
    axes[1, i].set_title(f'{img_type.upper()} - Corrupted')

plt.tight_layout()
plt.show()
```

### Tutorial 2: Creating a PyTorch Dataset

```python
from torch.utils.data import Dataset
import json
from pathlib import Path
from PIL import Image

class TimeSeriesDataset(Dataset):
    def __init__(self, data_dir='stdiff_training_data'):
        with open(Path(data_dir) / 'dataset_summary.json') as f:
            self.summary = json.load(f)
        self.samples = self.summary['samples']
        self.data_dir = Path(data_dir)
    
    def __len__(self):
        return len(self.samples) * 4  # 4 image types
    
    def __getitem__(self, idx):
        sample_idx = idx // 4
        img_type = ['gaf', 'mtf', 'rp', 'spec'][idx % 4]
        sample = self.samples[sample_idx]
        
        orig = Image.open(self.data_dir / f"original/{sample['series_id']:06d}_{img_type}.png")
        miss = Image.open(self.data_dir / f"missing/{sample['series_id']:06d}_{img_type}.png")
        
        return {'original': orig, 'corrupted': miss}

# Usage
dataset = TimeSeriesDataset()
print(f"Dataset size: {len(dataset)}")
```

### Tutorial 3: Generating Custom Dataset

```bash
# Small test dataset
python generate_training_dataset.py --samples 100 --output test_data

# Large production dataset
python generate_training_dataset.py --samples 5000 --output large_data

# Custom time series lengths
python generate_training_dataset.py \
    --samples 2000 \
    --min_length 200 \
    --max_length 800 \
    --output custom_lengths
```

---

## 📈 Statistics Summary

Generated dataset (`stdiff_training_data/`) contains:

### Pattern Distribution (approximately uniform):
- Sine: ~285 samples (14.2%)
- Cosine: ~287 samples (14.4%)
- Trend: ~290 samples (14.5%)
- Seasonal: ~283 samples (14.2%)
- Noise: ~281 samples (14.1%)
- Spikes: ~288 samples (14.4%)
- Mixed: ~286 samples (14.3%)

### Missing Type Distribution (approximately uniform):
- Random: ~508 samples (25.4%)
- Block: ~495 samples (24.8%)
- Periodic: ~501 samples (25.1%)
- Edge: ~496 samples (24.8%)

### Missing Rate Range:
- Min: ~5%
- Max: ~30%
- Mean: ~17.5%

---

## ❓ FAQ

**Q: Can I use this dataset for commercial purposes?**
A: Check the license file. Dataset is synthetically generated.

**Q: How do I extract the tar.gz archive?**
A: `tar -xzf stdiff_training_data.tar.gz`

**Q: Can I generate more samples?**
A: Yes! Use `generate_training_dataset.py --samples 5000`

**Q: What's the difference between image types?**
A: See `stdiff_training_data/README.md` → Image Types section

**Q: How long does training take?**
A: Depends on GPU. ~10-15 hours on NVIDIA TITAN RTX for 300 epochs.

**Q: Can I use only specific image types?**
A: Yes, filter in your DataLoader or modify the generation script.

---

## 📞 Support

For detailed information, refer to:

1. **Using the dataset?** → `stdiff_training_data/README.md`
2. **Generating new data?** → `GENERATE_TRAINING_DATASET_README.md`
3. **Training models?** → `finetune_stable_diffusion.py` documentation
4. **Issues?** → Check troubleshooting sections in respective READMEs

---

**Last Updated:** 2024-12-01
**Dataset Version:** 1.0
