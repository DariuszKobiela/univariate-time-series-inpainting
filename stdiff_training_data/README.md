# Stable Diffusion 2 Training Dataset for Time Series Inpainting

This directory contains a synthetically generated dataset for fine-tuning Stable Diffusion 2 models on time series inpainting tasks. The dataset consists of paired images showing original and corrupted time series representations in multiple imaging modalities.

---

## 📁 Directory Structure

```
stdiff_training_data/
├── README.md                    # This file
├── dataset_summary.json         # Complete dataset metadata (1.5 MB)
│
├── original/                    # Original (complete) time series images
│   ├── 000000_gaf.png          # Gramian Angular Field
│   ├── 000000_mtf.png          # Markov Transition Field
│   ├── 000000_rp.png           # Recurrence Plot
│   ├── 000000_spec.png         # Spectrogram
│   ├── 000001_gaf.png
│   └── ... (8,000 images total: 2,000 samples × 4 types)
│
├── missing/                     # Corrupted time series images (with missing data)
│   ├── 000000_gaf.png
│   ├── 000000_mtf.png
│   ├── 000000_rp.png
│   ├── 000000_spec.png
│   └── ... (8,000 images total: 2,000 samples × 4 types)
│
└── masks/                       # Metadata for each sample
    ├── 000000_metadata.json    # Contains pattern type, missing type, rates, etc.
    ├── 000001_metadata.json
    └── ... (2,000 JSON files)
```

---

## 📊 Dataset Statistics

| Property | Value |
|----------|-------|
| **Total Samples** | 2,000 time series |
| **Total Images** | 16,000 (8,000 original + 8,000 corrupted) |
| **Image Types** | 4 (GAF, MTF, RP, Spectrogram) |
| **Image Size** | 64×64 pixels (RGB) |
| **Disk Size** | ~1.2 GB (compressed to ~1.2 GB tar.gz) |
| **Time Series Length** | Variable (100-1,000 points) |
| **Missing Data Rate** | 5% - 30% per sample |

---

## 🎨 Image Types

Each time series is transformed into four different 2D representations:

### 1. **GAF** (Gramian Angular Field)
- Encodes temporal correlation as angular summation
- Preserves temporal dynamics in polar coordinates
- Best for: periodic and oscillatory patterns

### 2. **MTF** (Markov Transition Field)
- Captures state transition probabilities
- Encodes temporal dependencies as transitions
- Best for: sequential patterns and state changes

### 3. **RP** (Recurrence Plot)
- Visualizes recurrence of states in phase space
- Shows periodicity and chaos
- Best for: complex dynamical systems

### 4. **SPEC** (Spectrogram)
- Frequency-domain representation
- Shows time-frequency evolution
- Best for: frequency analysis and spectral patterns

---

## 🔢 Sample Patterns

The dataset includes diverse time series patterns:

| Pattern Type | Description | Count |
|--------------|-------------|-------|
| **sine** | Simple sinusoidal waves | ~285 |
| **cosine** | Cosine waves | ~285 |
| **trend** | Linear/quadratic trends | ~285 |
| **seasonal** | Multiple frequency components | ~285 |
| **noise** | Random walk/noise | ~285 |
| **spikes** | Sudden jumps/spikes | ~285 |
| **mixed** | Combination of above | ~290 |

*Distribution is approximately uniform across all pattern types.*

---

## 🕳️ Missing Data Types

Four types of missingness are simulated:

### 1. **Random** (~25% of samples)
- Randomly scattered missing points
- Simulates: sensor dropouts, random failures

### 2. **Block** (~25% of samples)
- Contiguous blocks of missing data
- Simulates: sensor downtime, power outages

### 3. **Periodic** (~25% of samples)
- Regularly spaced missing points
- Simulates: scheduled maintenance, sampling issues

### 4. **Edge** (~25% of samples)
- Missing data at beginning/end
- Simulates: late sensor activation, early shutdown

---

## 📋 Metadata Format

Each `XXXXXX_metadata.json` file contains:

```json
{
  "series_id": 0,
  "pattern_type": "sine",
  "missing_type": "random",
  "missing_rate": 0.2098,
  "length": 859,
  "file_paths": {
    "original_gaf": "stdiff_training_data/original/000000_gaf.png",
    "missing_gaf": "stdiff_training_data/missing/000000_gaf.png",
    "original_mtf": "stdiff_training_data/original/000000_mtf.png",
    "missing_mtf": "stdiff_training_data/missing/000000_mtf.png",
    "original_rp": "stdiff_training_data/original/000000_rp.png",
    "missing_rp": "stdiff_training_data/missing/000000_rp.png",
    "original_spec": "stdiff_training_data/original/000000_spec.png",
    "missing_spec": "stdiff_training_data/missing/000000_spec.png"
  }
}
```

---

## 🎯 Intended Use

This dataset is designed for:

1. **Fine-tuning Stable Diffusion 2** for time series inpainting
2. Training **image-to-image translation models** (U-Net, pix2pix, etc.)
3. **Research** on time series reconstruction via image inpainting
4. **Benchmarking** deep learning inpainting methods

---

## 🚀 Usage Example

### Loading the Dataset in Python

```python
import json
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset

class TimeSeriesInpaintingDataset(Dataset):
    def __init__(self, data_dir='stdiff_training_data'):
        self.data_dir = Path(data_dir)
        
        # Load dataset summary
        with open(self.data_dir / 'dataset_summary.json', 'r') as f:
            self.summary = json.load(f)
        
        self.samples = self.summary['samples']
        self.image_types = ['gaf', 'mtf', 'rp', 'spec']
    
    def __len__(self):
        return len(self.samples) * len(self.image_types)
    
    def __getitem__(self, idx):
        sample_idx = idx // len(self.image_types)
        img_type = self.image_types[idx % len(self.image_types)]
        
        sample = self.samples[sample_idx]
        
        # Load images
        original_path = self.data_dir / f"original/{sample['series_id']:06d}_{img_type}.png"
        missing_path = self.data_dir / f"missing/{sample['series_id']:06d}_{img_type}.png"
        
        original_img = Image.open(original_path).convert('RGB')
        missing_img = Image.open(missing_path).convert('RGB')
        
        return {
            'original': original_img,
            'missing': missing_img,
            'metadata': sample
        }

# Usage
dataset = TimeSeriesInpaintingDataset()
print(f"Total training pairs: {len(dataset)}")
```

### Using with Stable Diffusion Fine-tuning

```python
from torch.utils.data import DataLoader

# Create dataset
dataset = TimeSeriesInpaintingDataset()

# Split train/val
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size]
)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# Use with your Stable Diffusion training script
# See: finetune_stable_diffusion.py
```

---

## 🔧 Regenerating the Dataset

If you need to regenerate or create a custom dataset, use the `generate_training_dataset.py` script:

### Basic Usage

```bash
# Generate 2,000 samples (default)
python generate_training_dataset.py --samples 2000

# Generate larger dataset
python generate_training_dataset.py --samples 5000 --output stdiff_training_data_large

# Custom seed for reproducibility
python generate_training_dataset.py --samples 1000 --seed 42
```

### Advanced Options

```bash
python generate_training_dataset.py \
    --samples 3000 \
    --output my_custom_dataset \
    --seed 123 \
    --min_length 200 \
    --max_length 800
```

See [generate_training_dataset.py Documentation](#generate-training-dataset-script) below for full details.

---

## 📈 Quality Assurance

### Image Quality
- All images are 64×64 RGB PNG files
- Normalized to [0, 255] range
- Anti-aliasing applied during generation

### Data Quality
- No duplicate time series
- All samples validated for:
  - Non-empty time series
  - Valid missing data masks
  - Successful image generation
- Failed samples are logged and skipped

### Validation Checks

```python
import json
from pathlib import Path

# Load summary
with open('stdiff_training_data/dataset_summary.json', 'r') as f:
    summary = json.load(f)

# Check completeness
print(f"Total samples: {summary['total_samples']}")
print(f"Image types: {summary['image_types']}")

# Verify files exist
data_dir = Path('stdiff_training_data')
for sample in summary['samples'][:10]:  # Check first 10
    series_id = sample['series_id']
    for img_type in ['gaf', 'mtf', 'rp', 'spec']:
        orig = data_dir / f"original/{series_id:06d}_{img_type}.png"
        miss = data_dir / f"missing/{series_id:06d}_{img_type}.png"
        assert orig.exists(), f"Missing: {orig}"
        assert miss.exists(), f"Missing: {miss}"
        
print("✅ Dataset validation passed!")
```

---

## 📝 Citation

If you use this dataset in your research, please cite:

```bibtex
@misc{timeseries_inpainting_dataset_2024,
  title={Synthetic Time Series Inpainting Dataset for Stable Diffusion},
  author={[Your Name]},
  year={2024},
  howpublished={https://github.com/[your-repo]},
  note={Dataset of 2,000 synthetic time series with 16,000 images in GAF, MTF, RP, and Spectrogram representations}
}
```

---

## 🔗 Related Files

- **Training Script:** `finetune_stable_diffusion.py`
- **Generation Script:** `generate_training_dataset.py`
- **Image Encoders:** `ts_image_inpainting.py`
- **Experiment Runner:** `iterative_experiment.py`

---

## 📞 Support

For issues or questions:
1. Check the main project README
2. Review `generate_training_dataset.py` documentation
3. Examine example metadata in `dataset_summary.json`

---

**Generated:** 2025-12-01 
**Dataset Version:** 1.0  
**Format Version:** 1.0

