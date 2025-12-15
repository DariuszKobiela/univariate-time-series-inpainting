# generate_training_dataset.py - Documentation

## Overview

`generate_training_dataset.py` is a Python script that generates synthetic time series datasets for training image-based inpainting models (particularly Stable Diffusion 2). It creates paired images showing original and corrupted time series in multiple representation formats (GAF, MTF, RP, Spectrogram).

---

## 🎯 Purpose

This script addresses the challenge of training deep learning models for time series inpainting by:

1. **Generating diverse synthetic time series** with various patterns
2. **Introducing controlled missingness** with different mechanisms
3. **Converting to image representations** (GAF, MTF, RP, Spectrogram)
4. **Creating paired training data** (original vs. corrupted)
5. **Organizing metadata** for easy training pipeline integration

---

## 📋 Requirements

### Python Packages

```bash
pip install numpy pandas matplotlib pillow tqdm
```

### Project Dependencies

The script depends on the following functions from `ts_image_inpainting.py`:
- `to_gaf()` - Gramian Angular Field transformation
- `to_mtf()` - Markov Transition Field transformation
- `to_rp()` - Recurrence Plot transformation
- `to_spectrogram()` - Spectrogram transformation
- `save_image()` - Image saving utility

---

## 🚀 Usage

### Basic Command

```bash
python generate_training_dataset.py --samples 2000
```

This generates:
- 2,000 time series samples
- 8,000 original images (2,000 × 4 types)
- 8,000 corrupted images (2,000 × 4 types)
- 2,000 metadata JSON files
- 1 dataset summary JSON file

### All Command-Line Arguments

```bash
python generate_training_dataset.py \
    --samples 2000 \              # Number of time series to generate
    --output stdiff_training_data \  # Output directory
    --seed 42 \                   # Random seed for reproducibility
    --min_length 100 \            # Minimum time series length
    --max_length 1000             # Maximum time series length
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--samples` | int | 2000 | Number of time series samples to generate |
| `--output` | str | "stdiff_training_data" | Output directory path |
| `--seed` | int | 42 | Random seed for reproducibility |
| `--min_length` | int | 100 | Minimum length of generated time series |
| `--max_length` | int | 1000 | Maximum length of generated time series |

---

## 🏗️ Architecture

The script consists of three main classes:

### 1. **TimeSeriesGenerator**

Generates synthetic time series with various patterns.

**Pattern Types:**
- `sine` - Sinusoidal waves
- `cosine` - Cosine waves
- `trend` - Linear/quadratic trends
- `seasonal` - Multi-frequency seasonal patterns
- `noise` - Random walks and noise
- `spikes` - Sudden jumps and spikes
- `mixed` - Random combination of multiple patterns

**Example:**

```python
generator = TimeSeriesGenerator(min_length=100, max_length=1000)

# Generate specific pattern
sine_series = generator.generate_synthetic_series(length=500, pattern_type="sine")

# Generate random pattern
random_series = generator.generate_synthetic_series(pattern_type="mixed")
```

### 2. **MissingDataGenerator**

Creates realistic missing data masks.

**Missing Types:**
- `random` - Randomly scattered missing points
- `block` - Contiguous blocks of missing data
- `periodic` - Regularly spaced missing points
- `edge` - Missing data at beginning/end

**Example:**

```python
generator = MissingDataGenerator()

# Create random mask (20% missing)
mask = generator.create_random_mask(length=1000, missing_rate=0.20)

# Create block mask (30% missing)
mask = generator.create_block_mask(length=1000, missing_rate=0.30)

# Create periodic mask (15% missing)
mask = generator.create_periodic_mask(length=1000, missing_rate=0.15)
```

### 3. **TrainingDatasetGenerator**

Main orchestrator that:
- Generates time series
- Applies missing data
- Converts to images
- Saves organized dataset
- Creates metadata

**Example:**

```python
generator = TrainingDatasetGenerator(output_dir="my_dataset")

# Generate single training pair
pair = generator.generate_training_pair(
    series_id=0,
    pattern_type="sine",
    missing_rate=0.20,
    missing_type="random"
)

# Save to disk
file_paths = generator.save_training_pair(pair)
```

---

## 📊 Output Structure

```
stdiff_training_data/
├── original/
│   ├── 000000_gaf.png      # Original GAF image
│   ├── 000000_mtf.png      # Original MTF image
│   ├── 000000_rp.png       # Original RP image
│   ├── 000000_spec.png     # Original Spectrogram
│   ├── 000001_gaf.png
│   └── ...
│
├── missing/
│   ├── 000000_gaf.png      # Corrupted GAF image
│   ├── 000000_mtf.png      # Corrupted MTF image
│   ├── 000000_rp.png       # Corrupted RP image
│   ├── 000000_spec.png     # Corrupted Spectrogram
│   └── ...
│
├── masks/
│   ├── 000000_metadata.json  # Metadata for sample 0
│   ├── 000001_metadata.json
│   └── ...
│
└── dataset_summary.json    # Complete dataset summary
```

---

## 📄 Metadata Format

### Per-Sample Metadata (`XXXXXX_metadata.json`)

```json
{
  "series_id": 0,
  "pattern_type": "sine",
  "missing_type": "random",
  "missing_rate": 0.20985669961447095,
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

### Dataset Summary (`dataset_summary.json`)

```json
{
  "total_samples": 2000,
  "image_types": ["gaf", "mtf", "rp", "spec"],
  "samples": [
    {
      "series_id": 0,
      "pattern_type": "sine",
      "missing_type": "random",
      "missing_rate": 0.20985669961447095,
      "length": 859,
      "file_paths": { ... }
    },
    ...
  ]
}
```

---

## 🔧 Advanced Usage

### Custom Pattern Distribution

Modify the pattern distribution by editing the `generate_dataset()` method:

```python
# In TrainingDatasetGenerator class

def generate_dataset(self, n_samples: int = 1000):
    pattern_types = ["sine", "cosine", "trend", "seasonal", "noise", "spikes", "mixed"]
    
    # Custom distribution (e.g., more sine waves)
    custom_distribution = {
        "sine": 0.30,      # 30%
        "cosine": 0.10,    # 10%
        "trend": 0.15,     # 15%
        "seasonal": 0.15,  # 15%
        "noise": 0.10,     # 10%
        "spikes": 0.10,    # 10%
        "mixed": 0.10      # 10%
    }
```

### Custom Missing Rate Range

```python
# Modify in generate_training_pair() method
missing_rate = random.uniform(0.10, 0.40)  # 10-40% instead of 5-30%
```

### Filtering Specific Image Types

Generate only specific image types:

```python
generator = TrainingDatasetGenerator()
generator.image_types = ["gaf", "mtf"]  # Only GAF and MTF
generator.encoders = {
    "gaf": to_gaf,
    "mtf": to_mtf
}
```

---

## 📊 Statistics & Quality Control

The script includes built-in statistics reporting:

```
=== DATASET GENERATION STATISTICS ===
Total samples generated: 2000
Pattern type distribution:
  - sine: 285 (14.2%)
  - cosine: 287 (14.4%)
  - trend: 290 (14.5%)
  - seasonal: 283 (14.2%)
  - noise: 281 (14.1%)
  - spikes: 288 (14.4%)
  - mixed: 286 (14.3%)

Missing type distribution:
  - random: 508 (25.4%)
  - block: 495 (24.8%)
  - periodic: 501 (25.1%)
  - edge: 496 (24.8%)

Missing rate statistics:
  - Min: 5.02%
  - Max: 29.98%
  - Mean: 17.45%
  - Median: 17.38%

Time series length statistics:
  - Min: 102
  - Max: 999
  - Mean: 551
  - Median: 552
```

---

## 🐛 Error Handling

The script includes robust error handling:

```python
try:
    image = encoder(series_filled)
    images[img_type] = image
except Exception as e:
    print(f"Warning: Failed to create {img_type} image: {e}")
    # Creates fallback empty image
    images[img_type] = np.zeros((64, 64), dtype=np.float32)
```

Failed samples are logged but don't stop generation.

---

## 🔬 Integration with Training Pipeline

### With Stable Diffusion Fine-tuning

```bash
# 1. Generate dataset
python generate_training_dataset.py --samples 2000

# 2. Train Stable Diffusion model
python finetune_stable_diffusion.py \
    --data_dir stdiff_training_data \
    --output_dir models/stable_diffusion_2_all_4 \
    --max_samples 4000 \
    --batch_size 4 \
    --max_epochs 300
```

### With Custom PyTorch DataLoader

```python
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json

class SD2InpaintingDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        
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
        series_id = sample['series_id']
        
        # Load images
        original = Image.open(
            self.data_dir / f"original/{series_id:06d}_{img_type}.png"
        )
        missing = Image.open(
            self.data_dir / f"missing/{series_id:06d}_{img_type}.png"
        )
        
        if self.transform:
            original = self.transform(original)
            missing = self.transform(missing)
        
        return {
            'original': original,
            'missing': missing,
            'metadata': sample
        }

# Usage
dataset = SD2InpaintingDataset('stdiff_training_data')
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
```

---

## 🎨 Visualization

Quick visualization of generated samples:

```python
import matplotlib.pyplot as plt
from PIL import Image

def visualize_sample(sample_id=0, data_dir='stdiff_training_data'):
    """Visualize one sample with all image types"""
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    image_types = ['gaf', 'mtf', 'rp', 'spec']
    
    for i, img_type in enumerate(image_types):
        # Original
        orig = Image.open(f"{data_dir}/original/{sample_id:06d}_{img_type}.png")
        axes[0, i].imshow(orig)
        axes[0, i].set_title(f"{img_type.upper()} - Original")
        axes[0, i].axis('off')
        
        # Missing
        miss = Image.open(f"{data_dir}/missing/{sample_id:06d}_{img_type}.png")
        axes[1, i].imshow(miss)
        axes[1, i].set_title(f"{img_type.upper()} - Corrupted")
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'sample_{sample_id}_visualization.png', dpi=150)
    plt.show()

# Visualize first 5 samples
for i in range(5):
    visualize_sample(i)
```

---

## ⚡ Performance Tips

### Speed Optimization

1. **Use multiprocessing for large datasets:**

```python
from multiprocessing import Pool

def generate_sample_wrapper(args):
    series_id, generator = args
    return generator.generate_training_pair(series_id)

# Use with Pool
with Pool(processes=8) as pool:
    results = pool.map(generate_sample_wrapper, [(i, gen) for i in range(n_samples)])
```

2. **Reduce image resolution** (faster encoding):

```python
# In ts_image_inpainting.py, modify image size
IMAGE_SIZE = 32  # Instead of 64
```

3. **Skip expensive transformations** during testing:

```python
generator.image_types = ["gaf"]  # Only one type for quick testing
```

### Memory Optimization

- Generate in batches if memory is limited
- Clear caches between iterations
- Use generators instead of loading all at once

---

## 📝 Example Scripts

### Complete Generation Script

```python
#!/usr/bin/env python3
"""
Complete example of dataset generation with custom settings
"""

from generate_training_dataset import TrainingDatasetGenerator

# Initialize generator
generator = TrainingDatasetGenerator(
    output_dir="my_custom_dataset"
)

# Custom settings
generator.ts_generator.min_length = 200
generator.ts_generator.max_length = 800

# Generate dataset
print("Generating custom dataset...")
generator.generate_dataset(
    n_samples=1000,
    pattern_distribution={
        "sine": 0.2,
        "trend": 0.2,
        "seasonal": 0.3,
        "mixed": 0.3
    }
)

print("✅ Dataset generation complete!")
```

### Validation Script

```python
#!/usr/bin/env python3
"""
Validate generated dataset integrity
"""

import json
from pathlib import Path
from PIL import Image

def validate_dataset(data_dir='stdiff_training_data'):
    print(f"Validating dataset: {data_dir}")
    
    # Load summary
    with open(Path(data_dir) / 'dataset_summary.json', 'r') as f:
        summary = json.load(f)
    
    n_samples = summary['total_samples']
    image_types = summary['image_types']
    
    print(f"Expected samples: {n_samples}")
    print(f"Image types: {image_types}")
    
    # Check all files exist
    missing_files = []
    corrupted_images = []
    
    for sample in summary['samples']:
        series_id = sample['series_id']
        
        for img_type in image_types:
            # Check original
            orig_path = Path(data_dir) / f"original/{series_id:06d}_{img_type}.png"
            if not orig_path.exists():
                missing_files.append(str(orig_path))
            else:
                try:
                    img = Image.open(orig_path)
                    img.verify()
                except Exception as e:
                    corrupted_images.append((str(orig_path), str(e)))
            
            # Check missing
            miss_path = Path(data_dir) / f"missing/{series_id:06d}_{img_type}.png"
            if not miss_path.exists():
                missing_files.append(str(miss_path))
            else:
                try:
                    img = Image.open(miss_path)
                    img.verify()
                except Exception as e:
                    corrupted_images.append((str(miss_path), str(e)))
    
    # Report
    if missing_files:
        print(f"\n❌ Missing files ({len(missing_files)}):")
        for f in missing_files[:10]:
            print(f"  - {f}")
    
    if corrupted_images:
        print(f"\n❌ Corrupted images ({len(corrupted_images)}):")
        for f, e in corrupted_images[:10]:
            print(f"  - {f}: {e}")
    
    if not missing_files and not corrupted_images:
        print("\n✅ Dataset validation passed!")
        print(f"   All {n_samples * len(image_types) * 2} images are present and valid.")
    
    return len(missing_files) == 0 and len(corrupted_images) == 0

if __name__ == "__main__":
    validate_dataset()
```

---

## 🔗 Related Documentation

- **Dataset README:** `stdiff_training_data/README.md`
- **Training Script:** `finetune_stable_diffusion.py`
- **Main Experiments:** `iterative_experiment.py`
- **Image Encoders:** `ts_image_inpainting.py`

---

## 📧 Troubleshooting

### Common Issues

**Issue:** `ImportError: cannot import name 'to_gaf'`
- **Solution:** Ensure `ts_image_inpainting.py` is in the same directory

**Issue:** Out of memory during generation
- **Solution:** Reduce `--samples` or generate in batches

**Issue:** Image encoding fails for certain series
- **Solution:** Check time series length and adjust `min_length`/`max_length`

**Issue:** Unbalanced pattern distribution
- **Solution:** Set fixed random seed: `--seed 42`

---

## 📚 References

1. **Gramian Angular Field (GAF):** Wang & Oates (2015)
2. **Markov Transition Field (MTF):** Wang & Oates (2015)
3. **Recurrence Plots (RP):** Eckmann et al. (1987)
4. **Spectrograms:** Standard signal processing technique

---

**Script Version:** 1.0  
**Last Updated:** 2024-09-08  
**Author:** [Your Name]

