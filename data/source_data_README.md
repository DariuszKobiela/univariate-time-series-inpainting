# Experimental Datasets - Time Series Inpainting

This directory contains all datasets used in the time series inpainting experiments, including original data, corrupted versions, reconstructed results, and intermediate image representations.

---

## 📁 Directory Structure

```
data/
├── README.md                      # This file
│
├── 0_source_data/                 # Original univariate industrial time series (7 files, 8.5 MB)
│   ├── boiler_outlet_temp_univ.csv
│   ├── pump_sensor_28_univ.csv
│   ├── vibration_sensor_S1.csv
│   ├── water_level_sensors_2010_L300.csv
│   ├── water_level_sensors_2010_L308.csv
│   └── water_level_sensors_2010_L311.csv
│
├── 1_missing_data/                # Time series with injected missing values (721 files, 964 MB)
│   ├── [dataset]_[mechanism]_[rate]_[iter].csv
│   └── ...
│
├── 2_fixed_data/                  # Reconstructed time series (16,201 files, 25 GB)
│   ├── [dataset]_[mechanism]_[rate]_[iter]_[method].csv
│   └── ...
│
└── images_inpainting/             # Image representations (16,324 files, 4 GB)
    ├── 0_original_images/         # Original GAF/MTF/RP/SPEC images
    ├── 1_missing_images/          # Corrupted images
    ├── 2_fixed_images/            # Reconstructed images
    └── 3_difference_images/       # Visualization of differences
```

---

## 📊 Dataset Statistics

| Directory | Files | Size | Description |
|-----------|-------|------|-------------|
| **0_source_data** | 7 | 8.5 MB | Original industrial time series |
| **1_missing_data** | 721 | 964 MB | Corrupted versions (systematic missingness) |
| **2_fixed_data** | 16,201 | 25 GB | Reconstructed by 31 methods |
| **images_inpainting** | 16,324 | 4 GB | Image transformations (GAF/MTF/RP/SPEC) |
| **Total** | **33,253** | **~30 GB** | Complete experimental dataset |

---

## 📈 0_source_data/

### Description

Original, complete industrial time series datasets without any missing values. These serve as ground truth for all experiments.

### Contents

| File | Source | Points | Sampling | Description |
|------|--------|--------|----------|-------------|
| `boiler_outlet_temp_univ.csv` | Industrial boiler | ~10,000 | 1 min | Temperature sensor readings |
| `pump_sensor_28_univ.csv` | Industrial pump | ~10,000 | 1 min | Pressure/flow measurements |
| `vibration_sensor_S1.csv` | Vibration monitor | ~10,000 | High freq | Mechanical vibration data |
| `water_level_sensors_2010_L300.csv` | Water monitoring | ~8,760 | 1 hour | Water level station L300 |
| `water_level_sensors_2010_L308.csv` | Water monitoring | ~8,760 | 1 hour | Water level station L308 |
| `water_level_sensors_2010_L311.csv` | Water monitoring | ~8,760 | 1 hour | Water level station L311 |

### Format

All files are CSV with two columns:
```csv
timestamp,value
2023-01-01 00:00:00,42.5
2023-01-01 00:01:00,42.8
...
```

### Archive

```bash
# Compressed archive
data_0_source.tar.gz (1.8 MB)

# Extract
tar -xzf ../data_0_source.tar.gz
```

---

## 🕳️ 1_missing_data/

### Description

Time series with systematically injected missing values. Each file corresponds to one experimental configuration with controlled missingness.

### Naming Convention

```
[dataset]_[mechanism]_[rate]_[iteration].csv
```

**Examples:**
- `boiler_MCAR_2p_1.csv` - Boiler, MCAR mechanism, 2% missing, iteration 1
- `pump_MAR_5p_3.csv` - Pump, MAR mechanism, 5% missing, iteration 3
- `vibration_MNAR_10p_7.csv` - Vibration, MNAR mechanism, 10% missing, iteration 7

### Experimental Design

| Parameter | Values | Count |
|-----------|--------|-------|
| **Datasets** | 3 (boiler, pump, vibration) | 3 |
| **Mechanisms** | MCAR, MAR, MNAR | 3 |
| **Missing Rates** | 2%, 5%, 10% | 3 |
| **Iterations** | 1-10 | 10 |
| **Total Configs** | 3 × 3 × 3 × 10 | **270** per dataset |

**Note:** Water level datasets (L300, L308, L311) were excluded from main experiments.

### Missingness Mechanisms

**MCAR (Missing Completely At Random):**
- Random points removed
- No systematic pattern
- Simulates: random sensor failures

**MAR (Missing At Random):**
- Probability depends on observed values
- Conditional missingness
- Simulates: threshold-based dropout

**MNAR (Missing Not At Random):**
- Probability depends on missing values themselves
- Systematic bias
- Simulates: sensor saturation, range limitations

### Format

Same as source data, but with `NaN` values:
```csv
timestamp,value
2023-01-01 00:00:00,42.5
2023-01-01 00:01:00,NaN
2023-01-01 00:02:00,42.8
...
```

### Archive

```bash
# Compressed archive
data_1_missing.tar.gz (192 MB)

# Extract
tar -xzf ../data_1_missing.tar.gz
```

---

## 🔧 2_fixed_data/

### Description

Time series reconstructed using 31 different methods. This is the main experimental output containing all reconstruction results.

### Naming Convention

```
[dataset]_[mechanism]_[rate]_[iteration]_[method].csv
```

**Examples:**
- `boiler_MCAR_2p_1_imputemean.csv` - Mean imputation
- `pump_MAR_5p_3_interpolatecubic.csv` - Cubic interpolation
- `vibration_MNAR_10p_7_gafunet.csv` - GAF + U-Net inpainting
- `boiler_MCAR_2p_1_rpsd2all4.csv` - RP + Stable Diffusion 2

### Reconstruction Methods (31 total)

#### Classical Methods (15)

**Statistical Imputation (3):**
- `imputemean` - Mean of observed values
- `imputemedian` - Median of observed values
- `imputemode` - Mode of observed values

**Directional Fill (2):**
- `imputeffill` - Forward fill (propagate last valid)
- `imputebfill` - Backward fill (propagate next valid)

**Interpolation (9):**
- `interpolatenearest` - Nearest neighbor
- `interpolatelinear` - Linear interpolation
- `interpolateindex` - Index-based
- `interpolatequadratic` - Quadratic polynomial
- `interpolatecubic` - Cubic spline
- `interpolatepolynomial` - High-order polynomial
- `interpolatepchip` - Piecewise Cubic Hermite
- `interpolateakima` - Akima spline
- `interpolatespline` - Smoothing spline

**Machine Learning (1):**
- `knn` - K-Nearest Neighbors
- `sarimax` - SARIMAX time series model

#### Image-based Methods (16)

**U-Net Models (4):**
- `gafunet` - GAF + U-Net
- `mtfunet` - MTF + U-Net
- `rpunet` - RP + U-Net
- `specunet` - Spectrogram + U-Net

**Stable Diffusion 2 - Fine-tuned (4):**
- `gafsd2all4` - GAF + SD2 (trained on all 4 types)
- `mtfsd2all4` - MTF + SD2 (trained on all 4 types)
- `rpsd2all4` - RP + SD2 (trained on all 4 types)
- `specsd2all4` - SPEC + SD2 (trained on all 4 types)

### Pipeline: Image-based Methods

```
Time Series → Image Transform → Inpaint → Inverse Transform → Time Series
    (CSV)    → (GAF/MTF/RP/SPEC) → (U-Net/SD2) → (Inverse) → (CSV)
```

### Format

Same as source data, with reconstructed values:
```csv
timestamp,value
2023-01-01 00:00:00,42.5
2023-01-01 00:01:00,42.7    # Reconstructed (was NaN)
2023-01-01 00:02:00,42.8
...
```

### Statistics

- **Total files:** 16,201
- **Per dataset:** ~5,400 files
- **Per config:** 31 methods
- **Size:** ~25 GB total (~1.5 MB per file average)

### Archive

```bash
# Compressed archive
data_2_fixed.tar.gz (5.3 GB)

# Extract
tar -xzf ../data_2_fixed.tar.gz
```

---

## 🎨 images_inpainting/

### Description

Image representations of time series at various stages of the inpainting pipeline. Used for visualization and debugging.

### Subdirectories

#### 0_original_images/
Complete time series converted to images (ground truth).

**Format:** `[dataset]_[config]_[type].png`

#### 1_missing_images/
Corrupted time series as images (input to inpainting models).

**Format:** `[dataset]_[config]_[type]_missing.png`

#### 2_fixed_images/
Inpainted images (output from U-Net/SD2).

**Format:** `[dataset]_[config]_[method]_[type].png`

#### 3_difference_images/
Visual differences between original and reconstructed.

**Format:** `[dataset]_[config]_[method]_[type]_diff.png`

### Image Types

| Type | Full Name | Size | Description |
|------|-----------|------|-------------|
| **gaf** | Gramian Angular Field | 64×64 | Polar encoding of temporal correlations |
| **mtf** | Markov Transition Field | 64×64 | State transition probabilities |
| **rp** | Recurrence Plot | 64×64 | Phase space recurrence patterns |
| **spec** | Spectrogram | 64×64 | Time-frequency representation |

### Statistics

- **Total images:** 16,324
- **Original:** ~4,000 images
- **Missing:** ~4,000 images
- **Fixed:** ~6,000 images
- **Differences:** ~2,324 images
- **Size:** ~4 GB total

### Archive

```bash
# Compressed archive
data_images.tar.gz (4.0 GB)

# Extract
tar -xzf ../data_images.tar.gz
```

---

## 🔬 Experimental Workflow

### Complete Pipeline

```
1. SOURCE DATA (0_source_data/)
   ↓ Inject missing values (MCAR/MAR/MNAR)
   
2. MISSING DATA (1_missing_data/)
   ↓ Convert to images (optional for image methods)
   
3. IMAGES - MISSING (images_inpainting/1_missing_images/)
   ↓ Apply reconstruction (31 methods)
   
4A. IMAGES - FIXED (images_inpainting/2_fixed_images/)
    ↓ Inverse transform to time series
    
4B. FIXED DATA (2_fixed_data/)
    ↓ Compute metrics (MAPE, MAE, RMSE)
    
5. RESULTS (results/quick_experiment/df_final_*.csv)
```

### Files Generated Per Configuration

For each experimental configuration (e.g., `boiler_MCAR_2p_1`):

1. **1 missing file** → `1_missing_data/boiler_MCAR_2p_1.csv`
2. **4 original images** → `images_inpainting/0_original_images/boiler_MCAR_2p_1_[gaf|mtf|rp|spec].png`
3. **4 missing images** → `images_inpainting/1_missing_images/...`
4. **31 reconstructed files** → `2_fixed_data/boiler_MCAR_2p_1_[method].csv`
5. **16 reconstructed images** → `images_inpainting/2_fixed_images/...` (4 methods × 4 types)

**Total per config:** ~50 files

---

## 📊 Data Generation Scripts

### Source Data
Original data was collected from industrial systems. No generation script needed.

### Missing Data
```bash
# Generated by iterative_experiment.py
python iterative_experiment.py --generate-missing-only
```

### Reconstructed Data
```bash
# Generated by main experiment
python iterative_experiment.py
```

### Images
```bash
# Generated automatically during image-based reconstruction
# See: ts_image_inpainting.py
```

---

## 💾 Storage & Archives

### Individual Archives

| Archive | Original Size | Compressed Size | Compression Ratio |
|---------|---------------|-----------------|-------------------|
| `data_0_source.tar.gz` | 8.5 MB | 1.8 MB | 4.7:1 |
| `data_1_missing.tar.gz` | 964 MB | 192 MB | 5.0:1 |
| `data_2_fixed.tar.gz` | 25 GB | 5.3 GB | 4.7:1 |
| `data_images.tar.gz` | 4 GB | 4.0 GB | 1.0:1 (PNG) |
| **Total** | **~30 GB** | **~10.5 GB** | **2.9:1** |

### Extracting Archives

```bash
# Extract all
tar -xzf data_0_source.tar.gz
tar -xzf data_1_missing.tar.gz
tar -xzf data_2_fixed.tar.gz
tar -xzf data_images.tar.gz

# Extract to specific directory
tar -xzf data_2_fixed.tar.gz -C /path/to/destination/
```

### Creating Archives (for backup)

```bash
# Compress individual directories
tar -czf data_0_source.tar.gz data/0_source_data/
tar -czf data_1_missing.tar.gz data/1_missing_data/
tar -czf data_2_fixed.tar.gz data/2_fixed_data/
tar -czf data_images.tar.gz data/images_inpainting/

# Compress everything
tar -czf data_complete.tar.gz data/
```

---

## 🔍 Data Access Examples

### Loading Source Data

```python
import pandas as pd

# Load original time series
df = pd.read_csv('data/0_source_data/boiler_outlet_temp_univ.csv', 
                 index_col='timestamp', parse_dates=True)

print(f"Length: {len(df)}")
print(f"Missing: {df['value'].isna().sum()}")
```

### Loading Corrupted Data

```python
# Load corrupted version
df_missing = pd.read_csv('data/1_missing_data/boiler_MCAR_2p_1.csv',
                         index_col='timestamp', parse_dates=True)

# Count missing values
n_missing = df_missing['value'].isna().sum()
missing_rate = n_missing / len(df_missing) * 100

print(f"Missing: {n_missing} ({missing_rate:.1f}%)")
```

### Loading Reconstructed Data

```python
# Load reconstruction
df_fixed = pd.read_csv('data/2_fixed_data/boiler_MCAR_2p_1_gafunet.csv',
                       index_col='timestamp', parse_dates=True)

# Compare with original
df_original = pd.read_csv('data/0_source_data/boiler_outlet_temp_univ.csv',
                          index_col='timestamp', parse_dates=True)

# Compute error on missing regions only
mask = df_missing['value'].isna()
errors = abs(df_original.loc[mask, 'value'] - df_fixed.loc[mask, 'value'])
mae = errors.mean()

print(f"MAE on missing regions: {mae:.4f}")
```

### Loading Images

```python
from PIL import Image
import numpy as np

# Load original image
img = Image.open('data/images_inpainting/0_original_images/boiler_MCAR_2p_1_gaf.png')
img_array = np.array(img)

print(f"Shape: {img_array.shape}")
print(f"Type: {img_array.dtype}")
```

---

## 📈 Dataset Characteristics

### Time Series Properties

| Dataset | Length | Min | Max | Mean | Std | Trend | Seasonality |
|---------|--------|-----|-----|------|-----|-------|-------------|
| **Boiler** | ~10k | 25.3 | 89.7 | 58.2 | 12.4 | Stable | Weak |
| **Pump** | ~10k | 1.2 | 98.5 | 45.6 | 28.7 | None | Strong |
| **Vibration** | ~10k | 0.001 | 2.456 | 0.542 | 0.389 | Increasing | None |
| **Water L300** | ~8.7k | 2.1 | 5.8 | 3.4 | 0.8 | Seasonal | Strong |
| **Water L308** | ~8.7k | 1.8 | 6.2 | 3.2 | 0.9 | Seasonal | Strong |
| **Water L311** | ~8.7k | 2.4 | 5.5 | 3.6 | 0.7 | Seasonal | Strong |

### Missing Data Distribution

**By Mechanism:**
- MCAR: 33.3% of configs (90 per dataset)
- MAR: 33.3% of configs (90 per dataset)
- MNAR: 33.3% of configs (90 per dataset)

**By Rate:**
- 2%: 33.3% of configs (90 per dataset)
- 5%: 33.3% of configs (90 per dataset)
- 10%: 33.3% of configs (90 per dataset)

**By Dataset:**
- Boiler: 270 configs
- Pump: 270 configs
- Vibration: 270 configs
- **Total:** 810 configs

---

## 🔧 Maintenance & Cleanup

### Checking Data Integrity

```bash
# Count files in each directory
echo "Source: $(ls data/0_source_data/*.csv 2>/dev/null | wc -l)"
echo "Missing: $(ls data/1_missing_data/*.csv 2>/dev/null | wc -l)"
echo "Fixed: $(ls data/2_fixed_data/*.csv 2>/dev/null | wc -l)"

# Check for empty files
find data/ -type f -empty
```

### Disk Space Management

```bash
# Check space usage
du -sh data/*/

# Remove intermediate images (if needed)
rm -rf data/images_inpainting/3_difference_images/

# Keep only essential data
# Warning: This removes all but source and final results
rm -rf data/1_missing_data/
rm -rf data/images_inpainting/
```

### Regenerating Data

```bash
# Regenerate missing data
python iterative_experiment.py --force-regenerate-missing

# Regenerate specific method
python iterative_experiment.py --methods gafunet --force-reprocess
```

---

## 📚 Related Files & Documentation

### Scripts
- `iterative_experiment.py` - Main experiment runner (generates 1_missing_data/, 2_fixed_data/)
- `ts_image_inpainting.py` - Image transformation functions
- `calculate_differences.py` - Computes metrics from reconstructed data
- `generate_training_dataset.py` - Generates synthetic training data (separate from this)

### Results
- `results/quick_experiment/df_final_*.csv` - Aggregated metrics
- `results/quick_experiment/final_results.json` - Complete experimental results

### Documentation
- `EXPERIMENT_1_DESCRIPTION.md` - Experiment methodology
- `README.md` (project root) - Main project documentation

---

## ⚠️ Important Notes

### Data Provenance

- **Source data:** Real industrial sensors (anonymized)
- **Missing data:** Synthetically generated with controlled mechanisms
- **Reconstructions:** Generated by automated pipeline
- **Images:** Deterministic transformations from time series

### Reproducibility

All data can be regenerated from source:
1. Source data (0_source_data/) - original, no changes
2. Missing data (1_missing_data/) - regenerate with same seeds
3. Fixed data (2_fixed_data/) - regenerate by running experiments
4. Images (images_inpainting/) - regenerate during reconstruction

### Data Quality

**Validated:**
- ✅ All files have correct format
- ✅ No corrupted CSV files
- ✅ Missing rates match specifications
- ✅ All methods completed successfully

**Known Issues:**
- Some image-based methods may fail on extreme cases (handled gracefully)
- Large file count may cause filesystem issues on some systems

---

## 🔒 Data Usage & Citation

### Usage Guidelines

This dataset is for **research purposes** as part of the time series inpainting project.

**Allowed:**
- Analysis and visualization
- Method comparison
- Result reproduction
- Academic publication

**Restrictions:**
- Do not redistribute source data without permission
- Cite original data sources when publishing
- Acknowledge reconstruction methods used

### Citation

If you use this dataset in your research:

```bibtex
@misc{timeseries_inpainting_data_2025,
  title={Time Series Inpainting Experimental Dataset},
  author={Dariusz Kobiela and Jarosław Kobiela and Adam Kurowski and Agnieszka Landowska},
  year={2025},
  note={Dataset containing 270 experimental configurations across 3 industrial time series, 
        reconstructed using 31 methods}
}
```

---

## 📧 Support

### Questions?

1. Check file naming conventions above
2. Review experimental workflow
3. Consult related scripts documentation
4. Check project main README

### Issues?

- **Missing files:** Regenerate using experiment scripts
- **Corrupted data:** Re-run specific configurations
- **Disk space:** Use archives, remove intermediate data
- **Performance:** Use archived versions when possible

---

**Dataset Version:** 1.0  
**Last Updated:** 1.12.2025  
**Total Size:** ~30 GB (uncompressed), ~10.5 GB (compressed)  
**Status:** Complete ✅

