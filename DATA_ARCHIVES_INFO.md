# 📦 Data Archives Information

Quick reference for all data archives in the project.

---

## 📚 Available Archives

### Main Data Archives (from `data/` folder)

| Archive | Original Size | Compressed | Files | Description |
|---------|---------------|------------|-------|-------------|
| `data_0_source.tar.gz` | 8.5 MB | 1.8 MB | 7 | Original time series |
| `data_1_missing.tar.gz` | 964 MB | 192 MB | 721 | Corrupted versions |
| `data_2_fixed.tar.gz` | 25 GB | 5.3 GB | 16,201 | Reconstructed data |
| `data_images.tar.gz` | 4 GB | 4.0 GB | 16,324 | Image transformations |

### Training Data Archive

| Archive | Original Size | Compressed | Files | Description |
|---------|---------------|------------|-------|-------------|
| `stdiff_training_data.tar.gz` | 1.2 GB | 1.2 GB | 18,000 | SD2 training images |

---

## 🎯 Quick Extract Commands

### Extract Specific Archive

```bash
# Source data (smallest)
tar -xzf data_0_source.tar.gz

# Missing data
tar -xzf data_1_missing.tar.gz

# Reconstructed data (largest - takes time!)
tar -xzf data_2_fixed.tar.gz

# Images
tar -xzf data_images.tar.gz

# Training data
tar -xzf stdiff_training_data.tar.gz
```

### Extract All

```bash
# Extract all archives at once
for archive in data_*.tar.gz stdiff_training_data.tar.gz; do
    echo "Extracting $archive..."
    tar -xzf "$archive"
done
```

### List Contents Without Extracting

```bash
# View what's inside
tar -tzf data_0_source.tar.gz | head -20

# Count files
tar -tzf data_2_fixed.tar.gz | wc -l
```

---

## 💾 Storage Requirements

### If You Need Everything

| Scenario | Disk Space Needed |
|----------|-------------------|
| **Archives only** | ~11 GB |
| **Extracted only** | ~30 GB |
| **Both** | ~41 GB |

### Minimal Setup

If you have limited space, extract only what you need:

```bash
# For analysis only (no images)
tar -xzf data_0_source.tar.gz      # 8.5 MB
tar -xzf data_2_fixed.tar.gz       # 25 GB
# Total: ~25 GB
```

```bash
# For visualization
tar -xzf data_0_source.tar.gz      # 8.5 MB
tar -xzf data_images.tar.gz        # 4 GB  
# Total: ~4 GB
```

```bash
# For training models
tar -xzf stdiff_training_data.tar.gz  # 1.2 GB
# Total: 1.2 GB
```

---

## 🔄 Regenerating Archives

If you need to recreate archives:

```bash
# From data/ directory
cd data/

# Create source archive
tar -czf ../data_0_source.tar.gz 0_source_data/

# Create missing data archive
tar -czf ../data_1_missing.tar.gz 1_missing_data/

# Create fixed data archive (takes time!)
tar -czf ../data_2_fixed.tar.gz 2_fixed_data/

# Create images archive
tar -czf ../data_images.tar.gz images_inpainting/
```

---

## 📊 Archive Details

### data_0_source.tar.gz

**Contents:** Original industrial time series
- 6 CSV files (boiler, pump, vibration, 3× water level)
- No missing values
- Ground truth for experiments

**When to extract:**
- Need original data for comparison
- Regenerating experiments
- Computing baseline metrics

---

### data_1_missing.tar.gz

**Contents:** Time series with injected missing values
- 721 CSV files
- 270 configs per dataset (3 datasets)
- MCAR/MAR/MNAR mechanisms
- 2%, 5%, 10% missing rates

**When to extract:**
- Analyzing missing data patterns
- Regenerating reconstructions
- Understanding experimental setup

---

### data_2_fixed.tar.gz ⚠️ LARGE

**Contents:** Reconstructed time series (all methods)
- 16,201 CSV files
- 31 methods × 270 configs
- Main experimental output

**When to extract:**
- Computing metrics
- Analyzing reconstruction quality
- Comparing methods

**Warning:** This is 25 GB! Make sure you have space.

---

### data_images.tar.gz

**Contents:** Image representations
- 16,324 PNG files
- GAF/MTF/RP/Spectrogram
- Original, missing, fixed, and difference images

**When to extract:**
- Visualizing transformations
- Debugging image-based methods
- Creating figures for papers

---

### stdiff_training_data.tar.gz

**Contents:** Stable Diffusion 2 training dataset
- 8,000 original images
- 8,000 corrupted images
- 2,000 metadata JSON files
- Synthetic time series patterns

**When to extract:**
- Training SD2 model
- Analyzing training data
- Creating custom datasets

---

## 🔍 Verification

### Check Archive Integrity

```bash
# Test archive integrity
tar -tzf data_0_source.tar.gz > /dev/null && echo "✅ OK" || echo "❌ Corrupted"

# Check all archives
for archive in data_*.tar.gz stdiff_training_data.tar.gz; do
    echo -n "Testing $archive... "
    tar -tzf "$archive" > /dev/null 2>&1 && echo "✅" || echo "❌"
done
```

### Verify File Counts After Extraction

```bash
# Expected counts
echo "Source data: $(ls data/0_source_data/*.csv 2>/dev/null | wc -l) (expected: 7)"
echo "Missing data: $(ls data/1_missing_data/*.csv 2>/dev/null | wc -l) (expected: 721)"
echo "Fixed data: $(ls data/2_fixed_data/*.csv 2>/dev/null | wc -l) (expected: 16201)"
echo "Images: $(find data/images_inpainting -type f 2>/dev/null | wc -l) (expected: 16324)"
```

---

## 📖 Documentation References

### Full Documentation

- **Data folder:** `data/README.md` - Complete dataset documentation
- **Training data:** `stdiff_training_data/README.md` - Training dataset details
- **Generation:** `GENERATE_TRAINING_DATASET_README.md` - How to create datasets

### Related Scripts

- `iterative_experiment.py` - Generates missing and fixed data
- `generate_training_dataset.py` - Generates training data
- `calculate_differences.py` - Computes metrics

---

## 💡 Tips

### Extract to Different Location

```bash
# Extract to specific directory
mkdir -p /path/to/extracted/
tar -xzf data_2_fixed.tar.gz -C /path/to/extracted/
```

### Extract Specific Files Only

```bash
# Extract only one file
tar -xzf data_2_fixed.tar.gz data/2_fixed_data/boiler_MCAR_2p_1_gafunet.csv

# Extract files matching pattern
tar -xzf data_2_fixed.tar.gz --wildcards 'data/2_fixed_data/boiler_MCAR_*'
```

### Streaming (No Disk Extraction)

```bash
# Process without extracting
tar -xzf data_0_source.tar.gz -O | grep "boiler"
```

---

## 🗑️ Cleanup

### Remove Extracted Data (Keep Archives)

```bash
# Remove extracted folders but keep archives
rm -rf data/0_source_data/
rm -rf data/1_missing_data/
rm -rf data/2_fixed_data/
rm -rf data/images_inpainting/
rm -rf stdiff_training_data/

# Verify archives still exist
ls -lh *.tar.gz
```

### Remove Archives (Keep Extracted)

```bash
# If you have extracted data and need space
rm data_0_source.tar.gz
rm data_1_missing.tar.gz
rm data_2_fixed.tar.gz
rm data_images.tar.gz
rm stdiff_training_data.tar.gz
```

---

**Last Updated:** 1.12.2025  
**Total Archive Size:** ~10.5 GB  
**Total Extracted Size:** ~30 GB
