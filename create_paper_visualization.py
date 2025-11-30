#!/usr/bin/env python3
"""
Create visualization for paper: 
- Time series lineplot (original, missing, 4 reconstructed)
- Image grids for each transformation (GAF, MTF, RP, Spectrogram)
"""

print("🚀 Starting script...")
import pandas as pd
print("   ✓ pandas imported")
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
print("   ✓ matplotlib imported")
import numpy as np
print("   ✓ numpy imported")
from PIL import Image
print("   ✓ PIL imported")
import os
print("   ✓ os imported")

# Configuration
DATASET = "vibr"
MISSINGNESS_TYPE = "MCAR"
MISSING_RATE = "50p"
ITERATION = "1"

# File paths
BASE_NAME = f"{DATASET}_{MISSINGNESS_TYPE}_{MISSING_RATE}_{ITERATION}"
ORIGINAL_FILE = f"data/0_source_data/vibration_sensor_S1.csv"
MISSING_FILE = f"data/1_missing_data/{BASE_NAME}.csv"

# Reconstructed files (one per transformation using sd2all4 model)
RECONSTRUCTED_FILES = {
    'GAF': f"data/2_fixed_data/{BASE_NAME}_gafsd2all4.csv",
    'MTF': f"data/2_fixed_data/{BASE_NAME}_mtfsd2all4.csv",
    'RP': f"data/2_fixed_data/{BASE_NAME}_rpsd2all4.csv",
    'SPEC': f"data/2_fixed_data/{BASE_NAME}_specsd2all4.csv",
}

# Image paths
TRANSFORMS = ['gaf', 'mtf', 'rp', 'spec']
TRANSFORM_NAMES = {
    'gaf': 'GAF',
    'mtf': 'MTF',
    'rp': 'RP',
    'spec': 'Spectrogram'
}

OUTPUT_DIR = "results/paper_visualization"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_timeseries_with_timestamps(filepath):
    """Load time series from CSV file with timestamps"""
    df = pd.read_csv(filepath)
    # Convert timestamp to datetime
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    # Sort by time
    df = df.sort_values("Timestamp")
    # Return timestamps and values
    return df["Timestamp"].values, df["Vibration (Hz)"].values

def create_single_timeseries_plot(time, values, title, color, filename):
    """Create a single time series plot"""
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # Plot
    ax.plot(time, values, color=color, linewidth=1.5)
    
    # Styling
    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('Vibration (Hz)', fontsize=11)
    ax.set_title(title, fontsize=12)
    
    # Grid
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, color='gray')
    ax.set_axisbelow(True)
    
    # Save
    output_path = f"{OUTPUT_DIR}/{filename}"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

def create_combined_reconstructions_plot(time_orig, original, reconstructed_data, colors):
    """Create a plot overlaying all 4 reconstruction methods with original as black baseline"""
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # First plot original as black baseline
    ax.plot(time_orig, original, color='black', linewidth=2, label='Original', alpha=0.7, zorder=5)
    
    # Then plot each reconstructed series on top
    for name, (time, values) in reconstructed_data.items():
        ax.plot(time, values, color=colors[name], linewidth=1.5, label=f'{name} (Reconstructed)', alpha=0.8, zorder=10)
    
    # Styling
    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('Vibration (Hz)', fontsize=11)
    ax.set_title('Vibration Sensor S1 - Comparison of Reconstruction Methods', fontsize=12)
    ax.legend(loc='upper right', fontsize=9, frameon=True, framealpha=0.95, ncol=2)
    
    # Grid
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, color='gray')
    ax.set_axisbelow(True)
    
    # Save
    output_path = f"{OUTPUT_DIR}/timeseries_reconstructed_comparison.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

def create_timeseries_plots():
    """Create separate plots for original, missing, and each reconstructed series"""
    print("🎨 Creating time series plots...")
    
    # Load data with timestamps
    time_orig, original = load_timeseries_with_timestamps(ORIGINAL_FILE)
    time_missing, missing = load_timeseries_with_timestamps(MISSING_FILE)
    
    # Original is longer - cut last 10 points
    print(f"   📏 Original length: {len(original)} → {len(original) - 10} (cutting last 10 points)")
    original = original[:-10]
    time_orig = time_orig[:-10]
    
    print(f"   📏 Missing length: {len(missing)}")
    
    # For visualization, downsample if series is too long (>10k points)
    downsample_factor = 1
    if len(original) > 10000:
        downsample_factor = len(original) // 5000  # Keep ~5000 points
        print(f"   📉 Downsampling for visualization: every {downsample_factor} points")
        original = original[::downsample_factor]
        missing = missing[::downsample_factor]
        time_orig = time_orig[::downsample_factor]
        time_missing = time_missing[::downsample_factor]
        print(f"   📏 New length: {len(original)} points")
    
    # 1. Original plot
    print(f"   📊 Creating Original plot...")
    path = create_single_timeseries_plot(
        time_orig, original, 
        'Vibration Sensor S1 - Original Time Series',
        '#F5A623', 'timeseries_original.png'
    )
    print(f"      ✅ Saved: {path}")
    
    # 2. Missing plot
    print(f"   📊 Creating Missing plot...")
    path = create_single_timeseries_plot(
        time_missing, missing, 
        'Vibration Sensor S1 - Missing Data Time Series',
        '#1976D2', 'timeseries_missing.png'
    )
    print(f"      ✅ Saved: {path}")
    
    # 3-6. Reconstructed plots for each transformation
    colors = {
        'GAF': '#2E7D32',  # Green
        'MTF': '#E74C3C',  # Red
        'RP': '#9C27B0',   # Purple
        'SPEC': '#FF6F00'  # Orange
    }
    
    reconstructed_data = {}
    
    for name, filepath in RECONSTRUCTED_FILES.items():
        print(f"   📊 Creating Reconstructed ({name}) plot...")
        time_recon, reconstructed = load_timeseries_with_timestamps(filepath)
        
        if len(reconstructed) > 10000 and downsample_factor > 1:
            reconstructed = reconstructed[::downsample_factor]
            time_recon = time_recon[::downsample_factor]
        
        # Store for combined plot
        reconstructed_data[name] = (time_recon, reconstructed)
        
        path = create_single_timeseries_plot(
            time_recon, reconstructed, 
            f'Vibration Sensor S1 - Reconstructed ({name}) Time Series',
            colors[name], f'timeseries_reconstructed_{name.lower()}.png'
        )
        print(f"      ✅ Saved: {path}")
    
    # 7. Combined reconstruction comparison plot
    print(f"   📊 Creating Combined Reconstructions Comparison plot...")
    path = create_combined_reconstructions_plot(time_orig, original, reconstructed_data, colors)
    print(f"      ✅ Saved: {path}")

def save_single_image(img, title, filename):
    """Save a single image with title"""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    ax.axis('off')
    
    output_path = f"{OUTPUT_DIR}/{filename}"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

def create_image_grid(transform_short):
    """Create 3x1 grid: original, missing, reconstructed for one transformation + save each individually"""
    transform_name = TRANSFORM_NAMES[transform_short]
    print(f"🎨 Creating image grid for {transform_name}...")
    
    # Image paths
    original_img_path = f"data/images_inpainting/0_original_images/{DATASET}_image_{transform_short}.png"
    missing_img_path = f"data/images_inpainting/1_missing_images/{BASE_NAME}_imagemissing_{transform_short}.png"
    fixed_img_path = f"data/images_inpainting/2_fixed_images/{BASE_NAME}_imagefixed_{transform_short}_sd2-all4.png"
    
    # Check if files exist
    for path in [original_img_path, missing_img_path, fixed_img_path]:
        if not os.path.exists(path):
            print(f"   ⚠️  Warning: {path} not found!")
            return
    
    # Load images
    img_original = Image.open(original_img_path)
    img_missing = Image.open(missing_img_path)
    img_fixed = Image.open(fixed_img_path)
    
    # Save individual images
    print(f"   📸 Saving individual images...")
    path = save_single_image(img_original, f'{transform_name} - Original', 
                            f'image_{transform_short}_original.png')
    print(f"      ✅ {path}")
    
    path = save_single_image(img_missing, f'{transform_name} - Missing', 
                            f'image_{transform_short}_missing.png')
    print(f"      ✅ {path}")
    
    path = save_single_image(img_fixed, f'{transform_name} - Reconstructed', 
                            f'image_{transform_short}_reconstructed.png')
    print(f"      ✅ {path}")
    
    # Create grid with 1 row, 3 columns
    print(f"   🖼️  Creating grid...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot images
    images = [img_original, img_missing, img_fixed]
    titles = ['Original', 'Missing', 'Reconstructed\n(Inpainting)']
    
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
        ax.axis('off')
    
    # Main title
    fig.suptitle(f'{transform_name} Transformation', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Save grid
    output_path = f"{OUTPUT_DIR}/images_{transform_short}_grid.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Saved grid: {output_path}")

def main():
    """Main function"""
    print("="*80)
    print("📊 CREATING PAPER VISUALIZATION")
    print("="*80)
    print(f"Dataset: {DATASET} (vibration_sensor_S1.csv - 210 points)")
    print(f"Missingness: {MISSINGNESS_TYPE}, Rate: {MISSING_RATE}, Iteration: {ITERATION}")
    print()
    
    # Create time series plots (separate for each)
    create_timeseries_plots()
    print()
    
    # Create image grids for each transformation
    for transform in TRANSFORMS:
        create_image_grid(transform)
    
    print()
    print("="*80)
    print("✅ ALL VISUALIZATIONS CREATED!")
    print("="*80)
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print()
    print("Generated files:")
    print(f"  📊 Time series plots:")
    print(f"     1. {OUTPUT_DIR}/timeseries_original.png")
    print(f"     2. {OUTPUT_DIR}/timeseries_missing.png")
    print(f"     3. {OUTPUT_DIR}/timeseries_reconstructed_gaf.png")
    print(f"     4. {OUTPUT_DIR}/timeseries_reconstructed_mtf.png")
    print(f"     5. {OUTPUT_DIR}/timeseries_reconstructed_rp.png")
    print(f"     6. {OUTPUT_DIR}/timeseries_reconstructed_spec.png")
    print(f"     7. {OUTPUT_DIR}/timeseries_reconstructed_comparison.png (4 methods overlay)")
    print(f"  🖼️  Transformation images:")
    for transform in TRANSFORMS:
        print(f"     {TRANSFORM_NAMES[transform]}:")
        print(f"       - {OUTPUT_DIR}/image_{transform}_original.png")
        print(f"       - {OUTPUT_DIR}/image_{transform}_missing.png")
        print(f"       - {OUTPUT_DIR}/image_{transform}_reconstructed.png")
        print(f"       - {OUTPUT_DIR}/images_{transform}_grid.png (combined)")

if __name__ == "__main__":
    main()

