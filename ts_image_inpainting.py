import os
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from pyts.image import GramianAngularField, MarkovTransitionField, RecurrencePlot
from scipy.signal import spectrogram
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import psutil

# ---------------------------- ENCODERS -------------------------------------

def get_optimal_image_size(series_length):
    """Calculate optimal image size based on available memory"""
    try:
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        
        # Estimate memory needed for image processing
        # Each pixel needs ~4 bytes (float32) + overhead
        if available_gb > 8:  # 8+ GB available
            max_size = 1024
        elif available_gb > 4:  # 4+ GB available
            max_size = 512
        elif available_gb > 2:  # 2+ GB available
            max_size = 256
        else:  # Less than 2 GB
            max_size = 128
            
        # Don't exceed series length
        optimal_size = min(max_size, series_length)
        
        # Ensure it's a reasonable size for the model
        if optimal_size > 512:
            optimal_size = 512
        elif optimal_size < 64:
            optimal_size = 64
            
        return optimal_size
    except:
        return 256  # Fallback size

def to_gaf(series, max_size=None):
    """Convert series to GAF with automatic memory optimization"""
    if max_size is None:
        max_size = get_optimal_image_size(len(series))
    
    print(f"  📊 GAF: Converting {len(series)} points to {max_size}x{max_size} image")
    transformer = GramianAngularField(method="summation", image_size=max_size)
    X = transformer.fit_transform(series.values.reshape(1, -1))
    return X[0]

def to_mtf(series, max_size=None):
    """Convert series to MTF with automatic memory optimization"""
    if max_size is None:
        max_size = get_optimal_image_size(len(series))
    
    print(f"  📊 MTF: Converting {len(series)} points to {max_size}x{max_size} image")
    transformer = MarkovTransitionField(image_size=max_size)
    X = transformer.fit_transform(series.values.reshape(1, -1))
    return X[0]

def to_rp(series, max_size=None):
    """Convert series to Recurrence Plot with automatic memory optimization"""
    if max_size is None:
        max_size = get_optimal_image_size(len(series))
    
    print(f"  📊 RP: Converting {len(series)} points to {max_size}x{max_size} image")
    transformer = RecurrencePlot(image_size=max_size)
    X = transformer.fit_transform(series.values.reshape(1, -1))
    return X[0]

def to_spectrogram(series, window=32, target_size=None):
    """Return a square spectrogram (target_size x target_size).

    The raw spectrogram has shape (freq_bins, time_bins). We generate it with
    hop length = 1 (noverlap = window-1) to maximise the time resolution, then
    resample both axes with a bilinear zoom so the result is `target_size`×`target_size`.
    If `target_size` is not given, it defaults to the length of the input series.
    """
    if target_size is None:
        target_size = len(series)

    f, t, Sxx = spectrogram(
        series.values,
        fs=1,
        nperseg=window,
        noverlap=window - 1,
        mode="magnitude",
    )

    # scale to square using scipy.ndimage.zoom
    zoom_r = target_size / Sxx.shape[0]
    zoom_c = target_size / Sxx.shape[1]
    Sxx_sq = zoom(Sxx, (zoom_r, zoom_c), order=1)  # bilinear
    return Sxx_sq

# ---------------------------- INPAINTERS ------------------------------

class UnetInpainter:
    """
    An inpainter that uses a pretrained U-Net model from Hugging Face.
    Specifically, it uses the Stable Diffusion v1.5 inpainting pipeline,
    which has a U-Net backbone. It will be downloaded on the first run.
    """
    def __init__(self, device=None):
        # Check CUDA availability and set device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
                print(f"🎮 CUDA detected: {torch.cuda.get_device_name(0)}")
                print(f"🎮 CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            else:
                device = "cpu"
                print("⚠️ CUDA not available, using CPU")
        
        self.device = device
        print(f"🔧 Initializing U-Net Inpainter on device: {self.device}")
        
        # Switched to a more recent inpainting model to resolve loading issues.
        model_id = "stabilityai/stable-diffusion-2-inpainting"
        try:
            # Set torch dtype based on device
            torch_dtype = torch.float16 if "cuda" in device else torch.float32
            print(f"🔧 Loading model with dtype: {torch_dtype}")
            
            self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
            ).to(device)
            
            print(f"✅ U-Net Inpainter loaded successfully on {self.device}")
            self.is_functional = True
            
            # Print GPU memory info if using CUDA
            if "cuda" in device:
                print(f"🎮 GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
                print(f"🎮 GPU memory cached: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
                
        except Exception as e:
            print(f"❌ Could not load U-Net model from Hugging Face. U-Net will be disabled.")
            print(f"❌ Error: {e}")
            self.is_functional = False

    def __call__(self, image: np.ndarray, mask: pd.Series, enc_name: str) -> np.ndarray:
        if not self.is_functional:
            return image.copy()  # Passthrough if model failed to load

        # Check memory before processing
        memory_before = psutil.virtual_memory()
        print(f"  💾 Memory before {enc_name}-unet: {memory_before.percent:.1f}% ({memory_before.available / (1024**3):.1f} GB available)")
        
        # Check GPU memory if using CUDA
        if "cuda" in self.device:
            gpu_memory_before = torch.cuda.memory_allocated(0) / 1024**3
            print(f"  🎮 GPU memory before {enc_name}-unet: {gpu_memory_before:.2f} GB")
        
        # CRITICAL FIX: Always resize mask to match image dimensions
        print(f"  🔧 Original mask size: {len(mask)} points")
        print(f"  🔧 Target image size: {image.shape[0]}x{image.shape[1]}")
        
        # Resize mask to match image size (this is the key fix!)
        if len(mask) != image.shape[0]:
            print(f"  🔄 Resizing mask from {len(mask)} to {image.shape[0]} elements...")
            from scipy.ndimage import zoom
            zoom_factor = image.shape[0] / len(mask)
            mask_1d_resized = zoom(mask.values.astype(float), zoom_factor, order=0)
            mask_1d_resized = mask_1d_resized > 0.5  # Convert back to boolean
            print(f"  ✅ Mask resized to: {len(mask_1d_resized)} elements")
        else:
            mask_1d_resized = mask.values

        # Check if image is too large and resize if needed
        if image.shape[0] > 512 or image.shape[1] > 512:
            print(f"  ⚠️ Large image detected: {image.shape[0]}x{image.shape[1]}")
            print(f"  🔄 Resizing to 512x512 for memory efficiency...")
            from scipy.ndimage import zoom
            zoom_factor = 512 / max(image.shape)
            image_resized = zoom(image, zoom_factor, order=1)
            # Also resize the mask
            mask_1d_final = zoom(mask_1d_resized.astype(float), zoom_factor, order=0)
            mask_1d_final = mask_1d_final > 0.5  # Convert back to boolean
            print(f"  ✅ Image resized to: {image_resized.shape[0]}x{image_resized.shape[1]}")
            print(f"  ✅ Mask resized to: {len(mask_1d_final)} elements")
        else:
            image_resized = image
            mask_1d_final = mask_1d_resized

        print(f"  📐 Final sizes: Image {image_resized.shape}, Mask {len(mask_1d_final)} elements")

        # 1. Prepare the mask: create 2D numpy mask from 1D mask
        if enc_name in ["gaf", "mtf", "rp"]:
            print(f"  🎭 Creating 2D mask: {len(mask_1d_final)} x {len(mask_1d_final)} = {len(mask_1d_final)**2:,} pixels")
            patch_mask = np.logical_or.outer(mask_1d_final, mask_1d_final)
        else:  # spec
            print(f"  🎭 Creating spec mask: {image_resized.shape[0]} x {len(mask_1d_final)}")
            patch_mask = np.tile(mask_1d_final, (image_resized.shape[0], 1))

        # 2. Convert inputs to PIL Images
        init_image_pil = Image.fromarray((image_resized * 255).astype(np.uint8)).convert("RGB")
        mask_image_pil = Image.fromarray((patch_mask * 255).astype(np.uint8)).convert("RGB")

        # 3. Run the inpainting pipeline
        # Note: The model operates at a fixed resolution (e.g., 512x512),
        # so we must resize the output back to the original image size.
        original_size = (init_image_pil.width, init_image_pil.height)
        inpainted_result = self.pipe(
            prompt="high quality, professional, studio lighting",
            image=init_image_pil,
            mask_image=mask_image_pil,
            strength=0.99,
        ).images[0]

        # Resize back to original dimensions if necessary
        if inpainted_result.size != original_size:
            inpainted_result = inpainted_result.resize(original_size, Image.Resampling.LANCZOS)

        # 4. Convert output back to a numpy array and blend with original
        inpainted_np = np.array(inpainted_result).astype(np.float32) / 255.0
        inpainted_gray = np.mean(inpainted_np, axis=2)  # Convert RGB to grayscale
        
        # Resize back to original image size if we resized earlier
        if image.shape != image_resized.shape:
            from scipy.ndimage import zoom
            zoom_factor = image.shape[0] / inpainted_gray.shape[0]
            inpainted_gray = zoom(inpainted_gray, zoom_factor, order=1)
            patch_mask = zoom(patch_mask, zoom_factor, order=0)
            patch_mask = patch_mask > 0.5  # Convert back to boolean
        
        # Enforce that unmasked areas remain unchanged
        inpainted_gray[~patch_mask] = image[~patch_mask]

        # Check memory after processing
        memory_after = psutil.virtual_memory()
        print(f"  💾 Memory after {enc_name}-unet: {memory_after.percent:.1f}% ({memory_after.available / (1024**3):.1f} GB available)")
        memory_used = memory_before.available - memory_after.available
        print(f"  📊 Memory used: {memory_used / (1024**3):.2f} GB")
        
        # Check GPU memory after processing if using CUDA
        if "cuda" in self.device:
            gpu_memory_after = torch.cuda.memory_allocated(0) / 1024**3
            gpu_memory_used = gpu_memory_after - gpu_memory_before
            print(f"  🎮 GPU memory after {enc_name}-unet: {gpu_memory_after:.2f} GB")
            print(f"  🎮 GPU memory used: {gpu_memory_used:.2f} GB")

        return inpainted_gray

def passthrough_inpainter(image, mask, enc_name: str):
    """Placeholder that just returns the input image (no inpainting)."""
    return image.copy()

# Instantiate the U-Net model once
unet_model = UnetInpainter()

INPAINTERS = {
    "unet": unet_model,  # The __call__ method will be invoked
    "gated_conv": passthrough_inpainter,
    "ddpm": passthrough_inpainter,
    "ca_gan": passthrough_inpainter,
}

# --------------------------- INVERSE (stub) --------------------------------

def inverse_identity(image, original_length=None):
    """Convert image back to time series by flattening the main diagonal and interpolating to original length"""
    diag = np.diag(image)
    
    # If original_length is provided, interpolate to match it
    if original_length is not None and len(diag) != original_length:
        from scipy.interpolate import interp1d
        x_old = np.linspace(0, 1, len(diag))
        x_new = np.linspace(0, 1, original_length)
        f = interp1d(x_old, diag, kind='linear', bounds_error=False, fill_value='extrapolate')
        return pd.Series(f(x_new))
    
    return pd.Series(diag)

def inverse_spectrogram(img, original_length=None):
    # img.shape == (freq_bins, time_bins)
    series = pd.Series(img.mean(axis=0))
    
    # If original_length is provided, interpolate to match it
    if original_length is not None and len(series) != original_length:
        from scipy.interpolate import interp1d
        x_old = np.linspace(0, 1, len(series))
        x_new = np.linspace(0, 1, original_length)
        f = interp1d(x_old, series.values, kind='linear', bounds_error=False, fill_value='extrapolate')
        return pd.Series(f(x_new))
    
    return series

INVERTERS = {
    "gaf": inverse_identity,
    "mtf": inverse_identity,
    "rp": inverse_identity,
    "spec": inverse_spectrogram,
}

ENCODERS = {
    "gaf": to_gaf,
    "mtf": to_mtf,
    "rp": to_rp,
    "spec": to_spectrogram,
}

# --------------------------- PIPELINE --------------------------------------

def save_image(img, path):
    plt.imsave(path, img, cmap="viridis")

def process_series(series: pd.Series, out_dir: Path, dataset_name: str, selected_inpaint_names: list):
    out_dir.mkdir(parents=True, exist_ok=True)
    mask = series.isna()
    series_filled = series.fillna(0)  # zeros for encoding

    selected_inp_fns = {name: INPAINTERS[name] for name in selected_inpaint_names if name in INPAINTERS}
    if not selected_inp_fns:
        print("None of the selected inpainters are available. Exiting.")
        return

    for enc_name, encoder in ENCODERS.items():
        img = encoder(series_filled)
        # save damaged image
        img_path = f"data/2_degraded/{dataset_name}_{enc_name}_damaged.png"
        print(f"Processing {img_path}")
        save_image(img, img_path)

        for inp_name, inp_fn in selected_inp_fns.items():
            inpainted = inp_fn(img, mask, enc_name)
            
            # Construct unique filenames that include the dataset, encoder, and inpainter names
            base_filename = f"{dataset_name}-fixed-{enc_name}-{inp_name}"

            # Save the repaired image directly to the output directory
            save_image(inpainted, out_dir / f"{base_filename}.png")

            # Create and save the difference image
            diff_img = np.abs(inpainted - img)
            save_image(diff_img, out_dir / f"{base_filename}_diff.png")

            # inverse transform (stub)
            inv_fn = INVERTERS[enc_name]
            recon_series = inv_fn(inpainted)
            # align length
            recon_series.index = series.index[: len(recon_series)]
            merged = series.copy()
            merged[mask] = recon_series[mask]

            # save only predicted values (NaNs elsewhere)
            recon_out = series.copy()
            recon_out[:] = np.nan
            recon_out[mask] = recon_series[mask]

            recon_out.to_csv(out_dir / f"{base_filename}_predicted_values.csv", index_label="year")

            # save merged final series
            merged.to_csv(out_dir / f"{base_filename}.csv", index_label="year")


# --------------------------- CLI -------------------------------------------

def main():
    """Main function to run the script from the command line."""
    parser = argparse.ArgumentParser(description="TS image inpainting scaffold")
    parser.add_argument("--input", nargs='+', help="One or more CSV file paths of the degraded data.")
    parser.add_argument("--column", default="deaths_rate_per_100k", help="Name of the time series column in the CSV")
    parser.add_argument("--output", default="data/3_repaired", help="Directory to save the repaired data and images")
    
    inpainter_choices = list(INPAINTERS.keys())
    parser.add_argument(
        "--inpainters",
        nargs="+",
        choices=inpainter_choices + ["all"],
        help=f"Select inpainting models to run. Choices: {inpainter_choices}. Use 'all' to run all."
    )

    args = parser.parse_args()

    selected_inpaint_names = args.inpainters
    if not selected_inpaint_names:
        print("No inpainting model selected. Please use the --inpainters argument (e.g., --inpainters unet).")
        print(f"Available models: {list(INPAINTERS.keys())}")
        return
        
    if "all" in selected_inpaint_names:
        selected_inpaint_names = list(INPAINTERS.keys())
    
    selected_inpaint_names = sorted(list(set(selected_inpaint_names)))
    print(f"Selected inpainters: {selected_inpaint_names}")

    if not args.input:
        print("No input file(s) specified. Please use the --input argument.")
        return

    out_dir = Path(args.output)
    for input_path in args.input:
        print(f"\n--- Processing file: {input_path} ---")
        try:
            df = pd.read_csv(input_path, index_col=0)
            series = df[args.column]
            dataset_name = Path(input_path).stem
            process_series(series, out_dir, dataset_name, selected_inpaint_names)
        except FileNotFoundError:
            print(f"  Error: File not found at '{input_path}'. Skipping.")
        except KeyError:
            print(f"  Error: Column '{args.column}' not found in '{input_path}'. Skipping.")
        except Exception as e:
            print(f"  An unexpected error occurred while processing '{input_path}': {e}. Skipping.")


if __name__ == "__main__":
    main() 