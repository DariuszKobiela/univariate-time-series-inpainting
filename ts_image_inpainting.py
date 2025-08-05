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

# ---------------------------- ENCODERS -------------------------------------

def to_gaf(series):
    transformer = GramianAngularField(method="summation")
    X = transformer.fit_transform(series.values.reshape(1, -1))
    return X[0]

def to_mtf(series):
    transformer = MarkovTransitionField()
    X = transformer.fit_transform(series.values.reshape(1, -1))
    return X[0]

def to_rp(series):
    transformer = RecurrencePlot()
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
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        # Switched to a more recent inpainting model to resolve loading issues.
        model_id = "stabilityai/stable-diffusion-2-inpainting"
        try:
            self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if "cuda" in device else torch.float32,
            ).to(device)
            self.device = device
            print(f"U-Net Inpainter loaded on {self.device}")
            self.is_functional = True
        except Exception as e:
            print(f"Could not load U-Net model from Hugging Face. U-Net will be disabled.")
            print(f"Error: {e}")
            self.is_functional = False

    def __call__(self, image: np.ndarray, mask: pd.Series, enc_name: str) -> np.ndarray:
        if not self.is_functional:
            return image.copy()  # Passthrough if model failed to load

        # 1. Prepare the mask: create 2D numpy mask from 1D pandas mask
        mask_np = mask.values
        if enc_name in ["gaf", "mtf", "rp"]:
            patch_mask = np.logical_or.outer(mask_np, mask_np)
        else:  # spec
            patch_mask = np.tile(mask_np, (image.shape[0], 1))

        # 2. Convert inputs to PIL Images
        init_image_pil = Image.fromarray((image * 255).astype(np.uint8)).convert("RGB")
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
        
        # Enforce that unmasked areas remain unchanged
        inpainted_gray[~patch_mask] = image[~patch_mask]

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

def inverse_identity(image):
    """Placeholder inverse that flattens the main diagonal (for demo only)."""
    diag = np.diag(image)
    return pd.Series(diag)

def inverse_spectrogram(img):
    # img.shape == (freq_bins, time_bins)
    return pd.Series(img.mean(axis=0))

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