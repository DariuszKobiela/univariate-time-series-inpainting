"""
Custom Stable Diffusion 2 Inpainter for Mathematical Images

Ten model wykorzystuje fine-tuned Stable Diffusion 2 specjalnie wytrenowany
na obrazach GAF, MTF, RP i Spec do naprawiania szeregów czasowych.
"""

import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class StableDiffusion2MathInpainter:
    """
    Inpainter wykorzystujący Stable Diffusion 2 fine-tuned na obrazach matematycznych
    """
    
    def __init__(self, model_path: str = None, device: str = None, is_multi_type: bool = None):
        """
        Args:
            model_path: ścieżka do fine-tuned modelu (jeśli None, użyje base model)
            device: device do użycia (cuda/cpu)
            is_multi_type: czy model jest trenowany na wszystkich typach obrazów (auto-detect jeśli None)
        """
        self.model_path = model_path
        
        # Ustaw device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Auto-detect multi-type model na podstawie ścieżki
        if is_multi_type is None:
            self.is_multi_type = (
                model_path and 
                ("stable_diffusion_2_all" in str(model_path) or 
                 "sd2-all" in str(model_path) or
                 "multi" in str(model_path).lower())
            )
        else:
            self.is_multi_type = is_multi_type
            
        # Mapowanie typów obrazów na prompty
        self.prompts = {
            "gaf": "high quality gramian angular field time series visualization, mathematical representation, clean and precise",
            "mtf": "high quality markov transition field time series visualization, mathematical representation, clean and precise", 
            "rp": "high quality recurrence plot time series visualization, mathematical representation, clean and precise",
            "spec": "high quality spectrogram time series visualization, mathematical representation, clean and precise"
        }
        
        # Universal prompt for multi-type model
        self.universal_prompt = "high quality mathematical time series visualization, scientific data representation, clean and precise"
        
        self.pipeline = None
        self._load_model()
    
    def _load_model(self):
        """Ładuje model Stable Diffusion"""
        try:
            if self.model_path and os.path.exists(self.model_path):
                # Sprawdź czy to jest model z cross-validation
                if os.path.isdir(self.model_path):
                    # Sprawdź czy istnieje best_model link (nowy system)
                    best_model_link = os.path.join(self.model_path, "best_model")
                    if os.path.exists(best_model_link):
                        logger.info(f"Loading best cross-validation model from: {best_model_link}")
                        model_path = best_model_link
                    else:
                        # Legacy: Sprawdź czy istnieje checkpoint-final
                        final_checkpoint = os.path.join(self.model_path, "checkpoint-final")
                        if os.path.exists(final_checkpoint):
                            logger.info(f"Loading CV-trained model from: {final_checkpoint}")
                            model_path = final_checkpoint
                        else:
                            # Legacy: Sprawdź czy istnieją foldery fold_*
                            fold_dirs = [d for d in os.listdir(self.model_path) if d.startswith('fold_')]
                            if fold_dirs:
                                # Załaduj najlepszy fold (domyślnie fold_1)
                                best_fold = os.path.join(self.model_path, "fold_1", "checkpoint-fold_1_final")
                                if os.path.exists(best_fold):
                                    logger.info(f"Loading best fold model from: {best_fold}")
                                    model_path = best_fold
                                else:
                                    logger.info(f"Loading CV model from: {self.model_path}")
                                    model_path = self.model_path
                            else:
                                logger.info(f"Loading fine-tuned model from: {self.model_path}")
                                model_path = self.model_path
                else:
                    # To jest plik, nie folder
                    logger.info(f"Loading fine-tuned model from: {self.model_path}")
                    model_path = self.model_path
            else:
                # Fallback do base model
                logger.info("Loading base Stable Diffusion 2 inpainting model")
                model_path = "stabilityai/stable-diffusion-2-inpainting"
            
            # Ustaw torch dtype na podstawie device
            torch_dtype = torch.float16 if "cuda" in self.device else torch.float32
            
            self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                use_safetensors=True if "cuda" in self.device else False,
                safety_checker=None,
                feature_extractor=None,
            ).to(self.device)
            
            # Optymalizacje pamięci
            if "cuda" in self.device:
                # Attention slicing (dla diffusers 0.34.0+)
                try:
                    # Nowa składnia dla diffusers 0.34.0+
                    self.pipeline.enable_attention_slicing("auto")
                    logger.info("✅ Attention slicing enabled")
                except Exception as e:
                    try:
                        # Fallback dla starszych wersji
                        self.pipeline.enable_attention_slicing()
                        logger.info("✅ Attention slicing enabled (fallback)")
                    except Exception as e2:
                        logger.warning(f"Could not enable attention slicing: {e}, fallback failed: {e2}")
                
                # XFormers (opcjonalne)
                try:
                    self.pipeline.enable_xformers_memory_efficient_attention()
                    logger.info("✅ XFormers memory optimization enabled")
                except Exception as e:
                    logger.warning(f"XFormers not available (this is OK): {e}")
                    
                logger.info("✅ Memory optimizations setup completed")
            
            logger.info(f"✅ Model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise
    
    def inpaint(self, image: np.ndarray, mask, image_type: str = "gaf") -> np.ndarray:
        """
        Wykonuje inpainting na obrazie matematycznym
        
        Args:
            image: obraz wejściowy jako numpy array [H, W] lub [H, W, C]
            mask: maska jako numpy array [H, W] lub pandas Series, True = obszar do naprawy
            image_type: typ obrazu ("gaf", "mtf", "rp", "spec")
            
        Returns:
            naprawiony obraz jako numpy array [H, W] lub [H, W, C]
        """
        try:
            # Konwertuj na PIL Images
            pil_image = self._numpy_to_pil(image)
            pil_mask = self._mask_to_pil(mask)
            
            # Wybierz odpowiedni prompt (użyj universal prompt dla multi-type model)
            if hasattr(self, 'is_multi_type') and self.is_multi_type:
                prompt = self.universal_prompt
            else:
                prompt = self.prompts.get(image_type, self.prompts["gaf"])
            
            # Wykonaj inpainting
            result = self.pipeline(
                prompt=prompt,
                image=pil_image,
                mask_image=pil_mask,
                num_inference_steps=50,
                guidance_scale=7.5,
                strength=0.8,
                generator=torch.manual_seed(42)  # Dla reprodukowalności
            ).images[0]
            
            # Konwertuj z powrotem na numpy
            result_array = self._pil_to_numpy(result, original_shape=image.shape)
            
            return result_array
            
        except Exception as e:
            logger.error(f"Inpainting failed: {e}")
            # Fallback - zwróć oryginalny obraz
            return image.copy()
    
    def _numpy_to_pil(self, image: np.ndarray) -> Image.Image:
        """Konwertuje numpy array na PIL Image"""
        # Normalizuj do [0, 1] jeśli potrzeba
        if image.max() <= 1.0:
            image = image * 255
        
        # Ogranicz do [0, 255]
        image = np.clip(image, 0, 255).astype(np.uint8)
        
        # Obsłuż różne formaty
        if len(image.shape) == 2:
            # Grayscale -> RGB
            image = np.stack([image, image, image], axis=-1)
        elif len(image.shape) == 3 and image.shape[2] == 1:
            # Single channel -> RGB
            image = np.repeat(image, 3, axis=2)
        
        return Image.fromarray(image, mode='RGB')
    
    def _mask_to_pil(self, mask) -> Image.Image:
        """Konwertuje maskę na PIL Image"""
        # Mask: True = obszar do inpainting, False = obszar zachowany
        # Stable Diffusion: white (255) = inpaint, black (0) = keep
        
        # Handle pandas Series
        if hasattr(mask, 'values'):
            mask = mask.values
        
        # Convert to numpy array if not already
        mask = np.array(mask)
        
        if mask.dtype == bool:
            mask_array = mask.astype(np.uint8) * 255
        else:
            # Zakładamy że mask jest już w formacie 0-1 lub 0-255
            if mask.max() <= 1.0:
                mask_array = (mask * 255).astype(np.uint8)
            else:
                mask_array = mask.astype(np.uint8)
        
        return Image.fromarray(mask_array, mode='L')
    
    def _pil_to_numpy(self, image: Image.Image, original_shape: tuple) -> np.ndarray:
        """Konwertuje PIL Image z powrotem na numpy array"""
        # Konwertuj na numpy
        result_array = np.array(image)
        
        # Przywróć oryginalny format jeśli był grayscale
        if len(original_shape) == 2:
            # Konwertuj RGB na grayscale
            if len(result_array.shape) == 3:
                result_array = np.mean(result_array, axis=2)
        elif len(original_shape) == 3 and original_shape[2] == 1:
            # Konwertuj RGB na single channel
            if len(result_array.shape) == 3:
                result_array = np.mean(result_array, axis=2, keepdims=True)
        
        # Normalizuj do [0, 1] jeśli oryginalny obraz był w tym zakresie
        original_max = 1.0 if np.all(np.array(original_shape) <= 1.0) else 255.0
        if original_max == 1.0:
            result_array = result_array / 255.0
        
        return result_array.astype(np.float32)
    
    def __call__(self, image: np.ndarray, mask, enc_name: str = "gaf") -> np.ndarray:
        """Callable interface dla kompatybilności z istniejącym systemem"""
        return self.inpaint(image, mask, enc_name)


# Funkcja pomocnicza do trenowania modelu
def train_custom_model(dataset_dir: str = "stdiff_training_data", 
                      output_dir: str = "models/stable_diffusion_2_all",
                      max_samples: int = 8000,
                      batch_size: int = 1,
                      max_epochs: int = 100,
                      n_folds: int = 10,
                      early_stop_patience: int = 5,
                      learning_rate: float = 1e-5):
    """
    Trenuje custom model Stable Diffusion 2 na wszystkich typach obrazów matematycznych
    
    Args:
        dataset_dir: ścieżka do wygenerowanego datasetu treningowego
        output_dir: folder wyjściowy dla modelu
        max_samples: maksymalna liczba próbek do treningu
        batch_size: rozmiar batcha
        max_epochs: maksymalna liczba epok
        n_folds: liczba folds dla cross-validation
        early_stop_patience: patience dla early stopping
        learning_rate: learning rate
    """
    import subprocess
    import sys
    
    cmd = [
        sys.executable, "finetune_stable_diffusion.py",
        "--data_dir", dataset_dir,
        "--output_dir", output_dir,
        "--max_samples", str(max_samples),
        "--batch_size", str(batch_size),
        "--max_epochs", str(max_epochs),
        "--n_folds", str(n_folds),
        "--early_stop_patience", str(early_stop_patience),
        "--learning_rate", str(learning_rate)
    ]
    
    print(f"🚀 Starting cross-validation training with command:")
    print(f"   {' '.join(cmd)}")
    print(f"📊 Training on all image types: GAF, MTF, RP, SPEC")
    print(f"🔄 Cross-validation: {n_folds} folds")
    print(f"🛑 Early stopping: {early_stop_patience} epochs patience")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Cross-validation training completed successfully!")
        print(f"📁 Models saved to: {output_dir}")
        print(f"🏆 Best model available at: {output_dir}/best_model")
        return f"{output_dir}/best_model"
    else:
        print(f"❌ Training failed: {result.stderr}")
        if result.stdout:
            print(f"📝 Training output: {result.stdout}")
        return None


# Przykład użycia
if __name__ == "__main__":
    # Test funkcjonalności
    print("🧪 Testing Stable Diffusion 2 Math Inpainter")
    
    # Stwórz testowy obraz (symulacja GAF)
    test_image = np.random.random((64, 64)).astype(np.float32)
    test_mask = np.zeros((64, 64), dtype=bool)
    test_mask[20:40, 20:40] = True  # Prostokątny obszar do inpainting
    
    # Test inpaintera
    try:
        # Test z base model
        print("🧪 Testing with base Stable Diffusion 2 model...")
        inpainter = StableDiffusion2MathInpainter()
        result = inpainter(test_image, test_mask, "gaf")
        print(f"✅ Base model test passed! Result shape: {result.shape}")
        
        # Test multi-type detection
        print("🧪 Testing multi-type model detection...")
        multi_inpainter = StableDiffusion2MathInpainter(
            model_path="models/stable_diffusion_2_all/best_model",
            is_multi_type=True
        )
        print(f"✅ Multi-type model: {multi_inpainter.is_multi_type}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("Make sure you have the required dependencies installed:")
        print("pip install diffusers transformers accelerate torch torchvision")
        print("\nFor training, additionally install:")
        print("pip install sklearn tqdm matplotlib accelerate")