#!/usr/bin/env python3
"""
Fine-tuning Stable Diffusion 2 dla obrazów matematycznych (GAF, MTF, RP, SPEC)
z cross-validation i early stopping

Skrypt ładuje Stable Diffusion 2 z HuggingFace i dotrenowuje go na datasecie 
wszystkich typów obrazów matematycznych jednocześnie.
"""

import os
import sys
import json
import argparse
import logging
import warnings
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from diffusers import (
    StableDiffusionInpaintPipeline,
    UNet2DConditionModel,
    AutoencoderKL,
    DDPMScheduler,
)
from transformers import CLIPTextModel, CLIPTokenizer
from accelerate import Accelerator

# Wyłącz ostrzeżenia
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Ustawienia loggingu
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MathImageInpaintingDataset(Dataset):
    """Dataset dla inpaintingu obrazów matematycznych"""
    
    def __init__(self, 
                 data_dir: str,
                 indices: List[int],
                 image_types: List[str] = ["gaf", "mtf", "rp", "spec"],
                 image_size: int = 512):
        """
        Args:
            data_dir: ścieżka do folderu z danymi treningowymi
            indices: lista indeksów próbek do użycia
            image_types: typy obrazów do treningu
            image_size: rozmiar obrazów (512 dla Stable Diffusion 2)
        """
        self.data_dir = Path(data_dir)
        self.indices = indices
        self.image_types = image_types
        self.image_size = image_size
        
        # Ładuj metadane
        with open(self.data_dir / "dataset_summary.json", 'r') as f:
            self.metadata = json.load(f)
        
        # Stwórz listę wszystkich par obrazów dla podanych indeksów
        self.samples = []
        for idx in indices:
            sample_meta = self.metadata["samples"][idx]
            for img_type in image_types:
                self.samples.append({
                    "series_id": idx,
                    "image_type": img_type,
                    "original_path": self.data_dir / "original" / f"{idx:06d}_{img_type}.png",
                    "missing_path": self.data_dir / "missing" / f"{idx:06d}_{img_type}.png",
                    "metadata": sample_meta
                })
        
        logger.info(f"Dataset created with {len(self.samples)} image pairs")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Ładuj obrazy
        original_img = Image.open(sample["original_path"]).convert("RGB")
        missing_img = Image.open(sample["missing_path"]).convert("RGB")
        
        # Zmień rozmiar do 512x512 (dla Stable Diffusion 2)
        original_img = original_img.resize((self.image_size, self.image_size), Image.LANCZOS)
        missing_img = missing_img.resize((self.image_size, self.image_size), Image.LANCZOS)
        
        # Konwertuj na tensor [0, 1]
        original_tensor = torch.tensor(np.array(original_img)).permute(2, 0, 1).float() / 255.0
        missing_tensor = torch.tensor(np.array(missing_img)).permute(2, 0, 1).float() / 255.0
        
        # Stwórz maskę (obszary różniące się między original a missing)
        mask = torch.abs(original_tensor - missing_tensor).mean(dim=0, keepdim=True)
        mask = (mask > 0.01).float()  # Próg dla wykrycia różnic
        
        # Rozszerz maskę żeby była bardziej widoczna
        mask = torch.nn.functional.max_pool2d(mask.unsqueeze(0), kernel_size=3, stride=1, padding=1).squeeze(0)
        
        # Skaluj do [-1, 1] dla Stable Diffusion
        original_tensor = original_tensor * 2.0 - 1.0
        missing_tensor = missing_tensor * 2.0 - 1.0
        
        # Prompt specyficzny dla typu obrazu
        prompts = {
            "gaf": "high quality gramian angular field mathematical visualization",
            "mtf": "high quality markov transition field mathematical visualization", 
            "rp": "high quality recurrence plot mathematical visualization",
            "spec": "high quality spectrogram mathematical visualization"
        }
        prompt = prompts.get(sample["image_type"], "high quality mathematical visualization")
        
        return {
            "image": original_tensor,  # Target image
            "masked_image": missing_tensor,  # Input with missing parts
            "mask": mask,  # Inpainting mask
            "prompt": prompt
        }


class StableDiffusionTrainer:
    """Trainer dla fine-tuningu Stable Diffusion 2"""
    
    def __init__(self, 
                 model_id: str = "stabilityai/stable-diffusion-2-inpainting",
                 output_dir: str = "models/stable_diffusion_2_all",
                 mixed_precision: str = "fp16"):
        """
        Args:
            model_id: HuggingFace model ID
            output_dir: folder wyjściowy
            mixed_precision: tryb mixed precision (fp16/bf16/no)
        """
        self.model_id = model_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Accelerator dla optymalizacji
        self.accelerator = Accelerator(mixed_precision=mixed_precision)
        
        # Ładuj komponenty modelu
        self._load_model_components()
        
        # Optymalizacje pamięci
        self._setup_memory_optimizations()
        
    def _load_model_components(self):
        """Ładuje komponenty Stable Diffusion 2"""
        logger.info(f"Loading model components from {self.model_id}")
        
        # Tokenizer i text encoder
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.model_id, subfolder="tokenizer"
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.model_id, subfolder="text_encoder"
        )
        
        # VAE
        self.vae = AutoencoderKL.from_pretrained(
            self.model_id, subfolder="vae"
        )
        
        # UNet - to będzie trenowane
        self.unet = UNet2DConditionModel.from_pretrained(
            self.model_id, subfolder="unet"
        )
        
        # Scheduler
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            self.model_id, subfolder="scheduler"
        )
        
        # Zamroź wszystko oprócz UNet
        self.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(True)  # Tylko UNet będzie trenowany
        
        logger.info("Model components loaded successfully")
    
    def _setup_memory_optimizations(self):
        """Ustawia optymalizacje pamięci"""
        if torch.cuda.is_available():
            # Przenieś komponenty na GPU z optymalizacjami
            self.vae = self.vae.to(self.accelerator.device)
            self.text_encoder = self.text_encoder.to(self.accelerator.device) 
            self.unet = self.unet.to(self.accelerator.device)
            
            # Włącz attention slicing (dla diffusers 0.34.0+)
            try:
                # Nowa składnia dla diffusers 0.34.0+
                self.unet.set_attention_slice("auto")
                logger.info("✅ Attention slicing enabled")
            except Exception as e:
                try:
                    # Fallback dla starszych wersji
                    self.unet.enable_attention_slicing()
                    logger.info("✅ Attention slicing enabled (fallback)")
                except Exception as e2:
                    logger.warning(f"Could not enable attention slicing: {e}, fallback failed: {e2}")
            
            # Próbuj włączyć xformers (opcjonalne)
            try:
                self.unet.enable_xformers_memory_efficient_attention()
                logger.info("✅ XFormers memory efficient attention enabled")
            except Exception as e:
                logger.warning(f"XFormers not available (this is OK): {e}")
                logger.info("Continuing without XFormers - training will still work")
            
            # Gradient checkpointing dla oszczędności pamięci
            try:
                self.unet.enable_gradient_checkpointing()
                logger.info("✅ Gradient checkpointing enabled")
            except Exception as e:
                logger.warning(f"Could not enable gradient checkpointing: {e}")
            
            logger.info("✅ Memory optimizations setup completed")
    
    def encode_prompt(self, prompts: List[str]) -> torch.Tensor:
        """Koduje prompty tekstowe"""
        tokens = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            encoder_hidden_states = self.text_encoder(
                tokens.input_ids.to(self.accelerator.device)
            )[0]
        
        return encoder_hidden_states
    
    def compute_loss(self, batch):
        """Oblicza loss dla batcha"""
        device = self.accelerator.device
        
        # Przenieś batch na device
        images = batch["image"].to(device)
        masked_images = batch["masked_image"].to(device)
        masks = batch["mask"].to(device)
        prompts = batch["prompt"]
        
        # Encode prompts
        encoder_hidden_states = self.encode_prompt(prompts)
        
        # Encode images to latent space
        with torch.no_grad():
            latents = self.vae.encode(images).latent_dist.sample() * self.vae.config.scaling_factor
            masked_latents = self.vae.encode(masked_images).latent_dist.sample() * self.vae.config.scaling_factor
            
            # Resize mask to latent dimensions
            mask_latents = F.interpolate(masks, size=latents.shape[-2:], mode="nearest")
        
        # Sample noise
        noise = torch.randn_like(latents)
        batch_size = latents.shape[0]
        
        # Sample timesteps
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (batch_size,), device=device
        ).long()
        
        # Add noise to latents
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        
        # Concatenate masked latents and mask for inpainting
        latent_model_input = torch.cat([noisy_latents, mask_latents, masked_latents], dim=1)
        
        # Predict noise
        model_pred = self.unet(
            latent_model_input,
            timesteps,
            encoder_hidden_states=encoder_hidden_states
        ).sample
        
        # Compute loss
        loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
        
        return loss
    
    def train_epoch(self, dataloader, optimizer, scaler=None):
        """Trenuje jeden epoch"""
        self.unet.train()
        total_loss = 0
        
        progress_bar = tqdm(dataloader, desc="Training", leave=True, ncols=100)
        for step, batch in enumerate(progress_bar):
            optimizer.zero_grad()
            
            if scaler is not None:
                # Mixed precision training
                with autocast():
                    loss = self.compute_loss(batch)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Regular training
                loss = self.compute_loss(batch)
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.6f}", "avg_loss": f"{total_loss/(step+1):.6f}"})
        
        return total_loss / len(dataloader)
    
    def validate_epoch(self, dataloader):
        """Waliduje model"""
        self.unet.eval()
        total_loss = 0
        
        with torch.no_grad():
            for step, batch in enumerate(tqdm(dataloader, desc="Validation", leave=True, ncols=100)):
                loss = self.compute_loss(batch)
                total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def save_checkpoint(self, fold: int, epoch: int, train_loss: float, val_loss: float):
        """Zapisuje checkpoint"""
        checkpoint_dir = self.output_dir / f"fold_{fold}" / f"checkpoint-epoch-{epoch}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Zapisz UNet
        self.unet.save_pretrained(checkpoint_dir / "unet")
        
        # Zapisz metadane
        metadata = {
            "fold": fold,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "model_id": self.model_id
        }
        
        with open(checkpoint_dir / "training_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Checkpoint saved: {checkpoint_dir}")
        
        return checkpoint_dir
    
    def save_final_model(self, fold: int):
        """Zapisuje finalny model po treningu"""
        final_dir = self.output_dir / f"fold_{fold}" / f"checkpoint-fold_{fold}_final"
        final_dir.mkdir(parents=True, exist_ok=True)
        
        # Zapisz cały pipeline
        pipeline = StableDiffusionInpaintPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet,
            scheduler=self.noise_scheduler,
            safety_checker=None,
            feature_extractor=None
        )
        
        pipeline.save_pretrained(final_dir)
        logger.info(f"Final model saved: {final_dir}")
        
        return final_dir


class EarlyStopping:
    """Early stopping dla treningu"""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


def set_random_seeds(seed: int = 42):
    """Ustawia random seedy dla reprodukowalności"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Stable Diffusion 2 for math images")
    parser.add_argument("--data_dir", type=str, default="stdiff_training_data",
                        help="Path to training dataset")
    parser.add_argument("--output_dir", type=str, default="models/stable_diffusion_2_all_4",
                        help="Output directory for trained model")
    parser.add_argument("--max_samples", type=int, default=4000,
                        help="Number of training image pairs per fold (not total across all folds)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Training batch size (start with 1 for 22GB GPU)")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--max_epochs", type=int, default=300,
                        help="Maximum number of epochs")
    parser.add_argument("--n_folds", type=int, default=2,
                        help="Number of training runs (each with different random seed)")
    parser.add_argument("--train_ratio", type=float, default=0.75,
                        help="Ratio of data used for training (rest for validation, e.g. 0.8 = 80% train, 20% val)")
    parser.add_argument("--early_stop_patience", type=int, default=5,
                        help="Early stopping patience")
    parser.add_argument("--mixed_precision", type=str, default="no",
                        choices=["no", "fp16", "bf16"], help="Mixed precision training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--resume_fold", type=int, default=None,
                        help="Resume training from specific fold")
    
    args = parser.parse_args()
    
    # Ustawienia
    set_random_seeds(42)
    
    logger.info("🚀 STARTING STABLE DIFFUSION 2 FINE-TUNING")
    logger.info("=" * 60)
    logger.info(f"📁 Data directory: {args.data_dir}")
    logger.info(f"📁 Output directory: {args.output_dir}")
    logger.info(f"🎯 Max samples: {args.max_samples}")
    logger.info(f"📊 Batch size: {args.batch_size}")
    logger.info(f"🎓 Learning rate: {args.learning_rate}")
    logger.info(f"📈 Training runs: {args.n_folds}")
    logger.info(f"📊 Train/Val split: {args.train_ratio:.0%}/{1-args.train_ratio:.0%}")
    
    # Sprawdź dostępność CUDA
    if not torch.cuda.is_available():
        logger.error("CUDA is not available! This script requires GPU.")
        sys.exit(1)
    
    logger.info(f"🔧 GPU: {torch.cuda.get_device_name()}")
    logger.info(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Ładuj dataset metadata
    data_dir = Path(args.data_dir)
    with open(data_dir / "dataset_summary.json", 'r') as f:
        metadata = json.load(f)
    
    total_samples = metadata["total_samples"]
    logger.info(f"📚 Total samples in dataset: {total_samples}")
    
    # Określ które próbki użyć (ograniczone przez max_samples PER FOLD)
    # max_samples oznacza liczbę pairs do trenowania w każdym fold
    # Każda próbka ma 4 typy obrazów, więc max_samples // 4 = base samples per fold
    
    # Train/validation split ratio (konfigurowalne przez --train_ratio)
    train_ratio = args.train_ratio
    base_samples_per_fold = args.max_samples // 4
    
    # Oblicz łączną liczbę base samples potrzebną (teraz train_ratio to rzeczywisty % na training)
    total_base_samples_needed = int(base_samples_per_fold / train_ratio)
    max_base_samples = min(total_base_samples_needed, total_samples)
    
    # Walidacja czy mamy wystarczająco danych
    actual_train_pairs_per_fold = int(max_base_samples * train_ratio * 4)
    actual_val_pairs_per_fold = int(max_base_samples * (1 - train_ratio) * 4)
    
    if total_base_samples_needed > total_samples:
        logger.warning(f"⚠️ Dataset ma tylko {total_samples} base samples ({total_samples * 4} pairs)")
        logger.warning(f"⚠️ Dla {args.max_samples} pairs per fold potrzeba {total_base_samples_needed} base samples")
        logger.warning(f"⚠️ Zmniejszam do dostępnych danych: {actual_train_pairs_per_fold} pairs per fold")
    
    sample_indices = list(range(max_base_samples))
    logger.info(f"🎯 Target: {args.max_samples} training pairs per fold")
    logger.info(f"🎯 Using {max_base_samples} base samples = {max_base_samples * 4} total image pairs")
    logger.info(f"📊 Split ratio: {train_ratio:.0%} train / {1-train_ratio:.0%} validation")
    logger.info(f"📊 Actual per fold: {actual_train_pairs_per_fold} training pairs + {actual_val_pairs_per_fold} validation pairs")
    
    # Informacje o pamięci
    logger.info("\n🔧 MEMORY MANAGEMENT SETTINGS")
    logger.info(f"Mixed precision: {args.mixed_precision}")
    logger.info(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
    logger.info("Additional optimizations: attention slicing, gradient checkpointing")
    
    # Wyniki treningu
    fold_results = []
    
    # Training loop (możliwość wielokrotnego trenowania z różnymi seed'ami)
    for fold in range(args.n_folds):
        if args.resume_fold is not None and fold < args.resume_fold:
            logger.info(f"⏭️ Skipping fold {fold + 1} (resuming from fold {args.resume_fold + 1})")
            continue
            
        logger.info(f"\n🔄 TRAINING RUN {fold + 1}/{args.n_folds}")
        logger.info("=" * 40)
        
        # Podział na train/validation z custom ratio (np. 80/20)
        train_sample_indices, val_sample_indices = train_test_split(
            sample_indices, 
            train_size=train_ratio,
            random_state=42 + fold,  # Różne seed dla każdego run'a
            shuffle=True
        )
        
        logger.info(f"📊 Train samples: {len(train_sample_indices)} ({len(train_sample_indices) * 4} pairs)")
        logger.info(f"📊 Val samples: {len(val_sample_indices)} ({len(val_sample_indices) * 4} pairs)")
        
        # Stwórz datasets
        train_dataset = MathImageInpaintingDataset(
            data_dir=args.data_dir,
            indices=train_sample_indices
        )
        val_dataset = MathImageInpaintingDataset(
            data_dir=args.data_dir,
            indices=val_sample_indices
        )
        
        # Stwórz dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        # Stwórz trainer
        trainer = StableDiffusionTrainer(
            output_dir=args.output_dir,
            mixed_precision=args.mixed_precision
        )
        
        # Optimizer
        optimizer = optim.AdamW(
            trainer.unet.parameters(),
            lr=args.learning_rate,
            weight_decay=0.01
        )
        
        # Scaler dla mixed precision
        scaler = GradScaler() if args.mixed_precision == "fp16" else None
        
        # Early stopping
        early_stopping = EarlyStopping(patience=args.early_stop_patience)
        
        # Training loop dla tego fold
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        
        logger.info(f"🏃 Starting training for fold {fold + 1}")
        
        for epoch in range(args.max_epochs):
            print(f"\n📅 FOLD {fold + 1}/{args.n_folds} - EPOCH {epoch + 1}/{args.max_epochs}")
            print("-" * 60)
            
            # Trenuj epoch
            print(f"🏃 Training epoch {epoch + 1}...")
            train_loss = trainer.train_epoch(train_loader, optimizer, scaler)
            
            # Waliduj epoch  
            print(f"🔍 Validating epoch {epoch + 1}...")
            val_loss = trainer.validate_epoch(val_loader)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            print(f"📊 RESULTS - Epoch {epoch + 1}:")
            print(f"   📈 Train Loss: {train_loss:.6f}")
            print(f"   📉 Val Loss: {val_loss:.6f}")
            
            # Zapisz checkpoint jeśli val loss się poprawił
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                trainer.save_checkpoint(fold + 1, epoch + 1, train_loss, val_loss)
                logger.info(f"✅ New best validation loss: {val_loss:.6f}")
            
            # Early stopping check
            if early_stopping(val_loss):
                logger.info(f"🛑 Early stopping triggered at epoch {epoch + 1}")
                break
        
        # Zapisz final model dla tego fold
        final_model_path = trainer.save_final_model(fold + 1)
        
        # Zapisz wyniki fold
        fold_result = {
            "fold": fold + 1,
            "best_val_loss": best_val_loss,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "model_path": str(final_model_path)
        }
        fold_results.append(fold_result)
        
        logger.info(f"✅ Fold {fold + 1} completed. Best val loss: {best_val_loss:.6f}")
        
        # Wyczyść pamięć GPU
        torch.cuda.empty_cache()
    
    # Zapisz wyniki wszystkich folds
    results_file = Path(args.output_dir) / "cross_validation_results.json"
    with open(results_file, 'w') as f:
        json.dump(fold_results, f, indent=2)
    
    # Podsumowanie
    logger.info(f"\n🎉 CROSS-VALIDATION COMPLETED!")
    logger.info("=" * 50)
    
    val_losses = [result["best_val_loss"] for result in fold_results]
    mean_val_loss = np.mean(val_losses)
    std_val_loss = np.std(val_losses)
    
    logger.info(f"📊 Mean validation loss: {mean_val_loss:.6f} ± {std_val_loss:.6f}")
    logger.info(f"📁 Results saved to: {results_file}")
    logger.info(f"📁 Models saved in: {args.output_dir}")
    
    # Znajdź najlepszy fold
    best_fold_idx = np.argmin(val_losses)
    best_fold = fold_results[best_fold_idx]
    logger.info(f"🏆 Best fold: {best_fold['fold']} (val_loss: {best_fold['best_val_loss']:.6f})")
    logger.info(f"🏆 Best model: {best_fold['model_path']}")
    
    # Stwórz symboliczny link do najlepszego modelu  
    best_model_link = Path(args.output_dir) / "best_model"
    if best_model_link.exists():
        best_model_link.unlink()
    
    # Stwórz relatywną ścieżkę do najlepszego modelu
    relative_path = Path(best_fold['model_path']).relative_to(Path(args.output_dir))
    best_model_link.symlink_to(relative_path)
    logger.info(f"🔗 Best model linked as: {best_model_link} -> {relative_path}")
    
    print(f"\n🎯 NEXT STEPS:")
    print(f"📁 Models saved in: {args.output_dir}")
    print(f"🏆 Best model: {args.output_dir}/best_model")
    print(f"")
    print(f"🔗 To integrate with system, run:")
    print(f"   python integrate_custom_model.py --model_path {args.output_dir}/best_model --model_name sd2-all --test")


if __name__ == "__main__":
    main()
