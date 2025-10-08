#!/usr/bin/env python3
"""
Generator bazy danych obrazów matematycznych do fine-tuningu Stable Diffusion 2

Ten skrypt generuje różnorodne szeregi czasowe i konwertuje je na obrazy
GAF, MTF, RP i Spec z różnymi wzorami brakujących danych.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import random
from typing import Tuple, List
import argparse
from tqdm import tqdm

# Import our existing encoders
from ts_image_inpainting import to_gaf, to_mtf, to_rp, to_spectrogram, save_image

class TimeSeriesGenerator:
    """Generator różnorodnych szeregów czasowych"""
    
    def __init__(self, min_length=100, max_length=1000):
        self.min_length = min_length
        self.max_length = max_length
        
    def generate_synthetic_series(self, length: int = None, pattern_type: str = "mixed") -> np.ndarray:
        """
        Generuje syntetyczny szereg czasowy o zadanym wzorcu
        
        Args:
            length: długość szeregu (jeśli None, losowa)
            pattern_type: typ wzorca ("sine", "trend", "seasonal", "noise", "mixed", etc.)
        """
        if length is None:
            length = random.randint(self.min_length, self.max_length)
            
        t = np.linspace(0, 10, length)
        
        if pattern_type == "sine":
            # Proste fale sinusoidalne z losowymi parametrami
            freq = random.uniform(0.5, 3.0)
            amplitude = random.uniform(0.5, 2.0)
            phase = random.uniform(0, 2*np.pi)
            series = amplitude * np.sin(freq * t + phase)
            
        elif pattern_type == "cosine":
            # Fale cosinusoidalne
            freq = random.uniform(0.5, 3.0)
            amplitude = random.uniform(0.5, 2.0)
            phase = random.uniform(0, 2*np.pi)
            series = amplitude * np.cos(freq * t + phase)
            
        elif pattern_type == "trend":
            # Trendy liniowe i kwadratowe
            if random.choice([True, False]):
                # Trend liniowy
                slope = random.uniform(-1, 1)
                intercept = random.uniform(-2, 2)
                series = slope * t + intercept
            else:
                # Trend kwadratowy
                a = random.uniform(-0.1, 0.1)
                b = random.uniform(-1, 1)
                c = random.uniform(-2, 2)
                series = a * t**2 + b * t + c
                
        elif pattern_type == "seasonal":
            # Wzorce sezonowe (wielokrotne częstotliwości)
            series = np.zeros_like(t)
            n_components = random.randint(2, 4)
            for _ in range(n_components):
                freq = random.uniform(0.5, 5.0)
                amplitude = random.uniform(0.2, 1.0)
                phase = random.uniform(0, 2*np.pi)
                series += amplitude * np.sin(freq * t + phase)
                
        elif pattern_type == "noise":
            # Różne typy szumu
            noise_type = random.choice(["white", "brownian", "pink"])
            if noise_type == "white":
                series = np.random.normal(0, 1, length)
            elif noise_type == "brownian":
                series = np.cumsum(np.random.normal(0, 0.1, length))
            else:  # pink noise
                series = self._generate_pink_noise(length)
                
        elif pattern_type == "exponential":
            # Wzrost/spadek eksponencjalny
            rate = random.uniform(-0.5, 0.5)
            series = np.exp(rate * t)
            
        elif pattern_type == "spikes":
            # Szeregi z gwałtownymi skokami
            base = random.uniform(-1, 1)
            series = np.full(length, base)
            n_spikes = random.randint(3, 10)
            for _ in range(n_spikes):
                pos = random.randint(0, length-1)
                spike_height = random.uniform(-3, 3)
                series[pos] = spike_height
                
        elif pattern_type == "mixed":
            # Kombinacja różnych wzorców
            components = []
            n_components = random.randint(2, 4)
            
            base_types = ["sine", "trend", "seasonal", "noise"]
            for _ in range(n_components):
                comp_type = random.choice(base_types)
                weight = random.uniform(0.3, 1.0)
                component = weight * self.generate_synthetic_series(length, comp_type)
                components.append(component)
            
            series = np.sum(components, axis=0)
            
        else:
            raise ValueError(f"Unknown pattern type: {pattern_type}")
        
        # Dodaj trochę szumu do każdego szeregu
        noise_level = random.uniform(0.01, 0.1)
        series += np.random.normal(0, noise_level, length)
        
        # Normalizuj do zakresu [0, 1] dla lepszej stabilności obrazów
        series = (series - series.min()) / (series.max() - series.min() + 1e-8)
        
        return series
    
    def _generate_pink_noise(self, length: int) -> np.ndarray:
        """Generuje różowy szum (1/f noise)"""
        # Prosta implementacja różowego szumu
        white = np.random.randn(length)
        # Filtr 1/f w dziedzinie częstotliwości
        fft = np.fft.fft(white)
        freqs = np.fft.fftfreq(length)[1:]  # Pomijamy DC
        fft[1:] = fft[1:] / np.sqrt(freqs)
        pink = np.real(np.fft.ifft(fft))
        return pink

class MissingDataGenerator:
    """Generator różnych wzorców brakujących danych"""
    
    @staticmethod
    def create_random_mask(length: int, missing_rate: float) -> np.ndarray:
        """Tworzy losową maskę brakujących danych"""
        mask = np.ones(length, dtype=bool)
        n_missing = int(length * missing_rate)
        missing_indices = np.random.choice(length, n_missing, replace=False)
        mask[missing_indices] = False
        return mask
    
    @staticmethod
    def create_block_mask(length: int, missing_rate: float) -> np.ndarray:
        """Tworzy maskę z blokami brakujących danych"""
        mask = np.ones(length, dtype=bool)
        n_missing = int(length * missing_rate)
        
        # Stwórz kilka bloków
        n_blocks = random.randint(1, max(1, n_missing // 10))
        block_sizes = np.random.multinomial(n_missing, [1/n_blocks] * n_blocks)
        
        for block_size in block_sizes:
            if block_size > 0:
                start = random.randint(0, max(0, length - block_size))
                end = min(start + block_size, length)
                mask[start:end] = False
                
        return mask
    
    @staticmethod
    def create_periodic_mask(length: int, missing_rate: float) -> np.ndarray:
        """Tworzy maskę z periodycznymi brakami"""
        mask = np.ones(length, dtype=bool)
        period = random.randint(5, 20)
        missing_width = max(1, int(period * missing_rate))
        
        for i in range(0, length, period):
            end = min(i + missing_width, length)
            mask[i:end] = False
            
        return mask
    
    @staticmethod
    def create_edge_mask(length: int, missing_rate: float) -> np.ndarray:
        """Tworzy maskę z brakami na początku/końcu"""
        mask = np.ones(length, dtype=bool)
        n_missing = int(length * missing_rate)
        
        if random.choice([True, False]):  # Początek
            mask[:n_missing] = False
        else:  # Koniec
            mask[-n_missing:] = False
            
        return mask

class TrainingDatasetGenerator:
    """Główna klasa generująca dataset treningowy"""
    
    def __init__(self, output_dir: str = "stdiff_training_data"):
        self.output_dir = Path(output_dir)
        self.ts_generator = TimeSeriesGenerator()
        self.missing_generator = MissingDataGenerator()
        
        # Stwórz foldery
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "original").mkdir(exist_ok=True)
        (self.output_dir / "missing").mkdir(exist_ok=True)
        (self.output_dir / "masks").mkdir(exist_ok=True)
        
        # Typy obrazów do generowania
        self.image_types = ["gaf", "mtf", "rp", "spec"]
        self.encoders = {
            "gaf": to_gaf,
            "mtf": to_mtf, 
            "rp": to_rp,
            "spec": to_spectrogram
        }
        
    def generate_training_pair(self, series_id: int, pattern_type: str = "mixed", 
                              missing_rate: float = None, missing_type: str = None) -> dict:
        """
        Generuje jedną parę treningową (original, missing, mask)
        
        Returns:
            dict: {'original_series', 'missing_series', 'mask', 'metadata'}
        """
        # Parametry domyślne
        if missing_rate is None:
            missing_rate = random.uniform(0.05, 0.30)  # 5-30% missing
        if missing_type is None:
            missing_type = random.choice(["random", "block", "periodic", "edge"])
            
        # Generuj szereg czasowy
        series = self.ts_generator.generate_synthetic_series(pattern_type=pattern_type)
        
        # Generuj maskę brakujących danych
        if missing_type == "random":
            mask = self.missing_generator.create_random_mask(len(series), missing_rate)
        elif missing_type == "block":
            mask = self.missing_generator.create_block_mask(len(series), missing_rate)
        elif missing_type == "periodic":
            mask = self.missing_generator.create_periodic_mask(len(series), missing_rate)
        elif missing_type == "edge":
            mask = self.missing_generator.create_edge_mask(len(series), missing_rate)
        else:
            raise ValueError(f"Unknown missing type: {missing_type}")
            
        # Stwórz szereg z brakującymi danymi
        missing_series = series.copy()
        missing_series[~mask] = np.nan
        
        return {
            "original_series": series,
            "missing_series": missing_series,
            "mask": mask,
            "metadata": {
                "series_id": series_id,
                "pattern_type": pattern_type,
                "missing_type": missing_type,
                "missing_rate": missing_rate,
                "length": len(series)
            }
        }
    
    def series_to_images(self, series: np.ndarray, prefix: str) -> dict:
        """Konwertuje szereg na obrazy wszystkich typów"""
        images = {}
        
        # Wypełnij NaN dla encoderów (potrzebują kompletnych danych)
        series_filled = pd.Series(series).interpolate(method='linear').fillna(0)
        
        for img_type in self.image_types:
            try:
                encoder = self.encoders[img_type]
                image = encoder(series_filled)
                images[img_type] = image
            except Exception as e:
                print(f"Warning: Failed to create {img_type} image: {e}")
                # Stwórz pustą obrazek jako fallback
                images[img_type] = np.zeros((64, 64), dtype=np.float32)
                
        return images
    
    def save_training_pair(self, pair_data: dict, save_images: bool = True) -> dict:
        """Zapisuje parę treningową na dysk"""
        metadata = pair_data["metadata"]
        series_id = metadata["series_id"]
        
        file_paths = {}
        
        if save_images:
            # Konwertuj na obrazy
            original_images = self.series_to_images(pair_data["original_series"], f"original_{series_id}")
            missing_images = self.series_to_images(pair_data["missing_series"], f"missing_{series_id}")
            
            # Zapisz obrazy
            for img_type in self.image_types:
                # Obrazy oryginalne
                original_path = self.output_dir / "original" / f"{series_id:06d}_{img_type}.png"
                save_image(original_images[img_type], original_path)
                
                # Obrazy z brakami
                missing_path = self.output_dir / "missing" / f"{series_id:06d}_{img_type}.png"
                save_image(missing_images[img_type], missing_path)
                
                file_paths[f"original_{img_type}"] = str(original_path)
                file_paths[f"missing_{img_type}"] = str(missing_path)
        
        # Zapisz metadane
        metadata_path = self.output_dir / "masks" / f"{series_id:06d}_metadata.json"
        import json
        with open(metadata_path, 'w') as f:
            json.dump({
                **metadata,
                "file_paths": file_paths
            }, f, indent=2)
            
        return file_paths
    
    def generate_dataset(self, n_samples: int = 1000, pattern_distribution: dict = None):
        """
        Generuje kompletny dataset treningowy
        
        Args:
            n_samples: liczba par treningowych do wygenerowania
            pattern_distribution: rozkład typów wzorców (domyślnie równomierny)
        """
        if pattern_distribution is None:
            pattern_distribution = {
                "mixed": 0.3,
                "sine": 0.15,
                "seasonal": 0.15,
                "trend": 0.1,
                "noise": 0.1,
                "spikes": 0.1,
                "exponential": 0.1
            }
        
        # Sprawdź czy suma prawdopodobieństw = 1
        total_prob = sum(pattern_distribution.values())
        if abs(total_prob - 1.0) > 1e-6:
            # Znormalizuj
            pattern_distribution = {k: v/total_prob for k, v in pattern_distribution.items()}
        
        patterns = list(pattern_distribution.keys())
        probabilities = list(pattern_distribution.values())
        
        print(f"🎯 Generowanie {n_samples} par treningowych...")
        print(f"📊 Rozkład wzorców: {pattern_distribution}")
        
        metadata_summary = []
        
        for i in tqdm(range(n_samples), desc="Generating training pairs"):
            # Wybierz typ wzorca na podstawie rozkładu
            pattern_type = np.random.choice(patterns, p=probabilities)
            
            # Generuj parę treningową
            pair_data = self.generate_training_pair(
                series_id=i,
                pattern_type=pattern_type
            )
            
            # Zapisz na dysk
            file_paths = self.save_training_pair(pair_data)
            
            # Dodaj do podsumowania
            summary_entry = {
                **pair_data["metadata"],
                "file_paths": file_paths
            }
            metadata_summary.append(summary_entry)
        
        # Zapisz podsumowanie całego datasetu
        summary_path = self.output_dir / "dataset_summary.json"
        with open(summary_path, 'w') as f:
            json.dump({
                "total_samples": n_samples,
                "pattern_distribution": pattern_distribution,
                "image_types": self.image_types,
                "samples": metadata_summary
            }, f, indent=2)
        
        print(f"✅ Dataset wygenerowany w: {self.output_dir}")
        print(f"📋 Podsumowanie zapisane w: {summary_path}")
        
        # Wyświetl statystyki
        self._print_dataset_stats(metadata_summary)
    
    def _print_dataset_stats(self, metadata_summary: List[dict]):
        """Wyświetla statystyki wygenerowanego datasetu"""
        print(f"\n📊 STATYSTYKI DATASETU:")
        print(f"=" * 50)
        
        # Statystyki wzorców
        pattern_counts = {}
        missing_type_counts = {}
        missing_rates = []
        lengths = []
        
        for sample in metadata_summary:
            pattern = sample["pattern_type"]
            missing_type = sample["missing_type"]
            
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
            missing_type_counts[missing_type] = missing_type_counts.get(missing_type, 0) + 1
            missing_rates.append(sample["missing_rate"])
            lengths.append(sample["length"])
        
        print(f"🔢 Wzorce szeregów:")
        for pattern, count in sorted(pattern_counts.items()):
            percentage = count / len(metadata_summary) * 100
            print(f"  {pattern}: {count} ({percentage:.1f}%)")
        
        print(f"\n🕳️ Typy brakujących danych:")
        for missing_type, count in sorted(missing_type_counts.items()):
            percentage = count / len(metadata_summary) * 100
            print(f"  {missing_type}: {count} ({percentage:.1f}%)")
        
        print(f"\n📏 Statystyki numeryczne:")
        print(f"  Długość szeregów: {np.min(lengths)}-{np.max(lengths)} (avg: {np.mean(lengths):.0f})")
        print(f"  Missing rate: {np.min(missing_rates):.2f}-{np.max(missing_rates):.2f} (avg: {np.mean(missing_rates):.2f})")
        print(f"  Całkowita liczba obrazów: {len(metadata_summary) * len(self.image_types) * 2}")

def main():
    parser = argparse.ArgumentParser(description="Generate training dataset for Stable Diffusion fine-tuning")
    parser.add_argument("--samples", type=int, default=1000, help="Number of training pairs to generate")
    parser.add_argument("--output", default="stdiff_training_data", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Ustaw seed dla reprodukowalności
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    print(f"🚀 GENERATOR DATASETU TRENINGOWEGO")
    print(f"=" * 60)
    print(f"📁 Folder wyjściowy: {args.output}")
    print(f"🔢 Liczba próbek: {args.samples}")
    print(f"🌱 Random seed: {args.seed}")
    print()
    
    # Stwórz generator i wygeneruj dataset
    generator = TrainingDatasetGenerator(args.output)
    generator.generate_dataset(n_samples=args.samples)
    
    print(f"\n🎉 GOTOWE! Dataset gotowy do fine-tuningu Stable Diffusion 2")

if __name__ == "__main__":
    main()
