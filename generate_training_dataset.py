#!/usr/bin/env python3
"""
Training dataset generator for fine-tuning Stable Diffusion 2 on mathematical images

This script generates diverse time series and converts them to GAF, MTF, RP, 
and Spectrogram images with various missing data patterns.
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
    """Generator for diverse synthetic time series patterns"""
    
    def __init__(self, min_length=100, max_length=1000):
        self.min_length = min_length
        self.max_length = max_length
        
    def generate_synthetic_series(self, length: int = None, pattern_type: str = "mixed") -> np.ndarray:
        """
        Generates a synthetic time series with a specified pattern
        
        Args:
            length: Length of the series (if None, random)
            pattern_type: Pattern type ("sine", "trend", "seasonal", "noise", "mixed", etc.)
        """
        if length is None:
            length = random.randint(self.min_length, self.max_length)
            
        t = np.linspace(0, 10, length)
        
        if pattern_type == "sine":
            # Simple sinusoidal waves with random parameters
            freq = random.uniform(0.5, 3.0)
            amplitude = random.uniform(0.5, 2.0)
            phase = random.uniform(0, 2*np.pi)
            series = amplitude * np.sin(freq * t + phase)
            
        elif pattern_type == "cosine":
            # Cosine waves
            freq = random.uniform(0.5, 3.0)
            amplitude = random.uniform(0.5, 2.0)
            phase = random.uniform(0, 2*np.pi)
            series = amplitude * np.cos(freq * t + phase)
            
        elif pattern_type == "trend":
            # Linear and quadratic trends
            if random.choice([True, False]):
                # Linear trend
                slope = random.uniform(-1, 1)
                intercept = random.uniform(-2, 2)
                series = slope * t + intercept
            else:
                # Quadratic trend
                a = random.uniform(-0.1, 0.1)
                b = random.uniform(-1, 1)
                c = random.uniform(-2, 2)
                series = a * t**2 + b * t + c
                
        elif pattern_type == "seasonal":
            # Seasonal patterns (multiple frequencies)
            series = np.zeros_like(t)
            n_components = random.randint(2, 4)
            for _ in range(n_components):
                freq = random.uniform(0.5, 5.0)
                amplitude = random.uniform(0.2, 1.0)
                phase = random.uniform(0, 2*np.pi)
                series += amplitude * np.sin(freq * t + phase)
                
        elif pattern_type == "noise":
            # Various noise types
            noise_type = random.choice(["white", "brownian", "pink"])
            if noise_type == "white":
                series = np.random.normal(0, 1, length)
            elif noise_type == "brownian":
                series = np.cumsum(np.random.normal(0, 0.1, length))
            else:  # pink noise
                series = self._generate_pink_noise(length)
                
        elif pattern_type == "exponential":
            # Exponential growth/decay
            rate = random.uniform(-0.5, 0.5)
            series = np.exp(rate * t)
            
        elif pattern_type == "spikes":
            # Series with sudden jumps
            base = random.uniform(-1, 1)
            series = np.full(length, base)
            n_spikes = random.randint(3, 10)
            for _ in range(n_spikes):
                pos = random.randint(0, length-1)
                spike_height = random.uniform(-3, 3)
                series[pos] = spike_height
                
        elif pattern_type == "mixed":
            # Combination of different patterns
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
        
        # Add some noise to every series
        noise_level = random.uniform(0.01, 0.1)
        series += np.random.normal(0, noise_level, length)
        
        # Normalize to [0, 1] range for better image stability
        series = (series - series.min()) / (series.max() - series.min() + 1e-8)
        
        return series
    
    def _generate_pink_noise(self, length: int) -> np.ndarray:
        """Generates pink noise (1/f noise)"""
        # Simple pink noise implementation
        white = np.random.randn(length)
        # Apply 1/f filter in frequency domain
        fft = np.fft.fft(white)
        freqs = np.fft.fftfreq(length)[1:]  # Skip DC component
        fft[1:] = fft[1:] / np.sqrt(freqs)
        pink = np.real(np.fft.ifft(fft))
        return pink

class MissingDataGenerator:
    """Generator for various missing data patterns"""
    
    @staticmethod
    def create_random_mask(length: int, missing_rate: float) -> np.ndarray:
        """Creates a random missing data mask"""
        mask = np.ones(length, dtype=bool)
        n_missing = int(length * missing_rate)
        missing_indices = np.random.choice(length, n_missing, replace=False)
        mask[missing_indices] = False
        return mask
    
    @staticmethod
    def create_block_mask(length: int, missing_rate: float) -> np.ndarray:
        """Creates a mask with blocks of missing data"""
        mask = np.ones(length, dtype=bool)
        n_missing = int(length * missing_rate)
        
        # Create several blocks
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
        """Creates a mask with periodic missing data"""
        mask = np.ones(length, dtype=bool)
        period = random.randint(5, 20)
        missing_width = max(1, int(period * missing_rate))
        
        for i in range(0, length, period):
            end = min(i + missing_width, length)
            mask[i:end] = False
            
        return mask
    
    @staticmethod
    def create_edge_mask(length: int, missing_rate: float) -> np.ndarray:
        """Creates a mask with missing data at beginning/end"""
        mask = np.ones(length, dtype=bool)
        n_missing = int(length * missing_rate)
        
        if random.choice([True, False]):  # Beginning
            mask[:n_missing] = False
        else:  # End
            mask[-n_missing:] = False
            
        return mask

class TrainingDatasetGenerator:
    """Main class for generating training datasets"""
    
    def __init__(self, output_dir: str = "stdiff_training_data"):
        self.output_dir = Path(output_dir)
        self.ts_generator = TimeSeriesGenerator()
        self.missing_generator = MissingDataGenerator()
        
        # Create folders
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "original").mkdir(exist_ok=True)
        (self.output_dir / "missing").mkdir(exist_ok=True)
        (self.output_dir / "masks").mkdir(exist_ok=True)
        
        # Image types to generate
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
        Generates one training pair (original, missing, mask)
        
        Returns:
            dict: {'original_series', 'missing_series', 'mask', 'metadata'}
        """
        # Default parameters
        if missing_rate is None:
            missing_rate = random.uniform(0.05, 0.30)  # 5-30% missing
        if missing_type is None:
            missing_type = random.choice(["random", "block", "periodic", "edge"])
            
        # Generate time series
        series = self.ts_generator.generate_synthetic_series(pattern_type=pattern_type)
        
        # Generate missing data mask
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
            
        # Create series with missing data
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
        """Converts time series to images of all types"""
        images = {}
        
        # Fill NaN for encoders (they need complete data)
        series_filled = pd.Series(series).interpolate(method='linear').fillna(0)
        
        for img_type in self.image_types:
            try:
                encoder = self.encoders[img_type]
                image = encoder(series_filled)
                images[img_type] = image
            except Exception as e:
                print(f"Warning: Failed to create {img_type} image: {e}")
                # Create empty image as fallback
                images[img_type] = np.zeros((64, 64), dtype=np.float32)
                
        return images
    
    def save_training_pair(self, pair_data: dict, save_images: bool = True) -> dict:
        """Saves training pair to disk"""
        metadata = pair_data["metadata"]
        series_id = metadata["series_id"]
        
        file_paths = {}
        
        if save_images:
            # Convert to images
            original_images = self.series_to_images(pair_data["original_series"], f"original_{series_id}")
            missing_images = self.series_to_images(pair_data["missing_series"], f"missing_{series_id}")
            
            # Save images
            for img_type in self.image_types:
                # Original images
                original_path = self.output_dir / "original" / f"{series_id:06d}_{img_type}.png"
                save_image(original_images[img_type], original_path)
                
                # Images with missing data
                missing_path = self.output_dir / "missing" / f"{series_id:06d}_{img_type}.png"
                save_image(missing_images[img_type], missing_path)
                
                file_paths[f"original_{img_type}"] = str(original_path)
                file_paths[f"missing_{img_type}"] = str(missing_path)
        
        # Save metadata
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
        Generates complete training dataset
        
        Args:
            n_samples: Number of training pairs to generate
            pattern_distribution: Distribution of pattern types (default: uniform)
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
        
        # Check if probabilities sum to 1
        total_prob = sum(pattern_distribution.values())
        if abs(total_prob - 1.0) > 1e-6:
            # Normalize
            pattern_distribution = {k: v/total_prob for k, v in pattern_distribution.items()}
        
        patterns = list(pattern_distribution.keys())
        probabilities = list(pattern_distribution.values())
        
        print(f"🎯 Generating {n_samples} training pairs...")
        print(f"📊 Pattern distribution: {pattern_distribution}")
        
        metadata_summary = []
        
        for i in tqdm(range(n_samples), desc="Generating training pairs"):
            # Select pattern type based on distribution
            pattern_type = np.random.choice(patterns, p=probabilities)
            
            # Generate training pair
            pair_data = self.generate_training_pair(
                series_id=i,
                pattern_type=pattern_type
            )
            
            # Save to disk
            file_paths = self.save_training_pair(pair_data)
            
            # Add to summary
            summary_entry = {
                **pair_data["metadata"],
                "file_paths": file_paths
            }
            metadata_summary.append(summary_entry)
        
        # Save complete dataset summary
        summary_path = self.output_dir / "dataset_summary.json"
        with open(summary_path, 'w') as f:
            json.dump({
                "total_samples": n_samples,
                "pattern_distribution": pattern_distribution,
                "image_types": self.image_types,
                "samples": metadata_summary
            }, f, indent=2)
        
        print(f"✅ Dataset generated in: {self.output_dir}")
        print(f"📋 Summary saved in: {summary_path}")
        
        # Display statistics
        self._print_dataset_stats(metadata_summary)
    
    def _print_dataset_stats(self, metadata_summary: List[dict]):
        """Displays statistics of the generated dataset"""
        print(f"\n📊 DATASET STATISTICS:")
        print(f"=" * 50)
        
        # Pattern statistics
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
        
        print(f"🔢 Time series patterns:")
        for pattern, count in sorted(pattern_counts.items()):
            percentage = count / len(metadata_summary) * 100
            print(f"  {pattern}: {count} ({percentage:.1f}%)")
        
        print(f"\n🕳️ Missing data types:")
        for missing_type, count in sorted(missing_type_counts.items()):
            percentage = count / len(metadata_summary) * 100
            print(f"  {missing_type}: {count} ({percentage:.1f}%)")
        
        print(f"\n📏 Numerical statistics:")
        print(f"  Series length: {np.min(lengths)}-{np.max(lengths)} (avg: {np.mean(lengths):.0f})")
        print(f"  Missing rate: {np.min(missing_rates):.2f}-{np.max(missing_rates):.2f} (avg: {np.mean(missing_rates):.2f})")
        print(f"  Total images: {len(metadata_summary) * len(self.image_types) * 2}")

def main():
    parser = argparse.ArgumentParser(description="Generate training dataset for Stable Diffusion fine-tuning")
    parser.add_argument("--samples", type=int, default=1000, help="Number of training pairs to generate")
    parser.add_argument("--output", default="stdiff_training_data", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    print(f"🚀 TRAINING DATASET GENERATOR")
    print(f"=" * 60)
    print(f"📁 Output directory: {args.output}")
    print(f"🔢 Number of samples: {args.samples}")
    print(f"🌱 Random seed: {args.seed}")
    print()
    
    # Create generator and generate dataset
    generator = TrainingDatasetGenerator(args.output)
    generator.generate_dataset(n_samples=args.samples)
    
    print(f"\n🎉 DONE! Dataset ready for Stable Diffusion 2 fine-tuning")

if __name__ == "__main__":
    main()
