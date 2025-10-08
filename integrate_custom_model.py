#!/usr/bin/env python3
"""
Skrypt integracji custom Stable Diffusion 2 z systemem eksperymentów

Ten skrypt dodaje wytrenowany model do INPAINTERS w ts_image_inpainting.py
"""

import os
import sys
from pathlib import Path
import argparse

def integrate_custom_model(model_path: str, model_name: str = "stdiff"):
    """
    Integruje custom model z istniejącym systemem
    
    Args:
        model_path: ścieżka do wytrenowanego modelu
        model_name: nazwa modelu w systemie (np. "stdiff", "sd2-gaf")
    """
    
    print(f"🔗 INTEGRACJA CUSTOM MODELU STABLE DIFFUSION 2")
    print(f"=" * 60)
    print(f"📁 Model path: {model_path}")
    print(f"🏷️ Model name: {model_name}")
    
    # Sprawdź czy model istnieje
    if not os.path.exists(model_path):
        print(f"❌ Model path not found: {model_path}")
        return False
    
    # Import custom inpainter
    try:
        from models.stdiff import StableDiffusion2MathInpainter
        print("✅ Custom inpainter class imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import custom inpainter: {e}")
        return False
    
    # Stwórz instancję modelu
    try:
        custom_inpainter = StableDiffusion2MathInpainter(model_path=model_path)
        print("✅ Custom model instance created successfully")
    except Exception as e:
        print(f"❌ Failed to create model instance: {e}")
        return False
    
    # Modyfikuj ts_image_inpainting.py
    ts_file = "ts_image_inpainting.py"
    
    # Czytaj plik
    with open(ts_file, 'r') as f:
        content = f.read()
    
    # Znajdź sekcję INPAINTERS
    inpainters_start = content.find("INPAINTERS = {")
    if inpainters_start == -1:
        print(f"❌ Could not find INPAINTERS dictionary in {ts_file}")
        return False
    
    # Znajdź koniec słownika INPAINTERS
    inpainters_end = content.find("}", inpainters_start)
    if inpainters_end == -1:
        print(f"❌ Could not find end of INPAINTERS dictionary")
        return False
    
    # Sprawdź czy model już istnieje
    if f'"{model_name}":' in content[inpainters_start:inpainters_end]:
        print(f"⚠️ Model {model_name} already exists in INPAINTERS")
        print("Use a different name or remove existing entry manually")
        return False
    
    # Dodaj import na górze pliku
    import_line = f"from models.stdiff import StableDiffusion2MathInpainter\n"
    
    # Sprawdź czy import już istnieje
    if "from models.stdiff import" not in content:
        # Znajdź miejsce na dodanie importu (po ostatnim import)
        lines = content.split('\n')
        last_import_idx = -1
        
        for i, line in enumerate(lines):
            if line.strip().startswith('import ') or line.strip().startswith('from '):
                last_import_idx = i
        
        if last_import_idx >= 0:
            lines.insert(last_import_idx + 1, import_line.strip())
            content = '\n'.join(lines)
            print("✅ Added import statement")
    
    # Stwórz instancję modelu przed INPAINTERS
    # Auto-detect multi-type model
    is_multi_type = ("stable_diffusion_2_all" in model_path or 
                     "sd2-all" in model_name or 
                     "multi" in model_path.lower())
    
    if is_multi_type:
        model_instance_code = f'\n# Custom Stable Diffusion 2 multi-type model (trained on GAF, MTF, RP, SPEC)\ncustom_stdiff_model = StableDiffusion2MathInpainter(model_path="{model_path}", is_multi_type=True)\n'
    else:
        model_instance_code = f'\n# Custom Stable Diffusion 2 model\ncustom_stdiff_model = StableDiffusion2MathInpainter(model_path="{model_path}")\n'
    
    # Znajdź miejsce przed INPAINTERS
    unet_model_line = content.find("unet_model = UnetInpainter()")
    if unet_model_line != -1:
        # Znajdź koniec tej linii
        line_end = content.find('\n', unet_model_line)
        content = content[:line_end] + model_instance_code + content[line_end:]
        print("✅ Added model instance")
    
    # Dodaj model do INPAINTERS
    new_entry = f'    "{model_name}": custom_stdiff_model,  # Custom Stable Diffusion 2 for math images\n'
    
    # Wstaw przed zamknięciem słownika
    content = content[:inpainters_end] + new_entry + content[inpainters_end:]
    
    # Zapisz backup
    backup_file = f"{ts_file}.backup"
    with open(backup_file, 'w') as f:
        f.write(open(ts_file, 'r').read())
    print(f"✅ Created backup: {backup_file}")
    
    # Zapisz zmodyfikowany plik
    with open(ts_file, 'w') as f:
        f.write(content)
    
    print(f"✅ Successfully integrated model '{model_name}' into {ts_file}")
    print(f"🎯 Model can now be used in experiments with: --inpainting_models {model_name}")
    
    return True

def test_integration(model_name: str = "stdiff"):
    """Testuje czy integracja działa"""
    print(f"\n🧪 TESTING INTEGRATION")
    print(f"=" * 30)
    
    try:
        # Import ts_image_inpainting
        import ts_image_inpainting
        
        # Sprawdź czy model jest w INPAINTERS
        if model_name in ts_image_inpainting.INPAINTERS:
            print(f"✅ Model '{model_name}' found in INPAINTERS")
            
            # Test basic functionality
            inpainter = ts_image_inpainting.INPAINTERS[model_name]
            print(f"✅ Model instance accessible")
            print(f"✅ Integration successful!")
            return True
        else:
            print(f"❌ Model '{model_name}' not found in INPAINTERS")
            print(f"Available models: {list(ts_image_inpainting.INPAINTERS.keys())}")
            return False
            
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Integrate custom Stable Diffusion 2 model")
    parser.add_argument("--model_path", required=True, help="Path to trained model")
    parser.add_argument("--model_name", default="stdiff", help="Name for the model in system")
    parser.add_argument("--test", action="store_true", help="Test integration after adding")
    
    args = parser.parse_args()
    
    # Integruj model
    success = integrate_custom_model(args.model_path, args.model_name)
    
    if success and args.test:
        # Testuj integrację
        test_integration(args.model_name)
    
    if success:
        print(f"\n🎉 INTEGRATION COMPLETED!")
        print(f"🚀 Usage in experiments:")
        print(f"   python run_improved_experiment.py --quick --inpainting_models {args.model_name}")
        print(f"   python run_improved_experiment.py --inpainting_models unet {args.model_name}")

if __name__ == "__main__":
    main()
