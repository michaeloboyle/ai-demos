#!/usr/bin/env python3
"""
Local Image Processing Setup

Sets up directories and basic tools for PURSUIT helmet defect overlay generation.
Simplified approach: Use existing 2 angles + physics-based defect overlays.

Storage location: /Volumes/black box - Backup Data 2020/defense-ai-models/
"""

import os
import sys
import time
import shutil
from pathlib import Path

# External drive paths
EXTERNAL_DRIVE = "/Volumes/black box - Backup Data 2020"
MODEL_BASE_PATH = os.path.join(EXTERNAL_DRIVE, "defense-ai-models")
IMAGE_MODELS_PATH = os.path.join(MODEL_BASE_PATH, "image-generation")

def check_external_drive():
    """Check if external drive is available"""

    print("🔍 Checking external drive availability...")

    if not os.path.exists(EXTERNAL_DRIVE):
        print(f"❌ External drive not found: {EXTERNAL_DRIVE}")
        print("📝 Please ensure external drive 'black box' is mounted")
        return False

    # Check available space (need ~10GB)
    statvfs = os.statvfs(EXTERNAL_DRIVE)
    available_gb = (statvfs.f_bavail * statvfs.f_frsize) / (1024**3)

    print(f"✅ External drive found: {EXTERNAL_DRIVE}")
    print(f"💾 Available space: {available_gb:.1f} GB")

    if available_gb < 12:
        print(f"⚠️  Warning: Low space. Recommend 12+ GB available")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return False

    return True

def setup_model_directories():
    """Create directory structure on external drive"""

    print("📁 Setting up model directories...")

    directories = [
        MODEL_BASE_PATH,
        IMAGE_MODELS_PATH,
        os.path.join(IMAGE_MODELS_PATH, "stable-diffusion-xl"),
        os.path.join(IMAGE_MODELS_PATH, "controlnet"),
        os.path.join(IMAGE_MODELS_PATH, "generated-helmets"),
        os.path.join(IMAGE_MODELS_PATH, "generated-helmets", "angles"),
        os.path.join(IMAGE_MODELS_PATH, "generated-helmets", "defects"),
        os.path.join(IMAGE_MODELS_PATH, "generated-helmets", "variations")
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"  ✅ Created: {directory}")

    return True

def check_dependencies():
    """Check required Python packages"""

    print("🐍 Checking Python dependencies...")

    required_packages = [
        "torch",
        "diffusers",
        "transformers",
        "accelerate",
        "PIL",  # Pillow installs as PIL
        "numpy"
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            if package == "PIL":
                missing_packages.append("pillow")
                print(f"  ❌ pillow - Missing")
            else:
                missing_packages.append(package)
                print(f"  ❌ {package} - Missing")

    if missing_packages:
        print(f"\n📦 Install missing packages:")
        print(f"pip install {' '.join(missing_packages)}")
        return False

    return True

def download_stable_diffusion_xl():
    """Download Stable Diffusion XL to external drive"""

    print("\n🎨 Downloading Stable Diffusion XL...")
    print("📊 Model size: ~7GB")
    print("⏱️  Estimated time: 10-30 minutes")

    try:
        from diffusers import StableDiffusionXLPipeline
        import torch

        model_path = os.path.join(IMAGE_MODELS_PATH, "stable-diffusion-xl")

        # Check if already exists
        if os.path.exists(os.path.join(model_path, "unet")):
            print("  ✅ Stable Diffusion XL already downloaded")
            return True

        print("  📥 Downloading Stable Diffusion XL base model...")

        # Download with float16 to save space
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16"
        )

        print(f"  💾 Saving to external drive: {model_path}")
        pipeline.save_pretrained(model_path)

        # Clean up memory
        del pipeline

        print("  ✅ Stable Diffusion XL downloaded successfully")
        return True

    except Exception as e:
        print(f"  ❌ Error downloading SDXL: {e}")
        return False

def download_controlnet_models():
    """Download ControlNet models for precise generation"""

    print("\n🎯 Downloading ControlNet models...")

    controlnet_models = [
        {
            "name": "canny",
            "model_id": "diffusers/controlnet-canny-sdxl-1.0",
            "description": "Edge-guided generation"
        },
        {
            "name": "depth",
            "model_id": "diffusers/controlnet-depth-sdxl-1.0",
            "description": "Depth-guided generation"
        }
    ]

    try:
        from diffusers import ControlNetModel

        for model_info in controlnet_models:
            model_path = os.path.join(IMAGE_MODELS_PATH, "controlnet", model_info["name"])

            # Check if already exists
            if os.path.exists(os.path.join(model_path, "config.json")):
                print(f"  ✅ ControlNet {model_info['name']} already downloaded")
                continue

            print(f"  📥 Downloading ControlNet {model_info['name']}...")
            print(f"      {model_info['description']}")

            controlnet = ControlNetModel.from_pretrained(
                model_info["model_id"],
                torch_dtype=torch.float16
            )

            controlnet.save_pretrained(model_path)
            del controlnet

            print(f"  ✅ ControlNet {model_info['name']} downloaded")

        return True

    except Exception as e:
        print(f"  ❌ Error downloading ControlNet: {e}")
        return False

def download_inpainting_model():
    """Download inpainting model for defect generation"""

    print("\n🎨 Downloading inpainting model...")

    try:
        from diffusers import StableDiffusionXLInpaintPipeline

        model_path = os.path.join(IMAGE_MODELS_PATH, "stable-diffusion-xl-inpaint")

        if os.path.exists(os.path.join(model_path, "unet")):
            print("  ✅ Inpainting model already downloaded")
            return True

        print("  📥 Downloading SDXL inpainting model...")

        pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(
            "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
            torch_dtype=torch.float16,
            variant="fp16"
        )

        pipeline.save_pretrained(model_path)
        del pipeline

        print("  ✅ Inpainting model downloaded")
        return True

    except Exception as e:
        print(f"  ❌ Error downloading inpainting model: {e}")
        return False

def create_model_info():
    """Create model information file"""

    print("\n📝 Creating model information...")

    model_info = {
        "downloaded_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models": {
            "stable-diffusion-xl": {
                "path": "stable-diffusion-xl/",
                "size_gb": 7,
                "description": "Base SDXL model for helmet generation",
                "use_case": "Generate missing helmet angles"
            },
            "controlnet-canny": {
                "path": "controlnet/canny/",
                "size_gb": 1.5,
                "description": "Edge-guided generation",
                "use_case": "Maintain helmet shape consistency"
            },
            "controlnet-depth": {
                "path": "controlnet/depth/",
                "size_gb": 1.5,
                "description": "Depth-guided generation",
                "use_case": "Generate different viewing angles"
            },
            "inpainting": {
                "path": "stable-diffusion-xl-inpaint/",
                "size_gb": 7,
                "description": "Inpainting model for defects",
                "use_case": "Add realistic damage patterns"
            }
        },
        "total_size_gb": 17,
        "usage": {
            "memory_requirement": "10-12GB RAM during generation",
            "generation_time": "30-60 seconds per image",
            "resolution": "1024x1024 pixels"
        }
    }

    info_path = os.path.join(IMAGE_MODELS_PATH, "model_info.json")

    import json
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)

    print(f"  ✅ Model info saved: {info_path}")
    return True

def main():
    """Main download function"""

    print("🚀 Local Image Model Downloader for PURSUIT Helmet Generation\n")
    print("📋 This will download ~10GB of image generation models")
    print("💾 Models will be stored on external drive for Mac Mini M2 Pro\n")

    # Pre-flight checks
    if not check_external_drive():
        return False

    if not setup_model_directories():
        return False

    if not check_dependencies():
        return False

    # Download models
    print("\n" + "="*60)
    print("🎯 Starting Model Downloads")
    print("="*60)

    download_tasks = [
        ("Base Generation Model", download_stable_diffusion_xl),
        ("ControlNet Models", download_controlnet_models),
        ("Inpainting Model", download_inpainting_model),
        ("Model Information", create_model_info)
    ]

    for task_name, task_func in download_tasks:
        print(f"\n📦 {task_name}")
        if not task_func():
            print(f"❌ Failed: {task_name}")
            return False

    print("\n" + "="*60)
    print("✅ All Models Downloaded Successfully!")
    print("="*60)

    print(f"📁 Models location: {IMAGE_MODELS_PATH}")
    print(f"💾 Total size: ~10GB")

    print(f"\n🎯 Next Steps:")
    print("1. Run: python scripts/generate_helmet_variations.py")
    print("2. Generate missing PURSUIT helmet angles")
    print("3. Create realistic defect variations")

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)