# Local Image Generation Research for PURSUIT Helmet Dataset

## Constraints (Mac Mini M2 Pro)
- **Memory**: 16GB unified memory
- **Storage**: External drive required for models (`/Volumes/black box/`)
- **Requirements**: Local open-source models only
- **Current Stack**: Ollama (text: Llama 3.1 8B, vision: LLaVA 1.6 7B)

## Local Image Generation Options

### 1. **Stable Diffusion XL** (Recommended)
**Model Size**: ~7GB
**Requirements**: 8-12GB RAM during generation
**Storage**: `/Volumes/black box/defense-ai-models/stable-diffusion-xl/`

**Capabilities:**
- High-quality image generation (1024x1024)
- LoRA fine-tuning support
- ControlNet integration for precise control
- Inpainting for defect overlays

**Setup:**
```bash
# Install dependencies
pip install diffusers transformers accelerate

# Download model to external drive
python -c "
from diffusers import StableDiffusionXLPipeline
import torch

pipeline = StableDiffusionXLPipeline.from_pretrained(
    'stabilityai/stable-diffusion-xl-base-1.0',
    torch_dtype=torch.float16
)
pipeline.save_pretrained('/Volumes/black box/defense-ai-models/stable-diffusion-xl/')
"
```

### 2. **ControlNet for Precise Generation**
**Model Size**: ~3GB additional
**Use Case**: Generate exact viewing angles from reference

**ControlNet Types for Helmets:**
- **Canny Edge**: Maintain helmet outline/shape
- **Depth**: Generate different viewing angles
- **Pose**: Control helmet orientation
- **Inpaint**: Add realistic defects

### 3. **LCM-LoRA** (Speed Optimization)
**Model Size**: ~100MB
**Benefit**: 4-8 steps instead of 20-50 (4x faster)
**Quality**: Slightly reduced but acceptable for variations

### 4. **Alternative: Flux.1-dev**
**Model Size**: ~12GB
**Quality**: Higher than SDXL
**Concern**: May exceed 16GB RAM limit with other models

## Recommended Local Workflow

### Phase 1: Setup Local Image Generation
```bash
# Create model directory on external drive
mkdir -p "/Volumes/black box/defense-ai-models/image-generation"

# Install dependencies
pip install diffusers transformers accelerate controlnet_aux

# Download SDXL to external drive
python scripts/download_image_models.py
```

### Phase 2: Generate Missing Angles
```python
from diffusers import StableDiffusionXLPipeline, ControlNetModel
import torch

# Load from external drive
model_path = "/Volumes/black box/defense-ai-models/stable-diffusion-xl"
pipeline = StableDiffusionXLPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.float16
)

# Generate left profile from reference
prompt = """
Professional product photograph of Gentex PURSUIT flight helmet,
carbon fiber shell, left three-quarter profile view,
studio lighting, white background, ultra-high resolution,
detailed hardware, communication systems visible
"""

image = pipeline(
    prompt=prompt,
    height=1024,
    width=1024,
    num_inference_steps=20,
    guidance_scale=7.5
).images[0]
```

### Phase 3: Defect Generation via Inpainting
```python
from diffusers import StableDiffusionXLInpaintPipeline

# Load inpainting pipeline
inpaint_pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(
    model_path + "/inpaint"
)

# Apply realistic defect
defect_prompt = """
realistic ballistic impact damage on carbon fiber helmet,
circular impact crater, radial cracks, fiber delamination,
professional lighting
"""

damaged_helmet = inpaint_pipeline(
    prompt=defect_prompt,
    image=clean_helmet_image,
    mask_image=impact_zone_mask,
    height=1024,
    width=1024
).images[0]
```

## Memory Management Strategy

### Sequential Model Loading (16GB RAM)
```python
import gc
import torch

def generate_helmet_variations():
    # 1. Unload Ollama models temporarily
    ollama.stop()
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # 2. Load SDXL pipeline
    pipeline = load_sdxl_pipeline()

    # 3. Generate images
    variations = generate_angles_and_defects(pipeline)

    # 4. Cleanup and restart Ollama
    del pipeline
    gc.collect()
    ollama.start()

    return variations
```

### Storage Organization
```
/Volumes/black box/defense-ai-models/
├── stable-diffusion-xl/          # ~7GB
│   ├── unet/
│   ├── vae/
│   ├── text_encoder/
│   └── scheduler/
├── controlnet/                   # ~3GB
│   ├── canny/
│   ├── depth/
│   └── inpaint/
├── lcm-lora/                     # ~100MB
└── generated-helmets/            # Output storage
    ├── angles/
    ├── defects/
    └── variations/
```

## Implementation Plan

### 1. Model Download Script
```python
# scripts/download_image_models.py
def download_to_external_drive():
    models = [
        "stabilityai/stable-diffusion-xl-base-1.0",
        "diffusers/controlnet-canny-sdxl-1.0",
        "diffusers/controlnet-depth-sdxl-1.0"
    ]

    base_path = "/Volumes/black box/defense-ai-models/"

    for model in models:
        print(f"Downloading {model}...")
        # Download and save to external drive
```

### 2. Helmet Generation Pipeline
```python
# scripts/generate_helmet_variations.py
class PursuitHelmetGenerator:
    def __init__(self):
        self.model_path = "/Volumes/black box/defense-ai-models/"
        self.reference_images = load_pursuit_references()

    def generate_missing_angles(self):
        # Left profile, top view, rear view, bottom view
        pass

    def generate_defect_variations(self):
        # Apply physics-based defect patterns
        pass

    def optimize_for_qc(self):
        # Ensure QC-suitable quality and detail
        pass
```

### 3. Quality Validation
- **Resolution**: 1024x1024 minimum (vs reference 3560x3776)
- **Realism**: Compare against reference using CLIP similarity
- **Consistency**: Maintain PURSUIT helmet design elements
- **Defect Accuracy**: Validate against physics patterns

## Next Steps

1. **Download SDXL** to external drive (~7GB)
2. **Test generation** with PURSUIT reference images
3. **Generate missing angles** (5 angles × 2-3 variations)
4. **Create defect overlays** using inpainting
5. **Build QC dataset** with 50-100 total images

## Expected Performance
- **Generation Time**: 30-60 seconds per image
- **Memory Usage**: 10-12GB during generation
- **Storage Need**: ~500MB for generated dataset
- **Quality**: High enough for QC training/demo purposes

This local approach gives us complete control and privacy while working within Mac Mini constraints!