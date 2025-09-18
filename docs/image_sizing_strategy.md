# Image Sizing Strategy for PURSUIT Helmet Generation

## Current Situation
- **Reference Images**: 3560×3776px (~13.4MP, 6.5MB PNG)
- **Target Generation**: Need multiple angles + defect variations
- **Constraints**: Mac Mini M2 Pro 16GB RAM, local models only

## Performance vs Quality Analysis

### Current Reference Size: 3560×3776px
**Pros:**
- Ultra-high detail for professional QC analysis
- Can see carbon fiber weave texture
- Perfect for real production use

**Cons:**
- **Memory intensive**: 3560×3776×4 channels = ~203MB per image in memory
- **Generation time**: SDXL at this resolution would take 3-5 minutes per image
- **RAM pressure**: May cause memory issues with 16GB constraint

### Recommended Sizes for AI Generation

#### 1. **1024×1024px** (SDXL Native)
**Use Case**: Primary generation size
- **Memory**: ~16MB per image
- **Generation time**: 30-60 seconds
- **Quality**: High enough for QC demos
- **SDXL optimized**: Native resolution for best quality

#### 2. **2048×2048px** (High Quality)
**Use Case**: Final output after upscaling
- **Memory**: ~64MB per image
- **Generation time**: 2-3 minutes
- **Quality**: Near-reference level
- **Method**: Generate 1024px → AI upscale to 2048px

#### 3. **512×512px** (Rapid Prototyping)
**Use Case**: Fast iteration during development
- **Memory**: ~4MB per image
- **Generation time**: 10-20 seconds
- **Quality**: Suitable for testing defect patterns

## Recommended Workflow

### Phase 1: Reference Preprocessing
```python
def resize_references_for_generation():
    """Resize PURSUIT references for different use cases"""

    reference_sizes = {
        "original": (3560, 3776),      # Keep originals
        "generation_input": (1024, 1024),  # For SDXL conditioning
        "quick_test": (512, 512),      # For rapid iteration
        "final_output": (2048, 2048)   # Target quality
    }

    # Resize maintaining aspect ratio with smart cropping
    for size_name, dimensions in reference_sizes.items():
        resize_and_save(original_image, dimensions, f"reference_{size_name}.png")
```

### Phase 2: Multi-Resolution Generation
```python
def generate_helmet_variations():
    """Generate at optimal size, upscale if needed"""

    # 1. Generate at 1024×1024 (SDXL native)
    base_image = generate_with_sdxl(
        prompt=helmet_prompt,
        width=1024,
        height=1024,
        num_inference_steps=20
    )

    # 2. AI upscale to 2048×2048 if needed
    if need_high_resolution:
        upscaled_image = upscale_with_esrgan(
            base_image,
            target_size=(2048, 2048)
        )

    return base_image, upscaled_image
```

### Phase 3: Smart Defect Application
```python
def apply_defects_efficiently():
    """Apply defects at appropriate resolution"""

    # Generate defects at 1024×1024 for speed
    damaged_helmet = inpaint_defect(
        clean_image_1024,
        defect_mask,
        defect_prompt
    )

    # Upscale final result if needed
    if need_high_res:
        return upscale_image(damaged_helmet, (2048, 2048))

    return damaged_helmet
```

## Memory Optimization Benefits

### Before (3560×3776px):
- **Single image RAM**: ~203MB
- **Batch generation**: Would exceed 16GB quickly
- **Generation time**: 3-5 minutes each

### After (1024×1024px):
- **Single image RAM**: ~16MB
- **Batch generation**: Can handle 10+ images in memory
- **Generation time**: 30-60 seconds each
- **Total dataset**: Generate 50-100 images in reasonable time

## Quality Validation

### QC Analysis Requirements
- **Defect detection**: 1024×1024 sufficient for AI training
- **Surface detail**: Adequate for crack/impact pattern recognition
- **Hardware inspection**: All mounting points clearly visible
- **Demonstration**: Perfect for notebook demos

### When to Use Higher Resolution
- **Final production models**: Upscale to 2048×2048
- **Print materials**: Keep some originals at full resolution
- **Detail inspection**: Use original 3560×3776 references

## Implementation Strategy

### 1. **Multi-Size Asset Pipeline**
```
assets/helmet_images/
├── downloads/
│   ├── original/                    # 3560×3776 (references)
│   ├── generation_ready/            # 1024×1024 (for SDXL)
│   ├── quick_test/                  # 512×512 (rapid iteration)
│   └── final_output/                # 2048×2048 (upscaled results)
```

### 2. **Memory-Aware Generation**
```python
# Start with smaller sizes, scale up as needed
generation_pipeline = {
    "prototype": 512,      # Fast iteration
    "production": 1024,    # Main generation
    "final": 2048         # Upscaled output
}
```

### 3. **Smart Batch Processing**
```python
# Generate multiple variations efficiently
batch_size = 4  # 4×1024px images = ~64MB vs single 3560px = 203MB
for batch in helmet_variations:
    generate_batch(batch, size=1024)
```

## Recommendation

**Start with 1024×1024px for all AI generation:**
- ✅ **4x faster generation**
- ✅ **12x less memory usage**
- ✅ **SDXL native resolution**
- ✅ **High enough quality for QC demos**
- ✅ **Can upscale winners to 2048px**

This approach gives us speed + quality flexibility while respecting Mac Mini constraints!