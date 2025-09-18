# AI Image Generation Research for PURSUIT Helmet Dataset

## Current Status
- **Reference Quality**: Excellent (13MP professional photos)
- **Available Angles**: Right 3/4 profile, Front view
- **Target**: Generate missing angles + defect variations

## AI Model Options for Helmet Generation

### 1. **DALL-E 3** (Recommended for Variations)
**Capabilities:**
- Excellent object understanding
- High-quality realistic outputs
- Good at maintaining object consistency
- Strong prompt following

**For Our Use:**
- Generate missing viewing angles
- Create defect overlays
- Maintain PURSUIT helmet design consistency

**API Access:** OpenAI API ($0.04/image standard, $0.08/HD)

**Prompts Strategy:**
```
"Professional product photograph of a Gentex PURSUIT flight helmet, carbon fiber shell, from [angle], studio lighting, white background, ultra-high resolution"

"Same PURSUIT helmet with realistic [defect type] damage, showing [specific damage], maintaining professional photography quality"
```

### 2. **Midjourney v6** (Best Quality)
**Capabilities:**
- Exceptional photorealistic quality
- Excellent materials rendering (perfect for carbon fiber)
- Superior lighting and detail

**Limitations:**
- No direct API (requires Discord interface)
- Less precise control over specific elements
- Higher cost per image

**For Our Use:**
- Generate stunning missing angles
- Create ultra-realistic defect variations

### 3. **Stable Diffusion XL + ControlNet** (Best Control)
**Capabilities:**
- Local deployment (privacy)
- Precise control with ControlNet
- Free to run locally
- Inpainting for defect overlays

**ControlNet Options:**
- **Canny**: Edge-guided generation
- **Pose**: For maintaining helmet orientation
- **Depth**: For 3D consistency

**For Our Use:**
- Generate exact angles using depth/pose control
- Precise defect placement via inpainting

### 4. **Adobe Firefly** (Commercial Safe)
**Capabilities:**
- Commercial-use trained
- Good object consistency
- Integrated with Photoshop

**For Our Use:**
- Generate variations for commercial demos
- Inpainting defects in Photoshop

## Recommended Workflow

### Phase 1: Missing Angles Generation
1. **Use DALL-E 3** for initial angle generation
2. **Refine with Stable Diffusion XL** + ControlNet for precision
3. **Target angles**: Left profile, Top view, Rear view, Bottom view

### Phase 2: Defect Generation
1. **Segment helmet components** (shell, visor, hardware)
2. **Apply physics-based defect models** from our patterns
3. **Use inpainting** to overlay defects realistically
4. **Match lighting** to maintain consistency

### Phase 3: Quality Control Dataset
1. **Generate 20-30 variations** per defect type
2. **Multiple severity levels** (minor, major, critical)
3. **Different angles** for each defect
4. **Validation** against real defect references

## Technical Implementation

### DALL-E 3 Integration
```python
import openai

def generate_helmet_angle(angle_description):
    response = openai.images.generate(
        model="dall-e-3",
        prompt=f"Professional product photograph of Gentex PURSUIT flight helmet, carbon fiber shell, {angle_description}, studio lighting, white background, ultra-high resolution",
        size="1024x1024",
        quality="hd",
        n=1
    )
    return response.data[0].url
```

### Stable Diffusion XL Setup
```bash
# Local installation
pip install diffusers transformers
python scripts/generate_helmet_angles.py --reference main_pursuit.png --angle "left_profile"
```

### Defect Overlay Pipeline
1. **Load reference image**
2. **Apply physics-based defect pattern**
3. **Generate realistic damage texture**
4. **Composite with proper lighting**
5. **Validate realism score**

## Cost Analysis

### DALL-E 3 Approach
- Missing angles (5): $0.40
- Defect variations (50): $4.00
- **Total: ~$5 for complete dataset**

### Midjourney Approach
- ~$30/month subscription
- Unlimited generations
- **Best quality, higher cost**

### Stable Diffusion (Local)
- Free after initial setup
- Requires GPU (8GB+ VRAM recommended)
- **Most cost-effective for large datasets**

## Recommended Approach

**For PURSUIT Helmet Project:**
1. **Start with DALL-E 3** - Quick, high-quality results
2. **Use Stable Diffusion XL** for fine-tuning and defects
3. **Combine both** for best results

**Next Steps:**
1. Implement DALL-E 3 integration
2. Generate missing viewing angles
3. Create defect overlay system
4. Build comprehensive QC dataset

## Quality Validation Criteria
- **Resolution**: Minimum 2048x2048 for QC analysis
- **Realism**: >95% photorealistic
- **Consistency**: Matches reference lighting/materials
- **Defect Accuracy**: Physics-compliant damage patterns
- **Detection Suitability**: Clear defect visibility for AI training