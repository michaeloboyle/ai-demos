# Scripts Directory

Essential scripts for data generation and knowledge base management in the Defense Manufacturing AI Demo Portfolio.

## Current Scripts (3 essential)

### **1. `build_knowledge_base.py`**
**Purpose**: Integrates all collected data sources into comprehensive knowledge base
**Usage**: `python build_knowledge_base.py`
**Generates**: `assets/knowledge_base/` - Complete integrated database

**What it does**:
- Loads MIL standards, equipment data, helmet images, defect patterns
- Creates unified knowledge base with cross-references
- Generates individual component files for focused access
- Validates data integrity and completeness

**Used by**: All demonstration notebooks for data access

---

### **2. `download_image_models.py`**
**Purpose**: Downloads Stable Diffusion XL models to external drive for local image generation
**Usage**: `python download_image_models.py`
**Downloads to**: `/Volumes/black box/defense-ai-models/image-generation/`

**What it downloads** (~17GB total):
- Stable Diffusion XL Base (7GB) - Core image generation
- SDXL Inpainting (7GB) - Defect overlay creation
- ControlNet Canny (1.5GB) - Edge-guided generation
- ControlNet Depth (1.5GB) - 3D perspective control

**Prerequisites**:
- External drive "black box" mounted
- 12GB+ available space
- Python packages: `diffusers`, `transformers`, `torch`

---

### **3. `generate_defect_patterns.py`**
**Purpose**: Creates physics-based synthetic defect patterns for helmet QC analysis
**Usage**: `python generate_defect_patterns.py`
**Generates**: `assets/defect_patterns/` - 15 realistic defect models

**Defect categories generated**:
- **Impact damage**: Ballistic, blunt force, fragmentation
- **Material degradation**: UV, thermal, chemical exposure
- **Manufacturing defects**: Voids, misalignment, delamination
- **Wear patterns**: Contact wear, abrasion, compression
- **Environmental damage**: Corrosion, sand abrasion, humidity

**Used by**: `helmet_qc_demo.ipynb` for realistic defect overlay generation

## Removed Scripts (Data Collection Complete ✅)

The following scripts served their purpose and were removed after successful data collection:

- `collect_mil_standards.py` - ✅ 3 government standards collected
- `collect_equipment_data.py` - ✅ 10 equipment records collected
- `collect_helmet_images.py` - ✅ PURSUIT references obtained
- `scrape_gentex_images.py` - ✅ Verification complete
- `download_images.py` - ✅ Images downloaded and stored
- `assess_reference_images.py` - ✅ Quality assessment complete
- `cleanup_data.py` - ✅ Repository cleanup complete
- `cleanup_obsolete_files.py` - ✅ File optimization complete

## Workflow for Image Generation Phase

### **Phase 1: Setup External Models**
```bash
# Ensure external drive is mounted
ls "/Volumes/black box/"

# Download SDXL models (one-time setup)
python download_image_models.py
```

### **Phase 2: Generate Missing Helmet Angles**
```python
# Will be created: generate_helmet_variations.py
# Uses: SDXL + ControlNet to generate left profile, top, rear, bottom views
# Input: PURSUIT reference images (2 × 13.3MP)
# Output: 5 additional viewing angles at 1024×1024
```

### **Phase 3: Create Defect Overlays**
```python
# Will be created: generate_defect_overlays.py
# Uses: SDXL Inpainting + physics-based patterns
# Input: Clean helmet images + defect patterns database
# Output: 45-60 realistic defected helmet variations
```

### **Phase 4: Update Knowledge Base**
```bash
# Integrate generated images into knowledge base
python build_knowledge_base.py
```

## External Dependencies

### **Local AI Models** (Mac Mini M2 Pro)
- **Ollama models**: LLaMA 3.1 8B, LLaVA 1.6 7B (existing)
- **SDXL models**: Downloaded by `download_image_models.py` (~17GB)
- **Memory management**: Sequential loading for 16GB RAM constraint

### **External Drive Storage**
- **Path**: `/Volumes/black box/defense-ai-models/`
- **Required space**: ~17GB for image generation models
- **Organization**: Separate directories for each model type

## File Size Summary
- **Current scripts**: ~52KB total (3 essential files)
- **Generated assets**: ~13MB (after data collection)
- **External models**: ~17GB (image generation capability)

## Quality Assurance

**Data Integrity**: ✅
- All scripts validate input/output data
- Error handling for network and disk operations
- Graceful fallbacks for offline operation

**Mac Mini Optimization**: ✅
- Memory-conscious model loading
- External drive storage management
- Performance monitoring and optimization

**Production Ready**: ✅
- Professional error handling
- Comprehensive logging
- Progress reporting and validation

Last updated: 2025-09-18