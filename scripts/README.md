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

### **2. `setup_defect_generation.py`**
**Purpose**: Sets up directories for physics-based defect overlay generation
**Usage**: `python setup_defect_generation.py`
**Creates**: Directory structure for defect overlays on existing angles

**What it sets up**:
- Output directories for defect variations
- Image processing workspace
- Quality validation structure
- Basic dependencies check

**Prerequisites**:
- External drive mounted and writable
- Python packages: `PIL`, `opencv-python`, `numpy`

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

# Set up defect generation directories
python setup_defect_generation.py
```

### **Phase 2: Create Defect Overlays**
```python
# Will be created: generate_defect_overlays.py
# Uses: Physics-based defect patterns + image composition
# Input: PURSUIT reference images (2 angles) + defect patterns database
# Output: 30 realistic defected helmet variations
```

### **Phase 3: Update Knowledge Base**
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