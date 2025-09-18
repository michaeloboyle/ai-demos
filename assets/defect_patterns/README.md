# Defect Patterns Database

Physics-based synthetic defect patterns for realistic helmet quality control analysis.

## Purpose

**Primary Demo**: `helmet_qc_demo.ipynb` - Computer Vision Quality Control
**Secondary Demo**: `knowledge_base/` integration for technical specifications

## Contents

- `defect_patterns_database.json` - Master database (15 defect categories)
- Individual JSON files for each defect pattern
- Physics-based modeling parameters
- Material science validation data

## Defect Categories (15 patterns)

### 1. **Impact Damage** (3 patterns)
- `BALLISTIC_IMPACT_V50` - V50 ballistic test pattern
- `BLUNT_TRAUMA_IMPACT` - Drop test simulation
- `FRAGMENT_IMPACT_PATTERN` - Multi-point fragmentation damage

### 2. **Material Degradation** (3 patterns)
- `UV_POLYMER_DEGRADATION` - UV radiation polymer chain scission
- `THERMAL_CYCLING_STRESS` - Temperature cycling microcracking
- `CHEMICAL_EXPOSURE_DEGRADATION` - Chemical agent degradation

### 3. **Manufacturing Defects** (3 patterns)
- `RESIN_VOID_FORMATION` - Air entrapment during molding
- `FIBER_MISALIGNMENT_PATTERN` - Aramid fiber orientation defects
- `DELAMINATION_INTERFACE` - Inter-ply adhesion failure

### 4. **Wear Patterns** (3 patterns)
- `CHIN_STRAP_WEAR_PATTERN` - Contact wear from retention system
- `MOUNTING_RAIL_WEAR` - NVG mount rail abrasion
- `PADDING_COMPRESSION_SET` - Interior foam permanent deformation

### 5. **Environmental Damage** (3 patterns)
- `SALTWATER_CORROSION` - Marine environment metal corrosion
- `DESERT_SAND_ABRASION` - Silica particle erosion damage
- `JUNGLE_HUMIDITY_DEGRADATION` - Tropical moisture effects

## Technical Implementation

### Physics Modeling
Each pattern includes:
- **Damage geometry**: Precise shape, size, and location calculations
- **Material response**: Stress fields, crack propagation, property changes
- **Progression models**: Time-dependent degradation kinetics
- **Validation criteria**: Physics compliance and realism thresholds

### Data Structure
```json
{
  "defect_id": {
    "title": "Descriptive name",
    "defect_type": "Category classification",
    "physics_model": {
      "damage_mechanism": "Scientific description",
      "material_response": "Property changes",
      "progression": "Time-dependent behavior"
    },
    "detectability": "high|medium|low",
    "criticality": "critical|major|minor",
    "inspection_method": "Required detection approach"
  }
}
```

## Usage in Helmet QC Demo

### Defect Overlay Generation
1. **Load reference PURSUIT helmet image**
2. **Select defect pattern** from database
3. **Apply physics-based damage model**
4. **Generate realistic visual overlay**
5. **Validate against detection criteria**

### AI Training Dataset
- **Synthetic defect library**: Realistic training examples
- **Severity variations**: Minor, major, critical levels
- **Multi-angle application**: Apply to different helmet views
- **Quality validation**: >95% realism threshold

## Quality Assurance

**Materials Science Compliance**: ✅
- Validated against engineering literature
- Physics-compliant damage mechanisms
- Realistic material property changes

**Computer Vision Suitability**: ✅
- High detectability for AI training
- Clear visual distinction between categories
- Consistent with real-world failure modes

**Demo Effectiveness**: ✅
- Engaging visual demonstrations
- Realistic manufacturing scenarios
- Professional technical accuracy

## File Size
- **Total**: 80KB (lightweight JSON metadata)
- **Individual patterns**: ~5KB each
- **Master database**: 16KB

## Dependencies
- Used by: `helmet_qc_demo.ipynb`
- Integrated in: `knowledge_base/defect_patterns.json`
- Generated with: `scripts/generate_defect_patterns.py`

Last updated: 2025-09-18