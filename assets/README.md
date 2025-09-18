# Assets Directory

This directory contains all data assets for the Defense Manufacturing AI Demo Portfolio.

## Directory Structure

```
assets/
├── defect_patterns/         # Physics-based defect models for QC demo
├── equipment_manuals/       # Equipment database for field support demo
├── helmet_images/           # PURSUIT helmet references for QC demo
├── knowledge_base/          # Integrated database for all demos
└── mil_standards/           # Government standards for compliance demo
```

## Demonstration Mapping

### 1. **Portfolio Launcher** (`defense_ai_portfolio.ipynb`)
**Uses**: `knowledge_base/` (metrics and overview data)
- Business metrics and ROI calculations
- Demo navigation and executive summary

### 2. **Military Compliance Assistant** (`compliance_demo.ipynb`)
**Uses**: `mil_standards/` + `knowledge_base/`
- Government standards validation
- MIL-STD compliance checking
- Regulatory requirement analysis

### 3. **Helmet QC Vision System** (`helmet_qc_demo.ipynb`)
**Uses**: `helmet_images/` + `defect_patterns/` + `knowledge_base/`
- PURSUIT helmet computer vision analysis
- Physics-based defect detection
- Quality control automation

### 4. **Field Support Chatbot** (`field_support_demo.ipynb`)
**Uses**: `equipment_manuals/` + `knowledge_base/`
- Equipment troubleshooting database
- Multi-turn conversation support
- Technical procedure lookup

## Data Sources

All assets are derived from **real, verified sources**:
- **Government APIs**: ASSIST, GSA, NASA (standards)
- **Official databases**: NTSB, FAA (equipment data)
- **Professional photography**: Gentex Corporation (helmet images)
- **Materials science**: Physics-based modeling (defect patterns)

## Total Size
- **Git repository**: ~13MB (high-quality references and metadata)
- **External models**: ~17GB (SDXL image generation on external drive)
- **Generated dataset**: ~100MB (when image generation complete)

## Quality Assurance
- ✅ All data sources verified and curl-tested
- ✅ Professional-grade photography (13.3MP PURSUIT helmets)
- ✅ Physics-compliant defect modeling
- ✅ Official government standard compliance

Last updated: 2025-09-18