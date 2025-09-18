# PURSUIT Helmet Images Database

This directory contains high-quality PURSUIT helmet reference images for AI-powered QC analysis.

## Contents

- `helmet_images_database.json` - Complete image database with metadata
- `downloads/` - Ultra-high resolution PURSUIT helmet images

## PURSUIT Helmet Images

**Model**: Gentex PURSUIT™ Fixed Wing Flight Helmet
**Quality**: Professional studio photography (13.3MP each)
**Format**: PNG with transparency
**Total Size**: 13.2MB

### Image Details

1. **main_pursuit_pdp_gallery_2025__39745.png** (6.5MB)
   - **View**: Right 3/4 profile with visor down
   - **Resolution**: 3560×3732 (13.3MP)
   - **Features**: Carbon fiber shell, communication equipment, mounting hardware

2. **img2_pursuit_pdp_gallery_2025__92469.png** (6.8MB)
   - **View**: Front view with visor down
   - **Resolution**: 3560×3776 (13.4MP)
   - **Features**: Visor system, retention system, front mounting points

## Image Generation Pipeline

These reference images serve as the foundation for:

- **Missing angle generation** (left profile, top, rear, bottom views)
- **Defect overlay creation** (impact, wear, environmental damage)
- **Accessory variations** (visor positions, NVG mounts)
- **QC training dataset** (70-80 total generated images)

## Technical Specifications

- **Source**: Gentex Corporation Product Gallery
- **License**: Commercial Product Catalog
- **QC Suitable**: ✅ Professional quality for defect detection
- **AI Ready**: ✅ Optimized for Stable Diffusion XL generation
- **Resolution Strategy**: 3560px → 1024px (generation) → 2048px (output)

## Usage

Images are ready for:
- Local SDXL image generation and variation
- Physics-based defect pattern overlay
- Computer vision QC system training
- Defense manufacturing AI demonstrations

## Data Permanence

✅ **Self-contained**: No external dependencies
✅ **Offline capable**: Full local access
✅ **Version controlled**: Git-tracked for collaboration
✅ **High quality**: Professional photography standards

Last updated: 2025-09-18 11:25:00