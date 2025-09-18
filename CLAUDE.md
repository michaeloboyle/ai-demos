# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains AI demonstration notebooks showcasing practical AI/ML applications in defense manufacturing contexts including quality control, compliance validation, and field support. The project consists of 4 interactive Jupyter notebooks demonstrating production-ready AI solutions.

## Architecture

### Repository Structure
```
ai-demos/
├── defense_ai_portfolio.ipynb         # Executive dashboard and demo launcher
├── compliance_demo.ipynb              # MIL-STD compliance validation system
├── helmet_qc_demo.ipynb               # Computer vision quality control
├── field_support_demo.ipynb           # AI field technician chatbot
├── assets/                            # Supporting data and resources
│   ├── helmet_samples/                # Sample images for QC demo
│   ├── mil_standards/                 # Military standards database
│   └── equipment_manuals/             # Equipment knowledge base
├── docs/                              # Technical specifications
└── requirements.txt                   # Python dependencies
```

### Technical Stack
- **Environment**: Jupyter Notebook (.ipynb files)
- **Python**: 3.8+ required
- **Core AI**: Anthropic Claude API for LLM and vision capabilities
- **Computer Vision**: OpenCV + PIL for image processing
- **Data Science**: NumPy, Pandas, Matplotlib for analysis
- **UI**: ipywidgets for interactive notebook components

### Demo Portfolio Components

1. **Portfolio Launcher** (`defense_ai_portfolio.ipynb`)
   - Executive dashboard with business metrics
   - One-click navigation to specific demos
   - ROI calculator and implementation timeline

2. **Compliance Assistant** (`compliance_demo.ipynb`)
   - AI-powered document analysis against MIL-STD requirements
   - Automated requirement extraction and validation
   - Traffic light compliance matrix with remediation suggestions

3. **Helmet QC Vision System** (`helmet_qc_demo.ipynb`)
   - Computer vision defect detection using Claude Vision API
   - Synthetic image generation with controlled defects
   - Visual overlay and severity classification

4. **Field Support Chatbot** (`field_support_demo.ipynb`)
   - Multi-turn conversation AI for equipment troubleshooting
   - Equipment-specific knowledge base integration
   - Real-time part number and procedure lookup

## Development Commands

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Set up API credentials
export ANTHROPIC_API_KEY="your_api_key_here"

# Enable Jupyter widgets
jupyter nbextension enable --py widgetsnbextension
```

### Running Demos
```bash
# Launch Jupyter environment
jupyter notebook

# Start with portfolio launcher
# Open: defense_ai_portfolio.ipynb

# Run individual demos
# Open: compliance_demo.ipynb
# Open: helmet_qc_demo.ipynb
# Open: field_support_demo.ipynb
```

### Development Workflow
```bash
# Check notebook execution
jupyter nbconvert --execute --to notebook compliance_demo.ipynb
jupyter nbconvert --execute --to notebook helmet_qc_demo.ipynb
jupyter nbconvert --execute --to notebook field_support_demo.ipynb

# Validate requirements
pip check

# Clean notebook outputs (before commits)
jupyter nbconvert --clear-output --inplace *.ipynb
```

## Key Implementation Patterns

### Claude API Integration
All notebooks use consistent patterns for Claude API calls:
- Structured prompts with manufacturing domain context
- JSON response parsing for structured data extraction
- Error handling with fallback to cached responses
- Rate limiting and timeout management

### Interactive Widgets
Standard widget patterns across demos:
- File upload widgets for custom data input
- Output widgets for real-time analysis display
- Button widgets with event handlers for demo flow
- Text widgets for user input and conversation

### Image Processing Pipeline
Computer vision demos follow this flow:
1. Image loading and preprocessing with PIL/OpenCV
2. Base64 encoding for Claude Vision API transmission
3. Structured prompt for defect analysis
4. JSON response parsing for defect locations/classifications
5. Visual overlay generation with matplotlib

### Knowledge Base Architecture
Equipment and standards data structured as:
- Nested dictionaries for hierarchical data
- JSON serialization for API transmission
- Semantic search using Claude for query matching
- Context-aware response generation

## Business Context

This project demonstrates AI integration capabilities for:
- **Quality Control**: 90% reduction in inspection time through automated defect detection
- **Compliance**: 95% faster MIL-STD validation with AI document analysis
- **Field Support**: 24/7 availability with instant troubleshooting responses
- **Cost Savings**: Significant labor reduction across multiple manufacturing processes

Target equipment includes ballistic helmets (SPH-4, ACH, PURSUIT), PAPR systems, and military/law enforcement protective gear.

## API Requirements

- **Anthropic API Key**: Required for all Claude integrations
- **Claude Vision**: Used for image analysis in QC demo
- **Rate Limits**: Standard tier sufficient for demo purposes
- **Offline Fallback**: Cached responses for demo reliability

## File Conventions

- Notebooks use descriptive cell titles and markdown documentation
- Asset files organized by demo type in `/assets/` subdirectories
- Generated content saved to `/assets/generated/` (gitignored)
- API keys stored in environment variables, never committed
- Image files optimized for notebook display and API transmission