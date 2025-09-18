# Gentex AI Demo Portfolio - Technical Specification

## Project Overview

**Purpose**: Demonstrate AI/ML integration capabilities for Gentex Corporation IT Applications Principal interview
**Format**: Self-contained Jupyter notebooks showcasing manufacturing-relevant AI solutions
**Timeline**: 6.5 hours total development time
**Delivery**: GitHub repository with 4 interactive notebooks

## Business Context

Gentex Corporation manufactures defense and safety equipment including:
- Ballistic helmets (SPH-4, ACH, PURSUIT)
- Powered Air Purifying Respirators (PAPR)
- Industrial respiratory protection systems
- Military and law enforcement protective gear

**Target Markets**: Global defense forces, law enforcement, emergency responders, industrial personnel

## Technical Architecture

### Platform Requirements
- **Environment**: Jupyter Notebook (.ipynb files)
- **Python Version**: 3.8+
- **Core Dependencies**:
  - `anthropic` (Claude API integration)
  - `opencv-python` (computer vision)
  - `matplotlib` (visualization)
  - `ipywidgets` (interactive components)
  - `pandas` (data manipulation)
  - `PIL/Pillow` (image processing)

### API Requirements
- **Claude API**: Anthropic API key for LLM integration
- **Claude Vision**: Image analysis capabilities
- **Rate Limits**: Standard tier sufficient for demo purposes

## Demo Portfolio Structure

### 1. Portfolio Launcher (`gentex_ai_portfolio.ipynb`)

**Purpose**: Executive dashboard and demo navigation
**Development Time**: 30 minutes

#### Components:
```python
# Executive Summary Section
- Company overview and AI opportunity assessment
- Business value proposition for each demo
- ROI calculator with realistic manufacturing metrics

# Navigation Interface
- Interactive buttons for each demo notebook
- Quick launch capabilities
- Demo sequence guidance

# Metrics Dashboard
- Projected time savings
- Cost reduction estimates
- Implementation timeline
```

#### User Experience:
1. Professional introduction with Gentex context
2. One-click navigation to specific demos
3. Business impact quantification
4. Implementation roadmap visualization

---

### 2. Military Compliance Assistant (`compliance_demo.ipynb`)

**Purpose**: AI-powered compliance validation for defense contracts
**Development Time**: 2 hours
**Priority**: High (leverages existing document processing expertise)

#### Technical Components:

##### Cell 1: Environment Setup
```python
import re, pandas as pd, json
from anthropic import Anthropic
from IPython.display import display, HTML
import ipywidgets as widgets
```

##### Cell 2: Standards Database
```python
# MIL-STD Reference Database
mil_standards = {
    "MIL-DTL-44099": {
        "title": "Ballistic Helmet Requirements",
        "requirements": ["V50 ballistic resistance", "Weight limits", "Retention system"]
    },
    "MIL-STD-662F": {
        "title": "V50 Ballistic Test Protocol",
        "requirements": ["Test setup", "Projectile specifications", "Velocity measurements"]
    },
    "MIL-STD-810G": {
        "title": "Environmental Engineering Considerations",
        "requirements": ["Temperature range", "Humidity resistance", "Shock/vibration"]
    }
}
```

##### Cell 3: Document Parser
```python
def parse_spec_document(spec_text):
    """Extract structured requirements from specification document"""
    # NLP processing using Claude API
    # Return categorized requirements list

def extract_requirements(document):
    """Identify testable requirements and specifications"""
    # Pattern matching for technical specifications
    # Weight, dimensions, performance criteria
```

##### Cell 4: Compliance Engine
```python
def check_compliance(requirements, mil_standards):
    """Validate requirements against MIL-STD database"""
    # Claude API integration for semantic matching
    # Generate compliance matrix with confidence scores

def generate_compliance_report(results):
    """Create formatted compliance assessment"""
    # Traffic light system (Green/Yellow/Red)
    # Detailed non-compliance explanations
```

##### Cell 5: Interactive Demo
```python
# Sample helmet specification document
sample_spec = """
HELMET SPECIFICATION - MODEL ACH-2024
Ballistic Protection: Must defeat 9mm FMJ at 1400 fps ±30 fps
Weight: Maximum 3.5 lbs including all components
Temperature Range: -40°F to +160°F operational
Materials: Aramid fiber composite construction
Retention System: 4-point chinstrap with quick-release
Size Range: Small, Medium, Large, X-Large
Color: Olive Drab Green per Federal Standard 595C
"""

# Interactive compliance checker
compliance_widget = widgets.interactive(...)
```

#### Demo Flow:
1. **Input**: Sample helmet specification document
2. **Processing**: AI extracts 12+ requirements automatically
3. **Analysis**: Cross-reference against MIL-STD database
4. **Output**: Compliance matrix with traffic light indicators
5. **Reporting**: Detailed non-compliance issues with remediation suggestions

#### Business Value:
- **Time Reduction**: Compliance review from 4 hours → 15 minutes
- **Accuracy**: 95% requirement extraction accuracy
- **Risk Mitigation**: Early identification of non-compliance issues
- **Cost Savings**: Reduced rework and contract modifications

---

### 3. Helmet Quality Control Vision AI (`helmet_qc_demo.ipynb`)

**Purpose**: Computer vision for automated defect detection
**Development Time**: 3 hours
**Priority**: High (most visually impressive demo)

#### Technical Components:

##### Cell 1: Computer Vision Setup
```python
import cv2, numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFilter
import base64, io
from anthropic import Anthropic
```

##### Cell 2: Synthetic Data Generation
```python
def create_helmet_images():
    """Generate realistic helmet images with controlled defects"""
    # Base helmet template (circular/oval shape)
    # Simulated defects: scratches, cracks, dents, material flaws
    # Return image set: clean, minor defects, major defects

def add_scratch_defect(image, severity='minor'):
    """Add realistic scratch patterns"""
    # Random scratch location and orientation
    # Varying line thickness and opacity

def add_crack_defect(image, severity='major'):
    """Add crack patterns with realistic propagation"""
    # Branching crack structures
    # Depth variation simulation
```

##### Cell 3: Claude Vision Integration
```python
def analyze_helmet_defects(image_path):
    """Send helmet image to Claude Vision API"""
    # Base64 encoding for API transmission
    # Structured prompt for defect identification
    # Return JSON with defect locations and classifications

def classify_defect_severity(defects):
    """Categorize defects by manufacturing impact"""
    # Pass/Fail/Rework classifications
    # Confidence scoring for each detection
```

##### Cell 4: Visualization Engine
```python
def overlay_defect_detection(original_image, defects):
    """Create visual overlay showing detected defects"""
    # Bounding boxes around defect areas
    # Color coding by severity level
    # Confidence score annotations

def generate_qc_report(image_name, defects, classification):
    """Create formatted quality control report"""
    # Pass/Fail recommendation
    # Defect inventory with locations
    # Suggested remediation actions
```

##### Cell 5: Interactive Demo Interface
```python
# Sample helmet image gallery
helmet_samples = [
    "helmet_clean.jpg",
    "helmet_scratched.jpg",
    "helmet_cracked.jpg",
    "helmet_multiple_defects.jpg"
]

# Upload widget for custom images
upload_widget = widgets.FileUpload(...)

# Real-time analysis display
analysis_output = widgets.Output()
```

#### Demo Flow:
1. **Input**: Display 4 helmet sample images (clean → severely damaged)
2. **Analysis**: AI analyzes each image for defects
3. **Detection**: Visual overlay showing defect locations
4. **Classification**: Severity scoring and pass/fail recommendations
5. **Reporting**: Formatted QC report with actionable recommendations

#### Business Value:
- **Speed**: QC inspection time from 5 minutes → 30 seconds
- **Consistency**: Eliminates human inspector variability
- **Accuracy**: 90%+ defect detection rate
- **Cost Reduction**: 60% reduction in QC labor costs
- **Traceability**: Digital record of all inspections

---

### 4. Field Support Assistant (`field_support_demo.ipynb`)

**Purpose**: AI chatbot for field technician support
**Development Time**: 1 hour
**Priority**: Medium (demonstrates end-user focus)

#### Technical Components:

##### Cell 1: Chatbot Infrastructure
```python
from anthropic import Anthropic
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
import json, datetime
```

##### Cell 2: Equipment Knowledge Base
```python
equipment_kb = {
    "SPH-4": {
        "description": "Aviation helmet system",
        "common_issues": ["Visor cracking", "Chin strap failure", "Comm system issues"],
        "field_repairs": ["Temporary visor replacement", "Emergency strap repair"],
        "part_numbers": ["12345-A", "12345-B", "12345-C"]
    },
    "ACH": {
        "description": "Advanced Combat Helmet",
        "common_issues": ["Strap adjustment", "Padding degradation", "Mount failures"],
        "field_repairs": ["Padding replacement", "Strap readjustment"],
        "part_numbers": ["ACH-001", "ACH-002", "ACH-003"]
    },
    "PAPR": {
        "description": "Powered Air Purifying Respirator",
        "common_issues": ["Filter clogging", "Battery depletion", "Motor failure"],
        "field_repairs": ["Filter replacement", "Battery swap", "Emergency bypass"],
        "part_numbers": ["PAPR-F01", "PAPR-B01", "PAPR-M01"]
    }
}
```

##### Cell 3: Conversation Engine
```python
def field_support_chat(user_input, conversation_history, equipment_context):
    """Process field technician queries with equipment context"""
    # Claude API with equipment-specific knowledge
    # Multi-turn conversation capability
    # Context-aware responses with part numbers and procedures

def extract_equipment_type(query):
    """Identify relevant equipment from user query"""
    # NLP processing to determine equipment type
    # Return relevant knowledge base section
```

##### Cell 4: Interactive Chat Interface
```python
class FieldSupportWidget:
    def __init__(self):
        self.conversation_history = []
        self.chat_output = widgets.Output()
        self.input_text = widgets.Text(placeholder="Ask about equipment issues...")
        self.send_button = widgets.Button(description="Send")

    def send_message(self, sender):
        # Process user input and generate AI response
        # Update conversation display
        # Maintain conversation context
```

##### Cell 5: Pre-Loaded Scenarios
```python
# Realistic field scenarios
demo_scenarios = [
    {
        "title": "Cracked SPH-4 Visor Emergency",
        "query": "SPH-4 visor cracked during mission - need immediate field repair options",
        "expected_response": "Emergency procedures, temporary solutions, part numbers"
    },
    {
        "title": "PAPR Filter Indicator Red",
        "query": "PAPR filter indicator showing red - what's the replacement procedure?",
        "expected_response": "Step-by-step filter replacement, safety protocols"
    },
    {
        "title": "ACH Chin Strap Failure",
        "query": "ACH chin strap broke - field repair options before mission?",
        "expected_response": "Temporary repair methods, safety considerations"
    }
]
```

#### Demo Flow:
1. **Launch**: Interactive chat interface within notebook
2. **Scenarios**: Pre-loaded realistic field problems
3. **Conversation**: Multi-turn dialogue demonstrating context awareness
4. **Solutions**: Practical repair procedures with part numbers
5. **Documentation**: Conversation log with timestamp and equipment references

#### Business Value:
- **24/7 Availability**: No timezone limitations for field support
- **Consistency**: Standardized troubleshooting procedures
- **Speed**: Instant responses vs. waiting for human support
- **Cost Reduction**: Reduced support ticket volume
- **Knowledge Retention**: Capture and share field expertise

---

## Implementation Timeline

### Phase 1: Foundation (Day 1)
- **Hour 1-2**: Portfolio notebook structure and navigation
- **Hour 3-4**: Compliance Assistant core functionality
- **Hour 5-6**: Compliance demo with sample data

### Phase 2: Computer Vision (Day 2)
- **Hour 1-3**: Helmet QC vision system development
- **Hour 4**: Integration with portfolio launcher
- **Hour 5-6**: Testing and refinement

### Phase 3: Chatbot Integration (Day 3)
- **Hour 1**: Field Support Assistant implementation
- **Hour 2**: Final portfolio integration and testing
- **Hour 3**: Documentation and demo preparation

## Technical Requirements

### Development Environment
```bash
# Required Python packages
pip install anthropic opencv-python matplotlib ipywidgets pandas pillow jupyter

# API Configuration
export ANTHROPIC_API_KEY="your_api_key_here"

# Jupyter Extensions
jupyter nbextension enable --py widgetsnbextension
```

### File Structure
```
gentex-ai-demos/
├── gentex_ai_portfolio.ipynb          # Main launcher
├── compliance_demo.ipynb              # Compliance assistant
├── helmet_qc_demo.ipynb               # Quality control vision
├── field_support_demo.ipynb           # Field support chatbot
├── assets/
│   ├── helmet_samples/                # Sample images
│   ├── mil_standards/                 # Standards database
│   └── equipment_manuals/             # KB content
├── requirements.txt                   # Python dependencies
└── README.md                          # Setup instructions
```

## Success Metrics

### Technical Metrics
- **Portfolio Load Time**: < 5 seconds
- **Demo Response Time**: < 3 seconds per interaction
- **API Reliability**: 99%+ uptime during demo
- **Cross-Platform Compatibility**: Windows, macOS, Linux

### Business Metrics
- **Interview Impact**: Demonstrate manufacturing domain understanding
- **AI Capability Showcase**: Practical, implementable solutions
- **Technical Depth**: Production-ready code quality
- **Business Alignment**: Clear ROI and implementation path

## Risk Mitigation

### Technical Risks
- **API Availability**: Offline fallback with cached responses
- **Performance**: Optimized image processing and caching
- **Dependencies**: Minimal external library requirements
- **Compatibility**: Tested across Python 3.8-3.11

### Demo Risks
- **Internet Connectivity**: Local data and cached examples
- **Time Constraints**: Modular demos for flexible presentation
- **Technical Issues**: Backup static screenshots for each demo
- **Audience Engagement**: Interactive elements with clear business value

## Conclusion

This specification outlines a comprehensive AI demo portfolio specifically designed for Gentex Corporation's manufacturing context. The self-contained Jupyter notebook approach ensures:

1. **Professional Presentation**: Clean, interactive demos with clear business value
2. **Technical Credibility**: Production-ready code demonstrating AI/ML expertise
3. **Domain Relevance**: Manufacturing-specific use cases aligned with Gentex's products
4. **Implementation Feasibility**: Realistic development timeline with concrete deliverables

The portfolio demonstrates practical AI integration capabilities while showcasing understanding of Gentex's business challenges and opportunities.