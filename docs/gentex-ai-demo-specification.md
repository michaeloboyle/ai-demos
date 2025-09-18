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
- **Hardware**: Mac Mini M2 Pro (16GB unified memory)
- **Core Dependencies**:
  - `ollama` (local model serving)
  - `transformers` (model loading)
  - `sentence-transformers` (embeddings)
  - `opencv-python` (computer vision)
  - `matplotlib` (visualization)
  - `ipywidgets` (interactive components)
  - `pandas` (data manipulation)
  - `PIL/Pillow` (image processing)

### Local AI Stack
- **Text Model**: Llama 3.1 8B (4.9GB quantized)
- **Vision Model**: LLaVA 1.6 7B (4.4GB quantized)
- **Embeddings**: all-MiniLM-L6-v2 (384 dimensions)
- **Inference Server**: Ollama for unified model management
- **Memory Management**: Sequential model loading to optimize 16GB RAM

### Performance Specifications
- **Text Generation**: 15-25 tokens/second
- **Vision Analysis**: 3-5 seconds per image
- **Document Processing**: 2-4 seconds per document
- **Peak Memory Usage**: 6-8GB per active model
- **Model Switching**: <10 seconds between tasks

### Technical Architecture Overview

```mermaid
graph TD
    subgraph "Mac Mini M2 Pro (16GB)"
        subgraph "Jupyter Environment"
            PL[Portfolio Launcher<br/>• Executive dashboard<br/>• ROI calculator<br/>• Demo navigation]
            CD[Compliance Demo<br/>• MIL-STD validation<br/>• Document parsing<br/>• Traffic light matrix]
            HQ[Helmet QC Demo<br/>• Computer vision<br/>• Defect detection<br/>• Visual overlays]
            FS[Field Support Demo<br/>• Multi-turn chat<br/>• Equipment KB lookup<br/>• Part number search]
        end

        subgraph "Ollama Service"
            LLM[Llama 3.1 8B<br/>• 4.9GB quantized<br/>• 15-25 tokens/sec<br/>• Document analysis]
            VLM[LLaVA 1.6 7B<br/>• 4.4GB quantized<br/>• 3-5 sec/image<br/>• Defect detection]
            EMB[MiniLM-L6-v2<br/>• 384 dimensions<br/>• Semantic search<br/>• Fast embedding]
        end

        subgraph "Data Layer"
            MS[MIL Standards DB<br/>• MIL-DTL-44099<br/>• MIL-STD-662F<br/>• MIL-STD-810G]
            EK[Equipment KB<br/>• SPH-4 procedures<br/>• ACH troubleshooting<br/>• PAPR manuals]
            HI[Helmet Images<br/>• Clean samples<br/>• Defect patterns<br/>• Synthetic data]
            SC[Sample Content<br/>• Cached AI responses<br/>• Demo screenshots<br/>• Fallback examples]
        end
    end

    PL --> CD
    PL --> HQ
    PL --> FS

    CD --> LLM
    CD --> EMB
    CD --> MS

    HQ --> VLM
    HQ --> LLM
    HQ --> HI

    FS --> LLM
    FS --> EK

    %% Fallback connections for offline/cached content
    CD -.-> SC
    HQ -.-> SC
    FS -.-> SC

    style LLM fill:#e1f5fe
    style VLM fill:#f3e5f5
    style EMB fill:#e8f5e8
    style SC fill:#fff3e0
```

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
import re, pandas as pd, json, os
import ollama
from sentence_transformers import SentenceTransformer
from IPython.display import display, HTML
import ipywidgets as widgets

# External drive asset paths
DEMO_ASSETS = os.environ.get('DEMO_ASSETS', '/Volumes/black box/gentex-demo-assets')
MIL_STANDARDS_PATH = f"{DEMO_ASSETS}/mil_standards"
SAMPLE_CONTENT_PATH = f"{DEMO_ASSETS}/sample_content"
```

##### Cell 2: Standards Database
```python
# Load MIL-STD Reference Database from external drive
def load_mil_standards():
    """Load MIL standards from external drive JSON files"""
    standards = {}
    for std_file in os.listdir(MIL_STANDARDS_PATH):
        if std_file.endswith('.json'):
            with open(f"{MIL_STANDARDS_PATH}/{std_file}", 'r') as f:
                std_name = std_file.replace('.json', '')
                standards[std_name] = json.load(f)
    return standards

# Fallback for offline demo
mil_standards_fallback = {
    "MIL-DTL-44099": {
        "title": "Ballistic Helmet Requirements",
        "requirements": ["V50 ballistic resistance", "Weight limits", "Retention system"]
    },
    "MIL-STD-662F": {
        "title": "V50 Ballistic Test Protocol",
        "requirements": ["Test setup", "Projectile specifications", "Velocity measurements"]
    }
}

# Load from external drive or use fallback
try:
    mil_standards = load_mil_standards()
except:
    mil_standards = mil_standards_fallback
```

##### Cell 3: Document Parser
```python
def parse_spec_document(spec_text):
    """Extract structured requirements from specification document"""
    # NLP processing using local Llama 3.1 model
    # Return categorized requirements list

def extract_requirements(document):
    """Identify testable requirements and specifications"""
    # Pattern matching for technical specifications
    # Weight, dimensions, performance criteria
    response = ollama.chat(model='llama3.1:8b-instruct-q4_K_M',
                          messages=[{'role': 'user', 'content': f'Extract requirements: {document}'}])
```

##### Cell 4: Compliance Engine
```python
def check_compliance(requirements, mil_standards):
    """Validate requirements against MIL-STD database"""
    # Local semantic matching using sentence transformers
    # Generate compliance matrix with confidence scores
    embedder = SentenceTransformer('all-MiniLM-L6-v2')

def generate_compliance_report(results):
    """Create formatted compliance assessment"""
    # Traffic light system (Green/Yellow/Red)
    # Detailed non-compliance explanations using Llama 3.1
    report = ollama.chat(model='llama3.1:8b-instruct-q4_K_M',
                        messages=[{'role': 'user', 'content': f'Generate compliance report: {results}'}])
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

```mermaid
sequenceDiagram
    participant U as User
    participant J as Jupyter Notebook
    participant L as Llama 3.1 8B
    participant E as Embeddings
    participant DB as MIL-STD Database

    U->>J: Upload specification document
    J->>L: Extract requirements from text
    L->>J: Structured requirements list
    J->>E: Generate embeddings for requirements
    E->>J: Requirement vectors
    J->>DB: Search similar MIL standards
    DB->>J: Matching standards
    J->>L: Generate compliance matrix
    L->>J: Traffic light assessment
    J->>U: Display compliance report
```

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
import ollama
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

##### Cell 3: LLaVA Vision Integration
```python
def analyze_helmet_defects(image_path):
    """Send helmet image to local LLaVA model"""
    # Base64 encoding for model transmission
    # Structured prompt for defect identification
    # Return JSON with defect locations and classifications
    with open(image_path, "rb") as img_file:
        img_data = base64.b64encode(img_file.read()).decode()

    response = ollama.chat(model='llava:7b-v1.6-mistral-q4_0',
                          messages=[{'role': 'user',
                                   'content': 'Analyze this helmet for defects',
                                   'images': [img_data]}])

def classify_defect_severity(defects):
    """Categorize defects by manufacturing impact"""
    # Pass/Fail/Rework classifications using local model
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

```mermaid
sequenceDiagram
    participant U as User
    participant J as Jupyter Notebook
    participant V as LLaVA 1.6 7B
    participant L as Llama 3.1 8B
    participant CV as OpenCV

    U->>J: Select helmet image
    J->>CV: Preprocess image (resize, normalize)
    CV->>J: Processed image
    J->>V: Analyze image for defects
    V->>J: Defect locations & descriptions
    J->>L: Classify defect severity
    L->>J: Pass/Fail/Rework classification
    J->>CV: Generate visual overlay
    CV->>J: Annotated image
    J->>U: Display QC report with overlay
```

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
import ollama
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
    # Local Llama 3.1 with equipment-specific knowledge
    # Multi-turn conversation capability
    # Context-aware responses with part numbers and procedures

    messages = [{'role': 'system', 'content': f'Equipment context: {equipment_context}'}]
    messages.extend(conversation_history)
    messages.append({'role': 'user', 'content': user_input})

    response = ollama.chat(model='llama3.1:8b-instruct-q4_K_M', messages=messages)

def extract_equipment_type(query):
    """Identify relevant equipment from user query"""
    # NLP processing using local model to determine equipment type
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

```mermaid
sequenceDiagram
    participant U as User/Technician
    participant J as Jupyter Notebook
    participant L as Llama 3.1 8B
    participant KB as Equipment KB

    U->>J: Enter equipment issue
    J->>KB: Identify equipment type
    KB->>J: Equipment context & procedures
    J->>L: Process query with context
    L->>J: Contextual response
    J->>U: Display solution with part numbers

    Note over U,L: Multi-turn conversation
    U->>J: Follow-up question
    J->>L: Continue conversation with history
    L->>J: Updated response
    J->>U: Refined solution
```

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

```mermaid
gantt
    title Gentex AI Demo Development Timeline
    dateFormat  X
    axisFormat  %L

    section Phase 1: Foundation
    Portfolio Structure         :milestone, m1, 0, 0
    Portfolio Navigation        :p1, 0, 2
    Compliance Core             :p2, 2, 4
    Compliance Demo             :p3, 4, 6
    Phase 1 Complete           :milestone, m2, 6, 6

    section Phase 2: Computer Vision
    Vision System Setup         :cv1, 6, 7
    Helmet Image Generation     :cv2, 7, 8
    LLaVA Integration          :cv3, 8, 9
    QC Demo Interface          :cv4, 9, 10
    Portfolio Integration      :cv5, 10, 11
    Testing & Refinement       :cv6, 11, 12
    Phase 2 Complete           :milestone, m3, 12, 12

    section Phase 3: Chatbot & Final
    Field Support Assistant     :fs1, 12, 13
    Portfolio Integration      :fs2, 13, 14
    Documentation & Prep       :fs3, 14, 15
    Final Testing              :fs4, 15, 16
    Demo Ready                 :milestone, m4, 16, 16
```

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
# CRITICAL: Automated asset setup for external drive
# Run the setup script to handle environment-aware paths
python scripts/setup_external.py

# The script will:
# 1. Detect available storage (internal vs external)
# 2. Set OLLAMA_MODELS and DEMO_ASSETS appropriately
# 3. Create directory structure
# 4. Download models if needed

# Manual setup (if script fails):
export OLLAMA_MODELS="/Volumes/black box/gentex-demo-assets/models"
export DEMO_ASSETS="/Volumes/black box/gentex-demo-assets"

# Install Ollama for model management
brew install ollama

# Download required models to external drive (total ~10GB)
ollama pull llama3.1:8b-instruct-q4_K_M    # 4.9GB text model
ollama pull llava:7b-v1.6-mistral-q4_0     # 4.4GB vision model

# Verify external drive setup
ollama list
ls -la "$DEMO_ASSETS"
df -h "/Volumes/black box"

# Required Python packages
pip install ollama transformers sentence-transformers opencv-python matplotlib ipywidgets pandas pillow jupyter

# Jupyter Extensions
jupyter nbextension enable --py widgetsnbextension

# Start Ollama service (will use external drive models)
ollama serve
```

### Storage Requirements
- **Internal Drive**: Only 19GB available (code only, ~50MB)
- **External Drive**: 5.8TB available on "black box" (all assets ~11GB total)
- **Asset Breakdown**:
  - **AI Models**: ~10GB (Llama 3.1 + LLaVA 1.6)
  - **Helmet Images**: ~500MB (sample datasets)
  - **MIL Standards**: ~100MB (compliance database)
  - **Equipment KB**: ~200MB (manuals and procedures)
  - **Sample Content**: ~50MB (cached responses)
  - **Total External**: ~11GB (well within 5.8TB capacity)

### File Structure & Source Control Strategy

```
gentex-ai-demos/                       # Git repository (internal drive)
├── gentex_ai_portfolio.ipynb          # Main launcher
├── compliance_demo.ipynb              # Compliance assistant
├── helmet_qc_demo.ipynb               # Quality control vision
├── field_support_demo.ipynb           # Field support chatbot
├── requirements.txt                   # Python dependencies
├── README.md                          # Setup instructions
├── .gitignore                         # Exclude large assets
├── config/
│   └── asset_paths.py                 # Configurable asset paths
├── assets/                            # Small assets (Git tracked)
│   ├── sample_data/                   # Tiny demo samples (<1MB each)
│   ├── schemas/                       # JSON schemas and templates
│   └── fallback_content/              # Minimal offline fallbacks
└── scripts/
    ├── download_models.py             # Model download automation
    ├── generate_assets.py             # Asset generation scripts
    └── setup_external.py              # External drive setup

/Volumes/black box/gentex-demo-assets/  # Large assets (NOT in Git)
├── models/                            # Downloaded via script (~10GB)
├── helmet_samples/                    # Generated via script (~500MB)
├── mil_standards/                     # Downloaded/generated (~100MB)
├── equipment_manuals/                 # Generated content (~200MB)
└── sample_content/                    # Cached responses (~50MB)
```

### Source Control Impact Analysis

#### **Git Repository (Tracked):**
- **Size**: ~50-100MB (notebooks + small samples + scripts)
- **Content**: Source code, schemas, generation scripts, tiny samples
- **Portable**: Works across different development environments
- **Collaborative**: Full team access and version history

#### **External Assets (Not Tracked):**
- **Size**: ~11GB (models, images, databases)
- **Content**: Generated/downloaded content via automation scripts
- **Local**: Machine-specific, regenerated as needed
- **Excluded**: .gitignore prevents accidental commits

#### **Asset Management Strategy:**
```python
# config/asset_paths.py - Environment-aware path configuration
import os
import platform

def get_asset_base_path():
    """Get asset path based on environment"""
    if platform.system() == "Darwin":  # macOS
        # Check for external drive first
        if os.path.exists("/Volumes/black box"):
            return "/Volumes/black box/gentex-demo-assets"

    # Fallback to local directory for other systems
    return os.path.join(os.getcwd(), "local_assets")

def setup_asset_directories():
    """Create asset directory structure"""
    base_path = get_asset_base_path()
    directories = [
        "models", "helmet_samples", "mil_standards",
        "equipment_manuals", "sample_content"
    ]
    for directory in directories:
        os.makedirs(os.path.join(base_path, directory), exist_ok=True)
```

### Repository Management Best Practices

#### **Source Control Impact Summary:**

**✅ Benefits of This Approach:**
- **Small Git repo**: ~100MB vs 11GB (keeps GitHub free tier)
- **Fast clones**: Quick repository access for collaboration
- **Portable code**: Works across different environments
- **Automated setup**: Scripts handle asset generation
- **Version controlled logic**: All code and schemas tracked

**⚠️ Trade-offs:**
- **Setup complexity**: Initial asset generation required
- **Asset regeneration**: Large files need to be recreated locally
- **Documentation dependency**: Clear setup instructions critical

**🔧 Implementation Benefits:**
- **Environment detection**: Auto-configures paths based on available storage
- **Graceful fallbacks**: Works without external drive (smaller assets)
- **Collaboration friendly**: Other developers can run setup scripts
- **Demo reliability**: Both live AI and cached fallback content

#### **Recommended .gitignore Configuration:**
```gitignore
# Large AI models and assets (external drive content)
/local_assets/
*.gguf
*.safetensors
*.bin
models/

# Generated content
/assets/generated/
**/sample_content/compliance_reports/
**/sample_content/qc_analyses/
**/sample_content/field_responses/

# Large image datasets
**/helmet_samples/clean/
**/helmet_samples/scratched/
**/helmet_samples/cracked/
**/helmet_samples/multiple_defects/

# Jupyter notebook outputs (optional)
*.ipynb_checkpoints/

# Environment files
.env
.venv/
*.pyc
__pycache__/

# macOS
.DS_Store
```

#### **Development Workflow:**
1. **Initial Setup**: `git clone` → `python scripts/setup_external.py`
2. **Asset Generation**: Scripts download models and generate content
3. **Development**: Work with notebooks, commit code changes only
4. **Collaboration**: Other developers repeat setup process
5. **Demo Preparation**: Verify external drive mounting and asset availability

This hybrid approach keeps the Git repository clean and collaborative while handling the Mac Mini's storage constraints through automated asset management.

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

### Model Memory Management Strategy

```mermaid
flowchart TD
    Start([Demo Launch]) --> Check{Check Current Task}

    Check -->|Compliance| LoadText[Load Llama 3.1 8B<br/>Unload Vision Models]
    Check -->|Vision QC| LoadVision[Load LLaVA 1.6 7B<br/>Unload Text Models]
    Check -->|Field Support| LoadText

    LoadText --> TextReady[Text Model Ready<br/>6GB Memory]
    LoadVision --> VisionReady[Vision Model Ready<br/>6GB Memory]

    TextReady --> ProcessText[Process Text/Chat]
    VisionReady --> ProcessVision[Process Images]

    ProcessText --> TaskComplete{Task Complete?}
    ProcessVision --> TaskComplete

    TaskComplete -->|No| ProcessText
    TaskComplete -->|No| ProcessVision
    TaskComplete -->|Yes| Unload[Unload Current Model]

    Unload --> Check

    style LoadText fill:#e1f5fe
    style LoadVision fill:#f3e5f5
    style TextReady fill:#e1f5fe
    style VisionReady fill:#f3e5f5
```

### Technical Risks
- **Model Loading**: Sequential model management for 16GB RAM constraints
- **Storage Limitations**: Internal drive insufficient - external drive required
- **External Drive Dependency**: Models must be accessible on "black box" drive
- **Asset Availability**: Large assets not in Git - must be generated locally
- **Setup Complexity**: Initial environment requires script execution
- **Performance**: Optimized inference with quantized models on external storage
- **Dependencies**: Local model availability and Ollama service reliability
- **Compatibility**: Tested on M2 Pro with macOS Sonoma

### Demo Risks
- **External Drive Access**: Ensure "black box" drive mounted before demo start
- **Asset Generation Failure**: Fallback content available for offline demos
- **Model Performance**: Pre-warmed models and cached responses for smooth demos
- **Time Constraints**: Modular demos for flexible presentation
- **Technical Issues**: Backup static screenshots and fallback responses
- **Memory Constraints**: Aggressive model unloading and restart procedures
- **Collaboration Setup**: Clear documentation for team member onboarding

## Conclusion

This specification outlines a comprehensive AI demo portfolio specifically designed for Gentex Corporation's manufacturing context. The self-contained Jupyter notebook approach ensures:

1. **Professional Presentation**: Clean, interactive demos with clear business value
2. **Technical Credibility**: Production-ready code demonstrating AI/ML expertise
3. **Domain Relevance**: Manufacturing-specific use cases aligned with Gentex's products
4. **Implementation Feasibility**: Realistic development timeline with concrete deliverables

The portfolio demonstrates practical AI integration capabilities while showcasing understanding of Gentex's business challenges and opportunities.