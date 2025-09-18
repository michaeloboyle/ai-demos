"""
Test-Driven Development tests for helmet_qc_demo.ipynb

This test file defines the expected behavior and structure of the helmet QC demo notebook
following TDD principles. Tests are written FIRST, then the notebook is implemented to pass them.

SwarmAgent1 - Helmet QC Computer Vision Demo Tests
"""

import unittest
import json
import os
import sys
from pathlib import Path

# Add the project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class TestHelmetQCNotebookStructure(unittest.TestCase):
    """Test the structural requirements of the helmet QC notebook"""

    def setUp(self):
        self.notebook_path = project_root / "helmet_qc_demo.ipynb"
        self.assets_path = project_root / "assets"

    def test_notebook_exists(self):
        """Test that the helmet QC notebook file exists"""
        self.assertTrue(self.notebook_path.exists(),
                       "helmet_qc_demo.ipynb must exist")

    def test_notebook_is_valid_json(self):
        """Test that the notebook is valid JSON format"""
        with open(self.notebook_path, 'r') as f:
            try:
                notebook_data = json.load(f)
                self.assertIsInstance(notebook_data, dict)
            except json.JSONDecodeError:
                self.fail("Notebook must be valid JSON format")

    def test_required_cells_exist(self):
        """Test that all required cells are present in correct order"""
        with open(self.notebook_path, 'r') as f:
            notebook = json.load(f)

        cells = notebook.get('cells', [])
        cell_sources = [' '.join(cell.get('source', [])) for cell in cells]

        required_sections = [
            "# Gentex Helmet Quality Control AI Demo",
            "## System Architecture",
            "## Dependencies and Imports",
            "## Configuration and Setup",
            "## Defect Detection Models",
            "## Image Upload Interface",
            "## Computer Vision Analysis Engine",
            "## Visual Overlay System",
            "## Traffic Light Classification",
            "## Business Metrics Dashboard",
            "## Demo Execution Interface"
        ]

        for section in required_sections:
            found = any(section in source for source in cell_sources)
            self.assertTrue(found, f"Required section '{section}' not found in notebook")

class TestDefectDetectionFunctionality(unittest.TestCase):
    """Test the defect detection core functionality"""

    def setUp(self):
        self.defect_db_path = project_root / "assets" / "defect_patterns" / "defect_database.json"
        self.generated_images_path = project_root / "assets" / "defect_patterns" / "generated"

    def test_defect_database_accessible(self):
        """Test that defect database is accessible and properly formatted"""
        self.assertTrue(self.defect_db_path.exists(),
                       "Defect database must exist")

        with open(self.defect_db_path, 'r') as f:
            defect_db = json.load(f)

        required_keys = ['generated_at', 'total_defects', 'defect_types', 'defects']
        for key in required_keys:
            self.assertIn(key, defect_db, f"Defect database must contain '{key}'")

    def test_generated_images_available(self):
        """Test that generated defect images are available"""
        self.assertTrue(self.generated_images_path.exists(),
                       "Generated images directory must exist")

        # Count PNG files in all subdirectories
        png_files = list(self.generated_images_path.rglob("*.png"))
        self.assertGreaterEqual(len(png_files), 30,
                               "At least 30 generated defect images must be available")

    def test_defect_types_coverage(self):
        """Test that all required defect types are represented"""
        required_defect_types = [
            "ballistic_impact",
            "blunt_force",
            "thermal_damage",
            "surface_wear",
            "environmental"
        ]

        for defect_type in required_defect_types:
            defect_dir = self.generated_images_path / defect_type
            self.assertTrue(defect_dir.exists(),
                           f"Directory for defect type '{defect_type}' must exist")

            images = list(defect_dir.glob("*.png"))
            self.assertGreater(len(images), 0,
                              f"At least one image for defect type '{defect_type}' must exist")

class TestNotebookCodeCells(unittest.TestCase):
    """Test the executable code within notebook cells"""

    def setUp(self):
        self.notebook_path = project_root / "helmet_qc_demo.ipynb"

    def test_imports_cell_contains_required_libraries(self):
        """Test that imports cell includes all necessary libraries"""
        with open(self.notebook_path, 'r') as f:
            notebook = json.load(f)

        # Find the imports cell (look for code cells with imports)
        imports_cell = None
        for cell in notebook.get('cells', []):
            if cell.get('cell_type') == 'code':
                source = ' '.join(cell.get('source', []))
                if 'import numpy' in source and 'import pandas' in source:
                    imports_cell = source
                    break

        self.assertIsNotNone(imports_cell, "Imports cell must exist")

        required_imports = [
            'import numpy',
            'import pandas',
            'import matplotlib',
            'from PIL import',  # Matches "from PIL import Image, ImageDraw, ImageFont"
            'import cv2',
            'import json',
            'import base64',
            'from pathlib import Path',
            'import ipywidgets'
        ]

        for import_stmt in required_imports:
            self.assertIn(import_stmt, imports_cell,
                         f"Required import '{import_stmt}' missing")

    def test_configuration_cell_defines_constants(self):
        """Test that configuration cell defines necessary constants"""
        with open(self.notebook_path, 'r') as f:
            notebook = json.load(f)

        config_cell = None
        for cell in notebook.get('cells', []):
            if cell.get('cell_type') == 'code':
                source = ' '.join(cell.get('source', []))
                if 'ANTHROPIC_API_KEY' in source and 'DEFECT_THRESHOLD' in source:
                    config_cell = source
                    break

        self.assertIsNotNone(config_cell, "Configuration cell must exist")

        required_configs = [
            'ANTHROPIC_API_KEY',
            'DEFECT_THRESHOLD',
            'CONFIDENCE_LEVELS',
            'ASSET_PATH'
        ]

        for config in required_configs:
            self.assertIn(config, config_cell,
                         f"Required configuration '{config}' missing")

class TestBusinessMetrics(unittest.TestCase):
    """Test business metrics and reporting functionality"""

    def test_qc_efficiency_metrics_defined(self):
        """Test that QC efficiency metrics are properly defined"""
        with open(project_root / "helmet_qc_demo.ipynb", 'r') as f:
            notebook = json.load(f)

        # Look for business metrics in any cell
        metrics_found = False
        for cell in notebook.get('cells', []):
            source = ' '.join(cell.get('source', []))
            if '90%' in source and 'reduction' in source and 'inspection time' in source:
                metrics_found = True
                break

        self.assertTrue(metrics_found,
                       "Business metric '90% reduction in inspection time' must be present")

    def test_traffic_light_system_implemented(self):
        """Test that traffic light classification system is implemented"""
        with open(project_root / "helmet_qc_demo.ipynb", 'r') as f:
            notebook = json.load(f)

        traffic_light_found = False
        for cell in notebook.get('cells', []):
            source = ' '.join(cell.get('source', []))
            if all(keyword in source.lower() for keyword in ['pass', 'fail', 'rework']):
                traffic_light_found = True
                break

        self.assertTrue(traffic_light_found,
                       "Traffic light system (Pass/Fail/Rework) must be implemented")

class TestIntegrationRequirements(unittest.TestCase):
    """Test integration with swarm state and external systems"""

    def test_swarm_state_integration(self):
        """Test that notebook includes swarm state tracking"""
        with open(project_root / "helmet_qc_demo.ipynb", 'r') as f:
            notebook = json.load(f)

        swarm_integration = False
        for cell in notebook.get('cells', []):
            source = ' '.join(cell.get('source', []))
            if '.swarm' in source or 'swarm_state' in source:
                swarm_integration = True
                break

        self.assertTrue(swarm_integration,
                       "Swarm state integration must be present")

    def test_offline_fallback_capability(self):
        """Test that offline demo capability is included"""
        with open(project_root / "helmet_qc_demo.ipynb", 'r') as f:
            notebook = json.load(f)

        offline_fallback = False
        for cell in notebook.get('cells', []):
            source = ' '.join(cell.get('source', []))
            if 'offline' in source.lower() or 'fallback' in source.lower():
                offline_fallback = True
                break

        self.assertTrue(offline_fallback,
                       "Offline fallback capability must be included")

class TestAPIIntegration(unittest.TestCase):
    """Test Claude Vision API integration patterns"""

    def test_claude_vision_api_integration(self):
        """Test that Claude Vision API integration is properly implemented"""
        with open(project_root / "helmet_qc_demo.ipynb", 'r') as f:
            notebook = json.load(f)

        api_integration = False
        for cell in notebook.get('cells', []):
            source = ' '.join(cell.get('source', []))
            if 'claude' in source.lower() and ('vision' in source.lower() or 'image' in source.lower()):
                api_integration = True
                break

        self.assertTrue(api_integration,
                       "Claude Vision API integration must be present")

    def test_base64_image_encoding(self):
        """Test that base64 image encoding is implemented for API calls"""
        with open(project_root / "helmet_qc_demo.ipynb", 'r') as f:
            notebook = json.load(f)

        base64_encoding = False
        for cell in notebook.get('cells', []):
            source = ' '.join(cell.get('source', []))
            if 'base64' in source and 'encode' in source:
                base64_encoding = True
                break

        self.assertTrue(base64_encoding,
                       "Base64 image encoding must be implemented")

if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)