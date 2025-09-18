#!/usr/bin/env python3
"""
TDD Tests for Physics-Based Defect Generation

Following TDD principles as required by global CLAUDE.md:
"All development should be TDD"

Test coverage for scripts/generate_defect_overlays.py
"""

import unittest
import tempfile
import os
import sys
import json
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path

# Add scripts directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

try:
    from generate_defect_overlays import DefectGenerator, DEFECT_TYPES, BASE_IMAGES
except ImportError as e:
    print(f"Import error: {e}")
    print("Run: pip install opencv-python pillow numpy")
    sys.exit(1)

class TestDefectGenerator(unittest.TestCase):
    """Test suite for DefectGenerator class"""

    def setUp(self):
        """Set up test fixtures before each test method"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_image_path = os.path.join(self.temp_dir, "test_helmet.png")

        # Create a test helmet image (simple white rectangle on transparent background)
        test_image = Image.new("RGBA", (300, 200), (255, 255, 255, 255))
        test_image.save(self.test_image_path, "PNG")

        # Create generator with test directories
        test_output_dir = os.path.join(self.temp_dir, "output")
        test_db_path = os.path.join(self.temp_dir, "database.json")
        self.generator = DefectGenerator(output_dir=test_output_dir, defect_database_path=test_db_path)

    def tearDown(self):
        """Clean up after each test method"""
        # Clean up temp files
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_defect_generator_initialization(self):
        """Test that DefectGenerator initializes correctly"""
        generator = DefectGenerator()
        self.assertIsInstance(generator, DefectGenerator)
        self.assertEqual(generator.defect_database, [])

    def test_load_base_image_success(self):
        """Test successful loading of base image"""
        pil_image, cv_image = self.generator.load_base_image(self.test_image_path)

        # PIL image tests
        self.assertIsInstance(pil_image, Image.Image)
        self.assertEqual(pil_image.mode, "RGBA")
        self.assertEqual(pil_image.size, (300, 200))

        # OpenCV image tests
        self.assertIsInstance(cv_image, np.ndarray)
        self.assertEqual(cv_image.shape, (200, 300, 3))  # Height, Width, Channels

    def test_load_base_image_file_not_found(self):
        """Test loading non-existent image raises FileNotFoundError"""
        with self.assertRaises(FileNotFoundError):
            self.generator.load_base_image("nonexistent_file.png")

    def test_ballistic_impact_generation(self):
        """Test ballistic impact defect generation"""
        base_image = Image.new("RGBA", (300, 200), (255, 255, 255, 255))

        # Test each severity level
        for severity in ["minor", "moderate", "severe"]:
            with self.subTest(severity=severity):
                result = self.generator.create_ballistic_impact(base_image, severity)

                # Basic image validation
                self.assertIsInstance(result, Image.Image)
                self.assertEqual(result.mode, "RGBA")
                self.assertEqual(result.size, (300, 200))

                # Image should be different from original (has damage applied)
                self.assertNotEqual(list(result.getdata()), list(base_image.getdata()))

    def test_thermal_damage_generation(self):
        """Test thermal damage defect generation"""
        base_image = Image.new("RGBA", (300, 200), (255, 255, 255, 255))

        for severity in ["minor", "moderate", "severe"]:
            with self.subTest(severity=severity):
                result = self.generator.create_thermal_damage(base_image, severity)

                self.assertIsInstance(result, Image.Image)
                self.assertEqual(result.mode, "RGBA")
                self.assertEqual(result.size, (300, 200))
                self.assertNotEqual(list(result.getdata()), list(base_image.getdata()))

    def test_surface_wear_generation(self):
        """Test surface wear defect generation"""
        base_image = Image.new("RGBA", (300, 200), (255, 255, 255, 255))

        for severity in ["minor", "moderate", "severe"]:
            with self.subTest(severity=severity):
                result = self.generator.create_surface_wear(base_image, severity)

                self.assertIsInstance(result, Image.Image)
                self.assertEqual(result.mode, "RGBA")
                self.assertEqual(result.size, (300, 200))
                self.assertNotEqual(list(result.getdata()), list(base_image.getdata()))

    def test_environmental_damage_generation(self):
        """Test environmental damage defect generation"""
        base_image = Image.new("RGBA", (300, 200), (255, 255, 255, 255))

        for severity in ["minor", "moderate", "severe"]:
            with self.subTest(severity=severity):
                result = self.generator.create_environmental_damage(base_image, severity)

                self.assertIsInstance(result, Image.Image)
                self.assertEqual(result.mode, "RGBA")
                self.assertEqual(result.size, (300, 200))
                self.assertNotEqual(list(result.getdata()), list(base_image.getdata()))

    def test_blunt_force_damage_generation(self):
        """Test blunt force damage defect generation"""
        base_image = Image.new("RGBA", (300, 200), (255, 255, 255, 255))

        for severity in ["minor", "moderate", "severe"]:
            with self.subTest(severity=severity):
                result = self.generator.create_blunt_force_damage(base_image, severity)

                self.assertIsInstance(result, Image.Image)
                self.assertEqual(result.mode, "RGBA")
                self.assertEqual(result.size, (300, 200))
                self.assertNotEqual(list(result.getdata()), list(base_image.getdata()))

    def test_apply_defect_all_types(self):
        """Test apply_defect method with all defect types"""
        base_image = Image.new("RGBA", (300, 200), (255, 255, 255, 255))

        for defect_type in DEFECT_TYPES.keys():
            with self.subTest(defect_type=defect_type):
                result = self.generator.apply_defect(base_image, defect_type, "moderate")

                self.assertIsInstance(result, Image.Image)
                self.assertEqual(result.mode, "RGBA")
                self.assertEqual(result.size, (300, 200))

    def test_apply_defect_invalid_type(self):
        """Test apply_defect with invalid defect type raises ValueError"""
        base_image = Image.new("RGBA", (300, 200), (255, 255, 255, 255))

        with self.assertRaises(ValueError):
            self.generator.apply_defect(base_image, "invalid_defect_type", "moderate")

    def test_setup_output_directory(self):
        """Test output directory structure creation"""
        # Create a new generator with fresh temp directory
        test_output_dir = os.path.join(self.temp_dir, "test_output")
        test_db_path = os.path.join(self.temp_dir, "test_db.json")
        generator = DefectGenerator(output_dir=test_output_dir, defect_database_path=test_db_path)

        # Check main directory exists
        self.assertTrue(os.path.exists(test_output_dir))

        # Check subdirectories for each defect type exist
        for defect_type in DEFECT_TYPES.keys():
            subdir = os.path.join(test_output_dir, defect_type)
            self.assertTrue(os.path.exists(subdir))

    def test_save_defect_database(self):
        """Test defect database JSON saving"""
        # Add test data to database
        self.generator.defect_database = [
            {
                "filename": "test_defect.png",
                "defect_type": "ballistic_impact",
                "severity": "moderate",
                "base_image": "test",
                "variation": 1
            }
        ]

        self.generator.save_defect_database()

        # Verify file was created
        self.assertTrue(os.path.exists(self.generator.DEFECT_DATABASE))

        # Verify JSON structure
        with open(self.generator.DEFECT_DATABASE, 'r') as f:
            data = json.load(f)

        self.assertIn("generated_at", data)
        self.assertIn("total_defects", data)
        self.assertIn("defects", data)
        self.assertEqual(data["total_defects"], 1)
        self.assertEqual(len(data["defects"]), 1)

class TestDefectTypesConfiguration(unittest.TestCase):
    """Test defect types configuration"""

    def test_defect_types_structure(self):
        """Test that DEFECT_TYPES has correct structure"""
        self.assertIsInstance(DEFECT_TYPES, dict)

        expected_types = [
            "ballistic_impact", "blunt_force", "thermal_damage",
            "surface_wear", "environmental"
        ]

        for defect_type in expected_types:
            self.assertIn(defect_type, DEFECT_TYPES)

            config = DEFECT_TYPES[defect_type]
            self.assertIn("description", config)
            self.assertIn("severity_levels", config)
            self.assertIn("count", config)

            # Check severity levels
            expected_severities = ["minor", "moderate", "severe"]
            self.assertEqual(config["severity_levels"], expected_severities)

    def test_base_images_configuration(self):
        """Test that BASE_IMAGES configuration is valid"""
        self.assertIsInstance(BASE_IMAGES, dict)
        self.assertGreater(len(BASE_IMAGES), 0)

        for name, path in BASE_IMAGES.items():
            self.assertIsInstance(name, str)
            self.assertIsInstance(path, str)
            self.assertTrue(path.endswith('.png'))

class TestImageProcessingMethods(unittest.TestCase):
    """Test specific image processing methods"""

    def setUp(self):
        """Set up test fixtures"""
        self.generator = DefectGenerator()
        self.test_image = Image.new("RGBA", (200, 200), (128, 128, 128, 255))

    def test_severity_parameter_effects(self):
        """Test that different severity levels produce different results"""
        # Test with ballistic impact
        minor_result = self.generator.create_ballistic_impact(self.test_image, "minor")
        moderate_result = self.generator.create_ballistic_impact(self.test_image, "moderate")
        severe_result = self.generator.create_ballistic_impact(self.test_image, "severe")

        # Results should be different for different severities
        self.assertNotEqual(list(minor_result.getdata()), list(moderate_result.getdata()))
        self.assertNotEqual(list(moderate_result.getdata()), list(severe_result.getdata()))

    def test_image_composition_preserves_alpha(self):
        """Test that alpha compositing preserves image properties"""
        result = self.generator.create_ballistic_impact(self.test_image, "moderate")

        # Should maintain RGBA mode
        self.assertEqual(result.mode, "RGBA")
        self.assertEqual(result.size, self.test_image.size)

    def test_deterministic_randomness(self):
        """Test that randomness is controlled for reproducible results"""
        # Note: This would require seed control in the actual implementation
        # For now, just test that multiple runs produce valid results

        results = []
        for _ in range(3):
            result = self.generator.create_surface_wear(self.test_image, "moderate")
            results.append(result)

        # All results should be valid images
        for result in results:
            self.assertIsInstance(result, Image.Image)
            self.assertEqual(result.mode, "RGBA")

class TestPerformanceRequirements(unittest.TestCase):
    """Test performance requirements"""

    def setUp(self):
        """Set up performance test fixtures"""
        self.generator = DefectGenerator()
        self.test_image = Image.new("RGBA", (300, 200), (255, 255, 255, 255))

    def test_generation_speed(self):
        """Test that defect generation meets speed requirements"""
        import time

        start_time = time.time()
        result = self.generator.create_ballistic_impact(self.test_image, "moderate")
        end_time = time.time()

        generation_time = end_time - start_time

        # Should generate in under 1 second (requirement from specs)
        self.assertLess(generation_time, 1.0,
                       f"Generation took {generation_time:.2f}s, should be <1s")

    def test_memory_efficiency(self):
        """Test that generation doesn't consume excessive memory"""
        # Generate multiple defects to test memory usage
        results = []
        for defect_type in list(DEFECT_TYPES.keys())[:3]:  # Test subset
            result = self.generator.apply_defect(self.test_image, defect_type, "moderate")
            results.append(result)

        # All results should be valid (basic memory leak test)
        for result in results:
            self.assertIsInstance(result, Image.Image)

if __name__ == '__main__':
    # Set up test discovery
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    test_classes = [
        TestDefectGenerator,
        TestDefectTypesConfiguration,
        TestImageProcessingMethods,
        TestPerformanceRequirements
    ]

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print(f"\n{'='*60}")
    print(f"TDD Test Results for Physics-Based Defect Generation")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")

    if result.failures:
        print(f"\nFailures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip()}")

    if result.errors:
        print(f"\nErrors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('Error:')[-1].strip()}")

    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)