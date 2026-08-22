"""
Universal Dental AI - Preprocessor Unit Tests
Comprehensive robustness tests for image resizing, CLAHE enhancement, and edge-case handling.
"""

import pytest
import numpy as np
import cv2
from universal_dental_ai.core.preprocessor import DentalImagePreprocessor

def test_preprocessor_pipeline_output_shape():
    """Verifies that the preprocessing pipeline always outputs the exact neural network dimensions."""
    preprocessor = DentalImagePreprocessor(target_size=(1024, 2048))
    dummy_raw = np.random.randint(0, 255, (500, 1200), dtype=np.uint8)
    
    processed = preprocessor.process_pipeline(dummy_raw)
    
    assert processed.shape == (1024, 2048)
    assert processed.dtype == np.uint8

def test_preprocessor_invalid_input_type():
    """Ensures the pipeline gracefully raises an exception when passed faulty data structures."""
    preprocessor = DentalImagePreprocessor()
    with pytest.raises(Exception):
        # Passing a string instead of a numpy matrix should trigger a fail-safe crash
        preprocessor.process_pipeline("invalid_image_path_string")

def test_preprocessor_handles_extreme_aspect_ratios():
    """Verifies that the aspect-ratio preservation with padding works on ultra-narrow images."""
    preprocessor = DentalImagePreprocessor(target_size=(1024, 2048))
    # Simulating a very weirdly cropped radiograph slice (100 width, 800 height)
    narrow_image = np.zeros((800, 100), dtype=np.uint8)
    
    processed = preprocessor.process_pipeline(narrow_image)
    
    assert processed.shape == (1024, 2048)
    # The borders should stay pure black (0) due to safe padding padding
    assert processed[0, 0] == 0
    assert processed[-1, -1] == 0
