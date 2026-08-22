"""
Universal Dental AI - AI Inference Engine Unit Tests
Deep validation of the 32-tooth FDI numbering sorter and neural pipeline integrity.
"""

import pytest
import numpy as np
from universal_dental_ai.core.segmenter import DentalInferenceEngine

def test_inference_engine_32_teeth_mapping():
    """Verifies that the FDI dynamic tooth assignment assigns legal dental codes (11-48)."""
    engine = DentalInferenceEngine()
    dummy_processed = np.zeros((1024, 2048), dtype=np.uint8)
    
    findings = engine.predict_radiograph(dummy_processed)
    
    assert len(findings) > 0
    for finding in findings:
        # Enforce that every tooth identified falls exactly into the global ISO 3950 definition
        assert 11 <= finding.tooth_number <= 48
        # Ensure quadrant numbers 19-20, 29-30, 39-40 don't exist in FDI notation
        assert not (19 <= finding.tooth_number <= 20)
        assert not (29 <= finding.tooth_number <= 30)
        assert not (39 <= finding.tooth_number <= 40)

def test_fdi_sorting_logic_consistency():
    """Ensures spatial coordinate mapping correctly separates maxilla (upper) from mandible (lower)."""
    engine = DentalInferenceEngine()
    
    # Simulate 2 distinct bounding boxes: One clearly on top, one on bottom
    mock_boxes = [,  # Upper left zone
        [200, 750, 250, 850]   # Lower left zone
    ]
    
    fdi_results = engine.generate_fdi_mapping(mock_boxes)
    
    # Box 1 (Y=150) must be mapped to upper jaw (Quadrant 1 or 2 -> tens digit 1 or 2)
    assert fdi_results[0] in range(11, 29)
    # Box 2 (Y=750) must be mapped to lower jaw (Quadrant 3 or 4 -> tens digit 3 or 4)
    assert fdi_results[1] in range(31, 49)
