import pytest
import numpy as np
from universal_dental_ai.core.segmenter import DentalInferenceEngine
from universal_dental_ai.schema.report_schema import PathologyType

def test_inference_engine_32_teeth_mapping():
    """Verifies that the FDI dynamic tooth assignment assigns legal dental codes (11-48)."""
    engine = DentalInferenceEngine()
    dummy_processed = np.zeros((1024, 2048), dtype=np.uint8)
    
    findings = engine.predict_radiograph(dummy_processed)
    
    assert len(findings) > 0, "AI Inference failed to generate simulated dental findings."
    for finding in findings:
        assert 11 <= finding.tooth_number <= 48, f"Illegal FDI tooth number detected: {finding.tooth_number}"
        assert finding.confidence_score >= 0.0 and finding.confidence_score <= 1.0
