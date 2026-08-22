import pytest
import numpy as np
from universal_dental_ai.core.preprocessor import DentalImagePreprocessor

def test_preprocessor_pipeline_output_shape():
    """Verifies that the preprocessing pipeline always outputs the exact neural network dimensions."""
    preprocessor = DentalImagePreprocessor(target_size=(1024, 2048))
    # Create a dummy raw image (e.g., 500x500 grayscale)
    dummy_raw = np.random.randint(0, 255, (500, 500), dtype=np.uint8)
    
    processed = preprocessor.process_pipeline(dummy_raw)
    
    assert processed.shape == (1024, 2048), "Preprocessed image dimension mismatch!"
    assert processed.dtype == np.uint8, "Preprocessed image must be 8-bit unsigned integer."
