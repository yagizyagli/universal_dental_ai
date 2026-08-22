"""
Universal Dental AI - Image Preprocessing Module
Optimizes dental radiographs (Panoramic, Periapical) for high-precision deep learning analysis.
"""

import cv2
import numpy as np
import logging

# Configure professional logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DentalImagePreprocessor:
    def __init__(self, target_size: tuple = (1024, 2048)):
        """
        Initializes the preprocessor with standard deep learning input dimensions.
        :param target_size: Tuple of (height, width) for scaling the radiograph.
        """
        self.target_size = target_size
        # CLAHE (Contrast Limited Adaptive Histogram Equalization) for localized contrast enhancement
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Resizes the dental image while maintaining critical anatomical aspect ratios using padding.
        """
        h, w = image.shape[:2]
        th, tw = self.target_size
        
        # Calculate aspect ratio
        scale = min(tw / w, th / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # Create a professional black canvas for padding
        padded = np.zeros((th, tw), dtype=np.uint8) if len(image.shape) == 2 else np.zeros((th, tw, 3), dtype=np.uint8)
        
        # Center the resized image on the canvas
        dx, dy = (tw - new_w) // 2, (th - new_h) // 2
        padded[dy:dy+new_h, dx:dx+new_w] = resized
        
        logger.info(f"Image resized from {w}x{h} to target {tw}x{th} with padding.")
        return padded

    def enhance_contrast(self, grayscale_image: np.ndarray) -> np.ndarray:
        """
        Applies localized adaptive histogram equalization to make micro-caries and bone loss visible.
        """
        if len(grayscale_image.shape) == 3:
            grayscale_image = cv2.cvtColor(grayscale_image, cv2.COLOR_BGR2GRAY)
            
        enhanced = self.clahe.apply(grayscale_image)
        logger.info("Adaptive localized contrast enhancement (CLAHE) applied successfully.")
        return enhanced

    def remove_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Removes radiograph sensor noise without blurring sharp tooth edges (critical for enamel analysis).
        """
        # Bilateral filter preserves sharp enamel-dentin boundaries while smoothing out sensor noise
        denoised = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
        logger.info("Bilateral edge-preserving denoising completed.")
        return denoised

    def process_pipeline(self, raw_image: np.ndarray) -> np.ndarray:
        """
        Executes the full preprocessing pipeline required before feeding data into the AI models.
        """
        try:
            logger.info("Starting dental image preprocessing pipeline...")
            
            # Step 1: Ensure grayscale
            if len(raw_image.shape) == 3:
                raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
                
            # Step 2: Denoise sensor artifacts
            denoised = self.remove_noise(raw_image)
            
            # Step 3: Equalize contrast for pathology clarity
            enhanced = self.enhance_contrast(denoised)
            
            # Step 4: Standardize dimensions for neural network input
            final_image = self.resize_image(enhanced)
            
            logger.info("Preprocessing pipeline executed successfully. Image ready for inference.")
            return final_image
            
        except Exception as e:
            logger.error(f"Error during preprocessing pipeline: {str(e)}")
            raise e
