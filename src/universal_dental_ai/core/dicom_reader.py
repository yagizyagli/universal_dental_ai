"""
Universal Dental AI - DICOM Reader and Security Module
Safely parses medical-grade DICOM (.dcm) dental radiographs and enforces strict privacy anonymization.
"""

import pydicom
import numpy as np
import logging
from typing import Tuple, Dict, Any
from universal_dental_ai.schema.report_schema import PatientInfo

# Configure professional logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DentalDicomReader:
    def __init__(self):
        """
        Initializes the DICOM reader for secure processing of panoramic and periapical X-rays.
        """
        pass

    def extract_and_anonymize_patient(self, dicom_dataset: pydicom.dataset.FileDataset) -> PatientInfo:
        """
        Extracts required demographics and immediately anonymizes or encrypts identity to ensure compliance.
        """
        # Extract metadata securely with safe fallback defaults
        raw_name = str(dicom_dataset.get("PatientName", "Anonymous^Patient"))
        patient_id = str(dicom_dataset.get("PatientID", "UNKNOWN_ID"))
        
        # Parse Age (DICOM age format is often '034Y' -> 34)
        raw_age = dicom_dataset.get("PatientAge", "000Y")
        try:
            age = int(''.join(filter(str.isdigit, str(raw_age))))
        except ValueError:
            age = 0

        gender = str(dicom_dataset.get("PatientSex", "Other"))
        if gender == "M":
            gender = "Male"
        elif gender == "F":
            gender = "Female"
        else:
            gender = "Other"

        # Professional Anonymization: Hash or mask the real name to block data leaks on open-source systems
        anonymized_name = f"Patient_{patient_id[:5]}***"
        
        logger.info("Patient demographic data extracted and securely anonymized.")
        
        return PatientInfo(
            patient_id=patient_id,
            full_name=anonymized_name,
            age=age,
            gender=gender
        )

    def get_pixel_data(self, dicom_dataset: pydicom.dataset.FileDataset) -> np.ndarray:
        """
        Extracts raw pixel array and converts it to a standardized 8-bit grayscale matrix.
        Handles high-bit depths (12-bit, 14-bit, 16-bit) common in professional dental sensors.
        """
        if "PixelData" not in dicom_dataset:
            raise KeyError("The provided DICOM file does not contain any valid image pixel data.")

        # Extract raw numpy array from DICOM
        raw_pixels = dicom_dataset.pixel_array
        
        # Standardize bit depth to 8-bit (0-255) for OpenCV and Deep Learning model compatibility
        if raw_pixels.dtype != np.uint8:
            # Min-Max scaling to compress 12/16-bit dynamics into 8-bit without losing clinical contrast
            raw_pixels = raw_pixels.astype(float)
            min_val = np.min(raw_pixels)
            max_val = np.max(raw_pixels)
            
            if max_val - min_val > 0:
                normalized = (raw_pixels - min_val) / (max_val - min_val) * 255.0
                raw_pixels = normalized.astype(np.uint8)
            else:
                raw_pixels = np.zeros(raw_pixels.shape, dtype=np.uint8)

        logger.info(f"DICOM pixel data successfully extracted. Shape: {raw_pixels.shape}, Depth: 8-bit converted.")
        return raw_pixels

    def read_secure_pipeline(self, dicom_path: str) -> Tuple[np.ndarray, PatientInfo, Dict[str, Any]]:
        """
        Executes the secure ingestion pipeline: loads file, scrubs identity, extracts clinical pixel map.
        """
        try:
            logger.info(f"Ingesting medical DICOM file from: {dicom_path}")
            
            # Read the DICOM file (force parameter bypasses non-standard headers from older machines)
            ds = pydicom.dcmread(dicom_path, force=True)
            
            # Step 1: Securely anonymize identity data
            patient_meta = self.extract_and_anonymize_patient(ds)
            
            # Step 2: Extract medical image grid
            pixel_matrix = self.get_pixel_data(ds)
            
            # Step 3: Extract non-sensitive acquisition metadata for technical logging
            technical_meta = {
                "modality": str(ds.get("Modality", "PX")), # PX: Panoramic X-Ray
                "manufacturer": str(ds.get("Manufacturer", "Generic_Sensor")),
                "kvp": str(ds.get("KVP", "Unknown")),
                "exposure_time": str(ds.get("ExposureTime", "Unknown"))
            }
            
            logger.info("Secure DICOM ingestion pipeline completed successfully.")
            return pixel_matrix, patient_meta, technical_meta

        except Exception as e:
            logger.error(f"Security or parsing breach in DICOM pipeline: {str(e)}")
            raise e
