"""
Universal Dental AI - Data Schema and Validation Module
Defines rigorous Pydantic data models for patient metadata, AI detections, 
and doctor validation workflows to guarantee state-of-the-art diagnostic integrity.
"""

from datetime import datetime
from typing import List, Optional
from enum import Enum
from pydantic import BaseModel, Field

class FindingStatus(str, Enum):
    """Tracks whether a dental finding is proposed by AI or validated by a doctor."""
    AI_PROPOSED = "AI_PROPOSED"
    DOCTOR_APPROVED = "DOCTOR_APPROVED"
    DOCTOR_REJECTED = "DOCTOR_REJECTED"
    DOCTOR_ADDED = "DOCTOR_ADDED"

class PathologyType(str, Enum):
    """Supported dental pathologies for comprehensive 32-tooth analysis."""
    CARIES = "CARIES"
    PERIAPICAL_LESION = "PERIAPICAL_LESION"
    BONE_LOSS = "BONE_LOSS"
    IMPACTED_TOOTH = "IMPACTED_TOOTH"
    MISSING_TOOTH = "MISSING_TOOTH"

class DentalFinding(BaseModel):
    """Represents a single clinical finding mapped to a specific tooth in the 32-tooth layout."""
    finding_id: str = Field(..., description="Unique tracking identifier for the finding")
    tooth_number: int = Field(..., ge=11, le=48, description="FDI World Dental Federation notation (11-48)")
    pathology: PathologyType = Field(..., description="Type of detected dental pathology")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="AI confidence metrics (0.0 to 1.0)")
    status: FindingStatus = Field(default=FindingStatus.AI_PROPOSED, description="Current status in clinical workflow")
    bounding_box: Optional[List[int]] = Field(default=None, description="Coordinates [xmin, ymin, xmax, ymax] for image overlay")
    doctor_notes: Optional[str] = Field(default=None, description="Optional clinical commentary appended by the reviewer")

class PatientInfo(BaseModel):
    """Demographics and security metadata of the examined individual."""
    patient_id: str = Field(..., description="Anonymized patient unique token")
    full_name: str = Field(..., description="Masked patient name for GDPR/KVKK compliance")
    age: int = Field(..., ge=0, le=120, description="Patient age in years")
    gender: str = Field(..., description="Male, Female, or Other")

class DentalAnalysisReport(BaseModel):
    """The master framework combining patient tracking, AI output logs, and official doctor signatures."""
    report_id: str = Field(..., description="Unique clinical report master identifier")
    patient: PatientInfo
    radiograph_type: str = Field(default="PANORAMIC", description="Type of X-Ray (e.g., PANORAMIC, PERIAPICAL, BITEWING)")
    findings: List[DentalFinding] = Field(default=[], description="List of all mapped dental anomalies")
    is_doctor_approved: bool = Field(default=False, description="Flag indicating if the doctor has locked and signed the report")
    approved_by_doctor_name: Optional[str] = Field(default=None, description="Name of the validating clinical provider")
    diploma_number: Optional[str] = Field(default=None, description="Official medical registration or license code")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    def approve_report(self, doctor_name: str, diploma_no: str):
        """
        Locks the report, signs off the pipeline, and transitions the report into a immutable clinical record.
        """
        self.is_doctor_approved = True
        self.approved_by_doctor_name = doctor_name
        self.diploma_number = diploma_no
        self.updated_at = datetime.utcnow()
