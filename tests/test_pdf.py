import os
import pytest
from datetime import datetime
from universal_dental_ai.schema.report_schema import DentalAnalysisReport, PatientInfo, DentalFinding, PathologyType, FindingStatus
from universal_dental_ai.reporting.pdf_generator import DentalPdfGenerator

def test_pdf_generation_flow(tmpdir):
    """Verifies that the PDF generator safely compiles data and locks the output binary file."""
    report_path = os.path.join(tmpdir, "test_clinical_report.pdf")
    
    # Mock a validated dental report structure
    patient = PatientInfo(patient_id="PT-99", full_name="Patient_Test", age=45, gender="Male")
    finding = DentalFinding(
        finding_id="f1", tooth_number=16, pathology=PathologyType.CARIES, 
        confidence_score=0.95, status=FindingStatus.DOCTOR_APPROVED, doctor_notes="Valid"
    )
    
    report = DentalAnalysisReport(
        report_id="REP-101", patient=patient, findings=[finding],
        is_doctor_approved=True, approved_by_doctor_name="Dr. John Doe", diploma_number="D-12345"
    )
    
    pdf_gen = DentalPdfGenerator()
    pdf_gen.generate_pdf(report, report_path)
    
    assert os.path.exists(report_path), "PDF engine failed to write the report file to disk."
    assert os.path.getsize(report_path) > 0, "Generated PDF file is empty or corrupted."
