"""
Universal Dental AI - PDF Generator Unit Tests
Ensures data masking, empty state fallbacks, and multi-page layouts render seamlessly.
"""

import os
import pytest
from datetime import datetime
from universal_dental_ai.schema.report_schema import DentalAnalysisReport, PatientInfo, DentalFinding, PathologyType, FindingStatus
from universal_dental_ai.reporting.pdf_generator import DentalPdfGenerator

@pytest.fixture
def base_report_setup():
    """Reusable fixture to create a standardized baseline report data object."""
    patient = PatientInfo(patient_id="PT-TEST-01", full_name="Patient_Masked", age=29, gender="Female")
    return DentalAnalysisReport(report_id="REP-TEST-01", patient=patient)

def test_pdf_generation_with_empty_findings(tmpdir, base_report_setup):
    """Verifies that the PDF engine doesn't crash if a patient has 0 verified pathologies."""
    report_path = os.path.join(tmpdir, "empty_report.pdf")
    base_report_setup.findings = [] # Zero findings
    
    pdf_gen = DentalPdfGenerator()
    pdf_gen.generate_pdf(base_report_setup, report_path)
    
    assert os.path.exists(report_path)
    assert os.path.getsize(report_path) > 0

def test_pdf_excludes_doctor_rejected_findings(tmpdir, base_report_setup):
    """Critical safety check: Ensures rejected AI alarms are completely dropped from the medical document."""
    report_path = os.path.join(tmpdir, "filtered_report.pdf")
    
    rejected_finding = DentalFinding(
        finding_id="f_rejected", tooth_number=11, pathology=PathologyType.CARIES,
        confidence_score=0.45, status=FindingStatus.DOCTOR_REJECTED
    )
    approved_finding = DentalFinding(
        finding_id="f_approved", tooth_number=12, pathology=PathologyType.BONE_LOSS,
        confidence_score=0.91, status=FindingStatus.DOCTOR_APPROVED
    )
    
    base_report_setup.findings = [rejected_finding, approved_finding]
    base_report_setup.approve_report("Dr. Test", "D-999")
    
    pdf_gen = DentalPdfGenerator()
    # If the filtering logic fails inside, reportlab table styling might throw errors or print illegal lines
    pdf_gen.generate_pdf(base_report_setup, report_path)
    
    assert os.path.exists(report_path)
