"""
Universal Dental AI - End-to-End Simulation Example
Demonstrates reading, preprocessing, AI inference over 32 teeth, doctor modification, and PDF generation.
"""

import os
import sys
from datetime import datetime

# Adjust path to access local src repository during development
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from universal_dental_ai.core.preprocessor import DentalImagePreprocessor
from universal_dental_ai.core.segmenter import DentalInferenceEngine
from universal_dental_ai.schema.report_schema import DentalAnalysisReport, PatientInfo, FindingStatus, PathologyType, DentalFinding
from universal_dental_ai.reporting.pdf_generator import DentalPdfGenerator

def main():
    print("[STEP 1] Initializing Universal Dental AI clinical modules...")
    preprocessor = DentalImagePreprocessor(target_size=(1024, 2048))
    inference_engine = DentalInferenceEngine()
    pdf_generator = DentalPdfGenerator()

    # Create dummy matrix simulating an ingested raw panoramic radiograph
    import numpy as np
    raw_xray_image = np.random.randint(50, 200, (800, 1600), dtype=np.uint8)

    print("[STEP 2] Running edge-preserving and contrast-limiting preprocessing...")
    processed_xray = preprocessor.process_pipeline(raw_xray_image)

    print("[STEP 3] Deploying Deep Learning 32-tooth assessment engine...")
    ai_findings = inference_engine.predict_radiograph(processed_xray)

    # Establish initial database report state with mock patient tracking data
    mock_patient = PatientInfo(patient_id="PATIENT-2026-X89", full_name="Patient_78a***", age=34, gender="Female")
    report = DentalAnalysisReport(
        report_id="REPORT-2026-001",
        patient=mock_patient,
        radiograph_type="PANORAMIC",
        findings=ai_findings
    )

    print(f"[STEP 4] Simulating Human-in-the-Loop Web UI for the Dentist...")
    print(f" -> AI suggested {len(report.findings)} pathologies.")

    # DOCTOR WORKFLOW SIMULATION:
    # 1. Doctor reviews and APPROVES the Caries on Tooth 16
    for finding in report.findings:
        if finding.tooth_number == 16:
            finding.status = FindingStatus.DOCTOR_APPROVED
            finding.doctor_notes = "Confirmed severe distal caries. Scheduled for composite restoration."
            print(" -> Doctor Action: Approved AI finding on Tooth 16.")

        # 2. Doctor REJECTS the Impacted Tooth on Tooth 48 (False positive override)
        if finding.tooth_number == 48:
            finding.status = FindingStatus.DOCTOR_REJECTED
            print(" -> Doctor Action: Rejected AI finding on Tooth 48 (Anatomical variant, not pathology).")

    # 3. Doctor manually ADDS a missing tooth finding that AI overlooked (Tooth 24)
    manual_finding = DentalFinding(
        finding_id="doctor_manual_add_24",
        tooth_number=24,
        pathology=PathologyType.MISSING_TOOTH,
        confidence_score=1.0, # Human absolute validation
        status=FindingStatus.DOCTOR_ADDED,
        doctor_notes="Extracted 3 years ago according to patient history."
    )
    report.findings.append(manual_finding)
    print(" -> Doctor Action: Manually added Missing Tooth 24.")

    print("[STEP 5] Finalizing and signing off clinical report...")
    # Doctor signs the digital legal workflow
    report.approve_report(doctor_name="Dr. Elizabeth Blackwell, DDS", diploma_no="DIP-2026-8874")

    # Target directory path for outputting local artifacts securely
    output_pdf_path = os.path.join(os.path.dirname(__file__), "outputs", "final_clinical_report.pdf")
    
    print(f"[STEP 6] Compiling locked binary PDF Report to: {output_pdf_path}")
    pdf_generator.generate_pdf(report, output_pdf_path)
    print("=" * 70)
    print("SUCCESS: End-to-end dental AI pipeline test simulated without errors.")
    print("=" * 70)

if __name__ == "__main__":
    main()
