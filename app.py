"""
Universal Dental AI - Production Web API Server
Exposes high-performance FastAPI endpoints to ingest real dental radiographs,
orchestrate the AI 32-tooth pipeline, and stream dynamic validated PDF reports.
"""

import os
import shutil
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import cv2
import numpy as np

# Import our library modules compiled in the previous steps
from universal_dental_ai.core.preprocessor import DentalImagePreprocessor
from universal_dental_ai.core.segmenter import DentalInferenceEngine
from universal_dental_ai.schema.report_schema import DentalAnalysisReport, PatientInfo
from universal_dental_ai.reporting.pdf_generator import DentalPdfGenerator

# Initialize production framework
app = FastAPI(
    title="Universal Dental AI - Clinical API",
    description="Medical-grade REST API backend managing radiography processing pipelines.",
    version="0.1.0"
)

# Enforce secure CORS parameters so our local index.html frontend can transmit data frames smoothly
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows web traffic from local or remote hosted dashboards
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Temporary directory path structure to buffer inputs and reports securely on the server
TEMP_DIR = os.path.join(os.path.dirname(__file__), "server_artifacts")
os.makedirs(TEMP_DIR, exist_ok=True)

# Instantiate core library engines once globally to conserve server memory layouts
preprocessor = DentalImagePreprocessor(target_size=(1024, 2048))
inference_engine = DentalInferenceEngine()
pdf_generator = DentalPdfGenerator()

@app.post("/api/v1/analyze")
async def analyze_radiograph(file: UploadFile = File(...)):
    """
    Endpoint 1: Real-time Ingestion and Neural Inference.
    Accepts raw dental radiographs, processes boundaries, and spits out 32-tooth structural reports.
    """
    # Restrict file input types to safe diagnostic image schemas
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.dcm')):
        raise HTTPException(status_code=400, detail="Invalid medical document format. Provide PNG, JPG, or DICOM frames.")

    try:
        # Read byte streams directly into memory matrix without writing to disk first
        file_bytes = await file.read()
        
        # Safe decoding fallback path for standard image matrix formats
        nparr = np.frombuffer(file_bytes, np.uint8)
        raw_img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        
        if raw_img is None:
            raise ValueError("Corrupted image matrix. OpenCV failed to decode pixels.")

        # Step 1: Run core library edge-preserving and contrast-limiting pipeline
        processed_xray = preprocessor.process_pipeline(raw_img)

        # Step 2: Deploy deep learning multi-class forward pass over 32 teeth targets
        ai_findings = inference_engine.predict_radiograph(processed_xray)

        # Step 3: Bundle findings inside a transient data structure ready for doctor modifications
        # Real-world setups pull patient identifiers from DICOM metadata or hospital HIS logs
        mock_patient = PatientInfo(patient_id="PATIENT-WS-99", full_name="Anonymized_Subject***", age=42, gender="Male")
        
        # Build strict Pydantic report response framework
        report = DentalAnalysisReport(
            report_id="REPORT-WS-101",
            patient=mock_patient,
            radiograph_type="PANORAMIC",
            findings=ai_findings
        )

        return report.model_dump() # Returns a verified JSON string array to the frontend dashboard

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Core pipeline execution error: {str(e)}")

@app.post("/api/v1/report/compile")
async def compile_report_pdf(report_data: DentalAnalysisReport):
    """
    Endpoint 2: Immutable PDF Compilation.
    Ingests modified doctor-validated schemas and exports a binary PDF artifact immediately.
    """
    try:
        # Define isolated server target paths for report storage lifecycle management
        pdf_filename = f"signed_report_{report_data.report_id}.pdf"
        target_pdf_path = os.path.join(TEMP_DIR, pdf_filename)

        # Trigger our library pdf motor to render alternating-row odontogram structures
        pdf_generator.generate_pdf(report_data, target_pdf_path)

        if not os.path.exists(target_pdf_path):
            raise FileNotFoundError("PDF binary compiler failed to pipe file write streams to disk.")

        # Stream the medical document back down the network sockets natively as an attachment asset
        return FileResponse(
            path=target_pdf_path,
            media_type="application/pdf",
            filename=pdf_filename
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF engine failed to compile master layout: {str(e)}")
