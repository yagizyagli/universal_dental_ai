# Universal Dental AI (`universal_dental_ai`)

A state-of-the-art, open-source Python library for medical-grade dental radiography analysis. Features high-precision 32-teeth instance segmentation, automatic FDI (ISO 3950) numbering, and a robust human-in-the-loop validation framework for doctor-approved clinical PDF reporting.

---

## 👑 Architect & Author
This project is completely designed, developed, and maintained by:
*   **Yağız Yağlı** ([@yagizyagli](https://github.com/yagizyagli))

---

## ⭐ Support the Project
If you find this library useful for your clinical research, medical applications, or AI development, **please give this repository a Star!** It helps the project grow and stay actively maintained.

---

## 🏗️ 32-Tooth Core Architecture Workflow
The library processes raw radiography streams through a strict, decoupled 5-stage pipeline to achieve medical-grade data safety and precision:

```text
 [Raw X-Ray (.dcm)] ──> [1. DentalDicomReader] ──> scrubs and anonymizes patient metadata (GDPR/KVKK)
                                 │
                                 ▼
                    [2. DentalImagePreprocessor] ──> applies localized CLAHE & Bilateral filtering
                                 │
                                 ▼
                     [3. DentalInferenceEngine] ──> runs YOLOv11-Seg & sorts coordinates into 32 FDI teeth
                                 │
                                 ▼
                       [4. Web UI/Schema Verification] ──> locks Doctor Approvals/Overrides (Pydantic)
                                 │
                                 ▼
                     [5. DentalPdfGenerator] ──> compiles signed, locked, hospital-ready PDF reports
```

---

## 🌟 Key Features

*   **Secure DICOM Ingestion:** Automatically scrubs patient identity strings to ensure strict GDPR, HIPAA, and KVKK privacy compliance before any data leaves the local infrastructure.
*   **Clinical Image Enhancement:** Deploys Contrast Limited Adaptive Histogram Equalization (CLAHE) and edge-preserving Bilateral Filters to maximize micro-caries visibility without losing structural boundary precision.
*   **Complete 32-Tooth Mapping:** Algorithmic quadrant sorting that matches neural network bounding coordinates to strict FDI notation (11-48).
*   **Human-in-the-Loop Validation:** Immutable Pydantic schemas built to handle doctor overrides (approve, reject, or manually add findings) directly from web interfaces.
*   **Automated PDF Reporting:** Compiles final clinical assessments into crisp, hospital-ready PDF files featuring alternating-row odontogram data structures and embedded legal disclaimers.

---

## 🛠️ Installation

Clone the repository and install the library in editable development mode:

```bash
git clone https://github.com
cd universal_dental_ai
pip install -r requirements.txt
pip install -e .
```

---

## 🚀 Quick Start (Simulation)

To see the entire pipeline interact from reading an X-ray to compiling a signed doctor report, simply execute the end-to-end example:

```bash
python examples/analyze_and_report.py
```

### Basic Programmatic Usage

```python
from universal_dental_ai.core import DentalDicomReader, DentalImagePreprocessor, DentalInferenceEngine
from universal_dental_ai.reporting import DentalPdfGenerator

# 1. Load and anonymize patient data
reader = DentalDicomReader()
pixels, patient_meta, tech_meta = reader.read_secure_pipeline("path_to_xray.dcm")

# 2. Process image for AI
preprocessor = DentalImagePreprocessor(target_size=(1024, 2048))
ready_image = preprocessor.process_pipeline(pixels)

# 3. Analyze via deep learning engine
engine = DentalInferenceEngine()
ai_findings = engine.predict_radiograph(ready_image)
```

---

## 🧪 Testing

Run the rigorous automated test suite via the root test index to verify image pipeline boundaries, tooth sorting coordinates, and PDF lock states:

```bash
python test_index.py
```

---

## ⚖️ License & Medical Disclaimer

Distributed under the **Apache License 2.0**. This library acts strictly as a clinical decision support tool. Ultimate therapeutic and diagnostic accountability remains entirely with the licensed signing healthcare practitioner.
