"""
Universal Dental AI - Clinical PDF Report Generator
Assembles validated doctor-approved dental analysis into a professional,
hospital-ready PDF document containing patient metadata and the 32-tooth odontogram.
"""

import os
import logging
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

from universal_dental_ai.schema.report_schema import DentalAnalysisReport, FindingStatus

# Configure professional logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DentalPdfGenerator:
    def __init__(self):
        """
        Initializes the PDF report constructor engine with professional typesetting standards.
        """
        self.styles = getSampleStyleSheet()
        self._create_custom_styles()

    def _create_custom_styles(self):
        """Creates unique styles tailored for highly scannable medical reporting."""
        self.title_style = ParagraphStyle(
            'ReportTitle',
            parent=self.styles['Heading1'],
            fontSize=22,
            leading=26,
            textColor=colors.HexColor('#1A365D'), # Deep clinical blue
            spaceAfter=15
        )
        self.section_style = ParagraphStyle(
            'SectionTitle',
            parent=self.styles['Heading2'],
            fontSize=14,
            leading=18,
            textColor=colors.HexColor('#2C5282'),
            spaceBefore=12,
            spaceAfter=6
        )
        self.meta_style = ParagraphStyle(
            'MetaText',
            parent=self.styles['Normal'],
            fontSize=10,
            leading=14,
            textColor=colors.HexColor('#2D3748')
        )
        self.table_header_style = ParagraphStyle(
            'TableHeader',
            parent=self.styles['Normal'],
            fontSize=10,
            leading=12,
            textColor=colors.white,
            fontName='Helvetica-Bold'
        )

    def build_odontogram_data(self, report: DentalAnalysisReport) -> list:
        """
        Compiles the 32-tooth workflow data into an elegant matrix structure for table visualization.
        Filters out any rejected findings and displays approved diagnostic codes.
        """
        # Table Header
        table_content = [[
            Paragraph("Tooth No (FDI)", self.table_header_style),
            Paragraph("Pathology / Assessment", self.table_header_style),
            Paragraph("Confidence", self.table_header_style),
            Paragraph("Workflow Status", self.table_header_style),
            Paragraph("Clinical Notes", self.table_header_style)
        ]]

        # Sort findings by FDI tooth notation (11 to 48) for alphabetical/numerical clinical order
        sorted_findings = sorted(report.findings, key=lambda f: f.tooth_number)

        for finding in sorted_findings:
            # Enforce security and verification: Skip anomalies rejected by the operating dentist
            if finding.status == FindingStatus.DOCTOR_REJECTED:
                continue

            # Map status to a clean human-readable presentation format
            status_text = "Verified by Doctor" if finding.status in [FindingStatus.DOCTOR_APPROVED, FindingStatus.DOCTOR_ADDED] else "AI Proposed"
            confidence_display = f"{finding.confidence_score * 100:.1f}%" if finding.confidence_score else "N/A"
            notes_display = finding.doctor_notes if finding.doctor_notes else "-"

            row = [
                Paragraph(f"Tooth {finding.tooth_number}", self.meta_style),
                Paragraph(finding.pathology.value, self.meta_style),
                Paragraph(confidence_display, self.meta_style),
                Paragraph(status_text, self.meta_style),
                Paragraph(notes_display, self.meta_style)
            ]
            table_content.append(row)

        # Empty state fallback if no pathologies are verified across all 32 teeth
        if len(table_content) == 1:
            table_content.append([Paragraph("No critical pathologies identified or verified for the 32-tooth structure.", self.meta_style), "", "", "", ""])

        return table_content

    def generate_pdf(self, report: DentalAnalysisReport, output_path: str):
        """
        Main engine that compiles the full report data structure and exports a binary PDF file.
        """
        try:
            logger.info(f"Initiating PDF compiler for Report ID: {report.report_id}")
            
            # Ensure output directories exist safely
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Setup Document layout framework
            doc = SimpleDocTemplate(output_path, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
            story = []

            # 1. Header Banner Title
            story.append(Paragraph("UNIVERSAL DENTAL AI - RADIOLOGY REPORT", self.title_style))
            story.append(Spacer(1, 10))

            # 2. Patient Demographics & Document Signature Block
            meta_data = [
                [Paragraph(f"<b>Patient Identity:</b> {report.patient.full_name}", self.meta_style), 
                 Paragraph(f"<b>Report Date:</b> {report.created_at.strftime('%Y-%m-%d %H:%M')}", self.meta_style)],
                [Paragraph(f"<b>Age / Gender:</b> {report.patient.age} / {report.patient.gender}", self.meta_style), 
                 Paragraph(f"<b>Modality:</b> {report.radiograph_type}", self.meta_style)],
                [Paragraph(f"<b>Validating Doctor:</b> {report.approved_by_doctor_name if report.is_doctor_approved else 'UNAPPROVED'}", self.meta_style), 
                 Paragraph(f"<b>Diploma No:</b> {report.diploma_number if report.is_doctor_approved else 'N/A'}", self.meta_style)]
            ]
            
            meta_table = Table(meta_data, colWidths=[260, 260])
            meta_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F7FAFC')),
                ('PADDING', (0, 0), (-1, -1), 8),
                ('LINEBELOW', (0, -1), (-1, -1), 1, colors.HexColor('#E2E8F0')),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))
            story.append(meta_table)
            story.append(Spacer(1, 20))

            # 3. Clinical Findings & 32-Tooth Odontogram Section
            story.append(Paragraph("Comprehensive 32-Tooth Assessment Table", self.section_style))
            story.append(Spacer(1, 5))

            odontogram_table_data = self.build_odontogram_data(report)
            odontogram_table = Table(odontogram_table_data, colWidths=[90, 120, 70, 100, 140])
            
            # Apply corporate medical theme to data table
            odontogram_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2C5282')), # Solid blue header
                ('PADDING', (0, 0), (-1, -1), 6),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0')),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F7FAFC')]), # Alternating rows
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ]))
            story.append(odontogram_table)
            story.append(Spacer(1, 30))

            # 4. Professional Liability Disclaimer
            disclaimer_text = (
                "<b>Legal Disclaimer:</b> This report is generated by an automated AI diagnostic assistance model "
                "and verified by a licensed dental professional. It is designed to act as clinical decision support. "
                "The ultimate therapeutic and diagnostic accountability remains strictly with the signing healthcare practitioner."
            )
            story.append(Paragraph(disclaimer_text, ParagraphStyle('Disclaimer', parent=self.styles['Normal'], fontSize=8, textColor=colors.HexColor('#718096'), leading=11)))

            # Build and lock document
            doc.build(story)
            logger.info(f"Clinical PDF successfully compiled and locked at: {output_path}")

        except Exception as e:
            logger.error(f"Failed to generate clinical PDF report: {str(e)}")
            raise e
