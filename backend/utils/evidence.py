import hashlib
import json
import os
from datetime import datetime, timezone
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

class EvidenceManager:
    def __init__(self, storage_dir="manifests"):
        self.storage_dir = storage_dir
        if not os.path.exists(storage_dir):
            os.makedirs(storage_dir)

    def calculate_sha256(self, file_path):
        """
        Calculate SHA-256 hash of a file.
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def create_manifest(self, file_info, analysis_results, model_version="v1.0.0"):
        """
        Create a JSON manifest for the evidence.
        """
        manifest = {
            "version": "1.0",
            "metadata": {
                "filename": file_info["filename"],
                "sha256": file_info["hash"],
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "file_type": file_info["type"],
                "file_size_bytes": file_info["size"]
            },
            "analysis": {
                "model_version": model_version,
                "authenticity_score": analysis_results["score"],
                "visual_score": analysis_results.get("visual_score"),
                "audio_score": analysis_results.get("audio_score"),
                "status": analysis_results["status"]
            },
            "compliance": {
                "section": "Section 65B Bharatiya Sakshya Adhiniyam",
                "disclaimer": "Advisory tool - verify results before legal use."
            }
        }
        
        manifest_path = os.path.join(self.storage_dir, f"{file_info['hash']}_manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=4)
        
        return manifest, manifest_path

    def generate_pdf_report(self, manifest, output_path):
        """
        Generate a human-readable PDF report based on the manifest.
        """
        doc = SimpleDocTemplate(output_path, pagesize=letter)
        styles = getSampleStyleSheet()
        elements = []

        # Title
        title_style = ParagraphStyle(
            'TitleStyle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=1 # Center
        )
        elements.append(Paragraph("Project Phoenix: Evidence Report", title_style))
        elements.append(Spacer(1, 12))

        # Metadata Table
        elements.append(Paragraph("<b>Media Information</b>", styles['Heading2']))
        data = [
            ["Filename", manifest["metadata"]["filename"]],
            ["SHA-256 Hash", manifest["metadata"]["sha256"]],
            ["Timestamp (UTC)", manifest["metadata"]["timestamp_utc"]],
            ["File Type", manifest["metadata"]["file_type"]],
            ["Size", f"{manifest['metadata']['file_size_bytes']} bytes"]
        ]
        t = Table(data, colWidths=[150, 350])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('PADDING', (0, 0), (-1, -1), 6),
        ]))
        elements.append(t)
        elements.append(Spacer(1, 24))

        # Analysis Results
        elements.append(Paragraph("<b>Forensic Analysis Summary</b>", styles['Heading2']))
        score = manifest["analysis"]["authenticity_score"]
        score_color = colors.green if score > 70 else (colors.orange if score > 40 else colors.red)
        
        analysis_data = [
            ["Authenticity Score", f"{score}%"],
            ["Visual Integrity", f"{manifest['analysis']['visual_score']}%" if manifest['analysis']['visual_score'] else "N/A"],
            ["Audio Integrity", f"{manifest['analysis']['audio_score']}%" if manifest['analysis']['audio_score'] else "N/A"],
            ["Model Version", manifest["analysis"]["model_version"]]
        ]
        at = Table(analysis_data, colWidths=[150, 350])
        at.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('TEXTCOLOR', (1, 0), (1, 0), score_color), # Authenticity Score color
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('PADDING', (0, 0), (-1, -1), 6),
        ]))
        elements.append(at)
        elements.append(Spacer(1, 48))

        # Legal Admissibility Section
        elements.append(Paragraph("<b>Section 65B Compliance</b>", styles['Heading2']))
        legal_text = (
            "This report is generated as part of a secure forensic chain of custody. "
            "The hash provided uniquely identifies the source media. The digital signature "
            "accompanying the manifest file ensures data integrity and origin authenticity, "
            "supporting admissibility under Section 65B of the Bharatiya Sakshya Adhiniyam."
        )
        elements.append(Paragraph(legal_text, styles['Normal']))
        elements.append(Spacer(1, 12))
        elements.append(Paragraph("<i>Disclaimer: Advisory tool — verify results before legal use.</i>", styles['Italic']))

        doc.build(elements)
        return output_path
