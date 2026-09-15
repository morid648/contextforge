"""Shared test fixtures."""

from pathlib import Path
import pytest
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


def create_pdf_at_path(pdf_path: Path):
    """Generates a multi-page test PDF document with realistic content."""
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(pdf_path), pagesize=letter)

    # Page 1
    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, 720, "Multi-Agent Context Engineering Systems")
    c.setFont("Helvetica", 11)
    c.drawString(72, 690, "Abstract: This research paper evaluates how context engineering coordinates")
    c.drawString(72, 675, "heterogeneous context sources (RAG, memory, web search, external APIs).")
    c.setFont("Helvetica-Bold", 13)
    c.drawString(72, 640, "1. Introduction")
    c.setFont("Helvetica", 11)
    c.drawString(72, 620, "Large Language Models often suffer from parametric hallucination when answering")
    c.drawString(72, 605, "complex analytical questions. Context engineering filters candidate context.")
    c.showPage()

    # Page 2
    c.setFont("Helvetica-Bold", 13)
    c.drawString(72, 720, "2. Key Findings and Methodology")
    c.setFont("Helvetica", 11)
    c.drawString(72, 700, "Our empirical results indicate that parallel context gathering reduces latency")
    c.drawString(72, 685, "by 62 percent compared to sequential execution across 4 distinct data sources.")
    c.drawString(72, 670, "Furthermore, evaluator filtering dropped irrelevant noise with 94 percent accuracy.")
    c.save()


@pytest.fixture
def sample_pdf_path(tmp_path) -> Path:
    """Creates a temporary sample PDF with text and pages."""
    pdf_path = tmp_path / "test_research_paper.pdf"
    create_pdf_at_path(pdf_path)
    return pdf_path
