import logging
from pathlib import Path

import fitz  # PyMuPDF

from app.core.exceptions import PDFParsingError
from app.ingestion.parsers.base import BasePDFParser

logger = logging.getLogger(__name__)


class PyMuPDFParser(BasePDFParser):
    """Extracts text from digitally-native PDFs using PyMuPDF.

    Note: this does not handle scanned/image-only PDFs — those return
    empty strings per page. OCR fallback (e.g. via Tesseract) is a
    planned v2 addition for that case.
    """

    def extract_text(self, file_path: Path) -> list[tuple[int, str]]:
        if not file_path.exists():
            raise PDFParsingError(f"File not found: {file_path}")

        try:
            doc = fitz.open(file_path)
        except Exception as exc:
            raise PDFParsingError(f"Failed to open PDF {file_path.name}: {exc}") from exc

        pages: list[tuple[int, str]] = []
        try:
            for page_index in range(len(doc)):
                page = doc[page_index]
                text = page.get_text().strip()
                pages.append((page_index + 1, text))
        finally:
            doc.close()

        non_empty_pages = [p for p in pages if p[1]]
        if not non_empty_pages:
            raise PDFParsingError(
                f"No extractable text found in {file_path.name} "
                "(likely a scanned/image-only PDF — OCR not yet implemented)"
            )

        logger.info(f"Extracted text from {len(non_empty_pages)}/{len(pages)} pages in {file_path.name}")
        return pages