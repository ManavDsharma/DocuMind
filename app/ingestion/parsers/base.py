from abc import ABC, abstractmethod
from pathlib import Path


class BasePDFParser(ABC):
    """Interface for PDF text extraction strategies.

    Concrete implementations might use PyMuPDF, pdfplumber, or an
    OCR-based pipeline for scanned documents. Callers depend on this
    interface, not a specific library, so the extraction strategy
    can change without touching downstream code.
    """

    @abstractmethod
    def extract_text(self, file_path: Path) -> list[tuple[int, str]]:
        """Extract text from a PDF.

        Returns:
            A list of (page_number, page_text) tuples, 1-indexed.
        """
        raise NotImplementedError