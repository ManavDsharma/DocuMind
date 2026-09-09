class DocuMindError(Exception):
    """Base exception for all DocuMind application errors."""


class PDFParsingError(DocuMindError):
    """Raised when a PDF cannot be parsed or contains no extractable text."""


class EmbeddingError(DocuMindError):
    """Raised when the embedding provider fails to return vectors."""


class VectorStoreError(DocuMindError):
    """Raised when Qdrant read/write operations fail."""