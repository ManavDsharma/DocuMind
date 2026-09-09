from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """A single chunk of a source document, ready for embedding."""

    chunk_id: str = Field(..., description="Unique ID: {document_id}_{index}")
    document_id: str
    document_name: str
    chunk_index: int
    text: str
    page_number: int | None = None


class EmbeddedChunk(DocumentChunk):
    """A DocumentChunk with its embedding vector attached."""

    embedding: list[float]