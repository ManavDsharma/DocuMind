from pydantic import BaseModel


class RetrievedChunk(BaseModel):
    """A chunk returned from any retrieval stage, with its score."""

    chunk_id: str
    document_id: str
    document_name: str
    page_number: int | None
    text: str
    score: float