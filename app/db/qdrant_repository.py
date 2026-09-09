import logging
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.core.exceptions import VectorStoreError
from app.ingestion.schemas import EmbeddedChunk

logger = logging.getLogger(__name__)

#We use a random UUID as the point ID rather than our own chunk_id string: Qdrant point IDs must be a UUID or unsigned integer — this is a Qdrant-specific constraint, so we keep our own human-readable chunk_id in the payload instead, where it's just metadata with no format restriction
class QdrantRepository:
    """Encapsulates all Qdrant read/write operations.

    Callers never touch the qdrant_client library directly — they go
    through this repository. This means if we ever swap Qdrant for
    another vector DB, only this file changes.
    """

    def __init__(self, url: str, collection_name: str = "documind_chunks", vector_size: int = 768) -> None:
        self._client = QdrantClient(url=url)
        self._collection_name = collection_name
        self._vector_size = vector_size

    def ensure_collection(self) -> None:
        """Create the collection if it doesn't already exist. Idempotent."""
        existing = [c.name for c in self._client.get_collections().collections]
        if self._collection_name in existing:
            logger.info(f"Collection '{self._collection_name}' already exists")
            return

        self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config=VectorParams(size=self._vector_size, distance=Distance.COSINE),
        )
        logger.info(f"Created collection '{self._collection_name}'")

    def upsert_chunks(self, chunks: list[EmbeddedChunk]) -> int:
        if not chunks:
            return 0
        try:
            points = [
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=chunk.embedding,
                    payload={
                        "chunk_id": chunk.chunk_id,
                        "document_id": chunk.document_id,
                        "document_name": chunk.document_name,
                        "chunk_index": chunk.chunk_index,
                        "text": chunk.text,
                        "page_number": chunk.page_number,
                    },
                )
                for chunk in chunks
            ]
            self._client.upsert(collection_name=self._collection_name, points=points)
        except Exception as exc:
            raise VectorStoreError(f"Failed to upsert {len(chunks)} chunks: {exc}") from exc

        logger.info(f"Upserted {len(points)} chunks into '{self._collection_name}'")
        return len(points)