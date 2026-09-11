import logging

from google import genai
from google.genai import types
from qdrant_client import QdrantClient

from app.retrieval.schemas import RetrievedChunk

logger = logging.getLogger(__name__)


class VectorSearcher:
    """Semantic search over Qdrant using Gemini query embeddings.

    Uses task_type="RETRIEVAL_QUERY" (not RETRIEVAL_DOCUMENT) — Gemini's
    embedding model produces asymmetric embeddings: queries and documents
    are embedded slightly differently to improve retrieval accuracy. Using
    the wrong task_type for either side is a subtle bug that silently
    degrades results without throwing any error.
    """

    def __init__(self, qdrant_url: str, gemini_api_key: str, collection_name: str = "documind_chunks", output_dim: int = 768) -> None:
        self._client = QdrantClient(url=qdrant_url)
        self._genai_client = genai.Client(api_key=gemini_api_key)
        self._collection_name = collection_name
        self._output_dim = output_dim

    def search(self, query: str, top_k: int = 20) -> list[RetrievedChunk]:
        response = self._genai_client.models.embed_content(
            model="gemini-embedding-001",
            contents=[query],
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=self._output_dim,
            ),
        )
        query_vector = response.embeddings[0].values

        results = self._client.query_points(
            collection_name=self._collection_name,
            query=query_vector,
            limit=top_k,
        ).points

        return [
            RetrievedChunk(
                chunk_id=r.payload["chunk_id"],
                document_id=r.payload["document_id"],
                document_name=r.payload["document_name"],
                page_number=r.payload.get("page_number"),
                text=r.payload["text"],
                score=r.score,
            )
            for r in results
        ]