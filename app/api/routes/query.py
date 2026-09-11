from fastapi import APIRouter, Depends

from app.config import settings
from app.db.postgres_repository import PostgresChunkRepository
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.vector_search import VectorSearcher
from app.services.retrieval_service import HybridRetrievalService
from pydantic import BaseModel

router = APIRouter(prefix="/query", tags=["retrieval"])

# Loaded once at import time, not per-request — the model takes time to load
# and is expensive to keep re-instantiating.
_reranker = CrossEncoderReranker()


def get_retrieval_service() -> HybridRetrievalService:
    return HybridRetrievalService(
        vector_searcher=VectorSearcher(qdrant_url=settings.qdrant_url, gemini_api_key=settings.gemini_api_key),
        bm25_repository=PostgresChunkRepository(dsn=settings.postgres_url),
        reranker=_reranker,
    )


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


@router.post("")
async def query_documents(
    request: QueryRequest,
    service: HybridRetrievalService = Depends(get_retrieval_service),
):
    results = service.retrieve(request.query, final_k=request.top_k)
    return {"query": request.query, "results": [r.model_dump() for r in results]}