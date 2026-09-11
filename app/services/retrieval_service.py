import logging

from app.db.postgres_repository import PostgresChunkRepository
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.schemas import RetrievedChunk
from app.retrieval.vector_search import VectorSearcher

logger = logging.getLogger(__name__)


class HybridRetrievalService:
    """Orchestrates: vector search + BM25 search -> RRF fuse -> rerank."""

    def __init__(
        self,
        vector_searcher: VectorSearcher,
        bm25_repository: PostgresChunkRepository,
        reranker: CrossEncoderReranker,
    ) -> None:
        self._vector_searcher = vector_searcher
        self._bm25_repository = bm25_repository
        self._reranker = reranker

    def retrieve(self, query: str, candidate_k: int = 20, final_k: int = 5) -> list[RetrievedChunk]:
        vector_results = self._vector_searcher.search(query, top_k=candidate_k)
        bm25_results = self._bm25_repository.bm25_search(query, top_k=candidate_k)

        logger.info(f"Vector search: {len(vector_results)} results | BM25 search: {len(bm25_results)} results")

        fused = reciprocal_rank_fusion([vector_results, bm25_results], top_k=candidate_k)
        reranked = self._reranker.rerank(query, fused, top_k=final_k)

        return reranked