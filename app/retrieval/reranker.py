import logging

from sentence_transformers import CrossEncoder

from app.retrieval.schemas import RetrievedChunk

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Reranks candidates by scoring (query, chunk) pairs jointly.

    Unlike vector search (which compares independently pre-computed
    embeddings), a cross-encoder feeds the query and candidate text
    together through the model — more accurate, much slower, hence
    only applied to a small candidate set (~20), not the whole corpus.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        logger.info(f"Loading cross-encoder model: {model_name}")
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[RetrievedChunk], top_k: int = 5) -> list[RetrievedChunk]:
        if not candidates:
            return []

        pairs = [(query, c.text) for c in candidates]
        scores = self._model.predict(pairs)

        reranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)

        return [
            chunk.model_copy(update={"score": float(score)})
            for chunk, score in reranked[:top_k]
        ]