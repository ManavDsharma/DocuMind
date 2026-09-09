import logging

from google import genai
from google.genai import types
from google.genai.errors import ClientError
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from app.core.exceptions import EmbeddingError
from app.ingestion.embedders.base import BaseEmbedder

logger = logging.getLogger(__name__)


def _is_rate_limit_error(exc: BaseException) -> bool:
    """True only for 429 RESOURCE_EXHAUSTED — a transient, retry-worthy error.

    Other 4xx errors (e.g. 400 INVALID_ARGUMENT) mean the request itself
    is malformed, and retrying an identical request will never succeed.
    """
    return isinstance(exc, ClientError) and getattr(exc, "code", None) == 429


class GeminiEmbedder(BaseEmbedder):
    """Generates embeddings using Google's Gemini embedding API.

    Free-tier quota for gemini-embedding-001 is easy to exhaust — in
    practice, a single 100-text batch can consume the entire per-minute
    allowance. We retry on 429 with exponential backoff (long waits,
    since the quota resets on a ~60s window) but fail fast on any other
    4xx, since those indicate a bug in our request, not a transient issue.
    """

    MAX_BATCH_SIZE = 100

    def __init__(self, api_key: str, model: str = "gemini-embedding-001", output_dim: int = 768) -> None:
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._output_dim = output_dim

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        all_embeddings: list[list[float]] = []
        num_batches = (len(texts) + self.MAX_BATCH_SIZE - 1) // self.MAX_BATCH_SIZE

        for batch_num, i in enumerate(range(0, len(texts), self.MAX_BATCH_SIZE), start=1):
            batch = texts[i : i + self.MAX_BATCH_SIZE]
            logger.info(f"Embedding batch {batch_num}/{num_batches}: {len(batch)} texts")
            batch_embeddings = await self._embed_batch(batch)
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        try:
            response = await self._call_gemini(texts)
        except ClientError as exc:
            logger.error(f"Gemini rejected the embedding request: {exc}")
            raise EmbeddingError(f"Embedding request failed for {len(texts)} texts: {exc}") from exc
        except Exception as exc:
            logger.error(f"Unexpected error calling Gemini: {exc}")
            raise EmbeddingError(f"Failed to embed {len(texts)} texts: {exc}") from exc

        if len(response.embeddings) != len(texts):
            raise EmbeddingError(f"Expected {len(texts)} embeddings, got {len(response.embeddings)}")

        return [e.values for e in response.embeddings]

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=15, max=70),
        retry=retry_if_exception(_is_rate_limit_error),
        reraise=True,
    )
    async def _call_gemini(self, texts: list[str]):
        """Raw API call, isolated so tenacity sees the real ClientError type
        (not our wrapped EmbeddingError) when deciding whether to retry."""
        return self._client.models.embed_content(
            model=self._model,
            contents=texts,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=self._output_dim,
            ),
        )