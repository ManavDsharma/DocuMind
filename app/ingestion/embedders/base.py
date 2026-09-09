from abc import ABC, abstractmethod


class BaseEmbedder(ABC):
    """Interface for text embedding providers."""

    @abstractmethod
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns one vector per input, same order."""
        raise NotImplementedError