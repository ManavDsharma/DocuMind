import logging

logger = logging.getLogger(__name__)


class RecursiveCharacterChunker:
    """Splits text into overlapping chunks using a hierarchy of separators.

    Strategy: try splitting on the largest structural boundary first
    (paragraphs), and only fall back to smaller boundaries (sentences,
    words) if a piece is still too large. This keeps semantically
    related text together far better than naive fixed-size slicing.

    Overlap between consecutive chunks preserves context across chunk
    boundaries — without it, a sentence split across two chunks loses
    meaning in both.
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        separators: list[str] | None = None,
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS

    def split(self, text: str) -> list[str]:
        chunks = self._recursive_split(text, self.separators)
        return [c.strip() for c in chunks if c.strip()]

    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text]

        if not separators:
            # Base case: no separators left, hard-split by size.
            return self._merge_with_overlap(
                [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]
            )

        separator, remaining_separators = separators[0], separators[1:]
        pieces = text.split(separator) if separator else list(text)

        # Recursively split any piece still too large.
        expanded: list[str] = []
        for piece in pieces:
            if len(piece) > self.chunk_size:
                expanded.extend(self._recursive_split(piece, remaining_separators))
            else:
                expanded.append(piece)

        return self._merge_with_overlap(expanded, separator)

    def _merge_with_overlap(self, pieces: list[str], separator: str = "") -> list[str]:
        """Greedily merge small pieces back into chunk_size-sized chunks with overlap."""
        merged: list[str] = []
        current = ""

        for piece in pieces:
            candidate = current + separator + piece if current else piece
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    merged.append(current)
                # Start next chunk with overlap from the end of the previous one.
                overlap_text = current[-self.chunk_overlap :] if current else ""
                current = (overlap_text + separator + piece).strip()
        if current:
            merged.append(current)

        return merged