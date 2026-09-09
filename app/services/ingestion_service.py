import logging
from pathlib import Path

from app.db.qdrant_repository import QdrantRepository
from app.ingestion.chunker import RecursiveCharacterChunker
from app.ingestion.embedders.base import BaseEmbedder
from app.ingestion.parsers.base import BasePDFParser
from app.ingestion.schemas import DocumentChunk, EmbeddedChunk

logger = logging.getLogger(__name__)


class IngestionService:
    """Orchestrates: parse PDF -> chunk -> embed -> store.

    Depends on abstractions (BasePDFParser, BaseEmbedder), not concrete
    classes — this is dependency injection. Tests can pass in fake/mock
    implementations without touching real PDFs, Gemini, or Qdrant.
    """

    def __init__(
        self,
        parser: BasePDFParser,
        chunker: RecursiveCharacterChunker,
        embedder: BaseEmbedder,
        repository: QdrantRepository,
    ) -> None:
        self._parser = parser
        self._chunker = chunker
        self._embedder = embedder
        self._repository = repository

    async def ingest_pdf(self, file_path: Path, document_id: str) -> dict:
        document_name = file_path.name
        pages = self._parser.extract_text(file_path)

        chunks: list[DocumentChunk] = []
        for page_number, page_text in pages:
            for i, chunk_text in enumerate(self._chunker.split(page_text)):
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{document_id}_p{page_number}_c{i}",
                        document_id=document_id,
                        document_name=document_name,
                        chunk_index=len(chunks),
                        text=chunk_text,
                        page_number=page_number,
                    )
                )

        logger.info(f"Split '{document_name}' into {len(chunks)} chunks")

        embeddings = await self._embedder.embed_texts([c.text for c in chunks])
        embedded_chunks = [
            EmbeddedChunk(**chunk.model_dump(), embedding=vector)
            for chunk, vector in zip(chunks, embeddings)
        ]

        self._repository.ensure_collection()
        stored_count = self._repository.upsert_chunks(embedded_chunks)

        return {
            "document_id": document_id,
            "document_name": document_name,
            "pages_processed": len(pages),
            "chunks_created": len(chunks),
            "chunks_stored": stored_count,
        }