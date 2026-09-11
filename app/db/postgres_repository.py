import logging

import psycopg
from psycopg.rows import dict_row

from app.core.exceptions import VectorStoreError
from app.ingestion.schemas import DocumentChunk
from app.retrieval.schemas import RetrievedChunk

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS document_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chunk_id TEXT UNIQUE NOT NULL,
    document_id TEXT NOT NULL,
    document_name TEXT NOT NULL,
    chunk_index INT NOT NULL,
    page_number INT,
    text TEXT NOT NULL,
    text_tsv TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
);
CREATE INDEX IF NOT EXISTS idx_document_chunks_tsv ON document_chunks USING GIN (text_tsv);
"""


class PostgresChunkRepository:
    """Stores chunk text in Postgres and serves BM25-style keyword search
    via native full-text search (tsvector/tsquery + GIN index).

    Note: a real production system would manage this schema via a proper
    migration tool (Alembic), not raw DDL on startup — we're keeping it
    simple here since this project doesn't have a schema that evolves
    often, but this is a known simplification worth naming as such.
    """

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn

    def ensure_schema(self) -> None:
        with psycopg.connect(self._dsn) as conn:
            conn.execute(_SCHEMA)
            conn.commit()
        logger.info("Postgres schema ensured (document_chunks table)")

    def insert_chunks(self, chunks: list[DocumentChunk]) -> int:
        if not chunks:
            return 0
        try:
            with psycopg.connect(self._dsn) as conn:
                with conn.cursor() as cur:
                    cur.executemany(
                        """
                        INSERT INTO document_chunks
                            (chunk_id, document_id, document_name, chunk_index, page_number, text)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT (chunk_id) DO NOTHING
                        """,
                        [
                            (c.chunk_id, c.document_id, c.document_name, c.chunk_index, c.page_number, c.text)
                            for c in chunks
                        ],
                    )
                conn.commit()
        except Exception as exc:
            raise VectorStoreError(f"Failed to insert {len(chunks)} chunks into Postgres: {exc}") from exc

        logger.info(f"Inserted {len(chunks)} chunks into Postgres")
        return len(chunks)

    def bm25_search(self, query: str, top_k: int = 20) -> list[RetrievedChunk]:
        sql = """
            SELECT chunk_id, document_id, document_name, page_number, text,
                   ts_rank_cd(text_tsv, plainto_tsquery('english', %s)) AS score
            FROM document_chunks
            WHERE text_tsv @@ plainto_tsquery('english', %s)
            ORDER BY score DESC
            LIMIT %s
        """
        try:
            with psycopg.connect(self._dsn, row_factory=dict_row) as conn:
                rows = conn.execute(sql, (query, query, top_k)).fetchall()
        except Exception as exc:
            raise VectorStoreError(f"BM25 search failed: {exc}") from exc

        return [
            RetrievedChunk(
                chunk_id=r["chunk_id"],
                document_id=r["document_id"],
                document_name=r["document_name"],
                page_number=r["page_number"],
                text=r["text"],
                score=r["score"],
            )
            for r in rows
        ]