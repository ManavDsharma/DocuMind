# backfill_postgres.py (run once, from project root)
from qdrant_client import QdrantClient
from app.config import settings
from app.db.postgres_repository import PostgresChunkRepository
from app.ingestion.schemas import DocumentChunk

qdrant = QdrantClient(url=settings.qdrant_url)
pg = PostgresChunkRepository(dsn=settings.postgres_url)
pg.ensure_schema()

points, _ = qdrant.scroll(collection_name="documind_chunks", limit=1000, with_payload=True)
chunks = [
    DocumentChunk(
        chunk_id=p.payload["chunk_id"],
        document_id=p.payload["document_id"],
        document_name=p.payload["document_name"],
        chunk_index=p.payload["chunk_index"],
        text=p.payload["text"],
        page_number=p.payload.get("page_number"),
    )
    for p in points
]
inserted = pg.insert_chunks(chunks)
print(f"Backfilled {inserted} chunks into Postgres")