import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File

from app.config import settings
from app.core.exceptions import DocuMindError
from app.db.qdrant_repository import QdrantRepository
from app.ingestion.chunker import RecursiveCharacterChunker
from app.ingestion.embedders.gemini_embedder import GeminiEmbedder
from app.ingestion.parsers.pymupdf_parser import PyMuPDFParser
from app.services.ingestion_service import IngestionService

router = APIRouter(prefix="/ingest", tags=["ingestion"])

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def get_ingestion_service() -> IngestionService:
    """FastAPI dependency: builds an IngestionService with real implementations.

    This is the one place concrete classes get wired together — swap
    any of these for a different implementation without touching the
    route handler or the service logic itself.
    """
    return IngestionService(
        parser=PyMuPDFParser(),
        chunker=RecursiveCharacterChunker(chunk_size=800, chunk_overlap=150),
        embedder=GeminiEmbedder(api_key=settings.gemini_api_key),
        repository=QdrantRepository(url=settings.qdrant_url),
    )


@router.post("")
async def ingest_document(
    file: UploadFile = File(...),
    service: IngestionService = Depends(get_ingestion_service),
):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    document_id = str(uuid.uuid4())
    saved_path = UPLOAD_DIR / f"{document_id}.pdf"

    with saved_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        result = await service.ingest_pdf(saved_path, document_id)
    except DocuMindError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return result