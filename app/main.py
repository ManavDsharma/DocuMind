from fastapi import FastAPI

from app.api.routes import ingestion
from app.config import settings
from app.core.logging_config import configure_logging

configure_logging()

app = FastAPI(title="DocuMind API")
app.include_router(ingestion.router)


@app.get("/health")
async def health():
    return {"status": "ok", "qdrant_url": settings.qdrant_url}