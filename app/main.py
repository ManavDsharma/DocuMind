from fastapi import FastAPI

from app.api.routes import ingestion
from app.config import settings
from app.core.logging_config import configure_logging
from app.api.routes import query

configure_logging()

app = FastAPI(title="DocuMind API")
app.include_router(ingestion.router)
app.include_router(query.router)


@app.get("/health")
async def health():
    return {"status": "ok", "qdrant_url": settings.qdrant_url}
