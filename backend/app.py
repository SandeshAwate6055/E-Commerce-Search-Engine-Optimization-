"""
FastAPI application entry point.

Run with:
    cd d:/Ecommerce_search_engine
    .venv_gpu/Scripts/uvicorn backend.app:app --host 0.0.0.0 --port 8000
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.config import settings
from backend.routes import health_router, search_router
from backend.services.model_loader import load_all

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


# ── Lifespan (startup/shutdown) ───────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — loading models and indexes...")
    load_all()
    logger.info("Startup complete. API is ready.")
    yield
    logger.info("Shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Multimodal E-Commerce Search API",
    description=(
        "End-to-end multimodal product search using "
        "Qwen LLM, Qwen Vision, CLIP, and FAISS.\n\n"
        "**Endpoints:**\n"
        "- `GET  /health`           — system status\n"
        "- `POST /search/text`      — text-only search\n"
        "- `POST /search/image`     — image-only search\n"
        "- `POST /search/multimodal`— text + image search\n"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS (permissive for local dev — restrict in production) ──────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(health_router)
app.include_router(search_router)

# ── Static files — serve product images ──────────────────────────────────────
from pathlib import Path as _Path
_images_dir = _Path(settings.IMAGES_DIR)
if _images_dir.exists():
    app.mount("/images", StaticFiles(directory=str(_images_dir)), name="images")


@app.get("/", tags=["Root"])
def root():
    return {
        "message": "Multimodal E-Commerce Search API",
        "docs":    "/docs",
        "health":  "/health",
    }
