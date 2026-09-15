"""
GET /health — system status endpoint.
"""
import torch
from fastapi import APIRouter

import backend.services.model_loader as ml
from backend.models.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Return the current status of models, indexes, and GPU."""
    vram_gb = None
    if torch.cuda.is_available():
        vram_gb = round(torch.cuda.memory_allocated() / 1024**3, 2)

    return HealthResponse(
        status="ok" if ml._loaded else "loading",
        device=ml.DEVICE,
        models_loaded=ml._loaded,
        products_count=len(ml.products_df) if ml.products_df is not None else 0,
        text_index_size=(ml.bge_index.ntotal if ml.bge_index is not None 
                         else (ml.text_index.ntotal if ml.text_index is not None else 0)),
        image_index_size=ml.image_index.ntotal if ml.image_index is not None else 0,
        vram_used_gb=vram_gb,
    )
