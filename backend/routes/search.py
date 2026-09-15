"""
Search endpoints:
  POST /search/text        — text-only search
  POST /search/image       — image-only search
  POST /search/multimodal  — text + image search
"""
import logging
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from backend.config import settings
from backend.models.schemas import (
    MultimodalSearchRequest,
    SearchResponse,
    TextSearchRequest,
)
from backend.services.search_service import search
from backend.utils.file_utils import cleanup_upload, save_upload

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/search", tags=["Search"])


def _build_response(raw: dict) -> SearchResponse:
    """Convert search_service output dict → SearchResponse, handling None gracefully."""
    from backend.models.schemas import QwenTextParse, QwenVisionParse, ProductResult

    text_parse = None
    if raw.get("text_parse"):
        tp = raw["text_parse"]
        text_parse = QwenTextParse(
            original_query=tp.get("original_query", ""),
            semantic_query=tp.get("semantic_query", ""),
            category_hint=tp.get("category_hint"),
            color=tp.get("color"),
            gender=tp.get("gender"),
            brand=tp.get("brand"),
            min_price=tp.get("min_price"),
            max_price=tp.get("max_price"),
            attributes=tp.get("attributes", []),
            intent_summary=tp.get("intent_summary", ""),
        )

    vision_parse = None
    if raw.get("vision_parse"):
        vp = raw["vision_parse"]
        vision_parse = QwenVisionParse(
            image_path="[upload]",  # do not expose server filesystem path
            product_type=vp.get("product_type"),
            colors=vp.get("colors", []),
            style=vp.get("style"),
            pattern=vp.get("pattern"),
            material_appearance=vp.get("material_appearance"),
            gender_appearance=vp.get("gender_appearance"),
            key_visual_attributes=vp.get("key_visual_attributes", []),
            visual_description=vp.get("visual_description", ""),
        )

    results = [
        ProductResult(
            rank=r["rank"],
            pid=r["pid"],
            product_name=r["product_name"],
            main_category=r["main_category"],
            brand=r.get("brand"),
            retail_price=r.get("retail_price"),
            discounted_price=r.get("discounted_price"),
            image_path=r["image_path"],   # relative path, safe to expose
            text_score=r.get("text_score"),
            image_score=r.get("image_score"),
            norm_text_score=r["norm_text_score"],
            norm_image_score=r["norm_image_score"],
            final_score=r["final_score"],
            clip_score=r.get("clip_score"),
            raw_caption=r.get("raw_caption"),
            norm_text=r.get("norm_text"),
            retrieved_by=r["retrieved_by"],
        )
        for r in raw.get("results", [])
    ]

    return SearchResponse(
        mode=raw["mode"],
        result_count=raw["result_count"],
        text_parse=text_parse,
        vision_parse=vision_parse,
        results=results,
    )


# ── POST /search/text ─────────────────────────────────────────────────────────

@router.post("/text", response_model=SearchResponse)
def text_search(req: TextSearchRequest):
    """Search using a natural-language text query via Qwen LLM + CLIP + FAISS."""
    try:
        raw = search(
            text=req.query,
            top_k=req.top_k,
            retrieval_k=settings.DEFAULT_RETRIEVAL_K,
            text_weight=req.text_weight,
            image_weight=0.0,
            apply_filters=req.apply_filters,
        )
        return _build_response(raw)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("Text search error")
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")


# ── POST /search/image ────────────────────────────────────────────────────────

@router.post("/image", response_model=SearchResponse)
async def image_search(
    file:  UploadFile = File(..., description="Product image to search with"),
    top_k: int        = Form(10, ge=1, le=50),
):
    """Search by uploading an image — uses Qwen Vision + CLIP + FAISS."""
    saved_path = None
    try:
        saved_path = await save_upload(file)
        raw = search(
            image_path=str(saved_path),
            top_k=top_k,
            retrieval_k=settings.DEFAULT_RETRIEVAL_K,
            text_weight=0.0,
            image_weight=1.0,
            apply_filters=False,
        )
        return _build_response(raw)
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("Image search error")
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")
    finally:
        if saved_path:
            cleanup_upload(saved_path)


# ── POST /search/multimodal ───────────────────────────────────────────────────

@router.post("/multimodal", response_model=SearchResponse)
async def multimodal_search(
    text:         str | None  = Form(None,  description="Optional text query"),
    top_k:        int         = Form(10,    ge=1, le=50),
    text_weight:  float       = Form(0.5,   ge=0.0, le=1.0),
    image_weight: float       = Form(0.5,   ge=0.0, le=1.0),
    apply_filters: bool       = Form(True),
    file:         UploadFile | None = File(None, description="Optional query image"),
):
    """
    Full multimodal search — accepts text and/or image.
    At least one of text or file must be provided.
    """
    if (not text or not text.strip()) and file is None:
        raise HTTPException(
            status_code=422,
            detail="At least one of 'text' or 'file' must be provided.",
        )

    saved_path = None
    try:
        if file is not None:
            saved_path = await save_upload(file)

        raw = search(
            text=text if text and text.strip() else None,
            image_path=str(saved_path) if saved_path else None,
            top_k=top_k,
            retrieval_k=settings.DEFAULT_RETRIEVAL_K,
            text_weight=text_weight,
            image_weight=image_weight,
            apply_filters=apply_filters,
        )
        return _build_response(raw)
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("Multimodal search error")
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")
    finally:
        if saved_path:
            cleanup_upload(saved_path)
