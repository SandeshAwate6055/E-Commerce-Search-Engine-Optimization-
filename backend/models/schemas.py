"""
Pydantic request/response schemas for the API.
"""
from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


# ── Request bodies ────────────────────────────────────────────────────────────

class TextSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=512, description="Natural-language search query")
    top_k: int = Field(10, ge=1, le=50, description="Number of results to return")
    text_weight: float = Field(1.0, ge=0.0, le=1.0)
    apply_filters: bool = Field(True, description="Apply Qwen-inferred category/price filters")


class MultimodalSearchRequest(BaseModel):
    text: Optional[str] = Field(None, max_length=512)
    top_k: int = Field(10, ge=1, le=50)
    text_weight: float = Field(0.5, ge=0.0, le=1.0)
    image_weight: float = Field(0.5, ge=0.0, le=1.0)
    apply_filters: bool = Field(True)

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v):
        if v is not None and v.strip() == "":
            return None
        return v


# ── Response bodies ───────────────────────────────────────────────────────────

class ProductResult(BaseModel):
    rank: int
    pid: str
    product_name: str
    main_category: str
    brand: Optional[str]
    retail_price: Optional[float]
    discounted_price: Optional[float]
    image_path: str
    text_score: Optional[float]
    image_score: Optional[float]
    norm_text_score: float
    norm_image_score: float
    final_score: float
    clip_score: Optional[float] = None
    raw_caption: Optional[str] = None
    norm_text: Optional[str] = None
    retrieved_by: str


class QwenTextParse(BaseModel):
    original_query: str
    semantic_query: str
    category_hint: Optional[str]
    color: Optional[str]
    gender: Optional[str]
    brand: Optional[str]
    min_price: Optional[float]
    max_price: Optional[float]
    attributes: List[str]
    intent_summary: str


class QwenVisionParse(BaseModel):
    image_path: str
    product_type: Optional[str]
    colors: List[str]
    style: Optional[str]
    pattern: Optional[str]
    material_appearance: Optional[str]
    gender_appearance: Optional[str]
    key_visual_attributes: List[str]
    visual_description: str


class SearchResponse(BaseModel):
    mode: str
    result_count: int
    text_parse: Optional[QwenTextParse]
    vision_parse: Optional[QwenVisionParse]
    results: List[ProductResult]


class HealthResponse(BaseModel):
    status: str
    device: str
    models_loaded: bool
    products_count: int
    text_index_size: int
    image_index_size: int
    vram_used_gb: Optional[float]


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
