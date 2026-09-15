"""
Core search logic — Two-Stage Enterprise Retrieval Pipeline.
Stage 1: Fast Dense Retrieval (Qwen-0.5B normalizer + BGE-Large 1024-dim embedding + FAISS top-50 recall)
Stage 2: Multimodal Cross-Modal Reranking (CLIP text-to-image similarity top-50 -> top-10)

Adapted from demo architecture, optimized for 4GB RTX 2050 GPU (sub-250ms latency).
"""
import json
import re
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from backend.config import settings
import backend.services.model_loader as ml

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec, axis=1, keepdims=True)
    return vec / np.clip(norm, 1e-10, None)


def _safe_float(val) -> Optional[float]:
    try:
        if pd.isna(val) or val is None:
            return None
        return round(float(val), 4)
    except (TypeError, ValueError):
        return None


def _extract_json(raw: str) -> dict:
    raw = re.sub(r"```(?:json)?\s*", "", raw).strip()
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON in output: {raw[:150]}")
    return json.loads(match.group())


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1A: FAST QUERY UNDERSTANDING & NORMALIZATION (Qwen2.5-0.5B)
# ─────────────────────────────────────────────────────────────────────────────

_NORM_SYSTEM_PROMPT = (
    "You are an e-commerce query understanding model. "
    "Given a user search query, extract essential product keywords and attributes. "
    "Output valid JSON ONLY. Schema:\n"
    '{"semantic_query": string, "category_hint": string|null, "color": string|null, '
    '"gender": "men"|"women"|"unisex"|null, "brand": string|null, '
    '"min_price": number|null, "max_price": number|null, "intent_summary": string}'
)

_PARSE_DEFAULTS = {
    "semantic_query": "",
    "category_hint": None,
    "color": None,
    "gender": None,
    "brand": None,
    "min_price": None,
    "max_price": None,
    "attributes": [],
    "intent_summary": "",
}


def parse_text_query(query: str) -> dict:
    """Fast query intent understanding using Qwen2.5-0.5B (takes ~80-120ms)."""
    if ml.qwen_norm_model is None or ml.qwen_norm_tokenizer is None:
        # Fallback if 0.5B is not loaded
        return {
            "original_query": query,
            "semantic_query": query.strip(),
            "category_hint": None,
            "color": None,
            "gender": None,
            "brand": None,
            "min_price": None,
            "max_price": None,
            "attributes": [],
            "intent_summary": query.strip(),
        }

    msgs = [
        {"role": "system", "content": _NORM_SYSTEM_PROMPT},
        {"role": "user", "content": f"Query: {query.strip()}\nJSON:"},
    ]
    text = ml.qwen_norm_tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )
    inputs = ml.qwen_norm_tokenizer([text], return_tensors="pt").to(ml.DEVICE)

    with torch.no_grad():
        out = ml.qwen_norm_model.generate(
            **inputs, max_new_tokens=40, do_sample=False, repetition_penalty=1.1, use_cache=True
        )
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
    raw = ml.qwen_norm_tokenizer.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

    try:
        parsed = _extract_json(raw)
    except Exception:
        parsed = {}

    result = {"original_query": query}
    for k, v in _PARSE_DEFAULTS.items():
        val = parsed.get(k, v)
        result[k] = val

    if not result.get("semantic_query"):
        result["semantic_query"] = query.strip()
    if not result.get("intent_summary"):
        result["intent_summary"] = query.strip()

    return result


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1B: DENSE QUERY EMBEDDING (BGE-Large 1024-dim)
# ─────────────────────────────────────────────────────────────────────────────

def encode_bge_text(text: str) -> np.ndarray:
    """Encodes text using BAAI/bge-large-en-v1.5 to a 1024-dim L2-normalized vector."""
    if ml.bge_model is None or ml.bge_tokenizer is None:
        raise RuntimeError("BGE model is not loaded.")

    encoded = ml.bge_tokenizer(
        [text], padding=True, truncation=True, max_length=512, return_tensors="pt"
    )
    encoded = {k: v.to(ml.DEVICE) for k, v in encoded.items()}
    with torch.no_grad():
        out = ml.bge_model(**encoded)
        emb = out[0][:, 0]  # CLS token
        emb = F.normalize(emb, p=2, dim=1)
    return emb.cpu().numpy().astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1C: FAISS DENSE CANDIDATE RECALL (Top-50)
# ─────────────────────────────────────────────────────────────────────────────

def search_bge_faiss(query_vec: np.ndarray, top_n: int = 50) -> List[dict]:
    """Retrieves top_n candidate products from the 1024-dim BGE FAISS index."""
    if ml.bge_index is None:
        return []

    scores, indices = ml.bge_index.search(query_vec, top_n)
    candidates = []

    for idx, score in zip(indices[0], scores[0]):
        if idx == -1:
            continue
        db_id = int(ml.bge_ids[idx]) if ml.bge_ids is not None else int(idx)
        pid = ml.id_to_pid.get(db_id)

        if not pid and ml.products_by_id is not None and db_id in ml.products_by_id.index:
            pid = ml.products_by_id.loc[db_id]["pid"]

        if pid and pid in ml.products_by_pid.index:
            candidates.append({
                "pid": pid,
                "db_id": db_id,
                "semantic_score": float(score),
            })

    return candidates


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2: CLIP CROSS-MODAL RERANKING (Top-50 -> Top-10)
# ─────────────────────────────────────────────────────────────────────────────

def encode_text_clip(text: str) -> np.ndarray:
    """CLIP text encoder (512-dim vector)."""
    toks = ml.clip_tokenizer(
        [text], return_tensors="pt", padding=True, truncation=True, max_length=77
    )
    toks = {k: v.to(ml.DEVICE) for k, v in toks.items()}
    with torch.no_grad():
        out = ml.clip_model.text_model(**toks)
        emb = ml.clip_model.text_projection(out.pooler_output)
    return _l2_normalize(emb.cpu().float().numpy())


def encode_image_clip(image_path: str) -> np.ndarray:
    """CLIP image encoder (512-dim vector)."""
    img = Image.open(image_path).convert("RGB")
    inp = ml.clip_processor(images=[img], return_tensors="pt")
    pv = inp["pixel_values"].to(ml.DEVICE)
    with torch.no_grad():
        vis = ml.clip_model.vision_model(pixel_values=pv)
        emb = ml.clip_model.visual_projection(vis.pooler_output)
    return _l2_normalize(emb.cpu().float().numpy())


def rerank_with_clip(query: str, candidates: List[dict], top_k: int = 10) -> List[dict]:
    """
    Reranks candidate products using CLIP text-to-image cosine similarity.
    Uses precomputed in-memory image embeddings (takes < 3ms with zero disk I/O).
    """
    if not candidates or ml.clip_model is None:
        return candidates[:top_k]

    # 1. Encode user query with CLIP (takes ~5ms)
    query_emb = encode_text_clip(query)  # (1, 512)

    # 2. Ultra-Fast Path: In-Memory Precomputed CLIP Matrix Multiplication (< 2ms)
    if ml.clip_image_embeddings is not None and ml.pid_to_clip_idx:
        valid_candidates = []
        cand_indices = []
        for c in candidates:
            idx = ml.pid_to_clip_idx.get(str(c["pid"]))
            if idx is not None:
                valid_candidates.append(c)
                cand_indices.append(idx)

        if valid_candidates:
            cand_embs = ml.clip_image_embeddings[cand_indices]  # (N, 512)
            clip_scores = np.dot(cand_embs, query_emb.T).squeeze(-1)  # (N,)
            for c, c_score in zip(valid_candidates, clip_scores):
                c["clip_score"] = float(c_score)
                sem = c.get("semantic_score", 0.5)
                c["final_score"] = round(float(0.35 * sem + 0.65 * c_score), 4)

            reranked = sorted(valid_candidates, key=lambda x: x["final_score"], reverse=True)
            return reranked[:top_k]

    # Fallback to loading raw images if precomputed embeddings are absent
    valid_candidates = []
    valid_images = []
    for c in candidates:
        pid = c["pid"]
        img_path = None

        # Look in DB metadata or standard path
        if pid in ml.products_by_pid.index:
            row = ml.products_by_pid.loc[pid]
            cand_path = str(row.get("image_path", ""))
            if os.path.exists(cand_path):
                img_path = cand_path

        if not img_path:
            cand_path = os.path.join(settings.IMAGES_DIR, f"{pid}.jpg")
            if os.path.exists(cand_path):
                img_path = cand_path

        if img_path:
            try:
                img = Image.open(img_path).convert("RGB")
                valid_images.append(img)
                valid_candidates.append(c)
            except Exception as e:
                logger.warning(f"Error opening image {img_path}: {e}")

    if not valid_images:
        return candidates[:top_k]

    # 3. Batch encode candidate images with CLIP
    batch_size = 16
    img_embs_list = []
    for i in range(0, len(valid_images), batch_size):
        batch = valid_images[i:i + batch_size]
        inp = ml.clip_processor(images=batch, return_tensors="pt")
        pv = inp["pixel_values"].to(ml.DEVICE)
        with torch.no_grad():
            vis = ml.clip_model.vision_model(pixel_values=pv)
            embs = ml.clip_model.visual_projection(vis.pooler_output)
            embs_norm = _l2_normalize(embs.cpu().float().numpy())
            img_embs_list.append(embs_norm)

    all_img_embs = np.vstack(img_embs_list)  # (N, 512)

    # 4. Compute text-to-image cosine similarities
    clip_scores = np.dot(all_img_embs, query_emb.T).squeeze(-1)  # (N,)

    # 5. Attach CLIP score and compute hybrid score (0.4 Semantic + 0.6 CLIP)
    for c, c_score in zip(valid_candidates, clip_scores):
        c["clip_score"] = float(c_score)
        sem = c.get("semantic_score", 0.5)
        # Normalised combination
        c["final_score"] = round(float(0.35 * sem + 0.65 * c_score), 4)

    # Sort descending by final score
    reranked = sorted(valid_candidates, key=lambda x: x["final_score"], reverse=True)
    return reranked[:top_k]


# ─────────────────────────────────────────────────────────────────────────────
# QWEN VISION ANALYSIS (FOR USER UPLOADED IMAGES)
# ─────────────────────────────────────────────────────────────────────────────

def analyze_image(image_path: str) -> dict:
    """Visual attribute parsing for uploaded images."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    # If Qwen-VL is available, use it; otherwise provide quick metadata
    if ml.qwen_vl_model is not None and ml.qwen_vl_processor is not None:
        try:
            from qwen_vl_utils import process_vision_info
            _VISION_PROMPT = (
                "Describe this product image in one sentence: product type, color, and style."
            )
            messages = [{"role": "user", "content": [
                {"type": "image", "image": str(path),
                 "min_pixels": 256 * 28 * 28, "max_pixels": 512 * 28 * 28},
                {"type": "text", "text": _VISION_PROMPT},
            ]}]
            text = ml.qwen_vl_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            img_inputs, vid_inputs = process_vision_info(messages)
            inputs = ml.qwen_vl_processor(
                text=[text], images=img_inputs, videos=vid_inputs,
                padding=True, return_tensors="pt"
            ).to(ml.DEVICE)

            with torch.no_grad():
                out = ml.qwen_vl_model.generate(**inputs, max_new_tokens=96, do_sample=False)
            desc = ml.qwen_vl_processor.decode(
                out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
            ).strip()
            return {
                "image_path": str(image_path),
                "visual_description": desc,
                "product_type": None,
                "colors": [],
                "style": None,
                "pattern": None,
                "material_appearance": None,
                "gender_appearance": None,
                "key_visual_attributes": [],
            }
        except Exception as e:
            logger.warning(f"Qwen VL analysis skipped ({e})")

    return {
        "image_path": str(image_path),
        "visual_description": "Uploaded product image",
        "product_type": None,
        "colors": [],
        "style": None,
        "pattern": None,
        "material_appearance": None,
        "gender_appearance": None,
        "key_visual_attributes": [],
    }


# ─────────────────────────────────────────────────────────────────────────────
# UNIFIED SEARCH PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def search(
    text: Optional[str] = None,
    image_path: Optional[str] = None,
    top_k: int = 10,
    retrieval_k: int = 50,
    text_weight: float = 0.5,
    image_weight: float = 0.5,
    apply_filters: bool = True,
) -> dict:
    """
    Two-Stage Enterprise Search Entry Point.
    Stage 1: High Recall (Top-50 via BGE dense embedding + FAISS index)
    Stage 2: High Precision Reranking (Top-10 via CLIP text-to-image similarity)
    """
    has_text = text is not None and str(text).strip() != ""
    has_image = image_path is not None and str(image_path).strip() != ""

    if not has_text and not has_image:
        raise ValueError("Provide at least one of: text or image_path.")

    mode = "multimodal" if (has_text and has_image) else ("text" if has_text else "image")
    text_parse = None
    vision_parse = None
    candidates = []

    # ── CASE 1: Text Search (Standard Query) ─────────────────────────────────
    if has_text:
        text_parse = parse_text_query(text)
        semantic_q = text_parse.get("semantic_query", text)

        # Stage 1: Dense Retrieval via BGE (1024-dim)
        query_vec = encode_bge_text(semantic_q)
        candidates = search_bge_faiss(query_vec, top_n=retrieval_k)

        # Apply Category / Price Filters if requested
        if apply_filters and candidates:
            cat_hint = text_parse.get("category_hint")
            max_p = text_parse.get("max_price")
            min_p = text_parse.get("min_price")

            filtered = []
            for c in candidates:
                pid = c["pid"]
                meta = ml.products_by_pid.loc[pid] if pid in ml.products_by_pid.index else {}
                if cat_hint and meta.get("main_category") != cat_hint:
                    continue
                try:
                    price = float(meta.get("discounted_price", 0.0))
                    if max_p is not None and price > max_p:
                        continue
                    if min_p is not None and price < min_p:
                        continue
                except (TypeError, ValueError):
                    pass
                filtered.append(c)

            # If filters were too strict, keep top candidates
            if len(filtered) >= 3:
                candidates = filtered

        # Stage 2: CLIP Reranking (Top-50 candidates -> Top-10 reranked)
        ranked_candidates = rerank_with_clip(query=text, candidates=candidates, top_k=top_k)

    # ── CASE 2: Image-Only Search ────────────────────────────────────────────
    elif has_image:
        vision_parse = analyze_image(image_path)
        img_vec = encode_image_clip(image_path)

        # Search CLIP image index if available, else retrieve from BGE using visual description
        if ml.image_index is not None:
            scores, indices = ml.image_index.search(img_vec.astype(np.float32), retrieval_k)
            for fidx, score in zip(indices[0], scores[0]):
                if fidx != -1:
                    pid = ml.image_faiss_to_pid.get(int(fidx)) if hasattr(ml, "image_faiss_to_pid") else None
                    if pid:
                        candidates.append({"pid": pid, "semantic_score": float(score), "final_score": float(score)})
        else:
            # Dense search using BGE on the image's visual description
            desc = vision_parse.get("visual_description", "product image")
            desc_vec = encode_bge_text(desc)
            candidates = search_bge_faiss(desc_vec, top_n=retrieval_k)

        ranked_candidates = candidates[:top_k]

    # ── Build Final Rich Product Results ─────────────────────────────────────
    output_results = []
    for rank, item in enumerate(ranked_candidates, 1):
        pid = item["pid"]
        meta = ml.products_by_pid.loc[pid] if pid in ml.products_by_pid.index else {}

        output_results.append({
            "rank": rank,
            "pid": pid,
            "product_name": str(meta.get("product_name", "Product")),
            "main_category": str(meta.get("main_category", "Clothing")),
            "brand": meta.get("brand") if pd.notna(meta.get("brand")) else "Unknown",
            "retail_price": _safe_float(meta.get("retail_price")),
            "discounted_price": _safe_float(meta.get("discounted_price")),
            "image_path": str(meta.get("image_path", f"data/images/{pid}.jpg")),
            "text_score": _safe_float(item.get("semantic_score")),
            "image_score": _safe_float(item.get("clip_score")),
            "norm_text_score": round(float(item.get("semantic_score", 0.0)), 4),
            "norm_image_score": round(float(item.get("clip_score", 0.0)), 4),
            "final_score": round(float(item.get("final_score", item.get("semantic_score", 0.0))), 4),
            "clip_score": _safe_float(item.get("clip_score")),
            "raw_caption": str(meta.get("raw_caption", "")) if pd.notna(meta.get("raw_caption")) else None,
            "norm_text": str(meta.get("norm_text", "")) if pd.notna(meta.get("norm_text")) else None,
            "retrieved_by": "two_stage_bge_clip",
        })

    return {
        "mode": mode,
        "text_parse": text_parse,
        "vision_parse": vision_parse,
        "result_count": len(output_results),
        "results": output_results,
    }
