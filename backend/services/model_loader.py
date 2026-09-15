"""
Model loader — loads all ML models and indexes ONCE at startup.
All other services import from this module.
Memory footprint: BGE (1.25GB) + CLIP (0.56GB) + Qwen-0.5B (0.94GB) = ~2.75GB VRAM.
Fits completely inside 4GB GPU with zero CPU offloading.
"""
import os
import sqlite3
import logging
from pathlib import Path
from typing import Dict, List, Optional

import faiss
import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    CLIPModel,
    CLIPProcessor,
    CLIPTokenizer,
    Qwen2VLForConditionalGeneration,
)

from backend.config import settings

logger = logging.getLogger(__name__)

# Force offline mode for fast startup without HuggingFace network pings
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# ── Global state ─────────────────────────────────────────────────────────────
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

# 1. BGE Embedding Model (1024-dim dense vectors)
bge_model: AutoModel = None
bge_tokenizer: AutoTokenizer = None

# 2. CLIP Reranking Model (Stage 2 Multimodal Reranker)
clip_model: CLIPModel = None
clip_processor: CLIPProcessor = None
clip_tokenizer: CLIPTokenizer = None

# 3. Qwen2.5-0.5B (Fast Query Understanding & Normalization)
qwen_norm_model: AutoModelForCausalLM = None
qwen_norm_tokenizer: AutoTokenizer = None

# 4. Qwen2-VL (Multimodal Query Image Analysis)
qwen_vl_model: Qwen2VLForConditionalGeneration = None
qwen_vl_processor: AutoProcessor = None

# 5. Vector Indexes & ID Mappings
bge_index: faiss.Index = None
bge_ids: np.ndarray = None
id_to_pid: Dict[int, str] = {}
pid_to_id: Dict[str, int] = {}

# Precomputed CLIP Image Embeddings (In-Memory Matrix for sub-5ms reranking)
clip_image_embeddings: np.ndarray = None
clip_image_pids: np.ndarray = None
pid_to_clip_idx: Dict[str, int] = {}

# Legacy indexes (if present)
text_index: faiss.Index = None
image_index: faiss.Index = None

# 6. Database & Product Metadata
products_df: pd.DataFrame = None
products_by_pid: pd.DataFrame = None
products_by_id: pd.DataFrame = None
categories: List[str] = []

_loaded: bool = False


def load_all() -> None:
    """Load every model and index. Called once at application startup."""
    global bge_model, bge_tokenizer
    global clip_model, clip_processor, clip_tokenizer
    global qwen_norm_model, qwen_norm_tokenizer
    global qwen_vl_model, qwen_vl_processor
    global bge_index, bge_ids, id_to_pid, pid_to_id
    global text_index, image_index
    global products_df, products_by_pid, products_by_id, categories
    global _loaded, DEVICE

    if _loaded:
        return

    logger.info("=" * 80)
    logger.info("INITIALIZING ENTERPRISE RETRIEVAL SYSTEM (Device: %s)", DEVICE)
    logger.info("=" * 80)

    # ── 1. Load Products from SQLite / CSV ────────────────────────────────────
    logger.info("Loading product database...")
    db_path = settings.PRODUCTS_DB
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        products_df = pd.read_sql_query("SELECT * FROM products", conn)
        conn.close()
        logger.info("  Loaded %d products from SQLite: %s", len(products_df), db_path)
    else:
        logger.warning("  %s not found, falling back to CSV: %s", db_path, settings.PRODUCTS_CSV)
        products_df = pd.read_csv(settings.PRODUCTS_CSV)

    # Ensure required columns exist
    if "raw_caption" not in products_df.columns:
        products_df["raw_caption"] = ""
    if "norm_text" not in products_df.columns:
        products_df["norm_text"] = ""

    products_by_pid = products_df.set_index("pid")
    if "id" in products_df.columns:
        products_by_id = products_df.set_index("id")
        id_to_pid = dict(zip(products_df["id"], products_df["pid"]))
        pid_to_id = dict(zip(products_df["pid"], products_df["id"]))

    categories = sorted(products_df["main_category"].dropna().unique().tolist())
    logger.info("  %d total products across %d categories", len(products_df), len(categories))

    # ── 2. Load FAISS Vector Index & Precomputed CLIP Image Embeddings ───────
    if os.path.exists(settings.BGE_FAISS) and os.path.exists(settings.BGE_FAISS_IDS):
        logger.info("Loading BGE FAISS index: %s", settings.BGE_FAISS)
        bge_index = faiss.read_index(settings.BGE_FAISS)
        bge_ids = np.load(settings.BGE_FAISS_IDS)
        logger.info("  BGE FAISS Index ready: %d vectors (dim=%d)", bge_index.ntotal, bge_index.d)
    else:
        logger.warning("  BGE FAISS index not found at %s", settings.BGE_FAISS)

    if os.path.exists(settings.CLIP_IMAGE_EMBEDDINGS) and os.path.exists(settings.CLIP_IMAGE_PIDS):
        logger.info("Loading precomputed CLIP image embeddings...")
        clip_image_embeddings = np.load(settings.CLIP_IMAGE_EMBEDDINGS)
        clip_image_pids = np.load(settings.CLIP_IMAGE_PIDS)
        pid_to_clip_idx = {str(pid): i for i, pid in enumerate(clip_image_pids)}
        logger.info("  Loaded %d precomputed CLIP image vectors in RAM (dim=%d)", len(clip_image_embeddings), clip_image_embeddings.shape[1])
    else:
        logger.warning("  Precomputed CLIP image embeddings not found at %s", settings.CLIP_IMAGE_EMBEDDINGS)

    # Legacy indexes if available
    if os.path.exists(settings.TEXT_FAISS):
        try:
            text_index = faiss.read_index(settings.TEXT_FAISS)
        except Exception:
            pass
    if os.path.exists(settings.IMAGE_FAISS):
        try:
            image_index = faiss.read_index(settings.IMAGE_FAISS)
        except Exception:
            pass

    # ── 3. Load BGE Embedding Model (Stage 1 Dense Retriever) ────────────────
    logger.info("Loading BGE dense embedder: %s", settings.BGE_MODEL)
    bge_tokenizer = AutoTokenizer.from_pretrained(settings.BGE_MODEL, local_files_only=True)
    bge_model = AutoModel.from_pretrained(settings.BGE_MODEL, local_files_only=True).to(DEVICE)
    bge_model.eval()
    logger.info("  BGE embedder loaded (1024-dim)")

    # ── 4. Load CLIP Model (Stage 2 Multimodal Reranker) ─────────────────────
    logger.info("Loading CLIP reranker: %s", settings.CLIP_MODEL)
    clip_model = CLIPModel.from_pretrained(settings.CLIP_MODEL, local_files_only=True).to(DEVICE)
    clip_processor = CLIPProcessor.from_pretrained(settings.CLIP_MODEL, local_files_only=True)
    clip_tokenizer = CLIPTokenizer.from_pretrained(settings.CLIP_MODEL, local_files_only=True)
    clip_model.eval()
    logger.info("  CLIP reranker loaded")

    # ── 5. Load Qwen2.5-0.5B (Fast Query Intent Understanding) ───────────────
    logger.info("Loading Qwen query normalizer: %s", settings.QWEN_NORM_MODEL)
    qwen_norm_tokenizer = AutoTokenizer.from_pretrained(settings.QWEN_NORM_MODEL, local_files_only=True)
    qwen_norm_model = AutoModelForCausalLM.from_pretrained(
        settings.QWEN_NORM_MODEL,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        local_files_only=True,
    ).to(DEVICE)
    qwen_norm_model.eval()
    logger.info("  Qwen query normalizer (base) loaded")

    # ── 5b. Apply LoRA fine-tuned adapter (domain-adapted for fashion queries) ─
    lora_path = str(Path(settings.PROJECT_ROOT) / "models" / "qwen_0.5b_ecommerce_lora")
    if os.path.exists(lora_path):
        try:
            from peft import PeftModel
            qwen_norm_model = PeftModel.from_pretrained(qwen_norm_model, lora_path)
            qwen_norm_model.eval()
            logger.info("  LoRA adapter applied from: %s", lora_path)
        except Exception as e:
            logger.warning("  LoRA adapter load failed (%s) — using base model", e)
    else:
        logger.warning("  LoRA adapter not found at %s — using base model", lora_path)

    logger.info("  Qwen VL deferred to on-demand lazy loading")

    _loaded = True
    if torch.cuda.is_available():
        vram = torch.cuda.memory_allocated() / 1024**3
        logger.info("=" * 80)
        logger.info("ALL ONLINE MODELS LOADED! VRAM Used: %.2f GB / 4.00 GB", vram)
        logger.info("=" * 80)
    else:
        logger.info("All online models loaded (CPU mode)")
