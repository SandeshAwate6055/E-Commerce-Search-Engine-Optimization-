"""
Offline Indexing Pipeline with Incremental Checkpointing.
Every batch of captions and normalized keywords is saved immediately to disk,
ensuring 100% crash resilience: if your machine restarts or stops, it resumes
exactly where it left off without re-captioning already processed images.

Run from project root:
  .venv_gpu\\Scripts\\python -m backend.indexing.run_indexing
"""

import os
import sys

# Force 100% offline mode so disconnecting internet never triggers HuggingFace network calls or crashes
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import argparse
import json
import yaml
import time
from pathlib import Path

import pandas as pd
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.indexing.models.caption_model import CaptionModel
from backend.indexing.models.normalization_model import NormalizationModel
from backend.indexing.models.embedding_model import EmbeddingModel

from backend.indexing.logic.caption_generator import CaptionGenerator
from backend.indexing.logic.text_normalizer import TextNormalizer
from backend.indexing.logic.embedding_generator import EmbeddingGenerator

from backend.indexing.storage.postgres_writer import DatabaseWriter
from backend.indexing.storage.faiss_writer import FAISSWriter

from backend.indexing.utils.logger import setup_logger
from backend.indexing.utils.batching import create_batches

logger = setup_logger(__name__)


def load_config() -> dict:
    config_path = os.path.join(os.path.dirname(__file__), 'config', 'indexing.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def resolve_path(relative_path: str) -> str:
    """Resolve path relative to project root."""
    p = Path(relative_path)
    if p.is_absolute():
        return str(p)
    return str(PROJECT_ROOT / p)


def load_json_cache(path: str) -> dict:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load cache from {path} ({e}), starting fresh.")
            return {}
    return {}


def save_json_cache(path: str, data: dict):
    tmp_path = path + ".tmp"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def main():
    parser = argparse.ArgumentParser(description="Offline E-Commerce Indexing Pipeline with Resumable Checkpoints")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N images (for testing)")
    parser.add_argument("--db", type=str, default="postgres", choices=["postgres", "sqlite"], help="Database backend (default: postgres with sqlite fallback)")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    args = parser.parse_args()

    total_start = time.time()

    # ── Load config ──────────────────────────────────────────────────────
    config = load_config()
    logger.info("Configuration loaded")

    # Checkpoint file paths
    captions_cache_path = resolve_path("data/processed/captions_checkpoint.json")
    normalized_cache_path = resolve_path("data/processed/normalized_checkpoint.json")

    # ── Load product catalogue ───────────────────────────────────────────
    logger.info("=" * 80)
    logger.info("STEP 1: Loading Product Catalogue")
    logger.info("=" * 80)

    csv_path = resolve_path(config['data']['products_csv'])
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} products from CSV")

    images_dir = resolve_path(config['data']['images_dir'])
    supported = set(config['data']['supported_formats'])

    # Filter to products with existing images
    products = []
    for _, row in df.iterrows():
        pid = row['pid']
        img_path = None
        for ext in supported:
            candidate = os.path.join(images_dir, f"{pid}{ext}")
            if os.path.exists(candidate):
                img_path = candidate
                break
        if img_path:
            products.append({
                'pid': pid,
                'image_path': img_path,
                'product_name': str(row.get('product_name', '')),
                'main_category': str(row.get('main_category', '')),
                'brand': str(row.get('brand', '')),
                'retail_price': float(row['retail_price']) if pd.notna(row.get('retail_price')) else 0.0,
                'discounted_price': float(row['discounted_price']) if pd.notna(row.get('discounted_price')) else 0.0,
            })

    logger.info(f"Found {len(products)} products with valid images")

    # ── Init Database ────────────────────────────────────────────────────
    logger.info("=" * 80)
    logger.info(f"STEP 2: Initializing Database (Target: {args.db})")
    logger.info("=" * 80)

    db_writer = DatabaseWriter(config['database']['postgres'], prefer_db=args.db)
    db_writer.connect()
    schema_path = os.path.join(os.path.dirname(__file__), 'storage', 'schema.sql')
    db_writer.create_table(schema_path)

    # Filter out already fully-indexed products from database
    existing_pids = db_writer.get_all_pids()
    unprocessed = [p for p in products if p['pid'] not in existing_pids]
    logger.info(f"Already in DB: {len(existing_pids)} | Remaining to process: {len(unprocessed)}")

    if len(unprocessed) == 0:
        logger.info("All products already indexed into database! Nothing to do.")
        db_writer.close()
        return

    # Apply limit if requested (e.g. for testing)
    if args.limit and args.limit > 0:
        unprocessed = unprocessed[:args.limit]
        logger.info(f"[TEST MODE] Limiting to next {len(unprocessed)} products")

    batch_size = args.batch_size or config['data']['batch_size']

    # ── PHASE A: Image Captioning with Incremental Checkpointing ────────
    logger.info("=" * 80)
    logger.info("STEP 3: Image Captioning (Qwen2-VL-2B-Instruct)")
    logger.info("=" * 80)

    # Load previously saved captions from disk checkpoint
    captions = load_json_cache(captions_cache_path)
    needs_caption = [p for p in unprocessed if p['pid'] not in captions]
    logger.info(f"Captions already in checkpoint: {len(captions)} | New to caption: {len(needs_caption)}")

    caption_time = 0.0
    if len(needs_caption) > 0:
        caption_model = CaptionModel(
            model_path=config['models']['caption']['path'],
            device=config['models']['caption']['device'],
        )
        caption_gen = CaptionGenerator(caption_model)
        caption_start = time.time()

        for batch_idx, batch in enumerate(create_batches(needs_caption, batch_size)):
            paths = [p['image_path'] for p in batch]
            pids = [p['pid'] for p in batch]
            batch_captions = caption_gen.process_batch(paths)
            for pid, cap in zip(pids, batch_captions):
                captions[pid] = cap
                logger.info(f"  [{pid}] Caption: {cap}")

            # CRITICAL: Save to disk checkpoint after EVERY batch!
            save_json_cache(captions_cache_path, captions)

            done = min((batch_idx + 1) * batch_size, len(needs_caption))
            logger.info(f"  Progress: {done}/{len(needs_caption)} images captioned (Checkpointed to disk)")

        caption_time = time.time() - caption_start
        per_cap_sec = caption_time / max(len(needs_caption), 1)
        logger.info(f"[OK] New captions done in {caption_time:.2f}s ({per_cap_sec:.2f}s per image)")
        caption_model.unload()
        logger.info("[OK] Caption model unloaded from VRAM")
    else:
        logger.info("[OK] All required captions already saved in checkpoint! Skipping Qwen2-VL model load.")

    # ── PHASE B: Text Normalization with Incremental Checkpointing ──────
    logger.info("=" * 80)
    logger.info("STEP 4: Text Normalization (Qwen2.5-0.5B-Instruct)")
    logger.info("=" * 80)

    # Load previously saved normalized keywords from disk checkpoint
    normalized = load_json_cache(normalized_cache_path)
    needs_norm = [p for p in unprocessed if p['pid'] not in normalized]
    logger.info(f"Normalized texts in checkpoint: {len(normalized)} | New to normalize: {len(needs_norm)}")

    norm_time = 0.0
    if len(needs_norm) > 0:
        norm_model = NormalizationModel(
            model_path=config['models']['normalization']['path'],
            device=config['models']['normalization']['device'],
        )
        normalizer = TextNormalizer(norm_model)
        norm_start = time.time()

        for batch_idx, batch in enumerate(create_batches(needs_norm, batch_size)):
            pids = [p['pid'] for p in batch]
            batch_captions = [captions.get(pid, '') for pid in pids]
            batch_norms = normalizer.process_batch(batch_captions)
            for pid, norm in zip(pids, batch_norms):
                normalized[pid] = norm
                logger.info(f"  [{pid}] Normalized: {norm}")

            # CRITICAL: Save to disk checkpoint after EVERY batch!
            save_json_cache(normalized_cache_path, normalized)

            done = min((batch_idx + 1) * batch_size, len(needs_norm))
            logger.info(f"  Progress: {done}/{len(needs_norm)} texts normalized (Checkpointed to disk)")

        norm_time = time.time() - norm_start
        per_norm_sec = norm_time / max(len(needs_norm), 1)
        logger.info(f"[OK] New normalizations done in {norm_time:.2f}s ({per_norm_sec:.3f}s per text)")
        norm_model.unload()
        logger.info("[OK] Normalization model unloaded from VRAM")
    else:
        logger.info("[OK] All required normalized texts already saved in checkpoint! Skipping Qwen2.5 model load.")

    # ── PHASE C: Store in Database ───────────────────────────────────────
    logger.info("=" * 80)
    logger.info(f"STEP 5: Inserting {len(unprocessed)} records into Database")
    logger.info("=" * 80)

    db_ids = {}  # pid -> db row id
    for p in unprocessed:
        pid = p['pid']
        row_id = db_writer.insert_product(
            pid=pid,
            image_path=p['image_path'],
            product_name=p['product_name'],
            main_category=p['main_category'],
            brand=p['brand'],
            retail_price=p['retail_price'],
            discounted_price=p['discounted_price'],
            raw_caption=captions.get(pid, ''),
            norm_text=normalized.get(pid, ''),
        )
        if row_id:
            db_ids[pid] = row_id

    logger.info(f"[OK] Inserted {len(db_ids)} records into {db_writer.backend}")

    # ── PHASE D: Embedding + FAISS (BGE-large) ──────────────────────────
    logger.info("=" * 80)
    logger.info("STEP 6: Embedding + FAISS Index (BAAI/bge-large-en-v1.5)")
    logger.info("=" * 80)

    embed_model = EmbeddingModel(
        model_path=config['models']['embedding']['path'],
        device=config['models']['embedding']['device'],
    )
    embed_gen = EmbeddingGenerator(embed_model)

    faiss_config = config['database']['faiss'].copy()
    faiss_config['index_path'] = resolve_path(faiss_config['index_path'])
    faiss_config['ids_path'] = resolve_path(faiss_config['ids_path'])
    faiss_writer = FAISSWriter(faiss_config)
    faiss_writer.load_or_create()

    save_interval = config['processing']['save_interval']
    total_embedded = 0
    embed_start = time.time()

    for batch_idx, batch in enumerate(create_batches(unprocessed, batch_size)):
        pids = [p['pid'] for p in batch]
        texts = [normalized.get(pid, '') for pid in pids]
        batch_db_ids = [db_ids[pid] for pid in pids if pid in db_ids]

        if not batch_db_ids:
            continue

        embeddings = embed_gen.process_batch(texts[:len(batch_db_ids)])
        faiss_writer.add_vectors_batch(batch_db_ids, embeddings)

        total_embedded += len(batch_db_ids)
        if total_embedded % save_interval == 0:
            faiss_writer.save()

        done = min((batch_idx + 1) * batch_size, len(unprocessed))
        logger.info(f"  Progress: {done}/{len(unprocessed)} vectors indexed (FAISS saved)")

    embed_time = time.time() - embed_start
    per_embed_sec = embed_time / max(total_embedded, 1)
    logger.info(f"[OK] Embedding done in {embed_time:.2f}s ({per_embed_sec:.3f}s per text)")

    # Final save
    faiss_writer.save()
    embed_model.unload()
    logger.info("[OK] Embedding model unloaded from VRAM")

    # ── DONE & BENCHMARK REPORT ──────────────────────────────────────────
    db_writer.close()
    total_time = time.time() - total_start
    per_item_total = total_time / max(len(unprocessed), 1)
    total_catalog_size = len(products)
    projected_seconds = per_item_total * total_catalog_size
    projected_hours = projected_seconds / 3600

    logger.info("=" * 80)
    logger.info("INDEXING PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("=" * 80)
    logger.info(f"  Products processed this run:   {len(unprocessed)}")
    logger.info(f"  Database backend used:         {db_writer.backend}")
    logger.info(f"  Total FAISS index size:        {faiss_writer.index.ntotal} vectors")
    logger.info(f"  Total Captions in Checkpoint:  {len(captions)}")
    logger.info(f"  Total Keywords in Checkpoint:  {len(normalized)}")
    logger.info(f"  Total Run Time:                {total_time:.2f}s")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
