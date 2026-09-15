"""
Precompute CLIP Image Embeddings for all 4,681 catalog items.
Saves precomputed 512-dim L2-normalized embeddings to disk.
Enables sub-5ms in-memory cross-modal reranking without reading raw image files during search!
"""
import os
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

def main():
    root = Path(__file__).resolve().parents[2]
    db_path = root / "data" / "processed" / "products.db"
    out_dir = root / "data" / "processed" / "faiss"
    out_dir.mkdir(parents=True, exist_ok=True)

    emb_path = out_dir / "clip_image_embeddings.npy"
    pids_path = out_dir / "clip_image_pids.npy"

    print("=" * 80)
    print("PRECOMPUTING CLIP IMAGE EMBEDDINGS (Offline Optimization)")
    print("=" * 80)

    # 1. Load products from SQLite
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT id, pid, image_path FROM products ORDER BY id ASC", conn)
    conn.close()
    print(f"Loaded {len(df)} products from {db_path}")

    # 2. Load CLIP
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP model on {device}...")
    model_name = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(model_name, local_files_only=True)
    model = CLIPModel.from_pretrained(model_name, local_files_only=True).to(device)
    model.eval()

    batch_size = 64
    all_embeddings = []
    all_pids = []
    valid_ids = []

    t0 = time.time()
    total = len(df)

    for i in range(0, total, batch_size):
        batch_df = df.iloc[i:i + batch_size]
        images = []
        batch_valid_pids = []

        for _, row in batch_df.iterrows():
            img_path = str(row["image_path"])
            if not os.path.exists(img_path):
                img_path = str(root / "data" / "images" / f"{row['pid']}.jpg")

            try:
                img = Image.open(img_path).convert("RGB")
                images.append(img)
                batch_valid_pids.append(row["pid"])
                valid_ids.append(row["id"])
            except Exception as e:
                print(f"Warning: could not open image {img_path}: {e}")

        if images:
            inp = processor(images=images, return_tensors="pt")
            pv = inp["pixel_values"].to(device)
            with torch.no_grad():
                vis = model.vision_model(pixel_values=pv)
                embs = model.visual_projection(vis.pooler_output)
                # L2 normalize
                norm = torch.norm(embs, p=2, dim=-1, keepdim=True)
                embs = embs / norm.clamp(min=1e-10)
                all_embeddings.append(embs.cpu().numpy().astype(np.float32))
                all_pids.extend(batch_valid_pids)

        done = min(i + batch_size, total)
        if (i // batch_size) % 10 == 0 or done == total:
            elapsed = time.time() - t0
            rate = done / max(elapsed, 0.001)
            print(f"  Processed {done}/{total} images ({rate:.1f} img/s)...")

    # 3. Stack and save
    final_embeddings = np.vstack(all_embeddings)  # Shape: (4681, 512)
    final_pids = np.array(all_pids)
    final_ids = np.array(valid_ids)

    np.save(str(emb_path), final_embeddings)
    np.save(str(pids_path), final_pids)
    np.save(str(out_dir / "clip_image_db_ids.npy"), final_ids)

    total_time = time.time() - t0
    print("\n" + "=" * 80)
    print(f"[OK] Precomputed {len(final_embeddings)} CLIP image embeddings in {total_time:.2f}s!")
    print(f"Embeddings Shape: {final_embeddings.shape} (dim=512, float32)")
    print(f"File size on disk: {os.path.getsize(emb_path) / (1024 * 1024):.2f} MB")
    print(f"Saved to: {emb_path}")
    print("=" * 80)

if __name__ == "__main__":
    main()
