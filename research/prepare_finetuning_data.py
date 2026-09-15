"""
Dataset Preparation for E-Commerce Query Understanding LoRA Fine-Tuning.
Generates realistic customer search queries paired with ground-truth catalog attributes.
Creates:
  - data/research/train_data.jsonl (for LoRA fine-tuning)
  - data/research/val_data.jsonl   (for IR evaluation)
"""
import os
import re
import json
import random
import sqlite3
from pathlib import Path
import pandas as pd

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"[^\w\s|,-]", " ", str(text))
    return " ".join(text.split()).strip()

def main():
    root = Path(__file__).resolve().parents[1]
    db_path = root / "data" / "processed" / "products.db"
    out_dir = root / "data" / "research"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PREPARING E-COMMERCE QUERY UNDERSTANDING TRAINING DATASET")
    print("=" * 80)

    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT pid, product_name, main_category, brand, discounted_price, raw_caption, norm_text FROM products", conn)
    conn.close()
    print(f"Loaded {len(df)} catalog records.")

    pairs = []
    random.seed(42)

    # Color keywords common in fashion
    colors = ["black", "white", "blue", "red", "purple", "pink", "green", "yellow", "orange", "gold", "silver", "navy", "maroon", "grey", "beige", "turquoise"]
    genders = ["men", "women", "unisex", "girls", "boys"]

    query_templates = [
        "{query}",
        "buy {query}",
        "show me {query}",
        "looking for {query}",
        "best {query} online",
        "{query} with good quality",
        "{query} under budget",
        "cheap {query}",
    ]

    for _, row in df.iterrows():
        pname = clean_text(row["product_name"])
        cat = clean_text(row["main_category"])
        brand = clean_text(row["brand"]) if row["brand"] and row["brand"] != "Unknown" else None
        caption = clean_text(row["raw_caption"])
        norm_text = clean_text(row["norm_text"])
        price = row["discounted_price"]

        # Extract detected color & gender from norm_text or caption
        combined_text = f"{pname} {caption} {norm_text}".lower()
        found_color = next((c for c in colors if c in combined_text), None)
        found_gender = next((g for g in genders if g in combined_text), None)

        # Generate 2-3 query variants per product
        # Variant 1: From product title keywords
        words = [w for w in pname.split() if len(w) > 2][:5]
        if words:
            q1 = " ".join(words)
            template = random.choice(query_templates)
            user_query = template.format(query=q1.lower())

            target = {
                "semantic_query": q1.lower(),
                "category_hint": cat,
                "color": found_color,
                "gender": found_gender,
                "brand": brand,
                "keywords": norm_text,
            }
            pairs.append({"query": user_query, "target": target})

        # Variant 2: Attribute-focused query (e.g. "black formal shirt")
        if norm_text:
            key_tokens = [k.strip() for k in norm_text.split("|") if k.strip()]
            if len(key_tokens) >= 2:
                sample_tokens = random.sample(key_tokens, min(3, len(key_tokens)))
                q2 = " ".join(sample_tokens)
                template = random.choice(query_templates)
                user_query2 = template.format(query=q2.lower())

                target2 = {
                    "semantic_query": q2.lower(),
                    "category_hint": cat,
                    "color": found_color,
                    "gender": found_gender,
                    "brand": brand,
                    "keywords": norm_text,
                }
                pairs.append({"query": user_query2, "target": target2})

    # Deduplicate by query
    unique_pairs = []
    seen_queries = set()
    for p in pairs:
        q_norm = p["query"].strip().lower()
        if q_norm not in seen_queries and len(q_norm) > 4:
            seen_queries.add(q_norm)
            unique_pairs.append(p)

    random.shuffle(unique_pairs)
    val_size = min(300, int(len(unique_pairs) * 0.1))
    val_data = unique_pairs[:val_size]
    train_data = unique_pairs[val_size:]

    train_file = out_dir / "train_data.jsonl"
    val_file = out_dir / "val_data.jsonl"

    with open(train_file, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(val_file, "w", encoding="utf-8") as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Total Unique Pairs: {len(unique_pairs)}")
    print(f"  Training Set:   {len(train_data)} pairs -> {train_file}")
    print(f"  Validation Set: {len(val_data)} pairs -> {val_file}")
    print("=" * 80)

if __name__ == "__main__":
    main()
