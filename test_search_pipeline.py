"""
Test script to verify Two-Stage Retrieval pipeline end-to-end.
"""
import time
from backend.services.model_loader import load_all
from backend.services.search_service import search

def main():
    print("=" * 80)
    print("STEP 1: LOADING MODELS & INDEXES")
    print("=" * 80)
    t0 = time.time()
    load_all()
    print(f"[OK] Startup load finished in {time.time() - t0:.2f}s")

    print("\n" + "=" * 80)
    print("STEP 2: RUNNING TWO-STAGE SEARCH TEST")
    print("=" * 80)
    queries = [
        "men stylish formal shirt black long sleeves",
        "women purple cotton casual leggings"
    ]

    for q in queries:
        print(f"\n>>> USER SEARCH: '{q}'")
        t_start = time.time()
        res = search(text=q, top_k=5)
        latency = (time.time() - t_start) * 1000

        print(f"[OK] Completed in {latency:.1f} ms!")
        print(f"Parsed Query: {res.get('text_parse')}")
        print("Top 5 Results:")
        for r in res.get("results", []):
            print(f"  #{r['rank']} [Final: {r['final_score']} | CLIP: {r['clip_score']} | BGE: {r['text_score']}]")
            print(f"     PID: {r['pid']} | {r['product_name']}")
            print(f"     Category: {r['main_category']} | Price: Rs. {r['discounted_price']}")
            print(f"     Keywords: {r['norm_text']}")

if __name__ == "__main__":
    main()
