import time
from backend.services.model_loader import load_all
import backend.services.search_service as ss

def main():
    load_all()

    # Warmup CUDA kernels
    q0 = "warmup shirt"
    ss.encode_bge_text(q0)
    ss.encode_text_clip(q0)

    print("\n" + "=" * 60)
    print("MEASURING TRUE IN-MEMORY RETRIEVAL LATENCY")
    print("=" * 60)

    queries = [
        "men black formal shirt",
        "women purple leggings",
        "gold mangalsutra pendant",
        "cotton casual kurti"
    ]

    for q in queries:
        t0 = time.time()
        vec = ss.encode_bge_text(q)
        cands = ss.search_bge_faiss(vec, top_n=50)
        reranked = ss.rerank_with_clip(q, cands, top_k=5)
        lat = (time.time() - t0) * 1000
        print(f"Query: '{q}' -> Latency: {lat:.1f} ms | Top PID: {reranked[0]['pid']} (Score: {reranked[0]['final_score']})")

if __name__ == "__main__":
    main()
