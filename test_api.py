import requests
import time

def test_api():
    print("=" * 60)
    print("1. Testing GET http://localhost:8000/health")
    print("=" * 60)
    r = requests.get("http://localhost:8000/health")
    print("HTTP Status:", r.status_code)
    print("Health JSON:", r.json())

    print("\n" + "=" * 60)
    print("2. Testing POST http://localhost:8000/search/text")
    print("=" * 60)
    payload = {"query": "formal black shirt for men", "top_k": 3}
    t0 = time.time()
    r2 = requests.post("http://localhost:8000/search/text", json=payload)
    latency = (time.time() - t0) * 1000
    print(f"HTTP Status: {r2.status_code} in {latency:.1f}ms")
    data = r2.json()
    print("Parsed Query:", data.get("text_parse"))
    print("Results:")
    for item in data.get("results", []):
        print(f"  #{item['rank']} [Score: {item['final_score']} | CLIP: {item['clip_score']}]")
        print(f"     {item['product_name']} (Rs. {item['discounted_price']})")
        print(f"     Image: {item['image_path']}")

if __name__ == "__main__":
    test_api()
