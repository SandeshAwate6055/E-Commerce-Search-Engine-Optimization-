# Multimodal E-Commerce Search — Backend API

## Architecture

```
User Request
  │
  ├─ POST /search/text        ─► Qwen LLM → CLIP text → Text FAISS
  ├─ POST /search/image       ─► Qwen Vision → CLIP image → Image FAISS
  └─ POST /search/multimodal  ─► both paths above
                                        │
                                  Candidate Pool
                                        │
                                  Score Normalization (min-max)
                                        │
                                  Score Fusion (configurable weights)
                                        │
                                  Metadata Filters (category, price)
                                        │
                                  Top-K JSON Response
```

## How to Run

```bash
# From project root (d:/Ecommerce_search_engine)
.venv_gpu/Scripts/uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

First startup takes ~30–60 seconds to load all models into GPU memory.

## Endpoints

### `GET /health`
Returns backend and model status.

**Response:**
```json
{
  "status": "ok",
  "device": "cuda",
  "models_loaded": true,
  "products_count": 4681,
  "text_index_size": 4681,
  "image_index_size": 4681,
  "vram_used_gb": 4.68
}
```

---

### `POST /search/text`
Text-only product search.

**Request body (JSON):**
```json
{
  "query": "men's formal shirt",
  "top_k": 10,
  "text_weight": 1.0,
  "apply_filters": true
}
```

**Flow:** `query → Qwen LLM → semantic_query → CLIP text embed → Text FAISS → filters → ranked results`

---

### `POST /search/image`
Image-only product search via file upload.

**Request:** `multipart/form-data`
- `file`: image file (JPEG/PNG/WebP, max 10 MB)
- `top_k`: integer (1–50, default 10)

**Flow:** `image → Qwen Vision → CLIP image embed → Image FAISS → ranked results`

---

### `POST /search/multimodal`
Combined text + image search.

**Request:** `multipart/form-data`
- `text`: optional query string
- `file`: optional image file
- `top_k`: integer (default 10)
- `text_weight`: float 0.0–1.0 (default 0.5)
- `image_weight`: float 0.0–1.0 (default 0.5)
- `apply_filters`: boolean (default true)

At least one of `text` or `file` must be provided.

---

## Response Format

All search endpoints return:

```json
{
  "mode": "text | image | multimodal",
  "result_count": 10,
  "text_parse": {
    "original_query": "...",
    "semantic_query": "...",
    "category_hint": "Clothing",
    "color": "black",
    "gender": "men",
    "max_price": 2000.0,
    "attributes": ["formal"],
    "intent_summary": "..."
  },
  "vision_parse": {
    "image_path": "[upload]",
    "product_type": "shoes",
    "colors": ["orange"],
    "style": "wedged",
    "visual_description": "...",
    ...
  },
  "results": [
    {
      "rank": 1,
      "pid": "SNDEDAPKZGGEGYHV",
      "product_name": "S.m.a.R.T FEET Women Wedges",
      "main_category": "Footwear",
      "brand": "Unknown",
      "retail_price": 999.0,
      "discounted_price": 499.0,
      "image_path": "../data/images/SNDEDAPKZGGEGYHV.jpg",
      "text_score": null,
      "image_score": 1.0,
      "norm_text_score": 0.0,
      "norm_image_score": 1.0,
      "final_score": 1.0,
      "retrieved_by": "image"
    }
  ]
}
```

## Configuration

Copy `backend/.env.example` to `backend/.env` and override any values.
All paths default to project-relative locations — no hardcoded machine paths.

Key settings:
| Variable | Default | Description |
|---|---|---|
| `CLIP_MODEL` | `openai/clip-vit-base-patch32` | CLIP model ID |
| `QWEN_LLM_MODEL` | `Qwen/Qwen2.5-3B-Instruct` | Text query understanding |
| `QWEN_VL_MODEL` | `Qwen/Qwen2-VL-2B-Instruct` | Visual understanding |
| `DEFAULT_TOP_K` | `10` | Default result count |
| `MAX_TOP_K` | `50` | Maximum allowed top_k |
| `MAX_UPLOAD_BYTES` | `10485760` | 10 MB upload limit |
| `PORT` | `8000` | Server port |

## Limitations

- Qwen LLM inference runs partially on CPU (offloaded layers) — ~20–60 sec per query
- RTX 2050 (4 GB VRAM): all models share GPU via `device_map="auto"`
- Category filter reduces recall if Qwen infers the wrong category
- Image upload is processed and immediately deleted after the request
- No authentication — restrict network access for production use

## File Structure

```
backend/
├── app.py                  # FastAPI app, startup lifespan
├── .env.example            # Configuration template
├── config/
│   └── settings.py         # Pydantic settings (env-configurable)
├── services/
│   ├── model_loader.py     # Load all models/indexes once at startup
│   └── search_service.py   # Core ML pipeline logic
├── models/
│   └── schemas.py          # Pydantic request/response schemas
├── routes/
│   ├── health.py           # GET /health
│   └── search.py           # POST /search/text, /image, /multimodal
└── utils/
    └── file_utils.py       # Upload validation and temp file handling
```
