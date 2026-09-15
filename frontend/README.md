# Multimodal E-Commerce Search Frontend

React frontend for the FastAPI multimodal e-commerce search backend.

## Install

```bash
cd frontend
npm install
```

## Configure Backend URL

The API base URL is read from `VITE_API_BASE_URL`.

Create `frontend/.env` if you want to override the default:

```env
VITE_API_BASE_URL=http://localhost:8000
```

If the variable is not set, the frontend uses `http://localhost:8000`.

## Start Frontend

```bash
cd frontend
npm run dev
```

Vite serves the app at `http://localhost:5173`.

## Start Backend

From the project root:

```bash
.venv_gpu/Scripts/uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

The backend loads Qwen, Qwen Vision, CLIP, FAISS indexes, and the product catalogue on startup.

## Search Modes

- Text search: query only, calls `POST /search/text`.
- Image search: image upload only, calls `POST /search/image` with `multipart/form-data`.
- Multimodal search: query plus image upload, calls `POST /search/multimodal` with `multipart/form-data`.

All product cards are rendered from backend API results. The frontend does not hardcode products, rankings, images, or prices.

## Architecture

```text
src/
├── App.jsx                 # Page shell, API health, search orchestration
├── components/
│   ├── SearchBar.jsx       # Text input, upload/drop area, controls
│   ├── ResultsGrid.jsx     # Loading, empty, and result grid states
│   └── ProductCard.jsx     # Dynamic product result display
├── hooks/
│   └── useSearch.js        # Query/image/search state and validation
├── services/
│   └── api.js              # All FastAPI communication
└── index.css               # Responsive UI styling
```

## API Integration

`src/services/api.js` owns all HTTP calls through an Axios client. It handles:

- `GET /health`
- `POST /search/text`
- `POST /search/image`
- `POST /search/multimodal`
- product image URLs served by the backend at `/images/<filename>`

The request timeout is set to 5 minutes because local Qwen inference can be slow.

## Troubleshooting

- If the UI shows `API unavailable`, start FastAPI on port `8000`.
- If searches fail with CORS errors, confirm the backend includes `http://localhost:5173` in CORS origins.
- If startup is slow, wait for the backend logs to show that models and indexes are loaded.
- If product images do not display, verify that `data/images` exists and the backend mounted `/images`.
- If image upload fails, use JPEG, PNG, or WebP under 10 MB.
