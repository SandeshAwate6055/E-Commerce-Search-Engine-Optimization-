# 🛍️ Intelligent Multimodal E-Commerce Search Engine

[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/Frontend-React%20%2B%20Vite-61DAFB.svg?logo=react&logoColor=black)](https://vitejs.dev/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x%20(CUDA)-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FAISS](https://img.shields.io/badge/Vector%20Search-FAISS-00599C.svg)](https://github.com/facebookresearch/faiss)
[![License](https://img.shields.io/badge/License-Academic%20%2F%20Research-blue.svg)](#)

An enterprise-grade, memory-efficient **Two-Stage Multimodal Search Engine** designed for fashion and e-commerce catalogs. It combines **Dense Semantic Recall (BGE-Large + FAISS)** with **In-Memory Cross-Modal Visual Reranking (CLIP ViT-B/32)** and a **Domain-Adapted Query Parser (LoRA-fine-tuned Qwen2.5-0.5B)**.

Optimized to run **100% on a consumer 4 GB GPU (NVIDIA RTX 2050)** without CPU memory offloading, reducing end-to-end search latency from **~35 seconds to ~1.1 seconds (~97% reduction)** and cutting reranking latency from **1,500 ms to < 5 ms (>300× speedup)**.

---

## 🚀 Key Highlights & Innovations

* **Two-Stage Decoupled Retrieval:**
  * **Stage 1 (Dense Semantic Recall):** `BAAI/bge-large-en-v1.5` encodes queries into 1024-dim vectors. FAISS (`IndexFlatIP`) recalls Top-50 candidates in **3.4 ms**.
  * **Stage 2 (In-Memory Visual Reranking):** Precomputed 512-dim CLIP image embeddings stored as a contiguous matrix in RAM. Candidate reranking runs via vectorized dot-product in **< 5 ms** with **zero disk I/O**.
* **Domain-Adapted Query Intent Parsing (LoRA):**
  * `Qwen2.5-0.5B-Instruct` fine-tuned using LoRA ($r=8, \alpha=16$) on 8,577 catalog triplets.
  * Validation loss reduced from **3.13 → 0.62**; extracts structured entities (`category`, `gender`, `color`, `brand`, `max_price`) using only **0.94 GB VRAM**.
* **Fast-Path Heuristic:** Clean keyword queries bypass the LLM entirely, executing in **< 150 ms**.
* **Zero CPU Spill on 4 GB VRAM:**
  $$\text{VRAM Total} = \text{BGE (1.25 GB)} + \text{CLIP (0.56 GB)} + \text{Qwen-LoRA (0.94 GB)} = \mathbf{2.75\text{ GB / 4.0 GB}}$$
* **Consumer-Grade UI:** Modern React interface with visual match confidence badges (e.g. `96% Match`), smart filter pills, and instant *"Find Similar"* image search.

---

## 🏗️ System Architecture & Execution Flow

### System Architecture Overview
The architecture completely decouples the heavy offline vision-language distillation and embedding generation from real-time online query serving:

![System Architecture](./Figure1_Architecture.jpg)

### Real-Time Query Execution Flowchart
Incoming queries pass through an automated fast-path or the fine-tuned LoRA intent parser, followed by two-stage dense recall and in-memory dot-product reranking:

![Online Query Flowchart](./Figure2_Flowchart.jpg)

---

## 📊 Evaluation & Benchmark Results

Evaluated on **4,681 fashion & lifestyle products** across a **100-query scientific validation set**:

| Retrieval Pipeline | NDCG@10 ↑ | MRR@10 ↑ | Precision@5 ↑ | Latency (ms) ↓ | VRAM Footprint |
|---|---|---|---|---|---|
| **Legacy 1-Stage Qwen-3B LLM** | ~0.6200 | ~0.6800 | ~0.6400 | ~35,000 ms | > 4.0 GB (CPU spill) |
| **Baseline 1-Stage BGE Dense** | 0.8241 | 0.8548 | 0.7520 | ~23.3 ms | 1.25 GB |
| **Proposed Two-Stage (BGE + CLIP)** | **0.8094** | **0.8468** | **0.7660** | **~1,131 ms** | **2.75 GB (Zero spill)** |

### ⏱️ Latency Reduction Breakdown

| Component | Before Optimization | After Optimization | Gain |
|---|---|---|---|
| **End-to-End Latency** | ~35,000 ms | **~1,131 ms** (Fast-path: <150 ms) | **~97% Faster (31×)** |
| **Visual Reranking** | ~1,500 ms (Disk I/O) | **< 5 ms** (In-memory matrix) | **>300× Faster** |
| **FAISS Vector Recall** | — | **3.4 ms** | Real-time |
| **GPU VRAM Utilization** | > 4.0 GB (Thrashing) | **2.75 GB / 4.00 GB** | **Fits 100% on GPU** |

---

## 📁 Repository Structure

```text
├── backend/
│   ├── app.py                      # FastAPI application entrypoint
│   ├── config/settings.py          # Centralized configuration & file paths
│   ├── indexing/
│   │   └── build_clip_image_embeddings.py  # Precomputes 512-dim RAM matrix
│   ├── models/schemas.py           # Pydantic API response & search models
│   ├── routes/                     # /search and /health endpoints
│   └── services/
│       ├── model_loader.py         # Unified single-pass GPU model initialization
│       └── search_service.py       # Two-Stage retrieval & score fusion logic
│
├── frontend/                       # Modern React + Vite User Interface
│   ├── src/
│   │   ├── App.jsx                 # Main consumer interface & filter pills
│   │   ├── components/             # SearchBar, ProductCard, ResultsGrid, DevPanel
│   │   └── services/api.js         # Backend API client with image URL handling
│   └── package.json
│
├── data/
│   └── processed/
│       ├── faiss/                  # BGE FAISS index & precomputed CLIP vectors
│       ├── captions_checkpoint.json# Qwen2-VL AI-generated captions (4,681 items)
│       ├── normalized_checkpoint.json# Qwen-0.5B extracted search tokens
│       └── products.db             # Relational SQLite metadata database
│
├── models/
│   └── qwen_0.5b_ecommerce_lora/   # Fine-tuned LoRA adapter (val loss: 0.62)
│
├── research/
│   ├── prepare_finetuning_data.py  # Synthesizes 8,577 training pairs
│   ├── train_lora_qwen.py          # PEFT LoRA training script
│   └── evaluate_ir_metrics.py      # Scientific IR benchmark (NDCG/MRR/P@5)
│
├── notebooks/                      # Exploratory data analysis & prototyping
│   ├── 01_dataset_exploration.ipynb
│   └── ...
│
├── Figure1_Architecture.jpg        # Architecture diagram
├── Figure2_Flowchart.jpg           # Online procedural flowchart
└── README.MD
```

---

## ⚡ Quick Start & Installation

### 1. Prerequisites
* **OS:** Windows 10/11 or Linux
* **GPU:** NVIDIA GPU with $\ge$ 4 GB VRAM (tested on RTX 2050)
* **Python:** 3.10+
* **Node.js:** 18+

### 2. Backend Setup
```bash
# Clone the repository
git clone https://github.com/SandeshAwate6055/E-Commerce-Search-Engine-Optimization-.git
cd E-Commerce-Search-Engine-Optimization-

# Create and activate virtual environment
python -m venv .venv_gpu
# Windows:
.venv_gpu\Scripts\activate
# Linux:
source .venv_gpu/bin/activate

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install fastapi uvicorn faiss-cpu transformers peft pandas pillow

# Start the FastAPI server
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```
Backend API will be live at `http://localhost:8000` (Swagger docs at `http://localhost:8000/docs`).

### 3. Frontend Setup
```bash
cd frontend
npm install
npm run dev -- --host
```
Frontend web application will be accessible at `http://localhost:5173`.

---
