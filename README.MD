# 🛍️ Intelligent Multimodal E-Commerce Search Engine Optimization

[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/Frontend-React%2018%20%2B%20Vite-61DAFB.svg?logo=react&logoColor=black)](https://vitejs.dev/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x%20(CUDA%2012)-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FAISS](https://img.shields.io/badge/Vector%20Search-FAISS%20IndexFlatIP-00599C.svg)](https://github.com/facebookresearch/faiss)
[![Hardware](https://img.shields.io/badge/Target%20GPU-NVIDIA%20RTX%202050%20(4GB)-76B900.svg?logo=nvidia&logoColor=white)](#)
[![Patent](https://img.shields.io/badge/Patent-Complete%20Specification%20(Form%202)-blueviolet.svg)](#)

An enterprise-grade, memory-efficient **Two-Stage Multimodal Search Engine** designed for large-scale e-commerce catalogs. It decouples heavy **Vision-Language Distillation (Qwen2-VL-2B + Qwen2.5-0.5B)** from real-time query serving, combining **Dense Semantic Recall (BGE-Large + FAISS)** with **In-Memory Cross-Modal Visual Reranking (CLIP ViT-B/32)** and a **Domain-Adapted Query Intent Parser (LoRA-fine-tuned Qwen2.5-0.5B)**.

Optimized to run **100% on a consumer 4 GB GPU (NVIDIA RTX 2050)** without CPU memory offloading, reducing end-to-end multimodal search latency from **~35 seconds to ~1.1 seconds (~97% reduction)** and cutting candidate visual reranking latency from **1,500 ms to < 5 ms (>300× speedup)**.

---

## 🚀 Key Highlights & Research Innovations

* **Two-Stage Decoupled Retrieval Paradigm:**
  * **Offline Distillation:** `Qwen2-VL-2B-Instruct` extracts latent visual features into captions; `Qwen2.5-0.5B` normalizes text into canonical tokens (`brand | category | gender | color | material`).
  * **Stage 1 (Dense Semantic Recall):** `BAAI/bge-large-en-v1.5` (1024-dim) vectorizes normalized text into FAISS (`IndexFlatIP`). Retrieves Top-50 candidates in **~3.4 ms**.
  * **Stage 2 (In-Memory Visual Reranking):** Precomputes 512-dim `CLIP ViT-B/32` image embeddings as a contiguous matrix in RAM. Candidate reranking executes via vectorized PyTorch dot-product in **< 5 ms** with **zero disk I/O reads**.
* **Domain-Adapted Query Intent Reasoning (LoRA):**
  * `Qwen2.5-0.5B-Instruct` fine-tuned with LoRA ($r=8, \alpha=16$) on 8,577 catalog triplets.
  * Validation loss reduced from **3.13 → 0.62**; extracts structured entities (`category`, `gender`, `color`, `brand`, `max_price`) in **< 100 ms** using only **0.94 GB VRAM**.
* **Heuristic Fast-Path Routing:** Clean brand/keyword queries bypass the LLM entirely, executing in **< 150 ms**.
* **Zero CPU Spill on 4 GB VRAM:**
  $$\text{VRAM Total} = \text{BGE (1.25 GB)} + \text{CLIP (0.55 GB)} + \text{Qwen-LoRA (0.94 GB)} = \mathbf{2.75\text{ GB / 4.00 GB}}$$
  Leaves $> 1.3\text{ GB}$ safety buffer; eliminates PCIe bus paging and system stalls.
* **Curated & Verified Multi-Category Catalog:**
  * Cleaned 6,881 total verified products across 32 consumer categories in SQLite (`products.db`).
  * Resolved generic mobile placeholder defects by scraping and verifying 200 authentic smartphone models (iPhone, Samsung, Pixel, OnePlus, Xiaomi, Sony) with **100% unique MD5 pixel checksums**.
* **Consumer-Grade UI:** Modern React 18 interface with instant suggestions, visual match confidence badges (e.g. `96% Match`), smart filter ribbons, and one-click *"Find Similar"* visual search.

---

## 🏗️ System Architecture & Workflow

### Two-Stage Decoupled Retrieval Pipeline
Heavy generative vision models run strictly during offline batch ingestion. The real-time serving loop performs zero generative image tokens and operates purely on vectorized tensor multiplications:

![System Architecture](system_architecture_1.jpg)

### Real-Time Query Execution Flowchart
Incoming queries route through an automated fast-path or the fine-tuned LoRA intent parser, followed by two-stage dense recall, SQLite constraint filtering, and in-memory dot-product reranking:

![Real-Time Query Execution Workflow](workflow.jpg)

---

## 📊 Evaluation & Empirical Benchmarks

Evaluated over a **100-query scientific validation suite** across **6,881 catalog items** on dedicated NVIDIA RTX 2050 (4 GB VRAM):

### 1. Comparative Benchmark Performance

| Retrieval Pipeline / Architecture | Precision@5 ↑ | NDCG@10 ↑ | MRR@10 ↑ | Latency (ms) ↓ | Peak GPU VRAM | CPU Swapping |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline 1: Monolithic VLM (Qwen-3B/7B)** | 0.6400 | ~0.6200 | ~0.6800 | ~35,000 ms | > 4.0 GB (OOM) | **Severe (Paging)** |
| **Baseline 2: 1-Stage BGE Dense Only** | 0.7520 | 0.8241 | 0.8548 | ~23.3 ms | 1.25 GB | None |
| **Proposed Two-Stage (BGE + In-Memory CLIP)** | **0.7660** | **0.8094** | **0.8468** | **~1,131 ms** *(Fast-path: <150ms)* | **2.75 GB / 4.0 GB** | **Zero (100% GPU)** |

### 2. Latency Reduction Breakdown

| Component | Before Optimization | After Optimization | Performance Gain |
|---|---|---|:---:|
| **End-to-End Query Latency** | ~35,000 ms | **~1,131 ms** (Fast-path: **<150 ms**) | **~97% Faster (31×)** |
| **Visual Reranking Latency** | ~1,500 ms (Disk I/O) | **< 5 ms** (In-Memory RAM Matrix) | **>300× Faster** |
| **FAISS Vector Recall (Top-50)** | — | **~3.4 ms** | Real-Time |
| **GPU VRAM Utilization** | > 4.0 GB (Thrashing) | **2.75 GB / 4.00 GB** | **Zero CPU Spill** |

### 3. Layer-by-Layer Ablation Study

| Pipeline Stage / Optimization | P50 Latency | P95 Latency | Peak VRAM | Key Result |
|---|:---:|:---:|:---:|---|
| **Vanilla 3B Model (Baseline)** | 26,500 ms | 48,000 ms | 3,850 MB + CPU Swap | Unusable in production |
| **Migrate to Qwen2.5-0.5B (Un-tuned)** | 65 ms | 95 ms | 980 MB | 100% GPU resident |
| **+ LoRA Domain Adaptation** | 62 ms | 88 ms | 980 MB | Slot accuracy: $41\% \rightarrow 98\%$ |
| **+ Heuristic Fast-Path Router** | **< 2 ms** | 4 ms | 0 MB (Regex) | Direct bypass for keywords |
| **+ In-Memory CLIP Vectorization** | **3.8 ms** | 4.9 ms | 13.7 MB (RAM) | Replaced 1,500ms disk read |

---

## 🔬 Research & Intellectual Property (Patent)

* **Patent Specification (Form 2 Complete Specification):**
  * **Title:** *A Two-Stage Multimodal Retrieval System and Method for Low-Latency, Memory-Efficient E-Commerce Product Search*
  * **Core Novel Claims:** Asymmetric vision-language distillation decoupling, in-memory contiguous CLIP binary matrix dot-product reranking (<5ms), and strict 4 GB VRAM edge concurrency budgeting.
* **Research Paper Specification:**
  * **Title:** *Decoupled Multimodal E-Commerce Retrieval: Sub-Second Cross-Modal Reranking and LoRA Intent Reasoning on 4GB Consumer Edge Hardware*
  * **Detailed Blueprint:** See [`offline_assets/docs/research.md`](offline_assets/docs/research.md).

---

## 📁 Repository Structure

```text
Ecommerce_Search_Engine/
├── Indexing_Pipeline/              # Complete Offline Indexing Subsystem
│   ├── config/indexing.yaml        # Model & storage paths configuration
│   ├── data/                       # Batch image loader & image registry
│   ├── logic/                      # Captioning, text normalization, dense embedding
│   ├── models/                     # Qwen2-VL, Qwen2.5-0.5B, BGE-Large wrappers
│   ├── storage/                    # FAISS IndexFlatIP & SQLite database writers
│   ├── utils/                      # Batching generators, logging, validation
│   ├── scripts/                    # Smartphone catalog scraping & indexing scripts
│   ├── run_indexing.py             # Main entry point for offline indexing
│   └── README.md
│
├── Retrieval_Pipeline/             # High-Speed Online Retrieval Subsystem
│   ├── config/retrieval.yaml       # Retrieval thresholds and model paths
│   ├── logic/                      # LoRA query parsing, FAISS search, CLIP reranker
│   ├── models/                     # CLIP wrapper, unified GPU model loader, schemas
│   ├── storage/                    # FAISS searcher & SQLite reader
│   ├── utils/                      # Logging utilities
│   ├── retrieval_pipeline.py       # Two-Stage enterprise search pipeline
│   ├── run_test.py                 # Standalone CLI test script
│   └── README.md
│
├── backend/                        # Production FastAPI REST API Server
│   ├── routes/                     # /search and /health endpoints
│   ├── config/settings.py          # Server settings & path resolutions
│   ├── services/                   # Shims delegating to Retrieval_Pipeline
│   └── app.py                      # FastAPI app entrypoint & static image server
│
├── frontend/                       # Modern React 18 + Vite User Interface
│   ├── src/
│   │   ├── App.jsx                 # Homepage layout & interactive state
│   │   ├── components/             # CategoryNavBar, HeroPromoBanners, SearchBar, ProductCard
│   │   └── services/api.js         # API client connecting to backend
│   └── package.json
│
├── data/                           # Dataset & Vector Indexes
│   ├── images/                     # Product images catalog (including 200 smartphones)
│   ├── processed/
│   │   ├── faiss/                  # BGE FAISS index & precomputed CLIP vectors
│   │   ├── products.db             # Relational SQLite metadata database (6,881 items)
│   │   ├── captions_checkpoint.json
│   │   └── normalized_checkpoint.json
│   └── raw/
│
├── models/
│   └── qwen_0.5b_ecommerce_lora/   # Fine-tuned LoRA adapter weights (val loss: 0.62)
│
├── research/                       # Model training & scientific IR evaluation
│   ├── prepare_finetuning_data.py
│   ├── train_lora_qwen.py
│   └── evaluate_ir_metrics.py
│
├── notebooks/                      # Exploratory data analysis notebooks
│
├── app.py                          # Root launcher: python app.py
├── requirements.txt                # Complete Python dependencies
├── how_to_run.txt                  # Step-by-step setup and run instructions
├── download_model.txt              # Model download links and commands
├── system_architecture_1.jpg       # Two-stage system architecture diagram
├── workflow.jpg                    # Real-time query execution flowchart
└── README.MD
```

---

## ⚡ Quick Start & Installation

### 1. Prerequisites
* **OS:** Windows 10/11 or Linux
* **GPU:** NVIDIA GPU with $\ge 4\text{ GB}$ VRAM (Tested on GeForce RTX 2050, Compute Capability 8.6)
* **Python:** 3.10+
* **Node.js:** 18+

### 2. Setup & Execution
```bash
# 1. Clone the repository
git clone https://github.com/SandeshAwate6055/E-Commerce-Search-Engine-Optimization-.git
cd E-Commerce-Search-Engine-Optimization-

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Start the Backend API Server
python app.py
```
Backend API will be live at `http://127.0.0.1:8000` (Interactive Swagger docs available at `http://127.0.0.1:8000/docs`).

### 3. Frontend Web Interface
```bash
# 4. In a separate terminal, launch the React UI
cd frontend
npm install
npm run dev -- --host 127.0.0.1
```
Frontend web application will be accessible at `http://127.0.0.1:5173`.

---

## 📄 License & Academic Citation

Developed as an academic and research engineering project at **Vishwakarma Institute of Technology, Pune**.  
For citations, patent references, or questions, please refer to [`offline_assets/docs/research.md`](offline_assets/docs/research.md).
