"""
Centralised configuration — all paths and model settings in one place.
Override any value with environment variables.
"""
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ── Project root (resolved relative to this file) ────────────────────────
    PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

    # ── Data paths ───────────────────────────────────────────────────────────
    PRODUCTS_CSV: str      = ""   # filled in model_post_init
    PRODUCTS_DB: str       = ""   # SQLite database with 4,681 products
    BGE_FAISS: str         = ""   # 1024-dim dense vector index
    BGE_FAISS_IDS: str     = ""   # ID mapping array for FAISS
    CLIP_IMAGE_EMBEDDINGS: str = "" # Precomputed 512-dim CLIP image embeddings
    CLIP_IMAGE_PIDS: str   = ""   # PID array for precomputed CLIP embeddings
    TEXT_FAISS: str        = ""
    IMAGE_FAISS: str       = ""
    TEXT_MAPPING: str      = ""
    IMAGE_MAPPING: str     = ""
    IMAGES_DIR: str        = ""

    # ── Model names ──────────────────────────────────────────────────────────
    CLIP_MODEL: str        = "openai/clip-vit-base-patch32"
    BGE_MODEL: str         = "BAAI/bge-large-en-v1.5"
    QWEN_NORM_MODEL: str   = "Qwen/Qwen2.5-0.5B-Instruct"
    QWEN_LLM_MODEL: str    = "Qwen/Qwen2.5-3B-Instruct"
    QWEN_VL_MODEL: str     = "Qwen/Qwen2-VL-2B-Instruct"

    # ── Search defaults ──────────────────────────────────────────────────────
    DEFAULT_TOP_K: int        = 10
    DEFAULT_RETRIEVAL_K: int  = 50    # Stage 1: Recall top-50 from FAISS
    MAX_TOP_K: int            = 50
    DEFAULT_TEXT_WEIGHT: float  = 0.5
    DEFAULT_IMAGE_WEIGHT: float = 0.5

    # ── Upload limits ────────────────────────────────────────────────────────
    MAX_UPLOAD_BYTES: int       = 10 * 1024 * 1024   # 10 MB
    ALLOWED_IMAGE_TYPES: list   = ["image/jpeg", "image/png", "image/webp"]
    UPLOAD_DIR: str             = ""   # filled in model_post_init

    # ── Server ───────────────────────────────────────────────────────────────
    HOST: str  = "0.0.0.0"
    PORT: int  = 8000
    RELOAD: bool = False

    def model_post_init(self, __context):
        processed = self.PROJECT_ROOT / "data" / "processed"
        faiss_dir = processed / "faiss"
        upload_dir = self.PROJECT_ROOT / "backend" / "uploads"

        if not self.PRODUCTS_CSV:
            object.__setattr__(self, "PRODUCTS_CSV", str(processed / "products_ml_ready.csv"))
        if not self.PRODUCTS_DB:
            object.__setattr__(self, "PRODUCTS_DB", str(processed / "products.db"))
        if not self.BGE_FAISS:
            object.__setattr__(self, "BGE_FAISS", str(faiss_dir / "bge_index.faiss"))
        if not self.BGE_FAISS_IDS:
            object.__setattr__(self, "BGE_FAISS_IDS", str(faiss_dir / "bge_index_ids.npy"))
        if not self.CLIP_IMAGE_EMBEDDINGS:
            object.__setattr__(self, "CLIP_IMAGE_EMBEDDINGS", str(faiss_dir / "clip_image_embeddings.npy"))
        if not self.CLIP_IMAGE_PIDS:
            object.__setattr__(self, "CLIP_IMAGE_PIDS", str(faiss_dir / "clip_image_pids.npy"))
        if not self.TEXT_FAISS:
            object.__setattr__(self, "TEXT_FAISS", str(faiss_dir / "text_index.faiss"))
        if not self.IMAGE_FAISS:
            object.__setattr__(self, "IMAGE_FAISS", str(faiss_dir / "image_index.faiss"))
        if not self.TEXT_MAPPING:
            object.__setattr__(self, "TEXT_MAPPING", str(faiss_dir / "text_index_mapping.csv"))
        if not self.IMAGE_MAPPING:
            object.__setattr__(self, "IMAGE_MAPPING", str(faiss_dir / "image_index_mapping.csv"))
        if not self.IMAGES_DIR:
            object.__setattr__(self, "IMAGES_DIR", str(self.PROJECT_ROOT / "data" / "images"))
        if not self.UPLOAD_DIR:
            object.__setattr__(self, "UPLOAD_DIR", str(upload_dir))
        upload_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
