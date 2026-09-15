"""Normalized text to embedding logic"""
import numpy as np
from typing import List
from backend.indexing.utils.logger import setup_logger

logger = setup_logger(__name__)


class EmbeddingGenerator:
    def __init__(self, model):
        self.model = model

    def process_single(self, text: str) -> np.ndarray:
        return self.model.embed(text)

    def process_batch(self, texts: List[str]) -> np.ndarray:
        logger.info(f"Embedding {len(texts)} texts")
        return self.model.embed_batch(texts)
