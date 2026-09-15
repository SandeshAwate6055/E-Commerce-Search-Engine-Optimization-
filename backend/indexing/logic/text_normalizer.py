"""Caption to normalized text logic"""
from typing import List
from backend.indexing.utils.logger import setup_logger

logger = setup_logger(__name__)


class TextNormalizer:
    def __init__(self, model):
        self.model = model

    def process_single(self, caption: str) -> str:
        return self.model.normalize(caption)

    def process_batch(self, captions: List[str]) -> List[str]:
        logger.info(f"Normalizing {len(captions)} captions")
        return self.model.normalize_batch(captions)
