"""Image to caption logic"""
from typing import List
from backend.indexing.utils.logger import setup_logger

logger = setup_logger(__name__)


class CaptionGenerator:
    def __init__(self, model):
        self.model = model

    def process_single(self, image_path: str) -> str:
        logger.info(f"Captioning: {image_path}")
        return self.model.generate_caption(image_path)

    def process_batch(self, image_paths: List[str]) -> List[str]:
        logger.info(f"Captioning {len(image_paths)} images")
        return self.model.generate_captions_batch(image_paths)
