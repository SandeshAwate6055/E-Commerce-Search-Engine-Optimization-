"""FAISS index builder for embeddings"""
import faiss
import numpy as np
from typing import List
import os
from backend.indexing.utils.logger import setup_logger

logger = setup_logger(__name__)


class FAISSWriter:
    def __init__(self, config: dict):
        self.index_path = config['index_path']
        self.ids_path = config.get('ids_path', self.index_path.replace('.faiss', '_ids.npy'))
        self.embedding_dim = config.get('embedding_dim', 1024)
        self.normalize = config.get('normalize_vectors', True)
        self.index = None
        self.image_ids = []

    def create_index(self):
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        logger.info(f"Created FAISS IndexFlatIP dim={self.embedding_dim}")

    def load_or_create(self):
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            ids_path = self.ids_path
            if os.path.exists(ids_path):
                self.image_ids = np.load(ids_path).tolist()
            logger.info(f"Loaded FAISS index: {self.index.ntotal} vectors")
        else:
            self.create_index()

    def add_vectors_batch(self, db_ids: List[int], embeddings: np.ndarray):
        if self.normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.clip(norms, 1e-10, None)
        self.index.add(embeddings.astype('float32'))
        self.image_ids.extend(db_ids)
        logger.info(f"Added {len(db_ids)} vectors (total: {self.index.ntotal})")

    def save(self):
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        np.save(self.ids_path, np.array(self.image_ids))
        logger.info(f"Saved FAISS index: {self.index.ntotal} vectors -> {self.index_path}")
