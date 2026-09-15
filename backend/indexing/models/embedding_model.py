"""BAAI/bge-large-en-v1.5 wrapper for text embedding"""
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import numpy as np
from typing import List


class EmbeddingModel:
    """Wrapper for BAAI/bge-large-en-v1.5 (1024-dim)"""

    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModel.from_pretrained(model_path, local_files_only=True)
        self.model.to(device)
        self.model.eval()

    def embed(self, text: str) -> np.ndarray:
        encoded = self.tokenizer(
            text, padding=True, truncation=True,
            max_length=512, return_tensors='pt'
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.no_grad():
            output = self.model(**encoded)
            embedding = output[0][:, 0]  # CLS token
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding.cpu().numpy()[0]

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        encoded = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=512, return_tensors='pt'
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.no_grad():
            output = self.model(**encoded)
            embeddings = output[0][:, 0]
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().numpy()

    def unload(self):
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()
