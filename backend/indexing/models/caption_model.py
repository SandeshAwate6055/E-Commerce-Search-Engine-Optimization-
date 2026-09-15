"""Qwen2-VL wrapper for image captioning with bounded image token resolution."""
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from PIL import Image
from typing import List


class CaptionModel:
    """Wrapper for Qwen2-VL-2B-Instruct with bounded visual resolution for fast GPU inference."""

    PROMPT = (
        "You are a professional e-commerce product image caption generator. "
        "Describe the product in ONE clear, short sentence. "
        "Include ONLY: product type, color, material if visible, style, "
        "and gender target if obvious. "
        "Do NOT guess or infer. Describe only what is visible."
    )

    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = device
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            local_files_only=True,
        )
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        if device == "cpu":
            self.model.to(device)

    def generate_caption(self, image_path: str) -> str:
        # Crucial for performance: Bound min/max pixels to prevent high-res images
        # (e.g. 4000x3000) from generating 20,000+ visual tokens that freeze the GPU.
        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                    "min_pixels": 256 * 28 * 28,  # ~200k pixels (~256 tokens)
                    "max_pixels": 512 * 28 * 28   # ~400k pixels (~512 tokens)
                },
                {"type": "text", "text": self.PROMPT},
            ],
        }]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=96)
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

    def generate_captions_batch(self, image_paths: List[str]) -> List[str]:
        captions = []
        for path in image_paths:
            try:
                captions.append(self.generate_caption(path))
            except Exception as e:
                print(f"Error captioning {path}: {e}")
                captions.append("")
        return captions

    def unload(self):
        """Free GPU memory"""
        del self.model
        del self.processor
        torch.cuda.empty_cache()
