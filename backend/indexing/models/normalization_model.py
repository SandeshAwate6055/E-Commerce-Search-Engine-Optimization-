"""Qwen2.5-0.5B wrapper for text normalization"""
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import List


class NormalizationModel:
    """Wrapper for Qwen2.5-0.5B-Instruct"""

    SYSTEM_PROMPT = (
        "You are an e-commerce product text normalization model. "
        "Extract ONLY essential product keywords from the input. "
        "Rules: "
        "1. Extract product type (shirt, shoes, watch, bag). "
        "2. Extract colors ONLY if mentioned. "
        "3. Extract material, pattern, gender if mentioned. "
        "4. Do NOT guess or add information. "
        "5. Output ONLY keywords separated by ' | '. "
        "6. Keep output minimal and consistent. "
        "Example Input: A men's formal blue cotton shirt with full sleeves. "
        "Example Output: men | formal | blue | cotton | shirt | full sleeves"
    )

    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            local_files_only=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        if device == "cpu":
            self.model.to(device)

    def normalize(self, caption: str) -> str:
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": f"Extract product keywords from: {caption}\n\nKeywords:"},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=64, do_sample=False,
                repetition_penalty=1.2,
            )
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
        return self.tokenizer.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

    def normalize_batch(self, captions: List[str]) -> List[str]:
        results = []
        for cap in captions:
            try:
                results.append(self.normalize(cap))
            except Exception as e:
                print(f"Error normalizing: {e}")
                results.append("")
        return results

    def unload(self):
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()
