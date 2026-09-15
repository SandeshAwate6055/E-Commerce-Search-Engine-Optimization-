"""
LoRA Fine-Tuning of Qwen2.5-0.5B for E-Commerce Query Understanding.
Fine-tunes the small model on RTX 2050 GPU using PEFT / LoRA.
Memory footprint: ~1.5 GB VRAM with FP16 and Gradient Accumulation.
"""
import os
import json
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

SYSTEM_PROMPT = (
    "You are an e-commerce query understanding model. "
    "Given a user search query, extract essential product keywords and attributes. "
    "Output valid JSON ONLY. Schema:\n"
    '{"semantic_query": string, "category_hint": string|null, "color": string|null, '
    '"gender": "men"|"women"|"unisex"|null, "brand": string|null, "keywords": string}'
)

class QueryDataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 192, limit: int = 2000):
        self.examples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                item = json.loads(line.strip())
                self.examples.append(item)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = self.examples[idx]
        user_query = item["query"]
        target_str = json.dumps(item["target"], ensure_ascii=False)

        prompt = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\nQuery: {user_query}\nJSON:<|im_end|>\n"
            f"<|im_start|>assistant\n{target_str}<|im_end|>"
        )

        enc = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

def main():
    root = Path(__file__).resolve().parents[1]
    train_path = root / "data" / "research" / "train_data.jsonl"
    val_path = root / "data" / "research" / "val_data.jsonl"
    out_dir = root / "models" / "qwen_0.5b_ecommerce_lora"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("STARTING LoRA FINE-TUNING: Qwen2.5-0.5B-Instruct")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    print(f"Loading base model and tokenizer from local cache on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        local_files_only=True,
    ).to(device)

    # Configure PEFT LoRA
    print("Applying LoRA adapter configuration (rank=8, target=[q_proj, v_proj])...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    # Create datasets
    print("Preparing training batches...")
    train_dataset = QueryDataset(str(train_path), tokenizer, max_length=160, limit=1600)
    val_dataset = QueryDataset(str(val_path), tokenizer, max_length=160, limit=200)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    epochs = 2
    grad_accum_steps = 4
    total_steps = (len(train_loader) // grad_accum_steps) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=20, num_training_steps=total_steps)

    print(f"Training parameters: Epochs={epochs} | Batch Size=4 | Grad Accum={grad_accum_steps} | Steps={total_steps}")
    print("Starting training loop...\n")

    t_start = time.time()
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        step_count = 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / grad_accum_steps
            loss.backward()

            epoch_loss += outputs.loss.item()
            step_count += 1

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if (step + 1) % 100 == 0:
                current_loss = epoch_loss / step_count
                print(f"  [Epoch {epoch + 1}/{epochs}] Step {step + 1}/{len(train_loader)} - Loss: {current_loss:.4f}")

        # Validation Loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_batch in val_loader:
                v_ids = val_batch["input_ids"].to(device)
                v_mask = val_batch["attention_mask"].to(device)
                v_labels = val_batch["labels"].to(device)
                v_out = model(input_ids=v_ids, attention_mask=v_mask, labels=v_labels)
                val_loss += v_out.loss.item()

        avg_val_loss = val_loss / max(len(val_loader), 1)
        print(f">>> [Epoch {epoch + 1} Complete] Validation Loss: {avg_val_loss:.4f}\n")

    total_training_time = time.time() - t_start
    print("=" * 80)
    print(f"TRAINING COMPLETE in {total_training_time:.1f}s ({total_training_time / 60:.1f} minutes)!")
    print(f"Saving fine-tuned LoRA weights to {out_dir}...")
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print("[OK] Adapter saved successfully!")
    print("=" * 80)

if __name__ == "__main__":
    main()
