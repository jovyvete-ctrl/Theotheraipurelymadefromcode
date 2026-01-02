"""
#!/usr/bin/env python3
"""

#!/usr/bin/env python3
"""
Minimal LoRA fine-tuning for causal LMs using Hugging Face Transformers + PEFT.

Input dataset: JSONL file with objects containing "prompt" and "response" fields.
 - "prompt" should include the instruction and any context, ending where the model should start to produce the "response".
This script creates labels masking the prompt (labels = -100 for prompt tokens) so the model only computes loss on the response.

Usage example:
python finetune_lora.py \
  --model_name_or_path facebook/llama-7b-hf \
  --train_file sample_dataset.jsonl \
  --output_dir lora_out \
  --epochs 3 \
  --batch_size 4
"""

import argparse
from pathlib import Path
from datasets import load_dataset
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", required=True)
    p.add_argument("--train_file", required=True)
    p.add_argument("--output_dir", default="lora_out")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--load_in_4bit", action="store_true", help="Use bitsandbytes 4-bit quantization if available")
    return p.parse_args()

def make_inputs(example, tokenizer, max_length):
    prompt = example.get("prompt", "")
    response = example.get("response", "")
    # Concatenate without extra separator assuming prompt already formatted
    full = prompt + response
    enc = tokenizer(full, truncation=True, max_length=max_length, padding="max_length")
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    # Build labels: mask prompt tokens with -100 so loss only computed on response
    prompt_enc = tokenizer(prompt, truncation=True, max_length=max_length, padding=False)
    prompt_len = len(prompt_enc["input_ids"])
    labels = [-100] * prompt_len + input_ids[prompt_len:]
    # pad labels to max_length
    labels = labels[:max_length] + [-100] * max(0, max_length - len(labels))

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

def main():
    args = parse_args()
    ds = load_dataset("json", data_files={"train": args.train_file})["train"]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    # Ensure pad token exists
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model (optionally 4-bit via bitsandbytes)
    model_kwargs = {"device_map": "auto"}
    if args.load_in_4bit:
        # Lazy import to avoid hard failure if bitsandbytes not installed
        try:
            from bitsandbytes import bnb
            from transformers import BitsAndBytesConfig
            model_kwargs.update({
                "load_in_4bit": True,
                "quantization_config": BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16),
            })
        except Exception:
            print("bitsandbytes or BitsAndBytesConfig not available; attempting normal load.")
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)

    # Prepare for k-bit training if quantized, then apply LoRA
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # Preprocess dataset into input_ids/labels
    def preprocess_batch(batch):
        return [make_inputs(ex, tokenizer, args.max_length) for ex in batch]

    # Using map with batched=False because we return dicts for each example
    ds_proc = ds.map(lambda ex: make_inputs(ex, tokenizer, args.max_length),
                     remove_columns=ds.column_names)

    # Convert lists to tensors during collation
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        fp16=torch.cuda.is_available(),
        logging_steps=10,
        save_total_limit=2,
        optim="paged_adamw_8bit" if args.load_in_4bit else "adamw_torch",
        remove_unused_columns=False,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_proc,
        data_collator=data_collator,
    )

    trainer.train()
    # Save the full adapter weights (PEFT)
    model.save_pretrained(args.output_dir)
    print(f"Saved LoRA adapter + model metadata to {args.output_dir}")

if __name__ == "__main__":
    main()
