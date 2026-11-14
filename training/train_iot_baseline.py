#!/usr/bin/env python
"""
Reusable QLoRA trainer for IoT CP baselines B2-B5.
Run one baseline per Kaggle session command to keep GPU memory isolated.

Examples:
  python train_iot_baseline.py --baseline B2 --preflight
  python train_iot_baseline.py --baseline B2
  python train_iot_baseline.py --baseline B3
"""

import argparse
import inspect
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DATA_ROOT = Path("/kaggle/working/iot_cp_experiments/splits")
OUTPUT_ROOT = Path("/kaggle/working/iot_cp_experiments/runs")
SEED = 42
MAX_LENGTH = 3072

BASELINE_FILES = {
    "B2": ("b2_train.jsonl", "b2_validation.jsonl"),
    "B3": ("b3_train.jsonl", "b3_validation.jsonl"),
    "B4": ("b4_train.jsonl", "b4_validation.jsonl"),
    "B5": ("b5_train.jsonl", "b5_validation.jsonl"),
}

CONFIG = {
    "epochs": 3,
    "learning_rate": 2e-4,
    "per_device_train_batch_size": 1,
    "per_device_eval_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "warmup_ratio": 0.03,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "lr_scheduler_type": "cosine",
    "logging_steps": 10,
    "save_strategy": "epoch",
    "eval_strategy": "epoch",
    "save_total_limit": 3,
    "fp16": True,
    "optim": "paged_adamw_8bit",
    "gradient_checkpointing": True,
    "dataloader_num_workers": 2,
}


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
    return rows


def format_user_content(row):
    instruction = row["instruction"].strip()
    extra_input = row.get("input", "")
    if extra_input and extra_input.strip():
        return instruction + "\n\nInput:\n" + extra_input.strip()
    return instruction


class AssistantOnlyDataset(Dataset):
    def __init__(self, rows, tokenizer, split_name):
        self.items = []
        self.split_name = split_name
        over_limit = []

        for index, row in enumerate(rows):
            if not row.get("instruction") or not row.get("output"):
                raise ValueError(f"{split_name} row {index} has empty instruction/output")

            messages = [
                {"role": "user", "content": format_user_content(row)},
                {"role": "assistant", "content": row["output"]},
            ]
            prompt_messages = [
                {"role": "user", "content": format_user_content(row)}
            ]

            prompt_result = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
            )
            full_result = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                return_dict=True,
            )

            prompt_ids = prompt_result["input_ids"]
            input_ids = full_result["input_ids"]

            if hasattr(prompt_ids, "tolist"):
                prompt_ids = prompt_ids.tolist()
            if hasattr(input_ids, "tolist"):
                input_ids = input_ids.tolist()
            if prompt_ids and isinstance(prompt_ids[0], list):
                prompt_ids = prompt_ids[0]
            if input_ids and isinstance(input_ids[0], list):
                input_ids = input_ids[0]

            if input_ids[:len(prompt_ids)] != prompt_ids:
                raise ValueError(
                    f"Chat-template prefix mismatch in {split_name} row {index}; "
                    "cannot safely mask assistant-only loss."
                )

            if len(input_ids) > MAX_LENGTH:
                over_limit.append((index, len(input_ids)))
                continue

            labels = list(input_ids)
            labels[:len(prompt_ids)] = [-100] * len(prompt_ids)

            if not any(label != -100 for label in labels):
                raise ValueError(f"No assistant tokens available for loss at {split_name} row {index}")

            self.items.append({
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels": labels,
            })

        if over_limit:
            preview = over_limit[:10]
            raise ValueError(
                f"{split_name}: {len(over_limit)} records exceed {MAX_LENGTH} tokens. "
                f"Examples (index, length): {preview}. No truncation was applied."
            )

        if not self.items:
            raise ValueError(f"{split_name} produced an empty tokenized dataset")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class AssistantOnlyCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        max_len = max(len(item["input_ids"]) for item in features)
        input_ids, attention_masks, labels = [], [], []

        for item in features:
            padding = max_len - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [self.pad_token_id] * padding)
            attention_masks.append(item["attention_mask"] + [0] * padding)
            labels.append(item["labels"] + [-100] * padding)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def make_training_arguments(output_dir, epochs, logging_steps):
    params = inspect.signature(TrainingArguments.__init__).parameters
    kwargs = {
        "output_dir": str(output_dir),
        "num_train_epochs": epochs,
        "learning_rate": CONFIG["learning_rate"],
        "per_device_train_batch_size": CONFIG["per_device_train_batch_size"],
        "per_device_eval_batch_size": CONFIG["per_device_eval_batch_size"],
        "gradient_accumulation_steps": CONFIG["gradient_accumulation_steps"],
        "warmup_ratio": CONFIG["warmup_ratio"],
        "weight_decay": CONFIG["weight_decay"],
        "max_grad_norm": CONFIG["max_grad_norm"],
        "lr_scheduler_type": CONFIG["lr_scheduler_type"],
        "logging_steps": logging_steps,
        "save_strategy": CONFIG["save_strategy"],
        "save_total_limit": CONFIG["save_total_limit"],
        "fp16": CONFIG["fp16"],
        "optim": CONFIG["optim"],
        "gradient_checkpointing": CONFIG["gradient_checkpointing"],
        "dataloader_num_workers": CONFIG["dataloader_num_workers"],
        "report_to": "none",
        "seed": SEED,
        "data_seed": SEED,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "remove_unused_columns": False,
        "logging_first_step": True,
        "save_safetensors": True,
    }

    # Transformers releases have used both names.
    if "eval_strategy" in params:
        kwargs["eval_strategy"] = CONFIG["eval_strategy"]
    elif "evaluation_strategy" in params:
        kwargs["evaluation_strategy"] = CONFIG["eval_strategy"]

    # Avoid passing options unsupported by the installed Transformers version.
    kwargs = {key: value for key, value in kwargs.items() if key in params}
    return TrainingArguments(**kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", choices=BASELINE_FILES, required=True)
    parser.add_argument("--preflight", action="store_true",
                        help="Validate splits, tokenization and loss masking without loading the model")
    parser.add_argument("--epochs", type=int, default=CONFIG["epochs"])
    parser.add_argument("--logging-steps", type=int, default=CONFIG["logging_steps"])
    parser.add_argument("--resume-from-checkpoint", default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this training run")

    seed_everything()
    train_file, val_file = BASELINE_FILES[args.baseline]
    train_path, val_path = DATA_ROOT / train_file, DATA_ROOT / val_file

    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(
            f"Split files not found under {DATA_ROOT}. "
            "Run the split-generation cell first."
        )

    train_rows = read_jsonl(train_path)
    val_rows = read_jsonl(val_path)

    print(f"Baseline: {args.baseline}")
    print(f"Model: {MODEL_ID}")
    print(f"Train records: {len(train_rows)}")
    print(f"Validation records: {len(val_rows)}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_dataset = AssistantOnlyDataset(train_rows, tokenizer, "train")
    eval_dataset = AssistantOnlyDataset(val_rows, tokenizer, "validation")
    collator = AssistantOnlyCollator(tokenizer.pad_token_id)

    print(f"Tokenized train: {len(train_dataset)}")
    print(f"Tokenized validation: {len(eval_dataset)}")
    print("Assistant-only loss masking: enabled")
    print("Maximum sequence length: 3072; truncation: disabled")

    # Validate the collator on a real batch before any model allocation.
    sample_batch = collator([train_dataset[0], train_dataset[min(1, len(train_dataset)-1)]])
    assert sample_batch["input_ids"].shape == sample_batch["labels"].shape
    assert (sample_batch["labels"] == -100).any()
    assert (sample_batch["labels"] != -100).any()

    print("Collator smoke test: PASS")

    if args.preflight:
        print("\nPREFLIGHT PASSED. No model was loaded and no training occurred.")
        return

    run_dir = OUTPUT_ROOT / args.baseline
    checkpoints_dir = run_dir / "checkpoints"
    final_dir = run_dir / "adapter_final"
    run_dir.mkdir(parents=True, exist_ok=True)

    if final_dir.exists():
        raise FileExistsError(
            f"{final_dir} already exists. Refusing to overwrite a completed adapter."
        )

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quant_config,
        device_map={"": 0},
        dtype=torch.float16,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = make_training_arguments(
        checkpoints_dir, args.epochs, args.logging_steps
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        processing_class=tokenizer,
    )

    run_metadata = {
        "baseline": args.baseline,
        "model_id": MODEL_ID,
        "seed": SEED,
        "max_length": MAX_LENGTH,
        "train_count": len(train_dataset),
        "validation_count": len(eval_dataset),
        "lora": {
            "r": 16, "alpha": 32, "dropout": 0.05,
            "target_modules": [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
        },
        "training_config": CONFIG,
        "loss_masking": "user/system tokens masked with -100; assistant tokens supervised",
        "started_at_unix": time.time(),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2), encoding="utf-8"
    )

    train_result = trainer.train(
        resume_from_checkpoint=args.resume_from_checkpoint
    )
    eval_metrics = trainer.evaluate()

    final_dir.mkdir(parents=True, exist_ok=False)
    trainer.model.save_pretrained(final_dir, safe_serialization=True)
    tokenizer.save_pretrained(final_dir)

    metrics = {
        "train_metrics": train_result.metrics,
        "eval_metrics": eval_metrics,
        "gpu_name": torch.cuda.get_device_name(0),
        "peak_gpu_memory_gb": round(
            torch.cuda.max_memory_allocated(0) / 1024**3, 3
        ),
        "finished_at_unix": time.time(),
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, default=str), encoding="utf-8"
    )

    print("\nTRAINING COMPLETE")
    print(json.dumps(metrics, indent=2, default=str))
    print(f"Adapter saved to: {final_dir}")


if __name__ == "__main__":
    main()