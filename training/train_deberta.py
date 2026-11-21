"""
Fine-tuning pipeline for microsoft/deberta-v3-small prompt injection classifier.

Compatible with local CUDA GPU or Google Colab free T4 GPU tier.
Uses Hugging Face Trainer with mixed precision (FP16), evaluation every epoch,
checkpoint saving, and metric tracking (F1, precision, recall, accuracy).
"""

import os
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

def compute_metrics(eval_pred):
    """Computes F1, precision, recall, and accuracy for Hugging Face Trainer."""
    import numpy as np
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


def load_jsonl_dataset(file_path: str):
    """Loads a JSONL split into Hugging Face Dataset format."""
    from datasets import Dataset

    texts = []
    labels = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            texts.append(item["text"])
            labels.append(int(item["label"]))

    return Dataset.from_dict({"text": texts, "label": labels})


def train_deberta(
    train_file: str = "data/processed/train.jsonl",
    val_file: str = "data/processed/val.jsonl",
    output_dir: str = "models/deberta-v3-firewall",
    model_name: str = "microsoft/deberta-v3-small",
    max_length: int = 512,
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    gradient_accumulation_steps: int = 2,
    seed: int = 42,
):
    """Fine-tunes DeBERTa-v3-small on the prompt injection dataset."""
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        Trainer,
        TrainingArguments,
        DataCollatorWithPadding,
    )

    print(f"--> Initializing fine-tuning for {model_name}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"    Execution device: {device} (CUDA available: {torch.cuda.is_available()})")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    print(f"--> Loading dataset splits: {train_file} and {val_file}...")
    train_ds = load_jsonl_dataset(train_file)
    val_ds = load_jsonl_dataset(val_file)

    def tokenize_fn(examples):
        return tokenizer(examples["text"], max_length=max_length, truncation=True)

    print("--> Tokenizing datasets...")
    tokenized_train = train_ds.map(tokenize_fn, batched=True)
    tokenized_val = val_ds.map(tokenize_fn, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(out_path / "checkpoints"),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        warmup_ratio=0.1,
        gradient_accumulation_steps=gradient_accumulation_steps,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        logging_steps=20,
        seed=seed,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("--> Starting model training...")
    trainer.train()

    print(f"--> Saving final best model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Evaluate on validation split
    eval_results = trainer.evaluate()
    print("\n" + "=" * 60)
    print("VALIDATION EVALUATION RESULTS:")
    for k, v in eval_results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
    print("=" * 60 + "\n")

    return eval_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune DeBERTa-v3 on prompt injection dataset.")
    parser.add_argument("--train-file", default="data/processed/train.jsonl")
    parser.add_argument("--val-file", default="data/processed/val.jsonl")
    parser.add_argument("--output-dir", default="models/deberta-v3-firewall")
    parser.add_argument("--model-name", default="microsoft/deberta-v3-small")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    train_deberta(
        train_file=args.train_file,
        val_file=args.val_file,
        output_dir=args.output_dir,
        model_name=args.model_name,
        max_length=args.max_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
