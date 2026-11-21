"""
Training script for lightweight tradeoff models (MiniLM / DistilBERT).

Trains a smaller, faster transformer model for latency vs accuracy trade-off analysis.
Supports:
- sentence-transformers/all-MiniLM-L6-v2 (compact 22M parameters, ultra-fast)
- distilbert-base-uncased (66M parameters, balanced speed/accuracy)
"""

import argparse
from pathlib import Path
from training.train_deberta import train_deberta


def train_tradeoff_model(
    model_name: str = "distilbert/distilbert-base-uncased",
    train_file: str = "data/processed/train.jsonl",
    val_file: str = "data/processed/val.jsonl",
    output_dir: str = "models/tradeoff-model",
    epochs: int = 3,
    batch_size: int = 32,
    learning_rate: float = 3e-5,
):
    print(f"--> Training Tradeoff Model: {model_name}...")
    return train_deberta(
        train_file=train_file,
        val_file=val_file,
        output_dir=output_dir,
        model_name=model_name,
        max_length=256,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train lightweight tradeoff model.")
    parser.add_argument("--model-name", default="distilbert/distilbert-base-uncased")
    parser.add_argument("--train-file", default="data/processed/train.jsonl")
    parser.add_argument("--val-file", default="data/processed/val.jsonl")
    parser.add_argument("--output-dir", default="models/tradeoff-model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-5)
    args = parser.parse_args()

    train_tradeoff_model(
        model_name=args.model_name,
        train_file=args.train_file,
        val_file=args.val_file,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
