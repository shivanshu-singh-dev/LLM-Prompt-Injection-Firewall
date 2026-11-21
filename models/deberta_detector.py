"""
Production Inference Detector for DeBERTa-v3 Prompt Injection Classifier.

Features:
- Sub-512 and multi-chunk inference with sliding window max-pooling for long inputs.
- Calibrated decision threshold enforcement.
- Graceful handling when weights or GPU/PyTorch are being configured.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import json
import numpy as np


class DeBERTaFirewallDetector:
    """Production detector wrapping fine-tuned DeBERTa-v3-small."""

    def __init__(
        self,
        model_path: str = "models/deberta-v3-firewall",
        threshold: Optional[float] = None,
        max_chunk_tokens: int = 512,
        stride: int = 256,
        device: Optional[str] = None,
    ):
        self.model_path = Path(model_path)
        self.max_chunk_tokens = max_chunk_tokens
        self.stride = stride

        # Load calibrated threshold from model directory if exists
        self.threshold = threshold
        threshold_file = self.model_path / "threshold.json"
        if self.threshold is None and threshold_file.exists():
            try:
                with open(threshold_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.threshold = float(data.get("calibrated_threshold", 0.5))
            except Exception:
                self.threshold = 0.5
        elif self.threshold is None:
            self.threshold = 0.5

        self.tokenizer = None
        self.model = None
        self.device = device
        self.is_loaded = False

        self._load_model()

    def _load_model(self):
        """Loads Hugging Face tokenizer and model if weights and torch exist."""
        if not self.model_path.exists():
            return

        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification

            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"

            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
            self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_path))
            self.model.to(self.device)
            self.model.eval()
            self.is_loaded = True
        except ImportError:
            # PyTorch or transformers not yet installed locally
            self.is_loaded = False
        except Exception as e:
            print(f"[DeBERTaFirewallDetector] Note: Model weights at {self.model_path} could not be loaded: {e}")
            self.is_loaded = False

    def chunk_text(self, text: str) -> List[str]:
        """Splits long prompts into overlapping token chunks."""
        if not self.is_loaded or self.tokenizer is None:
            # Fallback simple word chunker
            words = text.split()
            if len(words) <= self.max_chunk_tokens:
                return [text]
            chunks = []
            for i in range(0, len(words), self.stride):
                chunk = " ".join(words[i:i + self.max_chunk_tokens])
                chunks.append(chunk)
            return chunks if chunks else [text]

        tokens = self.tokenizer(text, truncation=False, return_tensors=None)["input_ids"]
        if len(tokens) <= self.max_chunk_tokens:
            return [text]

        chunks = []
        for i in range(0, len(tokens), self.stride):
            chunk_tokens = tokens[i:i + self.max_chunk_tokens]
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            if chunk_text.strip():
                chunks.append(chunk_text)
        return chunks if chunks else [text]

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Computes injection probabilities [0.0, 1.0].
        For prompts exceeding max_chunk_tokens, evaluates chunks and applies max-pooling.
        """
        if not self.is_loaded:
            raise RuntimeError(
                f"DeBERTa model is not loaded. Ensure weights are placed in {self.model_path} "
                "and 'torch' and 'transformers' are installed in the environment."
            )

        import torch

        probabilities = []
        for text in texts:
            chunks = self.chunk_text(text)
            chunk_probs = []

            for chunk in chunks:
                inputs = self.tokenizer(
                    chunk,
                    max_length=self.max_chunk_tokens,
                    truncation=True,
                    return_tensors="pt"
                ).to(self.device)

                with torch.no_grad():
                    logits = self.model(**inputs).logits
                    probs = torch.softmax(logits, dim=1)[0, 1].item()
                    chunk_probs.append(probs)

            # Max-pooling: take the highest injection score among all chunks
            max_prob = max(chunk_probs) if chunk_probs else 0.0
            probabilities.append(max_prob)

        return np.array(probabilities, dtype=float)

    def predict(self, texts: List[str], threshold: Optional[float] = None) -> np.ndarray:
        """Returns binary predictions (0=allow, 1=block) based on threshold."""
        thresh = threshold if threshold is not None else self.threshold
        probs = self.predict_proba(texts)
        return (probs >= thresh).astype(int)
