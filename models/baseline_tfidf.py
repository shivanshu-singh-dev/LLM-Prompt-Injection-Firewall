"""
TF-IDF + Logistic Regression Baseline Classifier.

Serves as the empirical baseline for prompt injection detection.
Provides fast inference, word/bigram feature extraction, balanced class weighting,
and threshold tuning for target false positive rates.
"""

from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


class BaselineTFIDFClassifier:
    """TF-IDF + Logistic Regression pipeline for prompt injection classification."""

    def __init__(
        self,
        ngram_range: Tuple[int, int] = (1, 2),
        max_features: int = 10000,
        sublinear_tf: bool = True,
        c_param: float = 1.0,
        random_state: int = 42,
    ):
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.sublinear_tf = sublinear_tf
        self.c_param = c_param
        self.random_state = random_state

        self.vectorizer = TfidfVectorizer(
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            sublinear_tf=self.sublinear_tf,
            strip_accents="unicode",
            lowercase=True,
        )

        self.classifier = LogisticRegression(
            C=self.c_param,
            class_weight="balanced",
            solver="liblinear",
            random_state=self.random_state,
        )

        self.pipeline: Optional[Pipeline] = None
        self.is_fitted: bool = False
        self.calibrated_threshold: float = 0.5

    def fit(self, texts: List[str], labels: List[int]) -> "BaselineTFIDFClassifier":
        """Trains the vectorizer and logistic regression classifier."""
        self.pipeline = Pipeline([
            ("tfidf", self.vectorizer),
            ("clf", self.classifier),
        ])
        y = np.asarray(labels, dtype=int)
        self.pipeline.fit(texts, y)
        self.is_fitted = True
        return self

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Returns the probability of injection (class 1) for each input text."""
        if not self.is_fitted or self.pipeline is None:
            raise RuntimeError("Model is not fitted. Call fit() or load() first.")
        # Class 1 is injection
        probs = self.pipeline.predict_proba(texts)
        return probs[:, 1]

    def predict(self, texts: List[str], threshold: Optional[float] = None) -> np.ndarray:
        """Returns binary predictions based on decision threshold."""
        thresh = threshold if threshold is not None else self.calibrated_threshold
        probs = self.predict_proba(texts)
        return (probs >= thresh).astype(int)

    def tune_threshold_on_validation(
        self,
        val_texts: List[str],
        val_labels: List[int],
        target_fpr: float = 0.02,
        strategy: str = "target_fpr",
    ) -> float:
        """
        Tunes the decision threshold on validation data.
        
        Strategies:
        - 'target_fpr': finds lowest threshold that satisfies FPR <= target_fpr
        - 'max_f1': finds threshold that maximizes validation F1 score
        """
        y_val = np.asarray(val_labels, dtype=int)
        probs = self.predict_proba(val_texts)

        thresholds = np.linspace(0.01, 0.99, 99)
        best_threshold = 0.5
        best_f1 = -1.0

        if strategy == "target_fpr":
            valid_thresholds = []
            for t in thresholds:
                preds = (probs >= t).astype(int)
                fp = np.sum((y_val == 0) & (preds == 1))
                tn = np.sum((y_val == 0) & (preds == 0))
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

                if fpr <= target_fpr:
                    valid_thresholds.append((t, fpr))

            if valid_thresholds:
                # Lowest threshold that satisfies FPR constraint gives maximum recall
                best_threshold = valid_thresholds[0][0]
            else:
                # Fallback to threshold that yields lowest FPR
                all_fprs = [
                    (t, np.sum((y_val == 0) & ((probs >= t).astype(int) == 1)) / max(1, np.sum(y_val == 0)))
                    for t in thresholds
                ]
                all_fprs.sort(key=lambda x: x[1])
                best_threshold = all_fprs[0][0]

        elif strategy == "max_f1":
            for t in thresholds:
                preds = (probs >= t).astype(int)
                tp = np.sum((y_val == 1) & (preds == 1))
                fp = np.sum((y_val == 0) & (preds == 1))
                fn = np.sum((y_val == 1) & (preds == 0))
                p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = t

        self.calibrated_threshold = float(best_threshold)
        return self.calibrated_threshold

    def save(self, file_path: str = "models/baseline/baseline_tfidf.joblib") -> Path:
        """Serializes the fitted pipeline and metadata to disk."""
        if not self.is_fitted or self.pipeline is None:
            raise RuntimeError("Cannot save unfitted model.")
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        artifact = {
            "pipeline": self.pipeline,
            "calibrated_threshold": self.calibrated_threshold,
            "params": {
                "ngram_range": self.ngram_range,
                "max_features": self.max_features,
                "sublinear_tf": self.sublinear_tf,
                "c_param": self.c_param,
                "random_state": self.random_state,
            },
        }
        joblib.dump(artifact, path)
        return path

    @classmethod
    def load(cls, file_path: str = "models/baseline/baseline_tfidf.joblib") -> "BaselineTFIDFClassifier":
        """Loads a serialized pipeline and metadata from disk."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found at {path}")

        artifact = joblib.load(path)
        params = artifact.get("params", {})
        instance = cls(**params)
        instance.pipeline = artifact["pipeline"]
        instance.calibrated_threshold = artifact.get("calibrated_threshold", 0.5)
        instance.is_fitted = True
        return instance
