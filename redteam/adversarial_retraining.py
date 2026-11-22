"""
Adversarial Retraining Pipeline (Phase 7).

1. Splits mutated attack dataset into 60% retraining augmentation and 40% strictly held-out test mutations.
2. Augments clean training data with the retraining split.
3. Retrains the classifier with character-level and word n-gram features to defend against leetspeak and encoding evasions.
4. Saves the hardened model to models/hardened/hardened_classifier.joblib.
"""

from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import json
import random
import numpy as np

from data.schema import PromptExample
from models.baseline_tfidf import BaselineTFIDFClassifier


def split_mutations_train_test(
    mutations: List[Dict[str, Any]],
    train_ratio: float = 0.60,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Splits mutated payloads into training and strictly held-out evaluation sets,
    stratified by mutation type.
    """
    rng = random.Random(seed)
    by_type: Dict[str, List[Dict[str, Any]]] = {}

    for m in mutations:
        t = m["mutation_type"]
        by_type.setdefault(t, []).append(m)

    train_mutations = []
    heldout_test_mutations = []

    for m_type, items in by_type.items():
        items_copy = list(items)
        rng.shuffle(items_copy)
        n = len(items_copy)
        if n == 1:
            if rng.random() < train_ratio:
                train_mutations.append(items_copy[0])
            else:
                heldout_test_mutations.append(items_copy[0])
        else:
            n_train = max(1, min(n - 1, int(round(n * train_ratio))))
            train_mutations.extend(items_copy[:n_train])
            heldout_test_mutations.extend(items_copy[n_train:])

    return train_mutations, heldout_test_mutations


def train_hardened_classifier(
    clean_train_path: str = "data/processed/train.jsonl",
    adversarial_train_items: Optional[List[Dict[str, Any]]] = None,
    output_path: str = "models/hardened/hardened_classifier.joblib",
) -> BaselineTFIDFClassifier:
    """
    Retrains the classifier on clean data augmented with adversarial mutations.
    Enhances feature extraction with character n-grams (2-5) and word n-grams (1-3)
    to neutralize leetspeak, encoding fragments, and delimiter splitting.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import FeatureUnion, Pipeline

    # 1. Load clean training data
    texts = []
    labels = []
    with open(clean_train_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                texts.append(item["text"])
                labels.append(int(item["label"]))

    # 2. Add adversarial training samples
    if adversarial_train_items:
        for m in adversarial_train_items:
            texts.append(m["mutated_prompt"])
            labels.append(1)  # All mutations are malicious prompt injections

    # 3. Build robust feature pipeline combining word and character n-grams
    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 3),
        max_features=12000,
        sublinear_tf=True,
        strip_accents="unicode",
        lowercase=True,
    )
    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=12000,
        sublinear_tf=True,
        lowercase=True,
    )

    combined_features = FeatureUnion([
        ("word_tfidf", word_vectorizer),
        ("char_tfidf", char_vectorizer),
    ])

    classifier = LogisticRegression(
        C=2.0,
        class_weight="balanced",
        solver="liblinear",
        random_state=42,
    )

    hardened_pipeline = Pipeline([
        ("features", combined_features),
        ("clf", classifier),
    ])

    print(f"--> Retraining hardened classifier on {len(texts)} samples ({labels.count(0)} benign, {labels.count(1)} malicious)...")
    hardened_pipeline.fit(texts, np.array(labels, dtype=int))

    # Wrap into BaselineTFIDFClassifier structure for compatibility
    model = BaselineTFIDFClassifier()
    model.pipeline = hardened_pipeline
    model.is_fitted = True
    model.calibrated_threshold = 0.50

    # Save artifact
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump({"pipeline": hardened_pipeline, "calibrated_threshold": 0.50}, out_file)
    print(f"--> Saved hardened classifier to {out_file}")

    return model


if __name__ == "__main__":
    from redteam.attack_firewall import load_seed_payloads, generate_mutated_attack_suite

    seed = load_seed_payloads()
    mutations = generate_mutated_attack_suite(seed)
    train_muts, test_muts = split_mutations_train_test(mutations)
    print(f"Split {len(mutations)} mutations into {len(train_muts)} training and {len(test_muts)} held-out test.")

    model = train_hardened_classifier(adversarial_train_items=train_muts)
