# LLM Prompt Injection Firewall & Security Scanner

A high-performance security framework and proxy that protects Large Language Models (LLMs) against prompt injection attacks, jailbreaks, and system prompt leakage in real time.

---

## Overview

The **Prompt Injection Firewall** intercepts and inspects user inputs before they reach the language model, combining fast deterministic heuristics with trained classification models to filter adversarial inputs with sub-millisecond to low-millisecond overhead.

The repository also retains the **Security Scanner** harness, which actively red-teams target models using an adversarial payload library and verifies safety compliance.

```
                    ┌────────────────────────────────────────────────────────┐
                    │               PROMPT INJECTION FIREWALL                │
                    │                                                        │
Incoming Prompt ───►│  [ Layer 1: Fast Heuristics / Known Pattern Rules ]    │
                    │                      │ (pass)                          │
                    │  [ Layer 2: DeBERTa-v3 / ML Classifier Scoring ]       │
                    │                      │ (allow)                         │
                    └──────────────────────┼─────────────────────────────────┘
                                           ▼
                                 [ Target LLM / Ollama ]
                                           │
                    ┌──────────────────────┼─────────────────────────────────┐
                    │  [ Layer 3: Output Canary / Secret Leak Inspection ]   │
                    └──────────────────────┼─────────────────────────────────┘
                                           ▼
                                   Safe LLM Response
```

---

## Key Features

- **Multi-Tiered Defense**: Fast regex/heuristic rules for known injection syntax + transformer classifier for semantic evasion.
- **Canary & Output Inspection**: Response verification for internal canary tokens (e.g., `NW-7734-ADMIN`) and policy compliance.
- **Zero-Day Generalization Testing**: Leave-One-Attack-Family-Out (LOFO) cross-validation evaluating robustness against completely unseen attack families.
- **Hard-Negative Evaluation**: Dedicated benchmark testing False Positive Rate (FPR) on benign prompts containing suspicious trigger keywords (`ignore`, `override`, `system`, technical logs).
- **Sub-512 & Long Prompt Chunking**: Production inference detector with sliding window max-pooling for inputs exceeding token length limits.
- **OpenAI-Compatible Proxy**: Drop-in reverse proxy compatible with any standard OpenAI chat completion client.
- **Red-Team Scanner Harness**: 69 seed adversarial test vectors across multiple injection techniques with CLI and web visualization.

---

## Repository Structure

```
.
├── benchmarks/             # Latency & throughput profiling
│   └── benchmark_latency.py# Per-request p50, p95, p99 latency benchmarking (batch_size=1)
├── data/                   # Data schemas, ingestion, deduplication, and split pipelines
│   ├── schema.py           # Canonical PromptExample schema
│   ├── convert_scanner.py  # Converts scanner YAML payloads to standard format
│   ├── download_datasets.py# Public dataset fetcher
│   ├── clean_datasets.py   # Dataset normalization and cleaning
│   ├── curate_hard_negatives.py # Benign queries with trigger keywords for FPR testing
│   ├── deduplicate.py      # Exact & n-gram fuzzy deduplication engine
│   ├── split.py            # Stratified train/val/test splitting
│   ├── generate_data_card.py# Data card generator
│   └── build_dataset.py    # Master dataset generation pipeline
├── eval/                   # Unified evaluation engine
│   ├── metrics.py          # Precision, Recall, F1, FPR, ROC/PR AUC, 95% Bootstrap CI
│   ├── evaluator.py        # Benchmark harness for test split and hard negatives
│   ├── calibrate_threshold.py # Decision threshold tuner for target FPR
│   └── evaluate_lofo.py    # Leave-One-Attack-Family-Out cross-validation
├── models/                 # Firewall classification models
│   ├── baseline_tfidf.py   # TF-IDF + Logistic Regression baseline pipeline
│   └── deberta_detector.py # Production DeBERTa inference detector with token chunking
├── notebooks/              # GPU training notebooks
│   └── train_deberta_colab.ipynb # Colab notebook for free T4 GPU fine-tuning
├── payloads/
│   └── payloads.yaml       # 69 seed adversarial test payloads
├── training/               # Fine-tuning pipelines
│   ├── train_deberta.py    # DeBERTa-v3-small Hugging Face Trainer script
│   └── train_tradeoff.py   # Lightweight tradeoff model trainer (MiniLM/DistilBERT)
├── detectors/              # Scanner detectors (red-team harness)
│   ├── heuristic.py        # Exact marker substring & regex matching
│   ├── embedding.py        # Semantic similarity via sentence-transformers
│   └── llm_judge.py        # Structured evaluation via LLM-as-judge
├── webapp/                 # Flask dashboard for visualization and scan control
├── runner.py               # Sends payloads to target chat completion API
├── report.py               # Aggregates verdicts into JSON and HTML reports
├── run_baseline.py         # Single-command runner for baseline training & evaluation
├── run_generalization_benchmark.py # Single-command runner for LOFO & tradeoff benchmarks
├── main.py                 # CLI entrypoint for scanner
├── DATA_CARD.md            # Dataset provenance, splits, and taxonomy documentation
└── requirements.txt
```

---

## Setup & Quickstart

### 1. Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com) (or any OpenAI-compatible API endpoint)

```bash
# Clone the repository
git clone https://github.com/shivanshu-singh-dev/LLM-Prompt-Injection-Scanner.git
cd LLM-Prompt-Injection-Scanner

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Setup
To build or regenerate the dataset splits:

```bash
# Download public datasets (deepset, jackhhao)
python -m data.download_datasets

# Run the ingestion, deduplication, and stratified 80/10/10 splitting pipeline
python -m data.build_dataset
```

See [**`DATA_CARD.md`**](DATA_CARD.md) for full dataset provenance, license attribution, and taxonomy breakdown.

---

## Benchmarks & Evaluation

### 1. Headline Benchmark Comparison

Run the full benchmark suite across all models, leave-one-family-out cross-validation, and latency profiling:

```bash
python run_generalization_benchmark.py
```

| Model | Random F1 (95% CI) | Macro LOFO F1 | Generalization Gap | Hard Negatives FPR | p95 Latency (CPU) |
|---|---|---|---|---|---|
| **TF-IDF Baseline** | 0.9518 [0.915 - 0.981] | 0.6835 | **-0.2682** | 13.75% | **1.97 ms** |
| **DeBERTa-v3-small** | *Trained on Colab* | *Target: >0.85* | *Target: <-0.10* | *Target: <2.0%* | ~25 ms |

### 2. Generalization Breakdown: Leave-One-Attack-Family-Out (LOFO)

To expose the vulnerability of keyword-based models to unseen attack families, the LOFO benchmark trains on 3 families and tests on the 4th held-out family:

| Held-Out Attack Family | Random Split Detection | LOFO (Zero-Day) Detection | Accuracy Drop |
|---|---|---|---|
| `role_override` | 100.00% | 100.00% | 0.00% |
| `jailbreak` | 100.00% | 32.73% | **-67.27%** |
| `indirect` | 100.00% | 100.00% | 0.00% |
| `obedience_exploit` | 100.00% | 100.00% | 0.00% |

> [!IMPORTANT]
> The baseline model suffers a catastrophic **67.27% accuracy drop** when confronted with unseen jailbreaks. This empirically proves why keyword/TF-IDF models fail in real-world deployment when adversaries employ persona-switching or novel framing techniques.

---

## Fine-Tuning DeBERTa-v3 on Google Colab

1. Open [`notebooks/train_deberta_colab.ipynb`](notebooks/train_deberta_colab.ipynb) in Google Colab.
2. Select **Runtime -> Change runtime type -> T4 GPU**.
3. Upload `data/processed/train.jsonl` and `data/processed/val.jsonl`.
4. Run the notebook to fine-tune `microsoft/deberta-v3-small` with FP16, calibrate decision threshold for $<1\%$ validation FPR, and download `deberta-v3-firewall.zip`.
5. Extract the downloaded weights into `models/deberta-v3-firewall/`.

Alternatively, if training locally with a CUDA GPU:
```bash
python -m training.train_deberta --train-file data/processed/train.jsonl --val-file data/processed/val.jsonl
```

---

## Running the Security Scanner

The repository includes the adversarial evaluation scanner to benchmark models against prompt injection attacks:

### CLI Scanner
```bash
python main.py
```

**Options:**
- `--payloads <path>`: Custom payload YAML path (default: `payloads/payloads.yaml`)
- `--output <dir>`: Custom results directory (default: `results/`)
- `--skip-judge`: Run heuristic and embedding detectors only

### Web Dashboard
```bash
cd webapp && python app.py
```
Open `http://localhost:5000` to inspect payload execution logs, view compromised status, and review detector verdicts.

---

## License

MIT License. See individual dataset licenses in [DATA_CARD.md](DATA_CARD.md).
