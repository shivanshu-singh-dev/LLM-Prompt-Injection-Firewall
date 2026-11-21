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
- **Robust Dataset & Benchmark**: Curated datasets categorized into 4 attack families (`role_override`, `jailbreak`, `indirect`, `obedience_exploit`) plus benign controls and hard negatives.
- **Hard-Negative Evaluation**: Dedicated benchmark testing False Positive Rate (FPR) on benign prompts containing suspicious trigger keywords (`ignore`, `override`, `system`, technical logs).
- **Zero-Day Generalization Testing**: Leave-One-Attack-Family-Out (LOFO) evaluation methodology to measure defense against unseen attack strategies.
- **OpenAI-Compatible Proxy**: Drop-in reverse proxy compatible with any standard OpenAI chat completion client.
- **Red-Team Scanner Harness**: 69 seed adversarial test vectors across multiple injection techniques with CLI and web visualization.

---

## Repository Structure

```
.
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
│   └── evaluator.py        # Benchmark harness for test split and hard negatives
├── models/                 # Firewall classification models
│   └── baseline_tfidf.py   # TF-IDF + Logistic Regression baseline pipeline
├── payloads/
│   └── payloads.yaml       # 69 seed adversarial test payloads
├── detectors/
│   ├── heuristic.py        # Exact marker substring & regex matching
│   ├── embedding.py        # Semantic similarity via sentence-transformers
│   └── llm_judge.py        # Structured evaluation via LLM-as-judge
├── webapp/                 # Flask dashboard for visualization and scan control
├── runner.py               # Sends payloads to target chat completion API
├── report.py               # Aggregates verdicts into JSON and HTML reports
├── run_baseline.py         # Single-command runner for baseline training & evaluation
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

## Baseline Classifier & Evaluation

To train the TF-IDF + Logistic Regression baseline and evaluate it against the held-out test set and hard negatives:

```bash
python run_baseline.py
```

### Baseline Benchmark Results

| Model | Test F1 (95% CI) | Test Recall | Test FPR (95% CI) | Hard Negatives FPR (95% CI) | p95 Latency (CPU) |
|---|---|---|---|---|---|
| **TF-IDF + LogReg (Default 0.50)** | 0.9518 [0.916 - 0.981] | 91.86% | 0.96% [0.0% - 3.1%] | **13.75%** [7.5% - 21.3%] | 1.97 ms |
| **TF-IDF + LogReg (Calibrated 0.47)** | 0.9581 [0.925 - 0.986] | 93.02% | 0.96% [0.0% - 3.1%] | **21.25%** [12.5% - 30.0%] | 1.97 ms |

> [!NOTE]
> The baseline achieves high F1 on random splits but flags **13.75% - 21.25%** of benign queries containing trigger words (e.g. `ignore`, `system`, `override`) as prompt injections. Minimizing this Hard Negative FPR is the primary objective of the upcoming DeBERTa-v3 transformer classifier.

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
