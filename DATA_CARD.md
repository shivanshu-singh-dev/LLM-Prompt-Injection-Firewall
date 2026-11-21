# Data Card: Prompt Injection Firewall Dataset

**Date Generated:** 2026-11-21  
**Project:** LLM Prompt Injection Firewall  
**Total Examples:** 1890  

---

## 1. Overview & Purpose
This dataset is curated specifically to train and evaluate production prompt injection firewalls. It combines red-team payloads from the original scanner test suite with verified public injection and jailbreak datasets, balanced by benign queries and hard negatives.

---

## 2. Dataset Sources & Licenses

| Source | Count | Proportion | License | Description |
|---|---|---|---|---|
| `scanner_payloads` | 69 | 3.7% | MIT | Original 69 adversarial payloads targeting system prompt extraction & instruction override |
| `deepset/prompt-injections` | 653 | 34.6% | Apache-2.0 | Curated public prompt injections and benign controls |
| `jackhhao/jailbreak-classification` | 1168 | 61.8% | MIT | Community jailbreak prompts and conversational benign requests |

---

## 3. Split Distribution (80 / 10 / 10 Stratified)

All splits are stratified across `(label, attack_family)` pairs with random seed `42`.

| Split | Count | Benign (0) | Malicious (1) | % of Total |
|---|---|---|---|---|
| **Train** | 1510 | 828 | 682 | 79.9% |
| **Validation** | 190 | 104 | 86 | 10.1% |
| **Test** | 190 | 104 | 86 | 10.1% |
| **Total** | **1890** | **1036** | **854** | **100.0%** |

> [!IMPORTANT]
> The **Test** split is strictly held out. It must never be used during model training, hyperparameter tuning, or threshold calibration.

---

## 4. Attack Families & Taxonomy

Every example is assigned an `attack_family` to support Leave-One-Family-Out (LOFO) generalization benchmarking:

| Attack Family | Count | Primary Focus |
|---|---|---|
| `benign` | 1036 | Legitimate user queries, conversational prompts, and benign helpdesk questions |
| `role_override` | 20 | Policy puppetry (XML/JSON), system role spoofing, authority framing, credential claims |
| `jailbreak` | 550 | Persona switching (DAN/AIM), hypothetical framing, emotional coercion |
| `indirect` | 20 | Embedded malicious instructions in secondary data (logs, tickets, emails) |
| `obedience_exploit` | 9 | Syntactic obedience traps (verbatim repetition, grammar correction, completion) |
| `prompt_injection` | 255 | Direct prompt injection patterns from public datasets |

---

## 5. Deduplication & Data Quality

To prevent data leakage across splits and avoid artificial accuracy inflation:
- **Exact matches removed:** 27
- **Near-duplicates removed:** 120 (evaluated via character 3-gram Jaccard similarity ≥ 0.88)
- Scanner payloads are given highest retention priority during deduplication.
