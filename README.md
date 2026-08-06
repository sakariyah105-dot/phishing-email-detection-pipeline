# Phishing Email Detector — Context-Aware Detection Pipeline

A phishing detector that combines a trained text-classification model with a relationship-history layer — built and measured specifically to catch what word-based detection alone misses.

![Pipeline architecture](detection_pipeline_v2.svg)

## The problem

Word-based phishing filters were built for a threat that's largely gone. Current data on phishing in 2026:

- **82–86% of phishing emails are now AI-generated** — fluent, well-punctuated, no typos
- AI-written phishing gets a **54% click rate**, versus **12%** for old-style manual phishing
- Security researchers now describe the "bad grammar = phishing" heuristic as dead — the signal most keyword/NLP filters were built around no longer reliably exists in real attacks

A model that scores emails purely on wording has a structural blind spot: a well-written, contextually-plausible request with no suspicious vocabulary at all is invisible to it, no matter how well-trained it is.

## What this project does about it

Rather than just building a better word classifier, this project **measures that blind spot directly**, then adds a second, independent signal designed specifically to close it.

1. **Built a hand-crafted benchmark** of 14 emails, in 7 matched attack/legitimate pairs, where the wording is *identical* within each pair — isolating context (sender history, request fit) as the only variable that separates a real attack from a real request.
2. **Measured the content-only model against it** — the honest baseline.
3. **Added a relationship-memory layer** and measured the same benchmark again — the honest improvement.

## Results

### Content model, standard test set
Trained on 65,980 real emails (CEAS_08, Nazario, Enron, SpamAssassin, Ling, Nigerian_Fraud — 82,475 total), evaluated on 16,495 held-out emails it never saw during training:

| Metric | Score |
|---|---|
| Accuracy | 0.9899 |
| Precision | 0.9892 |
| Recall | 0.9914 |
| F1 | 0.9903 |
| ROC-AUC | 0.9991 |

This is a strong result — on the kind of phishing this training data represents.

### The real test: the clean-worded benchmark

| Stage | Accuracy | Pairs correctly separated | Avg. attack-vs-legit score gap |
|---|---|---|---|
| Content model only | **50.0%** | 0 / 7 | +0.000 |
| Content + relationship memory | **78.6%** | 7 / 7 | +29.3 |

The content-only model scored *exactly identically* on every attack and its matched legitimate twin — proof that a 99%-accurate model on standard data is at chance level (a coin flip) against context-dependent, clean-worded attacks. Adding relationship history — with no NLP, no LLM, just SQLite lookups — closed most of that gap.

**Worked example** (`vendor_payment_attack`, first-ever contact requesting a bank detail change):
```
Content-model score alone:  28.1/100  (barely registers as risky)
Relationship-memory flags:
  - First-ever contact between this sender and recipient on record.
  - A 'payment' request arriving on the very first contact is a
    classic pretexting pattern.
Fused score: 54.0/100
```

## Architecture

One email is scored by two independent signals, then combined:

- **Content model** (`TF-IDF` + `Logistic Regression`) — scores based purely on wording. Blind to who sent it or whether it fits any pattern.
- **Relationship memory** (SQLite-backed history) — scores based purely on behavior: has this sender contacted this recipient before, has this specific request type happened between them before. Completely blind to wording.
- **Fusion** — `final_score = 50% content_score + 50% relationship_score`. Weights were fixed *before* running the benchmark, not tuned to it, so the reported improvement isn't circular.

## Repository structure

```
├── benchmark.py                          # 14-email clean-worded evaluation set (7 matched pairs)
├── relationship_memory.py                # SQLite-backed sender/recipient history + scoring
├── evaluate_content_only.py              # Step 2: baseline measurement
├── evaluate_content_plus_relationship.py # Step 3: measured improvement
├── phishing_model.joblib                 # pre-trained TF-IDF + Logistic Regression pipeline
├── detection_pipeline_v2.svg             # architecture diagram (this README's image)
└── tfidf_explained.md                    # worked example of one prediction, step by step
```

## Setup

```bash
pip install pandas scikit-learn joblib
```

## Usage

Run the two evaluation stages, in order, from the same folder as `benchmark.py`, `relationship_memory.py`, and `phishing_model.joblib`:

```bash
python3 evaluate_content_only.py               # Step 2: baseline (50.0%)
python3 evaluate_content_plus_relationship.py   # Step 3: with relationship memory (78.6%)
```

## Limitations

- **Benchmark size is small (n=14).** Enough to demonstrate the effect clearly and reproducibly, not enough to claim a precise, generalizable accuracy figure. A larger, more diverse benchmark would strengthen the finding.
- **2 of 14 benchmark emails are still misclassified** even with relationship memory — both involve request types (`urgent`, ambiguous first contact) where the current scoring logic is weaker. Documented, not hidden.
- **Scope is email text only.** Current phishing increasingly happens over voice, SMS, and deepfake video — entirely outside what this system can see.
- **Content model's training data is dated** (largely 2000s-era Enron/CEAS emails) — real-world accuracy against current phishing is very likely lower than the 98.99% reported on its own held-out set.
- **LLM reasoning layer (a third planned signal) was designed but not implemented** — it would ask an LLM to reason about intent/plausibility for ambiguous cases, as a further signal alongside content and relationship scoring. Requires a paid API and was out of scope for this version.
- **Fusion weights (50/50) are a reasonable starting choice, not learned or tuned** — a production version would fit these weights on a larger labeled dataset rather than fixing them by hand.

## Tech stack

Python, pandas, scikit-learn (TF-IDF, Logistic Regression), SQLite (built into Python's standard library), joblib.

## Why this approach

Real production phishing detection is layered — authentication checks, attachment scanning, content scoring, behavioral analysis, and human review, stacked together, because no single layer catches everything. This project demonstrates and measures one specific, well-documented gap in the content-scoring layer, and shows that a lightweight behavioral layer meaningfully closes it — without needing more training data, a bigger model, or an LLM.
