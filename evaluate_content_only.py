"""
evaluate_content_only.py

STEP 2 of the ablation study: baseline measurement.

Runs the EXISTING content-only model (TF-IDF + Logistic Regression, trained
on 65,980 real emails, 98.99% accuracy on its own held-out test set) against
the fixed benchmark.py set -- the clean-worded attack/legitimate pairs.

This model has never seen the benchmark before and gets no relationship
history, no LLM reasoning -- just the raw email text, exactly as it was
designed to work.

Usage:
    python3 evaluate_content_only.py
"""

import joblib
from benchmark import BENCHMARK

MODEL_PATH = "phishing_model.joblib"  # copy your trained model next to this script


def run():
    model = joblib.load(MODEL_PATH)

    print("=" * 78)
    print("STEP 2: CONTENT-ONLY MODEL vs. THE CLEAN-WORDED BENCHMARK")
    print("=" * 78)
    print(f"Model: TF-IDF + Logistic Regression (98.99% accuracy on its own test set)")
    print(f"Benchmark: {len(BENCHMARK)} emails, {len(BENCHMARK)//2} matched attack/legit pairs")
    print(f"Note: this model only ever sees email TEXT -- no sender history, no context.\n")

    results = []
    for entry in BENCHMARK:
        text = entry["subject"] + " " + entry["body"]
        prob = model.predict_proba([text])[0][1]
        pred_label = 1 if prob >= 0.5 else 0
        correct = (pred_label == entry["label"])
        results.append({**entry, "prob": prob, "pred_label": pred_label, "correct": correct})

    # ---- per-email results ----
    print(f"{'ID':30s} {'True':8s} {'Predicted':10s} {'Prob':>7s}  {'Correct?'}")
    print("-" * 78)
    for r in results:
        true_lbl = "ATTACK" if r["label"] == 1 else "legit"
        pred_lbl = "ATTACK" if r["pred_label"] == 1 else "legit"
        mark = "correct" if r["correct"] else "WRONG"
        print(f"{r['id']:30s} {true_lbl:8s} {pred_lbl:10s} {r['prob']:>6.3f}  {mark}")

    # ---- overall accuracy ----
    accuracy = sum(r["correct"] for r in results) / len(results)
    print("-" * 78)
    print(f"\nOverall accuracy on benchmark: {accuracy:.1%} ({sum(r['correct'] for r in results)}/{len(results)})")

    # ---- the real finding: within-pair discrimination ----
    print("\n" + "=" * 78)
    print("THE KEY QUESTION: can it tell the attack apart from its matched legit twin?")
    print("=" * 78)
    categories = sorted(set(r["category"] for r in results))
    pair_gaps = []
    for cat in categories:
        pair = [r for r in results if r["category"] == cat]
        attack = next(r for r in pair if r["label"] == 1)
        legit = next(r for r in pair if r["label"] == 0)
        gap = attack["prob"] - legit["prob"]
        pair_gaps.append(gap)
        verdict = "correctly separated" if gap > 0.05 else "could NOT meaningfully separate"
        print(f"  {cat:28s} attack={attack['prob']:.3f}  legit={legit['prob']:.3f}  "
              f"gap={gap:+.3f}   -> {verdict}")

    avg_gap = sum(pair_gaps) / len(pair_gaps)
    meaningfully_separated = sum(1 for g in pair_gaps if g > 0.05)

    print(f"\nAverage attack-vs-legit probability gap across all pairs: {avg_gap:+.3f}")
    print(f"Pairs meaningfully separated (gap > 0.05): {meaningfully_separated}/{len(pair_gaps)}")
    print("\nInterpretation: within each pair, the wording is IDENTICAL by design.")
    print("Any separation the model shows here is coming from something other than")
    print("the actual risk signal (sender history, request fit) -- it has no access")
    print("to that information at all. This number is the honest baseline that")
    print("Steps 3-5 (relationship memory, LLM reasoning, fusion) need to improve on.")


if __name__ == "__main__":
    run()
