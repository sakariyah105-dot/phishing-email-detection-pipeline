"""
evaluate_content_plus_relationship.py

STEP 3 of the ablation study: content model + relationship memory, combined.

Same fixed benchmark as Step 2 (benchmark.py), same content model
(phishing_model.joblib) -- the ONLY thing added is relationship_memory.py,
seeded with each entry's prior_history before scoring.

Fusion weights (content 50% / relationship 50%) are a reasonable starting
point chosen BEFORE looking at benchmark results, not tuned to this specific
benchmark -- tuning weights on the same set you're measuring against would
make the "improvement" number meaningless.

Usage:
    python3 evaluate_content_plus_relationship.py
"""

import sqlite3
import datetime
import joblib
from benchmark import BENCHMARK
from relationship_memory import RelationshipMemory

MODEL_PATH = "phishing_model.joblib"

CONTENT_WEIGHT = 0.5
RELATIONSHIP_WEIGHT = 0.5


def seed_history(rm: RelationshipMemory, sender, recipient, prior_history):
    """prior_history is a list of (request_type, days_ago) tuples from the
    benchmark. Convert to timestamps and record each one as real history,
    oldest first, so score_email() sees a realistic-looking timeline."""
    today = datetime.datetime.utcnow()
    # oldest first so first_contact_date comes out correct
    for request_type, days_ago in sorted(prior_history, key=lambda x: -x[1]):
        ts = (today - datetime.timedelta(days=days_ago)).isoformat()
        rm.record_email(sender, recipient, request_type, timestamp=ts)


def run():
    model = joblib.load(MODEL_PATH)

    print("=" * 82)
    print("STEP 3: CONTENT MODEL + RELATIONSHIP MEMORY vs. THE SAME BENCHMARK")
    print("=" * 82)
    print(f"Fusion: {CONTENT_WEIGHT:.0%} content score + {RELATIONSHIP_WEIGHT:.0%} relationship score")
    print("(weights fixed in advance, not tuned on this benchmark)\n")

    results = []
    for entry in BENCHMARK:
        # fresh in-memory relationship DB per email, seeded ONLY with this
        # entry's own history -- pairs must not leak context into each other
        rm = RelationshipMemory(":memory:")
        seed_history(rm, entry["sender"], entry["recipient"], entry["prior_history"])

        text = entry["subject"] + " " + entry["body"]
        content_prob = model.predict_proba([text])[0][1]
        content_score = content_prob * 100

        rel_result = rm.score_email(entry["sender"], entry["recipient"], entry["request_type"])
        rel_score = rel_result.risk_score

        fused = CONTENT_WEIGHT * content_score + RELATIONSHIP_WEIGHT * rel_score
        pred_label = 1 if fused >= 50 else 0
        correct = (pred_label == entry["label"])

        results.append({
            **entry, "content_score": content_score, "rel_score": rel_score,
            "fused": fused, "pred_label": pred_label, "correct": correct,
            "rel_flags": rel_result.flags,
        })
        rm.close()

    # ---- per-email results ----
    print(f"{'ID':30s} {'True':7s} {'Content':>8s} {'Relat.':>7s} {'Fused':>7s} {'Pred':>7s}  {'Correct?'}")
    print("-" * 82)
    for r in results:
        true_lbl = "ATTACK" if r["label"] == 1 else "legit"
        pred_lbl = "ATTACK" if r["pred_label"] == 1 else "legit"
        mark = "correct" if r["correct"] else "WRONG"
        print(f"{r['id']:30s} {true_lbl:7s} {r['content_score']:7.1f}  {r['rel_score']:6.1f} "
              f"{r['fused']:6.1f}  {pred_lbl:6s}  {mark}")

    accuracy = sum(r["correct"] for r in results) / len(results)
    print("-" * 82)
    print(f"\nOverall accuracy on benchmark: {accuracy:.1%} ({sum(r['correct'] for r in results)}/{len(results)})")

    # ---- pair separation ----
    print("\n" + "=" * 82)
    print("PAIR SEPARATION: fused score, attack vs. matched legit twin")
    print("=" * 82)
    categories = sorted(set(r["category"] for r in results))
    gaps = []
    for cat in categories:
        pair = [r for r in results if r["category"] == cat]
        attack = next(r for r in pair if r["label"] == 1)
        legit = next(r for r in pair if r["label"] == 0)
        gap = attack["fused"] - legit["fused"]
        gaps.append(gap)
        verdict = "correctly separated" if gap > 5 else "still NOT separated"
        print(f"  {cat:28s} attack={attack['fused']:5.1f}  legit={legit['fused']:5.1f}  "
              f"gap={gap:+6.1f}   -> {verdict}")

    avg_gap = sum(gaps) / len(gaps)
    separated = sum(1 for g in gaps if g > 5)

    print(f"\nAverage attack-vs-legit fused-score gap: {avg_gap:+.1f}")
    print(f"Pairs meaningfully separated (gap > 5): {separated}/{len(gaps)}")

    print("\n" + "=" * 82)
    print("COMPARISON TO STEP 2 BASELINE")
    print("=" * 82)
    print("  Step 2 (content only):              50.0% accuracy, 0/7 pairs separated, avg gap +0.000")
    print(f"  Step 3 (content + relationship):    {accuracy:.1%} accuracy, {separated}/7 pairs separated, avg gap {avg_gap:+.1f}")

    # show one worked example of WHY it improved
    print("\n" + "=" * 82)
    print("WORKED EXAMPLE: why the relationship layer catches what content alone missed")
    print("=" * 82)
    example = next(r for r in results if r["id"] == "vendor_payment_attack")
    print(f"Email: \"{example['subject']}\"")
    print(f"Content-model score alone: {example['content_score']:.1f}/100 (barely registers as risky)")
    print(f"Relationship-memory flags:")
    for f in example["rel_flags"]:
        print(f"  - {f}")
    print(f"Fused score: {example['fused']:.1f}/100")


if __name__ == "__main__":
    run()
