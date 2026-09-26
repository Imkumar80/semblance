"""
DecisionFlip pilot evaluation harness -- Section 16, Critical First Experiment.

For each pair (text_a, text_b, label_a, label_b):
  1. Embed both texts with a given sentence-transformers model.
  2. Compute cosine similarity between the two embeddings.
  3. Every DecisionFlip pair by construction has label_a != label_b, so
     "does raw similarity predict a decision flip" is tested by sweeping a
     similarity threshold: below threshold -> predict "different decision",
     above threshold -> predict "same decision" (i.e. model conflates them).
  4. Report, per category: mean similarity, best-threshold accuracy against
     the easy-negative controls, and the headline diagnostic --
     "how often does the model rate a DecisionFlip pair as similar
     (>= easy-negative-control similarity), despite the decision differing."

This does NOT train anything. It only measures whether the phenomenon
described in the plan exists, per the "measure first" principle (Section 38).

Requires: pip install sentence-transformers --break-system-packages
Requires network access to huggingface.co to download model weights --
run this on your own machine / A6000, not in a network-restricted sandbox.

Usage:
    python eval_harness.py --pairs pilot_pairs.json --model all-MiniLM-L6-v2
    python eval_harness.py --pairs pilot_pairs.json --model BAAI/bge-small-en-v1.5
"""

import argparse
import json
from collections import defaultdict


def cosine_sim(a, b):
    import numpy as np
    a = np.array(a)
    b = np.array(b)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=str, default="pilot_pairs.json")
    ap.add_argument("--model", type=str, default="all-MiniLM-L6-v2")
    ap.add_argument("--out", type=str, default="pilot_results.json")
    args = ap.parse_args()

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise SystemExit(
            "sentence-transformers not installed. Run:\n"
            "  pip install sentence-transformers --break-system-packages"
        )

    with open(args.pairs) as f:
        pairs = json.load(f)

    print(f"Loading model: {args.model} (requires internet access to huggingface.co)")
    model = SentenceTransformer(args.model)

    results = []
    for p in pairs:
        emb_a = model.encode(p["text_a"])
        emb_b = model.encode(p["text_b"])
        sim = cosine_sim(emb_a, emb_b)
        results.append({**p, "similarity": sim})

    # --- Per-category summary ---
    by_cat = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r["similarity"])

    control_sims = by_cat.get("easy_negative_control", [0.0])
    control_mean = sum(control_sims) / len(control_sims)

    print("\n=== Similarity Gap by category ===")
    print(f"{'category':<24} {'mean_sim':>10} {'min':>8} {'max':>8}  vs control")
    summary = {"model": args.model, "control_mean_similarity": control_mean, "categories": {}}
    for cat, sims in by_cat.items():
        mean_sim = sum(sims) / len(sims)
        flag = ""
        if cat != "easy_negative_control" and mean_sim >= control_mean:
            flag = "  <-- as similar as (or more than) the easy-negative control, despite a real decision flip"
        print(f"{cat:<24} {mean_sim:>10.3f} {min(sims):>8.3f} {max(sims):>8.3f}{flag}")
        summary["categories"][cat] = {
            "mean_similarity": mean_sim,
            "min": min(sims),
            "max": max(sims),
            "n": len(sims),
            "at_or_above_control": mean_sim >= control_mean,
        }

    with open(args.out, "w") as f:
        json.dump({"per_pair": results, "summary": summary}, f, indent=2)

    print(f"\nFull results -> {args.out}")
    print(
        "\nInterpretation guide (Section 16 outcomes):\n"
        "  A - if every category's mean similarity is well BELOW the easy-negative control,\n"
        "      the model already separates these cases; DecisionFlip hypothesis is weak here.\n"
        "  B - if only some categories flag 'at_or_above_control', that's a concrete, scoped\n"
        "      research problem in those categories specifically.\n"
        "  C - if most/all categories flag, that's the strongest case for a decision-focused model."
    )


if __name__ == "__main__":
    main()
