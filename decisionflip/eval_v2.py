"""
DecisionFlip pilot evaluation, v2 (Section 16 steps 3-5: measure similarity,
test decision discrimination, inspect failures).

Per category it reports:
  1. Similarity: cos(anchor, flip) vs cos(anchor, paraphrase) vs unrelated controls,
     plus the flip's percentile among ALL pairwise cosines in the corpus.
  2. Margin test: fraction of triples where the flip is at least as close to the
     anchor as a same-decision paraphrase is (= geometry does not separate them).
  3. Cosine AUROC: how well raw cosine separates "same decision" (paraphrase)
     from "different decision" (flip). 0.5 = no separation.
  4. Decision test:
       - numerical/scope/permission/action: logistic regression on frozen
         embeddings, leave-one-FAMILY-out (unseen templates, Section 12).
         Reports decision accuracy and flip consistency (anchor AND flip correct).
       - tool_function: hard-negative ranking against tool descriptions.
  5. Wilson 95% intervals, because n is small.

Usage:
    pip install sentence-transformers scikit-learn
    python eval_v2.py --data pilot_v2.json --model all-MiniLM-L6-v2
    python eval_v2.py --data pilot_v2.json --model BAAI/bge-small-en-v1.5
    python eval_v2.py --data pilot_v2.json --model hashing   # offline smoke test only
"""

import argparse
import json
import math
import random
import re
import zlib
from collections import defaultdict

import numpy as np


# ---------------------------------------------------------------------------
# encoders
# ---------------------------------------------------------------------------

def make_encoder(name):
    if name == "hashing":  # bag-of-words smoke-test encoder, NOT a real baseline
        def enc(texts, dim=256):
            E = np.zeros((len(texts), dim))
            for i, t in enumerate(texts):
                for tok in re.findall(r"[a-z0-9$#']+", t.lower()):
                    E[i, zlib.crc32(tok.encode()) % dim] += 1.0
            return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
        return enc
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(name)
    return lambda texts: model.encode(texts, normalize_embeddings=True, batch_size=64)


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def fmt(k, n):
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {k / max(n, 1):.2f} [{lo:.2f},{hi:.2f}]"


# ---------------------------------------------------------------------------
# grouped evaluation helpers
# ---------------------------------------------------------------------------

def evaluation_splits(records, split, n_folds=5, seed=13):
    if split == "within-family":
        rng = random.Random(seed)
        grouped = defaultdict(list)
        for record in records:
            grouped[record["family"]].append(record)
        folds = [[] for _ in range(n_folds)]
        for group_records in grouped.values():
            rng.shuffle(group_records)
            for index, record in enumerate(group_records):
                folds[index % n_folds].append(record)
        return [
            ([record for index, fold in enumerate(folds) if index != held for record in fold],
             folds[held])
            for held in range(n_folds) if folds[held]
        ]

    key_name = {
        "heldout-family": "family",
        "heldout-policy": "policy_id",
        "heldout-template": "template_id",
    }[split]
    if split != "heldout-family":
        missing = [record["id"] for record in records if key_name not in record]
        if missing:
            raise ValueError(f"{split} requires {key_name!r} on every record")
    groups = sorted({record.get(key_name, record["family"]) for record in records})
    return [
        ([record for record in records if record.get(key_name, record["family"]) != group],
         [record for record in records if record.get(key_name, record["family"]) == group])
        for group in groups
    ]


def bootstrap_auc_by_family(flips, paraphrases, records, roc_auc_score,
                            n_bootstrap=1000, seed=17):
    grouped = defaultdict(list)
    for index, record in enumerate(records):
        grouped[record["family"]].append(index)
    families = list(grouped.values())
    if not families:
        return {"low": float("nan"), "high": float("nan")}

    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_bootstrap):
        selected = rng.integers(0, len(families), size=len(families))
        indices = [index for family_index in selected for index in families[family_index]]
        labels = np.r_[np.ones(len(indices)), np.zeros(len(indices))]
        scores = np.r_[paraphrases[indices], flips[indices]]
        samples.append(roc_auc_score(labels, scores))
    low, high = np.percentile(samples, [2.5, 97.5])
    return {"low": float(low), "high": float(high)}


def evaluate_classifier(records, feature_mode, split, emb, LogisticRegression,
                        TfidfVectorizer):
    results = {"decision_acc": 0, "n_decisions": 0,
               "flip_consistency": 0, "n_pairs": 0, "failures": []}
    keys = ("text_a", "text_b", "text_para")
    label_keys = ("label_a", "label_b", "label_para")

    for train_records, test_records in evaluation_splits(records, split):
        if not train_records or not test_records:
            continue
        train_texts = [record[key] for record in train_records for key in keys]
        train_labels = [record[key] for record in train_records for key in label_keys]
        if len(set(train_labels)) < 2:
            continue

        if feature_mode == "embedding":
            train_features = np.array([emb(text) for text in train_texts])
            test_features = lambda record: np.array([emb(record[key]) for key in keys])
        else:
            vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
            train_features = vectorizer.fit_transform(train_texts)
            test_features = lambda record: vectorizer.transform(
                [record[key] for key in keys]
            )

        classifier = LogisticRegression(C=10.0, max_iter=2000)
        classifier.fit(train_features, train_labels)
        for record in test_records:
            predictions = classifier.predict(test_features(record))
            correct = [prediction == record[label_key]
                       for prediction, label_key in zip(predictions, label_keys)]
            results["decision_acc"] += sum(correct)
            results["n_decisions"] += len(correct)
            flip_correct = correct[0] and correct[1]
            results["flip_consistency"] += int(flip_correct)
            results["n_pairs"] += 1
            if not flip_correct:
                results["failures"].append({
                    "category": record["category"], "id": record["id"],
                    "system": feature_mode, "split": split,
                    "text_a": record["text_a"], "text_b": record["text_b"],
                    "prediction_a": str(predictions[0]),
                    "prediction_b": str(predictions[1]),
                    "label_a": record["label_a"], "label_b": record["label_b"],
                })

    results["decision_acc"] /= max(results["n_decisions"], 1)
    results["flip_consistency"] /= max(results["n_pairs"], 1)
    return results


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="pilot_v2.json")
    ap.add_argument("--model", default="all-MiniLM-L6-v2")
    ap.add_argument("--out", default="results_v2.json")
    args = ap.parse_args()

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    with open(args.data) as f:
        data = json.load(f)
    triples, controls = data["triples"], data["controls"]

    texts = set()
    for t in triples:
        texts.update([t["text_a"], t["text_b"], t["text_para"]])
        if "desc_a" in t:
            texts.update([t["desc_a"], t["desc_b"]])
    for c in controls:
        texts.update([c["text_a"], c["text_b"]])
    texts = sorted(texts)
    idx = {t: i for i, t in enumerate(texts)}

    print(f"Encoding {len(texts)} unique texts with {args.model} ...")
    E = np.asarray(make_encoder(args.model)(texts), dtype=float)
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)

    def emb(t):
        return E[idx[t]]

    # corpus-wide pairwise cosines for percentile reporting (Section 14)
    S = E @ E.T
    all_sims = np.sort(S[np.triu_indices(len(texts), k=1)])

    def pct(x):
        return float(np.searchsorted(all_sims, x) / len(all_sims))

    ctrl = [float(emb(c["text_a"]) @ emb(c["text_b"])) for c in controls]
    ctrl_mean = float(np.mean(ctrl))

    by_cat = defaultdict(list)
    for t in triples:
        by_cat[t["category"]].append(t)

    report = {"model": args.model, "control_mean": ctrl_mean, "categories": {},
              "bootstrap_unit": "family"}
    failures = []

    print(f"\nUnrelated-control mean cosine: {ctrl_mean:.3f} (n={len(ctrl)})")
    print("\n=== 1-3. Primary categories: similarity and cosine AUROC ===")
    print(f"{'category':<22}{'n':>3} {'flip_cos':>9} {'para_cos':>9} {'flip_pct':>9}  "
          f"{'flip>=para':<15}{'AUROC [family bootstrap 95% CI]':>34}")
    for cat, ts in by_cat.items():
        fc = np.array([emb(t["text_a"]) @ emb(t["text_b"]) for t in ts])
        pc = np.array([emb(t["text_a"]) @ emb(t["text_para"]) for t in ts])
        fails = int(np.sum(fc >= pc))
        y = np.r_[np.ones(len(pc)), np.zeros(len(fc))]  # 1 = same decision
        auroc = float(roc_auc_score(y, np.r_[pc, fc]))
        auc_ci = bootstrap_auc_by_family(fc, pc, ts, roc_auc_score)
        pcts = [pct(x) for x in fc]
        if cat != "numerical_threshold":
            print(f"{cat:<22}{len(ts):>3} {fc.mean():>9.3f} {pc.mean():>9.3f} "
                  f"{np.mean(pcts):>9.2f}  {fmt(fails, len(ts)):<15}"
                  f"{auroc:.2f} [{auc_ci['low']:.2f},{auc_ci['high']:.2f}]")
        report["categories"][cat] = {
            "role": "calibration" if cat == "numerical_threshold" else "primary",
            "n": len(ts), "flip_cos_mean": float(fc.mean()), "para_cos_mean": float(pc.mean()),
            "flip_percentile_mean": float(np.mean(pcts)),
            "margin_fail": fails, "cos_auroc": auroc,
            "cos_auroc_family_bootstrap_95ci": auc_ci,
        }
        for t, a, b in zip(ts, fc, pc):
            if a >= b:
                failures.append({"category": cat, "system": "cosine_margin",
                                 "id": t["id"], "text_a": t["text_a"],
                                 "text_b": t["text_b"],
                                 "detail": f"flip_cos={a:.3f} para_cos={b:.3f}"})

    if "numerical_threshold" in by_cat:
        numeric = report["categories"]["numerical_threshold"]
        print("\nNumerical threshold (calibration only): "
              f"n={numeric['n']}, AUROC={numeric['cos_auroc']:.2f}, "
              f"family-bootstrap 95% CI="
              f"[{numeric['cos_auroc_family_bootstrap_95ci']['low']:.2f},"
              f"{numeric['cos_auroc_family_bootstrap_95ci']['high']:.2f}]")

    print("\n=== 4. Decision probes: embedding vs TF-IDF ===")
    print("Paired records are kept intact; each fold tests A, B, and the paraphrase.")
    for cat, ts in by_cat.items():
        if cat == "tool_function":
            continue
        print(f"\n[{cat}]" + (" (calibration only)" if cat == "numerical_threshold" else ""))
        cat_results = {}
        for split in ("within-family", "heldout-family"):
            cat_results[split] = {}
            for feature_mode in ("embedding", "tfidf"):
                scores = evaluate_classifier(
                    ts, feature_mode, split, emb, LogisticRegression, TfidfVectorizer
                )
                cat_results[split][feature_mode] = {
                    key: value for key, value in scores.items() if key != "failures"
                }
                failures.extend(scores["failures"])
                print(f"  {split:<16} {feature_mode:<10} "
                      f"decision={scores['decision_acc']:.3f} "
                      f"flip={scores['flip_consistency']:.3f} "
                      f"(n={scores['n_pairs']})")
        report["categories"][cat]["decision_probes"] = cat_results

    if "tool_function" in by_cat:
        ts = by_cat["tool_function"]
        rank_ok = 0
        for record in ts:
            desc_a, desc_b = emb(record["desc_a"]), emb(record["desc_b"])
            correct_a = emb(record["text_a"]) @ desc_a > emb(record["text_a"]) @ desc_b
            correct_b = emb(record["text_b"]) @ desc_b > emb(record["text_b"]) @ desc_a
            rank_ok += int(correct_a and correct_b)
            if not (correct_a and correct_b):
                failures.append({"category": "tool_function", "system": "embedding_rank",
                                 "id": record["id"], "text_a": record["text_a"],
                                 "text_b": record["text_b"],
                                 "detail": f"a_ok={bool(correct_a)} b_ok={bool(correct_b)}"})
        report["categories"]["tool_function"]["embedding_rank_flip_consistency"] = (
            rank_ok / max(len(ts), 1)
        )

    failures_by_category = defaultdict(list)
    for failure in failures:
        failures_by_category[failure["category"]].append(failure)
    print("\n=== 5. All failures by category, split, and system ===")
    for cat in by_cat:
        category_failures = failures_by_category[cat]
        print(f"\n[{cat}] {len(category_failures)} failures")
        for failure in category_failures:
            print(f"  ({failure.get('split', 'cosine_or_rank')}/"
                  f"{failure['system']}) {failure['id']}: "
                  f"{failure['text_a']} -> {failure['text_b']}")
    report["n_failures"] = len(failures)
    report["failures_by_category"] = {
        cat: len(category_failures)
        for cat, category_failures in failures_by_category.items()
    }

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"\nResults -> {args.out}")
    print("\nInterpret these as pilot results; family counts, not just example counts, "
          "limit generalization claims.")


if __name__ == "__main__":
    main()
