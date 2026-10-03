"""
hard_negatives.py -- deterministic single-fact flips applied directly to a SUPPORTED
claim, independent of whether VitaminC happens to have a matching REFUTES edit.

This is additive to prepare_vitaminc.py's `refuting_premise` (a real Wikipedia edit) and
synth_multichunk.py's `hop_refute` (which uses it). Those only fire when VitaminC
naturally has a refuting pair for that claim; this generator fires on ANY supported claim
with a detectable flip target, so number/date/direction/entity flip types stay balanced
rather than following whatever Wikipedia editors happened to edit.

Input JSONL: {id, source, premise, claim, label, [refuting_premise], [split]}
  (the same file prepare_vitaminc.py / synth_multichunk.py already read/write)

Output: for each label==1 row with >=1 detectable flip, one extra row per flip type found:
  {chunks: [premise], claim: <flipped claim>, label: 0, type: "flip_number" |
   "flip_date" | "flip_direction" | "flip_entity", flip_detail: {...}, source, split}

The ORIGINAL supported row is not duplicated here -- synth_multichunk.py already emits
the positive (pad_pos) version. This script only adds the flipped negative, so the pair
stays matched: same claim-template, one word different, opposite label.

Requires: pip install spacy && python -m spacy download en_core_web_sm  (for entity flips;
number/date/direction flips are regex-only and work without spaCy).
"""

import argparse, json, random, re

DIRECTION_PAIRS = [
    ("increased", "decreased"),
    ("increase", "decrease"),
    ("rose", "fell"),
    ("rise", "fall"),
    ("grew", "shrank"),
    ("growth", "decline"),
    ("approved", "rejected"),
    ("before", "after"),
    ("more than", "less than"),
    ("above", "below"),
    ("gained", "lost"),
    ("won", "lost"),
    ("higher", "lower"),
    ("improved", "worsened"),
    ("expanded", "contracted"),
]
DIR_LOOKUP = {}
for a, b in DIRECTION_PAIRS:
    DIR_LOOKUP[a] = b
    DIR_LOOKUP[b] = a

NUMBER_RE = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(%|percent)?")
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
DIR_RE = re.compile(
    r"\b("
    + "|".join(sorted((re.escape(k) for k in DIR_LOOKUP), key=len, reverse=True))
    + r")\b",
    re.IGNORECASE,
)


def flip_number(text, rng):
    """Flip the first standalone number by +/-(10-50)%, min change of 1. Skips bare years."""
    for m in NUMBER_RE.finditer(text):
        raw = m.group(1)
        if YEAR_RE.fullmatch(raw) and not m.group(2):
            continue  # looks like a year, not a quantity -> leave to flip_date
        val = float(raw.replace(",", ""))
        delta = max(1.0, val * rng.uniform(0.10, 0.50))
        # Decreasing zero would leave the claim unchanged. Zero-valued quantities must
        # therefore always move upward to produce a valid hard negative.
        new_val = val + delta if val == 0.0 or rng.random() < 0.5 else max(0.0, val - delta)
        new_str = str(int(new_val)) if float(new_val).is_integer() else f"{new_val:.1f}"
        flipped = text[: m.start(1)] + new_str + text[m.end(1) :]
        return flipped, {"from": raw, "to": new_str}
    return None


def flip_date(text, rng):
    m = YEAR_RE.search(text)
    if not m:
        return None
    year = int(m.group(0))
    new_year = year + rng.choice([-3, -2, -1, 1, 2, 3])
    flipped = text[: m.start()] + str(new_year) + text[m.end() :]
    return flipped, {"from": str(year), "to": str(new_year)}


def flip_direction(text, rng):
    m = DIR_RE.search(text)
    if not m:
        return None
    word = m.group(1)
    repl = DIR_LOOKUP[word.lower()]
    if word[0].isupper():
        repl = repl.capitalize()
    flipped = text[: m.start(1)] + repl + text[m.end(1) :]
    return flipped, {"from": word, "to": repl}


def flip_entity(text, claim_pool, rng, nlp):
    """Swap a PERSON/ORG/GPE entity for a same-type entity drawn from other claims in the
    pool. Requires spaCy; returns None silently if nlp is not loaded (caller should then
    just skip entity flips rather than fail the whole run)."""
    if nlp is None:
        return None
    doc = nlp(text)
    ents = [e for e in doc.ents if e.label_ in ("PERSON", "ORG", "GPE")]
    if not ents:
        return None
    ent = rng.choice(ents)
    candidates = [
        e.text
        for c in claim_pool
        for e in nlp(c).ents
        if e.label_ == ent.label_ and e.text.lower() != ent.text.lower()
    ]
    if not candidates:
        return None
    repl = rng.choice(candidates)
    flipped = text[: ent.start_char] + repl + text[ent.end_char :]
    return flipped, {"from": ent.text, "to": repl, "entity_type": ent.label_}


def build(items, seed=0, use_spacy=True):
    rng = random.Random(seed)
    nlp = None
    if use_spacy:
        try:
            import spacy

            nlp = spacy.load("en_core_web_sm")
        except Exception as e:
            print(f"[hard_negatives] spaCy unavailable ({e}); skipping entity flips.")

    supported = [x for x in items if x["label"] == 1]
    claim_pool = [x["claim"] for x in supported]
    out = []
    for it in supported:
        claim = it["claim"]
        # parent_id traces every generated row back to the source record, which the
        # contamination check (training IDs disjoint from eval IDs) needs. Falls back to
        # a source+claim key if the input row has no "id" (e.g. a bare fixture).
        parent_id = it.get("id") or f"{it.get('source', '?')}::{claim}"
        for fn, ftype in (
            (lambda t, r: flip_number(t, r), "flip_number"),
            (lambda t, r: flip_date(t, r), "flip_date"),
            (lambda t, r: flip_direction(t, r), "flip_direction"),
        ):
            res = fn(claim, rng)
            if res:
                flipped, detail = res
                assert flipped != claim, (
                    f"flip produced no change: {ftype} on {claim!r}"
                )
                out.append(
                    dict(
                        id=f"{parent_id}-{ftype}-{len(out)}",
                        parent_id=parent_id,
                        chunks=[it["premise"]],
                        claim=flipped,
                        label=0,
                        type=ftype,
                        flip_detail=detail,
                        source=it["source"],
                        split=it.get("split"),
                    )
                )
        res = flip_entity(claim, claim_pool, rng, nlp)
        if res:
            flipped, detail = res
            assert flipped != claim, (
                f"flip produced no change: flip_entity on {claim!r}"
            )
            out.append(
                dict(
                    id=f"{parent_id}-flip_entity-{len(out)}",
                    parent_id=parent_id,
                    chunks=[it["premise"]],
                    claim=flipped,
                    label=0,
                    type="flip_entity",
                    flip_detail=detail,
                    source=it["source"],
                    split=it.get("split"),
                )
            )
    assert all(r["label"] == 0 for r in out), (
        "hard_negatives must only emit label=0 rows"
    )
    return out


def report(rows):
    from collections import Counter

    print("flip rows:", len(rows))
    print("by type:", dict(Counter(r["type"] for r in rows)))
    print("by split:", dict(Counter(r.get("split") for r in rows)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True, help="prepare_vitaminc.py-style JSONL")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-spacy", action="store_true", help="skip entity flips")
    a = ap.parse_args()
    with open(a.inp, encoding="utf-8") as stream:
        items = [json.loads(line) for line in stream if line.strip()]
    rows = build(items, a.seed, use_spacy=not a.no_spacy)
    with open(a.out, "w", encoding="utf-8") as stream:
        for r in rows:
            stream.write(json.dumps(r, ensure_ascii=False) + "\n")
    report(rows)
