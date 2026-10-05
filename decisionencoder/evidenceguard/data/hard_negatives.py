"""
hard_negatives.py -- deterministic single-fact flips applied directly to an English
SUPPORTED claim, independent of whether VitaminC happens to have a matching REFUTES edit.

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

Entity flips use spaCy/en_core_web_sm when available and a lower-precision, zero-dependency
capitalized-phrase fallback otherwise. Number/date/direction flips are regex-only.
"""

import argparse
import json
import random
import re
from pathlib import Path


DEFAULT_LEXICON_PATH = Path(__file__).with_name("lexicon_en.json")


def load_lexicon(path=DEFAULT_LEXICON_PATH):
    """Load an English flip lexicon, rejecting malformed or ambiguous mappings."""
    with Path(path).open(encoding="utf-8") as stream:
        lexicon = json.load(stream)
    if not isinstance(lexicon, dict):
        raise ValueError("lexicon root must be a JSON object")
    pairs = lexicon.get("direction_pairs")
    months = lexicon.get("months")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("lexicon direction_pairs must be a non-empty list")
    if any(
        not isinstance(pair, list)
        or len(pair) != 2
        or not all(isinstance(word, str) and word.strip() for word in pair)
        for pair in pairs
    ):
        raise ValueError("each direction pair must contain two non-empty strings")
    if (
        not isinstance(months, list)
        or len(months) != 12
        or not all(isinstance(month, str) and month.strip() for month in months)
    ):
        raise ValueError("lexicon months must contain twelve non-empty month names")
    lexicon["direction_pairs"] = pairs
    lexicon["months"] = months
    return lexicon


DEFAULT_LEXICON = load_lexicon()
DIRECTION_PAIRS = DEFAULT_LEXICON["direction_pairs"]
MONTHS = DEFAULT_LEXICON["months"]


def build_direction_lookup(pairs):
    lookup = {}
    for a, b in pairs:
        for src, dst in ((a, b), (b, a)):
            if src.lower() in lookup and lookup[src.lower()] != dst:
                raise ValueError(
                    f"DIRECTION_PAIRS collision: {src!r} already maps to "
                    f"{lookup[src.lower()]!r}, cannot also map to {dst!r}. "
                    "Pick a different word for one of the two pairs."
                )
            lookup[src.lower()] = dst
    return lookup


# "gained"<->"lost" and "won"<->"lost" used to share "lost", so whichever pair was built
# last in the dict silently overwrote the other (lost -> won always, never -> gained,
# losing the gained/declined direction entirely). Kept as two non-overlapping pairs
# instead (won<->lost for games/elections, gained<->declined for amounts/values), plus a
# load-time check so a future added pair can't reintroduce a silent collision.
DIR_LOOKUP = build_direction_lookup(DIRECTION_PAIRS)

NUMBER_RE = re.compile(
    r"(?<![\w.])(\d+(?:,\d{3})*(?:\.\d+)?)(?![\w.])\s*(%|percent)?"
)
YEAR_RE = re.compile(r"\b(?:17|18|19|20)\d{2}\b")
DECADE_RE = re.compile(r"\b(?:(?:17|18|19)\d|20\d)0s\b", re.IGNORECASE)
CENTURY_RE = re.compile(
    r"\b(?P<number>\d{1,2})(?P<suffix>st|nd|rd|th)\s+century\b", re.I
)
RELATIVE_DURATION_RES = (
    re.compile(
        r"\b(?P<value>\d{1,3})\s+(?P<unit>years?|months?|weeks?|days?)"
        r"\s+(?P<relative>ago|earlier|later|from now)\b",
        re.I,
    ),
    re.compile(
        r"\bin\s+(?P<value>\d{1,3})\s+(?P<unit>years?|months?|weeks?|days?)\b",
        re.I,
    ),
)
NON_ENTITY_STARTERS = {
    "a", "an", "and", "although", "at", "before", "by", "during", "he", "her",
    "here", "how", "however", "i", "in", "it", "its", "she", "since", "the",
    "they", "this", "those", "to", "we", "when", "where", "which", "who", "while",
}
PROPER_NOUN_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b")


def build_direction_regex(direction_lookup):
    return re.compile(
        r"\b("
        + "|".join(
            sorted((re.escape(k) for k in direction_lookup), key=len, reverse=True)
        )
        + r")\b",
        re.IGNORECASE,
    )


def build_month_regex(months):
    return re.compile(
        r"\b(" + "|".join(sorted((re.escape(m) for m in months), key=len, reverse=True)) + r")\b",
        re.IGNORECASE,
    )


DIR_RE = build_direction_regex(DIR_LOOKUP)
MONTH_RE = build_month_regex(MONTHS)


def _overlaps(span_a, span_b):
    return span_a[0] < span_b[1] and span_b[0] < span_a[1]


def _is_temporal_number(text, number_span, month_re=MONTH_RE):
    patterns = (YEAR_RE, DECADE_RE, CENTURY_RE, *RELATIVE_DURATION_RES)
    if any(
        _overlaps(number_span, match.span())
        for pattern in patterns
        for match in pattern.finditer(text)
    ):
        return True
    for month_match in month_re.finditer(text):
        day_match = re.match(
            r"\s+\d{1,2}(?:st|nd|rd|th)?(?:,\s*(?:17|18|19|20)\d{2})?",
            text[month_match.end():],
            re.IGNORECASE,
        )
        if day_match and _overlaps(
            number_span, (month_match.start(), month_match.end() + day_match.end())
        ):
            return True
    return False


def flip_number(text, rng, month_re=MONTH_RE):
    """Flip the first standalone number by +/-(10-50)%, min change of 1. Skips bare years.
    Preserves the original number's format (integer, decimal places, comma grouping) so a
    whole-number population count doesn't come out as e.g. "303604.9" -- a random
    percentage of a large integer is essentially never itself a whole number, so the
    output format must be derived from the ORIGINAL string, not from whether the flipped
    value happens to be an integer."""
    for m in NUMBER_RE.finditer(text):
        raw = m.group(1)
        if _is_temporal_number(text, m.span(1), month_re):
            continue  # looks like a year, not a quantity -> leave to flip_date
        has_comma = "," in raw
        decimals = len(raw.split(".", 1)[1]) if "." in raw else 0
        val = float(raw.replace(",", ""))
        delta = max(1.0, val * rng.uniform(0.10, 0.50))
        # Decreasing zero would leave the claim unchanged ("0" -> "0"). Zero-valued
        # quantities must therefore always move upward to produce a valid hard negative.
        new_val = (
            val + delta if val == 0.0 or rng.random() < 0.5 else max(0.0, val - delta)
        )
        if decimals:
            new_str = (
                f"{new_val:,.{decimals}f}" if has_comma else f"{new_val:.{decimals}f}"
            )
        else:
            new_str = f"{round(new_val):,}" if has_comma else str(round(new_val))
        if new_str == raw:
            continue  # rounding collapsed back to the original value; try the next number
        flipped = text[: m.start(1)] + new_str + text[m.end(1) :]
        return flipped, {"from": raw, "to": new_str}
    return None


def _ordinal_suffix(number):
    if number % 100 in (11, 12, 13):
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")


def _preserve_case(replacement, original):
    if original.isupper():
        return replacement.upper()
    if original.islower():
        return replacement.lower()
    if original.istitle():
        return replacement.title()
    return replacement


def flip_date(text, rng, months=None, month_re=None):
    """Perturb a year, decade, century, named month, or relative duration."""
    months = MONTHS if months is None else months
    month_re = MONTH_RE if month_re is None else month_re

    for pattern in RELATIVE_DURATION_RES:
        m = pattern.search(text)
        if m:
            value = int(m.group("value"))
            new_value = value + rng.choice((-3, -2, -1, 1, 2, 3))
            new_value = max(1, new_value)
            start, end = m.span("value")
            replacement = str(new_value)
            flipped = text[:start] + replacement + text[end:]
            return flipped, {
                "from": m.group("value") + " " + m.group("unit"),
                "to": replacement + " " + m.group("unit"),
            }

    m = month_re.search(text)
    if m:
        original = m.group(0)
        alternatives = [month for month in months if month.lower() != original.lower()]
        replacement = _preserve_case(rng.choice(alternatives), original)
        flipped = text[:m.start()] + replacement + text[m.end():]
        return flipped, {"from": original, "to": replacement}

    m = CENTURY_RE.search(text)
    if m:
        century = int(m.group("number"))
        if not 1 <= century <= 21:
            m = None
        else:
            candidates = [
                value
                for value in (century - 2, century - 1, century + 1, century + 2)
                if 1 <= value <= 21
            ]
            new_century = rng.choice(candidates)
            suffix = _ordinal_suffix(new_century)
            replacement = f"{new_century}{suffix}"
            start, end = m.span("number")
            flipped = text[:start] + replacement + text[end:]
            return flipped, {
                "from": m.group("number") + m.group("suffix"),
                "to": replacement,
            }

    m = DECADE_RE.search(text)
    if m:
        decade = int(m.group(0)[:-1])
        candidates = [value for value in (decade - 20, decade - 10, decade + 10, decade + 20) if 1700 <= value <= 2090]
        new_decade = rng.choice(candidates)
        replacement = f"{new_decade}s"
        flipped = text[:m.start()] + replacement + text[m.end():]
        return flipped, {"from": m.group(0), "to": replacement}

    m = YEAR_RE.search(text)
    if m:
        year = int(m.group(0))
        offsets = [offset for offset in (-3, -2, -1, 1, 2, 3) if 1700 <= year + offset <= 2099]
        new_year = year + rng.choice(offsets)
        flipped = text[:m.start()] + str(new_year) + text[m.end():]
        return flipped, {"from": str(year), "to": str(new_year)}
    return None


def flip_direction(text, rng, direction_lookup=None, direction_re=None):
    direction_lookup = DIR_LOOKUP if direction_lookup is None else direction_lookup
    direction_re = DIR_RE if direction_re is None else direction_re
    m = direction_re.search(text)
    if not m:
        return None
    word = m.group(1)
    repl = direction_lookup[word.lower()]
    if word[0].isupper():
        repl = repl.capitalize()
    flipped = text[: m.start(1)] + repl + text[m.end(1) :]
    return flipped, {"from": word, "to": repl}


ENTITY_TYPES = ("PERSON", "ORG", "GPE")
FALLBACK_ENTITIES = {
    "PERSON": ("James Wilson", "Maria Garcia", "David Chen", "Elena Rostova", "Marcus Johnson"),
    "ORG": ("Acme Corp", "Vanguard Group", "Novartis", "Siemens", "Airbus"),
    "GPE": ("Canada", "Germany", "Japan", "Brazil", "Australia"),
}
FALLBACK_ENTITY_NAMES = tuple(
    name for names in FALLBACK_ENTITIES.values() for name in names
)


def _fallback_entity_matches(text):
    matches = []
    for match in PROPER_NOUN_RE.finditer(text):
        words = list(re.finditer(r"[A-Z][a-z]+", match.group(0)))
        while words and words[0].group(0).lower() in NON_ENTITY_STARTERS:
            words.pop(0)
        if not words:
            continue
        start = match.start() + words[0].start()
        candidate = text[start:match.end()]
        if candidate.lower() in {month.lower() for month in MONTHS}:
            continue
        matches.append((start, match.end(), candidate))
    return matches


def build_entity_index(claims, nlp, batch_size=256):
    """Run NER over `claims` ONCE and return {entity_type: [unique entity_text, ...]}.

    flip_entity used to call nlp(c) on every claim in the candidate pool on EVERY call --
    with ~200k supported rows that is O(n^2) spaCy invocations (tens of billions of calls
    at this dataset's scale), which would not finish. Precomputing the index once per
    split, with nlp.pipe for batched processing, makes indexing O(n): each claim is NER'd
    exactly once. Deduplicated (case-insensitive, first-seen form kept) so a single very
    common entity (e.g. "the United States" mentioned thousands of times) doesn't dominate
    the list -- that would both skew which replacement gets picked and make the
    bounded-retry loop in flip_entity more likely to exhaust itself on a near-uniform pool."""
    index = {t: [] for t in ENTITY_TYPES}
    seen = {t: set() for t in ENTITY_TYPES}
    for doc in nlp.pipe(claims, batch_size=batch_size):
        for e in doc.ents:
            if e.label_ in index:
                key = e.text.lower()
                if key not in seen[e.label_]:
                    seen[e.label_].add(key)
                    index[e.label_].append(e.text)
    return index


def flip_entity(text, entity_index, rng, nlp, max_tries=5):
    """Swap a PERSON/ORG/GPE entity, or use a lower-precision regex fallback.

    Selection is expected-constant-time: rng.choice on a list is O(1), so up to max_tries
    draws is O(1) regardless of dataset size -- no per-call scan of the candidate list.
    (The previous version filtered the whole list on every call, which stayed O(n) per
    flip_entity call and so O(n*k) across a split even after NER itself was made O(n).)
    A list with only the original entity's own text (or empty) correctly returns None
    after max_tries rather than looping forever or silently picking a non-distinct value.
    Without usable spaCy entities, the fallback has no reliable entity type; it marks such
    rows UNKNOWN so they can be separated for quality review."""
    if nlp is not None:
        doc = nlp(text)
        ents = [e for e in doc.ents if e.label_ in ENTITY_TYPES]
        if ents:
            ent = rng.choice(ents)
            pool = entity_index.get(ent.label_, [])
            for _ in range(max_tries):
                if not pool:
                    break
                candidate = rng.choice(pool)
                if candidate.lower() != ent.text.lower():
                    flipped = text[:ent.start_char] + candidate + text[ent.end_char:]
                    return flipped, {
                        "from": ent.text,
                        "to": candidate,
                        "entity_type": ent.label_,
                    }

    candidates = _fallback_entity_matches(text)
    if not candidates:
        return None
    start, end, original = rng.choice(candidates)
    pool = [name for name in FALLBACK_ENTITY_NAMES if name.lower() != original.lower()]
    if not pool:
        return None
    replacement = rng.choice(pool)
    flipped = text[:start] + replacement + text[end:]
    return flipped, {
        "from": original,
        "to": replacement,
        "entity_type": "UNKNOWN",
        "method": "regex_fallback",
    }


def build(items, seed=0, use_spacy=True, lexicon_path=DEFAULT_LEXICON_PATH):
    rng = random.Random(seed)
    lexicon = load_lexicon(lexicon_path)
    direction_lookup = build_direction_lookup(lexicon["direction_pairs"])
    direction_re = build_direction_regex(direction_lookup)
    month_re = build_month_regex(lexicon["months"])
    nlp = None
    if use_spacy:
        try:
            import spacy

            nlp = spacy.load("en_core_web_sm")
        except (ImportError, OSError) as e:
            print(
                f"[hard_negatives] spaCy/en_core_web_sm unavailable ({e}); "
                "using lower-precision regex entity flips."
            )

    supported = [x for x in items if x["label"] == 1]
    # Entity-flip candidates must come from the SAME split only -- otherwise a train-split
    # row could borrow an entity string from a test-split claim, which violates the
    # train/eval disjointness the contamination manifest checks for. Built once per split
    # (not per claim -- see build_entity_index) so this stays O(n) instead of O(n^2).
    entity_index_by_split = {}
    if nlp is not None:
        claims_by_split = {}
        for x in supported:
            claims_by_split.setdefault(x.get("split"), []).append(x["claim"])
        for split, claims in claims_by_split.items():
            entity_index_by_split[split] = build_entity_index(claims, nlp)
    out = []
    for it in supported:
        claim = it["claim"]
        # parent_id traces every generated row back to the source record, which the
        # contamination check (training IDs disjoint from eval IDs) needs. Falls back to
        # a source+claim key if the input row has no "id" (e.g. a bare fixture).
        parent_id = it.get("id") or f"{it.get('source', '?')}::{claim}"
        for fn, ftype in (
            (lambda t, r: flip_number(t, r, month_re), "flip_number"),
            (lambda t, r: flip_date(t, r, lexicon["months"], month_re), "flip_date"),
            (
                lambda t, r: flip_direction(t, r, direction_lookup, direction_re),
                "flip_direction",
            ),
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
        split_index = entity_index_by_split.get(it.get("split"), {})
        res = flip_entity(claim, split_index, rng, nlp)
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
    ap.add_argument(
        "--no-spacy",
        action="store_true",
        help="force the lower-precision regex entity fallback",
    )
    ap.add_argument(
        "--lexicon",
        type=Path,
        default=DEFAULT_LEXICON_PATH,
        help="JSON lexicon containing direction_pairs and twelve month names",
    )
    a = ap.parse_args()
    with open(a.inp, encoding="utf-8") as stream:
        items = [json.loads(line) for line in stream if line.strip()]
    rows = build(items, a.seed, use_spacy=not a.no_spacy, lexicon_path=a.lexicon)
    with open(a.out, "w", encoding="utf-8") as stream:
        for r in rows:
            stream.write(json.dumps(r, ensure_ascii=False) + "\n")
    report(rows)
