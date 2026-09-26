"""
DecisionFlip pilot dataset generator.

Produces controlled pairs of the form:
    (text_a, text_b, label_a, label_b, category, rule)

Every pair is constructed so that text_a and text_b are near-identical
(one controlled variable changed) but the *correct decision* differs.
Ground truth comes entirely from the rule that generated the pair --
no model is asked to judge correctness (per plan Section 9).

We also generate "easy negative" pairs per category: same label change,
but with large surface-level difference, so we can check that the
DecisionFlip pairs really do sit in the "high similarity, different
decision" region rather than looking like ordinary hard negatives.

Usage:
    python generate_pilot.py --n-per-category 8 --out pilot_pairs.json
"""

import argparse
import itertools
import json
import random

random.seed(7)  # reproducibility for the pilot run

# ---------------------------------------------------------------------------
# Category 1: Numerical threshold (calibration / sanity-check category --
# see Section 4 caveat: this is the easiest case, not the headline case)
# ---------------------------------------------------------------------------

RECIPIENTS = ["Alice", "Bob", "the vendor", "Priya", "the contractor", "Rahul"]


def gen_numerical_threshold(n):
    pairs = []
    threshold = 1000
    for i in range(n):
        recipient = random.choice(RECIPIENTS)
        below = threshold - random.randint(1, 50)
        at_or_above = threshold + random.randint(0, 200)
        text_a = f"Transfer ${below} to {recipient}."
        text_b = f"Transfer ${at_or_above} to {recipient}."
        pairs.append({
            "id": f"numthresh_{i}",
            "category": "numerical_threshold",
            "rule": f"amount < ${threshold} -> ALLOW; amount >= ${threshold} -> REQUIRE_APPROVAL",
            "text_a": text_a, "label_a": "ALLOW",
            "text_b": text_b, "label_b": "REQUIRE_APPROVAL",
            "variables_changed": ["amount"],
        })
    return pairs


# ---------------------------------------------------------------------------
# Category 2: Scope (staging vs production / internal vs external)
# ---------------------------------------------------------------------------

ACTIONS_SCOPE = ["Deploy the new config to", "Push the release to", "Restart the service in",
                 "Run the migration script against", "Apply the firewall rule to"]
SCOPE_PAIRS = [("staging", "production"), ("the sandbox environment", "the live environment"),
               ("the internal network", "the public-facing network"), ("dev", "prod")]


def gen_scope(n):
    pairs = []
    for i in range(n):
        action = random.choice(ACTIONS_SCOPE)
        safe_scope, risky_scope = random.choice(SCOPE_PAIRS)
        text_a = f"{action} {safe_scope}."
        text_b = f"{action} {risky_scope}."
        pairs.append({
            "id": f"scope_{i}",
            "category": "scope",
            "rule": f"target=({safe_scope}) -> ALLOW; target=({risky_scope}) -> REQUIRE_APPROVAL",
            "text_a": text_a, "label_a": "ALLOW",
            "text_b": text_b, "label_b": "REQUIRE_APPROVAL",
            "variables_changed": ["scope"],
        })
    return pairs


# ---------------------------------------------------------------------------
# Category 3: Permission (authorized vs unauthorized / approved vs unapproved)
# ---------------------------------------------------------------------------

PERMISSION_TEMPLATES = [
    "{actor} is authorized to access the {resource}.",
    "{actor} has an approved request to modify the {resource}.",
    "{actor} has manager sign-off to delete the {resource}.",
]
RESOURCES = ["billing database", "customer records table", "production API keys",
             "shared drive folder", "HR compensation sheet"]
ACTORS = ["The contractor", "Intern account #4", "The on-call engineer", "The new hire"]


def gen_permission(n):
    pairs = []
    for i in range(n):
        template = random.choice(PERMISSION_TEMPLATES)
        actor = random.choice(ACTORS)
        resource = random.choice(RESOURCES)
        authorized_text = template.format(actor=actor, resource=resource)
        unauthorized_text = authorized_text.replace("is authorized", "is not authorized") \
            .replace("has an approved request", "has not requested authorization") \
            .replace("has manager sign-off", "does not have manager sign-off")
        pairs.append({
            "id": f"permission_{i}",
            "category": "permission",
            "rule": "authorized/approved -> ALLOW; not authorized/not approved -> DENY",
            "text_a": authorized_text, "label_a": "ALLOW",
            "text_b": unauthorized_text, "label_b": "DENY",
            "variables_changed": ["permission_status"],
        })
    return pairs


# ---------------------------------------------------------------------------
# Category 4: Action (create vs delete, enable vs disable)
# ---------------------------------------------------------------------------

ACTION_PAIR_TEMPLATES = [
    ("Create a new user account for {entity}.", "Delete the user account for {entity}."),
    ("Enable two-factor authentication for {entity}.", "Disable two-factor authentication for {entity}."),
    ("Grant admin access to {entity}.", "Revoke admin access from {entity}."),
]
ENTITIES = ["the new analyst", "the marketing team", "the shared service account", "the vendor login"]


def gen_action(n):
    pairs = []
    for i in range(n):
        safe_tmpl, risky_tmpl = random.choice(ACTION_PAIR_TEMPLATES)
        entity = random.choice(ENTITIES)
        text_a = safe_tmpl.format(entity=entity)
        text_b = risky_tmpl.format(entity=entity)
        pairs.append({
            "id": f"action_{i}",
            "category": "action",
            "rule": "create/enable/grant -> ALLOW; delete/disable/revoke -> REQUIRE_APPROVAL",
            "text_a": text_a, "label_a": "ALLOW",
            "text_b": text_b, "label_b": "REQUIRE_APPROVAL",
            "variables_changed": ["action_verb"],
        })
    return pairs


# ---------------------------------------------------------------------------
# Category 5: Tool function (create_event vs list_events -- flagged in plan
# Section 8 as having related prior work; included but not emphasized)
# ---------------------------------------------------------------------------

TOOL_PAIRS = [
    ("Add a meeting to my calendar for tomorrow at 3pm.", "calendar.create_event"),
    ("Show me what's on my calendar tomorrow at 3pm.", "calendar.list_events"),
    ("Send an email to the team about the deadline.", "email.send"),
    ("Show me emails from the team about the deadline.", "email.search"),
]


def gen_tool_function(n):
    pairs = []
    base = list(itertools.combinations(range(len(TOOL_PAIRS)), 1))
    for i in range(n):
        idx = (2 * i) % len(TOOL_PAIRS)
        idx_b = idx + 1 if idx + 1 < len(TOOL_PAIRS) else idx - 1
        text_a, label_a = TOOL_PAIRS[idx]
        text_b, label_b = TOOL_PAIRS[idx_b]
        pairs.append({
            "id": f"toolfn_{i}",
            "category": "tool_function",
            "rule": "mapped directly from intended tool call",
            "text_a": text_a, "label_a": label_a,
            "text_b": text_b, "label_b": label_b,
            "variables_changed": ["intent_verb"],
        })
    return pairs


# ---------------------------------------------------------------------------
# Easy negatives: same label change, but low surface similarity -- used to
# confirm the DecisionFlip pairs sit in a different region than ordinary
# hard negatives (Section 15 representation analysis needs this contrast).
# ---------------------------------------------------------------------------

EASY_NEGATIVES = [
    ("Approve the $50 reimbursement for office supplies.", "ALLOW",
     "Immediately wipe all production database backups.", "REQUIRE_APPROVAL"),
    ("Read the public documentation page.", "ALLOW",
     "Export the entire customer PII table to an external drive.", "DENY"),
]


def gen_easy_negatives(n):
    pairs = []
    for i in range(n):
        text_a, label_a, text_b, label_b = EASY_NEGATIVES[i % len(EASY_NEGATIVES)]
        pairs.append({
            "id": f"easyneg_{i}",
            "category": "easy_negative_control",
            "rule": "control pair: large surface difference, different decision",
            "text_a": text_a, "label_a": label_a,
            "text_b": text_b, "label_b": label_b,
            "variables_changed": ["everything"],
        })
    return pairs


GENERATORS = {
    "numerical_threshold": gen_numerical_threshold,
    "scope": gen_scope,
    "permission": gen_permission,
    "action": gen_action,
    "tool_function": gen_tool_function,
    "easy_negative_control": gen_easy_negatives,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-category", type=int, default=7,
                     help="pairs per category (default gives ~35-40 total pairs, "
                          "inside the Section 11 pilot range of 20-50)")
    ap.add_argument("--out", type=str, default="pilot_pairs.json")
    args = ap.parse_args()

    all_pairs = []
    for name, fn in GENERATORS.items():
        n = args.n_per_category if name != "easy_negative_control" else max(2, args.n_per_category // 3)
        all_pairs.extend(fn(n))

    with open(args.out, "w") as f:
        json.dump(all_pairs, f, indent=2)

    print(f"Generated {len(all_pairs)} pairs -> {args.out}")
    by_cat = {}
    for p in all_pairs:
        by_cat[p["category"]] = by_cat.get(p["category"], 0) + 1
    for cat, count in by_cat.items():
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()
