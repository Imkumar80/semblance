"""
DecisionFlip pilot generator, v2.

Changes from v1:
  * Every record is a TRIPLE, not a pair:
        anchor A  (decision X)
        flip   B  (decision Y != X)   -- exactly ONE slot changed vs A
        para   A' (decision X)        -- one wording slot changed vs A, decision unchanged
    The paraphrase A' is the missing control: it lets us ask whether the model
    treats a decision flip as MORE different than a harmless rewording.
  * Flips are single-slot (no "a new" -> "the", no "to" -> "from").
  * Sampling is WITHOUT replacement and balanced across template families,
    so there are no duplicate pairs.
    * Numerical controls change only the amount and stay on the same side of
        the threshold.
    * Scope controls change only the deployment target; the split family is the
        safe/risky scope pair, so evaluation holds out scope pairs.
    * Context-dependent records compose a request, actor context, and policy.
    * Each record has a `family` so the evaluator can hold out whole groups.
  * Easy-negative controls are built programmatically from unrelated anchors.
  * Tool-function records carry tool descriptions for a hard-negative ranking test.

Ground truth comes only from the rule functions below (Section 9).

Usage:
    python generate_pilot_v2.py --n-per-category 12 --out pilot_v2.json
"""

import argparse
import itertools
import json
import random

SEED = 7


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def rec(id_, category, family, rule, a, la, b, lb, p, lp, changed, **extra):
    assert la != lb, f"{id_}: flip must change the decision"
    assert la == lp, f"{id_}: paraphrase must keep the decision"
    assert len({a, b, p}) == 3, f"{id_}: texts must be distinct"
    return {
        "id": id_, "category": category, "family": family, "rule": rule,
        "text_a": a, "label_a": la,
        "text_b": b, "label_b": lb,
        "text_para": p, "label_para": lp,
        "variables_changed": changed, **extra,
    }


def sample_balanced(items, key, n, rng):
    """Pick up to n items without replacement, round-robin across key groups."""
    groups = {}
    for it in items:
        groups.setdefault(key(it), []).append(it)
    for g in groups.values():
        rng.shuffle(g)
    order = list(groups)
    rng.shuffle(order)
    out = []
    while len(out) < n and any(groups.values()):
        for k in order:
            if groups[k] and len(out) < n:
                out.append(groups[k].pop())
    return out


# ---------------------------------------------------------------------------
# 1. Numerical threshold (calibration category)
# ---------------------------------------------------------------------------

THRESHOLD = 1000


def rule_amount(x):
    return "ALLOW" if x < THRESHOLD else "REQUIRE_APPROVAL"


NUM_VERBS = [("Transfer", "Send"), ("Pay", "Remit"), ("Refund", "Return")]
RECIPIENTS = ["Alice", "Bob", "the vendor", "Priya", "the contractor", "Rahul"]
GAPS = [(999, 1000), (990, 1010), (950, 1050), (900, 1100)]  # (below, at/above)


def gen_numerical(n, rng):
    items = []
    for (v, v2), r, (lo, hi) in itertools.product(NUM_VERBS, RECIPIENTS, GAPS):
        items.append((v, v2, r, lo, hi))
    picked = sample_balanced(items, key=lambda t: (t[0], t[3]), n=n, rng=rng)
    out = []
    for i, (v, v2, r, lo, hi) in enumerate(picked):
        para_amount = 950 if lo == 900 else 900
        out.append(rec(
            f"num_{i}", "numerical_threshold", f"num:gap{lo}/{hi}",
            f"amount < ${THRESHOLD} -> ALLOW; >= -> REQUIRE_APPROVAL",
            f"{v} ${lo} to {r}.", rule_amount(lo),
            f"{v} ${hi} to {r}.", rule_amount(hi),
            f"{v} ${para_amount} to {r}.", rule_amount(para_amount),
            ["amount"], gap=f"{lo}/{hi}",
        ))
    return out


# ---------------------------------------------------------------------------
# 2. Scope
# ---------------------------------------------------------------------------

SCOPE_ACTIONS = [  # (phrase, paraphrase phrase)
    ("deploy_config", "Deploy the new config to", "Roll out the new config to"),
    ("push_release", "Push the release to", "Ship the release to"),
    ("restart_service", "Restart the service in", "Reboot the service in"),
    ("run_migration", "Run the migration script against", "Execute the migration script against"),
    ("publish_app", "Publish the application to", "Release the application to"),
    ("apply_update", "Apply the security update in", "Install the security update in"),
]
SCOPE_PAIRS = [
    ("staging", "production", "dev"),
    ("the sandbox environment", "the live environment", "the test environment"),
    ("the internal network", "the public-facing network", "the private network"),
    ("dev", "prod", "staging"),
    ("the test cluster", "the production cluster", "the QA cluster"),
    ("the QA environment", "the customer-facing environment", "the staging environment"),
    ("the canary environment", "the global environment", "the test environment"),
]


def gen_scope(n, rng):
    items = list(itertools.product(SCOPE_ACTIONS, SCOPE_PAIRS))
    picked = sample_balanced(items, key=lambda t: t[0][0], n=n, rng=rng)
    out = []
    for i, ((template_id, act, _), (safe, risky, safe_control)) in enumerate(picked):
        out.append(rec(
            f"scope_{i}", "scope", f"scope_template:{template_id}",
            "safe target -> ALLOW; production-like target -> REQUIRE_APPROVAL",
            f"{act} {safe}.", "ALLOW",
            f"{act} {risky}.", "REQUIRE_APPROVAL",
            f"{act} {safe_control}.", "ALLOW",
            ["scope"],
            template_id=template_id,
            policy_id=f"scope_pair:{safe}|{risky}",
        ))
    return out


# ---------------------------------------------------------------------------
# 3. Context-dependent policy flips
# ---------------------------------------------------------------------------

CONTEXT_REQUESTS = [
    ("Restart the production service", "restart"),
    ("Deploy the hotfix to production", "deploy"),
    ("Delete the archived customer record", "delete"),
    ("Export the customer account data", "export"),
    ("Rotate the production API key", "rotate"),
]
CONTEXT_ACTORS = ["on-call engineer", "security engineer", "contractor", "service owner"]
CONTEXT_POLICIES = [
    ("role_policy", "Only the {allowed_role} may {action} this resource.",
     "{allowed_role} is the sole role permitted to {action} this resource."),
    ("approval_policy", "The {allowed_role} is authorized to {action} without approval.",
     "Approval is not required when the actor is the {allowed_role} and must {action} this resource."),
    ("delegation_policy", "A {allowed_role} may {action} this resource under the current policy.",
     "The current policy permits the {allowed_role} to {action} this resource."),
]


def gen_context_policy(n, rng):
    items = list(itertools.product(CONTEXT_REQUESTS, CONTEXT_ACTORS, CONTEXT_POLICIES))
    picked = sample_balanced(items, key=lambda t: t[2][0], n=n, rng=rng)
    out = []
    for i, ((request, action), actor, (policy_family, policy, policy_para)) in enumerate(picked):
        other_roles = [role for role in CONTEXT_ACTORS if role != actor]
        allowed_role = rng.choice(other_roles)
        allow_text = (
            f"Request: {request}. Context: actor is the {actor}. "
            f"Policy: {policy.format(allowed_role=actor, action=action)}"
        )
        flip_text = (
            f"Request: {request}. Context: actor is the {actor}. "
            f"Policy: {policy.format(allowed_role=allowed_role, action=action)}"
        )
        para_text = (
            f"Request: {request}. Context: actor is the {actor}. "
            f"Policy: {policy_para.format(allowed_role=actor, action=action)}"
        )
        out.append(rec(
            f"context_{i}", "context_policy", f"context_policy:{policy_family}",
            "the actor's role must match the role permitted by the policy",
            allow_text, "ALLOW", flip_text, "DENY", para_text, "ALLOW",
            ["policy_allowed_role"],
            request=request, actor_context=actor, policy_family=policy_family,
            template_id=policy_family,
            policy_id=f"allowed_role:{allowed_role}",
        ))
    return out


# ---------------------------------------------------------------------------
# 3. Permission  (single-slot: negation, antonym and lexical variants)
# ---------------------------------------------------------------------------

PERM_TEMPLATES = [  # (family, positive, negative, paraphrased positive)
    ("auth_negation",
     "{a} is authorized to {v} the {r}.", "{a} is not authorized to {v} the {r}.",
     "{a} is permitted to {v} the {r}."),
    ("approval_lack",
     "{a} has manager approval to {v} the {r}.", "{a} has no manager approval to {v} the {r}.",
     "{a} has manager sign-off to {v} the {r}."),
    ("grant_deny",
     "{a} has been granted permission to {v} the {r}.",
     "{a} has been denied permission to {v} the {r}.",
     "{a} has been given permission to {v} the {r}."),
    ("request_status",
     "{a} has an approved request to {v} the {r}.", "{a} has a denied request to {v} the {r}.",
     "{a} has an accepted request to {v} the {r}."),
]
PERM_VERBS = ["access", "modify", "delete", "export"]
RESOURCES = ["billing database", "customer records table", "production API keys",
             "shared drive folder", "HR compensation sheet"]
ACTORS = ["The contractor", "Intern account #4", "The on-call engineer", "The new hire"]


def gen_permission(n, rng):
    items = list(itertools.product(PERM_TEMPLATES, PERM_VERBS, RESOURCES, ACTORS))
    picked = sample_balanced(items, key=lambda t: t[0][0], n=n, rng=rng)
    out = []
    for i, ((fam, pos, neg, para), v, r, a) in enumerate(picked):
        f = {"a": a, "v": v, "r": r}
        out.append(rec(
            f"perm_{i}", "permission", f"perm:{fam}",
            "authorized/approved -> ALLOW; not authorized/not approved -> DENY",
            pos.format(**f), "ALLOW", neg.format(**f), "DENY", para.format(**f), "ALLOW",
            ["permission_status"],
        ))
    return out


# ---------------------------------------------------------------------------
# 4. Action  (one verb swapped; object, article and preposition held fixed)
# ---------------------------------------------------------------------------

ACTION_TEMPLATES = [  # (family, safe, risky, paraphrased safe)
    ("2fa", "Enable two-factor authentication for {e}.",
     "Disable two-factor authentication for {e}.",
     "Turn on two-factor authentication for {e}."),
    ("admin", "Grant admin access to {e}.", "Revoke admin access to {e}.",
     "Give admin access to {e}."),
    ("account", "Create a user account for {e}.", "Delete a user account for {e}.",
     "Set up a user account for {e}."),
    ("apikey", "Activate the API key for {e}.", "Deactivate the API key for {e}.",
     "Enable the API key for {e}."),
    ("sshkey", "Add an SSH key for {e}.", "Remove an SSH key for {e}.",
     "Register an SSH key for {e}."),
]
ENTITIES = ["the new analyst", "the marketing team", "the vendor", "the intern",
            "the finance group", "the contractor"]


def gen_action(n, rng):
    items = list(itertools.product(ACTION_TEMPLATES, ENTITIES))
    picked = sample_balanced(items, key=lambda t: t[0][0], n=n, rng=rng)
    out = []
    for i, ((fam, safe, risky, para), e) in enumerate(picked):
        out.append(rec(
            f"action_{i}", "action", f"action:{fam}",
            "create/enable/grant -> ALLOW; delete/disable/revoke -> REQUIRE_APPROVAL "
            "(synthetic policy, see realism caveat)",
            safe.format(e=e), "ALLOW", risky.format(e=e), "REQUIRE_APPROVAL",
            para.format(e=e), "ALLOW", ["action_verb"],
        ))
    return out


# ---------------------------------------------------------------------------
# 5. Tool function (evaluated as hard-negative ranking against tool descriptions)
# ---------------------------------------------------------------------------

TOOL_DESC = {
    "calendar.create_event": "Create a new event on the user's calendar",
    "calendar.list_events": "List or look up existing events on the user's calendar",
    "email.send": "Send an email message to a recipient",
    "email.search": "Search the mailbox for existing emails",
    "ticket.create": "Create a new support ticket",
    "ticket.search": "Search for existing support tickets",
    "doc.create": "Create a new document",
    "doc.search": "Search for existing documents",
}
TOOL_TEMPLATES = [  # (family, text_a, tool_a, text_b, tool_b, paraphrase of a)
    ("calendar", "Create a calendar event with {p} at {t}.", "calendar.create_event",
     "List calendar events with {p} at {t}.", "calendar.list_events",
     "Schedule a calendar event with {p} at {t}."),
    ("email", "Send the email to {p} about {x}.", "email.send",
     "Find the email to {p} about {x}.", "email.search",
     "Dispatch the email to {p} about {x}."),
    ("ticket", "Create a ticket about {x}.", "ticket.create",
     "Find a ticket about {x}.", "ticket.search",
     "Open a ticket about {x}."),
    ("doc", "Create a document about {x}.", "doc.create",
     "Find a document about {x}.", "doc.search",
     "Draft a document about {x}."),
]
PEOPLE = ["Maya", "the design team", "Rahul", "the vendor"]
TIMES = ["3pm tomorrow", "noon on Friday", "9am Monday"]
TOPICS = ["the Q3 deadline", "the billing outage", "the onboarding plan", "the audit"]


def gen_tool(n, rng):
    items = list(itertools.product(TOOL_TEMPLATES, PEOPLE, TIMES, TOPICS))
    picked = sample_balanced(items, key=lambda t: t[0][0], n=n, rng=rng)
    out, seen = [], set()
    for (fam, ta, tool_a, tb, tool_b, tp), p, t, x in picked:
        f = {"p": p, "t": t, "x": x}
        a, b, pa = ta.format(**f), tb.format(**f), tp.format(**f)
        if (a, b) in seen:  # templates that ignore some slots can collide
            continue
        seen.add((a, b))
        out.append(rec(
            f"tool_{len(out)}", "tool_function", f"tool:{fam}",
            "label = intended tool call",
            a, tool_a, b, tool_b, pa, tool_a, ["intent_verb"],
            desc_a=TOOL_DESC[tool_a], desc_b=TOOL_DESC[tool_b],
        ))
    return out


# ---------------------------------------------------------------------------
# Unrelated controls built from anchors of different categories
# ---------------------------------------------------------------------------

def gen_controls(triples, n, rng):
    by_cat = {}
    for t in triples:
        by_cat.setdefault(t["category"], []).append(t)
    cats = list(by_cat)
    out = []
    for i in range(n):
        c1, c2 = rng.sample(cats, 2)
        out.append({
            "id": f"ctrl_{i}",
            "text_a": rng.choice(by_cat[c1])["text_a"],
            "text_b": rng.choice(by_cat[c2])["text_a"],
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-category", type=int, default=30)
    ap.add_argument("--n-controls", type=int, default=20)
    ap.add_argument("--out", default="pilot_v2.json")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    triples = []
    for fn in (gen_numerical, gen_scope, gen_permission, gen_action, gen_tool,
               gen_context_policy):
        triples.extend(fn(args.n_per_category, rng))

    pair_keys = [(t["text_a"], t["text_b"]) for t in triples]
    assert len(pair_keys) == len(set(pair_keys)), "duplicate pairs generated"

    controls = gen_controls(triples, args.n_controls, rng)
    with open(args.out, "w") as f:
        json.dump({"triples": triples, "controls": controls}, f, indent=2)

    print(f"Generated {len(triples)} triples + {len(controls)} controls -> {args.out}")
    counts = {}
    for t in triples:
        counts.setdefault(t["category"], set()).add(t["family"])
    for category, families in counts.items():
        n = sum(t["category"] == category for t in triples)
        print(f"  {category}: {n} triples, {len(families)} families")


if __name__ == "__main__":
    main()
