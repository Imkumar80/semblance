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
        "id": id_,
        "category": category,
        "family": family,
        "rule": rule,
        "text_a": a,
        "label_a": la,
        "text_b": b,
        "label_b": lb,
        "text_para": p,
        "label_para": lp,
        "variables_changed": changed,
        **extra,
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
        para_amount = lo - 1
        out.append(
            rec(
                f"num_{i}",
                "numerical_threshold",
                f"num:gap{lo}/{hi}",
                f"amount < ${THRESHOLD} -> ALLOW; >= -> REQUIRE_APPROVAL",
                f"{v} ${lo} to {r}.",
                rule_amount(lo),
                f"{v} ${hi} to {r}.",
                rule_amount(hi),
                f"{v} ${para_amount} to {r}.",
                rule_amount(para_amount),
                ["amount"],
                gap=f"{lo}/{hi}",
            )
        )
    return out


# ---------------------------------------------------------------------------
# 2. Scope
# ---------------------------------------------------------------------------

SCOPE_ACTIONS = [  # (phrase, paraphrase phrase)
    ("deploy_config", "Deploy the new config to", "Roll out the new config to"),
    ("push_release", "Push the release to", "Ship the release to"),
    ("restart_service", "Restart the service in", "Reboot the service in"),
    (
        "run_migration",
        "Run the migration script against",
        "Execute the migration script against",
    ),
    ("publish_app", "Publish the application to", "Release the application to"),
    ("apply_update", "Apply the security update in", "Install the security update in"),
]
SCOPE_PAIRS = [
    ("staging", "production", "development"),
    ("the sandbox environment", "the live environment", "the test environment"),
    ("the internal network", "the public-facing network", "the private network"),
    ("the QA cluster", "the production cluster", "the canary cluster"),
    (
        "the preproduction environment",
        "the customer-facing environment",
        "the integration environment",
    ),
    ("the regional test network", "the global network", "the isolated network"),
    ("the canary deployment", "all production regions", "the single-node test cluster"),
]


def gen_scope(n, rng):
    items = list(itertools.product(SCOPE_ACTIONS, SCOPE_PAIRS))
    picked = sample_balanced(items, key=lambda t: t[0][0], n=n, rng=rng)
    out = []
    for i, ((template_id, act, _), (safe, risky, safe_control)) in enumerate(picked):
        out.append(
            rec(
                f"scope_{i}",
                "scope",
                f"scope_template:{template_id}",
                "safe target -> ALLOW; production-like target -> REQUIRE_APPROVAL",
                f"{act} {safe}.",
                "ALLOW",
                f"{act} {risky}.",
                "REQUIRE_APPROVAL",
                f"{act} {safe_control}.",
                "ALLOW",
                ["scope"],
                template_id=template_id,
                policy_id=f"scope_pair:{safe}|{risky}",
            )
        )
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
    ("Revoke the temporary admin credential", "revoke"),
    ("Restore the latest database backup", "restore"),
    ("Disable the compromised user account", "disable"),
    ("Change the firewall rule for the payment service", "change"),
    ("Read the incident response log", "read"),
    ("Quarantine the workstation after the alert", "quarantine"),
    ("Approve the emergency production change", "approve"),
    ("Purge the expired access token", "purge"),
    ("Update the customer-facing status page", "update"),
    ("Unlock the suspended employee account", "unlock"),
    ("Modify the backup retention schedule", "modify"),
    ("Initiate a failover for the primary database", "initiate"),
    ("Share the incident report with the vendor", "share"),
    ("Close the active security investigation", "close"),
    ("Change the access control list for the service", "change"),
]
CONTEXT_ACTORS = [
    "on-call engineer",
    "security engineer",
    "contractor",
    "service owner",
    "database administrator",
    "incident commander",
    "platform engineer",
    "support analyst",
]
CONTEXT_POLICIES = [
    (
        "only_role",
        "Only the {allowed_role} may {action} this resource.",
        "{allowed_role} is the sole role permitted to {action} this resource.",
    ),
    (
        "authorized_role",
        "The {allowed_role} is authorized to {action} without approval.",
        "Approval is not required when the actor is the {allowed_role} and must {action} this resource.",
    ),
    (
        "delegated_role",
        "The {allowed_role} may {action} this resource under current policy.",
        "Current policy authorizes the {allowed_role} to {action} this resource.",
    ),
    (
        "role_restriction",
        "This operation is restricted to the {allowed_role} role.",
        "Only a person assigned the {allowed_role} role can perform this operation.",
    ),
    (
        "permission_record",
        "Policy grants the {allowed_role} permission to {action} this resource.",
        "The {allowed_role} role has permission under policy to {action} this resource.",
    ),
    (
        "responsible_role",
        "For this resource, only the {allowed_role} may carry out the requested operation.",
        "The requested operation on this resource may be carried out only by the {allowed_role}.",
    ),
    (
        "exclusive_role",
        "No role other than the {allowed_role} can {action} this resource.",
        "The {allowed_role} alone is allowed to {action} this resource.",
    ),
    (
        "role_scope",
        "For this resource, the {allowed_role} role is authorized for the requested operation.",
        "The role authorized for this resource's operation is the {allowed_role}.",
    ),
]


def gen_context_policy(n, rng):
    items = list(itertools.product(CONTEXT_REQUESTS, CONTEXT_ACTORS, CONTEXT_POLICIES))
    picked = sample_balanced(items, key=lambda item: item[2][0], n=n, rng=rng)
    out = []
    for i, ((request, action), actor, (template_id, policy, policy_para)) in enumerate(
        picked
    ):
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
        out.append(
            rec(
                f"context_{i}",
                "context_policy",
                f"context_policy:{template_id}",
                "the actor's role must match the role permitted by the policy",
                allow_text,
                "ALLOW",
                flip_text,
                "DENY",
                para_text,
                "ALLOW",
                ["policy_allowed_role"],
                request=request,
                actor_context=actor,
                policy_template=template_id,
                template_id=template_id,
                policy_id=f"{action}:allowed_role:{allowed_role}",
                scenario_id=f"{request}:{actor}:{template_id}",
            )
        )
    return out


# ---------------------------------------------------------------------------
# 3. Permission  (single-slot: negation, antonym and lexical variants)
# ---------------------------------------------------------------------------

PERM_TEMPLATES = [  # (family, positive, negative, paraphrased positive)
    (
        "auth_negation",
        "{a} is authorized to {v} the {r}.",
        "{a} is not authorized to {v} the {r}.",
        "{a} is permitted to {v} the {r}.",
    ),
    (
        "approval_lack",
        "{a} has manager approval to {v} the {r}.",
        "{a} has no manager approval to {v} the {r}.",
        "{a} has manager sign-off to {v} the {r}.",
    ),
    (
        "grant_deny",
        "{a} has been granted permission to {v} the {r}.",
        "{a} has been denied permission to {v} the {r}.",
        "{a} has been given permission to {v} the {r}.",
    ),
    (
        "request_status",
        "{a} has an approved request to {v} the {r}.",
        "{a} has a denied request to {v} the {r}.",
        "{a} has an accepted request to {v} the {r}.",
    ),
]
PERM_VERBS = ["access", "modify", "delete", "export"]
RESOURCES = [
    "billing database",
    "customer records table",
    "production API keys",
    "shared drive folder",
    "HR compensation sheet",
]
ACTORS = ["The contractor", "Intern account #4", "The on-call engineer", "The new hire"]


def gen_permission(n, rng):
    items = list(itertools.product(PERM_TEMPLATES, PERM_VERBS, RESOURCES, ACTORS))
    picked = sample_balanced(items, key=lambda t: t[0][0], n=n, rng=rng)
    out = []
    for i, ((fam, pos, neg, para), v, r, a) in enumerate(picked):
        f = {"a": a, "v": v, "r": r}
        out.append(
            rec(
                f"perm_{i}",
                "permission",
                f"perm:{fam}",
                "authorized/approved -> ALLOW; not authorized/not approved -> DENY",
                pos.format(**f),
                "ALLOW",
                neg.format(**f),
                "DENY",
                para.format(**f),
                "ALLOW",
                ["permission_status"],
            )
        )
    return out


# ---------------------------------------------------------------------------
# 4. Action  (one verb swapped; object, article and preposition held fixed)
# ---------------------------------------------------------------------------

ACTION_TEMPLATES = [  # (family, safe, risky, paraphrased safe)
    (
        "2fa",
        "Enable two-factor authentication for {e}.",
        "Disable two-factor authentication for {e}.",
        "Turn on two-factor authentication for {e}.",
    ),
    (
        "admin",
        "Grant admin access to {e}.",
        "Revoke admin access to {e}.",
        "Give admin access to {e}.",
    ),
    (
        "account",
        "Create a user account for {e}.",
        "Delete a user account for {e}.",
        "Set up a user account for {e}.",
    ),
    (
        "apikey",
        "Activate the API key for {e}.",
        "Deactivate the API key for {e}.",
        "Enable the API key for {e}.",
    ),
    (
        "sshkey",
        "Add an SSH key for {e}.",
        "Remove an SSH key for {e}.",
        "Register an SSH key for {e}.",
    ),
]
ENTITIES = [
    "the new analyst",
    "the marketing team",
    "the vendor",
    "the intern",
    "the finance group",
    "the contractor",
]


def gen_action(n, rng):
    items = list(itertools.product(ACTION_TEMPLATES, ENTITIES))
    picked = sample_balanced(items, key=lambda t: t[0][0], n=n, rng=rng)
    out = []
    for i, ((fam, safe, risky, para), e) in enumerate(picked):
        out.append(
            rec(
                f"action_{i}",
                "action",
                f"action:{fam}",
                "create/enable/grant -> ALLOW; delete/disable/revoke -> REQUIRE_APPROVAL "
                "(synthetic policy, see realism caveat)",
                safe.format(e=e),
                "ALLOW",
                risky.format(e=e),
                "REQUIRE_APPROVAL",
                para.format(e=e),
                "ALLOW",
                ["action_verb"],
            )
        )
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
    (
        "calendar",
        "Create a calendar event with {p} at {t}.",
        "calendar.create_event",
        "List calendar events with {p} at {t}.",
        "calendar.list_events",
        "Schedule a calendar event with {p} at {t}.",
    ),
    (
        "email",
        "Send the email to {p} about {x}.",
        "email.send",
        "Find the email to {p} about {x}.",
        "email.search",
        "Dispatch the email to {p} about {x}.",
    ),
    (
        "ticket",
        "Create a ticket about {x}.",
        "ticket.create",
        "Find a ticket about {x}.",
        "ticket.search",
        "Open a ticket about {x}.",
    ),
    (
        "doc",
        "Create a document about {x}.",
        "doc.create",
        "Find a document about {x}.",
        "doc.search",
        "Draft a document about {x}.",
    ),
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
        out.append(
            rec(
                f"tool_{len(out)}",
                "tool_function",
                f"tool:{fam}",
                "label = intended tool call",
                a,
                tool_a,
                b,
                tool_b,
                pa,
                tool_a,
                ["intent_verb"],
                desc_a=TOOL_DESC[tool_a],
                desc_b=TOOL_DESC[tool_b],
            )
        )
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
        out.append(
            {
                "id": f"ctrl_{i}",
                "text_a": rng.choice(by_cat[c1])["text_a"],
                "text_b": rng.choice(by_cat[c2])["text_a"],
            }
        )
    return out


def assign_dataset_splits(triples, seed):
    grouped = {}
    for record in triples:
        group_id = record.get("policy_id", record["family"])
        grouped.setdefault((record["category"], group_id), []).append(record)

    rng = random.Random(seed)
    by_category = {}
    for category, group_id in grouped:
        by_category.setdefault(category, []).append(group_id)

    split_manifest = {}
    for category, group_ids in by_category.items():
        unique_groups = sorted(set(group_ids))
        rng.shuffle(unique_groups)
        if len(unique_groups) < 3:
            for group_id in unique_groups:
                split_manifest[f"{category}:{group_id}"] = "train"
                for record in grouped[(category, group_id)]:
                    record["dataset_split"] = "train"
            continue
        n_test = max(1, round(len(unique_groups) * 0.15))
        n_dev = max(1, round(len(unique_groups) * 0.15))
        test_groups = set(unique_groups[:n_test])
        dev_groups = set(unique_groups[n_test : n_test + n_dev])
        for group_id in unique_groups:
            split = (
                "test"
                if group_id in test_groups
                else "dev"
                if group_id in dev_groups
                else "train"
            )
            split_manifest[f"{category}:{group_id}"] = split
            for record in grouped[(category, group_id)]:
                record["dataset_split"] = split
    return split_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-category", type=int, default=30)
    ap.add_argument("--n-controls", type=int, default=20)
    ap.add_argument("--out", default="pilot_v2.json")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    triples = []
    for fn in (
        gen_numerical,
        gen_scope,
        gen_permission,
        gen_action,
        gen_tool,
        gen_context_policy,
    ):
        triples.extend(fn(args.n_per_category, rng))

    split_manifest = assign_dataset_splits(triples, args.seed)

    pair_keys = [(t["text_a"], t["text_b"]) for t in triples]
    assert len(pair_keys) == len(set(pair_keys)), "duplicate pairs generated"

    controls = gen_controls(triples, args.n_controls, rng)
    with open(args.out, "w") as f:
        json.dump(
            {
                "triples": triples,
                "controls": controls,
                "split_seed": args.seed,
                "split_grouping": "category + policy_id (family fallback)",
                "split_manifest": split_manifest,
            },
            f,
            indent=2,
        )

    print(f"Generated {len(triples)} triples + {len(controls)} controls -> {args.out}")
    counts = {}
    for t in triples:
        counts.setdefault(t["category"], set()).add(t["family"])
    for category, families in counts.items():
        n = sum(t["category"] == category for t in triples)
        print(f"  {category}: {n} triples, {len(families)} families")


if __name__ == "__main__":
    main()
