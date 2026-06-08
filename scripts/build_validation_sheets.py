"""Build human-validation labeling sheets (no API calls).

Produces two labeling tasks for a single annotator (lead author):

  TASK A -- paraphrase equivalence (100 items)
    For each paraphrase_null perturbation: is the paraphrase semantically
    equivalent to the gold answer? Validates the negative-control assumption
    and the "LLM judges over-reject paraphrases" finding.

  TASK B -- perturbation faithfulness spot-check (50 items, 10 per adversarial
    operator) For each adversarial perturbation: is the perturbed answer
    faithful to the retrieved passages? Validates that the intended label
    (perturbed = unfaithful) is correct.

Outputs (under validation/):
    RUBRIC.md                              -- labeling criteria (read first)
    paraphrase_equivalence_packet.md       -- readable items, Task A
    paraphrase_equivalence_labels.csv      -- fill-in sheet, Task A
    perturbation_spotcheck_packet.md       -- readable items, Task B
    perturbation_spotcheck_labels.csv      -- fill-in sheet, Task B
    _answer_key.json                       -- DO NOT OPEN until labeling done

Blinding: packets do NOT show the operator or the intended/judge labels.
Task B items are shuffled so operators are not grouped. The answer key
(operator, intended label, per-judge verdicts) is written separately and
prefixed with underscore.
"""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from jfre.seeds.expertqa import load as load_expertqa
from jfre.seeds.hotpotqa import load as load_hotpotqa

OUT = Path("validation")
RNG = random.Random(20260602)  # deterministic sampling

ADVERSARIAL_OPS = ["entity_swap", "numeric_drift", "hedge_insertion",
                   "distractor_parroting", "citation_relocation"]
N_PER_OP = 10
N_PARAPHRASE = 100

FUNCTIONAL = [
    "claude_sonnet_4_6", "mistral_large_2", "faithjudge_style_sonnet",
    "hhem_2_1_open", "minicheck_flan_t5_large", "ragas_style_sonnet",
]


def build_passage_map():
    """seed_id -> list of (text, is_relevant). Covers all pilot seed_ids."""
    pmap = {}
    for seed in load_hotpotqa(n=1000):
        pmap[seed.seed_id] = [(p.text, p.is_relevant) for p in seed.passages]
    for seed in load_expertqa(n=1500):
        pmap[seed.seed_id] = [(p.text, p.is_relevant) for p in seed.passages]
    return pmap


def load_perturbations():
    """(seed_id, op) -> perturbation record, rule_passed only."""
    out = {}
    for base in ("results/preview_pilot/perturbations.jsonl",
                 "results/citation_relocation_pilot/perturbations.jsonl"):
        p = Path(base)
        if not p.exists():
            continue
        for line in p.open():
            r = json.loads(line)
            if r.get("rule_passed"):
                out[(r["seed_id"], r["operator"])] = r
    return out


def load_verdicts():
    """(seed_id, op) -> {judge: verdict} for functional judges (non-error)."""
    out = defaultdict(dict)
    for base in ("results/preview_pilot/verdicts.jsonl",
                 "results/citation_relocation_pilot/verdicts.jsonl"):
        p = Path(base)
        if not p.exists():
            continue
        for line in p.open():
            r = json.loads(line)
            if r["judge"] in FUNCTIONAL and not r.get("metadata", {}).get("error"):
                out[(r["seed_id"], r["operator"])][r["judge"]] = r["verdict"]
    return out


def fmt_passages(passages, reveal_relevance=False):
    """Render passages with [PASSAGE N] labels. Relevance hidden by default."""
    lines = []
    for i, (text, is_rel) in enumerate(passages):
        tag = f" (relevant={is_rel})" if reveal_relevance else ""
        lines.append(f"[PASSAGE {i+1}]{tag} {text}")
    return "\n\n".join(lines)


def main():
    OUT.mkdir(exist_ok=True)
    pmap = build_passage_map()
    perts = load_perturbations()
    verdicts = load_verdicts()
    answer_key = {}

    # ---------- TASK A: paraphrase equivalence ----------
    para_cells = [(sid, op) for (sid, op) in perts if op == "paraphrase_null"]
    RNG.shuffle(para_cells)
    para_sample = para_cells[:N_PARAPHRASE]

    packet_a = ["# Task A -- Paraphrase Equivalence (100 items)\n",
                "Read RUBRIC.md first. For each item, decide whether Answer B conveys",
                "the EXACT SAME factual content as Answer A. Record yes/no in",
                "paraphrase_equivalence_labels.csv (column `equivalent_yes_no`).\n",
                "Do NOT open _answer_key.json until you finish.\n", "---\n"]
    labels_a = []
    for idx, (sid, op) in enumerate(para_sample, 1):
        item_id = f"PN-{idx:03d}"
        rec = perts[(sid, op)]
        packet_a.append(f"## {item_id}\n")
        packet_a.append(f"**Question:** {rec['question']}\n")
        packet_a.append(f"**Answer A (original):** {rec['gold_answer']}\n")
        packet_a.append(f"**Answer B (paraphrase):** {rec['perturbed_answer']}\n")
        packet_a.append("**Equivalent? (yes/no):** ______\n\n---\n")
        labels_a.append({"item_id": item_id, "equivalent_yes_no": "", "notes": ""})
        answer_key[item_id] = {
            "seed_id": sid, "operator": op, "intended_label": "faithful",
            "judge_verdicts": verdicts.get((sid, op), {}),
        }

    # ---------- TASK B: perturbation faithfulness spot-check ----------
    spot_items = []
    for op in ADVERSARIAL_OPS:
        cells = [(sid, o) for (sid, o) in perts if o == op]
        RNG.shuffle(cells)
        spot_items.extend(cells[:N_PER_OP])
    RNG.shuffle(spot_items)  # mix operators so they're not grouped

    packet_b = ["# Task B -- Perturbation Faithfulness Spot-Check (50 items)\n",
                "Read RUBRIC.md first. For each item, decide whether the Candidate",
                "Answer is FAITHFUL or UNFAITHFUL to the retrieved passages, using the",
                "rubric definition. Record faithful/unfaithful in",
                "perturbation_spotcheck_labels.csv (column `faithful_yes_no`).\n",
                "Do NOT open _answer_key.json until you finish.\n", "---\n"]
    labels_b = []
    for idx, (sid, op) in enumerate(spot_items, 1):
        item_id = f"SP-{idx:03d}"
        rec = perts[(sid, op)]
        passages = pmap.get(sid, [])
        packet_b.append(f"## {item_id}\n")
        packet_b.append(f"**Question:** {rec['question']}\n")
        packet_b.append(f"**Retrieved passages:**\n\n{fmt_passages(passages)}\n")
        packet_b.append(f"**Candidate answer:** {rec['perturbed_answer']}\n")
        packet_b.append("**Faithful or Unfaithful?:** ______\n\n---\n")
        labels_b.append({"item_id": item_id, "faithful_yes_no": "", "notes": ""})
        answer_key[item_id] = {
            "seed_id": sid, "operator": op, "intended_label": "unfaithful",
            "judge_verdicts": verdicts.get((sid, op), {}),
            "n_passages": len(passages),
        }

    # ---------- write everything ----------
    (OUT / "paraphrase_equivalence_packet.md").write_text("\n".join(packet_a))
    (OUT / "perturbation_spotcheck_packet.md").write_text("\n".join(packet_b))

    with (OUT / "paraphrase_equivalence_labels.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["item_id", "equivalent_yes_no", "notes"])
        w.writeheader()
        w.writerows(labels_a)

    with (OUT / "perturbation_spotcheck_labels.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["item_id", "faithful_yes_no", "notes"])
        w.writeheader()
        w.writerows(labels_b)

    (OUT / "_answer_key.json").write_text(json.dumps(answer_key, indent=2))

    # counts by operator for the spot-check
    spot_op_counts = defaultdict(int)
    for item_id, meta in answer_key.items():
        if item_id.startswith("SP-"):
            spot_op_counts[meta["operator"]] += 1

    print(f"Wrote validation sheets to {OUT}/")
    print(f"  Task A (paraphrase equivalence): {len(labels_a)} items")
    print(f"  Task B (perturbation spot-check): {len(labels_b)} items")
    print(f"    by operator: {dict(spot_op_counts)}")
    print(f"  Answer key: {OUT}/_answer_key.json ({len(answer_key)} items) -- keep closed until done")


if __name__ == "__main__":
    main()
