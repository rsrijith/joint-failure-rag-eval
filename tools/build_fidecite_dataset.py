"""Build the Fidecite citation-relocation dataset for a Hugging Face release.

Joins the study's citation_relocation perturbations with the passages
reconstructed from the seed loaders (the perturbation files do not store
passages), and writes a single JSONL matching the card in ``dataset/README.md``.

Run this in an environment that has the seed datasets available (the study repo,
with ``jfre[data]`` installed and ``data/raw`` populated). It only READS the
perturbations and seeds; it writes the dataset under ``dataset/data/``.

    python tools/build_fidecite_dataset.py \
        --perturbations /path/to/GroundLM/results/citation_relocation_pilot/perturbations.jsonl \
        --out dataset/data/test.jsonl

Each output row:
    seed_id, source, question, passages (list[str]), cited_answer,
    relocated_answer, permutation (dict[str,int]), n_citations
Licensing per source is in ``dataset/NOTICES.md`` (ExpertQA/PubMedQA MIT,
HotpotQA CC-BY-SA 4.0). Reconcile counts to the paper's final table before release.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path


def _load_passages_by_seed_id() -> dict[str, list[str]]:
    """Reconstruct seed_id -> passages from the deterministic loaders.

    Loaders are imported lazily so this file imports without ``jfre[data]``.
    """
    mapping: dict[str, list[str]] = {}
    from jfre.seeds import expertqa, hotpotqa  # requires jfre[data] + data/raw

    for loader, n in ((expertqa.load, 1500), (hotpotqa.load, 1000)):
        try:
            for seed in loader(n):
                mapping[seed.seed_id] = [p.text for p in seed.passages]
        except Exception as e:  # a missing source just yields fewer rows
            print(f"  warning: loader {loader.__module__} failed: {e}")
    # PubMedQA is optional; include if present.
    try:
        from jfre.seeds import pubmedqa  # type: ignore

        for seed in pubmedqa.load(1000):
            mapping[seed.seed_id] = [p.text for p in seed.passages]
    except Exception:
        pass
    return mapping


def _coerce_dict(value) -> dict:
    if isinstance(value, dict):
        return value
    try:
        return ast.literal_eval(value) if isinstance(value, str) else {}
    except Exception:
        return {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--perturbations", required=True, type=Path)
    ap.add_argument("--out", default=Path("dataset/data/test.jsonl"), type=Path)
    args = ap.parse_args()

    print("Reconstructing passages from seed loaders...")
    passages_by_id = _load_passages_by_seed_id()
    print(f"  {len(passages_by_id)} seeds with passages")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = skipped = 0
    with args.perturbations.open() as fin, args.out.open("w") as fout:
        for line in fin:
            row = json.loads(line)
            if row.get("operator") != "citation_relocation":
                continue
            if str(row.get("rule_passed")).lower() != "true":
                continue
            passages = passages_by_id.get(row["seed_id"])
            if not passages:
                skipped += 1
                continue
            diff = _coerce_dict(row.get("edit_diff"))
            permutation = {str(k): int(v) for k, v in _coerce_dict(diff.get("permutation")).items()}
            out_row = {
                "seed_id": row["seed_id"],
                "source": row.get("source", "expertqa"),
                "question": row["question"],
                "passages": passages,
                "cited_answer": row.get("cited_answer") or diff.get("original_cited_answer", ""),
                "relocated_answer": row["perturbed_answer"],
                "permutation": permutation,
                "n_citations": diff.get("n_citations_swapped", len(permutation)),
            }
            fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} rows to {args.out} ({skipped} skipped for missing passages).")
    print("Next: copy dataset/NOTICES.md alongside the rows and push to the HF dataset repo.")


if __name__ == "__main__":
    main()
