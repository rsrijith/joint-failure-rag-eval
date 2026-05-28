"""ExpertQA seed loader (Malaviya et al., NAACL 2024). MIT-licensed.

ExpertQA has 484 expert-curated questions across 32 fields. Each question
has an LLM-generated answer with expert revisions and cited evidence from
web sources. The evidence string format is "[N] URL\\n\\n PASSAGE_TEXT" —
we extract PASSAGE_TEXT to use as our Passage.text.

ExpertQA evidence is all "supporting" by design — no distractor passages.
So distractor_parroting and citation_relocation operators will skip these
seeds (rule_passed=False with a "no distractor passages" note). The other
operators (entity_swap, numeric_drift, hedge_insertion, paraphrase_null)
benefit from ExpertQA's longer, more numeric-rich expert-revised answers.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Iterator
from pathlib import Path

from jfre.types import Passage, Seed


_LOCAL = Path("data/raw/expertqa/r2_compiled_anon.jsonl")

# Strip [N]-style citation markers from gold answers for fair perturbation.
_CITATION_RE = re.compile(r"\[\d+\]")


def _extract_passage_text(evidence_str: str) -> str | None:
    """ExpertQA evidence format: '[N] URL\n\n PASSAGE_TEXT'. Return PASSAGE_TEXT."""
    parts = evidence_str.split("\n\n", 1)
    if len(parts) < 2:
        return None
    text = parts[1].strip()
    return text if text else None


def _clean_answer(answer: str) -> str:
    """Strip [N] citation markers; collapse runs of whitespace introduced by removal."""
    cleaned = _CITATION_RE.sub("", answer)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def load(n: int, shuffle_seed: int = 42) -> Iterator[Seed]:
    """Yield up to N ExpertQA seeds.

    Skips seeds without an answer, without claims, or with fewer than 2
    unique evidence passages.
    """
    if not _LOCAL.exists():
        raise FileNotFoundError(
            f"{_LOCAL} not found. Run: "
            "curl -sL https://raw.githubusercontent.com/chaitanyamalaviya/ExpertQA/main/data/r2_compiled_anon.jsonl "
            f"-o {_LOCAL}"
        )

    with _LOCAL.open() as f:
        examples = [json.loads(line) for line in f]

    rng = random.Random(shuffle_seed)
    rng.shuffle(examples)

    yielded = 0
    for i, example in enumerate(examples):
        if yielded >= n:
            return

        if not example.get("answers"):
            continue
        # Use first available model variant (typically rr_gs_gpt4).
        model_name = next(iter(example["answers"].keys()))
        ans = example["answers"][model_name]

        # Prefer the expert-revised answer; fall back to the original GPT-4 string.
        raw_gold = ans.get("revised_answer_string") or ans.get("answer_string") or ""
        gold = _clean_answer(raw_gold)
        if not gold or len(gold.split()) < 8:
            continue  # too short to be informative

        # Only include evidence from claims that the EXPERT ANNOTATOR labeled
        # as having "Complete" support. ExpertQA web-scraped evidence is noisy
        # (URL bit-rot, wrong-page captures); the expert annotator's `support`
        # field is the ground-truth filter for whether a claim is actually
        # supported by its cited evidence.
        seen: set[str] = set()
        passages: list[Passage] = []
        n_supported_claims = 0
        for claim in ans.get("claims", []) or []:
            if claim.get("support") != "Complete":
                continue
            n_supported_claims += 1

            # revised_evidence may be a single STRING (the revised version of
            # the first evidence item) or a list. Original evidence is always
            # a list. Normalize to a list before iterating.
            revised = claim.get("revised_evidence")
            if isinstance(revised, str) and revised.strip():
                evidence_list = [revised]
            elif isinstance(revised, list) and revised:
                evidence_list = revised
            else:
                evidence_list = claim.get("evidence") or []

            for evid in evidence_list:
                if not isinstance(evid, str):
                    continue
                text = _extract_passage_text(evid)
                if text and text not in seen and len(text.split()) >= 10:
                    seen.add(text)
                    passages.append(Passage(text=text, is_relevant=True))

        if len(passages) < 2:
            continue  # need at least 2 passages for the multi-passage setting
        if n_supported_claims < 2:
            continue  # need at least 2 expert-confirmed-supported claims

        # Cap passages at 10 for parity with HotpotQA's distractor split.
        passages = passages[:10]

        yield Seed(
            seed_id=f"expertqa-{i:05d}",
            source="expertqa",
            question=example["question"].strip(),
            passages=passages,
            gold_answer=gold,
            metadata={
                "field": example.get("metadata", {}).get("field", "unknown"),
                "annotator_id": example.get("annotator_id", "unknown"),
                "model_variant": model_name,
            },
        )
        yielded += 1
