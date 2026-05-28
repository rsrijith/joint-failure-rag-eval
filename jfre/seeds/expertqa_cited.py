"""Citation-annotated ExpertQA seed loader for the citation_relocation operator.

This loader wraps `expertqa.load` and adds Claude-Sonnet annotation that
inserts [N] passage-index citation markers into the gold answer. The
output annotated answer becomes `seed.metadata['cited_answer']`, where
[N] refers to the 1-indexed passage in `seed.passages` (matching the
[PASSAGE N] format used by the LLM-judge prompt).

The citation_relocation operator then swaps these [N] markers according
to a non-identity permutation to produce attribution-broken-but-content-
intact perturbations.

Annotations are cached to data/cache/expertqa_cited.jsonl so re-runs do
not re-pay for the Claude calls.

Filter rule: a seed is included only if Claude's annotation produces >= 2
DISTINCT citation indices in the answer. Otherwise there is nothing for
the operator to relocate.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path

import anthropic

from jfre.judges._retry import retry_on_rate_limit
from jfre.seeds.expertqa import load as load_raw
from jfre.types import Seed


_CACHE_DIR = Path("data/cache")
_CACHE_FILE = _CACHE_DIR / "expertqa_cited.jsonl"
_CITATION_RE = re.compile(r"\[(\d+)\]")
_MODEL = "claude-sonnet-4-6"


_ANNOTATE_PROMPT = """You are given a question, a list of numbered retrieved passages, and a candidate answer (with no citations). For each factual claim in the answer, insert a citation marker `[N]` immediately after the claim, where N is the 1-indexed number of the passage that DIRECTLY supports that claim.

Rules:
- Only cite passages that explicitly contain the cited claim's content.
- One citation marker per claim, placed at the end of the sentence or clause containing the claim.
- If multiple passages support a claim, pick the most directly relevant one.
- If no passage supports a claim, leave it without a citation.
- Do not change any of the answer's wording. Only insert the [N] markers.

Question:
{question}

Retrieved passages:
{passages}

Answer (no citations):
{answer}

Return ONLY the annotated answer text with [N] markers inserted. No commentary, no JSON, no markdown fences."""


@lru_cache(maxsize=1)
def _client() -> anthropic.Anthropic:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    return anthropic.Anthropic(api_key=key)


def _format_passages(passages) -> str:
    return "\n\n".join(f"[PASSAGE {i + 1}] {p.text}" for i, p in enumerate(passages))


def _load_cache() -> dict[str, dict]:
    if not _CACHE_FILE.exists():
        return {}
    out: dict[str, dict] = {}
    for line in _CACHE_FILE.open():
        rec = json.loads(line)
        out[rec["seed_id"]] = rec
    return out


def _append_cache(record: dict) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with _CACHE_FILE.open("a") as f:
        f.write(json.dumps(record) + "\n")


@retry_on_rate_limit()
def _annotate(question: str, passages, answer: str) -> str:
    msg = _client().messages.create(
        model=_MODEL,
        max_tokens=2048,
        temperature=0,
        system=[{
            "type": "text",
            "text": "You annotate answers with passage-index citation markers. Return only the annotated answer text.",
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{
            "role": "user",
            "content": _ANNOTATE_PROMPT.format(
                question=question,
                passages=_format_passages(passages),
                answer=answer,
            ),
        }],
    )
    return "".join(b.text for b in msg.content if b.type == "text").strip()


def load(n: int, shuffle_seed: int = 42) -> Iterator[Seed]:
    """Yield up to N citation-annotated ExpertQA seeds.

    Each yielded seed has `metadata['cited_answer']` containing the gold
    answer with [N] citation markers inserted by Claude. Seeds where the
    annotation produces fewer than 2 distinct citation indices are skipped.
    """
    cache = _load_cache()
    yielded = 0

    # Over-request raw seeds so we can drop ones whose annotations don't
    # yield >= 2 distinct citations.
    raw_iter = load_raw(n=n * 3, shuffle_seed=shuffle_seed)

    for seed in raw_iter:
        if yielded >= n:
            return

        if seed.seed_id in cache:
            cited = cache[seed.seed_id]["cited_answer"]
            indices = cache[seed.seed_id]["distinct_indices"]
        else:
            try:
                cited = _annotate(seed.question, seed.passages, seed.gold_answer)
            except Exception as e:
                _append_cache({
                    "seed_id": seed.seed_id,
                    "cited_answer": "",
                    "distinct_indices": [],
                    "error": str(e)[:200],
                })
                continue

            # Extract distinct citation indices that fall in [1, N_passages].
            n_passages = len(seed.passages)
            distinct = sorted({
                int(m) for m in _CITATION_RE.findall(cited)
                if 1 <= int(m) <= n_passages
            })
            _append_cache({
                "seed_id": seed.seed_id,
                "cited_answer": cited,
                "distinct_indices": distinct,
            })
            indices = distinct

        if len(indices) < 2:
            continue

        seed.metadata["cited_answer"] = cited
        seed.metadata["distinct_citation_indices"] = indices
        yield seed
        yielded += 1
