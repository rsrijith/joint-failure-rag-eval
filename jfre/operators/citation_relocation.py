"""citation_relocation operator: swap [N] citation indices in a cited answer.

The 6th perturbation operator. Unlike the other 5 (entity_swap,
numeric_drift, hedge_insertion, distractor_parroting, paraphrase_null),
this one does NOT modify the answer's factual content. Every claim in
the perturbed answer remains supported by SOME passage in the context.
Only the citation-attribution mapping is corrupted.

Mechanism:
1. Read seed.metadata['cited_answer'] (set by expertqa_cited loader).
2. Identify the set of distinct [N] citation indices used.
3. Choose a non-identity permutation of those indices.
4. Replace each [N] with [σ(N)] throughout the answer.

The resulting answer is "claim-level faithful" (every claim is still in
context) but "attribution-level unfaithful" (each claim is cited to a
passage that does not support it). LLM-as-judges that verify claim-in-
context without checking citation correctness will tend to mark the
perturbed answer as faithful. Claim-decomposition judges (RAGAS) and
local NLI judges (HHEM, MiniCheck, AlignScore) likewise verify against
the full context rather than per-citation, so they should also miss
this perturbation. That is the paper's "joint failure" hypothesis for
this operator.

Requires the seed to have come from `expertqa_cited.load()`. For seeds
without a cited_answer in metadata, rule_passed=False.
"""

from __future__ import annotations

import random
import re

from jfre.types import OperatorName, Perturbation, Seed


OPERATOR_NAME: OperatorName = "citation_relocation"

_CITATION_RE = re.compile(r"\[(\d+)\]")
_PERMUTATION_RNG_SEED = 42


def _derangement(indices: list[int], rng: random.Random) -> list[int]:
    """Return a permutation of `indices` with no fixed points.

    For 2 indices, the only derangement is the swap. For 3+, we sample
    permutations and retry until none of σ(i) == i. Fixed-point-free
    permutations exist for any n >= 2 (count is the subfactorial !n).
    """
    if len(indices) < 2:
        return indices[:]
    for _ in range(100):
        shuffled = indices[:]
        rng.shuffle(shuffled)
        if all(a != b for a, b in zip(indices, shuffled)):
            return shuffled
    # Fallback: cyclic shift always avoids fixed points for n >= 2.
    return indices[1:] + indices[:1]


def generate(seed: Seed) -> Perturbation:
    cited = seed.metadata.get("cited_answer")
    if not cited:
        return Perturbation(
            seed_id=seed.seed_id,
            operator=OPERATOR_NAME,
            rule_passed=False,
            rule_notes="no cited_answer in metadata (seed not from expertqa_cited loader)",
            perturbed_answer="",
            edit_diff={},
        )

    distinct = sorted({int(m) for m in _CITATION_RE.findall(cited)})
    if len(distinct) < 2:
        return Perturbation(
            seed_id=seed.seed_id,
            operator=OPERATOR_NAME,
            rule_passed=False,
            rule_notes=f"only {len(distinct)} distinct citation indices; need >= 2 to relocate",
            perturbed_answer="",
            edit_diff={},
        )

    rng = random.Random(f"{_PERMUTATION_RNG_SEED}-{seed.seed_id}")
    permuted = _derangement(distinct, rng)
    mapping = dict(zip(distinct, permuted))

    # Replace each [N] with [σ(N)] in a single pass. We build a regex-callback
    # so that overlapping replacements (e.g., [10] -> [12]) don't double-fire.
    def _sub(match: re.Match) -> str:
        n = int(match.group(1))
        return f"[{mapping.get(n, n)}]"

    perturbed = _CITATION_RE.sub(_sub, cited)

    if perturbed == cited:
        return Perturbation(
            seed_id=seed.seed_id,
            operator=OPERATOR_NAME,
            rule_passed=False,
            rule_notes="derangement produced no change (should be impossible)",
            perturbed_answer="",
            edit_diff={},
        )

    return Perturbation(
        seed_id=seed.seed_id,
        operator=OPERATOR_NAME,
        rule_passed=True,
        rule_notes=f"swapped {len(distinct)} distinct citations via derangement",
        perturbed_answer=perturbed,
        edit_diff={
            "original_cited_answer": cited,
            "permutation": mapping,
            "n_citations_swapped": len(distinct),
        },
    )
