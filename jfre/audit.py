"""Audit your own faithfulness judge for the citation-attribution blind spot.

Most deployed RAG faithfulness judges check *content support*: is each claim
supported by the retrieved passages *somewhere*. They do not check
*attribution*: does each claim trace to the specific passage it actually cites.

This module lets you measure that gap on your own judge in a few lines, using
only the standard library. You bring a judge callable; we relocate the citation
markers in a cited answer so every claim ends up attributed to a passage that
does not support it (while the claim stays supported by some *other* passage),
then check whether your judge notices.

    from jfre import make_seed, audit_judge

    def my_judge(question, passages, answer) -> bool:   # True == "faithful"
        ...  # call your RAGAS / HHEM / LLM-as-judge / custom metric

    seeds = [
        make_seed(
            question="When did the two events happen?",
            passages=["Event A happened in 1991.", "Event B happened in 2003."],
            cited_answer="Event A happened in 1991 [1]. Event B happened in 2003 [2].",
        ),
    ]
    result = audit_judge(my_judge, seeds)
    print(result.summary())

A judge that catches attribution errors flips to "unfaithful" on the relocated
answer. A judge that still says "faithful" has the blind spot. The fraction of
relocated answers a judge still passes is its attribution false-negative rate
(FNR) -- the headline number from the audit (39%-94% across seven deployed
judges in the reference study).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from jfre.operators import citation_relocation
from jfre.types import Passage, Seed

# A judge takes (question, passages, answer) and returns True if it considers the
# answer faithful, False otherwise. ``passages`` is a list of plain strings; the
# answer's [N] markers are 1-indexed into that list.
Judge = Callable[[str, list[str], str], bool]


def make_seed(
    question: str,
    passages: Sequence[str],
    cited_answer: str,
    seed_id: str = "byo-0",
) -> Seed:
    """Build a seed from your own (question, passages, cited answer).

    ``cited_answer`` must contain ``[N]`` citation markers, where N is the
    1-indexed passage being cited (matching the order of ``passages``). At least
    two *distinct* markers are required for relocation to have something to do.
    """
    ps = [Passage(text=str(p), is_relevant=True) for p in passages]
    return Seed(
        seed_id=seed_id,
        source="expertqa",  # nominal; only used for bookkeeping
        question=question,
        passages=ps,
        gold_answer=cited_answer,
        metadata={"cited_answer": cited_answer},
    )


def relocate_citations(seed: Seed) -> str:
    """Return the attribution-broken answer for a seed.

    Every ``[N]`` marker is re-pointed by a fixed-point-free permutation, so each
    claim now cites a passage that does not support it while staying supported by
    some other passage. Raises ``ValueError`` if the seed cannot be relocated
    (no cited answer, or fewer than two distinct citation markers).
    """
    pert = citation_relocation.generate(seed)
    if not pert.rule_passed:
        raise ValueError(f"cannot relocate citations: {pert.rule_notes}")
    return pert.perturbed_answer


def _passage_texts(seed: Seed) -> list[str]:
    return [p.text for p in seed.passages]


@dataclass
class AuditResult:
    """Outcome of auditing one judge across a set of seeds.

    n_attackable    seeds with >= 2 distinct citation markers (relocatable).
    clean_pass      relocatable seeds the judge called faithful when correctly cited.
    attacked_pass   relocatable seeds the judge STILL called faithful after relocation.
                    These are misses: the attribution error went undetected.
    n_skipped       seeds that could not be relocated (too few citations).
    """

    n_total: int
    n_attackable: int
    clean_pass: int
    attacked_pass: int
    n_skipped: int

    @property
    def false_negative_rate(self) -> float:
        """Share of relocated answers the judge wrongly passes (the blind spot)."""
        if self.n_attackable == 0:
            return float("nan")
        return self.attacked_pass / self.n_attackable

    @property
    def clean_pass_rate(self) -> float:
        """Share of correctly-cited answers the judge passes (want this high)."""
        if self.n_attackable == 0:
            return float("nan")
        return self.clean_pass / self.n_attackable

    @property
    def catches_attribution(self) -> int:
        """Relocated answers the judge correctly flagged as unfaithful."""
        return self.n_attackable - self.attacked_pass

    def summary(self) -> str:
        if self.n_attackable == 0:
            return (
                f"No attackable seeds (need >= 2 distinct [N] markers); "
                f"{self.n_skipped} skipped of {self.n_total}."
            )
        fnr = self.false_negative_rate
        if fnr >= 0.5:
            verdict = "BLIND SPOT: judge misses most citation misattributions"
        elif fnr > 0:
            verdict = "PARTIAL: judge catches some citation misattributions, but not all"
        else:
            verdict = "CLEAN: judge caught every citation misattribution in this set"
        return (
            f"Attribution false-negative rate: {fnr:.0%} "
            f"({self.attacked_pass}/{self.n_attackable} relocated answers still passed).\n"
            f"Clean-pass rate: {self.clean_pass_rate:.0%} "
            f"({self.clean_pass}/{self.n_attackable} correctly-cited answers passed).\n"
            f"Skipped (too few citations): {self.n_skipped} of {self.n_total}.\n"
            f"=> {verdict}."
        )


def audit_judge(judge: Judge, seeds: Sequence[Seed]) -> AuditResult:
    """Audit ``judge`` for the citation-attribution blind spot across ``seeds``.

    For each seed we present the judge with (a) the correctly-cited answer, which
    a sound judge passes, and (b) the citation-relocated answer, which a sound
    judge rejects. We count how often the judge is fooled by (b).
    """
    n_total = len(seeds)
    n_attackable = clean_pass = attacked_pass = n_skipped = 0

    for seed in seeds:
        try:
            attacked_answer = relocate_citations(seed)
        except ValueError:
            n_skipped += 1
            continue

        n_attackable += 1
        passages = _passage_texts(seed)
        clean_answer = seed.metadata["cited_answer"]

        if judge(seed.question, passages, clean_answer):
            clean_pass += 1
        if judge(seed.question, passages, attacked_answer):
            attacked_pass += 1

    return AuditResult(
        n_total=n_total,
        n_attackable=n_attackable,
        clean_pass=clean_pass,
        attacked_pass=attacked_pass,
        n_skipped=n_skipped,
    )
