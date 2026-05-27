"""HHEM-2.1-Open: Vectara's fine-tuned hallucination detection model.

A small (~184M params) NLI-based hallucination detector. Runs locally on
CPU or MPS. Per the model card, returns a probability that the hypothesis
(answer) is consistent with the premise (retrieved passages); threshold 0.5
separates faithful from unfaithful.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from jfre.judges._llm_judge_prompt import format_passages
from jfre.types import JudgeVerdict, OperatorName, Seed


JUDGE_NAME = "hhem_2_1_open"
_MODEL_ID = "vectara/hallucination_evaluation_model"
_THRESHOLD = 0.5


@lru_cache(maxsize=1)
def _model():
    """Load HHEM-2.1-Open once. trust_remote_code is required because the
    model ships a custom predict() method in its repo."""
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(
        _MODEL_ID,
        trust_remote_code=True,
    )
    model.eval()
    return model


def score(
    seed: Seed,
    answer_to_judge: str,
    operator: OperatorName | Literal["clean"],
) -> JudgeVerdict:
    """Score one (seed, candidate answer) pair.

    Premise = concatenation of all retrieved passages (relevant + distractor),
    matching the assessor's full evidence pool. Hypothesis = the candidate
    answer. HHEM returns one probability in [0, 1].
    """
    premise = format_passages(seed.passages)
    pairs = [(premise, answer_to_judge)]

    scores = _model().predict(pairs)
    # predict() may return a tensor or a list; coerce to float.
    raw_score = float(scores[0] if hasattr(scores, "__getitem__") else scores)

    verdict = "faithful" if raw_score >= _THRESHOLD else "unfaithful"

    return JudgeVerdict(
        seed_id=seed.seed_id,
        operator=operator,
        judge_name=JUDGE_NAME,
        verdict=verdict,
        raw_score=raw_score,
        judge_metadata={
            "threshold": _THRESHOLD,
            "score": raw_score,
            "model": _MODEL_ID,
        },
    )
