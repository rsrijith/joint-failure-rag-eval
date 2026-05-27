"""MiniCheck-Flan-T5-Large: fine-tuned fact-checker (Tang et al., EMNLP 2024).

Sequence classification on (premise=passages, hypothesis=answer). Threshold 0.5.
Loads on first call (~3 GB download). Runs locally on CPU/MPS.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from jfre.judges._llm_judge_prompt import format_passages
from jfre.types import JudgeVerdict, OperatorName, Seed


JUDGE_NAME = "minicheck_flan_t5_large"
_MODEL_ID = "lytang/MiniCheck-Flan-T5-Large"
_THRESHOLD = 0.5


@lru_cache(maxsize=1)
def _scorer():
    """Use the official `minicheck` package if installed, else fall back to
    transformers. The official package handles the chunking + prompting that
    MiniCheck expects for long premises."""
    try:
        from minicheck.minicheck import MiniCheck

        return ("minicheck", MiniCheck(model_name="flan-t5-large", cache_dir="./hf_cache"))
    except ImportError:
        # Fallback: load via transformers and run a single forward pass
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(_MODEL_ID)
        mdl = AutoModelForSeq2SeqLM.from_pretrained(_MODEL_ID)
        mdl.eval()
        return ("transformers", (tok, mdl))


def _score_via_minicheck(scorer, premise: str, hypothesis: str) -> float:
    _, raw_probs, _, _ = scorer.score(docs=[premise], claims=[hypothesis])
    return float(raw_probs[0])


def _score_via_transformers(tok_and_mdl, premise: str, hypothesis: str) -> float:
    import torch

    tok, mdl = tok_and_mdl
    prompt = f"predict: {premise}</s> claim: {hypothesis}"
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=2048)
    with torch.no_grad():
        outputs = mdl.generate(**inputs, max_new_tokens=2, output_scores=True, return_dict_in_generate=True)
    # MiniCheck-Flan-T5 outputs "0" or "1"; we look at the probability of token "1".
    first_token_logits = outputs.scores[0][0]
    probs = torch.softmax(first_token_logits, dim=-1)
    token_one_id = tok("1", add_special_tokens=False).input_ids[0]
    return float(probs[token_one_id].item())


def score(
    seed: Seed,
    answer_to_judge: str,
    operator: OperatorName | Literal["clean"],
) -> JudgeVerdict:
    """Score one (seed, candidate answer) pair via MiniCheck."""
    premise = format_passages(seed.passages)
    backend, obj = _scorer()

    if backend == "minicheck":
        raw_score = _score_via_minicheck(obj, premise, answer_to_judge)
    else:
        raw_score = _score_via_transformers(obj, premise, answer_to_judge)

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
            "backend": backend,
        },
    )
