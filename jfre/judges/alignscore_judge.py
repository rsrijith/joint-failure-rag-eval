"""AlignScore-large judge (Zha et al., ACL 2023).

The official `alignscore` package is locked to torch<2 and pytorch-lightning<2.
We installed it with --no-deps and monkey-patched the small things it needs
from the new transformers (`AdamW` -> `torch.optim.AdamW`). Works because
the actual inference path uses RoBERTa-large via transformers, which works
fine under torch 2.x.

Checkpoint: AlignScore-large.ckpt (~4.7GB), downloaded from yzha/AlignScore
HuggingFace repo on first run.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

# Apply monkey patches before alignscore imports
import transformers
import torch as _torch

if not hasattr(transformers, "AdamW"):
    transformers.AdamW = _torch.optim.AdamW

from jfre.judges._llm_judge_prompt import format_passages
from jfre.types import JudgeVerdict, OperatorName, Seed


JUDGE_NAME = "alignscore_large"
_CKPT_REL = "data/raw/alignscore/models--yzha--AlignScore/snapshots/8509e78d25bb914939fc585c626500c9b2944249/AlignScore-large.ckpt"
_THRESHOLD = 0.5


@lru_cache(maxsize=1)
def _scorer():
    from alignscore import AlignScore

    ckpt = Path(_CKPT_REL).resolve()
    if not ckpt.exists():
        raise FileNotFoundError(
            f"AlignScore checkpoint not found at {ckpt}. "
            "Download with: python -c \"from huggingface_hub import hf_hub_download; "
            "hf_hub_download('yzha/AlignScore', 'AlignScore-large.ckpt', cache_dir='data/raw/alignscore')\""
        )
    return AlignScore(
        model="roberta-large",
        batch_size=8,
        device="cpu",
        ckpt_path=str(ckpt),
        evaluation_mode="nli_sp",
    )


def score(
    seed: Seed,
    answer_to_judge: str,
    operator: OperatorName | Literal["clean"],
) -> JudgeVerdict:
    premise = format_passages(seed.passages)
    try:
        scores = _scorer().score(contexts=[premise], claims=[answer_to_judge])
        raw_score = float(scores[0])
    except Exception as e:
        return JudgeVerdict(
            seed_id=seed.seed_id,
            operator=operator,
            judge_name=JUDGE_NAME,
            verdict="faithful",
            raw_score=None,
            judge_metadata={"error": str(e)[:200]},
        )

    verdict = "faithful" if raw_score >= _THRESHOLD else "unfaithful"
    return JudgeVerdict(
        seed_id=seed.seed_id,
        operator=operator,
        judge_name=JUDGE_NAME,
        verdict=verdict,
        raw_score=raw_score,
        judge_metadata={"threshold": _THRESHOLD, "score": raw_score},
    )
