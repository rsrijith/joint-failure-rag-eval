"""CONTROL for the attribution-prompt experiment.

Scores the CLEAN cited answers (correct [N] attributions) with the SAME
attribution-aware prompt, for the 3 LLM judges. If the attribution prompt is a
genuine discriminator it should call these FAITHFUL (citations are correct). If
it calls them unfaithful too, the low FNR on citation_relocation is just
strictness, not detection (the AlignScore trap).

Output: results/citation_relocation_pilot/verdicts_attribution_clean.jsonl
"""

from __future__ import annotations

import json
from pathlib import Path

from jfre.seeds.expertqa import load as load_expertqa
from scripts.attribution_prompt_experiment import (
    JUDGES, _ATTR_PROMPT, _parse,
)
from jfre.judges._llm_judge_prompt import format_passages

OUT = Path("results/citation_relocation_pilot/verdicts_attribution_clean.jsonl")


def main():
    seedmap = {s.seed_id: s for s in load_expertqa(n=1500)}
    cited = {}
    for line in Path("results/citation_relocation_pilot/seeds.jsonl").open():
        s = json.loads(line)
        if s.get("accepted") and s.get("cited_answer"):
            cited[s["seed_id"]] = s["cited_answer"]
    cells = [sid for sid in cited if sid in seedmap]

    done = set()
    if OUT.exists():
        for line in OUT.open():
            r = json.loads(line)
            done.add((r["seed_id"], r["judge"]))

    work = [(sid, j) for sid in cells for j in JUDGES if (sid, j) not in done]
    print(f"{len(work)} clean_cited calls (control)", flush=True)

    with OUT.open("a") as f:
        for i, (sid, judge) in enumerate(work, 1):
            seed = seedmap[sid]
            prompt = _ATTR_PROMPT.format(
                question=seed.question,
                passages=format_passages(seed.passages),
                answer=cited[sid],
            )
            try:
                verdict = _parse(JUDGES[judge](prompt))
            except Exception as e:
                verdict = "error"
            f.write(json.dumps({"seed_id": sid, "operator": "clean_cited",
                                "judge": judge, "verdict": verdict,
                                "prompt_type": "attribution_aware"}) + "\n")
            f.flush()
            if i % 30 == 0:
                print(f"  {i}/{len(work)}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
