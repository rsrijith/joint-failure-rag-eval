"""Export the 100 accepted citation seeds for cross-model (Gemini) annotation.

Produces:
  validation/gemini_annotation_input.jsonl   one row per seed: seed_id, question,
      n_passages, passages (numbered text), gold_answer (UNCITED)
  validation/gemini_annotation_prompt.txt    the exact annotation instruction
  validation/gemini_annotation_batches/      the same items split into paste-able
      markdown batches of 20 for the AI Studio / chat UI

The cross-model run re-annotates these gold answers with [N] markers using a
NON-judge model (Gemini), to break the Claude-annotates / Claude-judges
circularity in the attribution-prompt experiment.
"""

from __future__ import annotations

import json
from pathlib import Path

from jfre.seeds.expertqa import load as load_expertqa

OUTDIR = Path("validation")
BATCHDIR = OUTDIR / "gemini_annotation_batches"

PROMPT = """You are annotating an answer with citation markers.

You are given a QUESTION, a list of numbered PASSAGES, and an ANSWER that has no citations. For each factual claim in the answer, insert a citation marker [N] immediately after the claim, where N is the number of the passage that DIRECTLY supports that claim.

Rules:
- Only cite a passage that explicitly contains the cited claim's content.
- One marker per claim, placed at the end of the sentence or clause containing the claim.
- If several passages support a claim, pick the single most directly relevant one.
- If no passage supports a claim, leave it without a citation.
- Do NOT change any of the answer's wording. Only insert [N] markers.

Return ONLY the annotated answer text with [N] markers inserted. No commentary, no JSON, no markdown fences.
"""


def fmt_passages(passages) -> str:
    return "\n\n".join(f"[PASSAGE {i+1}] {p.text}" for i, p in enumerate(passages))


def main():
    OUTDIR.mkdir(exist_ok=True)
    BATCHDIR.mkdir(exist_ok=True)

    accepted = [json.loads(l)["seed_id"] for l in
                Path("results/citation_relocation_pilot/seeds.jsonl").open()
                if json.loads(l).get("accepted")]
    seedmap = {s.seed_id: s for s in load_expertqa(n=1500)}
    rows = []
    for sid in accepted:
        s = seedmap.get(sid)
        if s is None:
            continue
        rows.append({
            "seed_id": sid,
            "question": s.question,
            "n_passages": len(s.passages),
            "passages": fmt_passages(s.passages),
            "gold_answer": s.gold_answer,
        })

    # 1. JSONL
    with (OUTDIR / "gemini_annotation_input.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    # 2. prompt
    (OUTDIR / "gemini_annotation_prompt.txt").write_text(PROMPT)

    # 3. paste-able markdown batches of 20
    B = 20
    for bi in range(0, len(rows), B):
        chunk = rows[bi:bi + B]
        lines = [f"# Gemini annotation batch {bi//B + 1} (items {bi+1}-{bi+len(chunk)})\n",
                 "Apply the prompt in gemini_annotation_prompt.txt to EACH item below.",
                 "Return one line per item as: SEEDID<TAB>annotated answer text.\n", "---\n"]
        for r in chunk:
            lines.append(f"## {r['seed_id']}\n")
            lines.append(f"QUESTION: {r['question']}\n")
            lines.append(f"PASSAGES:\n{r['passages']}\n")
            lines.append(f"ANSWER (no citations): {r['gold_answer']}\n")
            lines.append("---\n")
        (BATCHDIR / f"batch_{bi//B + 1}.md").write_text("\n".join(lines))

    print(f"Wrote {len(rows)} seeds:")
    print(f"  validation/gemini_annotation_input.jsonl")
    print(f"  validation/gemini_annotation_prompt.txt")
    print(f"  validation/gemini_annotation_batches/batch_1..{(len(rows)+B-1)//B}.md")


if __name__ == "__main__":
    main()
