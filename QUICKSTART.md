# Quickstart: audit your own judge

You have a faithfulness check in your RAG stack — RAGAS, an HHEM/MiniCheck/
AlignScore NLI model, an LLM-as-judge, or your own metric. This shows whether it
catches **citation misattribution**: a claim cited to a passage that does not
support it (while the claim is supported by some *other* passage).

## 1. Install (no heavy dependencies)

```bash
# Not on PyPI yet; install from the repo.
pip install "git+https://github.com/rsrijith/joint-failure-rag-eval.git"
# POST-RELEASE:  pip install jfre
```

## 2. Wrap your judge behind the adapter

A `jfre` judge is any callable `(question, passages, answer) -> bool`, where
`True` means "faithful." `passages` is a list of plain strings; the answer's
`[N]` markers are 1-indexed into that list.

```python
def my_judge(question, passages, answer) -> bool:
    # Examples:
    #   RAGAS:        run faithfulness over (question, passages, answer); threshold the score
    #   HHEM/NLI:     entail(answer, premise=" ".join(passages)) > 0.5
    #   LLM-as-judge: prompt your model for faithful/unfaithful
    ...
```

## 3. Bring a few cited examples from your own data

Each example is a `(question, passages, cited_answer)` triple where the answer
carries `[N]` citation markers. Two or more distinct markers are needed so there
is something to relocate.

```python
from jfre import make_seed

seeds = [
    make_seed(
        question="When did each event occur?",
        passages=["The first satellite launched in 1957.",
                  "The first Moon landing was in 1969.",
                  "The first reusable rocket landing was in 2015."],
        cited_answer="The first satellite launched in 1957 [1]. "
                     "The first Moon landing was in 1969 [2]. "
                     "The first reusable rocket landing was in 2015 [3].",
    ),
    # ... more of your own seeds
]
```

## 4. Run the audit

```python
from jfre import audit_judge

result = audit_judge(my_judge, seeds)
print(result.summary())
print("attribution FNR:", result.false_negative_rate)   # 0.0 good, 1.0 = totally blind
```

`audit_judge` presents your judge with the correctly-cited answer (a sound judge
passes it) and the citation-relocated answer (a sound judge rejects it), and
reports the false-negative rate — how often relocation slipped past.

## 5. If your LLM judge is blind, apply the fix

```python
from jfre.fix import make_attribution_judge

def call_llm(prompt: str) -> str:
    ...   # your model, returns raw text

fixed = make_attribution_judge(call_llm)
print(audit_judge(fixed, seeds).summary())   # miss rate should fall toward ~3%
```

See [FIX.md](FIX.md) for the prompt and the NLI caveat.

## Inspecting one attack

```python
from jfre import make_seed, relocate_citations

seed = seeds[0]
print("clean :", seed.metadata["cited_answer"])
print("attack:", relocate_citations(seed))   # same prose, citations deranged
```
