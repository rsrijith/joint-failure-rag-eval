Title: Add an OpenAI LLM-as-judge adapter
Labels: good first issue, help wanted, adapter

## What

`jfre` ships adapters for Claude and Mistral as LLM-as-judge backends. There is no
OpenAI one, so anyone running GPT-4o or GPT-4.1 as their faithfulness judge has to
write the wrapper before they can audit it. That is the most common judge setup in
the wild and the most common reason someone bounces off the tool.

## Why it matters

The audit only needs a `(question, passages, answer) -> bool` callable, so this is
a thin wrapper, not new science. It removes the largest single "I'd have to write
glue first" barrier.

## What to do

1. Add `jfre/judges/openai_judge.py`.
2. Copy the structure of `jfre/judges/claude_judge.py`: read the key from the
   environment (`OPENAI_API_KEY`), `temperature=0`, JSON-only response, reuse
   `jfre.judges._llm_judge_prompt.render_prompt` and `parse_verdict` so the content
   prompt is identical to the other LLM judges. Comparability across judges
   depends on that prompt being byte-identical, so do not reword it.
3. Export a factory returning a public `Judge` callable:

   ```python
   def make_openai_judge(model: str = "gpt-4o", attribution_aware: bool = False) -> Judge:
       ...
   ```

   With `attribution_aware=True`, use `jfre.fix.render_attribution_prompt` instead.
   That gives a one-flag before/after demonstration of the fix on OpenAI models.
4. `import openai` **inside** the factory, not at module top level, and add
   `openai>=1.40` to the `judges` extra in `pyproject.toml`. Core
   `dependencies` must stay empty; see `CONTRIBUTING.md`.
5. Add `tests/test_openai_judge.py` using a stub client so the test needs no
   network and no key. `tests/test_fix.py` has the stub pattern.

## Definition of done

- `pytest` green with no API key set and no network.
- `import jfre` still works in an environment where `openai` is not installed.
- A short docstring example showing the adapter passed to `audit_judge`.

## Pointers

- `jfre/judges/claude_judge.py` — the closest model to copy.
- `jfre/judges/_llm_judge_prompt.py` — the shared content-support prompt.
- `jfre/fix.py` — `render_attribution_prompt`, `parse_verdict`.
- `FIX.md` — why the attribution-aware variant is worth wiring in.

Happy to review a draft. No need to ask before starting; comment that you are
taking it.
