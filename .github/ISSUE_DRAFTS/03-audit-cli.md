Title: Add a `jfre audit` CLI that runs the audit over a JSONL file
Labels: good first issue, help wanted

## What

Right now auditing a judge means writing a Python script. A small CLI would let
someone point the tool at a file of their own cited answers and get the number:

```bash
jfre audit --seeds my_seeds.jsonl --judge mymodule:my_judge
jfre audit --seeds my_seeds.jsonl --judge mymodule:my_judge --json
```

Where `my_seeds.jsonl` is one object per line:

```json
{"seed_id": "s1", "question": "...", "passages": ["...", "..."], "cited_answer": "... [1] ... [2]"}
```

## Why it matters

It shortens the first run from "write a script, figure out `make_seed`" to one
command, and it makes the audit usable from a CI step. `--json` output means a
build can fail when a judge's attribution FNR regresses past a threshold, which is
the shape most teams want.

## What to do

1. `jfre/cli.py` using `argparse` from the standard library. **No new dependency**,
   in core or otherwise. No click, no typer. See `CONTRIBUTING.md`.
2. Register the entry point in `pyproject.toml`:

   ```toml
   [project.scripts]
   jfre = "jfre.cli:main"
   ```

3. Resolve `--judge mymodule:my_judge` with `importlib.import_module` plus
   `getattr`. Fail with a readable message when the module is missing, the
   attribute is missing, or the object is not callable. Do not let a traceback be
   the error message.
4. Build seeds with `jfre.make_seed`, run `jfre.audit_judge`, print
   `result.summary()`. Under `--json`, emit the `AuditResult` fields plus
   `false_negative_rate` and `clean_pass_rate` as a JSON object on stdout, with
   nothing else on stdout.
5. Exit codes: `0` on a completed audit, `2` on bad input or an unloadable judge.
   Optionally `--fail-over 0.5` to exit `1` when the FNR exceeds a threshold, which
   is what makes it useful in CI.
6. Tests in `tests/test_cli.py`. Write a temporary JSONL fixture, point `--judge`
   at a callable defined in the test module, assert on stdout and the exit code.
   Cover the `--json` shape and at least one error path.

## Definition of done

- `pytest` green, offline.
- `jfre audit --help` reads clearly to someone who has not read the README.
- Seeds with fewer than two distinct `[N]` markers are reported as skipped rather
  than crashing. `audit_judge` already counts them in `n_skipped`; surface it.
- A short CLI section added to `QUICKSTART.md`.

## Pointers

- `jfre/audit.py` — `make_seed`, `audit_judge`, and the `AuditResult` fields.
- `examples/audit_your_own_judge.py` — the flow the CLI is wrapping.
