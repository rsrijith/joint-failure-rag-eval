# Release checklist — the steps only the owner can run

Everything in this file was deliberately **not** executed. No push, no upload, no
tag, no release, no remote on the leaderboard, nobody contacted. Run these
yourself, in order.

## 0a. Decide the copyright line

`LICENSE` in this repo says **"Copyright (c) 2026 The Authors"**, which is a
leftover from double-anonymous submission. Commit `0fbfe41` de-anonymized the
repo, and `pyproject.toml` and `CITATION.cff` both now name Srijith Ravikumar, so
the LICENSE is the last anonymous artifact. `../fidecite-leaderboard/LICENSE` was
copied verbatim and says the same thing, to keep the two consistent rather than
guess.

An MIT grant with no identified copyright holder is weak: a downstream user
cannot tell who granted the licence. Decide the line and change it in **both**
files before publishing. This was left alone on purpose because a copyright
attribution is the owner's call, not a cleanup task.

## 0. Decide the distribution name first

Read `dist-name.md`. If you want `fidecite` on PyPI, apply `dist-name.patch`
before step 3, because the first upload fixes the name permanently. Check
`https://pypi.org/project/fidecite/` for availability.

## 1. Blocker: the public default branch is 6 commits behind

`origin/main` still holds the pre-adoption research code: `jfre` 0.0.1, eight
hard dependencies, and **no `jfre/audit.py` and no `jfre/fix.py`**. So
`pip install "git+https://github.com/rsrijith/joint-failure-rag-eval.git"`
installs a package with no `audit_judge` and no `make_seed`, and the README
quickstart raises `ImportError` for anyone who tries it today.

The README's install line is written for the state *after* this push. Push first,
or the documented command is wrong.

```bash
cd joint-failure-rag-eval
git checkout main
git merge --ff-only prep/publication-readiness   # review the diff first
git push origin main
```

Then confirm the install path actually works, from a scratch venv outside the repo:

```bash
python3 -m venv /tmp/verify && /tmp/verify/bin/python -m pip install \
  "git+https://github.com/rsrijith/joint-failure-rag-eval.git"
cd /tmp && /tmp/verify/bin/python -c "from jfre import make_seed, audit_judge; print('ok')"
```

## 2. Tag

```bash
git tag -a v0.1.0 -F PUBLICATION_NOTES/tag-message-v0.1.0.txt
git push origin v0.1.0
```

## 3. Publish to PyPI

Build from a clean tree so no stale artifact ships:

```bash
rm -rf dist build *.egg-info
python -m build
twine check --strict dist/*
```

TestPyPI first. This is the only rehearsal you get:

```bash
twine upload --repository testpypi dist/*
python3 -m venv /tmp/tpypi && /tmp/tpypi/bin/python -m pip install \
  --index-url https://test.pypi.org/simple/ jfre
cd /tmp && /tmp/tpypi/bin/python -c "from jfre import audit_judge; print('ok')"
```

Then the real upload. Use a project-scoped API token, or configure Trusted
Publishing so no token lives on disk:

```bash
twine upload dist/*
```

After it lands, revert the README install block to the plain `pip install jfre`
form. The `POST-RELEASE` HTML comment in the Install section says exactly what to
delete.

## 4. GitHub release

Create the release against tag `v0.1.0`. Body is
`PUBLICATION_NOTES/release-body-v0.1.0.md` — **read its trailing HTML comment and
delete it** before pasting, and fix the install line to match what you actually
published. Attach `dist/jfre-0.1.0-py3-none-any.whl` and
`dist/jfre-0.1.0.tar.gz`.

## 5. Zenodo DOI

1. Sign in to Zenodo with GitHub, at https://zenodo.org/account/settings/github/.
2. Flip the toggle **on** for `rsrijith/joint-failure-rag-eval`. Zenodo only
   captures releases created *after* the toggle is on, so do this **before**
   step 4, or cut a `v0.1.1` afterwards to trigger capture.
3. Publishing the release mints the DOI. Zenodo reads `CITATION.cff` for the
   metadata, which is why it carries `type: software` and the abstract.
4. Put the concept DOI (the version-independent one) back into `CITATION.cff` as
   `doi:` and into the README, then push.

## 6. Finish `CITATION.cff`

- Uncomment `date-released` and set it to the tag date.
- Add `doi:` from step 5.
- Add `orcid:` in both author blocks once you have one, from
  https://orcid.org/register. The placeholders are commented out and marked; a
  fabricated ORCID resolves to a stranger, so leave them commented until real.
- Replace `preferred-citation` with the real paper block at acceptance. It is
  currently a stub carrying `notes:` to that effect.

## 7. Leaderboard

`../fidecite-leaderboard` has a local git repo with one commit and **no remote,
by design**. To publish:

```bash
cd ../fidecite-leaderboard
git remote add origin https://github.com/rsrijith/fidecite-leaderboard.git
git push -u origin main
```

For the Hugging Face Space, add the Space as a second remote and push there too.
The Space needs `app.py` and `requirements.txt` at the root, which is how the
directory is laid out.

Before pushing: `reference_judges.csv` carries the study's per-judge FNRs and the
README flags them for re-confirmation against the camera-ready table. The dataset
id `rsrijith/fidecite` in `app.py` and `scripts/submit.py` is a **placeholder**
and 404s until the Hugging Face dataset is published. Submissions cannot work
until that dataset exists.

## 8. Good first issues

`.github/ISSUE_DRAFTS/` holds three drafts. Nothing was filed. Paste each into a
new issue and add the `good first issue` label. Delete the drafts directory
afterwards, or keep it as the backlog.

## What was verified during the prep pass

- `pytest tests/` — 14 passed.
- `python -m build` — built `jfre-0.1.0-py3-none-any.whl` and `jfre-0.1.0.tar.gz`.
- `twine check --strict dist/*` — PASSED on both.
- Clean venv, wheel installed with no extras, `pip list` showed `jfre` and `pip`
  only, README quickstart run from outside the repo reproduced its documented
  output. Both `examples/` scripts ran there too.
- `cffconvert --validate` — valid against CFF schema 1.2.0.
- The `fidecite` rename dry-run built, installed clean, kept `import jfre`
  working, and passed `twine check --strict`. Reverted; nothing renamed.
- The leaderboard submission flow was walked end to end as a stranger would, from
  the clean venv with only the wheel installed: wrote a judge module and a
  three-seed JSONL, ran `scripts/submit.py`, got a valid entry, `scripts/validate.py`
  passed it, `scripts/build_table.py` put it on the board as row 8. Then removed
  and the board regenerated back to the 7 study rows. `validate.py` was also
  checked against deliberately broken files and caught every planted error.

## What was NOT verified

- **PyPI name availability**, for `jfre` or `fidecite`. No index was queried.
- **The `git+https://` install**, because it requires `origin/main` to be pushed
  first. See step 1.
- Anything needing API keys or the Hugging Face dataset: the seven reference
  judges and the seed loaders. The leaderboard flow was verified with
  `--seeds my_seeds.jsonl`; the published-set path through `load_dataset` could not
  be exercised because the dataset does not exist yet.
- The leaderboard Gradio app was not launched. `gradio` was not installed.

## Known issues left alone on purpose

- **`LICENSE` names no copyright holder** ("The Authors"). See section 0a. Same in
  the leaderboard's copy.
- **The Hugging Face dataset does not exist.** `rsrijith/fidecite` is a placeholder
  in `../fidecite-leaderboard/app.py` and `scripts/submit.py`. Until it is
  published, a stranger can only submit with `--seeds my_seeds.jsonl` against
  their own data, and those rows are not comparable to the study's. Publishing the
  dataset is the real prerequisite for advertising the leaderboard, more than any
  packaging step.
- **`examples/audit_your_own_judge.py` prints a 0% clean-pass rate** for the fixed
  judge, because `toy_attribution_llm` in that file is a stub that rejects
  everything once the citations move. The FNR half of the demonstration is
  correct; the clean-pass half reads as if the fix destroys the judge, which is
  the opposite of what `FIX.md` reports. Worth making the stub pass
  correctly-cited answers before anyone runs the example as their first contact
  with the tool.
- **`jfre/judges/` uses a different interface** from the public `Judge` callable:
  `score(seed, answer_to_judge, operator) -> JudgeVerdict`. Not a defect, but a
  contributor writing an adapter will notice, so `CONTRIBUTING.md` explains which
  shape to use. A thin conversion helper would remove the confusion.
- **`jfre/judges/glm_cerebras_judge.py` ships in the wheel** but commit `e04cf4f`
  dropped GLM from the paper and the reference set is seven judges. Dead code in a
  published distribution. Either delete it or note why it stays.
