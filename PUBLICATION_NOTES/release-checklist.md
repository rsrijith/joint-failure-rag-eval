# Release checklist — the steps only the owner can run

**Rewritten 2026-08-01.** The earlier version was written before the repo was
published and had gone stale in ways that mattered: it listed already-completed
work as blocking, told you to install `jfre` from TestPyPI after the distribution
had been renamed to `fidecite`, and — worst — carried a `git merge --ff-only
prep/publication-readiness && git push origin main` command that **cannot work and
whose obvious recovery destroys the public history**. That command is gone. See
"Retired steps" at the foot for what was removed and why, so nothing here gets
re-added from an old copy.

Target date for the remaining steps: **2026-08-15**, timed to the GroundLM
decision. See "Does GroundLM actually gate this?" below — the honest answer is no.

---

## Already done — do not redo

| Step | Status |
|---|---|
| LICENSE copyright holder named (was "The Authors") | DONE `eb235ec`, both repos |
| Distribution name decided and applied | DONE `bb4c344` — `fidecite` on PyPI, `import jfre` |
| PyPI name availability checked | DONE 2026-08-01 — 404 for both `fidecite` and `jfre`. **Not a reservation. Re-check at step 1.** |
| Public `main` carries the working package | DONE — PR #1 merged `7252375` |
| `git+https://` install verified against the public URL | DONE — clean venv, quickstart output matched |
| First-contact example repaired | DONE `eb235ec` — was printing 0% clean-pass, reading as though the fix destroys your judge |

---

## Does GroundLM actually gate this?

**No.** `CITATION.cff` deliberately claims no venue and carries
`notes: "Citation block finalized at paper acceptance / public preprint."`, and
`date-released` is commented out. So nothing in the release has to be redone
depending on the decision.

What the decision changes is only what you can *add*:

- **If GroundLM accepts** — put the venue into `CITATION.cff`'s
  `preferred-citation` and mention the talk in the GitHub release body.
- **If it does not** — release exactly as written here. Say nothing about a venue.

So if the decision slips past mid-August, **release anyway**. Do not hold the
package for it.

---

## The run, in order. The order matters twice.

Zenodo must be enabled **before** the GitHub release or no DOI is minted, and the
README must be finalized **before** the tag so the tag points at what people
actually install.

### 1. Pre-flight

```bash
cd ~/Documents/EB1/GitHub/joint-failure-rag-eval
git checkout main && git pull
git status --porcelain          # must be empty
curl -s -o /dev/null -w "%{http_code}\n" https://pypi.org/pypi/fidecite/json   # want 404
```

A `200` means somebody took the name since 2026-08-01. Stop and re-plan if so.

### 2. Finalize the README install block

The Install section and the top-of-file install line are written in the
pre-publication `git+https://` form, with a `POST-RELEASE` HTML comment stating
exactly what to delete. Apply it now, so the tag captures the published form:

- top of README: `pip install fidecite`
- Install section: the three plain `pip install fidecite` / `"fidecite[judges]"` /
  `"fidecite[data]"` lines
- delete the `POST-RELEASE` comment itself
- same in `QUICKSTART.md`

```bash
git commit -am "README/QUICKSTART: switch to the published pip install fidecite form"
git push origin main
```

### 3. Enable Zenodo — BEFORE the release, not after

1. Sign in at https://zenodo.org/account/settings/github/ with GitHub.
2. Toggle **on** for `rsrijith/joint-failure-rag-eval`.

Zenodo only captures releases created *after* the toggle is on. Miss this and the
fix is cutting a throwaway `v0.1.1` just to trigger capture.

### 4. Build clean and rehearse on TestPyPI

```bash
rm -rf dist build *.egg-info
python -m build
twine check --strict dist/*     # expect PASSED on both artifacts
```

Artifacts are now `fidecite-0.1.0-py3-none-any.whl` and `fidecite-0.1.0.tar.gz`.

TestPyPI is the only rehearsal you get. Note the install name is `fidecite` and
the import is `jfre` — that split is the thing being rehearsed:

```bash
twine upload --repository testpypi dist/*
python3 -m venv /tmp/tpypi && /tmp/tpypi/bin/python -m pip install \
  --index-url https://test.pypi.org/simple/ fidecite
cd /tmp && /tmp/tpypi/bin/python -c "from jfre import make_seed, audit_judge; print('ok')"
```

**Run that last line from `/tmp`, not the repo.** From the repo root `import jfre`
silently reads the source tree instead of site-packages and the check passes
regardless — this exact false pass happened during the 2026-07-31 integration.

### 5. Tag

```bash
git tag -a v0.1.0 -F PUBLICATION_NOTES/tag-message-v0.1.0.txt
git push origin v0.1.0
```

### 6. Publish to PyPI — irreversible

Use a project-scoped API token or Trusted Publishing so no token sits on disk.

```bash
twine upload dist/*
```

Then verify as a stranger, again from outside the repo:

```bash
python3 -m venv /tmp/real && /tmp/real/bin/python -m pip install fidecite
cd /tmp && /tmp/real/bin/python -c "import jfre; print(jfre.__version__)"
```

### 7. GitHub release — this mints the DOI

Create it against tag `v0.1.0`. Body is
`PUBLICATION_NOTES/release-body-v0.1.0.md` — **read its trailing HTML comment and
delete it before pasting.** Attach both `dist/fidecite-0.1.0-py3-none-any.whl` and
`dist/fidecite-0.1.0.tar.gz`. If GroundLM accepted, add the venue line here.

### 8. Finish `CITATION.cff`, then push

- `doi:` — the **concept** DOI from Zenodo (version-independent), not the
  version-specific one.
- `date-released:` — uncomment, set to the tag date.
- `orcid:` — only if you have registered one. **A fabricated ORCID resolves to a
  stranger.** Leave the placeholders commented otherwise.
- `preferred-citation:` — replace the stub at paper acceptance.

Put the DOI in the README too, then commit and push.

### 9. Register `jfre` as an alias distribution

Only after `fidecite` is live. A stub distribution named `jfre` whose sole
dependency is `fidecite`, so anyone who guesses the install name from an
`import jfre` snippet still lands correctly. Mark its description as an alias so
it does not read as a squat. Rationale in `dist-name.md`.

---

## Separate track: the leaderboard

**Its real prerequisite is the Hugging Face dataset, not any packaging step.**
`rsrijith/fidecite` is a placeholder in `../fidecite-leaderboard/app.py` and
`scripts/submit.py` and 404s today, so a stranger can only submit with
`--seeds my_seeds.jsonl` against their own data, and those rows are not comparable
to the study's. `submit.py` now exits with that guidance rather than an obscure hub
error.

**The dataset must ship with `dataset/NOTICES.md`.** HotpotQA-derived rows are
CC-BY-SA 4.0 and the share-alike clause propagates, so they cannot be bundled into
this MIT repo. That is a licence conflict, not an attribution lapse.

```bash
cd ../fidecite-leaderboard
git remote add origin https://github.com/rsrijith/fidecite-leaderboard.git
git push -u origin main
```

Before pushing, re-confirm `reference_judges.csv`'s per-judge FNRs against the
camera-ready table.

## Separate track: good first issues

`.github/ISSUE_DRAFTS/` holds three drafts, unfiled. Paste each into a new issue,
add the `good first issue` label, then either delete the directory or keep it as
the backlog.

---

## Known issues, deliberately left

- **`jfre/judges/glm_cerebras_judge.py` ships in the wheel** though commit
  `e04cf4f` dropped GLM from the paper's seven-judge set. **Decision 2026-07-31:
  keep it.** `scripts/smoke_three_judges.py` imports it, so deleting it breaks a
  research script. It is unimported dead weight in the distribution, which is
  cheaper than a broken script. Revisit only if the wheel size matters.
- **`jfre/judges/` uses a different interface** from the public `Judge` callable:
  `score(seed, answer_to_judge, operator) -> JudgeVerdict`. Not a defect;
  `CONTRIBUTING.md` explains which shape an adapter should use. A thin conversion
  helper would remove the confusion.
- **Anything needing API keys or the HF dataset is unverified**: the seven
  reference judges, the seed loaders, and the `load_dataset` path through the
  leaderboard. The Gradio app has never been launched.

---

## Retired steps — removed 2026-08-01, do not re-add

- **"Decide the copyright line."** Done in `eb235ec`.
- **"Decide the distribution name first."** Done in `bb4c344`.
- **"Blocker: the public default branch is 6 commits behind," with
  `git merge --ff-only prep/publication-readiness && git push origin main`.**
  Removed as **dangerous**. The local branches were an *orphan* history sharing no
  commit with `origin/main`, so that merge refuses outright, the push is rejected,
  and forcing past the rejection replaces a 123-file public research repo with a
  56-file packaging slice. Resolved instead by branching *from* `origin/main` and
  adding the packaging layer on top: PR #1, merged as `7252375`, nothing deleted.
- **"PyPI name availability — NOT verified."** Checked 2026-08-01.
- **"The `git+https://` install — NOT verified."** Verified against the public URL
  after PR #1 merged.
- **"`examples/audit_your_own_judge.py` prints a 0% clean-pass rate."** Fixed in
  `eb235ec`.
