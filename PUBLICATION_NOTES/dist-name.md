# Recommendation: publish the distribution as `fidecite`, keep the import `jfre`

**Status: NOT APPLIED.** The change is staged as `dist-name.patch` in this
directory and nothing in the repo has been renamed. Claiming a name on PyPI is
one-shot and effectively irreversible (a name cannot be transferred to a
different project or freed on request), so the owner makes this call.

## Recommendation

Set the PyPI **distribution** name to `fidecite`. Leave the **import** package as
`jfre`. Users would run:

```bash
pip install fidecite
```

```python
import jfre
```

This split is ordinary Python packaging, not a wart: `pip install scikit-learn` /
`import sklearn`, `pip install pillow` / `import PIL`, `pip install
beautifulsoup4` / `import bs4`, `pip install opencv-python` / `import cv2`.

## Why

The brand is Fidecite and the README, the leaderboard, and the dataset all lead
with Fidecite. `jfre` is an acronym for a framing the project has already moved
past ("joint-failure RAG eval"), and the post-hoc pivot is documented in
`POST_HOC_PIVOT.md`. The name a stranger types after reading the paper or the
leaderboard is the brand, not the internal acronym. `fidecite` is also
searchable; `jfre` collides with nothing but means nothing.

The import package must not move, for a reason that outranks the branding: the
name `jfre` and the metric name `citation-faithfulness` /
`CitationFaithfulnessMetric` are already shipped inside third-party
integrations. Renaming the import package would break them silently.

## Tradeoffs

**For `fidecite`:**

- Matches the brand, the leaderboard repo name, and the intended dataset id.
- Searchable and pronounceable, so word of mouth actually resolves to the package.
- Costs nothing at the code level. The patch touches one line of `pyproject.toml`.
- Verified end to end: built as `fidecite-0.1.0`, installed into a clean venv,
  `import jfre` worked, and the README quickstart produced its documented output.
  `twine check --strict` passed on both artifacts.

**Against, or at least the friction:**

- Two names to learn. Someone who reads `import jfre` in a snippet and tries
  `pip install jfre` gets nothing unless `jfre` is also registered.
- Every existing install instruction in the wild would need updating. Today that
  is only this repo and the leaderboard, so the cost is near zero, but it rises
  the moment a paper or a blog post is public.
- PyPI availability of `fidecite` is **unverified**. No package index was queried
  during this pass. Check `https://pypi.org/project/fidecite/` before committing
  to it.

**Mitigation for the two-names problem.** Register `jfre` on PyPI too, as a stub
distribution whose only dependency is `fidecite`. Then `pip install jfre` works
for anyone who guesses from the import line. Do this only after `fidecite` is
published, and mark the stub's description as an alias so it does not look like a
squat.

## If you decline the rename

Do nothing. `pyproject.toml` already says `name = "jfre"` and the build works.
Delete this directory's `dist-name.patch` so a later session does not apply it by
reflex, and drop the `NOTE:` comment above `name` in `pyproject.toml`.

## Applying it

```bash
cd /path/to/joint-failure-rag-eval
git apply PUBLICATION_NOTES/dist-name.patch
```

Then update these, which the patch deliberately does not touch, so the rename is
reviewable as one line:

1. `README.md` — the two install blocks. The `POST-RELEASE` HTML comment in the
   Install section already spells out the `fidecite` form.
2. `QUICKSTART.md` — step 1's `POST-RELEASE` comment.
3. `CHANGELOG.md` — add a line under 0.1.0 noting the distribution name.
4. `../fidecite-leaderboard/CONTRIBUTING.md` and `scripts/submit.py` — the
   `pip install` line in the submission instructions.
5. Rebuild and re-run the clean-venv check in `PUBLICATION_NOTES/release-checklist.md`.
