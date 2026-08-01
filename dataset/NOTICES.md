# Dataset NOTICES — per-source licensing

The citation-relocation set is derived from three upstream seed datasets. Each
released row carries the license of the upstream dataset it was derived from,
keyed by the `source` field. This file must accompany any release of the rows.

## Per-source terms

- **HotpotQA-derived rows** (`source = "hotpotqa"`)
  Upstream: HotpotQA (Yang et al., EMNLP 2018), https://hotpotqa.github.io/
  License: **CC-BY-SA 4.0**. The share-alike clause propagates: redistributions
  of these rows (and adaptations) must remain under CC-BY-SA 4.0 with attribution.

- **ExpertQA-derived rows** (`source = "expertqa"`)
  Upstream: ExpertQA (Malaviya et al., NAACL 2024),
  https://github.com/chaitanyamalaviya/ExpertQA
  License: **MIT** (source license preserved; attribution required).

- **PubMedQA-derived rows** (`source = "pubmedqa"`)
  Upstream: PubMedQA (Jin et al., EMNLP 2019), https://pubmedqa.github.io/
  License: **MIT** (source license preserved; attribution required).

## Derivative content

The `[N]` citation annotations and the relocation permutations are produced by
the `jfre` pipeline (MIT). They are layered on top of upstream passages and
answers; the underlying text retains its upstream license as above.

## Excluded source

RAGTruth (Niu et al., ACL 2024) was evaluated as a seed source and **excluded**:
its embedded MS MARCO and Yelp passages carry redistribution restrictions that
would propagate to any derivative release.

## ⚠️ DO NOT ship these rows inside this MIT repository

HotpotQA-derived rows are **CC-BY-SA 4.0 and the share-alike clause propagates**. A
redistribution of them, or of any adaptation, must itself remain CC-BY-SA 4.0 with
attribution. This repository and the `jfre` package are **MIT**. Bundling the HotpotQA-derived
rows into an MIT distribution would therefore be a licence conflict, not merely an attribution
lapse.

This is the reason the seed set is released through a separate dataset channel that can carry
per-row licence metadata, rather than committed here. If a future session is tempted to "just
ship the seeds so the leaderboard works offline", that is the trap. The options that actually
work:

1. Publish the full set on the dataset hub with this NOTICES file and per-row `source`, as
   planned. HotpotQA rows stay CC-BY-SA there, which is permitted.
2. Ship **only** the MIT-licensed subset (`source` in `expertqa`, `pubmedqa`) in-repo, clearly
   labelled as a subset, and state that scores from it are not comparable to the full table.

Recorded 2026-07-29 after the leaderboard scaffold was found to default to an unpublished
dataset id, which made shipping the rows locally look like the obvious fix.
