# joint-failure-rag-eval

Pre-experiment methodology and (forthcoming) artifacts for a study of *joint failure across automatic RAG faithfulness evaluators under single-edit adversarial perturbations*.

> See **[METHODOLOGY.md](METHODOLOGY.md)** for the locked design: 6 perturbation operators, 7 deployed judges, three statistical framings (joint-failure rate, attack universality, pairwise Cohen's κ), validity gates, and pre-committed go/no-go thresholds.

## What this is

Production RAG systems stack multiple faithfulness judges and aggregate their verdicts. This work measures whether those judges fail independently — or whether attacker-aligned single-edit perturbations induce *correlated* failure across the ensemble at non-trivial rates.

Target venue: **GroundLM 2026 @ EMNLP** (workshop, submission deadline 2026-06-29 AoE). Long paper, ACL Anthology archival, double-blind.

## Timeline

| Phase | Status |
|---|---|
| Methodology committed | ✓ 2026-05-26 |
| Pilot (50 examples × 2 operators × 7 judges) | targeting 2026-06-04 |
| Full experiment (500 × 6 × 7 = 21,000 judge inferences) | 2026-06-07 → 2026-06-13 |
| Human validity review (600 perturbations) | 2026-06-14 → 2026-06-20 |
| Paper draft + submission | 2026-06-21 → 2026-06-29 |

## What will land here

After the pilot go/no-go (Day 5):
- `operators/` — Python implementations of the 6 single-edit operators with their per-operator validation rules
- `judges/` — wrappers for RAGAS, HHEM-2.1-Open, MiniCheck-Flan-T5-L, AlignScore-large, FaithJudge, Claude 4 LLM-judge, Gemini 2.x LLM-judge
- `analysis/` — joint-failure rate, attack universality, pairwise Cohen's κ matrix computation
- `data/` — released perturbation set (post-acceptance)
- `NOTICES` — upstream attributions and per-subset license assignments

## License

- **Code:** MIT (see [LICENSE](LICENSE)).
- **Perturbation dataset (when released):** mixed per-subset. HotpotQA-derived rows inherit CC-BY-SA 4.0 (share-alike clause). ExpertQA-derived rows remain MIT. A per-row `NOTICES` file will accompany the released dataset.

Upstream datasets used as seeds:
- [HotpotQA](https://hotpotqa.github.io/) (Yang et al., EMNLP 2018) — CC-BY-SA 4.0
- [ExpertQA](https://github.com/chaitanyamalaviya/ExpertQA) (Malaviya et al., NAACL 2024) — MIT

RAGTruth (Niu et al., ACL 2024) was considered as a seed source and dropped after a license audit: the embedded MS MARCO and Yelp passages carry redistribution restrictions that propagate to derivative releases.
