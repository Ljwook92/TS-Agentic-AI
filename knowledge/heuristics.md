# Literature-Derived Heuristics

These heuristics are a first-pass synthesis from the papers under `References/papers/`.
They are meant to guide experiment selection, not replace direct metric validation.

## Model Choice
- If an AF/BA spatial baseline is weak, escalate to a spatiotemporal model before extending the search elsewhere.
- If a spatiotemporal model still underperforms, first try a stronger attention variant or larger hidden size before increasing sequence length.
- For daily progression mapping, architectures that explicitly model temporal dependence are favored over pure framewise segmentation.

## Temporal Context
- Short temporal windows can miss slow wildfire expansion and cloud-recovery patterns.
- Increasing `ts_length` is most justified after a reasonable spatiotemporal baseline already exists.
- When temporal context is extended, compare against the previous window length directly instead of mixing multiple changes at once.

## Feature Selection
- VIIRS-based progression mapping is a strong default for AF/BA work.
- Auxiliary variables should be treated as ablation candidates rather than always-on truths.
- For prediction tasks, compare the full feature stack against reduced variants that remove static or highly engineered auxiliary groups.
- Land-cover and static environmental features are plausible helpers for generalization, but they should be validated against leaner subsets.

## Evaluation
- Partial test coverage should not be interpreted as a final benchmark; mark it explicitly as subset evaluation.
- When a bounded metric exceeds 1.0, treat it as suspicious and do not use it as the primary score for replanning.
- Prefer per-fire metrics and explicit comparison against the prior run when generating reasoning.

## Reporting
- Reports should explain not only which run is best, but what changed relative to the prior best run.
- When a retry succeeds after a debug failure, frame it as a recovery step rather than a scientific improvement.
