# Reasoning Templates

These are phrasing patterns the agent can use when proposing or explaining a run.

## Upgrade Reasoning
- The spatial baseline was weak, so the agent escalated to a spatiotemporal architecture to capture temporal dependence.
- The current temporal model plateaued, so the agent switched attention variants before changing the sequence length.
- The temporal model plateaued again, so the agent increased model capacity before extending the sequence window.
- The model still failed to improve materially, so the agent lowered the learning rate to stabilize optimization.

## Temporal Reasoning
- The current sequence length may be too short to capture wildfire progression over multiple days, so the agent regenerated datasets with a longer temporal window.
- Increasing `ts_length` was deferred until after model-family and optimization changes were tested to avoid conflating causes.

## Feature-Set Reasoning
- The full feature stack may be redundant, so the agent scheduled a reduced-feature ablation to test whether the gain comes from a smaller subset.
- Static landscape variables may improve generalization, but their value should be demonstrated against a remote-sensing-only baseline.
- Engineered prediction inputs may help, but the agent should compare them against a remote-sensing-only or no-FirePred alternative before treating them as essential.

## Reporting Reasoning
- This run is a recovery step after a debug failure, so its outcome should not be interpreted as a clean scientific improvement.
- This run improved the primary metric relative to the prior measured run after changing {changed_params}.
- This run did not materially improve the primary metric, suggesting the current configuration has plateaued.
