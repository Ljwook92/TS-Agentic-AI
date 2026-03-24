# TS-SatFire Paper Summary

## Citation

- title: TS-SatFire: A Multi-Task Satellite Image Time-Series Dataset for Wildfire Detection and Prediction
- authors: Yu Zhao, Sebastian Gerard, Yifang Ban
- year: 2024 preprint, later published in Scientific Data in 2025
- link: [arXiv PDF](https://arxiv.org/pdf/2412.11555)

## Core Claims

- A single dataset can support active fire detection, burned area mapping, and next-day wildfire progression prediction.
- Combining spectral, spatial, and temporal information is important for wildfire monitoring.
- Prediction quality benefits from auxiliary environmental data beyond satellite imagery alone.

## Useful Facts

- 179 wildfire events are included.
- The dataset covers the contiguous U.S. from January 2017 to October 2021 for the core train/val/pred/BA setup.
- The paper defines 27 channels for the progression prediction input stack.
- The benchmark compares temporal, spatial, and spatial-temporal model families.

## Operational Rules Derived From The Paper

| Evidence | Rule | Confidence |
| --- | --- | --- |
| Burned area and prediction preprocessing aggregate I4/I5 over time | Prefer spatiotemporal or BA-aware preprocessing when the goal depends on cumulative heat signatures | high |
| Prediction labels use only newly burned area between days | Do not evaluate prediction by raw next-day burned mask alone | high |
| Auxiliary data include weather, forecast, topography, and land cover | Route `pred` tasks to tools that consume the full auxiliary stack | high |
| The paper benchmarks temporal, spatial, and spatiotemporal models | Use a cheaper spatial baseline first when doing staged search, then escalate | medium |

## Limits and Caveats

- AF/BA labels are partly curated and partly derived from upstream products, so label quality varies by task and split.
- The active fire test set has a different geographic scope than BA/prediction test sets.
- The paper is sufficient for task and source definitions, but not enough to recover every implementation detail in the `FirePred` export pipeline.

## How The Agent Should Use This

- planning: choose tools and parameter defaults by task type
- evaluator: interpret lower prediction scores differently from AF/BA scores because prediction is harder by design
- retry logic: when prediction fails, consider data availability and auxiliary stack quality before only changing model parameters
