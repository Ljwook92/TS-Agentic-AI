# Legacy Assets

This directory preserves the original TS-SatFire implementation so the new agentic layer can call it without rewriting the domain logic first.

Contents:

- `scripts/`: main dataset and model entrypoints
- `satimg_dataset_processor/`: dataset preparation code
- `spatial_models/`: spatial models
- `temporal_models/`: temporal models
- `roi/`: wildfire ROI metadata
- `support/`: helper utilities and one-off scripts that are not part of the initial agent loop
