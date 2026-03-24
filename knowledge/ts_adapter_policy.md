# Time-Series Adapter Policy

This document defines how each model family should consume TS-SatFire inputs when the raw dataset contains a time dimension.

## Why This Policy Exists

TS-SatFire datasets often expose image tensors with shape:

- `[B, C, T, H, W]`

Different model families do not consume that shape in the same way. The adapter policy makes this explicit so shape handling is reproducible instead of ad hoc.

## Policy Names

### `spatial_framewise`

- Target tools: `run_spatial_model`
- Input expectation: 2D spatial models expect `[B, C, H, W]`
- Adapter rule:
  - if input is `[B, C, T, H, W]`
  - reshape to `[B*T, C, H, W]`
  - reshape labels from `[B, 2, T, H, W]` to `[B*T, 2, H, W]`
- Rationale:
  - the 2D model cannot reason over time directly
  - each time step is treated as an independent frame
  - this preserves all observed frames without forcing a single-frame heuristic

### `spatiotemporal_native`

- Target tools: `run_spatial_temp_model`, `run_spatial_temp_model_pred`
- Input expectation: `[B, C, T, H, W]`
- Adapter rule:
  - keep the time dimension
- Rationale:
  - 3D and spatiotemporal models are built to consume the temporal axis directly

### `temporal_sequence`

- Target tools: `run_seq_model`
- Input expectation:
  - temporal sequence representation for per-pixel or sequence-oriented models
- Adapter rule:
  - preserve temporal ordering
  - project or reshape into the sequence layout expected by the TensorFlow temporal data generator
- Rationale:
  - temporal models should see ordered time information instead of independent frames

## Current Adopted Default

- `run_spatial_model`: `spatial_framewise`
- `run_spatial_temp_model`: `spatiotemporal_native`
- `run_spatial_temp_model_pred`: `spatiotemporal_native`
- `run_seq_model`: `temporal_sequence`

## Known Alternatives

The spatial baseline could have used other policies, but these are not currently adopted:

- `last_frame_only`
- `center_frame_only`
- `temporal_mean_projection`

These alternatives may be useful for ablations, but they should not silently replace the default policy.

## Agent Notes

- If a shape mismatch occurs, first identify which adapter policy should apply.
- Do not patch shapes inline without updating this document.
- If a different policy is chosen for a paper experiment, record it in the experiment log.
