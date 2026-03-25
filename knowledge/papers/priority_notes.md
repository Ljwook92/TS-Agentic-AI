# Paper Priorities

This file records the current user-guided literature priority.

## Critical
- `ts_satfire`
  - Treat this as the strongest source of truth for dataset semantics, split definitions, channel conventions, and benchmark framing.

## High Priority
- `using_deep_learning_for_spatial_and_temporal_analysis_of_wildfire_start_and_progression`
  - Use for progression-oriented spatial-temporal reasoning.
- `wildfire_progression_prediction_and_validation_using_satellite_data_and_remote_sensing_in_sonoma_california`
  - Use for prediction-oriented reasoning and validation framing.
- `wildfire_progression_time_series_mapping_with_interferometric_synthetic_aperture_radar_insar`
  - Use for progression time-series reasoning.
- `wildfire_s1s2_canada_a_large_scale_sentinel_1_2_wildfire_burned_area_mapping_dataset_based_on_the_20172019_wildfires_in_canada`
  - Use for burned-area-specific reasoning.

## Supporting
- `near_real_time_wildfire_progression_mapping_with_viirs_time_series_and_autoregressive_swinunetr`
- `satellite_based_fire_progression_mapping_a_comprehensive_assessment_for_large_fires_in_northern_california`

## How The Agent Should Use This
- When there is a conflict between generic wildfire heuristics and TS-SatFire-specific knowledge, prefer TS-SatFire.
- Prefer high-priority papers when proposing feature-set ablations or task-specific reasoning.
- Use supporting papers to widen the search space only after the critical and high-priority evidence has been respected.
