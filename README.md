# TS_Agentic_AI

`TS_Agentic_AI` reorganizes the original `TS-SatFire` repository into an agent-oriented remote sensing workflow.

The project now separates:

- `legacy/`: preserved wildfire analysis code copied from TS-SatFire
- `agents/`: planner, executor, evaluator scaffolding
- `tools/`: wrappers that execute legacy scripts as structured tools
- `schemas/`: shared state and result schemas
- `configs/`: default tool parameters and thresholds
- `memory/`: persisted run history for iterative analysis
- `knowledge/`: reference material for retrieval and planning
- `References/`: source link notes used to build the knowledge base

## What Was Kept

The following assets were preserved because they remain useful for the analysis layer:

- wildfire dataset generation scripts
- spatial, temporal, and spatial-temporal training scripts
- legacy model implementations
- dataset processing modules
- ROI metadata and variable inventory

These assets now live under `legacy/`.

## What Was Set Aside

The following code was not removed, but was moved out of the top-level workflow because it is not part of the first agentic MVP:

- GOES clipping and coverage utilities
- bbox extraction helpers
- calibration helpers
- one-off comparison scripts

These now live under `legacy/support/`.

## Agentic Layout

The first MVP uses three roles:

- `Planner`: selects the next analysis action
- `Executor`: runs a legacy tool with structured inputs
- `Evaluator`: inspects outputs, metrics, and errors to recommend retry, fallback, or completion

## Quick Start

Run a planning cycle:

```bash
python main.py --task af
```

For the original HPC layout, the legacy code expects data under `/home/jlc3q/data/SatFire`.
You can override this without editing scripts:

```bash
export SATFIRE_ROOT=/your/path/to/SatFire
export TS_SATFIRE_CODE_ROOT=/your/path/to/TS-SatFire
```

Run a specific legacy tool through the executor:

```bash
python main.py --tool run_spatial_temp_model --task af --model swinunetr3d
```

The first version is intentionally conservative:

- deterministic tool registry
- persisted JSON state
- heuristic evaluator
- planner that can be replaced later with an LLM + RAG policy

## OpenAI Planner

If you want to enable the LLM planner, set:

```bash
export OPENAI_API_KEY=...
export OPENAI_MODEL=gpt-4.1-mini
```

Optional:

```bash
export OPENAI_BASE_URL=https://api.openai.com/v1
```

When `OPENAI_API_KEY` is present, the planner uses the OpenAI-compatible chat completion API.
If the API call fails, the system falls back to the deterministic rule planner.

## Next Steps

- replace the heuristic planner with an LLM planner
- add retrieval over `knowledge/`, experiment logs, and script documentation
- expose legacy script outputs as normalized metrics and artifacts
- add reflection and retry strategies based on evaluator feedback
