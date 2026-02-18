# AIFS Input Preparation Implementation

This document summarizes the current AIFS input preparation path implemented in `multimodal_aifs/utils/zarr_data_loader.py`.

## Current Approach

When an AIFS `runner` is provided, `ZarrClimateLoader.to_aifs_tensor(...)` uses the runner's native preparation flow:

1. Extract physics variables from the loaded Zarr sample.
2. Build `input_state = {"date": ..., "fields": ...}`.
3. Initialize runner forcing inputs (if needed).
4. Call `runner.prepare_input_tensor(input_state)`.
5. Convert/reshape output to AIFS-compatible tensor layout used by the project pipeline.

## Variable Count Convention

- Physics variables are provided from dataset fields.
- Additional forcing variables are computed internally by the runner.
- Constants in `multimodal_aifs/constants.py` reflect this split:
  - `AIFS_PHYSICS_VARIABLES`
  - `AIFS_INPUT_VARIABLES`

## Why This Is the Active Path

- Keeps variable ordering consistent with the AIFS runtime
- Avoids duplicate forcing computation logic in project code
- Preserves compatibility with runner-driven input semantics

## Validation Coverage

This behavior is exercised by:

- `multimodal_aifs/tests/test_input_preparation.py`
- Zarr integration tests under `multimodal_aifs/tests/integration/zarr/`
