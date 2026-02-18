# AIFS SimpleRunner Analysis

This note documents the behavior relied upon by the current loader integration.

## Observed Runner Semantics

- `SimpleRunner.prepare_input_tensor(input_state)` prepares tensors using runner-native feature ordering.
- Forcing features are injected through runner-managed forcing inputs.
- The resulting array is used as the authoritative AIFS-formatted source before conversion to project tensor layout.

## Integration Point

`multimodal_aifs/utils/zarr_data_loader.py` uses this sequence when `runner` is available:

1. Build input state from Zarr data and timestamp.
2. Ensure forcing inputs are initialized on the runner.
3. Call `prepare_input_tensor(...)`.
4. Convert to torch tensor and reshape for downstream AIFS/fusion modules.

## Practical Implications

- Keep variable list/ordering definitions in sync with runner expectations.
- Prefer runner-native preparation over local reimplementation of forcing logic.
- Use test coverage in `multimodal_aifs/tests/test_input_preparation.py` and zarr integration tests to guard regressions.
