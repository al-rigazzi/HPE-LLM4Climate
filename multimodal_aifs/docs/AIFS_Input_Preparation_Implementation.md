# AIFS Input Preparation Pipeline - Implementation Summary

**Date**: October 13, 2025
**Status**: ✅ Implemented and Working

## Problem

We were passing 94 variables directly to the AIFS model, but:
1. The model expected 218 features after preprocessing
2. We were missing forcing variables (trigonometric encodings, insolation)
3. Dimension mismatches caused model failures

**Root Cause**: We were not including forcing variables in our input. `SimpleRunner` **computes and adds** forcing variables internally (8 trigonometric + insolation = 9 additional variables, total 103).

## Solution

**Use `SimpleRunner.prepare_input_tensor()` directly** instead of reimplementing forcing computation.

### Implementation in `zarr_data_loader.py`

The `ZarrClimateLoader.to_aifs_tensor()` method now:

1. Extracts 94 physics variables from zarr dataset
2. Creates `input_state` dict with fields and date
3. Calls `runner.prepare_input_tensor(input_state)`
4. SimpleRunner internally:
   - Adds 8 trigonometric forcings (cos/sin lat/lon/time/day)
   - Adds insolation
   - Assembles all 103 variables in correct order
   - Returns numpy array [timesteps, features, grid_points]
5. Reshapes to AIFS format [batch, time, ensemble, grid, vars]

### Key Code

```python
# In ZarrClimateLoader.to_aifs_tensor()
if runner is not None and self.is_aifs_format:
    # Extract 94 physics variables
    fields = {}
    for var in ALL_AIFS_VARIABLES:
        fields[var] = data[var].values

    # Create input_state
    input_state = {
        "date": pd.Timestamp(data.time.values[0]).to_pydatetime(),
        "fields": fields,
    }

    # Initialize forcing inputs if needed
    if not hasattr(runner, 'constant_forcings_inputs'):
        runner.constant_forcings_inputs = runner.create_constant_forcings_inputs(input_state)
    if not hasattr(runner, 'dynamic_forcings_inputs'):
        runner.dynamic_forcings_inputs = runner.create_dynamic_forcings_inputs(input_state)

    # Let SimpleRunner prepare tensor (adds forcings)
    input_tensor_numpy = runner.prepare_input_tensor(input_state)
    # Returns: [timesteps, 103, grid_points]

    # Reshape to AIFS format
    tensor = torch.from_numpy(input_tensor_numpy).float()
    tensor = tensor.permute(0, 2, 1)  # [timesteps, grid_points, features]
    tensor = tensor.unsqueeze(0).unsqueeze(2)  # [batch, time, ensemble, grid, vars]
```

### Updated Constants

`multimodal_aifs/constants.py`:
- `AIFS_INPUT_VARIABLES = 103` - Total features after SimpleRunner adds forcings
- `AIFS_PHYSICS_VARIABLES = 94` - User-provided physics variables only
- `ALL_AIFS_VARIABLES`: Contains 94 physics variables (no forcing variables)
  - 12 surface variables
  - 4 soil variables
  - 78 pressure level variables

`multimodal_aifs/tests/test_input_preparation.py`:
- Demonstrates complete workflow
- Loads SimpleRunner to get variable order
- Computes forcings and prepares tensor
- Validates output

## How It Works

### The SimpleRunner Pipeline

```
User provides 94 physics variables
  ↓
ZarrClimateLoader.to_aifs_tensor(data, runner=runner)
  ↓
Extract fields dict with 94 variables
  ↓
Create input_state = {"date": date, "fields": fields}
  ↓
runner.prepare_input_tensor(input_state)
  ├── Adds 8 trigonometric forcings (cos/sin lat/lon/time/day)
  ├── Adds insolation
  └── Assembles in model's expected order
  ↓
Returns: numpy array [timesteps, 103, grid_points]
  ↓
Reshape to: [batch=1, time, ensemble=1, grid, vars=103]
  ↓
AIFS Model Input (Ready!)
```

### Why This Works

1. **SimpleRunner knows the model**: It has the checkpoint and knows exactly what variables and order the model expects
2. **Forcing computation is automatic**: No need to manually compute with earthkit - SimpleRunner does it
3. **Correct ordering guaranteed**: Variables are assembled in the exact order from `variable_to_input_tensor_index`
4. **Multi-timestep support**: Handles lagged inputs automatically

## Variable Breakdown

### Physics Variables (94) - User Provides

**Surface (12)**:
- `10u`, `10v`, `2d`, `2t`, `msl`, `skt`, `sp`, `tcw`
- `lsm`, `z`, `slor`, `sdor`

**Soil (4)**:
- `stl1`, `stl2`, `swvl1`, `swvl2`

**Pressure Levels (78)**:
- 6 parameters × 13 levels
- Parameters: `t`, `u`, `v`, `w`, `q`, `z`
- Levels: 1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50 hPa
- Naming: `{param}_{level}` (e.g., `t_1000`, `u_925`)

### Forcing Variables (9) - Computed by SimpleRunner

**Trigonometric Spatial (4)**:
- `cos_latitude`, `sin_latitude`
- `cos_longitude`, `sin_longitude`

**Trigonometric Temporal (4)**:
- `cos_julian_day`, `sin_julian_day`
- `cos_local_time`, `sin_local_time`

**Solar (1)**:
- `insolation`

**Total Model Input**: 94 physics + 9 forcings = **103 variables**

## Test Results

✅ **Test PASSED**: `test_lightweight_mistral_zarr`

```
SimpleRunner created tensor: (2, 103, 542080)
Format: [batch=1, time=2, ensemble=1, grid_points=542080, vars=103]
Input is in AIFS format [batch, time, ensemble, grid, vars]
Processing full tensor with AIFS encoder: torch.Size([1, 2, 1, 542080, 103])
AIFS encoder output shape: torch.Size([1, 1, 542080, 218])
Climate tokens: torch.Size([1, 2, 218])
✓ Test PASSED in 73.73s
```

Pipeline successfully:
1. ✅ Loaded 94 physics variables from zarr
2. ✅ Called SimpleRunner.prepare_input_tensor()
3. ✅ SimpleRunner added 9 forcing variables → 103 total
4. ✅ Tensor recognized as AIFS format
5. ✅ AIFS encoder processed successfully
6. ✅ Output climate tokens generated

## Integration with Existing Code

### Test Updates

All integration tests updated to pass `runner` to loader:

```python
# In test files
def test_something(aifs_mistral_model, aifs_model, zarr_dataset_path):
    loader = ZarrClimateLoader(zarr_dataset_path)
    climate_data = loader.load_time_range(None, None)

    # Get runner from fixture
    runner = aifs_model.get("runner") if not aifs_model.get("is_mock") else None

    # Pass runner to enable proper input preparation
    climate_tensor = loader.to_aifs_tensor(
        climate_data,
        batch_size=1,
        normalize=True,
        device=device,
        runner=runner,  # <-- Key parameter
    )
```

Updated test files:
- `test_cpu_mistral_zarr.py` ✅
- `test_real_mistral_cpu_full.py` ✅
- `test_real_mistral_zarr.py` ✅

## Benefits

1. **Correctness**: Tensors match exactly what AIFS expects (103 variables in correct order)
2. **Simplicity**: Use SimpleRunner's own logic instead of reimplementing
3. **Reliability**: No earthkit API issues - SimpleRunner handles everything
4. **Maintainability**: Automatically compatible with any AIFS model version
5. **Performance**: No duplicate forcing computation

## Key Learnings

1. **SimpleRunner.prepare_input_tensor() is the solution**: It handles all forcing computation and ordering
2. **Just pass 94 physics variables**: Don't try to include forcings manually
3. **Initialize forcing attributes**: Set `constant_forcings_inputs` and `dynamic_forcings_inputs` before calling `prepare_input_tensor()`
4. **Update AIFS_INPUT_VARIABLES to 103**: The constant represents total features after forcings added
5. **Check notebook workflows**: Real-world examples show the correct usage pattern

## Files Modified

**Modified**:
- `multimodal_aifs/utils/zarr_data_loader.py` - Integrated SimpleRunner.prepare_input_tensor()
- `multimodal_aifs/constants.py` - Updated AIFS_INPUT_VARIABLES from 94 to 103
- `multimodal_aifs/tests/integration/zarr/*.py` - Pass runner to loader
- `multimodal_aifs/tests/test_input_preparation.py` - Demonstrates SimpleRunner usage
- `multimodal_aifs/docs/AIFS_Input_Preparation_Implementation.md` - Updated documentation
- `multimodal_aifs/docs/AIFS_SimpleRunner_Analysis.md` - Analysis and solution documentation

**Note**: An initial attempt to manually reimplement forcing computation was abandoned in favor of using SimpleRunner's built-in `prepare_input_tensor()` method, which proved simpler and more reliable.
