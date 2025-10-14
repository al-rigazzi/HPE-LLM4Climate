# AIFS SimpleRunner Input Processing Analysis

**Date**: October 13, 2025
**Model**: AIFS-Single v1.1
**Source**: anemoi-inference package
**Status**: ✅ Solution Implemented

## Executive Summary

`SimpleRunner.run()` **ADDS forcing variables** to the input state before creating the input tensor. Users provide only the **physics variables** (94 in AIFS v1.1), and SimpleRunner computes and adds forcing variables internally.

**Solution**: Use `SimpleRunner.prepare_input_tensor()` directly to prepare our input tensors, letting it handle all forcing computation automatically.

## Our Implementation

### Working Solution in `zarr_data_loader.py`

```python
def to_aifs_tensor(self, data, runner=None, ...):
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

        # Let SimpleRunner prepare tensor (adds 9 forcings → 103 total)
        input_tensor_numpy = runner.prepare_input_tensor(input_state)
        # Returns: [timesteps, 103, grid_points]

        # Reshape to AIFS format
        tensor = torch.from_numpy(input_tensor_numpy).float()
        tensor = tensor.permute(0, 2, 1)
        tensor = tensor.unsqueeze(0).unsqueeze(2)
        # Returns: [batch=1, time, ensemble=1, grid_points, vars=103]
```

### Test Results

✅ **All tests passing with real data**

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 pytest test_cpu_mistral_zarr.py -xvs
# PASSED in 73.73s

SimpleRunner created tensor: (2, 103, 542080)
Climate tensor: torch.Size([1, 2, 1, 542080, 103])
AIFS encoder output: torch.Size([1, 1, 542080, 218])
Climate tokens: torch.Size([1, 2, 218])
✓ Test PASSED
```

## Call Stack Analysis

### 1. `SimpleRunner.run(input_state, lead_time)`

```python
def run(self, *, input_state: State, lead_time: Union[str, int, datetime.timedelta]):
    # Shallow copy to avoid modifying user's input
    input_state = input_state.copy()
    input_state["fields"] = input_state["fields"].copy()

    # CREATE FORCINGS (this is the key step!)
    self.constant_forcings_inputs = self.create_constant_forcings_inputs(input_state)
    self.dynamic_forcings_inputs = self.create_dynamic_forcings_inputs(input_state)
    self.boundary_forcings_inputs = self.create_boundary_forcings_inputs(input_state)

    # Prepare input tensor (includes adding forcings to fields dict)
    input_tensor = self.prepare_input_tensor(input_state)

    # Run forecast
    yield from self.forecast(lead_time, input_tensor, input_state)
```

### 2. `SimpleRunner.prepare_input_tensor(input_state)`

```python
def prepare_input_tensor(self, input_state: State, dtype=np.float32):
    # Add latitudes/longitudes if not present
    if "latitudes" not in input_state:
        input_state["latitudes"] = self.checkpoint.latitudes
    if "longitudes" not in input_state:
        input_state["longitudes"] = self.checkpoint.longitudes

    # KEY STEP: Add forcing variables to input_state["fields"]
    self.add_initial_forcings_to_input_state(input_state)

    # Validate input state
    input_state = self.validate_input_state(input_state)

    # Create tensor array
    input_tensor_numpy = np.full(
        shape=(
            self.checkpoint.multi_step_input,
            self.checkpoint.number_of_input_features,  # <-- This is the TOTAL including forcings
            self.checkpoint.number_of_grid_points,
        ),
        fill_value=np.nan,
        dtype=dtype,
    )

    # Map variables to tensor indices
    for var, field in input_fields.items():
        i = variable_to_input_tensor_index[var]
        input_tensor_numpy[:, i] = field

    return input_tensor_numpy
```

### 3. `Runner.add_initial_forcings_to_input_state(input_state)`

This is the **critical method** that adds forcing variables:

```python
def add_initial_forcings_to_input_state(self, input_state: State):
    date = input_state["date"]
    fields = input_state["fields"]

    # Get dates for lagged inputs (typically 2 timesteps)
    dates = [date + h for h in self.checkpoint.lagged]

    # Get forcing sources
    initial_constant_forcings = self.initial_constant_forcings_inputs(self.constant_forcings_inputs)
    initial_dynamic_forcings = self.initial_dynamic_forcings_inputs(self.dynamic_forcings_inputs)

    # ADD CONSTANT FORCINGS TO FIELDS DICT
    for source in initial_constant_forcings:
        arrays = source.load_forcings_array(dates, input_state)
        for name, forcing in zip(source.variables, arrays):
            fields[name] = forcing  # <-- ADDS to input fields!

    # ADD DYNAMIC FORCINGS TO FIELDS DICT
    for source in initial_dynamic_forcings:
        arrays = source.load_forcings_array(dates, input_state)
        for name, forcing in zip(source.variables, arrays):
            fields[name] = forcing  # <-- ADDS to input fields!
```

### 4. `ComputedForcings.load_forcings_array(dates, current_state)`

For computed forcings (trigonometric encodings, insolation):

```python
def load_forcings_array(self, dates: List[Date], current_state: State):
    # Use earthkit.data to compute forcing variables from coordinates and dates
    source = UnstructuredGridFieldList.from_values(
        latitudes=current_state["latitudes"],
        longitudes=current_state["longitudes"],
    )

    # This computes cos_latitude, sin_latitude, cos_julian_day, etc.
    ds = ekd.from_source("forcings", source, date=dates, param=self.variables)

    return forcing.to_numpy(dtype=np.float32, flatten=True).reshape(len(self.variables), len(dates), -1)
```

## Input Variable Structure for AIFS v1.1

### User-Provided Physics Variables (94 total)

#### Surface Variables (12):
- `10u`, `10v`, `2d`, `2t`, `msl`, `skt`, `sp`, `tcw`
- `lsm`, `z`, `slor`, `sdor`

#### Soil Variables (4):
- `stl1`, `stl2`, `swvl1`, `swvl2`

#### Pressure Level Variables (78):
6 parameters × 13 levels = 78 variables
- Parameters: `t`, `u`, `v`, `w`, `q`, `z`
- Levels: 1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50 hPa
- Naming: `{param}_{level}` (e.g., `t_1000`, `u_925`, ...)
- **Order**: By parameter then by level (descending pressure)

**Total User Input: 12 + 4 + 78 = 94 variables**

### Forcing Variables ADDED by SimpleRunner

These are **computed internally** and should **NOT** be in user input:

#### Trigonometric Spatial Encodings (4):
- `cos_latitude`, `sin_latitude`
- `cos_longitude`, `sin_longitude`

#### Trigonometric Temporal Encodings (4):
- `cos_julian_day`, `sin_julian_day`
- `cos_local_time`, `sin_local_time`

#### Solar/Constant Forcings:
- `insolation` - computed from date/time and solar geometry
- Constant forcing fields: `lsm`, `z`, `sdor`, `slor` (already in physics variables, marked as constant)

**Forcing variables added: 8 trigonometric + insolation = 9 dynamic computed forcings**

**Total Model Input: 94 physics + 9 forcings = 103 variables**

### Note on AIFS v1.1 Additional Variables

The AIFS v1.1 README mentions new **output** parameters (not inputs):
- `ssrd`, `strd` (radiation)
- `lcc`, `mcc`, `hcc`, `tcc` (cloud cover)
- `sf` (snowfall)
- `100u`, `100v` (100m winds) - these are **outputs** not inputs
- `rowe` (runoff)

## Implementation Status

✅ **COMPLETE** - All tests passing with real ECMWF data

### Files Modified

1. **`multimodal_aifs/utils/zarr_data_loader.py`**
   - Integrated `SimpleRunner.prepare_input_tensor()`
   - Extracts 94 physics variables
   - Creates input_state dict
   - Calls runner to prepare tensor with forcings

2. **`multimodal_aifs/constants.py`**
   - `AIFS_INPUT_VARIABLES = 103` (total after forcings)
   - `AIFS_PHYSICS_VARIABLES = 94` (user-provided only)
   - Updated validation logic

3. **Integration Tests**
   - `test_cpu_mistral_zarr.py` ✅
   - `test_real_mistral_cpu_full.py` ✅
   - `test_real_mistral_zarr.py` ✅
   - All updated to pass `runner` parameter

### Key Insight

**Don't reimplement - just use SimpleRunner!**

The notebook showed us the correct pattern:
```python
# From run_AIFS_v1.1.ipynb
input_state = dict(date=DATE, fields=fields)  # 94 variables only
for state in runner.run(input_state=input_state, lead_time=12):
    ...
```

We adapted this for our use case by calling `prepare_input_tensor()` directly instead of `run()`, since we only need the encoder, not the full forecast.

## Implications for Our Code

### ✅ Solution Implemented

We now use `SimpleRunner.prepare_input_tensor()` directly:

1. **Extract 94 physics variables** from our zarr dataset (use `ALL_AIFS_VARIABLES`)
2. **Create input_state dict** with date and fields
3. **Initialize forcing inputs** (`constant_forcings_inputs`, `dynamic_forcings_inputs`)
4. **Call `runner.prepare_input_tensor(input_state)`** - it adds 9 forcings → returns 103 total
5. **Reshape** from [timesteps, features, grid] to [batch, time, ensemble, grid, vars]

### What We DON'T Do Anymore

❌ Don't manually compute forcing variables with earthkit
❌ Don't include trigonometric encodings in our dataset
❌ Don't try to reimplement SimpleRunner's logic
❌ Don't pass forcing variables in input

### Result

- Input: 94 physics variables
- SimpleRunner adds: 9 forcing variables
- Output: 103 total variables in correct order
- Model receives: Exactly what it expects! ✅

## Conclusion

**SimpleRunner.prepare_input_tensor() is the complete solution.**

By examining the AIFS v1.1 notebook, we learned:
1. Users provide only 94 physics variables
2. SimpleRunner computes and adds forcings internally
3. The correct approach is to call `prepare_input_tensor()` and let it handle everything

This simple insight solved our variable count/order issues and enabled successful integration with real ECMWF data.
