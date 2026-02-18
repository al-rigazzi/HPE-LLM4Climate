# Data Notes

This package-level data directory is documentation-only in the current layout.

## Active Dataset Path

The active project dataset used by loaders/tests is at repository level:

- `data/real_ecmwf_latest.zarr`

## Loading Path in Code

Data ingestion is handled by:

- `multimodal_aifs/utils/zarr_data_loader.py`

## Test Behavior

- `USE_REAL_ZARR=true` uses real dataset flow
- `USE_REAL_ZARR=false` enables mock Zarr generation in test fixtures

## Recommendation

Keep this directory for package documentation and avoid storing large runtime datasets here.
