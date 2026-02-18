# Examples

Current runnable examples in this directory:

- `zarr_aifs_multimodal_example.py`
- `aifs_mistral_example.py`
- `multimodal_timeseries_demo.py`

Notebook assets:

- `mistral_training_samples.ipynb`

## Run Examples

```bash
python multimodal_aifs/examples/zarr_aifs_multimodal_example.py
python multimodal_aifs/examples/aifs_mistral_example.py
python multimodal_aifs/examples/multimodal_timeseries_demo.py
```

## Notes

- Examples assume dependencies from root `requirements.txt` are installed.
- Some runs may require local access to real Zarr data and model weights.
- For lightweight development, set mock environment flags (e.g. `USE_MOCK_LLM=true`).
