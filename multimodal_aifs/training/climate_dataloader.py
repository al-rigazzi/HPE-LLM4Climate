# Copyright 2025 Hewlett Packard Enterprise Development LP
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Climate-Text DataLoader for multimodal training.

This module provides a comprehensive DataLoader that generates training samples
on-the-fly by:
1. Loading climate data from Zarr files
2. Selecting random geographic locations
3. Computing climate statistics
4. Generating prompts
5. Getting LLM responses
6. Preparing data for AIFS tokenizer
"""

from dataclasses import dataclass
from pathlib import Path

import torch
import zarr
from torch.utils.data import IterableDataset

from ..constants import AIFS_GRID_POINTS, ALL_AIFS_VARIABLES
from .location_masks import Location, LocationMaskGenerator
from .prompt_generator import ClimatePromptGenerator
from .statistics_computer import ClimateStatisticsComputer


@dataclass
class TrainingSample:
    """Complete training sample with all necessary components."""

    # Input data
    climate_tensor_t1: torch.Tensor  # First timestep [grid_points, n_vars]
    climate_tensor_t2: torch.Tensor  # Second timestep [grid_points, n_vars]
    aifs_input_tensor: torch.Tensor  # Formatted for AIFS [2, 1, grid_points, n_vars]

    # Location and mask
    location: Location
    mask: torch.Tensor  # Boolean mask [grid_points]

    # Statistics and prompt
    statistics_table: str
    prompt: str
    llm_response: str

    # Tokenized text (for training)
    prompt_tokens: dict[str, torch.Tensor]
    response_tokens: dict[str, torch.Tensor]

    # Metadata
    timestep_1_index: int
    timestep_2_index: int
    zarr_file_path: str


class ClimateTextDataLoader(IterableDataset):
    """
    DataLoader for on-the-fly climate-text sample generation.

    This class is designed for distributed training with FSDP/ZeRO and generates
    samples dynamically to avoid storing large datasets.

    Pipeline:
    1. Load two consecutive timesteps from Zarr
    2. Select random geographic location
    3. Apply location mask
    4. Compute statistics (min, max, mean, grad, rot, div)
    5. Format statistics table
    6. Generate prompt
    7. Get Mistral response
    8. Prepare tensors for AIFS tokenizer
    """

    def __init__(
        self,
        zarr_paths: list[str] | str,
        mistral_model_name: str = "mistralai/Ministral-8B-Instruct-2410",
        batch_size: int = 1,
        samples_per_epoch: int = 1000,
        cache_mistral_model: bool = True,
        device: str = "cpu",
        seed: int | None = None,
        max_prompt_length: int = 2048,
        max_response_length: int = 512,
    ):
        """
        Initialize the climate-text dataloader.

        Args:
            zarr_paths: Path(s) to Zarr files containing climate data
            mistral_model_name: HuggingFace model name for Mistral
            batch_size: Batch size for training
            samples_per_epoch: Number of samples to generate per epoch
            cache_mistral_model: Whether to keep Mistral loaded in memory
            device: Device for Mistral inference
            seed: Random seed for reproducibility
            max_prompt_length: Maximum prompt token length
            max_response_length: Maximum response token length
        """
        super().__init__()

        self.batch_size = batch_size
        self.samples_per_epoch = samples_per_epoch
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length
        self.device = device
        self.cache_mistral_model = cache_mistral_model

        # Initialize Zarr paths
        if isinstance(zarr_paths, str):
            zarr_paths = [zarr_paths]
        self.zarr_paths = [Path(p) for p in zarr_paths]

        # Validate Zarr files exist
        for path in self.zarr_paths:
            if not path.exists():
                raise FileNotFoundError(f"Zarr file not found: {path}")

        # Initialize components
        print("Initializing climate-text dataloader components...")

        # Load Zarr metadata first to determine grid size
        self._load_zarr_metadata()

        # Detect actual grid size from the first available dataset
        actual_grid_points = self._detect_grid_size()

        self.location_generator = LocationMaskGenerator(grid_points=actual_grid_points, seed=seed)
        self.statistics_computer = ClimateStatisticsComputer()
        self.prompt_generator = ClimatePromptGenerator()  # Uses default templates directory

        # Initialize tokenizer and model for Mistral
        print(f"Loading tokenizer: {mistral_model_name}")
        from transformers import AutoTokenizer

        self.mistral_tokenizer = AutoTokenizer.from_pretrained(mistral_model_name)

        # Ensure tokenizer has pad token (required for padding)
        if self.mistral_tokenizer.pad_token is None:
            self.mistral_tokenizer.pad_token = self.mistral_tokenizer.eos_token
            self.mistral_tokenizer.pad_token_id = self.mistral_tokenizer.eos_token_id

        print(f"Loading Mistral model: {mistral_model_name}")
        self._initialize_mistral_model(mistral_model_name)

        print("DataLoader initialized:")
        print(f"  - Zarr files: {len(self.zarr_paths)}")
        print(f"  - Samples per epoch: {samples_per_epoch}")
        print(f"  - Device: {device}")

    def _initialize_mistral_model(self, model_name: str) -> None:
        """Initialize Mistral model for inference."""
        if self.mistral_tokenizer.pad_token is None:
            self.mistral_tokenizer.pad_token = self.mistral_tokenizer.eos_token

        if self.cache_mistral_model:
            from transformers import AutoModelForCausalLM

            print("  Loading model in float16 for memory efficiency...")
            self.mistral_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
            self.mistral_model.eval()

            # Update device to match where the model was actually loaded
            if hasattr(self.mistral_model, "device"):
                self.device = self.mistral_model.device
            elif hasattr(self.mistral_model, "hf_device_map"):
                # Get the first device from the device map
                first_device = next(iter(self.mistral_model.hf_device_map.values()))
                self.device = torch.device(first_device)

            print(f"  ✓ Mistral model loaded in float16 mode on {self.device}")
        else:
            print("  ℹ Mistral will be loaded per-sample (memory efficient mode)")

    def _load_zarr_metadata(self) -> None:
        """Load metadata from Zarr files."""
        self.zarr_datasets = []
        self.total_timesteps = []

        for zarr_path in self.zarr_paths:
            try:
                ds = zarr.open(str(zarr_path), mode="r")
                self.zarr_datasets.append(ds)

                # Get number of timesteps
                if "time" in ds:
                    n_timesteps = ds["time"].shape[0]
                else:
                    # Infer from first variable
                    first_var = list(ds.keys())[0]
                    n_timesteps = ds[first_var].shape[0]

                self.total_timesteps.append(n_timesteps)
                print(f"  ✓ Loaded {zarr_path.name}: {n_timesteps} timesteps")

            except Exception as e:
                raise RuntimeError(f"Failed to load Zarr file {zarr_path}: {e}") from e

    def _detect_grid_size(self) -> int:
        """Detect the actual grid size from the loaded datasets."""
        if not self.zarr_datasets:
            return AIFS_GRID_POINTS  # Fallback to default

        # Check the first dataset for grid size
        ds = self.zarr_datasets[0]

        for var_name in ALL_AIFS_VARIABLES:
            if var_name in ds:
                var_shape = ds[var_name].shape
                if len(var_shape) == 1:
                    return var_shape[0]
                if len(var_shape) == 2:
                    return var_shape[1]  # [time, grid_points]

        # Fallback if no AIFS variables found
        return AIFS_GRID_POINTS

    def _load_climate_timesteps(
        self, zarr_idx: int, t1: int, t2: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Load two consecutive timesteps from Zarr file.

        Args:
            zarr_idx: Index of Zarr dataset to use
            t1: First timestep index
            t2: Second timestep index

        Returns:
            Tuple of (tensor_t1, tensor_t2) each [grid_points, n_vars]
        """
        ds = self.zarr_datasets[zarr_idx]

        # Detect actual grid size from the first available variable
        actual_grid_points = None
        for var_name in ALL_AIFS_VARIABLES:
            if var_name in ds:
                var_shape = ds[var_name].shape
                if len(var_shape) == 1:
                    actual_grid_points = var_shape[0]
                elif len(var_shape) == 2:
                    actual_grid_points = var_shape[1]  # [time, grid_points]
                break

        # Fallback to AIFS grid points if no variables found
        if actual_grid_points is None:
            actual_grid_points = AIFS_GRID_POINTS

        # Load all variables for both timesteps
        data_t1 = []
        data_t2 = []

        for var_name in ALL_AIFS_VARIABLES:
            if var_name not in ds:
                # Handle missing variables with zeros using actual grid size and correct device
                data_t1.append(
                    torch.zeros(actual_grid_points, dtype=torch.float32, device=self.device)
                )
                data_t2.append(
                    torch.zeros(actual_grid_points, dtype=torch.float32, device=self.device)
                )
                continue

            var_data = ds[var_name]

            # Handle different array shapes
            if len(var_data.shape) == 1:
                # Static field (no time dimension)
                field = torch.from_numpy(var_data[:]).float().to(self.device)
                data_t1.append(field)
                data_t2.append(field)
            else:
                # Time-varying field
                field_t1 = torch.from_numpy(var_data[t1, :]).float().to(self.device)
                field_t2 = torch.from_numpy(var_data[t2, :]).float().to(self.device)
                data_t1.append(field_t1)
                data_t2.append(field_t2)

        # Stack into tensors [grid_points, n_vars]
        tensor_t1 = torch.stack(data_t1, dim=1)
        tensor_t2 = torch.stack(data_t2, dim=1)

        return tensor_t1, tensor_t2

    def _create_aifs_input_tensor(
        self, tensor_t1: torch.Tensor, tensor_t2: torch.Tensor
    ) -> torch.Tensor:
        """
        Format tensors for AIFS tokenizer input.

        Args:
            tensor_t1: First timestep [grid_points, n_vars]
            tensor_t2: Second timestep [grid_points, n_vars]

        Returns:
            AIFS input tensor [batch=1, time=2, ensemble=1, grid_points, n_vars]
        """
        # Stack timesteps
        stacked = torch.stack([tensor_t1, tensor_t2], dim=0)  # [2, grid_points, n_vars]

        # Add batch and ensemble dimensions
        aifs_input = stacked.unsqueeze(0).unsqueeze(2)  # [1, 2, 1, grid_points, n_vars]

        return aifs_input

    def _generate_mistral_response(self, prompt: str) -> str:
        """
        Generate response from Mistral model.

        Args:
            prompt: Input prompt

        Returns:
            Generated text response
        """
        # Format prompt for Mistral
        formatted_prompt = self.prompt_generator.format_for_mistral(prompt)

        # Tokenize
        inputs = self.mistral_tokenizer(
            formatted_prompt,
            return_tensors="pt",
            max_length=self.max_prompt_length,
            truncation=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self.mistral_model.generate(
                **inputs,
                max_new_tokens=self.max_response_length,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
            )

        # Decode response (remove prompt)
        full_text = self.mistral_tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = full_text[len(formatted_prompt) :].strip()

        return response

    def _generate_sample(self, sample_idx: int) -> TrainingSample:
        """
        Generate a single training sample.

        This is the main pipeline that orchestrates all steps.

        Args:
            sample_idx: Sample index (for seed variation)

        Returns:
            Complete TrainingSample object
        """
        # Step 1: Select random Zarr file and timesteps
        zarr_idx = sample_idx % len(self.zarr_datasets)
        n_timesteps = self.total_timesteps[zarr_idx]

        if n_timesteps < 2:
            raise ValueError(
                f"Zarr file {self.zarr_paths[zarr_idx]} has only {n_timesteps} timesteps, "
                "but need at least 2 consecutive timesteps for training"
            )

        max_t = n_timesteps - 1  # Maximum valid starting timestep
        t1 = (sample_idx * 17) % max_t if max_t > 0 else 0
        t2 = t1 + 1

        # Step 2: Load climate data
        tensor_t1, tensor_t2 = self._load_climate_timesteps(zarr_idx, t1, t2)

        # Step 3: Select random location
        location = self.location_generator.get_random_location()
        mask = location.mask

        # Step 4: Compute statistics on masked region
        # Use average of two timesteps for statistics
        avg_tensor = (tensor_t1 + tensor_t2) / 2.0
        statistics = self.statistics_computer.compute_statistics(
            avg_tensor, mask, variable_names=ALL_AIFS_VARIABLES
        )

        # Step 5: Format statistics table
        statistics_table = self.statistics_computer.format_statistics_table(statistics, max_vars=20)

        # Step 6: Generate prompt
        prompt = self.prompt_generator.generate_multimodal_training_prompt(
            location, statistics, statistics_table
        )

        # Step 7: Get Mistral response
        llm_response = self._generate_mistral_response(prompt)

        # Step 8: Tokenize prompt and response
        prompt_tokens = self.mistral_tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.max_prompt_length,
            truncation=True,
            padding="max_length",
        )

        response_tokens = self.mistral_tokenizer(
            llm_response,
            return_tensors="pt",
            max_length=self.max_response_length,
            truncation=True,
            padding="max_length",
        )

        # Step 9: Create AIFS input tensor
        aifs_input = self._create_aifs_input_tensor(tensor_t1, tensor_t2)

        # Create training sample
        sample = TrainingSample(
            climate_tensor_t1=tensor_t1,
            climate_tensor_t2=tensor_t2,
            aifs_input_tensor=aifs_input,
            location=location,
            mask=mask,
            statistics_table=statistics_table,
            prompt=prompt,
            llm_response=llm_response,
            prompt_tokens=prompt_tokens,
            response_tokens=response_tokens,
            timestep_1_index=t1,
            timestep_2_index=t2,
            zarr_file_path=str(self.zarr_paths[zarr_idx]),
        )

        return sample

    def __iter__(self):
        """Iterate over samples."""
        for i in range(self.samples_per_epoch):
            try:
                sample = self._generate_sample(i)
                yield sample
            except Exception as e:
                print(f"⚠ Error generating sample {i}: {e}")
                continue

    def __getitem__(self, index):
        """Not used for IterableDataset, but required by abstract base class."""
        raise NotImplementedError(
            "ClimateTextDataLoader is an IterableDataset. Use __iter__ instead."
        )

    def __len__(self) -> int:
        """Return number of samples per epoch."""
        return self.samples_per_epoch

    def get_sample_preview(
        self, num_samples: int = 1, max_response_length: int | None = None
    ) -> list[TrainingSample]:
        """
        Generate preview samples for inspection.

        Args:
            num_samples: Number of samples to preview
            max_response_length: Override max response length for preview (None uses default)

        Returns:
            List of training samples
        """
        # Temporarily override max_response_length if specified
        original_max_response = self.max_response_length
        if max_response_length is not None:
            self.max_response_length = max_response_length

        try:
            samples = []
            for i in range(num_samples):
                sample = self._generate_sample(i)
                samples.append(sample)
            return samples
        finally:
            # Restore original value
            self.max_response_length = original_max_response

    def print_sample_info(self, sample: TrainingSample) -> None:
        """
        Print detailed information about a sample.

        Args:
            sample: Training sample to inspect
        """
        print("\n" + "=" * 80)
        print("TRAINING SAMPLE PREVIEW")
        print("=" * 80)

        print("\n📍 Location:")
        print(f"  Name: {sample.location.name}")
        print(f"  Type: {sample.location.location_type}")
        print(
            f"  Coordinates: {sample.location.center_lat:.2f}°N, {sample.location.center_lon:.2f}°E"
        )
        print(f"  Masked grid points: {sample.mask.sum().item()} / {len(sample.mask)}")

        print("\n🌍 Climate Data:")
        print(f"  Timestep 1: {sample.timestep_1_index}")
        print(f"  Timestep 2: {sample.timestep_2_index}")
        print(f"  Tensor shape (t1): {sample.climate_tensor_t1.shape}")
        print(f"  Tensor shape (t2): {sample.climate_tensor_t2.shape}")
        print(f"  AIFS input shape: {sample.aifs_input_tensor.shape}")
        print(f"  Data source: {Path(sample.zarr_file_path).name}")

        print("\n📊 Statistics Table:")
        print(sample.statistics_table)

        print(f"\n💬 Prompt ({len(sample.prompt)} chars):")
        print("-" * 80)
        print(sample.prompt[:500] + "..." if len(sample.prompt) > 500 else sample.prompt)

        print(f"\n🤖 LLM Response ({len(sample.llm_response)} chars):")
        print("-" * 80)
        print(
            sample.llm_response[:500] + "..."
            if len(sample.llm_response) > 500
            else sample.llm_response
        )

        print("\n🔢 Tokenization:")
        print(f"  Prompt tokens: {sample.prompt_tokens['input_ids'].shape}")
        print(f"  Response tokens: {sample.response_tokens['input_ids'].shape}")

        print("\n" + "=" * 80)
