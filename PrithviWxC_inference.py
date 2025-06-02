import random
from pathlib import Path

import yaml

import matplotlib.pyplot as plt
import numpy as np
import torch
from huggingface_hub import hf_hub_download, snapshot_download

from PrithviWxC.dataloaders.merra2 import Merra2Dataset
from PrithviWxC.dataloaders.merra2 import (
    input_scalers,
    output_scalers,
    static_input_scalers,
)

from PrithviWxC.model import PrithviWxC
from PrithviWxC.dataloaders.merra2 import preproc

torch.jit.enable_onednn_fusion(True)
if torch.cuda.is_available():
    print(f"Using device: {torch.cuda.get_device_name()}")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
torch.manual_seed(42)
np.random.seed(42)

# The model has approximately 2.3 billion parameters
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# The variables are comprised of surface variables, surface static variables, and variables at various vertical levels within the atmosphere.
surface_vars = [
    "EFLUX",
    "GWETROOT",
    "HFLUX",
    "LAI",
    "LWGAB",
    "LWGEM",
    "LWTUP",
    "PS",
    "QV2M",
    "SLP",
    "SWGNT",
    "SWTNT",
    "T2M",
    "TQI",
    "TQL",
    "TQV",
    "TS",
    "U10M",
    "V10M",
    "Z0M",
]
static_surface_vars = ["FRACI", "FRLAND", "FROCEAN", "PHIS"]
vertical_vars = ["CLOUD", "H", "OMEGA", "PL", "QI", "QL", "QV", "T", "U", "V"]
levels = [
    34.0,
    39.0,
    41.0,
    43.0,
    44.0,
    45.0,
    48.0,
    51.0,
    53.0,
    56.0,
    63.0,
    68.0,
    71.0,
    72.0,
]

# The MERRA-2 dataset includes data at longitudes of -180 and +180. This represents duplicate data, so we set a padding variable to remove it.
padding = {"level": [0, 0], "lat": [0, -1], "lon": [0, 0]}

# The input to the core model consists of these variables at two different times. 
# The time difference in hours between these samples is passed to the model and set in the input_time variable.
# The model's task is to predict the fixed set of variables at a target time, given the input data.
lead_times = [18]  # This variable can be change to change the task
input_times = [-6]  # This variable can be change to change the task

variable_names = surface_vars + [
    f'{var}_level_{level}' for var in vertical_vars for level in levels
]

# MERRA-2 data is available from 1980 to the present day, at 3-hour temporal resolution. 
time_range = ("2020-01-01T00:00:00", "2020-01-02T05:59:59")

surf_dir = Path("/data/merra-2")
#snapshot_download(
#    repo_id="ibm-nasa-geospatial/Prithvi-WxC-1.0-2300M",
#    allow_patterns="merra-2/MERRA2_sfc_2020010[1].nc",
#    local_dir="data",
#)

vert_dir = Path("/data/merra-2")
#snapshot_download(
#    repo_id="ibm-nasa-geospatial/Prithvi-WxC-1.0-2300M",
#    allow_patterns="merra-2/MERRA_pres_2020010[1].nc",
#    local_dir="data",
#)

# The PrithviWxC model was trained to calculate the output by producing a perturbation to the climatology at the target time (residual=climate option).
surf_clim_dir = Path("/data/climatology")
#snapshot_download(
#    repo_id="ibm-nasa-geospatial/Prithvi-WxC-1.0-2300M",
#    allow_patterns="climatology/climate_surface_doy00[1]*.nc",
#    local_dir="data",
#)

vert_clim_dir = Path("/data/climatology")
#snapshot_download(
#    repo_id="ibm-nasa-geospatial/Prithvi-WxC-1.0-2300M",
#    allow_patterns="climatology/climate_vertical_doy00[1]*.nc",
#    local_dir="data",
#)

# The position data is encoded in the model with two possible options, fourier or absolute.
positional_encoding = "fourier"

dataset = Merra2Dataset(
    time_range=time_range,
    lead_times=lead_times,
    input_times=input_times,
    data_path_surface=surf_dir,
    data_path_vertical=vert_dir,
    climatology_path_surface=surf_clim_dir,
    climatology_path_vertical=vert_clim_dir,
    surface_vars=surface_vars,
    static_surface_vars=static_surface_vars,
    vertical_vars=vertical_vars,
    levels=levels,
    positional_encoding=positional_encoding,
)
assert len(dataset) > 0, "There doesn't seem to be any valid data."

# The model takes as static parameters the mean and variance values of the input variables and the variance values of the target difference, i.e., the variance between climatology and instantaneous variables. 

surf_in_scal_path = Path("/data/climatology/musigma_surface.nc")
#hf_hub_download(
#    repo_id="ibm-nasa-geospatial/Prithvi-WxC-1.0-2300M",
#    filename=f"climatology/{surf_in_scal_path.name}",
#    local_dir="data",
#)

vert_in_scal_path = Path("/data/climatology/musigma_vertical.nc")
#hf_hub_download(
#    repo_id="ibm-nasa-geospatial/Prithvi-WxC-1.0-2300M",
#    filename=f"climatology/{vert_in_scal_path.name}",
#    local_dir="data",
#)

surf_out_scal_path = Path("/data/climatology/anomaly_variance_surface.nc")
#hf_hub_download(
#    repo_id="ibm-nasa-geospatial/Prithvi-WxC-1.0-2300M",
#    filename=f"climatology/{surf_out_scal_path.name}",
#    local_dir="data",
#)

vert_out_scal_path = Path("/data/climatology/anomaly_variance_vertical.nc")
#hf_hub_download(
#    repo_id="ibm-nasa-geospatial/Prithvi-WxC-1.0-2300M",
#    filename=f"climatology/{vert_out_scal_path.name}",
#    local_dir="data",
#)

in_mu, in_sig = input_scalers(
    surface_vars,
    vertical_vars,
    levels,
    surf_in_scal_path,
    vert_in_scal_path,
)

output_sig = output_scalers(
    surface_vars,
    vertical_vars,
    levels,
    surf_out_scal_path,
    vert_out_scal_path,
)

static_mu, static_sig = static_input_scalers(
    surf_in_scal_path,
    static_surface_vars,
)

residual = "climate"
masking_mode = "global"
encoder_shifting = True
decoder_shifting = True
masking_ratio = 0.0

#hf_hub_download(
#    repo_id="ibm-nasa-geospatial/Prithvi-WxC-1.0-2300M",
#    filename="config.yaml",
#    local_dir="data",
#)

with open("/data/config.yaml", "r") as f:
    config = yaml.safe_load(f)

model = PrithviWxC(
    in_channels=config["params"]["in_channels"],
    input_size_time=config["params"]["input_size_time"],
    in_channels_static=config["params"]["in_channels_static"],
    input_scalers_mu=in_mu,
    input_scalers_sigma=in_sig,
    input_scalers_epsilon=config["params"]["input_scalers_epsilon"],
    static_input_scalers_mu=static_mu,
    static_input_scalers_sigma=static_sig,
    static_input_scalers_epsilon=config["params"][
        "static_input_scalers_epsilon"
    ],
    output_scalers=output_sig**0.5,
    n_lats_px=config["params"]["n_lats_px"],
    n_lons_px=config["params"]["n_lons_px"],
    patch_size_px=config["params"]["patch_size_px"],
    mask_unit_size_px=config["params"]["mask_unit_size_px"],
    mask_ratio_inputs=masking_ratio,
    mask_ratio_targets=0.0,
    embed_dim=config["params"]["embed_dim"],
    n_blocks_encoder=config["params"]["n_blocks_encoder"],
    n_blocks_decoder=config["params"]["n_blocks_decoder"],
    mlp_multiplier=config["params"]["mlp_multiplier"],
    n_heads=config["params"]["n_heads"],
    dropout=config["params"]["dropout"],
    drop_path=config["params"]["drop_path"],
    parameter_dropout=config["params"]["parameter_dropout"],
    residual=residual,
    masking_mode=masking_mode,
    encoder_shifting=encoder_shifting,
    decoder_shifting=decoder_shifting,
    positional_encoding=positional_encoding,
    checkpoint_encoder=[],
    checkpoint_decoder=[],
)

weights_path = Path("/data/weights/prithvi.wxc.2300m.v1.pt")
#hf_hub_download(
#    repo_id="ibm-nasa-geospatial/Prithvi-WxC-1.0-2300M",
#    filename=weights_path.name,
#    local_dir="data/weights",
#)

state_dict = torch.load(weights_path, weights_only=False)
if "model_state" in state_dict:
    state_dict = state_dict["model_state"]
model.load_state_dict(state_dict, strict=True)

if (hasattr(model, "device") and model.device != device) or not hasattr(
    model, "device"
):
    model = model.to(device)

# Inference
data = next(iter(dataset))
batch = preproc([data], padding)

for k, v in batch.items():
    if isinstance(v, torch.Tensor):
        batch[k] = v.to(device)

rng_state_1 = torch.get_rng_state()
with torch.no_grad():
    model.eval()
    out = model(batch)

# plotting
t2m = out[0, variable_names.index("T2M")].cpu().numpy()

lat = np.linspace(-90, 90, out.shape[-2])
lon = np.linspace(-180, 180, out.shape[-1])
X, Y = np.meshgrid(lon, lat)

plt.contourf(X, Y, t2m, 100)
plt.gca().set_aspect("equal")
#plt.show()
plt.savefig('ex1.png')



