from utils import (
    era5_retrieval,
    data_imputation,
    map2hpx,
    windspeed,
    trailing_average,
    update_scaling,
)
from training.dlwp.data import data_loading as dl
import numpy as np
from omegaconf import OmegaConf


"""
This data pipline creates training, validation, and test data for a coupled deep learning ocean model (DLOM). The coupled DLOM will forecast on a HEALPix 32 mesh, have 48 hour resolution, and predict a single prognostic field: sea surface temperature (SST). The model is trained to receive 10m windspeed and geopotential height at 1000 hPa for the 4 day period forecast. 
"""

era5_requests = [
    # z150
    {
        "constant": False,
        "single_level_variable": False,
        "variable_name": "z",
        "pressure_level": "150",
        "grid": [1, 1],
        "year": [y for y in range(1950, 2023)],
        "month": [month + 1 for month in range(0, 12)],
        "day": [d + 1 for d in range(0, 31)],
        "time": np.arange(0, 24, 3).tolist(),
        "target_file": "/home/disk/rhodium/bowenliu/ERA5/era5_1950-2022_3h_1deg_z150.nc",
    },
    # z50
    {
        "constant": False,
        "single_level_variable": False,
        "variable_name": "z",
        "pressure_level": "50",
        "grid": [1, 1],
        "year": [y for y in range(1950, 2023)],
        "month": [month + 1 for month in range(0, 12)],
        "day": [d + 1 for d in range(0, 31)],
        "time": np.arange(0, 24, 3).tolist(),
        "target_file": "/home/disk/rhodium/bowenliu/ERA5/era5_1950-2022_3h_1deg_z50.nc",
    },
]
# parameters for healpix remapping
hpx_params = [
    {
        "file_name": "/home/disk/rhodium/bowenliu/ERA5/era5_1950-2022_3h_1deg_z150.nc",
        "target_variable_name": "z150",
        "file_variable_name": "z",
        "prefix": "/home/disk/rhodium/bowenliu/HPX32/era5_1deg_3h_HPX32_1950-2022_",
        "nside": 32,
        "order": "bilinear",
        "resolution_factor": 1.0,
        "visualize": False,
        "poolsize": 30 
    },
    {
        "file_name": "/home/disk/rhodium/bowenliu/ERA5/era5_1950-2022_3h_1deg_z50.nc",
        "target_variable_name": "z50",
        "file_variable_name": "z",
        "prefix": "/home/disk/rhodium/bowenliu/HPX32/era5_1deg_3h_HPX32_1950-2022_",
        "nside": 32,
        "order": "bilinear",
        "resolution_factor": 1.0,
        "visualize": False,
        "poolsize": 30 # The number of parallel processes to use
    },
]

# Define the parameters for updating the scaling parameters of various variables
update_scaling_params = {
    "scale_file": "/home/disk/brume/bowenliu/git_clone/zephyr/training/configs/data/scaling/hpx32.yaml",
    "variable_file_prefix": "/home/disk/rhodium/bowenliu/HPX32/era5_1deg_3h_HPX32_1950-2022_",
    "variable_names": [
        "z150",
        "z50",
    ],
    "selection_dict": {
        "sample": slice(np.datetime64("1950-01-01"), np.datetime64("2022-12-31"))
    },
    "overwrite": False,
    "chunks": None,
}
# parameters used to write optimized zarr file
zarr_params = {
    "src_directory": "/home/disk/rhodium/dlwp/data/HPX32/",
    "dst_directory": "/home/disk/rhodium/dlwp/data/HPX32/",
    "dataset_name": "hpx32_1950-2022_3h_sst_coupled",
    "input_variables": [
        "sst",
        "ws10-48H",
        "z1000-48H",
    ],
    "output_variables": [
        "sst",
    ],
    "constants": {"lsm": "lsm"},
    "prefix": "era5_1deg_3h_HPX32_1950-2022_",
    "batch_size": 16,
    "scaling": OmegaConf.load(
        update_scaling.create_yaml_if_not_exists(update_scaling_params["scale_file"])
    ),
    "overwrite": False,
}
# Retrive raw data
# for request in era5_requests:
#     era5_retrieval.main(request)
# Remap data to HPX mesh
for hpx_param in hpx_params:
    map2hpx.main(hpx_param)
# update scaling dictionary
update_scaling.main(update_scaling_params)
# # create zarr file for optimized training
# dl.create_time_series_dataset_classic(**zarr_params)
