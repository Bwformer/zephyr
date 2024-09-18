from utils import (
    map2hpx_cuda,
    map2hpx_cuda_gpu,
    update_scaling
)
import numpy as np
from omegaconf import OmegaConf


"""
This is a script to remap lat-lon to a HEALPix mesh (GPU, under remap_py310 kernel).
and update the scaling parameters for the remapped variables.
"""
# parameters for healpix remapping
hpx_params = [
    {
        "file_name": "/home/disk/rhodium/bowenliu/ERA5/era5_1950-2022_3h_1deg_z150.nc",
        "target_variable_name": "z150",
        "file_variable_name": "z",
        "prefix": "/home/disk/rhodium/bowenliu/HPX32/era5_1deg_3h_HPX32_1950-2022_",
        "nside": 32,
    },
    {
        "file_name": "/home/disk/rhodium/bowenliu/ERA5/era5_1950-2022_3h_1deg_z50.nc",
        "target_variable_name": "z50",
        "file_variable_name": "z",
        "prefix": "/home/disk/rhodium/bowenliu/HPX32/era5_1deg_3h_HPX32_1950-2022_",
        "nside": 32,
    },
]

# Remap data to HPX mesh
for hpx_param in hpx_params:
    # map2hpx_cuda.main(hpx_param)
    map2hpx_cuda_gpu.main(hpx_param)


