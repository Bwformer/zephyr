import xarray as xr
import numpy as np
import os
import warnings
import torch
import earth2grid # change to remap_py310 kernel
import argparse

EXAMPLE_PARAMS = {
    'file_name' : '/home/disk/rhodium/bowenliu/ERA5/era5_1950-2022_3h_1deg_z150.nc',
    'file_variable_name' : 'z', # this is used to extract the DataArray from the Dataset
    'target_variable_name' : 'z150', # this is how the variable will be saved in the new file 
    # this is the "prefix" for the output file. Should include the desired path to the output file
    'prefix' : '/home/disk/rhodium/bowenliu/HPX32/era5_1deg_3h_HPX32_1950-2022_',
    'nside' : 32,
}


def regrid_tensor(x, regrid_func, shape):
    data = regrid_func(torch.tensor(x, dtype=torch.double))
    return data.numpy().reshape(shape)


def hpx_regrid(
        ds: xr.Dataset, # lat lon grid dataset 
        lat_name: str = 'latitude', # contains the coordinate names to use for regridding
        lon_name: str = 'longitude',
        level: int = 6, # regrid resolution
        n_side: int = 64,
        ) -> xr.Dataset:
    # lat_long_names = dlwp_names.lat_lon_dims
    # longitude = lat_long_names[1]
    # latitude = lat_long_names[0]
    lons = ds[lon_name]
    lats = ds[lat_name]

    hpx = earth2grid.healpix.Grid(
    level=level, pixel_order=earth2grid.healpix.HEALPIX_PAD_XY
    )
    src = earth2grid.latlon.LatLonGrid(lat=list(lats), lon=list(lons))
    # Regridder
    regrid = earth2grid.get_regridder(src, hpx)


    ds_regridded = xr.apply_ufunc(
    regrid_tensor,
    ds,
    input_core_dims=[[lat_name, lon_name]],
    output_core_dims=[["face", "height", "width"]],
    output_sizes={"face": 12, "height": n_side, "width": n_side},
    output_dtypes=[float],
    dask="parallelized",
    vectorize=True,
    on_missing_core_dim="copy",
    kwargs={"regrid_func": regrid, "shape": (12, n_side, n_side)},
    dask_gufunc_kwargs={"allow_rechunk": True},
    )
    # Assign coordinates to the regridded dataset
    time_coords = ds.coords["time"]
    nside_coords = np.arange(n_side)
    grid_coords = np.arange(12)
    ds_regridded = ds_regridded.assign_coords(
    time=time_coords,
    face=grid_coords,
    height=nside_coords,
    width=nside_coords,
    )

    return ds_regridded

def main(params):
    # ingore the Future warning
    warnings.filterwarnings("ignore", category=FutureWarning)

    # create namespace object for quick attribute referencing
    args = argparse.Namespace(**params)

    if args.nside == 32:
        level = 5
    elif args.nside == 64:
        level = 6

    if not os.path.isfile(args.file_name):
        print(f'source file ~{args.file_name}~ not found. Aborting.')
        return
    if os.path.isfile(args.prefix+args.target_variable_name+'.nc'):
        print(f'target file ~{args.prefix+args.target_variable_name+".nc"}~ already exists. Aborting.')
        return


    ds = xr.open_dataset(args.file_name)
    ds_hpx = hpx_regrid(ds=ds[args.file_variable_name],
                        level=level,
                        n_side=args.nside,
                        )
    ds_hpx.name = args.target_variable_name
    ds_hpx.to_netcdf(args.prefix+args.target_variable_name+'.nc')
    print(f'file saved to {args.prefix+args.target_variable_name+".nc"}')
    print()

# # 1. map to healpix mesh
if __name__ == '__main__':
    # ingore the Future warning
    warnings.filterwarnings("ignore", category=FutureWarning)

    ds = xr.open_dataset("/home/disk/rhodium/bowenliu/ERA5/era5_1950-2022_3h_1deg_z150.nc")
    ds_hpx = hpx_regrid(ds=ds.z[:100,...],
                        level=5, # 6 for n_side=64
                        n_side=32, # 64
                        )
    print(f'output shape: {ds_hpx.shape}')
    print(f'output coords: {ds_hpx.coords}')
    print(f'output names: {ds_hpx.name}')
    ds_hpx.name = 'z150'
    print(f'output new names: {ds_hpx.name}')
    print()