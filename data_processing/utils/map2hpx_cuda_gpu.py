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
    if torch.cuda.is_available():
        device = torch.device("cuda:1")
    else:
        device = torch.device("cpu")
    x_tensor = torch.tensor(x, dtype=torch.double).to(device)
    regrid_func = regrid_func.to(device)
    data = regrid_func(x_tensor)
    return data.cpu().numpy().reshape(shape)


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


    ds_regridded = regrid_tensor(ds.values, regrid, (len(ds.time), 12, n_side, n_side))
    ds_regridded = xr.Dataset(
        {
            "var": (["time", "face", "height", "width"], ds_regridded),
        },
        coords={
            "time": ds.time,
            "face": np.arange(12),
            "height": np.arange(n_side),
            "width": np.arange(n_side),
        },
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