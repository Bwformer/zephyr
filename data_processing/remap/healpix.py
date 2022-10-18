#! env/bin/python3

"""
This class contains reprojection methods to convert latlon data to and from HEALPix data. In this implementation, the
HEALPix structure is translated from its 1D array into a 3D array structure [F, H, W], where F=12 is the number of
faces and H=W=nside of the HEALPix map. The HEALPix base faces are indiced as follows


         HEALPix                              Face order                 3D array representation
                                                                            -----------------
--------------------------               //\\  //\\  //\\  //\\             |   |   |   |   |
|| 0  |  1  |  2  |  3  ||              //  \\//  \\//  \\//  \\            |0  |1  |2  |3  |
|\\  //\\  //\\  //\\  //|             /\\0 //\\1 //\\2 //\\3 //            -----------------
| \\//  \\//  \\//  \\// |            // \\//  \\//  \\//  \\//             |   |   |   |   |
|4//\\5 //\\6 //\\7 //\\4|            \\4//\\5 //\\6 //\\7 //\\             |4  |5  |6  |7  |
|//  \\//  \\//  \\//  \\|             \\/  \\//  \\//  \\//  \\            -----------------
|| 8  |  9  |  10 |  11  |              \\8 //\\9 //\\10//\\11//            |   |   |   |   |
--------------------------               \\//  \\//  \\//  \\//             |8  |9  |10 |11 |
                                                                            -----------------
                                    "\\" are top and bottom, whereas
                                    "//" are left and right borders


Details on the HEALPix can be found at https://iopscience.iop.org/article/10.1086/427976
"""

import os
from tqdm import tqdm
import multiprocessing

import numpy as np
import healpy as hp
import xarray as xr
import reproject as rp
import astropy as ap

# https://stackoverflow.com/questions/57354700/starmap-combined-with-tqdm/57364423#57364423
from .istarmap import istarmap
from .base import _BaseRemap
from .cubesphere import to_chunked_dataset

import matplotlib.pyplot as plt


class HEALPixRemap(_BaseRemap):

    def __init__(
            self,
            latitudes: int,
            longitudes: int,
            nside: int,
            order: str = "bilinear",
            resolution_factor: float = 1.0,
            verbose: bool = True
            ):
        """
        Consructor

        :param latitudes: The number of pixels in vertical direction of the LatLon data
        :param longitudes: The number of pixels in horizontal direction of the LatLon data
        :param nside: The number of pixels each HEALPix face sides has
        :param order: (Optional) The interpolation scheme ("nearest-neighbor", "bilinear", "biquadratic", "bicubic"),
        :param resolution_factor: (Optional) In some cases, when choosing nside "too large" for the source data, the
            projection can contain NaN values. Choosing a resolution_factor > 1.0 can resolve this but requires careful
            inspection of the projected data.
        :param verbose: (Optional) Whether to print progress to console
        """
        super().__init__()
        self.latitudes = latitudes
        self.longitudes = longitudes
        self.nside = nside
        self.order = order
        self.nested = True  # RING representation not supported in this implementation
        self.verbose = verbose

        resolution = 360./longitudes
        self.npix = hp.nside2npix(nside)

        # Define and generate world coordinate systems (wcs) for forward and backward mapping. More information at
        # https://github.com/astropy/reproject/issues/87
        # https://docs.astropy.org/en/latest/wcs/supported_projections.html
        wcs_input_dict = {
            'CTYPE1': 'RA',  # can be further specified with, e.g., RA---MOL, GLON-MOL, ELON-MOL
            'CUNIT1': 'deg',
            'CDELT1': -resolution*resolution_factor,  # -r produces for some reason less NaNs
            'CRPIX1': (longitudes)/2,
            'CRVAL1': 180.0,
            'NAXIS1': longitudes,
            'CTYPE2': 'DEC',  # can be further specified with, e.g., DEC--MOL, GLAT-MOL, GLAT-MOL 
            'CUNIT2': 'deg',
            'CDELT2': -resolution,
            'CRPIX2': (latitudes+1)/2,
            'CRVAL2': 0.0,
            'NAXIS2': latitudes
        }
        self.wcs_ll2hpx = ap.wcs.WCS(wcs_input_dict)

        wcs_input_dict = {
            'CTYPE1': 'RA',  # can be further specified with, e.g., RA---MOL, GLON-MOL, ELON-MOL
            'CUNIT1': 'deg',
            'CDELT1': resolution*resolution_factor,
            'CRPIX1': (longitudes)/2,
            'CRVAL1': 180.0,
            'NAXIS1': longitudes,
            'CTYPE2': 'DEC',  # can be further specified with, e.g., DEC--MOL, GLAT-MOL, GLAT-MOL 
            'CUNIT2': 'deg',
            'CDELT2': resolution,
            'CRPIX2': (latitudes+1)/2,
            'CRVAL2': 0.0,
            'NAXIS2': latitudes
        }
        self.wcs_hpx2ll = ap.wcs.WCS(wcs_input_dict)

        # Determine HEALPix indices of the projected map
        #thetas, phis = hp.pix2ang(self.nside, np.arange(hp.nside2npix(self.nside)))
        #self.hpxidcs = hp.ang2pix(self.nside, thetas, phis)#, nest=self.nested)

    def remap(
            self,
            file_path: str,
            prefix: str = "era5_1deg_3h_HPX32_1979-2018_",
            target_variable_name: str = "z500",
            poolsize: int = 20,
            chunk_ds: bool = True,
            to_netcdf: bool = True,
            times: xr.DataArray = None,
            ) -> xr.Dataset:
        """
        Takes a (preprocessed) LatLon dataset of shape [sample, varlev, lat, lon] and converts it into the HEALPix
        geometry with shape [sample, varlev, face, height, width], writes it to file and returns it.

        :param file_path: The path to the dataset in LatLon convention
        :param prefix: First part of the target variable name
        :param target_variable_name: The name for the target variable (following the prefix)
        :param poolsize: Number of processes to be used for the parallel remapping
        :param chunk_ds: Whether to chunk the dataset (recommended for fast data loading)
        :param to_netcdf: Whether to write the dataset to file
        :param times: An xarray DataArray of desired time steps; or compatible, e.g., slice(start, stop)
        :return: The converted dataset in HPX convention
        """

        # Load .nc file in latlon format to extract latlon information and to initialize the remapper module
        ds_ll = xr.open_dataset(file_path)
        if times is not None: ds_ll = ds_ll.sel({"sample": times})

        # Determine whether a "constant" or "variable" is processed
        const = False if "predictors" in list(ds_ll.keys()) else True
        vname = list(ds_ll.keys())[0] if const else "predictors"

        # Set up coordinates and chunksizes for the HEALPix dataset
        coords = {}
        if not const:
            coords["sample"] = ds_ll.coords["sample"]
            coords["varlev"] = ds_ll.coords["varlev"]
        coords["face"] = np.array(range(12), dtype=np.int64)
        coords["height"] = np.array(range(self.nside), dtype=np.int64)
        coords["width"] = np.array(range(self.nside), dtype=np.int64)
        chunksizes = {coord: len(coords[coord]) for coord in coords}

        # Map the "constant" or "variable" to HEALPix
        if const:
            data_hpx = self.ll2hpx(data=ds_ll.variables[vname].values)
            ds_mean = ds_ll.variables[vname].mean()
            ds_std = ds_ll.variables[vname].std()
        else:
            dims = [len(coords[coord]) for coord in coords]

            if poolsize < 2:
                # Sequential sample mapping via for-loop

                # Allocate a (huge) array to store all samples (time steps) of the projected data
                data_hpx = np.zeros(dims, dtype=ds_ll.variables[vname])

                # Iterate over all samples and levels, project them to HEALPix and store them in the predictors array
                pbar = tqdm(ds_ll.coords["sample"], disable=not self.verbose)
                for s_idx, sample in enumerate(pbar):
                    pbar.set_description("Remapping time steps")
                    for l_idx, level in enumerate(ds_ll.coords["varlev"]):
                        data_hpx[s_idx, l_idx] = self.ll2hpx(data=ds_ll.variables[vname][s_idx, l_idx].values)
            else:
                # Parallel sample mapping with 'poolsize' processes

                # Collect the arguments for each remapping call
                arguments = []
                if self.verbose: print("Preparing arguments for parallel remapping")
                for s_idx in tqdm(range(ds_ll.dims["sample"]), disable=not self.verbose):
                    for l_idx, level in enumerate(ds_ll.coords["varlev"]):
                        arguments.append([self, ds_ll.variables[vname][s_idx, l_idx].values])

                # Run the remapping in parallel
                with multiprocessing.Pool(poolsize) as pool:
                    if self.verbose:
                        print(f"Remapping time steps with {poolsize} processes in parallel")
                        data_hpx = np.array(list(tqdm(pool.istarmap(remap_parallel, arguments), total=len(arguments))))
                    else:
                        data_hpx = pool.starmap(remap_parallel, arguments)
                    pool.terminate()
                    pool.join()
                    
                # If 'level=1', it will not be included as dimension and needs to be added manually
                if data_hpx.shape != len(dims): data_hpx = np.expand_dims(data_hpx, axis=1)

            ds_mean = ds_ll.variables["mean"]
            ds_std = ds_ll.variables["std"]
            
            # Sample and level are loaded separately in DLWP
            chunksizes["sample"] = 1
            chunksizes["varlev"] = 1

        # Map the latitude and longitude fields to HEALPix
        data_lat = ds_ll["lat"].values  # 1D [H]
        data_lon = ds_ll["lon"].values  # 1D [W]
        data_lat = np.repeat(a=np.expand_dims(data_lat, axis=1), repeats=self.longitudes, axis=1)  # 2D [H, W]
        data_lon = np.repeat(a=np.expand_dims(data_lon, axis=0), repeats=self.latitudes, axis=0)   # 2D [H, W]
        data_lat = self.ll2hpx(data_lat)  # 3D [F, H, W] (HEALPix)
        data_lon = self.ll2hpx(data_lon)  # 3D [F, H, W] (HEALPix)

        data_lat = np.array(data_lat, dtype=np.float64)
        data_lon = np.array(data_lon, dtype=np.float64)

        # Build HEALPix dataset and write it to file
        ds_hpx = xr.Dataset(
            coords=coords,
            data_vars={
                "lat": (["face", "height", "width"], data_lat),
                "lon": (["face", "height", "width"], data_lon),
                vname: (list(coords.keys()), data_hpx),
                "mean": ds_mean,
                "std": ds_std
                },
            attrs=ds_ll.attrs,
            )
        if chunk_ds:
            ds_hpx = to_chunked_dataset(ds=ds_hpx, chunking=chunksizes)
        if to_netcdf:
            if self.verbose: print("Dataset sucessfully built. Writing data to file...")
            ds_hpx.to_netcdf(prefix + target_variable_name + ".nc")
        #print(ds_hpx.chunk())
        #ds_hpx.to_zarr(prefix + target_variable_name + ".zarr", mode="w")
        return ds_hpx

    def inverse_remap(
            self,
            forecast_path: str,
            verification_path: str,
            prefix: str = "forecast_",
            model_name: str = "model-name",
            vname: str = "z500",
            poolsize: int = 20,
            to_netcdf: bool = True,
            times: xr.DataArray = None,
            ) -> xr.Dataset:
        """
        Takes a (forecast) HEALPix dataset of shape [time, step, face, height, width] and converts it into the LatLon
        convention with shape [time, step, lat, lon], writes it to file and returns it.

        :param forecast_path: The path to the forecast dataset in HPX geometry
        :param verification_path: The path to the according ground truth file in LatLon convention
        :param prefix: First part of the target variable name
        :param model_name: The name of the model (to construct the target file name)
        :param vname: The variable of interest's name
        :param poolsize: Number of processes to be used for the parallel remapping
        :param to_netcdf: Whether to write the LL dataset to file
        :param times: An xarray DataArray of desired time steps; or compatible, e.g., slice(start, stop)
        :return: The converted dataset in LatLon convention
        """
        # Load .nc file in HEALPix format to get nside information and to initialize the remapper module
        fc_ds_hpx = xr.open_dataset(forecast_path)
        if times is not None: fc_ds_hpx = fc_ds_hpx.sel({"time": times})

        dims = [fc_ds_hpx.dims["time"], fc_ds_hpx.dims["step"], self.latitudes, self.longitudes]
        
        if poolsize < 2:
            # Sequential sample mapping via for-loop

            # Allocate a (huge) array to store all samples (time steps) of the projected data
            fc_data_ll = np.zeros(dims, dtype=fc_ds_hpx.variables[vname])

            # Iterate over all samples and levels, project them to HEALPix and store them in the predictors array
            pbar = tqdm(fc_ds_hpx.coords["time"], disable=not self.verbose)
            for f_idx, forecast_start_time in enumerate(pbar):
                pbar.set_description("Remapping time steps")
                for s_idx, step in enumerate(fc_ds_hpx.coords["step"]):
                    fc_data_ll[f_idx, s_idx] = self.hpx2ll(data=fc_ds_hpx.variables[vname][f_idx, s_idx].values)
        else:
            # Parallel sample mapping with 'poolsize' processes
            
            # Collect the arguments for each remapping call
            arguments = []
            if self.verbose: print("Preparing arguments for parallel remapping")
            for f_idx in tqdm(range(fc_ds_hpx.dims["time"]), disable=not self.verbose):
                for s_idx, step in enumerate(fc_ds_hpx.coords["step"]):
                    arguments.append([self, fc_ds_hpx.variables[vname][f_idx, s_idx].values])

            # Run the remapping in parallel
            with multiprocessing.Pool(poolsize) as pool:
                if self.verbose:
                    print(f"Remapping time steps with {poolsize} processes in parallel")
                    fc_data_ll = np.array(list(tqdm(pool.istarmap(inverse_remap_parallel, arguments),
                                                    total=len(arguments))))
                else:
                    fc_data_ll = pool.starmap(inverse_remap_parallel, arguments)
                pool.terminate()
                pool.join()
            fc_data_ll = np.reshape(fc_data_ll, dims)  # [(f s) lat lon] -> [f s lat lon]
        
        # Convert latitudes and longitudes from HEALPix to LatLon
        gt_ds = xr.open_dataset(verification_path)
        lat, lon = gt_ds["latitude"], gt_ds["longitude"]

        # Set up coordinates and chunksizes for the LatLon dataset
        coords = {"time": fc_ds_hpx.coords["time"],
                  "step": fc_ds_hpx.coords["step"],
                  "lat": np.array(lat, dtype=np.int64),
                  "lon": np.array(lon, dtype=np.int64)}
        
        # Build LatLon forecast dataset
        fc_ds_ll = xr.Dataset(coords=coords,
                              data_vars={vname: (list(coords.keys()), fc_data_ll)})
        if to_netcdf: fc_ds_ll.to_netcdf(f"{prefix}LL_{model_name.lower().replace(' ', '_')}_{vname}")

        return fc_ds_ll

    def ll2hpx(self, data: np.array, visualize: bool = False, **kwargs) -> np.array:
        """
        Projects a given array from latitude longitude into the HEALPix representation.

        :param data: The data of shape [height, width] in latlon format
        :param visualize: (Optional) Whether to visualize the data or not
        :return: An array of shape [f=12, h=nside, w=nside] containing the HEALPix data
        """

        # Flip data horizontally to use 'CDELT1 = -r' in the wcs for the reprojection below
        data = np.flip(data, axis=1)

        # Reproject latlon to HEALPix
        hpx1d, hpx1d_mask = rp.reproject_to_healpix(
            input_data=(data, self.wcs_ll2hpx),
            coord_system_out="icrs",
            nside=self.nside,
            order=self.order,
            nested=self.nested
            )

        # Convert the 1D HEALPix array into an array of shape [faces=12, nside, nside]
        hpx3d = np.zeros(shape=(12, self.nside, self.nside), dtype=np.float32)
        for hpxidx in range(self.npix):
            f, y, x = self.hpxidx2fyx(hpxidx=hpxidx)
            hpx3d[f, x, y] = hpx1d[hpxidx]

        # Compensate array indices [0, 0] representing top left and not bottom right corner (caused by hpxidx2fyx)
        hpx3d = np.flip(hpx3d, axis=(1,2))
        
        # Face index reordering/correction. Somewhat arbitrary; no clue why this is necessary
        #hpx3d = hpx3d[[1, 0, 3, 2, 6, 5, 4, 7, 9, 8, 11, 10]]
        #hpx3d = hpx3d[[9, 8, 11, 10, 6, 5, 4, 7, 1, 0, 3, 2,]]
        #hpx3d = hpx3d[[8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3]]

        if visualize:
            hp.cartview(hpx1d, title="Flipped and shifted horizontally", nest=True, **kwargs)
            hp.graticule()
            plt.savefig("cartview.pdf", format="pdf")

        assert hpx1d_mask.all(), self.nans_found_in_data(data=hpx3d, visualize=visualize)

        return hpx3d

    def hpx2ll(self, data: np.array, visualize: bool = False, **kwargs) -> np.array:
        """
        Projects a given three dimensional HEALPix array to latitude longitude representation.

        :param data: The data of shape [faces=12, height=nside, width=nside] in HEALPix format
        :param visualize: (Optional) Whether to visualize the data or not
        :return: An array of shape [height=latitude, width=longitude] containing the latlon data
        """
        # Recompensate array indices [0, 0] representing top left and not bottom right corner (required for fyx2hpxidx)
        #data = data[[9, 8, 11, 10, 6, 5, 4, 7, 1, 0, 3, 2]]
        data = data[[8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3]]

        # Convert the 3D [face, nside, nside] array back into the 1D HEALPix array
        hpx1d = np.zeros(self.npix, dtype=np.float32)
        for f in range(12):
            for y in range(self.nside):
                for x in range(self.nside):
                    hpxidx = self.fyx2hpxidx(f=f, y=y, x=x)
                    hpx1d[hpxidx] = data[f, y, x]

        # Project 1D HEALPix to LatLon
        ll2d, ll2d_mask = rp.reproject_from_healpix(
            input_data=(hpx1d, "icrs"),
            output_projection=self.wcs_hpx2ll,
            shape_out=(self.latitudes, self.longitudes),
            nested=self.nested
            )
        #ll2d = np.flip(ll2d, axis=1)  # Compensate flip in reprojection function above

        if visualize:
            plt.imshow(ll2d, **kwargs)
            plt.title("HPX mapped to LL")
            plt.tight_layout()
            plt.savefig("hpx2ll.pdf", format="pdf")

        assert ll2d_mask.all(), ("Found NaN in the projected data. This can occur when the resolution of the "
                                 "HEALPix data is smaller than that of the target latlon grid.")
        return ll2d

    def hpxidx2fyx(self, hpxidx: int) -> (int, int, int):
        """
        Determines the face (f), column (x), and row (y) indices for a given HEALPix index under consideration of the base
        face index [0, 1, ..., 11] and the number of pixels each HEALPix face side has (nside).

        :param hpxidx: The HEALPix index
        :return: A tuple containing the face, y, and x indices of the given HEALPix index
        """
        f = hpxidx//(self.nside**2)
        assert 0 <= f <= 11, "Face index must be within [0, 1, ..., 11]"

        # Get bit representation of hpxidx and split it into even and odd bits
        hpxidx = format(hpxidx%(self.nside**2), "b").zfill(self.nside)
        bits_eve = hpxidx[::2]
        bits_odd = hpxidx[1::2]

        # Compute row and column indices of the HEALPix index in the according face
        y = int(bits_eve, 2)
        x = int(bits_odd, 2)

        return (f, y, x)

    def fyx2hpxidx(self, f: int, x: int, y: int) -> int:
        """
        Computes the HEALPix index from a given face (f), row (y), and column (x) under consideration of the number of
        pixels along a HEALPix face (nside).

        :param f: The face index
        :param y: The local row index within the given face
        :param x: The local column index within the given face
        :return: The HEALPix index
        """

        # Determine even and odd bits of the HEALPix index from the row (y, even) and column (x, odd)
        bits_eve = format(y, "b").zfill(self.nside//2)
        bits_odd = format(x, "b").zfill(self.nside//2)

        # Alternatingly join the two bit lists. E.g., ["1", "0"] and ["1", "0"] becomes ["1", "1", "0", "0"]
        bitstring = ""
        for bit_idx in range(len(bits_eve)):
            bitstring += bits_eve[bit_idx]
            bitstring += bits_odd[bit_idx]

        return int(bitstring, 2) + f*self.nside**2

    def manual_projection():
        # Manual projection, as suggested at and modified from
        # https://stackoverflow.com/questions/31573572/healpy-from-data-to-healpix-map

        lats_deg = ds_ll.coords["lat"].values
        lons_deg = ds_ll.coords["lon"].values

        lats_rad = np.deg2rad(lats_deg)
        lons_rad = np.deg2rad(lons_deg)

        # Convert to healpix theta and phi notations: lat \in [0, pi], lon \in [0, 2pi]
        lats_hp = lats_rad + np.pi/2
        lons_hp = lons_rad

        # HEALPix setup
        npix = hp.nside2npix(nside)
        thetas = np.repeat(a=np.expand_dims(lats_hp, 1), repeats=len(lons_deg), axis=1)
        phis = np.repeat(a=np.expand_dims(lons_hp, 0), repeats=len(lats_deg), axis=0)
        indices = hp.ang2pix(nside, thetas, phis)

        # Projection
        hpmap = np.zeros(npix, dtype=np.float64)
        normalizer = np.zeros_like(hpmap) + 1e-7  # Prevent division by zero
        for i in range(len(indices)):
            hpmap[indices[i]] += data[i]
            normalizer[indices[i]] += 1
        hpmap /= normalizer

        hp.mollview(hpmap, title="Mollview image RING")
        hp.graticule()
        plt.tight_layout()
        plt.savefig("mollview_plot.pdf", format="pdf")

    def nans_found_in_data(self, data: np.array, visualize: bool = True) -> str:
        """
        Unifies the twelve HEALPix faces into one array and visualizes it if desired. Returns an error message.

        :param data: The data array [start_time, forecast_step, face, height, width]
        :param visualize: (Optional) Whether to visualize the data in face-representation
        :return: Error message string specifying that nans were found in the projected data
        """

        # Concatenate the faces in a HEALPix-like diamond structure
        f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11 = data

        nans = np.full(f0.shape, np.nan)
        row0 = np.concatenate((nans, nans, nans, f3, nans), axis=1)
        row1 = np.concatenate((nans, nans, f2, f7, f11), axis=1)
        row2 = np.concatenate((nans, f1, f6, f10, nans), axis=1)
        row3 = np.concatenate((f0, f5, f9, nans, nans), axis=1)
        row4 = np.concatenate((f4, f8, nans, nans, nans), axis=1)
        data = np.concatenate((row0, row1, row2, row3, row4), axis=0)

        if visualize:
            plt.imshow(data)
            plt.savefig("hpx_plot_with_nans.pdf", format="pdf")

        return ("Found NaN in the projected data. This can occur when the resolution of the original data is too "
                "small for the chosen HEALPix grid. Increasing the 'resolution_factor' of the HEALPixRemap instance "
                "might help. You may want to set 'visualize=True' when calling 'self.ll2hpx()' to write a "
                "'hpx_plot_with_nans.pdf' plot to file.")


def remap_parallel(mapper: HEALPixRemap, data: np.array) -> np.array:
    """
    Helper function to apply the mapping of individual samples (time steps) in parallel.

    :param data: The numpy array containing the LatLon data
    :return: A numpy array containing the data remapped to the HEALPix
    """
    return mapper.ll2hpx(data)


def inverse_remap_parallel(mapper: HEALPixRemap, data: np.array) -> np.array:
    """
    Helper function to apply the inverse mapping of individual samples (time steps) in parallel.

    :param data: The numpy array containing the LatLon data
    :return: A numpy array containing the data remapped to LatLon
    """
    return mapper.hpx2ll(data)
