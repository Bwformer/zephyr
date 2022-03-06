import logging
import re

import numpy as np
import pandas as pd
from pytorch_lightning.callbacks import TQDMProgressBar

logger = logging.getLogger(__name__)


def configure_logging(verbose=1):
    verbose_levels = {
        0: logging.WARNING,
        1: logging.INFO,
        2: logging.DEBUG,
        3: logging.NOTSET
    }
    if verbose not in verbose_levels.keys():
        verbose = 1
    logger = logging.getLogger()
    logger.setLevel(verbose_levels[verbose])
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "[%(asctime)s][PID=%(process)d]"
        "[%(levelname)s %(filename)s:%(lineno)d] - %(message)s"))
    handler.setLevel(verbose_levels[verbose])
    logger.addHandler(handler)


def remove_chars(s):
    """
    Remove characters from a string that have unintended effects on file paths.
    :param s: str
    :return: str
    """
    return ''.join(re.split('[$/\\\\]', s))


def to_chunked_dataset(ds, chunking):
    """
    Create a chunked copy of a Dataset with proper encoding for netCDF export.
    :param ds: xarray.Dataset
    :param chunking: dict: chunking dictionary as passed to
        xarray.Dataset.chunk()
    :return: xarray.Dataset: chunked copy of ds with proper encoding
    """
    chunk_dict = dict(ds.dims)
    chunk_dict.update(chunking)
    for var in ds.data_vars:
        if 'coordinates' in ds[var].encoding:
            del ds[var].encoding['coordinates']
        ds[var].encoding['contiguous'] = False
        ds[var].encoding['original_shape'] = ds[var].shape
        ds[var].encoding['chunksizes'] = tuple([chunk_dict[d] for d in ds[var].dims])
        ds[var].encoding['chunks'] = tuple([chunk_dict[d] for d in ds[var].dims])
    return ds


def encode_variables_as_int(ds, dtype='int16', compress=0, exclude_vars=()):
    """
    Adds encoding to Dataset variables to export to int16 type instead of float at write time.
    :param ds: xarray Dataset
    :param dtype: as understood by numpy
    :param compress: int: if >1, also enable compression of the variable with this compression level
    :param exclude_vars: iterable of variable names which are not to be encoded
    :return: xarray Dataset with encoding applied
    """
    for var in ds.data_vars:
        if var in exclude_vars:
            continue
        var_max = float(ds[var].max())
        var_min = float(ds[var].min())
        var_offset = (var_max + var_min) / 2
        var_scale = (var_max - var_min) / (2 * np.iinfo(dtype).max)
        if var_scale == 0:
            logger.warning(f"min and max for variable {var} are both {var_max}")
            var_scale = 1.
        ds[var].encoding.update({
            'scale_factor': var_scale,
            'add_offset': var_offset,
            'dtype': dtype,
            '_Unsigned': not np.issubdtype(dtype, np.signedinteger),
            '_FillValue': np.iinfo(dtype).min,
        })
        if 'valid_range' in ds[var].attrs:
            del ds[var].attrs['valid_range']
        if compress > 0:
            ds[var].encoding.update({
                'zlib': True,
                'complevel': compress
            })
    return ds


def insolation(dates, lat, lon, S=1., daily=False, enforce_2d=False, clip_zero=True):
    """
    Calculate the approximate solar insolation for given dates.

    For an example reference, see:
    https://brian-rose.github.io/ClimateLaboratoryBook/courseware/insolation.html

    :param dates: 1d array: datetime or Timestamp
    :param lat: 1d or 2d array of latitudes
    :param lon: 1d or 2d array of longitudes (0-360deg). If 2d, must match the shape of lat.
    :param S: float: scaling factor (solar constant)
    :param daily: bool: if True, return the daily max solar radiation (lat and day of year dependent only)
    :param enforce_2d: bool: if True and lat/lon are 1-d arrays, turns them into 2d meshes.
    :param clip_zero: bool: if True, set values below 0 to 0
    :return: 3d array: insolation (date, lat, lon)
    """
    if len(lat.shape) != len(lon.shape):
        raise ValueError("'lat' and 'lon' must have the same number of dimensions")
    if len(lat.shape) >= 2 and lat.shape != lon.shape:
        raise ValueError(f"shape mismatch between lat ({lat.shape} and lon ({lon.shape})")
    if len(lat.shape) == 1 and enforce_2d:
        lon, lat = np.meshgrid(lon, lat)
    n_dim = len(lat.shape)

    # Constants for year 1995 (standard in climate modeling community)
    # Obliquity of Earth
    eps = 23.4441 * np.pi / 180.
    # Eccentricity of Earth's orbit
    ecc = 0.016715
    # Longitude of the orbit's perihelion (when Earth is closest to the sun)
    om = 282.7 * np.pi / 180.
    beta = np.sqrt(1 - ecc ** 2.)

    # Get the day of year as a float.
    start_years = np.array([pd.Timestamp(pd.Timestamp(d).year, 1, 1) for d in dates], dtype='datetime64')
    days_arr = (np.array(dates, dtype='datetime64') - start_years) / np.timedelta64(1, 'D')
    for d in range(n_dim):
        days_arr = np.expand_dims(days_arr, -1)
    # For daily max values, set the day to 0.5 and the longitude everywhere to 0 (this is approx noon)
    if daily:
        days_arr = 0.5 + np.round(days_arr)
        new_lon = lon.copy().astype(np.float32)
        new_lon[:] = 0.
    else:
        new_lon = lon.astype(np.float32)
    # Longitude of the earth relative to the orbit, 1st order approximation
    lambda_m0 = ecc * (1. + beta) * np.sin(om)
    lambda_m = lambda_m0 + 2. * np.pi * (days_arr - 80.5) / 365.
    lambda_ = lambda_m + 2. * ecc * np.sin(lambda_m - om)
    # Solar declination
    dec = np.arcsin(np.sin(eps) * np.sin(lambda_))
    # Hour angle
    h = 2 * np.pi * (days_arr + new_lon / 360.)
    # Distance
    rho = (1. - ecc ** 2.) / (1. + ecc * np.cos(lambda_ - om))

    # Insolation
    sol = S * (np.sin(np.pi / 180. * lat[None, ...]) * np.sin(dec) -
               np.cos(np.pi / 180. * lat[None, ...]) * np.cos(dec) *
               np.cos(h)) * rho ** -2.
    if clip_zero:
        sol[sol < 0.] = 0.

    return sol.astype(np.float32)


class CustomProgressBar(TQDMProgressBar):
    def get_metrics(self, *args, **kwargs):
        # don't show the version number
        items = super().get_metrics(*args, **kwargs)
        items.pop("v_num", None)
        return items
