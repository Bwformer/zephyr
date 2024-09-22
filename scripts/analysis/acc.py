# This module is used to calculate the anomaly correlation coefficeint (ACC) of a single dlesm forecast or a series of dlesm forecasts. 

# imports 
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import os
import logging 
from .variable_metas import variable_metas
logger = logging.getLogger(__name__)
logging.getLogger('matplotlib').setLevel(logging.ERROR)

# Template for param dictionary that can be used to call acc. Also used for testing the module. 
EXAMPLE_PARAMS = {
    # first param is a list of dictionaries. These contain mnecssary information for calculating the acc of that forecast:
    # they may also contain plotting kwargs if the user wants to plot the acc and a cache path to save the calculated skill
    'forecast_params': [
        {'file':'tests/analysis/test_rmse/example_forecast1.nc',
         'plot_kwargs': {'label': 'test1', 'color': 'blue', 'linewidth': 2},
         'verification_file': 'tests/analysis/test_acc/test_era5_hpx32_z500.nc',
         'climatology_file': None,
         'acc_cache': 'tests/analysis/test_acc/example_forecast1_ACC-z500.nc',
        },
        {'file':'tests/analysis/test_rmse/example_forecast2.nc',
         'plot_kwargs': {'label': 'test2', 'color': 'green', 'linewidth': 2},
         'verification_file': 'tests/analysis/test_rmse/test_era5_hpx32_z500.nc',
         'climatology_file': None,
         'acc_cache': 'tests/analysis/test_acc/example_forecast2_ACC-z500.nc',
        },
    ],
    # name of the cariable to be used in the rmse calculation
    'variable': 'z500',
    # path to save the plot of the rmse
    'plot_file': 'tests/analysis/test_acc/acc_plot.png',
    'xlim': {'left':0, 'right':2},
    # whether to return calculated rmse 
    'return_acc': True,
}

def acc(
        forecast_params,
        variable,
        plot_file=None,
        xlim=None,
        return_acc=False,
): 
    """
    Calculates the anomaly correlation coefficeint (ACC) for a set of forecasts and plots the results.

    Parameters:
    forecast_params (list of dict): A list of dictionaries with instrcutions for handling each forecast. Each may contain: 
        'file' (str): The file path to the forecast data.
        'verification_file' (str): The file path to the verification data to compare to the forecast. ALso used for calculating the climatology.
        'climatology_file' (str, optional): The file path to the climatology data. If None, the climatology is calculated from the verification data.
        'acc_cache' (str, optional): The file path where the calculated ACC should be cached. If None, the ACC is not cached.
        'plot_kwargs' (dict, optional): The keyword arguments to be passed to the plot function for that forecast.
    variable (str): The variable for which the ACC is to be calculated.
    plot_file (str, optional): The file path where the plot should be saved. If None, the plot is not saved. Defaults to None.
    xlim (dict, optional): The x-axis limits for the plot. Defaults to None.
    return_acc (bool, optional): If True, the function returns the calculated ACC. Defaults to False.

    Returns:
    list of xarray.DataArray: A list of ACC values for each forecast. Only returned if return_acc is True.

    """
    acc = []
    # iterate through forecats and obtain acc 
    for forecast_param in forecast_params:
        # if acc is cached already, load it
        if os.path.isfile(forecast_param.get('acc_cache','')):
            logger.info(f"Loading ACC from {forecast_param['acc_cache']}.")
            acc.append(xr.open_dataarray(forecast_param['acc_cache']))
        else:
            logger.info(f"Calculating ACC for {forecast_param['file']}.")
        
            # open forecast and verification data
            forecast = xr.open_dataset(forecast_param['file'])[variable]
            verif_raw = xr.open_dataset(forecast_param['verification_file'])[variable]

            # calculate climatology
            if forecast_param.get('climatology_file',None) is not None:
                if os.path.isfile(forecast_param['climatology_file']):
                    logger.info(f"Loading climatology from {forecast_param['climatology_file']}")
                    climo_raw = xr.open_dataset(forecast_param['climatology_file'])[variable]
                else:
                    logger.info(f"Calculating climatology from {forecast_param['verification_file']} and caching to {forecast_param['climatology_file']}.")
                    climo_raw = verif_raw.groupby('sample.dayofyear').mean(dim='sample')
                    climo_raw.to_netcdf(forecast_param['climatology_file'])
            else: 
                logger.info(f"Calculating climatology from {forecast_param['verification_file']}.")
                climo_raw = verif_raw.groupby('sample.dayofyear').mean(dim='sample')

            # align verification data with forecast data
            logger.info("Aligning verification and climatology with forecast data.")
            verif = xr.full_like(forecast, fill_value=np.nan)
            climo = xr.full_like(forecast, fill_value=np.nan)
            for time in forecast.time:
                for step in forecast.step:
                    verif.loc[{'time': time, 'step': step}] = verif_raw.sel(sample=time + step).values
                    climo.loc[{'time': time, 'step': step}] = climo_raw.sel(dayofyear=(time + step).dt.dayofyear).values

            # calculate anomalies 
            forec_anom = forecast - climo
            verif_anom = verif - climo
            
            # calculate acc
            axis_mean = ['time', 'face', 'height','width']
            acc.append((verif_anom * forec_anom).mean(dim=axis_mean, skipna=True)
                / np.sqrt((verif_anom**2).mean(dim=axis_mean, skipna=True) *
                          (forec_anom**2).mean(dim=axis_mean, skipna=True)))

            # cache acc if requested
            if forecast_param.get('acc_cache',None) is not None:
                logger.info(f"Caching ACC to {forecast_param['acc_cache']}.")
                acc[-1].to_netcdf(forecast_param['acc_cache'])

    # plot acc if requested
    if plot_file is not None:
        fig, ax = plt.subplots()
        for skill, plot_kwargs in zip(acc, [forecast_param['plot_kwargs'] for forecast_param in forecast_params]):
            ax.plot([s / np.timedelta64(1, 'D') for s in skill.step.values], # plot in days 
                    skill, # scale to physical units
                    **plot_kwargs) # style curve and label

        # style plot
        # ax.set_xlabel('Forecast Days')
        ax.set_xlabel('Forecast lead time [days]')
        ax.set_ylabel(f'ACC')
        ax.grid()
        ax.legend()
        ax.set_xlim(**{'left':0, 'right':max([t.step[-1].values / np.timedelta64(1, 'D') for t in acc])} if xlim is None else xlim)
        ax.set_xticks(np.arange(0, ax.get_xlim()[1], 2))
        ax.set_ylim(bottom=0, top=1)
        logger.info(f"Saving plot to {plot_file}.")
        fig.savefig(plot_file,dpi=200)

    # return acc if requested
    if return_acc:
        return acc
    else:
        return