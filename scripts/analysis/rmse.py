# This module is used to calculate the root mean square error (RMSE) of a single dlesm forecast or a series of dlesm forecasts. 

# imports 
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import os
import logging 
from .variable_metas import variable_metas
logger = logging.getLogger(__name__)
logging.getLogger('matplotlib').setLevel(logging.ERROR)

# Template for param dictionary that can be used to call rmse. Also used for testing the module. 
EXAMPLE_PARAMS = {
    # first param is a list of dictionaries. These contain mnecssary information for calculating the rmse of that forecast:
    # they may alsop contain plotting kwargs if the user wants to plot the rmse and a cache path to save the calculated skill
    'forecast_params': [
        {'file':'tests/analysis/test_rmse/example_forecast1.nc',
         'plot_kwargs': {'label': 'test1', 'color': 'blue', 'linewidth': 2},
         'verification_file': 'tests/analysis/test_rmse/test_era5_hpx32_z500.nc',
        },
        {'file':'tests/analysis/test_rmse/example_forecast2.nc',
         'plot_kwargs': {'label': 'test2', 'color': 'green', 'linewidth': 2},
         'verification_file': 'tests/analysis/test_rmse/test_era5_hpx32_z500.nc',
         'rmse_cache': 'tests/analysis/test_rmse/example_forecast2_RMSE-z500.nc',
        },
    ],
    # name of the cariable to be used in the rmse calculation
    'variable': 'z500',
    # path to save the plot of the rmse
    'plot_file': 'tests/analysis/test_rmse/rmse_plot.png',
    'xlim': {'left':0, 'right':2},
    # whether to return calculated rmse 
    'return_rmse': False,
}

def rmse(
        forecast_params,
        variable,
        plot_file=None,
        xlim=None,
        return_rmse=False,
): 
    """
    Calculates the Root Mean Square Error (RMSE) for a set of forecasts and plots the results.

    Parameters:
    forecast_params (list of dict): A list of dictionaries with instrcutions for handling each forecast. Each may contain: 
        'file' (str): The file path to the forecast data.
        'verification_file' (str): The file path to the verification data to be used for that forecast.
        'rmse_cache' (str, optional): The file path where the calculated RMSE should be cached. If None, the RMSE is not cached.
        'plot_kwargs' (dict, optional): The keyword arguments to be passed to the plot function for that forecast.
    variable (str): The variable for which the RMSE is to be calculated.
    plot_file (str, optional): The file path where the plot should be saved. If None, the plot is not saved. Defaults to None.
    xlim (dict, optional): The x-axis limits for the plot. Defaults to None.
    return_rmse (bool, optional): If True, the function returns the calculated RMSE. Defaults to False.

    Returns:
    list of xarray.DataArray: A list of RMSE values for each forecast. Only returned if return_rmse is True.

    """
    rmse = []
    # iterate through forecats and obtain rmse 
    for forecast_param in forecast_params:
        if os.path.isfile(forecast_param.get('rmse_cache','')):
            logger.info(f"Loading RMSE from {forecast_param['rmse_cache']}.")
            rmse.append(xr.open_dataarray(forecast_param['rmse_cache']))
        else: 
            logger.info(f"Calculating RMSE for {forecast_param['file']}.")
        
            # open forecast and verification data
            forecast = xr.open_dataset(forecast_param['file'])[variable]
            verif_raw = xr.open_dataset(forecast_param['verification_file'])[variable]

            # align verification data with forecast data
            verif = xr.full_like(forecast, fill_value=np.nan)
            for time in forecast.time:
                for step in forecast.step:
                    verif.loc[{'time': time, 'step': step}] = verif_raw.sel(sample=time + step).values

            # calculate rmse 
            rmse.append(np.sqrt(((forecast - verif) ** 2).mean(dim=['time', 'face', 'height','width'])))

            # cache rmse if requested
            if forecast_param.get('rmse_cache',None) is not None:
                logger.info(f"Caching RMSE to {forecast_param['rmse_cache']}.")
                rmse[-1].to_netcdf(forecast_param['rmse_cache'])

    # plot RMSE if requested
    if plot_file is not None:
        fig, ax = plt.subplots()
        for skill, plot_kwargs in zip(rmse, [forecast_param['plot_kwargs'] for forecast_param in forecast_params]):
            ax.plot([s / np.timedelta64(1, 'D') for s in skill.step.values], # plot in days 
                    skill * variable_metas[variable]['scale_factor'], # scale to physical units
                    **plot_kwargs) # style curve and label

        # style plot
        # ax.set_xlabel('Forecast Days')
        ax.set_xlabel('Forecast lead time [days]')
        ax.set_ylabel(f'RMSE [{variable_metas[variable]["units"]}]')
        # calculate y_max for plot
        y_max = max(max(arr.values.flatten()) for arr in rmse) * variable_metas[variable]['scale_factor'] * 1.1
        ax.grid()
        ax.legend()
        ax.set_xlim(**{'left':0, 'right':max([t.step[-1].values / np.timedelta64(1, 'D') for t in rmse])} if xlim is None else xlim)
        ax.set_xticks(np.arange(0, ax.get_xlim()[1], 2))
        ax.set_ylim(bottom=0, top=y_max)
        logger.info(f"Saving plot to {plot_file}.")
        fig.savefig(plot_file,dpi=200)

    # return rmse if requested
    if return_rmse:
        return rmse
    else:
        return