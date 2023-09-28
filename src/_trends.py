import numpy as np
import xarray as xr
import dask.array as da
from dask.delayed import delayed
import pymannkendall as mk
from  scipy import stats

def _calc_slope(ds):
    """return linear regression statistical variables"""
    mask = np.isfinite(ds)
    x = np.arange(len(ds))
    # return stats.linregress(x[mask], y[mask])
    return stats.mstats.theilslopes(x[mask], y[mask]) 

# regression function defition
def regression(ds):
    """apply linear regression function along time axis"""
    axis_num = ds.get_axis_num('time')
    return da.apply_along_axis(_calc_slope, axis_num, ds)

def linregress(ds):

    # fill pixels that are all-NaNs
    allnans = ds.isnull().all('time')
    ds = ds.where(~allnans, other=0)

    # regression analysis
    delayed_objs = delayed(regression)(ds).persist()

    # transforms dask.delayed to dask.array
    results = da.from_delayed(delayed_objs, shape=(5,
                    ds.shape[1:][0], ds.shape[1:][1]), dtype=np.float32)
    #results = results.compute()
    # results = results.compute() #need this twice haven't figured out why

    # coordination definition
    coords = {'y': ds.y, 'x': ds.x}

    # output xarray.Dataset definition
    # ds_out = xr.Dataset(
    #     data_vars=dict(slope=(["y", "x"], results[0]),
    #                    intercept=(["y", "x"], results[1]),
    #                    r_value=(["y", "x"], results[2]),
    #                    p_value=(["y", "x"], results[3]),
    #                    std_err=(["y", "x"], results[4]),
    #                   ),
    #     coords = coords)
    ds_out = xr.Dataset(
        data_vars=dict(slope=(["y", "x"], results.slope),
                       intercept=(["y", "x"], results.intercept),
                       Tau=(["y", "x"], results.Tau),
                       p_value=(["y", "x"], results.p),
                       z=(["y", "x"], results.z),
                      ),
        coords = coords)

    #remask all-NaN pixel
    return ds_out.where(~allnans)