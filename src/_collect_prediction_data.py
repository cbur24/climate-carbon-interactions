import os
import xarray as xr
import numpy as np
from odc.geo.xr import assign_crs


def allNaN_arg(da, dim, stat, idx=True):
    """
    Calculate da.argmax() or da.argmin() while handling
    all-NaN slices. Fills all-NaN locations with an
    float and then masks the offending cells.

    Parameters
    ----------
    da : xarray.DataArray
    dim : str
        Dimension over which to calculate argmax, argmin e.g. 'time'
    stat : str
        The statistic to calculte, either 'min' for argmin()
        or 'max' for .argmax()
    idx : bool
        If True then use da.idxmax() or da.idxmin(), otherwise
        use ds.argmax() or ds.argmin()

    Returns
    -------
    xarray.DataArray
    """
    # generate a mask where entire axis along dimension is NaN
    mask = da.isnull().all(dim)

    if stat == "max":
        y = da.fillna(float(da.min() - 1))
        if idx==True:
            y = y.idxmax(dim=dim, skipna=True).where(~mask)
        else:
            y = y.argmax(dim=dim, skipna=True).where(~mask)
        return y

    if stat == "min":
        y = da.fillna(float(da.max() + 1))
        if idx==True:
            y = y.idxmin(dim=dim, skipna=True).where(~mask)
        else:
            y = y.argmin(dim=dim, skipna=True).where(~mask)
        return y

def round_coords(ds):
    """
    Due to precision of float64 on coordinates, coordinates
    don't quite match after reprojection, resulting in adding spurious
    pixels after merge. Converting to float32 rounds coords so they match.
    """
    ds['latitude'] = ds.latitude.astype('float32')
    ds['longitude'] = ds.longitude.astype('float32')
    
    ds['latitude'] = np.array([round(i,4) for i in ds.latitude.values])
    ds['longitude'] = np.array([round(i,4) for i in ds.longitude.values])
    
    return ds
    

def collect_prediction_data(time_start,
                            time_end,
                            chunks=dict(latitude=1000, longitude=1000, time=1),
                            export=False,
                            verbose=True
                           ):
  
    # Grab a list of all datasets in the folder
    base='/g/data/os22/chad_tmp/climate-carbon-interactions/data/5km/' 
    covariables = [base+i for i in os.listdir(base) if i.endswith('.nc')]
    covariables.sort()

    #loop through datasets and append
    dss=[]
    for var in covariables:
        if verbose:
            print('Extracting', var.replace(base, ''))

        ds = assign_crs(xr.open_dataset(var, chunks=chunks), crs='EPSG:4326')
        ds = ds.sel(time=slice(time_start, time_end))
        ds = round_coords(ds)
        dss.append(ds)
    
    #merge all datasets together
    if verbose:
        print('   Merge datasets')
    
    data = xr.merge(dss, compat='override')

    # format
    data = data.rename({'latitude':'y', 'longitude':'x'}) #this helps with predict_xr
    data = data.astype('float32') #make sure all data is in float32
    data = assign_crs(data, crs='epsg:4326')

    if export:
        if verbose:
            print('   Exporting netcdf')
        data.compute().to_netcdf(export+'/prediction_data_'+time_start+'_'+time_end+'.nc')
    
    return data
