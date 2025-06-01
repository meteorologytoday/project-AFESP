import pandas as pd
import xarray as xr
import os
import numpy as np
from pathlib import Path

archive_root = Path("/data/SO2/t2hsu/dataset_A/ERA5-derived-daily-global")

mapping_longname_shortname = {
    'sea_ice_concentration'                  : 'sicon',
    'geopotential'                  : 'z',
    '10m_u_component_of_wind'       : 'u10',
    '10m_v_component_of_wind'       : 'v10',
    'mean_sea_level_pressure'       : 'msl',
    '2m_temperature'                : 't2m',
    'sea_surface_temperature'       : 'sst',
    'specific_humidity'             : 'q',
    'u_component_of_wind'           : 'u',
    'v_component_of_wind'           : 'v',
    'mean_surface_sensible_heat_flux'    : 'msshf',
    'mean_surface_latent_heat_flux'      : 'mslhf',
    'mean_surface_net_long_wave_radiation_flux'  : 'msnlwrf',
    'mean_surface_net_short_wave_radiation_flux' : 'msnswrf',
    "total_precipitation": "tp",
}

def generate_filename(varset, dt, freq, file_prefix="ERA5-derived-daily"):
   
    dt_str = pd.Timestamp(dt).strftime("%Y-%m-%d")

    save_dir = Path(archive_root) / freq / varset
    filename = os.path.join(
        save_dir,
        "{file_prefix:s}-{varset:s}-{time:s}.nc".format(
            file_prefix = file_prefix,
            freq = freq,
            varset = varset,
            time = dt_str,
        )
    )

 
    return save_dir / filename

"""
def open_dataset(varname_longname, dt, freq):
  
    loading_filename = generate_filename(varname_longname, dt, freq) 
    ds = xr.open_dataset(loading_filename)
    
    return ds    
"""

# Find the first value that is True
def findfirst(xs):

    for i, x in enumerate(xs):
        if x:
            return i

    return -1

# Read ERA5 data and by default convert to S2S project
# resolution (1deg x 1deg), and rearrange the axis in
# a positively increasing order.
def open_dataset(dts, freq, varset, if_downscale = True):

    
    if not hasattr(dts, '__iter__'):
        dts = [dts,]
 
   
    filenames = [
        generate_filename(varset, dt, freq) for dt in dts
    ]
    
    ds = xr.open_mfdataset(filenames)
    
    # flip latitude
    ds = ds.isel(latitude=slice(None, None, -1))

    lat = ds.coords["latitude"].to_numpy()
    lon = ds.coords["longitude"].to_numpy()

    # make longitude 0 the first element
    first_positive_idx = findfirst( lon >= 0 )
    if first_positive_idx != -1:
        roll_by = - first_positive_idx
        ds = ds.roll(longitude=roll_by).assign_coords(
            coords = {
                "longitude" : np.roll(
                    ds.coords["longitude"].to_numpy() % 360,
                    roll_by,
                )
            }
        )

    ds = ds.rename({"valid_time" : "time"})

    return ds



