import numpy as np
import netCDF4
import xarray as xr
import os
import pandas as pd
import scipy
import ERA5_tools



#archive_root = Path("/") / "SO3" / "S2S"
archive_root = Path("/data/SO2/t2hsu/dataset_A/ERA5-derived-daily-global")

ERA5_longshortname_mapping = {
    'sea_ice_cover'                         : 'siconc',
    'geopotential'                          : 'z',
    "surface_pressure"                      : "sp",
    "mean_sea_level_pressure"               : "msl",
    "10m_u_component_of_wind"               :  "u10",
    "10m_v_component_of_wind"               :  "v10",
    "IVT_x"                                 : "IVT_x",
    "IVT_y"                                 : "IVT_y",
    "IVT"                                   : "IVT",
    "IWV"                                   : "IWV",
    "sea_surface_temperature"               : "sst",
    'mean_surface_sensible_heat_flux'       : 'msshf',
    'mean_surface_latent_heat_flux'         : 'mslhf',
    "total_precipitation" : "tp",
}

ERA5_shortlongname_mapping = {
    shortname : longname
    for longname, shortname in ERA5_longshortname_mapping.items()
}

# Find the first value that is True
def findfirst(xs):

    for i, x in enumerate(xs):
        if x:
            return i

    return -1

# Read ERA5 data and by default convert to S2S project
# resolution (1deg x 1deg), and rearrange the axis in
# a positively increasing order.
def open_dataset_ERA5(dts, freq, varset, if_downscale = True):

    
    if not hasattr(dts, '__iter__'):
        dts = [dts,]
 
   
    filenames = [
        ERA5_tools.generate_filename(varset, dt, freq) for dt in dts
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

    return ds



if __name__ == "__main__":

    dt = pd.Timestamp("2011-01-02")
    freq = "inst"
    varname = "mean_sea_level_pressure"
    ds = open_dataset_ERA5(dt, freq, varname)

    varname_short = ERA5_longshortname_mapping[varname]

    print(ds)
   
    print("Data is read.") 
    print("Load matplotlib...")
    import matplotlib as mplt
    mplt.use("TkAgg")

    import matplotlib.pyplot as plt
    print("Done")

    fig, ax = plt.subplots(1, 1)
    
    mappable = ax.contourf(ds.coords["longitude"], ds.coords["latitude"], ds[varname_short].isel(time=0))
    plt.colorbar(mappable, ax = ax, orientation="vertical")
    
    plt.show()
