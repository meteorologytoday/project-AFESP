from multiprocessing import Pool
import multiprocessing
from pathlib import Path
import traceback
import argparse

#import cdsapi
import numpy as np
import pandas as pd
import xarray as xr

import S2S_tools
import ERA5_tools
import ens_tools
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--archive-root', type=str, help='Input directories.', required=True)
    parser.add_argument('--origin', type=str, help='Input directories.', required=True)
    parser.add_argument('--model-version', type=str)
    parser.add_argument('--start-time', type=str)
    parser.add_argument('--lead-day', type=int)
    parser.add_argument('--output', type=str, default="")
    parser.add_argument('--nwp-type', type=str, help='Type of NWP. Valid options: `forecast`, `hindcast`.', required=True, choices=["forecast", "hindcast"])
    parser.add_argument('--no-display', action="store_true", help='No display GUI.',)
    parser.add_argument('--plot-lat-rng', type=float, nargs=2, help='Plot range of latitude', default=[-90, 90])
    parser.add_argument('--plot-lon-rng', type=float, nargs=2, help='Plot range of latitude', default=[0, 360])
    parser.add_argument('--ens-range', type=str, help='Plot range of latitude', required=True)

    args = parser.parse_args()

    print(args)
    
    start_time = pd.Timestamp(args.start_time)
    lead_time = pd.Timedelta(days=args.lead_day)

    ens_range = ens_tools.parseRanges(args.ens_range) 

    ds = S2S_tools.essentials.open_dataset(
        origin = args.origin,
        model_version = args.model_version,
        nwp_type = args.nwp_type,
        varset = "surf_avg",
        start_time = start_time,
        numbers = ens_range,
        root = args.archive_root,
    )["sst"].isel(start_time=0).sel(lead_time=lead_time + pd.Timedelta(hours=12))
   

    ds_mean = ds.mean(dim="number")
    ds_stderr = ds.std(dim="number") #/ (len(ds.coords["number"])**0.5)
 
           
    ds_ERA5 = ERA5_tools.open_dataset(
        start_time + lead_time,
        "6_hourly",
        "sea_surface_temperature",
    )["sst"].isel(time=0)

    print("ERA5 loaded. Need to interpolate.")

    ds_ERA5 = ds_ERA5.interp(
        coords=dict(
            latitude  = ds.coords["latitude"],
            longitude = ds.coords["longitude"],
        ),
    )

    print(ds)
    print(ds_ERA5)

    print("Loading matplotlib...")

    import matplotlib as mplt
    if args.no_display:
        print("Use Agg")
        mplt.use("Agg")
    else:
        print("Use TkAgg")
        mplt.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import matplotlib.transforms as transforms
    from matplotlib.dates import DateFormatter
    import matplotlib.ticker as mticker
    from matplotlib import rcParams



    from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
    import cartopy.crs as ccrs
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

    rcParams['contour.negative_linestyle'] = 'dashed'

    import cmocean as cmo
    import tool_fig_config
    print("done")

    from scipy.stats import ttest_ind_from_stats

    plot_infos = dict(

        sst = dict(
            diff = dict(
                shading_levels = np.linspace(-1, 1, 21) * 1,
                contour_levels = np.linspace(0, 1, 5) * 1,
                cmap = cmo.cm.balance,
            ),

            abs = dict(
                shading_levels = np.arange(-2, 36, 2),
                cmap = "gnuplot",
            ),

            factor = 1,
            origin = 273.15,
            label = "SST",
            unit  = "$ \\mathrm{K} $",
        ),

    )



    title_font_size = 18


    plot_lon_l = args.plot_lon_rng[0] % 360.0
    plot_lon_r = args.plot_lon_rng[1] % 360.0
    plot_lat_b = args.plot_lat_rng[0]
    plot_lat_t = args.plot_lat_rng[1]

    if plot_lon_r == 0.0:
        plot_lon_r = 360.0 # exception

    # rotate
    if plot_lon_l > plot_lon_r:
        plot_lon_l -= 360.0
        
    cent_lon = (plot_lon_l + plot_lon_r) / 2
    map_projection = ccrs.PlateCarree(central_longitude=cent_lon)
    map_transform = ccrs.PlateCarree()

    ncol = 3
    nrow = 1

    h = 5
    w_over_h = (plot_lon_r - plot_lon_l) / (plot_lat_t - plot_lat_b)
    font_size_factor = 2.0

    w = h * w_over_h


    figsize, gridspec_kw = tool_fig_config.calFigParams(
        w = w,
        h = h,
        wspace = 1.0,
        hspace = 0.5,
        w_left = 1.0,
        w_right = 2.5,
        h_bottom = 1.0,
        h_top = 1.0,
        ncol = ncol,
        nrow = nrow,
    )


    fig, ax = plt.subplots(
        nrow, ncol,
        figsize=figsize,
        subplot_kw=dict(projection=map_projection, aspect="auto"),
        gridspec_kw=gridspec_kw,
        constrained_layout=False,
        squeeze=False,
    )


    fig.suptitle("Start time: %s, lead day: %d" % (start_time.strftime("%Y-%m-%d"), args.lead_day,))

    coords = ds.coords

    plot_info = plot_infos["sst"]

    ax_flatten = ax.flatten()

    # ERA5
    _ax = ax_flatten[0]
    _shading = ds_ERA5.to_numpy() - plot_info["origin"]
    mappable = _ax.contourf(
        coords["longitude"], coords["latitude"],
        _shading,
        levels=plot_info["abs"]["shading_levels"],
        cmap=plot_info["abs"]["cmap"], 
        extend="both", 
        transform=map_transform,
    )

    cax = tool_fig_config.addAxesNextToAxes(fig, _ax, "right", thickness=0.3, spacing=0.3, flag_ratio_thickness=False, flag_ratio_spacing=False)
    cb = plt.colorbar(mappable, cax=cax, orientation="vertical", pad=0.00)
    cb.ax.tick_params(axis='both', labelsize=15 * font_size_factor)
    unit_str = "" if plot_info["unit"] == "" else " [ %s ]" % (plot_info["unit"],)
    cb.ax.set_ylabel(unit_str, size=25 * font_size_factor)

    # ECCC
    _ax = ax_flatten[1]
    _shading = ds_mean.to_numpy() - plot_info["origin"]
    mappable = _ax.contourf(
        coords["longitude"], coords["latitude"],
        _shading,
        levels=plot_info["abs"]["shading_levels"],
        cmap=plot_info["abs"]["cmap"], 
        extend="both", 
        transform=map_transform,
    )
    cax = tool_fig_config.addAxesNextToAxes(fig, _ax, "right", thickness=0.3, spacing=0.3, flag_ratio_thickness=False, flag_ratio_spacing=False)
    cb = plt.colorbar(mappable, cax=cax, orientation="vertical", pad=0.00)
    cb.ax.tick_params(axis='both', labelsize=15 * font_size_factor)
    unit_str = "" if plot_info["unit"] == "" else " [ %s ]" % (plot_info["unit"],)
    cb.ax.set_ylabel(unit_str, size=25 * font_size_factor)



    # diff = ECCC - ERA5
    _ax = ax_flatten[2]
    _shading = ds_mean.to_numpy() - ds_ERA5.to_numpy()
    mappable = _ax.contourf(
        coords["longitude"], coords["latitude"],
        _shading,
        levels=plot_info["diff"]["shading_levels"],
        cmap=plot_info["diff"]["cmap"], 
        extend="both", 
        transform=map_transform,
    )
    cax = tool_fig_config.addAxesNextToAxes(fig, _ax, "right", thickness=0.3, spacing=0.3, flag_ratio_thickness=False, flag_ratio_spacing=False)
    cb = plt.colorbar(mappable, cax=cax, orientation="vertical", pad=0.00)
    cb.ax.tick_params(axis='both', labelsize=15 * font_size_factor)
    unit_str = "" if plot_info["unit"] == "" else " [ %s ]" % (plot_info["unit"],)
    cb.ax.set_ylabel(unit_str, size=25 * font_size_factor)


    # Plot the hatch to denote significant data
    _dot = np.zeros_like(_shading)
    #_dot[:] = np.nan

    _significant_idx =  np.abs(ds_mean.to_numpy() - ds_ERA5.to_numpy()) > ds_stderr.to_numpy() 
    _dot[ _significant_idx                 ] = 0.75
    _dot[ np.logical_not(_significant_idx) ] = 0.25
    cs = _ax.contourf(coords["longitude"], coords["latitude"], _dot, colors='none', levels=[0, 0.5, 1], hatches=[None, "..."], transform=map_transform)

    # Remove the contour lines for hatches
    for _, collection in enumerate(cs.collections):
        collection.set_edgecolor((.2, .2, .2))
        collection.set_linewidth(0.)





    ax_flatten[0].set_title("(a) ERA5")
    ax_flatten[1].set_title("(b) %s" % (args.origin,))
    ax_flatten[2].set_title("(c) %s - ERA5" % (args.origin,))

    for __ax in ax_flatten:
            
        __ax.tick_params(axis='both', labelsize=15 * font_size_factor)

        gl = __ax.gridlines(crs=map_transform, draw_labels=True,
                          linewidth=1, color='gray', alpha=0.5, linestyle='--')

        gl.xlabels_top   = False
        gl.ylabels_right = False

        #gl.xlocator = mticker.FixedLocator(np.arange(-180, 181, 30))
        #gl.xlocator = mticker.FixedLocator([120, 150, 180, -150, -120])#np.arange(-180, 181, 30))
        #gl.ylocator = mticker.FixedLocator([10, 20, 30, 40, 50])
        
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.xlabel_style = {'size': 12*font_size_factor, 'color': 'black'}
        gl.ylabel_style = {'size': 12*font_size_factor, 'color': 'black'}

        __ax.set_global()
        #__ax.gridlines()
        __ax.coastlines(color='gray')

        __ax.set_extent([plot_lon_l, plot_lon_r, plot_lat_b, plot_lat_t], crs=map_transform)


    if not args.no_display:
        print("Showing plotted outcome.")
        plt.show()

    if args.output != "":

        print("Saving output: ", args.output)    
        fig.savefig(args.output, dpi=200)
    
    print("Finished.")

