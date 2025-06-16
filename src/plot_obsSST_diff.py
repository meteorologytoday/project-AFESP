from multiprocessing import Pool
import multiprocessing
from pathlib import Path
import traceback
import argparse

#import cdsapi
import numpy as np
import pandas as pd
import xarray as xr

import regrid_tools
import oisst_data_loader



thumbnail_numbering = "abcdefghijklmn"


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--datasets', type=str, nargs="+", help='Test datasets. The first one will be the reference.', required=True)
    parser.add_argument('--archive-root', type=str, help='Input directories.', required=True)
    parser.add_argument('--beg-date', type=str, required=True)
    parser.add_argument('--end-date', type=str, required=True)
    parser.add_argument('--smooth-deg', type=float, default=1.0)
    parser.add_argument('--half-window-size', type=int, default=5)
    parser.add_argument('--output-root', type=str, required=True)
    parser.add_argument('--no-display', action="store_true", help='No display GUI.',)
    parser.add_argument('--plot-lat-rng', type=float, nargs=2, help='Plot range of latitude', default=[-90, 90])
    parser.add_argument('--plot-lon-rng', type=float, nargs=2, help='Plot range of latitude', default=[0, 360])

    args = parser.parse_args()

    print(args)

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
    from scipy.stats import ttest_ind_from_stats
    print("done")
    
    plot_infos = dict(

        sst = dict(
            diff = dict(
                shading_levels = np.linspace(-1, 1, 21) * 2,
                contour_levels = np.linspace(0, 1, 5) * 1,
                cmap = cmo.cm.balance,
            ),

            abs = dict(
                shading_levels = np.arange(-2, 36, 1),
                cmap = "gnuplot",
            ),

            factor = 1,
            origin = 273.15,
            label = "SST",
            unit  = "$ \\mathrm{K} $",
        ),

    )

  
    beg_date = pd.Timestamp(args.beg_date)
    end_date = pd.Timestamp(args.end_date)

    print("Beg date: ", beg_date)
    print("End date: ", end_date)

    for dt in pd.date_range(beg_date, end_date, freq="D", inclusive="both"):
    
        print("Doing date: ", dt)

        output_dir = Path(args.output_root) / "-".join(args.datasets)
        output_file = output_dir / ("%s.png" % (dt.strftime("%Y-%m-%d"),))
        
        if output_file.exists():
            print("Output file %s already exists. Skip it." % (str(output_file),))

        else:
            
            output_dir.mkdir(exist_ok=True, parents=True)

            data = dict()
            da_ref = None
            for i, dataset in enumerate(args.datasets):
            
                print("Loading dataset: ", dataset)
                da = oisst_data_loader.load_dataset(
                    dataset,
                    dt,
                    dt,
                    inclusive="both",
                    root=args.archive_root,
                )["sst"].isel(time=0)
                
                _coords = da.coords

                _lat = _coords["lat"].to_numpy()
                _lon = _coords["lon"].to_numpy()

                dlat = _lat[1] - _lat[0]
                dlon = _lon[1] - _lon[0]

                image = da.to_numpy()
                print("ORIGINAL SHAPE : ", image.shape)
                image = regrid_tools.doGaussianFilter(
                    image,
                    half_Nx = args.half_window_size,
                    half_Ny = args.half_window_size,
                    dx = dlon,
                    dy = dlat,
                    sig_x = args.smooth_deg,
                    sig_y = args.smooth_deg,
                    periodic_lon=False,
                )
                print("SMOOTHED SHAPE : ", image.shape)

            
                da.data[:] = image

                if i == 0:
                    da_ref = da
                
                else:

                    print("Not reference data. Interpolate now.")       
                    da = da.interp(
                        coords=dict(
                            lat = da_ref.coords["lat"],
                            lon = da_ref.coords["lon"],
                        ),
                    )


                da = da.where(
                    ( da.coords["lat"] >= args.plot_lat_rng[0] )
                    & ( da.coords["lat"] <= args.plot_lat_rng[1] )
                    & ( da.coords["lon"] >= args.plot_lon_rng[0] )
                    & ( da.coords["lon"] <= args.plot_lon_rng[1] )
                , drop=True)


                data[dataset] = da

            print("Plotting...")



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

            ncol = len(args.datasets)
            nrow = 1

            h = 5
            w_over_h = (plot_lon_r - plot_lon_l) / (plot_lat_t - plot_lat_b)
            font_size_factor = 2.0

            w = h * w_over_h


            figsize, gridspec_kw = tool_fig_config.calFigParams(
                w = w,
                h = h,
                wspace = 3.0,
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


            fig.suptitle("Date: %s" % (dt.strftime("%Y-%m-%d"),))

            plot_info = plot_infos["sst"]

            ax_flatten = ax.flatten()

            for i, dataset in enumerate(args.datasets):
                
                is_ref = i == 0
                _ax = ax_flatten[i] 

                ref_da = data[args.datasets[0]]            
                da = data[dataset]
                coords = ref_da.coords

                if is_ref:
                    
                    _shading = da.to_numpy() - plot_info["origin"]

                    mappable = _ax.contourf(
                        coords["lon"], coords["lat"],
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
                    _ax.set_title("(%s) %s" % (thumbnail_numbering[i], dataset,))

                else:
                    
                    # diff = ECCC - ERA5
                    _shading = da.to_numpy() - ref_da.to_numpy()
                    mappable = _ax.contourf(
                        coords["lon"], coords["lat"],
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


                    _ax.set_title("(%s) %s - %s" % (thumbnail_numbering[i], dataset, args.datasets[0]))

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

            print("Saving output: ", output_file) 
            fig.savefig(output_file, dpi=200)
        
            print("Finished.")

