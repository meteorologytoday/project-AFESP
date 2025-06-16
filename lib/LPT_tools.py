import numpy as np
import xarray as xr
from scipy.ndimage import label, generate_binary_structure
from scipy import spatial
from scipy import signal

def genGaussianKernel(half_Nx, half_Ny, dx, dy, sig_x, sig_y):

    Nx = 2 * half_Nx + 1
    Ny = 2 * half_Ny + 1
    
    x = np.arange(Nx) * dx
    y = np.arange(Nx) * dy

    x -= x[half_Nx]
    y -= y[half_Ny]
    
    yy, xx = np.meshgrid(y, x, indexing='ij')
    
    w = np.exp( - ( ( xx / sig_x )**2 + ( yy / sig_y )**2 / 2 ) )

    w /= np.sum(w)
    
    return w

def doGaussianFilter(image, half_Nx, half_Ny, dx, dy, sig_x, sig_y):
    kernel = genGaussianKernel(half_Nx, half_Ny, dx, dy, sig_x, sig_y)
    return signal.convolve2d(image, kernel, mode="full", fillvalue=0)


r_earth = 6.371e6

def getDistOnSphere(lat1, lon1, lat2, lon2, r=1.0):

    _lat1 = np.deg2rad(lat1)
    _lat2 = np.deg2rad(lat2)
    
    _lon1 = np.deg2rad(lon1)
    _lon2 = np.deg2rad(lon2)

    cosine = (
        np.cos(_lat1) * np.cos(_lat2) * np.cos(_lon1 - _lon2)
        + np.sin(_lat1) * np.sin(_lat2)
    )

    arc = np.arccos(cosine)

    return r * arc
    
    


"""
    `pts` must have the shape of (npts, dim), where dim=2 in AR detection
    
    This algorithm is copied from 
    https://stackoverflow.com/questions/50468643/finding-two-most-far-away-points-in-plot-with-many-points-in-python
"""
def getTheFarthestPtsOnSphere(pts):

    # Looking for the most distant points
    # two points which are fruthest apart will occur as vertices of the convex hulil

    try:
        candidates = pts[spatial.ConvexHull(pts).vertices, :]
    except Exception as e:
        print("Something happen with QhHull: ", str(e))

        candidates = pts

    # get distances between each pair of candidate points
    # dist_mat = spatial.distance_matrix(candidates, candidates)

    dist_mat = np.zeros((len(candidates), len(candidates)))

    for i in range(len(candidates)):
        for j in range(len(candidates)):

            if i >= j:
                dist_mat[i, j] = 0.0
                continue

            dist_mat[i, j] = getDistOnSphere(candidates[i, 0], candidates[i, 1], candidates[j, 0], candidates[j, 1], r=r_earth)

    # get indices of candidates that are furthest apart
    i, j = np.unravel_index(dist_mat.argmax(), dist_mat.shape)
            
    farthest_pair = ( candidates[i, :], candidates[j, :] )

    return farthest_pair, dist_mat[i, j]


def detectLPOs(precip, coord_lat, coord_lon, area, threshold, weight=None, filter_func=None):
 
    # 1. Generate object maps
    # 2. Compute objects' characteristics
       
    binary_map = np.zeros(precip.shape, dtype=int)
    binary_map[precip >= threshold] = 1    
   
    # Using the default connectedness: four sides
    labeled_array, num_features = label(binary_map)

    LPOs = []

    renumbered_feature = 1
    for feature_n in range(1, num_features+1): # numbering starts at 1 
        
        idx = labeled_array == feature_n
        covered_area = area[idx]
        sum_covered_area = np.sum(covered_area)
        
        Npts = np.sum(idx)
        pts = np.zeros((np.sum(idx), 2))
        pts[:, 0] = coord_lat[idx]
        pts[:, 1] = coord_lon[idx]
        
            
        #farthest_pair, farthest_dist = getTheFarthestPtsOnSphere(pts)

        if weight is None:
            _wgt = covered_area

        else:
            _wgt = covered_area * weight[idx]

        _sum_wgt = np.sum(_wgt)
        
        centroid = (
            np.sum(coord_lat[idx] * _wgt) / _sum_wgt,
            np.sum(coord_lon[idx] * _wgt) / _sum_wgt,
        )

 

        labeled_array[labeled_array == feature_n] = renumbered_feature
        LPO = dict(
            feature_n     = renumbered_feature,
            area          = sum_covered_area,
            centroid_lat  = centroid[0],
            centroid_lon  = centroid[1],
            #length        = farthest_dist,
            #farthest_pair = farthest_pair,
        )

        if (filter_func is not None) and (filter_func(LPO) is False):
            labeled_array[labeled_array == feature_n] = 0.0
            continue 

        LPOs.append(LPO)
        renumbered_feature += 1
         
    
    return labeled_array, LPOs


def basicLPOFilter(LPO):

    result = True

    if LPO['area'] < (100e3)**2:
        result = False
   
    if np.abs(LPO['centroid_lat']) > 10:
        result = False

    return result

if __name__  == "__main__" :
    
    import ERA5_tools
    import pandas as pd

    cent_dt = pd.Timestamp("2015-12-21")
    day = pd.Timedelta(days=1)
    dts = pd.date_range(cent_dt - day, cent_dt + day, freq="D", inclusive="both")
    ds = ERA5_tools.open_dataset("total_precipitation", dts)
    print(ds)


    ds = ds.mean(dim="valid_time")
   
    ds = ds.where(np.abs(ds.coords["latitude"]) < 20, drop=True)
 
    lat = ds.coords["latitude"].to_numpy() 
    lon = ds.coords["longitude"].to_numpy()  % 360
  
    llat, llon = np.meshgrid(lat, lon, indexing='ij')

    dlat = lat[0] - lat[1]
    dlon = lon[1] - lon[0]

    dlat_rad = np.deg2rad(dlat)
    dlon_rad = np.deg2rad(dlon)


    R_earth = r_earth
 
    area = R_earth**2 * np.cos(np.deg2rad(llat)) * dlon_rad * dlat_rad

    precip = ds["tp"].to_numpy() * 1e3 * 24
    print("Shape of precip: ", precip.shape)

    print("Compute LPO")

    detect_results = dict()
    half_Nlon = 50
    half_Nlat = 50
    for radius in [0.01, 2.5, 5]:
        print("Doing Radius: ", radius) 
        kernel = genGaussianKernel(half_Nlon, half_Nlat, dlon, dlat, radius, radius)
        #print(kernel)
        #print(np.max(kernel))
        #print(np.sum(kernel))
        precip_filtered = signal.convolve2d(precip, kernel, mode="same", fillvalue=0)
        detect_results["LPOBasic_%d" % radius] = dict(
            result = detectLPOs(precip_filtered, llat, llon, area, threshold=12.0, weight=None, filter_func = basicLPOFilter),
            data = precip_filtered,
        )

        print("Shape of filtered: ", precip_filtered.shape)
    
    
    
    print("Loading matplotlib") 
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.patches import Rectangle
    import matplotlib.transforms as transforms
    from matplotlib.dates import DateFormatter
    import matplotlib.ticker as mticker
    import cartopy.crs as ccrs
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

    import cmocean as cmo
    print("done")

    cent_lon = 180.0

    plot_lon_l = -180.0
    plot_lon_r = 180.0
    plot_lat_b = -20.0
    plot_lat_t = 20.0

    proj = ccrs.PlateCarree(central_longitude=cent_lon)
    transform = ccrs.PlateCarree()

    fig, ax = plt.subplots(
        len(list(detect_results.keys())), 1,
        figsize=(12, 8),
        subplot_kw=dict(projection=proj),
        gridspec_kw=dict(hspace=0.15, wspace=0.2),
        constrained_layout=False,
        squeeze=False,
    )


    for i, key in enumerate(detect_results.keys()):

        result = detect_results[key]

        print("Plotting :", key)
        
        _labeled_array = result["result"][0]
        _objs          = result["result"][1]

        _data = result["data"]

        _labeled_array = _labeled_array.astype(float)
        _labeled_array[_labeled_array != 0] = 1.0

        _ax = ax[i, 0]

        _ax.set_title(key)
        
        levs = np.linspace(0, 16, 9)
        cmap = "cmo.rain"

        _data[_data < 0.05] = np.nan
        mappable = _ax.contourf(lon, lat, _data, levels=levs, cmap=cmap,  transform=transform, extend="max")
        cax = plt.colorbar(mappable, ax=_ax, orientation="vertical")
        cax.set_label("[ mm / day ]")
        _ax.contour(lon, lat, _labeled_array, levels=[0.5,], colors='yellow',  transform=transform, zorder=98, linewidth=1)
        
        for i, _obj in enumerate(_objs):
            cent_lat = _obj["centroid_lat"]
            cent_lon = _obj["centroid_lon"]
            _ax.text(cent_lon, cent_lat, "%d" % (i+1), va="center", ha="center", color="cyan", transform=transform, zorder=100)

        _ax.set_global()
        _ax.coastlines()
        _ax.set_extent([plot_lon_l, plot_lon_r, plot_lat_b, plot_lat_t], crs=transform)

        gl = _ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=1, color='gray', alpha=0.5, linestyle='--')

        gl.xlabels_top   = False
        gl.ylabels_right = False

        #gl.xlocator = mticker.FixedLocator(np.arange(-180, 181, 30))
        #gl.xlocator = mticker.FixedLocator([120, 150, 180, -150, -120])#np.arange(-180, 181, 30))
        #gl.ylocator = mticker.FixedLocator([10, 20, 30, 40, 50])

        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.xlabel_style = {'size': 10, 'color': 'black'}
        gl.ylabel_style = {'size': 10, 'color': 'black'}

    fig.suptitle("Central date: %s" % (cent_dt.strftime("%Y/%m/%d"),))
    plt.show()


