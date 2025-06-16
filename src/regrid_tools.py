import numpy as np
import scipy.sparse
import scipy.signal
import argparse
#import xarray as xr
# It is assumed that if data is provided, then
# nan means missing data.
def genGaussianKernel(half_Nx, half_Ny, dx, dy, sig_x, sig_y, data=None):

    Nx = 2 * half_Nx + 1
    Ny = 2 * half_Ny + 1
    
    x = np.arange(Nx) * dx
    y = np.arange(Nx) * dy

    x -= x[half_Nx]
    y -= y[half_Ny]
    
    yy, xx = np.meshgrid(y, x, indexing='ij')
    
    w = np.exp( - ( ( xx / sig_x )**2 + ( yy / sig_y )**2 / 2 ) )

    if data is not None:
        valid_idx = np.isfinite(data)
        invalid_idx = np.isnan(data)
        w[invalid_idx] = 0.0

    w /= np.sum(w)
    
    return w

def genPeriodicExtendedDataMatrixInX(Ny, Nx, left_size, right_size):

    rrow, ccol = np.meshgrid(range(Ny), range(Nx), indexing="ij")

    rrow_flat = rrow.flatten()
    ccol_flat = ccol.flatten()

    Ny_extended = Ny
    Nx_extended = Nx + left_size + right_size

 
    numbering_original = np.arange(Nx * Ny, dtype=int).reshape((Ny, Nx))

    concat_arrs = []
 
    if left_size > 0:
        concat_arrs.append(numbering_original[:, -left_size:])
    
    concat_arrs.append(numbering_original)
   
    if right_size > 0:
        concat_arrs.append(numbering_original[:, :right_size])

    numbering_extended = np.concatenate(
        concat_arrs,
        axis=1,
    )

    row = np.arange(numbering_extended.size, dtype=int)
    col = numbering_extended.flatten()
    data = np.ones((len(col),))

    # This is the extend matrix    
    extend_P_original = scipy.sparse.bsr_array((data, (row, col)), shape=(numbering_extended.size, numbering_original.size))


    # Now construct the reverse
    numbering_extended = np.arange(Nx_extended * Ny_extended, dtype=int).reshape((Ny_extended, Nx_extended))
    row = np.arange(numbering_original.size)
    col = numbering_extended[:, left_size:left_size+Nx].flatten()
    data = np.ones((len(col),))
    original_P_extend = scipy.sparse.bsr_array((data, (row, col)), shape=(numbering_original.size, numbering_extended.size))
    
    return dict(
        forward_mtx = extend_P_original,
        backward_mtx = original_P_extend,
        extended_shape = numbering_extended.shape
    )


def doGaussianFilter(image, half_Nx, half_Ny, dx, dy, sig_x, sig_y, periodic_lon=False):

    kernel = genGaussianKernel(half_Nx, half_Ny, dx, dy, sig_x, sig_y)
  

    print("IMAGE SHAPE: ", image.shape)
    result = None 
    if periodic_lon:
        
        mtx_info = genPeriodicExtendedDataMatrixInX(*image.shape, half_Nx, half_Nx)
        image_extended = (mtx_info["forward_mtx"] @ image.flatten()).reshape(mtx_info["extended_shape"])
       
        image_extended_smoothed = scipy.signal.convolve2d(image_extended, kernel, mode="same", fillvalue=0)
        
        result = (mtx_info["backward_mtx"] @ image_extended_smoothed.flatten()).reshape(image.shape)
        
    else:
        print("HERE")
        result = scipy.signal.convolve2d(image, kernel, mode="same", fillvalue=0)
    
    print("RESULT SHAPE: ", result.shape)


    return result


if __name__ == "__main__":
   

    parser = argparse.ArgumentParser(description='The dlat and dlon are assumed uniform.')
    parser.add_argument('--test-file', type=str, help='The file that contains missing data, lat and lon information.', required=True)
    parser.add_argument('--output', type=str, help='WRF file that provide XLAT and XLONG.', required=True)
    parser.add_argument('--half-window-size', type=int, nargs=2, help="Size of the convolution window in lon (x) and lat (y) direction.", required=True)
    args = parser.parse_args()

    print(args)

    
    ds = xr.open_dataset(args.test_file)
    
    WRF_llat = WRF_ds.coords["XLAT"].isel(Time=0).to_numpy()
    WRF_llon = WRF_ds.coords["XLONG"].isel(Time=0).to_numpy() % 360.0

    WRF_lat_idx, WRF_lon_idx, lat_regrid_bnds, lon_regrid_bnds = computeBoxIndex(WRF_llat, WRF_llon, args.lat_rng, args.lon_rng, args.dlat, args.dlon)

    PRISM_ds = xr.open_dataset(args.PRISM_file)
    PRISM_lat = PRISM_ds["lat"].to_numpy()
    PRISM_lon = PRISM_ds["lon"].to_numpy() % 360.0
    PRISM_llat, PRISM_llon = np.meshgrid(PRISM_lat, PRISM_lon, indexing='ij')
    PRISM_lat_idx, PRISM_lon_idx, _, _ = computeBoxIndex(PRISM_llat, PRISM_llon, args.lat_rng, args.lon_rng, args.dlat, args.dlon)
   
    lat_regrid = ( lat_regrid_bnds[1:] + lat_regrid_bnds[:-1] ) / 2
    lon_regrid = ( lon_regrid_bnds[1:] + lon_regrid_bnds[:-1] ) / 2
 
    new_ds = xr.Dataset(
        data_vars = dict(
            WRF_lat_idx = (["south_north", "west_east"], WRF_lat_idx),
            WRF_lon_idx = (["south_north", "west_east"], WRF_lon_idx),
            PRISM_lat_idx = (["lat", "lon"], PRISM_lat_idx),
            PRISM_lon_idx = (["lat", "lon"], PRISM_lon_idx),
            lat_regrid_bnd = (["lat_regrid_bnd",], lat_regrid_bnds),
            lon_regrid_bnd = (["lon_regrid_bnd",], lon_regrid_bnds),
            lat_regrid = (["lat_regrid",], lat_regrid),
            lon_regrid = (["lon_regrid",], lon_regrid),
        ),
        coords = dict(
            XLAT = (["south_north", "west_east"], WRF_llat),
            XLONG = (["south_north", "west_east"], WRF_llon),
            lat = (["lat"], PRISM_lat),
            lon = (["lon"], PRISM_lon),
        ),
        attrs = dict(
            nlat_box = max_lat_idx+1,
            nlon_box = max_lon_idx+1,
        )
    )

    print("Output file: %s" % (args.output,))
    new_ds.to_netcdf(args.output)
    
