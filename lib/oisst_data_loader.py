import xarray as xr
import pandas as pd
import os
from pathlib import Path

data_archive = "data"

def getFilenameFromDatetime(dataset, dt, root="."):
    
    full_filename = Path(root) / dataset / "{dataset:s}_{datetime:s}.nc".format(
            dataset = dataset,
            datetime = dt.strftime("%Y-%m-%d"),
    )

    return full_filename 
 

def load_dataset(dataset, dt_beg, dt_end, inclusive="left", root="."):

    if dt_end < dt_beg:
        raise Exception("dt_end = %s should be later than dt_beg = %s" % ( str(dt_beg), str(dt_end), ))
    
    filenames = [
        getFilenameFromDatetime(dataset, dt, root = root)
        for dt in pd.date_range(dt_beg, dt_end, inclusive=inclusive)
    ]
    
    print("Open dataset using xr.open_mfdataset...")
    ds = xr.open_mfdataset(filenames)
    
    return ds
