# requirements: 
#     xarray
#     xesmf
#     dask

# This is a lightweight reference for incorporating reprojection into your own workflows.
# You can easily adapt the code to match your specific data and target grids.

import xarray as xr
import xesmf as xe
from dask.diagnostics import ProgressBar

# Step 1: Load dataset with **coarse Dask chunking**
era5_data = xr.open_dataset("datasets/era5.nc",
    chunks={"time": 10, "latitude": 300, "longitude": 300}  # Bigger = fewer chunks
)

# Step 2: Load target grid (static, no need for chunking)
vhr_q1_path = "datasets/vhr_q1.nc"
vhr_data1 = xr.open_dataset(vhr_q1_path, mask_and_scale=True)
vhr_q2_path = "datasets/vhr_q2.nc"
vhr_data2 = xr.open_dataset(vhr_q2_path, mask_and_scale=True)

target_grid = xr.concat([vhr_data1, vhr_data2], dim="time")

# Step 3: Create regridder with cached weights (FAST)
regridder = xe.Regridder(
    era5_data,
    target_grid,
    method="bilinear",
    periodic=True,
    filename="weights.nc",     # Saves weights for reuse
    reuse_weights=False
)

# Step 4: Regrid and rechunk output (prevents chunk explosion!)
output = regridder(era5_data).chunk({
    "valid_time": 10,
    "rlat": 100,
    "rlon": 100
})

# Step 5: Write output lazily, then compute with progress bar
delayed = output.to_netcdf("datasets/regridded_era5.nc", compute=False)

with ProgressBar():
    delayed.compute()
