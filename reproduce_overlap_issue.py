import xarray as xr
import glob
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

# Simulation of the issue
files = [
    '/data/reloclim/normal/CMIP6_STREAM/paper1-cmip-data/ua_regrid/ua_Amon_BCC-CSM2-MR_ssp585_r1i1p1f1_gn_205501-209412_regridded.nc',
    '/data/reloclim/normal/CMIP6_STREAM/paper1-cmip-data/ua_regrid/ua_Amon_BCC-CSM2-MR_ssp585_r1i1p1f1_gn_20550116-20970916_regridded.nc'
]

print(f"Attempting to open {len(files)} overlapping files...")

try:
    ds = xr.open_mfdataset(files, combine='nested', concat_dim='time', use_cftime=True, data_vars='minimal', coords='minimal', compat='override')
    print("Dataset opened successfully.")
    print("Time coordinate size:", ds.time.size)
    print("Is time monotonic?", ds.time.to_index().is_monotonic_increasing)
    
    # Try selection
    try:
        partial = ds.sel(time=slice('2060-01-01', '2070-12-31'))
        print("Selection 2060-2070 successful. Size:", partial.time.size)
    except Exception as e:
        print(f"Selection failed: {e}")

except Exception as e:
    print(f"Open_mfdataset failed: {e}")
