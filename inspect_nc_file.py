import xarray as xr
import sys

file_path = '/data/reloclim/normal/CMIP6_STREAM/paper1-cmip-data/ua_regrid/ua_Amon_BCC-CSM2-MR_ssp585_r1i1p1f1_gn_205501-209412_regridded.nc'

try:
    ds = xr.open_dataset(file_path, decode_times=False) # Open without decoding first to see raw units
    print("--- RAW DATASET INFO ---")
    print(ds)
    print("\n--- VARIABLES ---")
    print(ds.data_vars)
    print("\n--- COORDINATES ---")
    print(ds.coords)
    
    if 'time' in ds.coords:
        print("\n--- TIME ATTRIBUTES ---")
        print(ds['time'].attrs)
        print(f"Time Range (Raw): {ds['time'].values[0]} to {ds['time'].values[-1]}")

    ds.close()

    print("\n--- DECODING TIMES ---")
    ds = xr.open_dataset(file_path, use_cftime=True)
    print(ds['time'])
    print(f"Time Range (Decoded): {ds['time'].values[0]} to {ds['time'].values[-1]}")
    ds.close()

except Exception as e:
    print(f"Error: {e}")
