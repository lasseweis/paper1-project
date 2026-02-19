
import xarray as xr
import glob
import os

# Define the pattern from Config
CMIP6_RAW_PR_PATH_PATTERN = '/data/reloclim/normal/CMIP6_STREAM/paper1-cmip-data/pr_regrid/pr_Amon_{model}_{scenario}_*_*_regridded.nc'

model = "EC-Earth3-Veg-LR"
scenario = "ssp585"

search_pattern = CMIP6_RAW_PR_PATH_PATTERN.format(model=model, scenario=scenario)
found_files = sorted(glob.glob(search_pattern))

print(f"Found {len(found_files)} files.")

# Filter to just a few files around the problematic one to save time
# Problem file: pr_Amon_EC-Earth3-Veg-LR_ssp585_r1i1p1f1_gr_20420116-21001216_regridded.nc
problem_file = [f for f in found_files if "20420116-21001216" in f]
if problem_file:
    print(f"Problem file found: {os.path.basename(problem_file[0])}")
else:
    print("Problem file NOT found in the list!")

# Let's try to load just the problem file first
if problem_file:
    try:
        ds = xr.open_dataset(problem_file[0])
        print("Successfully opened problem file.")
        print(ds)
        ds.close()
    except Exception as e:
        print(f"Failed to open problem file: {e}")

# Now let's try to load a mix of files like the code does
# We'll take 2041, 2042, the problem file, and 2043
subset_files = [f for f in found_files if "2041" in f or "2042" in f or "2043" in f]
print(f"Subset for testing: {[os.path.basename(f) for f in subset_files]}")

try:
    ds = xr.open_mfdataset(subset_files, combine='nested', concat_dim='time', use_cftime=True)
    print("Successfully opened mfdataset.")
    print("Time values:", ds.time.values)
    
    # Try selection
    try:
        sel = ds.sel(time=slice('2041', '2043'))
        print("Successfully selected time slice 2041-2043")
        print(sel.time.values)
    except Exception as e:
        print(f"Failed selection: {e}")
        
    ds.close()
except Exception as e:
    print(f"Failed open_mfdataset: {e}")
