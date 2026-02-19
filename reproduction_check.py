
import glob
import os

# Define the pattern from Config
CMIP6_RAW_PR_PATH_PATTERN = '/data/reloclim/normal/CMIP6_STREAM/paper1-cmip-data/pr_regrid/pr_Amon_{model}_{scenario}_*_*_regridded.nc'

model = "EC-Earth3-Veg-LR"
scenario = "ssp585"

search_pattern = CMIP6_RAW_PR_PATH_PATTERN.format(model=model, scenario=scenario)

print(f"Testing Pattern: {search_pattern}")

found_files = glob.glob(search_pattern)

print(f"Found {len(found_files)} files.")
for f in found_files:
    print(f" - {os.path.basename(f)}")
