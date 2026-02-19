import sys
import os
import glob
import re
import logging
import xarray as xr

# Add current directory to path so we can import modules
sys.path.append(os.getcwd())

# Mock Config
class MockConfig:
    CMIP6_RAW_UA_PATH_PATTERN = '/data/reloclim/normal/CMIP6_STREAM/paper1-cmip-data/ua_regrid/ua_Amon_{model}_{scenario}_*_*_regridded.nc'

# Mock StorylineAnalyzer with only the find_cmip6_ua_file method
class MockStorylineAnalyzer:
    def __init__(self):
        self.config = MockConfig()
        
    def find_cmip6_ua_file(self, model, scenario):
        # We need to copy the *modified* method content here to test it independently 
        # OR we import the actual class. Importing is better.
        pass

# Mock missing modules
import sys
from unittest.mock import MagicMock
sys.modules['statsmodels'] = MagicMock()
sys.modules['statsmodels.api'] = MagicMock()
sys.modules['jet_analyzer'] = MagicMock()

# Import the actual module
try:
    from storyline import StorylineAnalyzer
    from config import Config
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

def verify_ua_fix():
    print("Verifying UA fix for BCC-CSM2-MR ssp585...")
    
    # Initialize analyzer
    config = Config()
    analyzer = StorylineAnalyzer(config)
    
    # 1. Call find_cmip6_ua_file
    files = analyzer.find_cmip6_ua_file('BCC-CSM2-MR', 'ssp585')
    print(f"\nFound {len(files)} files.")
    for f in files:
        print(f" - {os.path.basename(f)}")
        
    # Check if overlapping file is removed
    overlapping_file = 'ua_Amon_BCC-CSM2-MR_ssp585_r1i1p1f1_gn_20550116-20970916_regridded.nc'
    present = any(overlapping_file in f for f in files)
    
    if not present:
        print("\nSUCCESS: Overlapping non-standard file was correctly excluded.")
    else:
        print("\nFAILURE: Overlapping non-standard file is still present!")
        sys.exit(1)
        
    # 2. Try opening with xarray
    print("\nAttempting to open files with xarray...")
    try:
        ds = xr.open_mfdataset(files, combine='nested', concat_dim='time', use_cftime=True, data_vars='minimal', coords='minimal', compat='override')
        print("Dataset opened successfully.")
        
        is_monotonic = ds.time.to_index().is_monotonic_increasing
        print(f"Time monotonic: {is_monotonic}")
        
        if is_monotonic:
            print("SUCCESS: Time index is monotonic.")
        else:
            print("FAILURE: Time index is NOT monotonic.")
            sys.exit(1)
            
    except Exception as e:
        print(f"FAILURE: xarray open error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    verify_ua_fix()
