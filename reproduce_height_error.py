
import xarray as xr
import numpy as np

def reproduce():
    print("Creating dummy datasets...")
    # Dataset 1: With height coordinate (scalar)
    data1 = np.random.rand(10, 10)
    ds1 = xr.DataArray(data1, coords={'lat': np.arange(10), 'lon': np.arange(10), 'height': 2.0}, dims=('lat', 'lon'), name='tas')
    
    # Dataset 2: Without height coordinate
    data2 = np.random.rand(10, 10)
    ds2 = xr.DataArray(data2, coords={'lat': np.arange(10), 'lon': np.arange(10)}, dims=('lat', 'lon'), name='tas')
    
    print("Dataset 1 coords:", ds1.coords)
    print("Dataset 2 coords:", ds2.coords)

    # Attempt concat - this often works with NaN padding unless strict checks are on, but let's see.
    # The error message implies coords='different' was inferred or passed.
    # xarray.concat documentation says coords='different' computes union of coords.
    # But if 'height' is missing in one, maybe it complains?
    # Let's try default concat first.
    try:
        print("\nAttempting default concat...")
        out = xr.concat([ds1, ds2], dim='model')
        print("Concat successful (unexpected if error is reproducible this easily)")
        print("Result coords:", out.coords)
    except ValueError as e:
        print(f"Concat failed as expected: {e}")
    except Exception as e:
        print(f"Concat failed with other error: {e}")

    # Fix: Drop height
    print("\nApplying fix: Dropping 'height'...")
    if 'height' in ds1.coords:
        ds1_fixed = ds1.drop_vars('height')
    else:
        ds1_fixed = ds1

    if 'height' in ds2.coords:
        ds2_fixed = ds2.drop_vars('height')
    else:
        ds2_fixed = ds2
        
    print("Dataset 1 fixed coords:", ds1_fixed.coords)
    
    try:
        print("Attempting concat after fix...")
        out_fixed = xr.concat([ds1_fixed, ds2_fixed], dim='model')
        print("Concat successful!")
        print("Result coords:", out_fixed.coords)
    except Exception as e:
        print(f"Concat still failed: {e}")

if __name__ == "__main__":
    reproduce()
