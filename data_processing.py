"""
Data processing module for climate analysis.

This file contains the DataProcessor class, which handles all low-level
data loading, cleaning, and transformations, such as:
- Reading NetCDF files (20CRv3, ERA5)
- Calculating seasonal and spatial means
- Calculating anomalies
- Detrending data
"""
import pandas as pd
import numpy as np
import xarray as xr
import logging
import os
import traceback
import warnings
from functools import lru_cache
from scipy import signal

# Import the project's configuration
from config import Config

warnings.filterwarnings('ignore', category=UserWarning, message='Sending large graph')
warnings.filterwarnings('ignore', category=FutureWarning)

# This helper function is used as a preprocess step in open_mfdataset
def select_level_preprocess(ds, level_hpa=850):
    """
    Preprocessing function for xarray.open_mfdataset to select a specific pressure level.
    Handles different units (hPa or Pa) and potential variable names gracefully.
    """
    var_name = None
    potential_names = ['ua', 'u', 'uwnd']
    for name in potential_names:
        if name in ds.data_vars:
            var_name = name
            break

    if var_name is None:
        return ds  # Return original if wind variable not found

    level_coord_name = None
    if 'lev' in ds.coords:
        level_coord_name = 'lev'
    elif 'plev' in ds.coords:
        ds = ds.rename({'plev': 'lev'})
        level_coord_name = 'lev'
    elif 'level' in ds.coords:
        ds = ds.rename({'level': 'lev'})
        level_coord_name = 'lev'

    if level_coord_name:
        try:
            target_plev_pa = level_hpa * 100
            level_coord = ds[level_coord_name]
            lev_units = level_coord.attrs.get('units', '').lower()

            target_plev = level_hpa # Default to hPa
            if 'pa' in lev_units and 'hpa' not in lev_units:
                target_plev = target_plev_pa
            elif not lev_units and level_coord.max() > 50000: # Guess Pa if units are missing but values are large
                target_plev = target_plev_pa

            return ds.sel({level_coord_name: target_plev}, method='nearest')
        except Exception as e:
            filename = os.path.basename(ds.encoding.get("source", "Unknown file"))
            logging.error(f"ERROR (preprocess): Level selection failed for file '{filename}'. Target: {level_hpa}hPa. Error: {e}")
            return ds
    return ds


class DataProcessor:
    """Class for processing climate data."""

    @staticmethod
    @lru_cache(maxsize=16)
    def process_ncfile(file, var_name, var_out_name=None, level_val=None):
        """Process NetCDF file (e.g., from 20CRv3) and return monthly data. Results are cached."""
        logging.info(f"Processing 20CRv3 data from {file}...")
        try:
            ds = xr.open_dataset(file, decode_times=True, use_cftime=True, chunks={'time': 'auto'})
            
            var_mapping = {'pr': 'prate', 'tas': 'air', 'ua': 'uwnd'}
            actual_var = var_mapping.get(var_name, var_name)

            if actual_var not in ds.data_vars:
                logging.error(f"Variable {actual_var} (mapped from {var_name}) not found in {file}")
                ds.close()
                return None

            rename_dict = {}
            if 'longitude' in ds.dims: rename_dict['longitude'] = 'lon'
            if 'latitude' in ds.dims: rename_dict['latitude'] = 'lat'
            if rename_dict: ds = ds.rename(rename_dict)
            
            if 'lon' in ds.coords and np.any(ds.lon > 180):
                ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180)).sortby('lon')

            level_dim_name = next((name for name in ['level', 'plev', 'lev'] if name in ds.dims), None)
            if level_val is not None and level_dim_name:
                ds = ds.sel({level_dim_name: level_val}, method='nearest')
            
            if var_out_name and var_out_name != actual_var:
                 if actual_var in ds.data_vars:
                     ds = ds.rename({actual_var: var_out_name})
                     actual_var = var_out_name

            time_accessor = ds.time.dt
            ds = ds.assign_coords(year=("time", time_accessor.year.values), month=("time", time_accessor.month.values))
            
            ds_filtered = ds.sel(time=((ds.year >= Config.ANALYSIS_START_YEAR) & (ds.year <= Config.ANALYSIS_END_YEAR)))
            if ds_filtered.time.size == 0:
                logging.warning(f"No data found within analysis period for {file}.")
                ds.close(); return None

            ds.close()
            
            if var_name == 'pr' and ds_filtered[actual_var].attrs.get('units', '').lower() == 'kg/m^2/s':
                logging.info(f"Converting precipitation units to mm/day for {os.path.basename(file)}")
                ds_filtered[actual_var] *= 86400.0
                ds_filtered[actual_var].attrs['units'] = 'mm/day'
            
            da_to_return = ds_filtered[actual_var]
            da_to_return.attrs['dataset'] = Config.DATASET_20CRV3
            return da_to_return

        except Exception as e:
            logging.error(f"Error in process_ncfile for {file}: {e}")
            logging.error(traceback.format_exc())
            return None

    @staticmethod
    @lru_cache(maxsize=16)
    def process_era5_file(file, var_in_file, var_out_name=None, level_val=None):
        """Process ERA5 NetCDF file and return monthly data. Results are cached."""
        logging.info(f"Processing ERA5 data from {file}...")
        try:
            ds = xr.open_dataset(file, decode_times=True, use_cftime=True)
            
            var_name_used = var_in_file
            if var_name_used not in ds.data_vars:
                potential_names = {'pr': 'tp', 'tas': 't2m', 'ua': 'u'}
                if var_in_file in potential_names and potential_names[var_in_file] in ds.data_vars:
                    var_name_used = potential_names[var_in_file]
                else:
                    raise ValueError(f"Variable '{var_in_file}' or its alias not found in {file}")

            rename_dict = {}
            if 'latitude' in ds.dims or 'latitude' in ds.coords:
                rename_dict['latitude'] = 'lat'
            if 'longitude' in ds.dims or 'longitude' in ds.coords:
                rename_dict['longitude'] = 'lon'
            if rename_dict:
                ds = ds.rename(rename_dict)
            
            if 'lon' in ds.coords and np.any(ds.lon > 180):
                ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180)).sortby('lon')
            
            if level_val is not None and 'level' in ds.dims:
                ds = ds.sel(level=level_val)
            
            if var_out_name and var_out_name != var_name_used:
                ds = ds.rename({var_name_used: var_out_name})
                var_name_used = var_out_name

            ds_monthly = ds.resample(time='1MS').mean()
            
            time_coords = ds_monthly.time.dt
            ds_monthly = ds_monthly.assign_coords(year=("time", time_coords.year.values), month=("time", time_coords.month.values))
            
            ds_filtered = ds_monthly.sel(time=((ds_monthly.year >= Config.ANALYSIS_START_YEAR) & (ds_monthly.year <= Config.ANALYSIS_END_YEAR)))
            
            da_to_return = ds_filtered[var_name_used]
            da_to_return.attrs['dataset'] = Config.DATASET_ERA5
            return da_to_return

        except Exception as e:
            logging.error(f"Error in process_era5_file for {file}: {e}")
            logging.error(traceback.format_exc())
            return None

    @staticmethod
    def assign_season_to_dataarray(da):
        """Assign 'season' and 'season_year' coordinates to a DataArray."""
        if da is None: return None
            
        dt_index = da.time.to_index()
        df_time = pd.DataFrame({'year': dt_index.year, 'month': dt_index.month})
        
        month_map = {12: 'Winter', 1: 'Winter', 2: 'Winter',
                     3: 'Spring', 4: 'Spring', 5: 'Spring',
                     6: 'Summer', 7: 'Summer', 8: 'Summer',
                     9: 'Autumn', 10: 'Autumn', 11: 'Autumn'}
        df_time['season'] = df_time['month'].map(month_map)
        
        df_time['season_year'] = df_time['year']
        df_time.loc[df_time['month'] == 12, 'season_year'] += 1
        
        da = da.assign_coords(
            season=("time", df_time['season'].values),
            season_year=("time", df_time['season_year'].values)
        )
        
        season_key = [f"{sy}-{s}" for sy, s in zip(da['season_year'].values, da['season'].values)]
        return da.assign_coords(season_key=("time", season_key))

    @staticmethod
    def calculate_seasonal_means(da):
        """Calculate seasonal means from a DataArray with a 'season_key' coordinate."""
        if da is None or 'season_key' not in da.coords:
            logging.error("'season_key' coordinate missing in calculate_seasonal_means.")
            return None

        seasonal_mean = da.groupby("season_key").mean(dim="time", skipna=True)
        
        if seasonal_mean.season_key.size == 0:
            return None

        years_list = [int(key.split('-')[0]) for key in seasonal_mean['season_key'].values]
        seasons_list = [key.split('-')[1] for key in seasonal_mean['season_key'].values]

        seasonal_mean = seasonal_mean.assign_coords(
            temp_season_year=("season_key", years_list),
            temp_season_str=("season_key", seasons_list)
        )
        
        ds_unstacked = seasonal_mean.set_index(season_key=["temp_season_year", "temp_season_str"]).unstack("season_key")
        
        final_da = ds_unstacked.rename({"temp_season_year": "season_year", "temp_season_str": "season"})
        if 'dataset' in da.attrs:
            final_da.attrs['dataset'] = da.attrs['dataset']
            
        return final_da

    @staticmethod
    def calculate_anomalies(da, base_period_start=None, base_period_end=None, as_percentage=False):
        """Calculate anomalies relative to a climatological period."""
        if da is None: return None
            
        start = base_period_start if base_period_start is not None else Config.BASE_PERIOD_START_YEAR
        end = base_period_end if base_period_end is not None else Config.BASE_PERIOD_END_YEAR
        
        try:
            da = da.assign_coords(month=da.time.dt.month)
            da_ref = da.sel(time=((da.time.dt.year >= start) & (da.time.dt.year <= end)))
            climatology = da_ref.groupby('month').mean('time')
            
            if as_percentage:
                anomalies = da.groupby('month') / climatology * 100 - 100
            else:
                anomalies = da.groupby('month') - climatology
            
            if 'dataset' in da.attrs:
                anomalies.attrs['dataset'] = da.attrs['dataset']
            return anomalies
        except Exception as e:
            logging.error(f"Error in calculate_anomalies: {e}")
            return xr.full_like(da, np.nan)

    @staticmethod
    def calculate_spatial_mean(da, lat_min, lat_max, lon_min, lon_max):
        """Calculate area-weighted spatial mean within a box."""
        if da is None: return None
            
        try:
            # === KORRIGIERTER ABSCHNITT START ===
            # This logic checks if latitude is descending and reverses the slice if needed.
            lat_slice = slice(lat_min, lat_max)
            if 'lat' in da.coords and da.lat.size > 1:
                if da.lat.values[0] > da.lat.values[-1]: # Check for descending order
                    lat_slice = slice(lat_max, lat_min)
            
            domain = da.sel(lat=lat_slice, lon=slice(lon_min, lon_max))
            # === KORRIGIERTER ABSCHNITT ENDE ===
            
            if domain.lat.size == 0 or domain.lon.size == 0:
                logging.warning("Spatial domain selection resulted in zero size.")
                return None

            weights = np.cos(np.deg2rad(domain.lat))
            weights.name = "weights"
            
            weighted_mean = domain.weighted(weights).mean(dim=("lat", "lon"), skipna=True)
            weighted_mean.attrs = da.attrs
            return weighted_mean
        except Exception as e:
            logging.error(f"Error in calculate_spatial_mean: {e}")
            return None

    @staticmethod
    def filter_by_season(da, season_name):
        """Filter a DataArray for a specific season."""
        if da is None or 'season' not in da.coords: return None
            
        try:
            if 'season' in da.dims:
                filtered_da = da.sel(season=season_name)
            else:
                filtered_da = da.where(da.season == season_name, drop=True)
            
            if filtered_da.size == 0: return None
            return filtered_da
        except Exception as e:
            logging.error(f"Error in filter_by_season for '{season_name}': {e}")
            return None

    @staticmethod
    def detrend_data(data):
        """Detrend a time series or spatial DataArray using scipy.detrend or a polyfit fallback."""
        if data is None: return None
        
        time_dim = 'season_year'
        if time_dim not in data.dims or data[time_dim].size < 2:
            return data

        try:
            clean_data = data.dropna(dim=time_dim, how='all')
            if clean_data[time_dim].size < 2: return data 
            clean_data = clean_data.load()

            def detrend_if_possible(x):
                if np.all(np.isfinite(x)):
                    return signal.detrend(x)
                return x

            detrended_data = xr.apply_ufunc(
                detrend_if_possible,
                clean_data,
                input_core_dims=[[time_dim]],
                output_core_dims=[[time_dim]],
                exclude_dims=set((time_dim,)),
                output_dtypes=[data.dtype]
            )
            
            detrended_data = detrended_data.assign_coords(
                {time_dim: clean_data[time_dim]}
            )
            detrended_data.attrs = dict(data.attrs, detrended=True)
            return detrended_data.reindex_like(data)
            
        except Exception as e_scipy:
            logging.warning(f"Detrending with scipy failed: {e_scipy}. Trying polyfit fallback.")
            try:
                # Hier laden wir die Daten auch, falls der erste Versuch fehlschlägt
                clean_data = data.dropna(dim=time_dim, how='any').load()
                if clean_data[time_dim].size < 2: return data

                p = clean_data.polyfit(dim=time_dim, deg=1)
                fit = xr.polyval(data[time_dim], p.polyfit_coefficients)

                detrended_data = data - fit
                detrended_data.attrs = dict(data.attrs, detrended=True)
                return detrended_data
            except Exception as e_fallback:
                logging.error(f"Detrending also failed with polyfit fallback. Returning original data. Error: {e_fallback}")
                return data