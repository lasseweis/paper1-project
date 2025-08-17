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
from scipy.stats import fisk, norm

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

            # Standardize coordinate names (robust version)
            rename_dict = {}
            if 'longitude' in ds.dims: rename_dict['longitude'] = 'lon'
            if 'latitude' in ds.dims: rename_dict['latitude'] = 'lat'
            # Prüft auch Koordinaten, die keine Dimensionen sind
            if 'longitude' in ds.coords and 'longitude' not in ds.dims: rename_dict['longitude'] = 'lon'
            if 'latitude' in ds.coords and 'latitude' not in ds.dims: rename_dict['latitude'] = 'lat'
            
            if rename_dict:
                 ds = ds.rename(rename_dict)
                 logging.debug(f"Renamed coordinates/dims: {rename_dict}")
            
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
            
            # --- START: VERBESSERTER BLOCK ZUR EINHEITENUMRECHNUNG ---
            if var_name == 'pr':
                data_var = ds_filtered[actual_var]
                if data_var.size > 0:
                    unit_str = data_var.attrs.get('units', '').lower()
                    conversion_factor = None

                    # 1. Prüft auf 'kg/m^2/s', was mm/s entspricht. Umrechnung zu mm/Tag.
                    if ('kg' in unit_str and 'm-2' in unit_str and 's-1' in unit_str) or ('kg/m^2/s' in unit_str):
                        conversion_factor = 86400.0
                        logging.info(f"Converting 20CRv3 precipitation from '{unit_str}' to 'mm/day' for {os.path.basename(file)}.")

                    # 2. Prüft auf 'm' (Meter). Umrechnung zu mm.
                    elif unit_str == 'm':
                        conversion_factor = 1000.0
                        logging.info(f"Converting 20CRv3 precipitation from 'm' to 'mm/day' for {os.path.basename(file)}.")
                    
                    # 3. Fallback-Prüfung anhand der Daten-Größenordnung, falls Einheit fehlt.
                    elif not unit_str or unit_str in ['unknown', '']:
                        if data_var.max() < 1.0:
                            conversion_factor = 1000.0
                            logging.warning(f"Unit attribute missing for 20CRv3. Converting precipitation based on data magnitude, assuming 'm' to 'mm/day' for {os.path.basename(file)}.")
                    
                    # Wenn ein Umrechnungsfaktor gefunden wurde, anwenden und Einheit anpassen.
                    if conversion_factor is not None:
                        ds_filtered[actual_var] *= conversion_factor
                        ds_filtered[actual_var].attrs['units'] = 'mm/day'
            # --- ENDE: VERBESSERTER BLOCK ---
            
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
            
            # --- START: VERBESSERTER BLOCK ZUR EINHEITENUMRECHNUNG ---
            if var_in_file == 'pr':
                data_var = ds_filtered[var_name_used]
                if data_var.size > 0:
                    unit_str = data_var.attrs.get('units', '').lower()
                    conversion_factor = None
                    
                    # 1. Prüft auf 'kg m-2 s-1', was mm/s entspricht. Umrechnung zu mm/Tag.
                    if 'kg' in unit_str and 'm-2' in unit_str and 's-1' in unit_str:
                        conversion_factor = 86400.0
                        logging.info(f"Converting ERA5 precipitation from 'kg m-2 s-1' to 'mm/day' for {os.path.basename(file)}.")
                    
                    # 2. Prüft auf 'm' (Meter), typisch für ERA5-Tagesgesamtniederschlag. Umrechnung zu mm.
                    elif unit_str == 'm':
                        conversion_factor = 1000.0
                        logging.info(f"Converting ERA5 precipitation from 'm' to 'mm/day' for {os.path.basename(file)}.")
                    
                    # 3. Fallback-Prüfung anhand der Daten-Größenordnung, falls Einheit fehlt oder unbekannt ist.
                    elif not unit_str or unit_str in ['unknown', '']:
                        if data_var.max() < 1.0:
                            conversion_factor = 1000.0
                            logging.warning(f"Unit attribute missing. Converting ERA5 precipitation based on data magnitude, assuming 'm' to 'mm/day' for {os.path.basename(file)}.")

                    # Wenn ein Umrechnungsfaktor gefunden wurde, anwenden und Einheit anpassen.
                    if conversion_factor is not None:
                        ds_filtered[var_name_used] = ds_filtered[var_name_used] * conversion_factor
                        ds_filtered[var_name_used].attrs['units'] = 'mm/day'
            # --- ENDE: VERBESSERTER BLOCK ---

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
        """Calculate seasonal means from a DataArray with season coordinates.
           The output will have 'season_year' and 'season' as dimensions."""
        if da is None:
            return None
        if 'season_key' not in da.coords:
            logging.error("Koordinate 'season_key' fehlt in calculate_seasonal_means.")
            return None

        try:
            # Debug: Gib Infos über das Eingabe-DataArray aus
            logging.debug(f"--- calculate_seasonal_means INPUT ---")
            # logging.debug(f"Input DataArray (da) head:\n{da.head()}") # Kann bei großen Arrays zu viel Log erzeugen
            logging.debug(f"Input DataArray (da) dims: {da.dims}")
            logging.debug(f"Input DataArray (da) coords: {list(da.coords)}")
            if 'season_key' in da.coords:
                logging.debug(f"Input 'season_key' dtype: {da['season_key'].dtype}")
                # logging.debug(f"Input 'season_key' head:\n{da['season_key'].head().values}")


            # Filtere ungültige Keys (wie vorher)
            keys_da = da['season_key']
            if keys_da.dtype == object: # Behandlung von object-Typ, der Strings oder gemischte Typen enthalten kann
                # Konvertiere zu String und prüfe auf NaN-ähnliche Werte und leere Strings
                keys_str = keys_da.astype(str) # Sicherstellen, dass es Strings sind für die Prüfung
                valid_keys_mask = keys_str.notnull() & (keys_str != '') & (keys_str != 'nan') & \
                                  (keys_str != '<NA>') & (keys_str != 'None') & \
                                  (pd.Series(keys_str).str.contains('-').fillna(False).values) # Muss '-' enthalten
            elif pd.api.types.is_string_dtype(keys_da.dtype): # Explizit für String-Typen
                valid_keys_mask = keys_da.notnull() & (keys_da != '') & \
                                  (keys_da.str.contains('-').fillna(False))
            else: # Für andere Typen, nur auf Null prüfen (sollte nicht der Fall sein für season_key)
                valid_keys_mask = keys_da.notnull()


            if not valid_keys_mask.all():
                original_size = da.time.size
                da_filtered = da.where(valid_keys_mask, drop=True)
                removed_count = original_size - da_filtered.time.size
                if removed_count > 0:
                    logging.warning(f"Ungültige oder schlecht formatierte Einträge in 'season_key' gefunden ({removed_count} entfernt). Filtere sie vor der Gruppierung heraus.")
                if da_filtered.time.size == 0:
                    logging.error("Keine gültigen season_key Einträge nach Filterung übrig.")
                    return None
            else:
                da_filtered = da

            # Debug: Gib Infos nach dem Filtern aus
            logging.debug(f"--- calculate_seasonal_means AFTER FILTER ---")
            # logging.debug(f"Filtered DataArray (da_filtered) head:\n{da_filtered.head()}")
            if 'season_key' in da_filtered.coords:
                logging.debug(f"Filtered 'season_key' dtype: {da_filtered['season_key'].dtype}")
                # logging.debug(f"Filtered 'season_key' head:\n{da_filtered['season_key'].head().values}")
                logging.debug(f"Is 'season_key' a dimension? {'season_key' in da_filtered.dims}")


            # Groupby und Mittelwertbildung
            logging.debug(f"Versuche groupby('season_key').mean()...")
            ds_seasonal = da_filtered.groupby("season_key").mean(dim="time", skipna=True)
            logging.debug(f"Groupby erfolgreich. ds_seasonal dims: {ds_seasonal.dims}")

            # Entferne Einträge, bei denen der Key nach dem Groupby NaN war (falls groupby NaN-Keys erzeugt)
            # Normalerweise sollte groupby keine NaN-Keys erzeugen, wenn da_filtered keine hat.
            # Aber zur Sicherheit:
            if ds_seasonal['season_key'].isnull().any():
                 logging.warning("NaN-Werte in 'season_key' nach groupby gefunden. Entferne diese.")
                 ds_seasonal = ds_seasonal.sel(season_key=ds_seasonal.season_key.notnull())

            if ds_seasonal.season_key.size == 0:
                logging.warning("Keine gültigen season_keys nach Groupby und NaN-Filterung übrig.")
                return None

            # Erstelle 'year' und 'season_str' Koordinaten aus der 'season_key' Dimension
            season_key_values = ds_seasonal['season_key'].values
            
            years_list = []
            seasons_list = []
            valid_parsed_keys = []

            for key_str in season_key_values:
                if isinstance(key_str, str) and '-' in key_str:
                    try:
                        year_part, season_part = key_str.split('-', 1)
                        years_list.append(int(year_part))
                        seasons_list.append(season_part)
                        valid_parsed_keys.append(key_str)
                    except ValueError:
                        logging.warning(f"Fehler beim Parsen des season_key '{key_str}'. Überspringe.")
                else:
                    logging.warning(f"Unerwarteter Typ oder Format für season_key '{key_str}'. Überspringe.")
            
            if not valid_parsed_keys:
                logging.error("Keine season_keys konnten erfolgreich geparst werden.")
                return None
            
            # Filtere ds_seasonal, falls einige Keys nicht geparst werden konnten
            if len(valid_parsed_keys) < len(season_key_values):
                ds_seasonal = ds_seasonal.sel(season_key=valid_parsed_keys)
                if ds_seasonal.season_key.size == 0:
                    logging.error("Keine Daten übrig nach Filterung nicht parsbarer season_keys.")
                    return None

            ds_seasonal = ds_seasonal.assign_coords(
                temp_season_year=("season_key", years_list),
                temp_season_str=("season_key", seasons_list)
            )

            # Wandle die 'season_key'-Dimension in ein MultiIndex ('temp_season_year', 'temp_season_str') um
            # und entpacke ('unstack') diesen, um die neuen Dimensionen zu erstellen.
            try:
                logging.debug(f"Vor set_index, ds_seasonal dims: {ds_seasonal.dims}, ds_seasonal coords: {list(ds_seasonal.coords.keys())}")
                
                # Stelle sicher, dass 'season_key' eine Dimension ist
                if 'season_key' not in ds_seasonal.dims:
                    # Sollte nach groupby der Fall sein, aber zur Sicherheit
                    if 'season_key' in ds_seasonal.coords:
                         ds_seasonal = ds_seasonal.set_index(season_key='season_key')
                    else:
                         logging.error("Dimension oder Koordinate 'season_key' nicht in ds_seasonal für set_index vorhanden.")
                         return None

                ds_multi_indexed = ds_seasonal.set_index(
                    season_key=["temp_season_year", "temp_season_str"]
                )
                logging.debug(f"Nach set_index, ds_multi_indexed dims: {ds_multi_indexed.dims}")
                
                # Unstack der 'season_key' (jetzt MultiIndex) Dimension
                ds_unstacked = ds_multi_indexed.unstack("season_key")
                logging.debug(f"Nach unstack, ds_unstacked dims: {ds_unstacked.dims}")

                # Benenne die neuen Dimensionen um
                # Die Namen der neuen Dimensionen entsprechen den Namen, die im set_index verwendet wurden
                ds_final = ds_unstacked.rename({
                    "temp_season_year": "season_year",
                    "temp_season_str": "season"
                })
                logging.debug(f"Nach rename, ds_final dims: {ds_final.dims}")

                # Entferne die temporären Koordinaten, falls sie noch als nicht-dimensionale Koordinaten existieren
                coords_to_drop = [c for c in ['temp_season_year', 'temp_season_str', 'season_key'] if c in ds_final.coords and c not in ds_final.dims]
                if coords_to_drop:
                    ds_final = ds_final.drop_vars(coords_to_drop)

            except Exception as e_unstack:
                logging.exception(f"Fehler beim Umstrukturieren mit set_index/unstack: ")
                return None

            if 'dataset' in da.attrs:
                ds_final.attrs['dataset'] = da.attrs['dataset']

            logging.debug(f"calculate_seasonal_means output dims: {ds_final.dims}")
            return ds_final

        except ValueError as ve:
            if "Providing a combination of `group` and **groupers is not supported" in str(ve):
                logging.error(f"ValueError im groupby: {ve}. Prüfe die Struktur von da_filtered und season_key.")
            else:
                logging.exception(f"Anderer ValueError in calculate_seasonal_means:")
            return None
        except Exception as e:
            logging.exception(f"Allgemeiner Fehler in calculate_seasonal_means:")
            return None

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
            
    @staticmethod
    def _calculate_pet_thornthwaite(tas_monthly, lat):
        """
        Calculates Potential Evapotranspiration (PET) using the Thornthwaite (1948) method.
        Internal helper function.
        
        Parameters:
        -----------
        tas_monthly : xr.DataArray
            Monthly mean temperature (°C) as a time series.
        lat : float
            Latitude in degrees.

        Returns:
        --------
        xr.DataArray
            Monthly PET in mm/month.
        """
        if tas_monthly is None: return None
        
        # Clip temperature at 0, as Thornthwaite is not defined for negative monthly means
        t = tas_monthly.clip(min=0)
        
        # Calculate monthly heat index 'i'
        i = (t / 5)**1.514
        
        # Calculate annual heat index 'I'
        I_annual = i.groupby("time.year").sum("time")

        # --- KORREKTUR START ---
        # Map the annual values back to the monthly time series using .sel()
        # This correctly broadcasts the annual value to each month of that year.
        I = I_annual.sel(year=tas_monthly.time.dt.year)
        # --- KORREKTUR ENDE ---
        
        # Calculate exponent 'a'
        a = (6.75e-7 * I**3) - (7.71e-5 * I**2) + (1.792e-2 * I) + 0.49239
        
        # Calculate day length and sunlight hours correction factor
        month_days = tas_monthly.time.dt.daysinmonth
        day_of_year = tas_monthly.time.dt.dayofyear
        
        delta = 0.409 * np.sin(0.0172 * day_of_year - 1.39)
        lat_rad = np.deg2rad(lat)
        
        # --- START DER KORREKTUR GEGEN RUNTIMEWARNING ---
        # Clip the argument of arccos to the valid range [-1, 1] to avoid NaN values
        # in polar regions during perpetual day/night.
        argument_for_arccos = -np.tan(lat_rad) * np.tan(delta)
        sunset_hour_angle = np.arccos(np.clip(argument_for_arccos, -1.0, 1.0))
        # --- ENDE DER KORREKTUR ---
        
        N = 24 / np.pi * sunset_hour_angle # Daylight hours
        
        # Correction factor L for days in month and daylight hours
        L = (N / 12) * (month_days / 30)
        
        # Calculate unadjusted PET
        pet_unadjusted = 16 * (10 * t / I)**a
        
        # Adjust PET for month and latitude
        pet_monthly = pet_unadjusted * L
        pet_monthly = pet_monthly.where(pet_monthly > 0, 0) # Ensure PET is not negative
        pet_monthly.attrs = {'long_name': 'Potential Evapotranspiration', 'units': 'mm/month'}
        
        return pet_monthly

    # Komplette, korrigierte Funktion für data_processing.py

    @staticmethod
    def calculate_spei(pr_monthly, tas_monthly, lat, scale=4):
        """
        Calculates the Standardized Precipitation-Evapotranspiration Index (SPEI).
        Handles both 1D timeseries and 3D (gridded) DataArrays.
        """
        if pr_monthly is None or tas_monthly is None:
            logging.error("SPEI calculation failed: Input precipitation or temperature is None.")
            return None

        is_spatial = 'lat' in pr_monthly.dims and 'lon' in pr_monthly.dims
        logging.info(f"Calculating SPEI with a {scale}-month scale (Spatial: {is_spatial})...")

        if is_spatial:
            # For spatial data, we need to pass a latitude array to PET calculation
            lat_da = pr_monthly.lat
        else:
            lat_da = lat

        pet_monthly = DataProcessor._calculate_pet_thornthwaite(tas_monthly, lat_da)
        if pet_monthly is None:
            logging.error("SPEI calculation failed: PET calculation returned None.")
            return None
        
        water_balance = pr_monthly - pet_monthly
        aggregated_balance = water_balance.rolling(time=scale, min_periods=scale).sum()
        valid_balance = aggregated_balance.dropna(dim='time', how='all')

        def fit_and_transform(x):
            # This function will be applied along the time dimension
            # Drop NaNs for fitting, but keep track of their locations
            series = pd.Series(x)
            not_nan = series.notna()
            if not_nan.sum() < 10: # Need enough data to fit
                return np.full_like(x, np.nan)
            
            # Fit the distribution only to valid data
            params = fisk.fit(series[not_nan])
            
            # Calculate CDF for all points (NaNs will result in NaNs)
            cdf = fisk.cdf(series.values, *params)
            
            # Transform to standard normal
            spei_values = norm.ppf(cdf)
            return spei_values

        if is_spatial:
            # Use apply_ufunc for efficient grid-cell-wise computation
            spei_values = xr.apply_ufunc(
                fit_and_transform,
                valid_balance,
                input_core_dims=[['time']],
                output_core_dims=[['time']],
                vectorize=True,  # Ensures the function is applied over non-core dims
                output_dtypes=[valid_balance.dtype]
            )
            spei_ts = spei_values.rename(f'spei_{scale}')
            # The time coordinate is preserved by apply_ufunc with core_dims
        else:
            # Original logic for 1D timeseries
            params = fisk.fit(valid_balance.values)
            cdf = fisk.cdf(valid_balance.values, *params)
            spei_values = norm.ppf(cdf)
            spei_ts = xr.DataArray(
                spei_values,
                coords={'time': valid_balance.time},
                dims=['time'],
                name=f'spei_{scale}'
            )

        spei_ts.attrs = {
            'long_name': f'{scale}-month Standardized Precipitation-Evapotranspiration Index',
            'scale_months': scale
        }
        return spei_ts.reindex_like(pr_monthly)
    
    @staticmethod
    def process_discharge_data(filepath_ssp245, filepath_ssp585, models_to_include):
        """
        Loads, processes, and combines discharge data from scenario-specific CSV files.
        This function is tailored to the specific format of the Danube discharge data.

        Args:
            filepath_ssp245 (str): Path to the SSP2-4.5 discharge CSV file.
            filepath_ssp585 (str): Path to the SSP5-8.5 discharge CSV file.
            models_to_include (list): A list of model names to extract from the files.

        Returns:
            dict: A dictionary where keys are model_scenario (e.g., 'ACCESS-CM2_ssp585')
                  and values are xarray DataArrays of the monthly discharge time series.
        """
        logging.info("Processing Danube discharge data for specified CMIP6 models...")
        
        all_discharge_data = {}
        
        for scenario, filepath in [('ssp245', filepath_ssp245), ('ssp585', filepath_ssp585)]:
            if not os.path.exists(filepath):
                logging.warning(f"Discharge file not found: {filepath}. Skipping scenario {scenario}.")
                continue

            try:
                # Read the CSV with semicolon delimiter and comma as decimal separator
                df = pd.read_csv(filepath, sep=';', decimal=',', na_values=['-0,01'])
                
                # The first column is the date, set it as the index
                df = df.rename(columns={df.columns[0]: 'date'})
                df['time'] = pd.to_datetime(df['date'])
                df = df.set_index('time')
                df = df.drop(columns=['date'])
                
                # Convert all data columns to numeric, coercing errors
                for col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

                # Filter the columns to only include the models we need
                for model_name in models_to_include:
                    if model_name in df.columns:
                        model_scenario_key = f"{model_name}_{scenario}"
                        
                        # Extract the series, drop NaNs, and convert to an xarray DataArray
                        model_series = df[model_name].dropna()
                        if not model_series.empty:
                            da = model_series.to_xarray()
                            da.name = 'discharge'
                            da.attrs['units'] = 'm3/s'
                            da.attrs['long_name'] = 'Danube River Discharge at Vienna'
                            all_discharge_data[model_scenario_key] = da
                            logging.info(f"Successfully processed discharge data for {model_scenario_key}")
                            
            except Exception as e:
                logging.error(f"Error processing discharge file {filepath}: {e}")
                logging.error(traceback.format_exc())

        return all_discharge_data