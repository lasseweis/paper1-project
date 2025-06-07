"""
CMIP6 and Storyline analysis module.

Contains the StorylineAnalyzer class to manage the CMIP6 analysis workflow:
- Loading and preprocessing CMIP6 model data.
- Identifying Global Warming Level (GWL) threshold years.
- Calculating metric changes at different GWLs.
- Calculating the final storyline impacts based on CMIP6 projections and
  reanalysis-derived sensitivities (beta_obs).
"""
import glob
import fnmatch
import traceback
import xarray as xr
import numpy as np
import pandas as pd
import logging
import os
import json

# Import local modules
from config import Config
from data_processing import DataProcessor, select_level_preprocess
from jet_analyzer import JetStreamAnalyzer


class StorylineAnalyzer:
    """Class for analyzing CMIP6 data and calculating storyline impacts."""

    def __init__(self, config: Config):
        self.config = config
        self.data_processor = DataProcessor()
        self.jet_analyzer = JetStreamAnalyzer()

    @staticmethod
    def get_model_scenario_key(model, scenario):
        return f"{model}_{scenario}"

    def _find_cmip_files(self, variable, experiment, model='*', member='*', grid='*'):
        """Finds CMIP6 files matching the criteria, searching in variable-specific subdirectories."""
        actual_var_for_path = 'tas' if variable == 'tas_global' else variable
        base_path = self.config.CMIP6_VAR_PATH.format(variable=actual_var_for_path)

        if not os.path.exists(base_path):
             logging.debug(f"Path {base_path} not found for '{actual_var_for_path}', trying CMIP6_DATA_BASE_PATH directly.")
             base_path = self.config.CMIP6_DATA_BASE_PATH

        file_pattern = self.config.CMIP6_FILE_PATTERN.format(
            variable=actual_var_for_path, model=model, experiment=experiment,
            member=member, grid=grid
        )
        search_path = os.path.join(base_path, file_pattern)
        found_files = glob.glob(search_path)
        
        if member != '*' and found_files:
            found_files = [f for f in found_files if f"_{member}_" in os.path.basename(f)]

        return sorted(list(set(found_files)))

    def _load_and_preprocess_model_data(self, model, scenarios_to_load, variable, force_regional=False):
        """
        Loads, concatenates historical and scenario data, and preprocesses it
        for a single CMIP6 model.
        """
        logging.info(f"Loading {variable} for {model} (hist + {scenarios_to_load}), force_regional={force_regional}...")
        member = self.config.CMIP6_MEMBER_ID
        actual_var_name = 'tas' if variable == 'tas_global' else variable
        
        all_files = self._find_cmip_files(actual_var_name, self.config.CMIP6_HISTORICAL_EXPERIMENT_NAME, model=model, member=member)
        for scenario in scenarios_to_load:
            all_files.extend(self._find_cmip_files(actual_var_name, scenario, model=model, member=member))

        if not all_files:
            logging.warning(f"--> No files found for {actual_var_name}, {model}. Skipping.")
            return None
        
        all_files = sorted(list(set(all_files)))
        logging.info(f"Found {len(all_files)} files for {actual_var_name} ({variable} requested), {model}.")

        try:
            preprocess_func = None
            # Define common arguments for open_mfdataset
            common_args = {
                "paths": all_files,
                "parallel": False,
                "engine": 'netcdf4',
                "use_cftime": True,
                "coords": 'minimal',
                "data_vars": 'minimal',
                "compat": 'override',
                "chunks": {'time': 120}
            }

            # **FIX:** Choose the combine strategy based on the variable, like in the original script.
            if variable == 'ua':
                preprocess_func = lambda ds: select_level_preprocess(ds, level_hpa=self.config.CMIP6_LEVEL)
                ds = xr.open_mfdataset(**common_args, combine='by_coords', preprocess=preprocess_func)
            else:
                ds = xr.open_mfdataset(**common_args, combine='nested', concat_dim='time')

            data_var = ds[actual_var_name]

            # Standardize coordinates
            rename_map = {'latitude': 'lat', 'longitude': 'lon', 'plev': 'lev'}
            dims_to_rename = {k: v for k, v in rename_map.items() if k in data_var.dims}
            if dims_to_rename: data_var = data_var.rename(dims_to_rename)

            # Ensure time coordinate is clean
            if not data_var.indexes['time'].is_unique:
                _, index = np.unique(data_var['time'], return_index=True)
                data_var = data_var.isel(time=index)
            if not data_var.indexes['time'].is_monotonic_increasing:
                data_var = data_var.sortby('time')

            # Unit conversions
            if variable == 'pr' and data_var.attrs.get('units', '').lower() == 'kg m-2 s-1':
                data_var = data_var * 86400.0
                data_var.attrs['units'] = 'mm/day'
            if 'tas' in variable and data_var.attrs.get('units', '').lower() == 'k':
                data_var = data_var - 273.15
                data_var.attrs['units'] = 'degC'

            # Specific processing for global vs. regional
            is_global_tas = (variable == self.config.CMIP6_GLOBAL_TAS_VAR) and not force_regional
            if is_global_tas:
                if 'lat' in data_var.dims and 'lon' in data_var.dims:
                    weights = np.cos(np.deg2rad(data_var.lat))
                    data_var = data_var.weighted(weights).mean(dim=("lon", "lat"), skipna=True)
                if data_var.ndim > 1: # Ensure it's a 1D time series
                    data_var = data_var.squeeze(drop=True)
                if data_var.ndim != 1 or 'time' not in data_var.dims:
                    raise ValueError(f"Global TAS for {model} could not be reduced to 1D time series. Dims: {data_var.dims}")
            else: # Regional processing
                if 'lon' in data_var.coords and np.any(data_var.lon > 180):
                    data_var = data_var.assign_coords(lon=(((data_var.lon + 180) % 360) - 180)).sortby('lon')
            
            # Filter time up to 2300 for long-running scenarios
            if 'time' in data_var.coords and data_var.time.size > 0:
                data_var = data_var.sel(time=slice(None, '2300-12-31'))

            return data_var.load()

        except Exception as e:
            logging.error(f"FATAL ERROR loading/processing {variable} for {model}: {e}")
            logging.error(traceback.format_exc())
            return None

    @staticmethod
    def calculate_gwl_thresholds(model_tas_data, pre_industrial_period, smoothing_window, gwl_levels):
        """Calculates the first year when specific global warming levels (GWLs) are exceeded."""
        if model_tas_data is None or model_tas_data.size == 0:
            return None

        annual_tas = model_tas_data.groupby('time.year').mean('time', skipna=True)
        
        ref_start, ref_end = pre_industrial_period
        tas_ref = annual_tas.sel(year=slice(ref_start, ref_end))
        if tas_ref.year.size == 0:
            logging.warning(f"No data in pre-industrial reference period ({ref_start}-{ref_end}).")
            return None
        
        global_tas_ref_mean = tas_ref.mean(dim='year', skipna=True).item()
        tas_anom = annual_tas - global_tas_ref_mean
        
        if tas_anom.year.size < smoothing_window:
            tas_anom_smoothed = tas_anom
        else:
            tas_anom_smoothed = tas_anom.rolling(year=smoothing_window, center=True).mean().dropna(dim='year')
        
        gwl_years = {gwl: None for gwl in gwl_levels}
        for gwl in gwl_levels:
            exceed_years = tas_anom_smoothed.where(tas_anom_smoothed > gwl, drop=True).year
            if exceed_years.size > 0:
                gwl_years[gwl] = int(exceed_years.min().item())
        return gwl_years

    def _extract_gwl_means(self, index_timeseries, gwl_years_model, gwl):
        """Extracts the N-year mean of a time series centered around a GWL threshold year."""
        threshold_year = gwl_years_model.get(gwl)
        if threshold_year is None or index_timeseries is None:
            return xr.DataArray(np.nan)

        window = self.config.GWL_YEARS_WINDOW
        start_year, end_year = threshold_year - window // 2, threshold_year + (window - 1) // 2
        
        gwl_slice = index_timeseries.sel(season_year=slice(start_year, end_year))
        if gwl_slice.season_year.size < window / 2:
            logging.warning(f"Only {gwl_slice.season_year.size}/{window} data points in window for GWL {gwl}.")

        return gwl_slice.mean(dim='season_year', skipna=True).compute()

    def analyze_cmip6_changes_at_gwl(self, models_to_run=None):
        """The main workflow for analyzing CMIP6 data at specific GWLs."""
        logging.info("\n--- Starting CMIP6 Analysis at Global Warming Levels ---")
        
        # **FIX:** Determine the list of models to run *before* the loop.
        # If models_to_run is None, find all available models instead of using a wildcard '*'.
        if models_to_run is None:
            logging.info("No model list provided, scanning for available models...")
            # Scan for models based on a primary variable like 'ua'
            base_path_scan = self.config.CMIP6_VAR_PATH.format(variable='ua')
            if not os.path.exists(base_path_scan): base_path_scan = self.config.CMIP6_DATA_BASE_PATH
            
            search_pattern = os.path.join(base_path_scan, self.config.CMIP6_FILE_PATTERN.format(
                variable='ua', model='*', experiment='*', member=self.config.CMIP6_MEMBER_ID, grid='*'
            ))
            all_ua_files = glob.glob(search_pattern)
            models_found = sorted(list(set(os.path.basename(f).split('_')[2] for f in all_ua_files)))
            models_to_run = models_found
            logging.info(f"Found {len(models_to_run)} models to process: {models_to_run}")

        if not models_to_run:
            logging.error("No models to run for CMIP6 analysis. Aborting.")
            return {}

        # Step 1: Load data for all specified models and variables
        all_vars = self.config.CMIP6_VARIABLES_TO_LOAD + [self.config.CMIP6_GLOBAL_TAS_VAR]
        model_data = {}
        for model in models_to_run:
            for scenario in self.config.CMIP6_SCENARIOS:
                key = self.get_model_scenario_key(model, scenario)
                model_data[key] = {}
                is_valid = True
                for var in set(all_vars):
                    data = self._load_and_preprocess_model_data(model, [scenario], var)
                    if data is None:
                        is_valid = False
                        break
                    model_data[key][var] = data
                if not is_valid:
                    model_data.pop(key, None)

        if not model_data:
            logging.error("No CMIP6 models were successfully loaded. Aborting analysis.")
            return {}
            
        # Step 2: Calculate GWL threshold years for each model
        gwl_thresholds = {}
        for key, data in model_data.items():
            global_tas_var_name = self.config.CMIP6_GLOBAL_TAS_VAR
            if global_tas_var_name in data and data[global_tas_var_name] is not None:
                thresholds = self.calculate_gwl_thresholds(
                    data[global_tas_var_name],
                    (self.config.CMIP6_PRE_INDUSTRIAL_REF_START, self.config.CMIP6_PRE_INDUSTRIAL_REF_END),
                    self.config.GWL_TEMP_SMOOTHING_WINDOW,
                    self.config.GWL_FINE_STEPS_FOR_PLOT
                )
                if thresholds:
                    gwl_thresholds[key] = thresholds
            else:
                logging.warning(f"Global TAS variable '{global_tas_var_name}' not found for {key}. Cannot calculate GWL thresholds.")


        # Step 3: Calculate time series of all metrics (jet indices, box means)
        metric_timeseries = {}
        box_coords = (self.config.BOX_LAT_MIN, self.config.BOX_LAT_MAX, self.config.BOX_LON_MIN, self.config.BOX_LON_MAX)
        for key, data in model_data.items():
            metric_timeseries[key] = {}
            ua_seas = DataProcessor.calculate_seasonal_means(DataProcessor.assign_season_to_dataarray(data['ua']))
            pr_seas = DataProcessor.calculate_seasonal_means(DataProcessor.assign_season_to_dataarray(data['pr']))
            tas_seas = DataProcessor.calculate_seasonal_means(DataProcessor.assign_season_to_dataarray(data['tas']))
            
            pr_box = DataProcessor.calculate_spatial_mean(pr_seas, *box_coords)
            tas_box = DataProcessor.calculate_spatial_mean(tas_seas, *box_coords)

            for season in ['Winter', 'Summer']:
                ua_season_data = DataProcessor.filter_by_season(ua_seas, season)
                s_label = 'DJF' if season == 'Winter' else 'JJA'
                metric_timeseries[key][f'{s_label}_JetSpeed'] = self.jet_analyzer.calculate_jet_speed_index(ua_season_data)
                metric_timeseries[key][f'{s_label}_JetLat'] = self.jet_analyzer.calculate_jet_lat_index(ua_season_data)
                metric_timeseries[key][f'{s_label}_pr'] = DataProcessor.filter_by_season(pr_box, season)
                metric_timeseries[key][f'{s_label}_tas'] = DataProcessor.filter_by_season(tas_box, season)

        # Step 4: Calculate absolute metric values at historical reference and at each GWL
        metrics_list = list(metric_timeseries.get(next(iter(metric_timeseries)), {}).keys())
        hist_ref_start, hist_ref_end = self.config.CMIP6_ANOMALY_REF_START, self.config.CMIP6_ANOMALY_REF_END
        
        model_abs_means_at_gwl = {}
        model_abs_means_at_hist_ref = {}

        for key, ts_data in metric_timeseries.items():
            hist_means = {met: ts.sel(season_year=slice(hist_ref_start, hist_ref_end)).mean().item() for met, ts in ts_data.items() if ts is not None and ts.size > 0}
            if len(hist_means) == len(metrics_list):
                model_abs_means_at_hist_ref[key] = hist_means

            if key in gwl_thresholds:
                model_abs_means_at_gwl[key] = {
                    'gwl': {
                        gwl: {
                            met: self._extract_gwl_means(ts_data.get(met), gwl_thresholds[key], gwl).item()
                            for met in metrics_list
                        }
                        for gwl in self.config.GWL_FINE_STEPS_FOR_PLOT
                    }
                }
        
        # Step 5: Calculate deltas (GWL - historical ref) and MMM changes
        all_deltas = {met: {gwl: [] for gwl in self.config.GWL_FINE_STEPS_FOR_PLOT} for met in metrics_list}
        for key in model_abs_means_at_gwl:
            if key in model_abs_means_at_hist_ref:
                for gwl in self.config.GWL_FINE_STEPS_FOR_PLOT:
                    for met in metrics_list:
                        hist_val = model_abs_means_at_hist_ref[key].get(met)
                        gwl_val = model_abs_means_at_gwl[key]['gwl'][gwl].get(met)
                        if hist_val is not None and gwl_val is not None and not np.isnan(hist_val) and not np.isnan(gwl_val):
                            delta = gwl_val - hist_val
                            if '_pr' in met and abs(hist_val) > 1e-9:
                                delta = (delta / hist_val) * 100.0
                            all_deltas[met][gwl].append(delta)

        mmm_changes = {gwl: {met: np.mean(all_deltas[met][gwl]) if all_deltas[met][gwl] else np.nan for met in metrics_list} for gwl in self.config.GLOBAL_WARMING_LEVELS}

        return {
            'gwl_threshold_years': gwl_thresholds,
            'cmip6_model_data_loaded': model_data,
            'model_metric_timeseries': metric_timeseries,
            'model_data_at_hist_reference': model_abs_means_at_hist_ref,
            'model_data_at_gwl': model_abs_means_at_gwl,
            'all_individual_model_deltas_for_plot': all_deltas,
            'mmm_changes': mmm_changes
        }
    
    def calculate_storyline_impacts(self, cmip6_results, beta_obs_slopes):
        """Calculates the impact changes for each defined storyline."""
        logging.info("\n--- Calculating Storyline Impacts ---")
        if 'mmm_changes' not in cmip6_results or not beta_obs_slopes:
            logging.error("Cannot calculate storyline impacts due to missing MMM changes or beta_obs slopes.")
            return None

        storyline_impacts = {gwl: {} for gwl in self.config.GLOBAL_WARMING_LEVELS}
        storyline_defs = self.config.STORYLINE_JET_CHANGES
        mmm_changes = cmip6_results['mmm_changes']

        impact_map = {
            'DJF_pr': {'jet_index': 'DJF_JetSpeed', 'beta_key': 'DJF_JetSpeed_vs_pr'},
            'DJF_tas': {'jet_index': 'DJF_JetSpeed', 'beta_key': 'DJF_JetSpeed_vs_tas'},
            'JJA_pr': {'jet_index': 'JJA_JetLat', 'beta_key': 'JJA_JetLat_vs_pr'},
            'JJA_tas': {'jet_index': 'JJA_JetLat', 'beta_key': 'JJA_JetLat_vs_tas'},
        }

        for gwl in self.config.GLOBAL_WARMING_LEVELS:
            if mmm_changes.get(gwl) is None: continue
            logging.info(f"\n  Processing Impacts for GWL {gwl}°C...")

            for impact_var, mapping in impact_map.items():
                jet_index_name = mapping['jet_index']
                beta_key = mapping['beta_key']

                delta_impact_mmm = mmm_changes[gwl].get(impact_var)
                delta_jet_mmm = mmm_changes[gwl].get(jet_index_name)
                beta_obs = beta_obs_slopes.get(beta_key)

                if any(v is None or np.isnan(v) for v in [delta_impact_mmm, delta_jet_mmm, beta_obs]):
                    continue

                logging.info(f"    Calculating impacts for {impact_var} (driven by {jet_index_name})")
                storyline_impacts[gwl][impact_var] = {}
                for storyline_type, delta_jet_story in storyline_defs[jet_index_name][gwl].items():
                    impact_adjustment = beta_obs * (delta_jet_story - delta_jet_mmm)
                    impact_change = delta_impact_mmm + impact_adjustment
                    storyline_impacts[gwl][impact_var][storyline_type] = impact_change
                    logging.info(f"      {storyline_type:<12}: Total Change = {impact_change:+.2f}")

        return {gwl: impacts for gwl, impacts in storyline_impacts.items() if impacts}