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
            common_args = {
                "paths": all_files, "parallel": False, "engine": 'netcdf4',
                "use_cftime": True, "coords": 'minimal', "data_vars": 'minimal',
                "compat": 'override', "chunks": {'time': 120}
            }

            if variable == 'ua':
                preprocess_func = lambda ds: select_level_preprocess(ds, level_hpa=self.config.CMIP6_LEVEL)
                ds = xr.open_mfdataset(**common_args, combine='nested', concat_dim='time', preprocess=preprocess_func)
            else:
                ds = xr.open_mfdataset(**common_args, combine='nested', concat_dim='time')

            data_var = ds[actual_var_name]

            rename_map = {'latitude': 'lat', 'longitude': 'lon', 'plev': 'lev'}
            final_rename_map = {old: new for old, new in rename_map.items() if old in data_var.dims or old in data_var.coords}
            if final_rename_map:
                data_var = data_var.rename(final_rename_map)

            if not data_var.indexes['time'].is_unique:
                _, index = np.unique(data_var['time'], return_index=True)
                data_var = data_var.isel(time=index)
            if not data_var.indexes['time'].is_monotonic_increasing:
                data_var = data_var.sortby('time')

            if variable == 'pr' and data_var.attrs.get('units', '').lower() == 'kg m-2 s-1':
                data_var = data_var * 86400.0
                data_var.attrs['units'] = 'mm/day'
            if 'tas' in variable and data_var.attrs.get('units', '').lower() == 'k':
                data_var = data_var - 273.15
                data_var.attrs['units'] = 'degC'

            is_global_tas = (variable == self.config.CMIP6_GLOBAL_TAS_VAR) and not force_regional
            if is_global_tas:
                if 'lat' in data_var.dims and 'lon' in data_var.dims:
                    weights = np.cos(np.deg2rad(data_var.lat))
                    data_var = data_var.weighted(weights).mean(dim=("lon", "lat"), skipna=True)
                if data_var.ndim > 1:
                    data_var = data_var.squeeze(drop=True)
                if data_var.ndim != 1 or 'time' not in data_var.dims:
                    raise ValueError(f"Global TAS for {model} could not be reduced to 1D time series. Dims: {data_var.dims}")
            else:
                if 'lon' in data_var.coords and np.any(data_var.lon > 180):
                    data_var = data_var.assign_coords(lon=(((data_var.lon + 180) % 360) - 180)).sortby('lon')
            
            # Hier begrenzen wir die Zeitachse aller geladenen Modelldaten auf den 31.12.2099
            if 'time' in data_var.coords and data_var.time.size > 0:
                data_var = data_var.sel(time=slice(None, '2100-01-01'))

            return data_var.load()

        # --- KORRIGIERTER/ERWEITERTER FEHLER-BLOCK ---
        except (OSError, ValueError) as e:
            if isinstance(e, OSError) and "HDF" in str(e):
                logging.error(f"FATAL HDF ERROR (corrupted file) for {variable}, {model}: {e}")
            else:
                logging.error(f"FATAL ERROR loading/processing {variable} for {model}: {e}")
            logging.error(traceback.format_exc())
            return None
        except Exception as e:
            logging.error(f"UNEXPECTED FATAL ERROR loading/processing {variable} for {model}: {e}")
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
        
        if models_to_run is None:
            logging.info("No model list provided, scanning for available models...")
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

        all_models_attempted = set(models_to_run)

        # Step 1: Load data
        all_vars = self.config.CMIP6_VARIABLES_TO_LOAD + [self.config.CMIP6_GLOBAL_TAS_VAR]
        model_data = {}
        for model in models_to_run:
            for scenario in self.config.CMIP6_SCENARIOS:
                key = self.get_model_scenario_key(model, scenario)
                model_data[key] = {}
                
                is_valid_model = True
                for var in self.config.CMIP6_VARIABLES_TO_LOAD:
                    force_regional = (var == 'tas')
                    data = self._load_and_preprocess_model_data(model, [scenario], var, force_regional=force_regional)
                    if data is None:
                        logging.warning(f"Skipping model {key} due to missing regional variable: {var}")
                        is_valid_model = False
                        break
                    model_data[key][var] = data
                
                if not is_valid_model:
                    model_data.pop(key, None)
                    continue

                global_tas_var = self.config.CMIP6_GLOBAL_TAS_VAR
                data = self._load_and_preprocess_model_data(model, [scenario], global_tas_var, force_regional=False)
                if data is None:
                    logging.warning(f"Skipping model {key} due to missing global variable: {global_tas_var}")
                    model_data.pop(key, None)
                    continue
                
                model_data[key][f"{global_tas_var}_global"] = data

        if not model_data:
            logging.error("No CMIP6 models were successfully loaded. Aborting analysis.")
            return {}
            
        # Steps 2-5... (Diese Schritte bleiben unverändert und werden hier zur Kürze weggelassen)
        # Step 2: Calculate GWL threshold years
        gwl_thresholds = {}
        for key, data in model_data.items():
            global_tas_key = f"{self.config.CMIP6_GLOBAL_TAS_VAR}_global"
            if global_tas_key in data and data[global_tas_key] is not None:
                thresholds = self.calculate_gwl_thresholds(
                    data[global_tas_key],
                    (self.config.CMIP6_PRE_INDUSTRIAL_REF_START, self.config.CMIP6_PRE_INDUSTRIAL_REF_END),
                    self.config.GWL_TEMP_SMOOTHING_WINDOW,
                    self.config.GWL_FINE_STEPS_FOR_PLOT
                )
                if thresholds:
                    gwl_thresholds[key] = thresholds
            else:
                logging.warning(f"Global TAS variable '{global_tas_key}' not found for {key}. Cannot calculate GWL thresholds.")

        # Step 3: Calculate time series of all metrics
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
        
        # Step 5: Calculate deltas and MMM changes
        all_deltas = {met: {gwl: {} for gwl in self.config.GWL_FINE_STEPS_FOR_PLOT} for met in metrics_list}
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
                            all_deltas[met][gwl][key] = delta

        mmm_changes = {
            gwl: {
                met: np.mean(list(all_deltas[met][gwl].values())) if all_deltas[met][gwl] else np.nan
                for met in metrics_list
            }
            for gwl in self.config.GWL_FINE_STEPS_FOR_PLOT
        }
        
        # Create model run summary
        models_per_gwl = {}
        all_models_in_deltas = set()
        if metrics_list:
            primary_metric_data = all_deltas.get(metrics_list[0], {})
            for gwl, model_deltas in primary_metric_data.items():
                model_names = {key.split('_')[0] for key in model_deltas.keys()}
                models_per_gwl[gwl] = sorted(list(model_names))
                all_models_in_deltas.update(model_names)

        final_failed_models = sorted(list(all_models_attempted - all_models_in_deltas))
        
        model_run_status = {
            'successful_models_per_gwl': models_per_gwl,
            'failed_models': final_failed_models
        }

        # --- NEU: Klassifiziere Modelle in Storylines ---
        storyline_classification = self.classify_models_into_storylines(
            all_deltas, self.config.STORYLINE_JET_CHANGES
        )
        # --- ENDE NEU ---

        return {
            'gwl_threshold_years': gwl_thresholds,
            'cmip6_model_data_loaded': model_data,
            'model_metric_timeseries': metric_timeseries,
            'model_data_at_hist_reference': model_abs_means_at_hist_ref,
            'model_data_at_gwl': model_abs_means_at_gwl,
            'all_individual_model_deltas_for_plot': all_deltas,
            'mmm_changes': mmm_changes,
            'model_run_status': model_run_status,
            'storyline_classification': storyline_classification # Hinzugefügter Wert
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
    
    @staticmethod
    def classify_models_into_storylines(all_deltas, storyline_defs):
        """
        Classifies individual CMIP6 models into storylines based on their jet index changes.

        Parameters:
        -----------
        all_deltas : dict
            Dictionary containing the calculated changes (deltas) for each model, metric, and GWL.
            Format: {metric: {gwl: {model_key: delta}}}
        storyline_defs : dict
            The storyline definitions from the Config class.
            Format: {'IndexName': {GWL: {'StorylineType': Value}}}

        Returns:
        --------
        dict
            A nested dictionary with the classification results.
            Format: {gwl: {jet_index: {storyline_type: [list_of_models]}}}
        """
        logging.info("Classifying CMIP6 models into defined storylines...")
        classification_results = {}
        
        for jet_index, gwl_storylines in storyline_defs.items():
            # Bestimme die Toleranz basierend auf dem Indexnamen
            tolerance = 0.25 if 'Speed' in jet_index else 0.35
            logging.info(f"  Processing '{jet_index}' with tolerance ±{tolerance}")
            
            for gwl, storylines in gwl_storylines.items():
                if gwl not in classification_results:
                    classification_results[gwl] = {}
                if jet_index not in classification_results[gwl]:
                    classification_results[gwl][jet_index] = {}

                # Hole die individuellen Modelldeltas für diesen spezifischen Jet-Index und GWL
                model_deltas = all_deltas.get(jet_index, {}).get(gwl, {})
                if not model_deltas:
                    logging.warning(f"    No model delta data found for '{jet_index}' at GWL {gwl}. Skipping.")
                    continue

                for storyline_type, storyline_value in storylines.items():
                    lower_bound = storyline_value - tolerance
                    upper_bound = storyline_value + tolerance
                    
                    models_in_storyline = []
                    for model_key, delta_val in model_deltas.items():
                        if lower_bound <= delta_val <= upper_bound:
                            # Behalte den kompletten Schlüssel (z.B. 'ACCESS-CM2_ssp585')
                            models_in_storyline.append(model_key)
                    
                    classification_results[gwl][jet_index][storyline_type] = sorted(models_in_storyline)
        return classification_results
    
    def calculate_cmip6_u850_change_fields(self,
                                            models_to_run=None,
                                            future_scenario='ssp585',
                                            future_period=(2070, 2099),
                                            historical_period=(1995, 2014),
                                            preloaded_cmip6_data=None):
        """
        Calculates the MMM U850 change (future - historical) and the historical MMM U850 field.
        Utilizes preloaded CMIP6 data if provided. Includes detailed logging for debugging.
        """
        logging.info(f"\nCalculating CMIP6 MMM U850 change: {future_scenario} ({future_period[0]}-{future_period[1]}) vs Historical ({historical_period[0]}-{historical_period[1]})...")

        if models_to_run is None:
            logging.warning("models_to_run is None, scanning for all potential models.")
            potential_models_set = set()
            base_path_scan = self.config.CMIP6_VAR_PATH.format(variable='ua')
            file_pattern_scan = self.config.CMIP6_FILE_PATTERN.format(
                variable='ua', model='*', experiment='*',
                member=self.config.CMIP6_MEMBER_ID, grid='*', start_date='*', end_date='*'
            )
            all_found_files_scan = glob.glob(os.path.join(base_path_scan, file_pattern_scan))
            for f_scan in all_found_files_scan:
                try: potential_models_set.add(os.path.basename(f_scan).split('_')[2])
                except IndexError: logging.warning(f"Could not parse model name from: {f_scan}")
            models_to_process_list = sorted(list(potential_models_set))
        else:
            models_to_process_list = models_to_run

        logging.info(f"Processing {len(models_to_process_list)} models for U850 change fields: {models_to_process_list}")

        model_u850_changes = {'DJF': [], 'JJA': []}
        model_u850_historical_means = {'DJF': [], 'JJA': []}
        valid_models_processed = 0

        for model in models_to_process_list:
            logging.info(f"--- Attempting U850 change processing for model: {model} ---")
            model_processed_successfully = False

            try:
                ua_combined_monthly = None
                model_scenario_key = self.get_model_scenario_key(model, future_scenario)

                if preloaded_cmip6_data and model_scenario_key in preloaded_cmip6_data and 'ua' in preloaded_cmip6_data[model_scenario_key]:
                    ua_combined_monthly = preloaded_cmip6_data[model_scenario_key]['ua']
                else:
                    logging.info(f"    Directly loading 'ua' data for {model}.")
                    ua_combined_monthly = self._load_and_preprocess_model_data(model, [future_scenario], 'ua')

                if ua_combined_monthly is None:
                    logging.warning(f"    SKIP {model}: Could not load 'ua' data.")
                    continue

                ua_hist_monthly = ua_combined_monthly.sel(time=slice(str(historical_period[0]), str(historical_period[1])))
                ua_future_monthly = ua_combined_monthly.sel(time=slice(str(future_period[0]), str(future_period[1])))

                if ua_hist_monthly.time.size == 0 or ua_future_monthly.time.size == 0:
                    logging.warning(f"    SKIP {model}: Not enough data in historical or future period.")
                    continue

                ua_hist_seasonal = self.data_processor.calculate_seasonal_means(self.data_processor.assign_season_to_dataarray(ua_hist_monthly))
                ua_future_seasonal = self.data_processor.calculate_seasonal_means(self.data_processor.assign_season_to_dataarray(ua_future_monthly))
                
                if ua_hist_seasonal is None or ua_future_seasonal is None:
                    continue

                for season in ['Winter', 'Summer']:
                    s_key = 'DJF' if season == 'Winter' else 'JJA'
                    ua_hist_season = self.data_processor.filter_by_season(ua_hist_seasonal, season)
                    ua_future_season = self.data_processor.filter_by_season(ua_future_seasonal, season)

                    if ua_hist_season is not None and ua_future_season is not None:
                        hist_mean = ua_hist_season.mean(dim='season_year', skipna=True)
                        future_mean = ua_future_season.mean(dim='season_year', skipna=True)
                        
                        change = future_mean - hist_mean
                        model_u850_changes[s_key].append(change)
                        model_u850_historical_means[s_key].append(hist_mean)
                        model_processed_successfully = True

            except Exception as e:
                logging.error(f"    Error processing U850 change for {model}: {e}")
                traceback.print_exc()

            if model_processed_successfully:
                valid_models_processed += 1

        results = {}
        if valid_models_processed < 3:
            logging.warning(f"Not enough models ({valid_models_processed}) to compute reliable MMM for U850 change.")
            return None

        for season in ['DJF', 'JJA']:
            if model_u850_changes[season]:
                aligned_changes = [da.reindex_like(model_u850_changes[season][0], method='nearest') for da in model_u850_changes[season]]
                aligned_hist_means = [da.reindex_like(model_u850_historical_means[season][0], method='nearest') for da in model_u850_historical_means[season]]
                
                results[season] = {
                    'u850_change_mmm': xr.concat(aligned_changes, dim='model').mean(dim='model', skipna=True).load(),
                    'u850_historical_mean_mmm': xr.concat(aligned_hist_means, dim='model').mean(dim='model', skipna=True).load()
                }
            else:
                results[season] = None
                
        return results