"""
CMIP6 and Storyline analysis module.

Contains the StorylineAnalyzer class to manage the CMIP6 analysis workflow:
- Loading and preprocessing CMIP6 model data.
- Identifying Global Warming Level (GWL) threshold years.
- Calculating metric changes at different GWLs.
- Calculating the final storyline impacts based on CMIP6 projections and
  reanalysis-derived sensitivities (beta_obs).
- Orchestrating advanced analyses like regression and correlation maps.
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
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.stats import chi2

# Import local modules
from config import Config
from data_processing import DataProcessor, select_level_preprocess
from jet_analyzer import JetStreamAnalyzer
from stats_analyzer import StatsAnalyzer


class StorylineAnalyzer:
    """
    Class for analyzing CMIP6 data, calculating storyline impacts, and
    orchestrating all advanced climate analyses.
    """

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

    def analyze_cmip6_changes_at_gwl(self, models_to_run=None, scenario_to_process=None):
        """The main workflow for analyzing CMIP6 data at specific GWLs."""
        # MODIFIED: Added scenario_to_process parameter
        if scenario_to_process:
            logging.info(f"\n--- Starting CMIP6 Analysis for SCENARIO: {scenario_to_process} at Global Warming Levels ---")
        else:
            logging.info("\n--- Starting CMIP6 Analysis at Global Warming Levels (all scenarios) ---")

        # MODIFIED: Filter the models to load based on the specified scenario
        all_models_config = self.config.REQUIRED_MODEL_SCENARIOS
        if not all_models_config:
            logging.error("The REQUIRED_MODEL_SCENARIOS dictionary in config.py is empty. Aborting.")
            return {}

        if scenario_to_process:
            # Select only models that are configured for the specific scenario
            models_and_scenarios_to_load = {
                model: [scenario_to_process]
                for model, scenarios in all_models_config.items()
                if scenario_to_process in scenarios
            }
            logging.info(f"Processing {len(models_and_scenarios_to_load)} models for scenario {scenario_to_process}.")
        else:
            # Keep original behavior if no scenario is specified
            models_and_scenarios_to_load = all_models_config
            logging.info(f"Processing {len(models_and_scenarios_to_load)} models based on the configuration file.")
        
        if not models_and_scenarios_to_load:
            logging.error(f"No models found for scenario {scenario_to_process} in the configuration. Aborting.")
            return {}

        all_models_attempted = set(models_and_scenarios_to_load.keys())

        # Step 1: Load data based on the provided dictionary (excluding discharge for now)
        model_data = {}
        for model, scenarios_to_load in models_and_scenarios_to_load.items():
            for scenario in scenarios_to_load:
                key = self.get_model_scenario_key(model, scenario)
                model_data[key] = {}
                
                is_valid_model = True
                for var in ['ua', 'pr', 'tas']: # Discharge is loaded separately
                    force_regional = (var == 'tas')
                    data = self._load_and_preprocess_model_data(model, [scenario], var, force_regional=force_regional)
                    if data is None:
                        logging.warning(f"Skipping model-scenario {key} due to missing regional variable: {var}")
                        is_valid_model = False
                        break
                    model_data[key][var] = data
                
                if not is_valid_model:
                    model_data.pop(key, None)
                    continue

                global_tas_var = self.config.CMIP6_GLOBAL_TAS_VAR
                data = self._load_and_preprocess_model_data(model, [scenario], global_tas_var, force_regional=False)
                if data is None:
                    logging.warning(f"Skipping model-scenario {key} due to missing global variable: {global_tas_var}")
                    model_data.pop(key, None)
                    continue
                
                model_data[key][f"{global_tas_var}_global"] = data

        if not model_data:
            logging.error("No CMIP6 models were successfully loaded. Aborting analysis.")
            return {}

        # --- NEW: Step 1.5 - Load Danube Discharge Data ---
        logging.info("\n--- Loading Danube Discharge Data ---")
        models_with_data = {key.split('_')[0] for key in model_data.keys()}
        discharge_data = self.data_processor.process_discharge_data(
            self.config.DISCHARGE_SSP245_FILE,
            self.config.DISCHARGE_SSP585_FILE,
            list(models_with_data)
        )
        
        # Add the loaded discharge data to the main model_data dictionary
        for key, data in discharge_data.items():
            # MODIFIED: Check if the scenario matches
            if key in model_data and key.endswith(scenario_to_process if scenario_to_process else ''):
                model_data[key]['discharge'] = data
            
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
            
            if 'discharge' in data and data['discharge'] is not None:
                discharge_seas = DataProcessor.calculate_seasonal_means(DataProcessor.assign_season_to_dataarray(data['discharge']))
            else:
                discharge_seas = None

            pr_box = DataProcessor.calculate_spatial_mean(pr_seas, *box_coords)
            tas_box = DataProcessor.calculate_spatial_mean(tas_seas, *box_coords)

            for season in ['Winter', 'Summer']:
                ua_season_data = DataProcessor.filter_by_season(ua_seas, season)
                s_label = 'DJF' if season == 'Winter' else 'JJA'
                metric_timeseries[key][f'{s_label}_JetSpeed'] = self.jet_analyzer.calculate_jet_speed_index(ua_season_data)
                metric_timeseries[key][f'{s_label}_JetLat'] = self.jet_analyzer.calculate_jet_lat_index(ua_season_data)
                metric_timeseries[key][f'{s_label}_pr'] = DataProcessor.filter_by_season(pr_box, season)
                metric_timeseries[key][f'{s_label}_tas'] = DataProcessor.filter_by_season(tas_box, season)
                if discharge_seas is not None:
                    metric_timeseries[key][f'{s_label}_discharge'] = DataProcessor.filter_by_season(discharge_seas, season)

        # Step 4: Calculate absolute metric values at historical reference and at each GWL
        metrics_list = list(metric_timeseries.get(next(iter(metric_timeseries)), {}).keys())
        hist_ref_start, hist_ref_end = self.config.CMIP6_ANOMALY_REF_START, self.config.CMIP6_ANOMALY_REF_END
        
        model_abs_means_at_gwl = {}
        model_abs_means_at_hist_ref = {}

        for key, ts_data in metric_timeseries.items():
            hist_means = {met: ts.sel(season_year=slice(hist_ref_start, hist_ref_end)).mean().item() for met, ts in ts_data.items() if ts is not None and ts.size > 0}
            if len(hist_means) > 0: # Check if at least some metrics were calculated
                model_abs_means_at_hist_ref[key] = hist_means

            if key in gwl_thresholds:
                model_abs_means_at_gwl[key] = {
                    'gwl': {
                        gwl: {
                            met: self._extract_gwl_means(ts_data.get(met), gwl_thresholds[key], gwl).item()
                            for met in metrics_list if ts_data.get(met) is not None
                        }
                        for gwl in self.config.GWL_FINE_STEPS_FOR_PLOT
                    }
                }
        
        # Step 5: Calculate deltas (changes) and MMM changes
        all_deltas = {met: {gwl: {} for gwl in self.config.GWL_FINE_STEPS_FOR_PLOT} for met in metrics_list}
        for key in model_abs_means_at_gwl:
            if key in model_abs_means_at_hist_ref:
                for gwl in self.config.GWL_FINE_STEPS_FOR_PLOT:
                    for met in metrics_list:
                        hist_val = model_abs_means_at_hist_ref[key].get(met)
                        gwl_val = model_abs_means_at_gwl[key]['gwl'][gwl].get(met)
                        if hist_val is not None and gwl_val is not None and not np.isnan(hist_val) and not np.isnan(gwl_val):
                            delta = gwl_val - hist_val
                            # For precipitation and discharge, calculate percentage change
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
        
        ### DEBUG ###
        logging.info("--- DEBUG: CHECKING MMM_CHANGES ---")
        for gwl in self.config.GLOBAL_WARMING_LEVELS:
            if gwl in mmm_changes:
                pr_djf = mmm_changes[gwl].get('DJF_pr', 'N/A')
                pr_jja = mmm_changes[gwl].get('JJA_pr', 'N/A')
                logging.info(f"  GWL {gwl}°C -> DJF_pr: {pr_djf}, JJA_pr: {pr_jja}")
        logging.info("--- END DEBUG ---")
        
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

        # Classify models into storylines using the quadrant method
        storyline_classification_2d = StorylineAnalyzer.classify_models_into_storylines_2d(
            all_deltas,
            self.config.STORYLINE_JET_CHANGES_2D,
            self.config.STORYLINE_INNER_RADIUS # Use the new inner radius parameter
        )

        return {
            'gwl_threshold_years': gwl_thresholds,
            'cmip6_model_data_loaded': model_data,
            'model_metric_timeseries': metric_timeseries,
            'model_data_at_hist_reference': model_abs_means_at_hist_ref,
            'model_data_at_gwl': model_abs_means_at_gwl,
            'all_individual_model_deltas_for_plot': all_deltas,
            'mmm_changes': mmm_changes,
            'model_run_status': model_run_status,
            'storyline_classification_2d': storyline_classification_2d
        }

    @staticmethod
    def calculate_storyline_impacts(cmip6_results):
        """
        Calculates the final impact changes for each defined 2D storyline by
        averaging the impact changes of all models classified within that storyline.
        This approach is consistent with the direct impact calculation for discharge.
        """
        logging.info("\n--- Calculating Final Storyline Impacts (by averaging classified models) ---")
        
        classification = cmip6_results.get('storyline_classification_2d')
        all_deltas = cmip6_results.get('all_individual_model_deltas_for_plot')
        gwls = Config.GLOBAL_WARMING_LEVELS

        if not classification or not all_deltas:
            logging.error("Cannot calculate storyline impacts: Missing model classification or delta values.")
            return None

        final_impacts = {gwl: {} for gwl in gwls}
        impact_vars_to_process = ['tas', 'pr']

        for gwl in gwls:
            logging.info(f"\n  Processing Impacts for GWL +{gwl}°C...")
            
            # Initialisiere die Dictionaries für diese GWL-Stufe
            for season in ['DJF', 'JJA']:
                for var in impact_vars_to_process:
                    final_impacts[gwl][f'{season}_{var}'] = {}

            # Gehe durch die klassifizierten Storylines für diese GWL-Stufe
            for storyline_key, model_list in classification.get(gwl, {}).items():
                if not model_list:
                    continue # Überspringe leere Storylines
                
                season = 'DJF' if 'DJF' in storyline_key else 'JJA'
                storyline_name = storyline_key.replace(f'{season}_', '')
                
                # Berechne die mittleren Auswirkungen für jede Variable
                for var in impact_vars_to_process:
                    impact_key = f'{season}_{var}'
                    
                    # Sammle die Delta-Werte der Modelle in dieser Storyline
                    model_deltas_for_impact = []
                    for model_run_key in model_list:
                        delta_val = all_deltas.get(impact_key, {}).get(gwl, {}).get(model_run_key)
                        if delta_val is not None and np.isfinite(delta_val):
                            model_deltas_for_impact.append(delta_val)
                    
                    if model_deltas_for_impact:
                        # Berechne den Mittelwert und speichere ihn
                        mean_impact = np.mean(model_deltas_for_impact)
                        final_impacts[gwl][impact_key][storyline_name] = {'total': mean_impact}
                        logging.info(f"      -> {storyline_name:<28} ({impact_key}): Mean Impact = {mean_impact:+.2f} (from {len(model_deltas_for_impact)} models)")

        return {gwl: impacts for gwl, impacts in final_impacts.items() if impacts}
    
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
    
    @staticmethod
    def classify_models_into_storylines_2d(all_deltas, storyline_defs_2d, radius):
        """
        Classifies CMIP6 models into 2D storylines using a multi-zone method.
        MODIFIED: The 'MMM' storyline now includes ALL available models. Other storylines
        are classified relative to the multi-model mean (MMM).
        """
        logging.info(f"Classifying models into 2D, Axial, and Extreme storylines (Inner Radius for Axial: {radius} std. dev.)...")
        classification_results = {}
        
        # This value corresponds to the radius of the 80% confidence ellipse.
        outer_radius_t_value = np.sqrt(chi2.ppf(0.5, 2) / 2)

        for season, gwl_storylines in storyline_defs_2d.items():
            jet_speed_key = f'{season}_JetSpeed'
            jet_lat_key = f'{season}_JetLat'
            
            for gwl, storylines in gwl_storylines.items():
                if gwl not in classification_results:
                    classification_results[gwl] = {}

                model_deltas_speed = all_deltas.get(jet_speed_key, {}).get(gwl, {})
                model_deltas_lat = all_deltas.get(jet_lat_key, {}).get(gwl, {})
                common_models = sorted(list(set(model_deltas_speed.keys()) & set(model_deltas_lat.keys())))

                if not common_models: continue

                speed_values = np.array([model_deltas_speed[m] for m in common_models])
                lat_values = np.array([model_deltas_lat[m] for m in common_models])

                mmm_speed = np.mean(speed_values)
                mmm_lat = np.mean(lat_values)
                std_dev_speed = np.std(speed_values)
                std_dev_lat = np.std(lat_values)

                if std_dev_speed < 1e-9 or std_dev_lat < 1e-9: continue

                # Initialize all storyline categories from the extended config
                for storyline_type in storylines:
                    storyline_key = f"{season}_{storyline_type}"
                    classification_results[gwl][storyline_key] = []

                # --- NEW LOGIC ---
                # PART 1: Assign ALL models to the 'MMM' storyline.
                classification_results[gwl][f"{season}_MMM"] = common_models
                
                # PART 2: Classify models into EXTREME QUADRANTS (outside the 80% ellipse)
                for model_key in common_models:
                    norm_dist_sq = ((model_deltas_speed[model_key] - mmm_speed) / std_dev_speed)**2 + \
                                ((model_deltas_lat[model_key] - mmm_lat) / std_dev_lat)**2
                    norm_dist = np.sqrt(norm_dist_sq)

                    if norm_dist > outer_radius_t_value:
                        is_north = model_deltas_lat[model_key] > mmm_lat
                        is_fast = model_deltas_speed[model_key] > mmm_speed
                        storyline_name = ""
                        if is_fast and is_north:      storyline_name = 'Fast Jet & Northward Shift'
                        elif not is_fast and is_north:storyline_name = 'Slow Jet & Northward Shift'
                        elif not is_fast and not is_north: storyline_name = 'Slow Jet & Southward Shift'
                        elif is_fast and not is_north: storyline_name = 'Fast Jet & Southward Shift'
                        if storyline_name:
                            classification_results[gwl][f"{season}_{storyline_name}"].append(model_key)
                
                # --- PART 3: Define and classify axial storylines ---
                # --- AUSKOMMENTIERT START ---
                # extreme_storyline_means = {}
                # extreme_types_in_config = [s for s in storylines if 'Shift' in s and 'MMM' not in s and 'Only' not in s and 'Extreme' not in s]
                # 
                # for storyline_name in extreme_types_in_config:
                #     full_key = f"{season}_{storyline_name}"
                #     models_in_storyline = classification_results[gwl].get(full_key, [])
                #     if models_in_storyline:
                #         mean_speed = np.mean([model_deltas_speed[m] for m in models_in_storyline])
                #         mean_lat = np.mean([model_deltas_lat[m] for m in models_in_storyline])
                #         extreme_storyline_means[storyline_name] = {'speed': mean_speed, 'lat': mean_lat}
                # 
                # axial_centers = {}
                # if extreme_storyline_means:
                #     max_lat_storyline = max(extreme_storyline_means, key=lambda k: extreme_storyline_means[k]['lat'])
                #     min_lat_storyline = min(extreme_storyline_means, key=lambda k: extreme_storyline_means[k]['lat'])
                #     max_speed_storyline = max(extreme_storyline_means, key=lambda k: extreme_storyline_means[k]['speed'])
                #     min_speed_storyline = min(extreme_storyline_means, key=lambda k: extreme_storyline_means[k]['speed'])
                # 
                #     axial_centers['Northward Shift Only'] = {'speed': 0, 'lat': extreme_storyline_means[max_lat_storyline]['lat']}
                #     axial_centers['Southward Shift Only'] = {'speed': 0, 'lat': extreme_storyline_means[min_lat_storyline]['lat']}
                #     axial_centers['Fast Jet Only'] = {'speed': extreme_storyline_means[max_speed_storyline]['speed'], 'lat': 0}
                #     axial_centers['Slow Jet Only'] = {'speed': extreme_storyline_means[min_speed_storyline]['speed'], 'lat': 0}
                # 
                # for storyline_name, center in axial_centers.items():
                #     storyline_key = f"{season}_{storyline_name}"
                #     classified_models_for_axial = []
                #     
                #     for model_key in common_models:
                #         norm_dist_sq = ((model_deltas_speed[model_key] - center['speed']) / std_dev_speed)**2 + \
                #                     ((model_deltas_lat[model_key] - center['lat']) / std_dev_lat)**2
                #         if np.sqrt(norm_dist_sq) <= radius:
                #             classified_models_for_axial.append(model_key)
                #     
                #     classification_results[gwl][storyline_key] = classified_models_for_axial
                # --- AUSKOMMENTIERT ENDE ---
                    
                # --- PART 4 - Identify and add corner extreme storylines ---
                # --- AUSKOMMENTIERT START ---
                # if common_models:
                #     norm_speed = (speed_values - np.min(speed_values)) / (np.max(speed_values) - np.min(speed_values))
                #     norm_lat = (lat_values - np.min(lat_values)) / (np.max(lat_values) - np.min(lat_values))
                #     
                #     metric_nw = norm_lat - norm_speed
                #     metric_se = norm_speed - norm_lat
                #     
                #     model_list_common = list(common_models)
                #     model_nw = model_list_common[np.argmax(metric_nw)]
                #     model_se = model_list_common[np.argmax(metric_se)]
                # 
                #     classification_results[gwl][f"{season}_Extreme NW"] = [model_nw]
                #     classification_results[gwl][f"{season}_Extreme SE"] = [model_se]
                #     logging.info(f"    - Extreme NW model for {season} @ {gwl}C: {model_nw}")
                #     logging.info(f"    - Extreme SE model for {season} @ {gwl}C: {model_se}")
                # --- AUSKOMMENTIERT ENDE ---

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

    # --- HIER BEGINNEN DIE INTEGRIERTEN METHODEN AUS ADVANCED ANALYZER ---

    @staticmethod
    def calculate_cmip6_regression_maps(cmip6_data, historical_period=(1995, 2014)):
        """
        Calculate regressions between UA850 and PR/TAS box indices using CMIP6 ensemble data
        for a specified historical period. Calculates MMM first and normalizes predictors.
        """
        logging.info(f"\nCalculating CMIP6 regression maps (U850 vs Box Indices, normalized predictors) for hist. period {historical_period}...")
        models = list(cmip6_data.keys())
        if not models:
            logging.info("  No CMIP6 model data provided.")
            return {}

        model_seasonal_means = {'ua': {}, 'pr': {}, 'tas': {}}
        hist_start, hist_end = historical_period

        for model in models:
            logging.info(f"  Processing historical seasonal means for {model}...")
            model_has_data = all(v in cmip6_data.get(model, {}) and cmip6_data[model][v] is not None for v in ['ua', 'pr', 'tas'])
            if not model_has_data:
                logging.warning(f"    Warning: Skipping {model} due to missing ua, pr, or tas data.")
                continue

            try:
                for var in ['ua', 'pr', 'tas']:
                    monthly_hist = cmip6_data[model][var].sel(time=slice(str(hist_start), str(hist_end)))
                    if monthly_hist.time.size == 0:
                        raise ValueError(f"No data for variable {var} in historical period.")
                    seasonal_full = DataProcessor.assign_season_to_dataarray(monthly_hist)
                    seasonal_mean = DataProcessor.calculate_seasonal_means(seasonal_full)
                    if seasonal_mean is not None:
                        model_seasonal_means[var][model] = seasonal_mean.load()
                    else:
                        raise ValueError(f"Seasonal mean calculation failed for {var}.")
            except Exception as e:
                logging.error(f"    Error processing historical data for {model}: {e}")
                for var_dict in model_seasonal_means.values():
                    var_dict.pop(model, None)

        mmm_seasonal = {}
        for var, models_with_var in model_seasonal_means.items():
            valid_models = {m: d for m, d in models_with_var.items() if d is not None and d.size > 0}
            if len(valid_models) < 3:
                logging.warning(f"  Skipping MMM for '{var}': Not enough valid models ({len(valid_models)} < 3).")
                continue
            try:
                datasets_to_combine = [ds.to_dataset(name=var).expand_dims(model=[m]) for m, ds in valid_models.items()]
                combined = xr.combine_by_coords(datasets_to_combine, compat='override', join='inner', coords='minimal', combine_attrs='drop_conflicts')
                mmm_seasonal[var] = combined[var].mean(dim='model', skipna=True)
                logging.info(f"    MMM for '{var}' successfully calculated from {len(valid_models)} models.")
            except Exception as e:
                logging.error(f"    ERROR combining datasets for '{var}' to create MMM: {e}")
                traceback.print_exc()
        
        if not all(k in mmm_seasonal for k in ['ua', 'pr', 'tas']):
             logging.error("  Cannot calculate CMIP6 regression maps due to missing MMM data.")
             return {}

        box_coords = (Config.BOX_LAT_MIN, Config.BOX_LAT_MAX, Config.BOX_LON_MIN, Config.BOX_LON_MAX)
        mmm_pr_box = DataProcessor.calculate_spatial_mean(mmm_seasonal['pr'], *box_coords)
        mmm_tas_box = DataProcessor.calculate_spatial_mean(mmm_seasonal['tas'], *box_coords)

        mmm_pr_box_detrended = DataProcessor.detrend_data(mmm_pr_box)
        mmm_tas_box_detrended = DataProcessor.detrend_data(mmm_tas_box)
        mmm_ua_detrended = DataProcessor.detrend_data(mmm_seasonal['ua'])

        if not all(d is not None for d in [mmm_pr_box_detrended, mmm_tas_box_detrended, mmm_ua_detrended]):
            logging.error("  ERROR: Detrending of CMIP6 MMM data failed.")
            return {}

        cmip6_regression_results = {}
        for season in ['Winter', 'Summer']:
            logging.info(f"  - Processing {season} CMIP6 regression maps...")
            pr_idx_norm = StatsAnalyzer.normalize(DataProcessor.filter_by_season(mmm_pr_box_detrended, season))
            tas_idx_norm = StatsAnalyzer.normalize(DataProcessor.filter_by_season(mmm_tas_box_detrended, season))
            ua_season_detrended = DataProcessor.filter_by_season(mmm_ua_detrended, season)
            
            if not all(d is not None for d in [pr_idx_norm, tas_idx_norm, ua_season_detrended]):
                 logging.warning(f"    Skipping {season}: Missing detrended/normalized MMM data.")
                 continue

            slopes_pr, p_values_pr = StorylineAnalyzer._calculate_regression_for_variable(pr_idx_norm, ua_season_detrended)
            slopes_tas, p_values_tas = StorylineAnalyzer._calculate_regression_for_variable(tas_idx_norm, ua_season_detrended)
            
            ua850_mean_orig = DataProcessor.filter_by_season(mmm_seasonal['ua'], season).mean(dim='season_year', skipna=True).compute()

            cmip6_regression_results[season] = {
                'slopes_pr': slopes_pr, 'p_values_pr': p_values_pr,
                'slopes_tas': slopes_tas, 'p_values_tas': p_values_tas,
                'ua850_mean': ua850_mean_orig.values if ua850_mean_orig is not None else None,
                'lons': ua_season_detrended.lon.values if 'lon' in ua_season_detrended.coords else None,
                'lats': ua_season_detrended.lat.values if 'lat' in ua_season_detrended.coords else None,
                'std_dev_pr': DataProcessor.filter_by_season(mmm_pr_box_detrended, season).std().item(),
                'std_dev_tas': DataProcessor.filter_by_season(mmm_tas_box_detrended, season).std().item()
            }
        
        logging.info("Calculation of CMIP6 regression maps finished.")
        return cmip6_regression_results

    @staticmethod
    def calculate_historical_slopes_comparison(beta_obs_slopes, cmip6_data_loaded, jet_data_reanalysis, historical_period=(1981, 2010)):
        """
        Calculates regression slopes (jet vs impact) for each CMIP6 model over a historical period
        to compare with beta_obs from reanalysis.
        """
        logging.info(f"\nCalculating historical slopes for CMIP6 models ({historical_period})...")
        cmip6_historical_slopes = {key: [] for key in beta_obs_slopes}
        hist_start, hist_end = historical_period
        box_coords = (Config.BOX_LAT_MIN, Config.BOX_LAT_MAX, Config.BOX_LON_MIN, Config.BOX_LON_MAX)

        for model, data in cmip6_data_loaded.items():
            logging.info(f"  Processing historical slopes for CMIP6 model: {model}")
            try:
                ua_hist = data.get('ua').sel(time=slice(str(hist_start), str(hist_end)))
                pr_hist = data.get('pr').sel(time=slice(str(hist_start), str(hist_end)))
                tas_hist = data.get('tas').sel(time=slice(str(hist_start), str(hist_end)))
                
                if ua_hist.time.size == 0:
                    logging.info(f"    Skipping {model}: No 'ua' data in historical period.")
                    continue

                ua_seas = DataProcessor.calculate_seasonal_means(DataProcessor.assign_season_to_dataarray(ua_hist))
                pr_seas = DataProcessor.calculate_seasonal_means(DataProcessor.assign_season_to_dataarray(pr_hist))
                tas_seas = DataProcessor.calculate_seasonal_means(DataProcessor.assign_season_to_dataarray(tas_hist))

                pr_box = DataProcessor.calculate_spatial_mean(pr_seas, *box_coords)
                tas_box = DataProcessor.calculate_spatial_mean(tas_seas, *box_coords)

                model_jets_hist = {
                    'DJF_JetSpeed': JetStreamAnalyzer.calculate_jet_speed_index(DataProcessor.filter_by_season(ua_seas, 'Winter')),
                    'JJA_JetSpeed': JetStreamAnalyzer.calculate_jet_speed_index(DataProcessor.filter_by_season(ua_seas, 'Summer')),
                    'DJF_JetLat': JetStreamAnalyzer.calculate_jet_lat_index(DataProcessor.filter_by_season(ua_seas, 'Winter')),
                    'JJA_JetLat': JetStreamAnalyzer.calculate_jet_lat_index(DataProcessor.filter_by_season(ua_seas, 'Summer'))
                }
                model_impacts_hist = {
                    'DJF_pr': DataProcessor.filter_by_season(pr_box, 'Winter'), 'JJA_pr': DataProcessor.filter_by_season(pr_box, 'Summer'),
                    'DJF_tas': DataProcessor.filter_by_season(tas_box, 'Winter'), 'JJA_tas': DataProcessor.filter_by_season(tas_box, 'Summer')
                }

                for beta_key in beta_obs_slopes:
                    jet_key, impact_var_short = beta_key.split('_vs_')
                    season_short = jet_key.split('_')[0]
                    impact_key = f"{season_short}_{impact_var_short}"
                    
                    jet_ts = model_jets_hist.get(jet_key)
                    impact_ts = model_impacts_hist.get(impact_key)

                    if jet_ts is not None and impact_ts is not None:
                        jet_detrended = DataProcessor.detrend_data(jet_ts)
                        impact_detrended = DataProcessor.detrend_data(impact_ts)

                        if jet_detrended is not None and impact_detrended is not None:
                            common_years = np.intersect1d(jet_detrended.season_year.values, impact_detrended.season_year.values)
                            if len(common_years) >= 5:
                                slope, _, _, _, _ = StatsAnalyzer.calculate_regression(
                                    jet_detrended.sel(season_year=common_years).values,
                                    impact_detrended.sel(season_year=common_years).values
                                )
                                if not np.isnan(slope):
                                    cmip6_historical_slopes[beta_key].append(slope)
            except Exception as e:
                logging.error(f"    Error processing historical slopes for {model}: {e}")
                traceback.print_exc()

        logging.info("Finished calculating CMIP6 historical slopes.")
        for key, slopes_list in cmip6_historical_slopes.items():
            logging.info(f"  Historical slopes calculated for {key}: {len(slopes_list)} models")
        return cmip6_historical_slopes

    @staticmethod
    def calculate_regression_grid(index, ua_data, lat_indices, lon_indices, lat_dim_name, lon_dim_name):
        """Helper function for parallel regression calculation."""
        local_slopes = np.full((len(lat_indices), len(lon_indices)), np.nan)
        local_p_values = np.full((len(lat_indices), len(lon_indices)), np.nan)
        
        for i, lat_idx in enumerate(lat_indices):
            for j, lon_idx in enumerate(lon_indices):
                ua_timeseries = ua_data.isel({lat_dim_name: lat_idx, lon_dim_name: lon_idx}).values
                slope, _, _, p_value, _ = StatsAnalyzer.calculate_regression(index, ua_timeseries)
                local_slopes[i, j] = slope
                local_p_values[i, j] = p_value
        return local_slopes, local_p_values, lat_indices, lon_indices

    @staticmethod
    def _calculate_regression_for_variable(index_da, field_da):
        """Helper to run grid-cell regression in parallel, aligning data first."""
        if index_da is None or field_da is None:
            return None, None
            
        common_years = np.intersect1d(index_da.season_year.values, field_da.season_year.values)
        if len(common_years) < 5:
            logging.warning("Not enough common years for regression.")
            return None, None
            
        index_common = index_da.sel(season_year=common_years)
        field_common = field_da.sel(season_year=common_years)

        lat_dim, lon_dim = 'lat', 'lon'
        lat_size = field_common.sizes.get(lat_dim, 0)
        lon_size = field_common.sizes.get(lon_dim, 0)

        if lat_size == 0 or lon_size == 0:
            logging.warning("Field data has zero size in lat or lon dimension.")
            return None, None
            
        lat_indices = list(range(lat_size))
        lon_indices = list(range(lon_size))

        slopes = np.full((lat_size, lon_size), np.nan)
        p_values = np.full((lat_size, lon_size), np.nan)

        with ProcessPoolExecutor(max_workers=Config.N_PROCESSES) as executor:
            futures = [executor.submit(StorylineAnalyzer.calculate_regression_grid, index_common.values, field_common, [i], lon_indices, lat_dim, lon_dim) for i in lat_indices]
            for future in as_completed(futures):
                local_slopes, local_p_values, res_lat_chunk, _ = future.result()
                lat_idx = res_lat_chunk[0]
                slopes[lat_idx, :] = local_slopes[0, :]
                p_values[lat_idx, :] = local_p_values[0, :]
                
        return slopes, p_values

    @staticmethod
    def calculate_regression_maps(datasets, dataset_key, regression_period=None):
        """Calculate regressions between UA850 and PR/TAS for a given reanalysis dataset."""
        logging.info(f"Calculating regression maps for {dataset_key} (normalized predictors)...")
        if regression_period:
            logging.info(f"  Using specific regression period: {regression_period[0]}-{regression_period[1]}")
        
        pr_box = datasets.get(f'{dataset_key}_pr_box_mean')
        tas_box = datasets.get(f'{dataset_key}_tas_box_mean')
        ua850_seasonal = datasets.get(f'{dataset_key}_ua850_seasonal')

        if not all(d is not None for d in [pr_box, tas_box, ua850_seasonal]):
            logging.error(f"Missing input data for regression maps for {dataset_key}.")
            return {}

        if regression_period:
            start, end = regression_period
            pr_box = pr_box.sel(season_year=slice(start, end))
            tas_box = tas_box.sel(season_year=slice(start, end))
            ua850_seasonal = ua850_seasonal.sel(season_year=slice(start, end))

        pr_box_detrended = DataProcessor.detrend_data(pr_box)
        tas_box_detrended = DataProcessor.detrend_data(tas_box)
        ua850_detrended = DataProcessor.detrend_data(ua850_seasonal)

        all_season_data = {}
        for season in ['Winter', 'Summer']:
            pr_idx_norm = StatsAnalyzer.normalize(DataProcessor.filter_by_season(pr_box_detrended, season))
            tas_idx_norm = StatsAnalyzer.normalize(DataProcessor.filter_by_season(tas_box_detrended, season))
            ua_season_detrended = DataProcessor.filter_by_season(ua850_detrended, season)
            
            if not all(d is not None for d in [pr_idx_norm, tas_idx_norm, ua_season_detrended]):
                all_season_data[season] = {'slopes_pr': None, 'p_values_pr': None, 'slopes_tas': None, 'p_values_tas': None}
                continue

            slopes_pr, p_values_pr = StorylineAnalyzer._calculate_regression_for_variable(pr_idx_norm, ua_season_detrended)
            slopes_tas, p_values_tas = StorylineAnalyzer._calculate_regression_for_variable(tas_idx_norm, ua_season_detrended)
            
            all_season_data[season] = {
                'slopes_pr': slopes_pr, 'p_values_pr': p_values_pr,
                'slopes_tas': slopes_tas, 'p_values_tas': p_values_tas,
                'ua850_mean': DataProcessor.filter_by_season(ua850_seasonal, season).mean('season_year').values,
                'lons': ua850_seasonal.lon.values, 'lats': ua850_seasonal.lat.values,
                'std_dev_pr': DataProcessor.filter_by_season(pr_box_detrended, season).std().item(),
                'std_dev_tas': DataProcessor.filter_by_season(tas_box_detrended, season).std().item()
            }
        return all_season_data

    @staticmethod
    def calculate_jet_impact_maps(datasets, jet_data, dataset_key, season):
        """
        Calculate regression maps showing the impact of jet indices on PR and TAS.
        The jet index (predictor) is normalized, so the slope is per std. dev.
        """
        logging.info(f"Calculating jet impact maps for {dataset_key} ({season})...")
        impact_maps = {season: {}}
        
        tas_seasonal = datasets[f'{dataset_key}_tas_seasonal']
        pr_seasonal = datasets[f'{dataset_key}_pr_seasonal']
        
        for jet_type in ['speed', 'lat']:
            jet_data_key = f'{dataset_key}_{season.lower()}_{jet_type}_data'
            jet_bundle = jet_data.get(jet_data_key)
            if jet_bundle is None or 'jet' not in jet_bundle or jet_bundle['jet'] is None:
                logging.warning(f"No valid jet data found for key: {jet_data_key}")
                continue

            jet_index_detrended = jet_bundle['jet']
            
            # Normalize the predictor for the regression
            jet_index_normalized = StatsAnalyzer.normalize(jet_index_detrended)
            
            # Get the standard deviation of the original (detrended) predictor for the plot label
            std_dev_predictor = jet_index_detrended.std().item()

            for var_name, var_data in [('tas', tas_seasonal), ('pr', pr_seasonal)]:
                var_data_season = DataProcessor.filter_by_season(var_data, season)
                var_data_detrended = DataProcessor.detrend_data(var_data_season)
                
                if var_data_detrended is None:
                    continue

                # Use the NORMALIZED jet index for the regression
                slopes, p_values = StorylineAnalyzer._calculate_regression_for_variable(jet_index_normalized, var_data_detrended)
                
                if slopes is not None and p_values is not None:
                    impact_key = f'jet_{jet_type}_{var_name}'
                    impact_maps[season][impact_key] = {
                        'slopes': slopes,
                        'p_values': p_values,
                        'lons': var_data.lon.values,
                        'lats': var_data.lat.values,
                        'std_dev_predictor': std_dev_predictor,  # Store the std dev for the plot
                        'common_years': np.intersect1d(jet_index_normalized.season_year.values, var_data_detrended.season_year.values)
                    }
        return impact_maps

    @staticmethod
    def calculate_jet_correlation_maps(datasets, jet_data, dataset_key, season):
        """
        Calculate grid-cell correlations between jet indices and climate variables.
        This function handles the computation, returning data ready for plotting.
        """
        logging.info(f"Calculating jet correlation maps for {dataset_key} ({season})...")
        correlation_results = {}
        season_lower = season.lower()

        # Define the combinations of jet indices and variables to correlate
        correlation_configs = [
            {'jet_type': 'speed', 'var_type': 'tas'},
            {'jet_type': 'speed', 'var_type': 'pr'},
            {'jet_type': 'lat',   'var_type': 'tas'},
            {'jet_type': 'lat',   'var_type': 'pr'}
        ]

        # Get the base seasonal data fields for the dataset
        tas_seasonal = datasets.get(f'{dataset_key}_tas_seasonal')
        pr_seasonal = datasets.get(f'{dataset_key}_pr_seasonal')
        if tas_seasonal is None or pr_seasonal is None:
            logging.error(f"Missing seasonal TAS or PR data for {dataset_key}. Cannot calculate correlation maps.")
            return {}

        # Filter and detrend the data for the specified season
        var_data_detrended = {
            'tas': DataProcessor.detrend_data(DataProcessor.filter_by_season(tas_seasonal, season)),
            'pr': DataProcessor.detrend_data(DataProcessor.filter_by_season(pr_seasonal, season))
        }

        for config in correlation_configs:
            jet_type = config['jet_type']
            var_type = config['var_type']
            result_key = f"jet_{jet_type}_{var_type}"
            logging.debug(f"  Processing: {dataset_key} {season} - Jet {jet_type} vs. {var_type.upper()}")

            # Get the corresponding detrended jet index data
            jet_data_key = f'{dataset_key}_{season_lower}_{jet_type}_data'
            jet_bundle = jet_data.get(jet_data_key)
            var_field = var_data_detrended.get(var_type)

            if jet_bundle is None or 'jet' not in jet_bundle or jet_bundle['jet'] is None or var_field is None:
                logging.warning(f"    Skipping {result_key}: Missing jet data or variable field.")
                continue

            jet_index_ts = jet_bundle['jet']
            jet_index_normalized = StatsAnalyzer.normalize(jet_index_ts)
            std_dev_jet = jet_index_ts.std().item()

            # Find common years for the regression
            common_years = np.intersect1d(jet_index_normalized.season_year.values, var_field.season_year.values)
            if len(common_years) < 5:
                logging.warning(f"    Skipping {result_key}: Not enough common years ({len(common_years)}).")
                continue

            # Select data for common years
            jet_index_common = jet_index_normalized.sel(season_year=common_years)
            var_field_common = var_field.sel(season_year=common_years)
            
            # Use the helper to calculate regression slopes using the normalized predictor
            slopes, p_values = StorylineAnalyzer._calculate_regression_for_variable(jet_index_common, var_field_common)

            if slopes is not None and p_values is not None:
                correlation_results[result_key] = {
                    'slopes': slopes,
                    'p_values': p_values,
                    'lons': var_field.lon.values,
                    'lats': var_field.lat.values,
                    'std_dev_jet': std_dev_jet
                }
        return correlation_results

    @staticmethod
    def analyze_correlations(datasets, discharge_data, jet_data, dataset_key):
        """Analyze specific correlations between climate indices and target variables."""
        logging.info(f"Analyzing selected correlations for {dataset_key} (Winter & Summer)...")
        all_correlations = {'winter': {}, 'summer': {}, 'pr': {'winter': {}, 'summer': {}}, 'tas': {'winter': {}, 'summer': {}}}
        
        for season in ['winter', 'summer']:
            for var in ['discharge', 'extreme_flow']:
                key = f'{season}_{var}'
                if discharge_data and key in discharge_data:
                    discharge_ts = discharge_data[key]
                    jet_speed_bundle = jet_data.get(f'{dataset_key}_{season}_speed_tas_data')
                    if jet_speed_bundle and 'jet' in jet_speed_bundle:
                        jet_ts = jet_speed_bundle['jet']
                        common_years = np.intersect1d(discharge_ts.season_year, jet_ts.season_year)
                        if len(common_years) > 5:
                            s, i, r, p, e = StatsAnalyzer.calculate_regression(
                                jet_ts.sel(season_year=common_years).values,
                                discharge_ts.sel(season_year=common_years).values
                            )
                            if not np.isnan(r):
                                all_correlations[season][f'{var}_jet_speed'] = {'r_value': r, 'p_value': p, 'slope': s, 'intercept': i}

            for var in ['pr', 'tas']:
                for jet_type in ['speed', 'lat']:
                    bundle_key = f'{dataset_key}_{season}_{jet_type}_{var}_data'
                    bundle = jet_data.get(bundle_key)
                    if bundle and 'jet' in bundle and var in bundle:
                        s, i, r, p, e = StatsAnalyzer.calculate_regression(bundle['jet'].values, bundle[var].values)
                        if not np.isnan(r):
                            all_correlations[var][season][jet_type] = {'r_value': r, 'p_value': p, 'slope': s, 'intercept': i}

        return all_correlations

    @staticmethod
    def analyze_amo_jet_correlations(jet_data, amo_data, season, window_size=15):
        """Analyze correlations between AMO and jet indices with rolling means."""
        logging.info(f"Analyzing AMO-Jet correlations ({season}, {window_size}-year rolling means)...")
        correlations = {}
        season_lower = season.lower()
        amo_seasonal_detrended = amo_data.get(f'amo_{season_lower}_detrended')
        if amo_seasonal_detrended is None: return {}
        
        amo_smooth = StatsAnalyzer.calculate_rolling_mean(amo_seasonal_detrended, window=window_size)
        
        for dataset_key in [Config.DATASET_20CRV3, Config.DATASET_ERA5]:
            dataset_correlations = {}
            for jet_type in ['speed', 'lat']:
                # This is the line corrected in the previous step
                jet_bundle = jet_data.get(f'{dataset_key}_{season_lower}_{jet_type}_data')
                
                if jet_bundle and 'jet' in jet_bundle:
                    jet_detrended = jet_bundle['jet']
                    jet_smooth = StatsAnalyzer.calculate_rolling_mean(jet_detrended, window=window_size)
                    
                    common_years = np.intersect1d(jet_smooth.season_year.values, amo_smooth.season_year.values)
                    if len(common_years) > window_size:
                        s, i, r, p, e = StatsAnalyzer.calculate_regression(
                            amo_smooth.sel(season_year=common_years).values,
                            jet_smooth.sel(season_year=common_years).values
                        )
                        if not np.isnan(r):
                            # --- START MODIFICATION ---
                            # Use 'latitude' as the key to match the plotting function's expectation
                            result_key = 'latitude' if jet_type == 'lat' else 'speed'
                            
                            # Store the actual data needed for the plot, not just the result
                            dataset_correlations[result_key] = {
                                'r_value': r, 
                                'p_value': p, 
                                'window_size': window_size,
                                'common_years': common_years,
                                'amo_values': amo_smooth.sel(season_year=common_years).values,
                                'jet_values': jet_smooth.sel(season_year=common_years).values
                            }
                            # --- END MODIFICATION ---

            if dataset_correlations:
                correlations[dataset_key] = dataset_correlations
        return correlations

    @staticmethod
    def analyze_all_correlations_for_bar_chart(datasets_reanalysis, jet_data_reanalysis, discharge_data, amo_data, season):
        """
        Analyzes all required correlations for the summary bar chart plot for a specific season.
        This method gathers data, calculates correlations, and returns a structured DataFrame.
        NOW INCLUDES AMO-INDEX correlation with SPEI.
        """
        logging.info(f"Gathering all correlation data for summary bar chart ({season})...")
        all_results = []
        season_lower = season.lower()

        # Defines all correlation pairs to be calculated and plotted
        plot_configs = [
            {'title': f'{season} Temp vs Jet Speed',       'var1_key': 'tas',          'var2_key': 'speed', 'category': 'Temperature'},
            {'title': f'{season} Precip vs Jet Lat',       'var1_key': 'pr',           'var2_key': 'lat',   'category': 'Precipitation'},
            {'title': f'{season} Temp vs Jet Lat',         'var1_key': 'tas',          'var2_key': 'lat',   'category': 'Temperature'},
            {'title': f'{season} Discharge vs Jet Speed',  'var1_key': 'discharge',    'var2_key': 'speed', 'category': 'Discharge'},
            {'title': f'{season} Extreme Flow vs Jet Speed', 'var1_key': 'extreme_flow', 'var2_key': 'speed', 'category': 'Discharge'},
            {'title': f'{season} Precip vs Jet Speed',     'var1_key': 'pr',           'var2_key': 'speed', 'category': 'Precipitation'},
            {'title': f'{season} Discharge vs SPEI-4',     'var1_key': 'discharge',    'var2_key': 'spei4', 'category': 'Drought Index'},
            {'title': f'{season} Extreme Flow vs SPEI-4',  'var1_key': 'extreme_flow', 'var2_key': 'spei4', 'category': 'Drought Index'},
            # --- KORRIGIERTE EINTRÄGE (AMO statt NAO) ---
            {'title': f'{season} SPEI-4 vs. AMO Index',    'var1_key': 'spei4',        'var2_key': 'amo', 'category': 'Teleconnections'},
        ]

        for dataset_key in [Config.DATASET_20CRV3, Config.DATASET_ERA5]:
            for config in plot_configs:
                # 1. Get the first timeseries (impact variable)
                var1_ts = None
                if config['var1_key'] in ['discharge', 'extreme_flow']:
                    if discharge_data:
                        var1_ts = discharge_data.get(f"{season_lower}_{config['var1_key']}")
                elif config['var1_key'] == 'spei4':
                    spei_data = datasets_reanalysis.get(f"{dataset_key}_spei4")
                    if spei_data is not None:
                        var1_ts = DataProcessor.detrend_data(DataProcessor.filter_by_season(spei_data, season))
                else:  # 'pr' or 'tas'
                    pr_tas_seasonal = datasets_reanalysis.get(f"{dataset_key}_{config['var1_key']}_box_mean")
                    if pr_tas_seasonal is not None:
                        var1_ts = DataProcessor.detrend_data(DataProcessor.filter_by_season(pr_tas_seasonal, season))

                # 2. Get the second timeseries (predictor: jet, SPEI, or AMO)
                var2_ts = None
                if config['var2_key'] == 'spei4':
                    spei_data = datasets_reanalysis.get(f"{dataset_key}_spei4")
                    if spei_data is not None:
                        var2_ts = DataProcessor.detrend_data(DataProcessor.filter_by_season(spei_data, season))
                elif config['var2_key'] == 'amo':
                    if amo_data:
                        # For teleconnections, we often use the detrended index
                        var2_ts = amo_data.get(f"amo_{season_lower}_detrended")
                else: # It's a jet index
                    jet_data_key = f"{dataset_key}_{season_lower}_{config['var2_key']}_data"
                    var2_ts = jet_data_reanalysis.get(jet_data_key, {}).get('jet')

                if var1_ts is None or var2_ts is None or var1_ts.size == 0 or var2_ts.size == 0:
                    continue

                # 3. Find common years and calculate correlation
                common_years, idx1, idx2 = np.intersect1d(var1_ts.season_year.values, var2_ts.season_year.values, return_indices=True)
                if len(common_years) < 5:
                    continue

                vals1 = var1_ts.values[idx1]
                vals2 = var2_ts.values[idx2]

                # Predictor (var2) is X, Impact (var1) is Y
                _, _, r_val, p_val, _ = StatsAnalyzer.calculate_regression(vals2, vals1)

                if not np.isnan(r_val):
                    base_label = config['title'].replace(f"{season} ", "").replace("vs", "vs.")
                    if "Jet" in base_label:
                        base_label = base_label.replace("Jet", "Jet Index")

                    all_results.append({
                        'correlation': r_val,
                        'p_value': p_val,
                        'base_label': base_label,
                        'dataset': dataset_key,
                        'category': config['category']
                    })
        
        if not all_results:
            return pd.DataFrame()
            
        return pd.DataFrame(all_results)

    @staticmethod
    def analyze_amo_jet_correlations_for_plot(jet_data, amo_data, window_size=15):
        """
        Analyzes AMO-Jet correlations for both Winter and Summer and returns a
        dictionary structured for the comparison plot.
        """
        logging.info(f"Analyzing AMO-Jet correlations for Winter and Summer for plotting...")
        
        winter_correlations = StorylineAnalyzer.analyze_amo_jet_correlations(
            jet_data=jet_data, amo_data=amo_data, season='Winter', window_size=window_size
        )
        
        summer_correlations = StorylineAnalyzer.analyze_amo_jet_correlations(
            jet_data=jet_data, amo_data=amo_data, season='Summer', window_size=window_size
        )
        
        # This structure matches what the new plotting function expects
        return {
            'Winter': winter_correlations,
            'Summer': summer_correlations
        }

    @staticmethod
    def analyze_timeseries_for_projection_plot(cmip6_results, datasets_reanalysis, config):
        """
        Prepares CMIP6 and reanalysis data for the climate projection timeseries plot.
        This version includes more robust data cleaning to prevent warnings from invalid model values.
        """
        logging.info("Preparing data for climate projection timeseries plot (all four jet indices)...")
        
        # Initialisierung für alle vier Indizes
        cmip6_plot_data = {'Global_Tas': {'members': [], 'mmm': None},
                            'JJA_JetLat': {'members': [], 'mmm': None},
                            'DJF_JetSpeed': {'members': [], 'mmm': None},
                            'JJA_JetSpeed': {'members': [], 'mmm': None},
                            'DJF_JetLat': {'members': [], 'mmm': None}}
        reanalysis_plot_data = {'Global_Tas': {}, 'JJA_JetLat': {}, 'DJF_JetSpeed': {}, 
                                'JJA_JetSpeed': {}, 'DJF_JetLat': {}}

        rolling_window = 20
        pi_ref_start = config.CMIP6_PRE_INDUSTRIAL_REF_START
        pi_ref_end = config.CMIP6_PRE_INDUSTRIAL_REF_END

        def _get_anomaly_and_smooth(data_array, year_coord, ref_start, ref_end, window):
            """Interne Hilfsfunktion zur Berechnung von Anomalien und gleitenden Mitteln."""
            if data_array is None or data_array.size == 0: return None
            
            # [KORREKTUR] Filtere unrealistische Werte, bevor sie in die Berechnung eingehen
            sane_data = data_array.where(np.abs(data_array) < 1e10)

            try:
                ref_period_data = sane_data.sel({year_coord: slice(ref_start, ref_end)})
                if ref_period_data.sizes.get(year_coord, 0) == 0:
                    logging.warning(f"No valid data in reference period {ref_start}-{ref_end} for an index.")
                    # Fallback: Benutze alle verfügbaren Daten für den Referenzwert, wenn der Zeitraum leer ist
                    if sane_data.sizes.get(year_coord, 0) > 0:
                        ref_period_data = sane_data
                    else:
                        return None
                
                ref_mean = ref_period_data.mean(dim=year_coord, skipna=True)
                anomaly = sane_data - ref_mean
                
                if anomaly.sizes.get(year_coord, 0) >= window:
                    return anomaly.rolling({year_coord: window}, center=True).mean().dropna(dim=year_coord)
                return anomaly # Gib die nicht geglättete Anomalie zurück, wenn die Zeitreihe zu kurz ist
            except Exception as e:
                logging.error(f"Error in _get_anomaly_and_smooth: {e}")
                return None

        # --- 1. CMIP6-Daten verarbeiten ---
        if cmip6_results and 'cmip6_model_data_loaded' in cmip6_results and 'model_metric_timeseries' in cmip6_results:
            cmip6_data = cmip6_results['cmip6_model_data_loaded']
            cmip6_metrics = cmip6_results['model_metric_timeseries']

            for key, model_data in cmip6_data.items():
                tas_global_monthly = model_data.get(f"{config.CMIP6_GLOBAL_TAS_VAR}_global")
                if tas_global_monthly is not None:
                    tas_annual = tas_global_monthly.groupby('time.year').mean('time', skipna=True)
                    tas_annual.name = "Global_Tas"
                    processed_tas = _get_anomaly_and_smooth(tas_annual, 'year', pi_ref_start, pi_ref_end, rolling_window)
                    if processed_tas is not None:
                        cmip6_plot_data['Global_Tas']['members'].append(processed_tas)
            
            for jet_key in ['JJA_JetLat', 'DJF_JetSpeed', 'JJA_JetSpeed', 'DJF_JetLat']:
                for model_key, metrics in cmip6_metrics.items():
                    jet_timeseries = metrics.get(jet_key)
                    if jet_timeseries is not None:
                        processed_jet = _get_anomaly_and_smooth(jet_timeseries, 'season_year', pi_ref_start, pi_ref_end, rolling_window)
                        if processed_jet is not None:
                            cmip6_plot_data[jet_key]['members'].append(processed_jet)

            # MMM-Berechnung
            for key, data in cmip6_plot_data.items():
                if data['members']:
                    try:
                        time_coord = 'year' if key == 'Global_Tas' else 'season_year'
                        
                        # [KORREKTUR] Filtere None-Werte und leere Arrays robuster
                        valid_members = [m for m in data['members'] if m is not None and m.size > 0 and time_coord in m.dims]
                        if not valid_members:
                            logging.warning(f"No valid members found for MMM calculation of {key}.")
                            continue

                        # Sanitize coordinates before concatenation
                        sanitized_members = []
                        for m in valid_members:
                            coords_to_drop = [c for c in m.coords if c not in m.dims and c != time_coord]
                            if coords_to_drop:
                                sanitized_members.append(m.drop_vars(coords_to_drop, errors='ignore'))
                            else:
                                sanitized_members.append(m)

                        if sanitized_members:
                            combined = xr.concat(sanitized_members, dim='member', join='outer', combine_attrs="drop_conflicts")
                            # Berechne MMM aus den bereits bereinigten Member-Daten
                            cmip6_plot_data[key]['mmm'] = combined.mean('member', skipna=True)
                            logging.info(f"Successfully calculated MMM for CMIP6 {key} from {len(sanitized_members)} members.")
                        else:
                            logging.warning(f"MMM calculation for {key} skipped, no sanitized members available.")
                            
                    except Exception as e:
                        logging.error(f"Failed to calculate MMM for CMIP6 {key}: {e}")
                        traceback.print_exc()

        # --- 2. Reanalyse-Daten verarbeiten ---
        logging.info("Preparing reanalysis data (Global Tas and all four indices), aligning ERA5 to 20CRv3 baseline...")

        def _calculate_global_mean_tas(da):
            if da is None or 'lat' not in da.dims or 'lon' not in da.dims:
                return None
            weights = np.cos(np.deg2rad(da.lat))
            weights.name = "weights"
            global_mean = da.weighted(weights).mean(dim=("lat", "lon"), skipna=True)
            return global_mean.groupby('time.year').mean('time', skipna=True)

        tas_20crv3_monthly = datasets_reanalysis.get('20CRv3_tas_monthly')
        tas_era5_monthly = datasets_reanalysis.get('ERA5_tas_monthly')

        tas_20crv3_annual = _calculate_global_mean_tas(tas_20crv3_monthly)
        if tas_20crv3_annual is not None:
            processed_tas_20crv3 = _get_anomaly_and_smooth(tas_20crv3_annual, 'year', pi_ref_start, pi_ref_end, rolling_window)
            if processed_tas_20crv3 is not None:
                reanalysis_plot_data['Global_Tas']['20CRv3'] = processed_tas_20crv3

        tas_era5_annual = _calculate_global_mean_tas(tas_era5_monthly)
        if tas_era5_annual is not None and tas_20crv3_annual is not None:
            common_years_tas = np.intersect1d(tas_20crv3_annual.year.values, tas_era5_annual.year.values)
            if len(common_years_tas) > 5:
                pi_mean_20crv3 = tas_20crv3_annual.sel(year=slice(pi_ref_start, pi_ref_end)).mean(skipna=True).compute().item()
                mean_20crv3_overlap = tas_20crv3_annual.sel(year=common_years_tas).mean(skipna=True).compute().item()
                mean_era5_overlap = tas_era5_annual.sel(year=common_years_tas).mean(skipna=True).compute().item()
                offset = mean_era5_overlap - mean_20crv3_overlap
                
                adjusted_era5 = tas_era5_annual - offset
                anomaly_era5 = adjusted_era5 - pi_mean_20crv3
                
                if anomaly_era5.year.size >= rolling_window:
                    smoothed_anomaly_era5 = anomaly_era5.rolling({'year': rolling_window}, center=True).mean().dropna(dim='year')
                    reanalysis_plot_data['Global_Tas']['ERA5'] = smoothed_anomaly_era5
                else:
                    reanalysis_plot_data['Global_Tas']['ERA5'] = anomaly_era5

        def get_abs_indices(dset_key):
            ua_seasonal = datasets_reanalysis.get(f'{dset_key}_ua850_seasonal')
            if ua_seasonal is None: return {}
            djf_ua = DataProcessor.filter_by_season(ua_seasonal, 'Winter')
            jja_ua = DataProcessor.filter_by_season(ua_seasonal, 'Summer')
            return {
                'DJF_JetSpeed': JetStreamAnalyzer.calculate_jet_speed_index(djf_ua),
                'DJF_JetLat': JetStreamAnalyzer.calculate_jet_lat_index(djf_ua),
                'JJA_JetSpeed': JetStreamAnalyzer.calculate_jet_speed_index(jja_ua),
                'JJA_JetLat': JetStreamAnalyzer.calculate_jet_lat_index(jja_ua)
            }

        indices_20crv3 = get_abs_indices("20CRv3")
        indices_era5 = get_abs_indices("ERA5")

        for jet_key in ['DJF_JetSpeed', 'JJA_JetLat', 'DJF_JetLat', 'JJA_JetSpeed']:
            ts_20crv3 = indices_20crv3.get(jet_key)
            ts_era5 = indices_era5.get(jet_key)
            
            if ts_20crv3 is None: continue

            pi_mean_20crv3 = ts_20crv3.sel(season_year=slice(pi_ref_start, pi_ref_end)).mean(skipna=True).compute()
            anomaly_20crv3 = ts_20crv3 - pi_mean_20crv3
            reanalysis_plot_data[jet_key]['20CRv3'] = anomaly_20crv3.rolling({'season_year': rolling_window}, center=True).mean().dropna(dim='season_year')

            if ts_era5 is None: continue

            common_years = np.intersect1d(ts_20crv3.season_year.values, ts_era5.season_year.values)
            if len(common_years) > 5:
                mean_20crv3_overlap = ts_20crv3.sel(season_year=common_years).mean(skipna=True).compute()
                mean_era5_overlap = ts_era5.sel(season_year=common_years).mean(skipna=True).compute()
                offset = mean_era5_overlap - mean_20crv3_overlap
                adjusted_era5 = ts_era5 - offset
                anomaly_era5 = adjusted_era5 - pi_mean_20crv3
                reanalysis_plot_data[jet_key]['ERA5'] = anomaly_era5.rolling({'season_year': rolling_window}, center=True).mean().dropna(dim='season_year')

        return cmip6_plot_data, reanalysis_plot_data

    @staticmethod
    def calculate_reanalysis_betas(datasets_reanalysis, jet_data_reanalysis, dataset_key='ERA5'):
        """
        Calculates the regression slopes (beta_obs) between jet indices and climate impacts
        for a specific reanalysis dataset (typically ERA5).
        NOW CALCULATES ALL 8 COMBINATIONS.
        """
        logging.info(f"Calculating all 8 beta_obs slopes from {dataset_key} reanalysis...")
        beta_obs_slopes = {}
        
        # Define all 8 relationships needed for the full comparison
        beta_configs = [
            # Winter Relationships
            {'jet_key_part': 'DJF_JetSpeed', 'impact_var_key': 'tas', 'season': 'Winter', 'jet_type': 'speed'},
            {'jet_key_part': 'DJF_JetSpeed', 'impact_var_key': 'pr',  'season': 'Winter', 'jet_type': 'speed'},
            {'jet_key_part': 'DJF_JetLat',   'impact_var_key': 'tas', 'season': 'Winter', 'jet_type': 'lat'},
            {'jet_key_part': 'DJF_JetLat',   'impact_var_key': 'pr',  'season': 'Winter', 'jet_type': 'lat'},
            # Summer Relationships
            {'jet_key_part': 'JJA_JetSpeed', 'impact_var_key': 'tas', 'season': 'Summer', 'jet_type': 'speed'},
            {'jet_key_part': 'JJA_JetSpeed', 'impact_var_key': 'pr',  'season': 'Summer', 'jet_type': 'speed'},
            {'jet_key_part': 'JJA_JetLat',   'impact_var_key': 'tas', 'season': 'Summer', 'jet_type': 'lat'},
            {'jet_key_part': 'JJA_JetLat',   'impact_var_key': 'pr',  'season': 'Summer', 'jet_type': 'lat'}
        ]

        for config in beta_configs:
            # Construct the final key for the output dictionary (e.g., 'DJF_JetSpeed_vs_tas')
            beta_key_name = f"{config['jet_key_part']}_vs_{config['impact_var_key']}"
            
            # Get the jet timeseries (already detrended from main workflow)
            jet_data_key = f"{dataset_key}_{config['season'].lower()}_{config['jet_type']}_data"
            jet_ts = jet_data_reanalysis.get(jet_data_key, {}).get('jet')

            # Get the impact timeseries (detrended)
            impact_box_mean = datasets_reanalysis.get(f"{dataset_key}_{config['impact_var_key']}_box_mean")
            impact_ts = None
            if impact_box_mean is not None:
                impact_ts = DataProcessor.detrend_data(
                    DataProcessor.filter_by_season(impact_box_mean, config['season'])
                )
            
            if jet_ts is None or impact_ts is None:
                logging.warning(f"  Could not calculate beta for '{beta_key_name}': Missing jet or impact data.")
                beta_obs_slopes[beta_key_name] = np.nan
                continue

            # Find common years and calculate regression
            common_years, idx1, idx2 = np.intersect1d(jet_ts.season_year.values, impact_ts.season_year.values, return_indices=True)
            
            if len(common_years) < 5:
                logging.warning(f"  Could not calculate beta for '{beta_key_name}': Not enough common years.")
                beta_obs_slopes[beta_key_name] = np.nan
                continue

            jet_vals = jet_ts.values[idx1]
            impact_vals = impact_ts.values[idx2]

            slope, _, _, _, _ = StatsAnalyzer.calculate_regression(jet_vals, impact_vals)
            beta_obs_slopes[beta_key_name] = slope
            logging.info(f"  - Calculated Beta_obs for {beta_key_name}: {slope:.3f}")
            
        return beta_obs_slopes
    
    @staticmethod
    def calculate_future_temporal_slopes(cmip6_results, beta_keys, gwls_to_analyze):
        """
        Calculates temporal regression slopes for each CMIP6 model over its specific
        future GWL period. This results in a distribution of slopes for each GWL.

        Parameters:
        -----------
        cmip6_results : dict
            The main results dictionary containing model timeseries and GWL years.
        beta_keys : list
            A list of the beta relationship keys to analyze (e.g., 'DJF_JetSpeed_vs_tas').
        gwls_to_analyze : list
            A list of the global warming levels to analyze (e.g., [2.0, 3.0]).

        Returns:
        --------
        dict
            A nested dictionary with the slope distributions: {beta_key: {gwl: [list_of_slopes]}}
        """
        logging.info("Calculating future temporal slopes for each CMIP6 model and GWL...")
        future_slopes = {key: {gwl: [] for gwl in gwls_to_analyze} for key in beta_keys}

        model_timeseries = cmip6_results.get('model_metric_timeseries', {})
        gwl_years = cmip6_results.get('gwl_threshold_years', {})
        window = Config.GWL_YEARS_WINDOW

        for beta_key in beta_keys:
            jet_key, impact_key_part = beta_key.split('_vs_')
            season_prefix = jet_key.split('_')[0]
            impact_key = f"{season_prefix}_{impact_key_part}"

            for model_run_key, metrics in model_timeseries.items():
                for gwl in gwls_to_analyze:
                    threshold_year = gwl_years.get(model_run_key, {}).get(gwl)
                    if threshold_year is None:
                        continue

                    jet_ts = metrics.get(jet_key)
                    impact_ts = metrics.get(impact_key)

                    if jet_ts is None or impact_ts is None:
                        continue

                    # Define the N-year window around the GWL threshold year
                    start_year, end_year = threshold_year - window // 2, threshold_year + (window - 1) // 2
                    
                    jet_slice = jet_ts.sel(season_year=slice(start_year, end_year))
                    impact_slice = impact_ts.sel(season_year=slice(start_year, end_year))
                    
                    # Ensure we have enough data for a meaningful regression
                    common_years = np.intersect1d(jet_slice.season_year.values, impact_slice.season_year.values)
                    if len(common_years) < 5:
                        continue
                        
                    # Detrend the slices before calculating the slope
                    jet_detrended = DataProcessor.detrend_data(jet_slice)
                    impact_detrended = DataProcessor.detrend_data(impact_slice)

                    if jet_detrended is not None and impact_detrended is not None:
                        slope, _, _, _, _ = StatsAnalyzer.calculate_regression(
                            jet_detrended.values, impact_detrended.values
                        )
                        if not np.isnan(slope):
                            future_slopes[beta_key][gwl].append(slope)
        
        for key, gwl_data in future_slopes.items():
            for gwl, slopes in gwl_data.items():
                logging.info(f"  - Calculated {len(slopes)} future temporal slopes for {key} at +{gwl}°C GWL.")

        return future_slopes
    
    @staticmethod
    def calculate_spatial_spei(datasets, dataset_key, scale=4):
        """
        Calculates a gridded SPEI field for a given dataset.
        """
        logging.info(f"Calculating spatial SPEI-{scale} for {dataset_key}...")
        pr_monthly_full = datasets.get(f'{dataset_key}_pr_monthly')
        tas_monthly_full = datasets.get(f'{dataset_key}_tas_monthly')
        
        if pr_monthly_full is None or tas_monthly_full is None:
            logging.error(f"Cannot calculate spatial SPEI for {dataset_key}: Missing monthly pr or tas data.")
            return None
        
        # We need a representative latitude for the entire grid for Thornthwaite.
        # A central latitude is a reasonable approximation for this.
        if 'lat' in pr_monthly_full.dims:
            # For the gridded calculation, we pass the actual latitude DataArray
            lat_input = pr_monthly_full.lat
        else:
            logging.error("Cannot calculate spatial SPEI: 'lat' dimension missing.")
            return None
            
        spatial_spei = DataProcessor.calculate_spei(
            pr_monthly_full, tas_monthly_full, lat=lat_input, scale=scale
        )
        return spatial_spei

    @staticmethod
    def calculate_spei_on_discharge_map(
        spatial_spei_data,
        discharge_timeseries,
        season,
        analysis_type='correlation'
    ):
        """
        Calculates a spatial map of either correlation or regression slope, showing
        the relationship between the local SPEI at each grid cell (predictor, X) 
        and a single discharge timeseries (response, Y).
        """
        logging.info(f"Calculating {analysis_type} map for local SPEI's influence on discharge for {season}...")
        
        if spatial_spei_data is None or discharge_timeseries is None:
            logging.error(f"Cannot calculate {analysis_type} map: Missing SPEI or discharge data.")
            return None, None

        # 1. Sicherstellen, dass die saisonalen Koordinaten vorhanden sind.
        if 'season' not in spatial_spei_data.coords:
            spei_with_seasons = DataProcessor.assign_season_to_dataarray(spatial_spei_data)
            if spei_with_seasons is None:
                logging.error("Failed to assign season coordinates to SPEI data.")
                return None, None
        else:
            spei_with_seasons = spatial_spei_data
        
        # 2. Nach Saison filtern und zu saisonalen Mittelwerten aggregieren
        spei_filtered = DataProcessor.filter_by_season(spei_with_seasons, season)
        if spei_filtered is None:
            logging.error(f"Filtering SPEI data for season '{season}' resulted in None.")
            return None, None
        
        spei_seasonal_mean = spei_filtered.groupby('season_year').mean(dim='time', skipna=True)
        
        # 3. Detrenden
        spei_detrended = DataProcessor.detrend_data(spei_seasonal_mean)
        discharge_detrended = discharge_timeseries # Ist bereits detrended
        
        if spei_detrended is None or discharge_detrended is None:
            logging.error("Detrending failed for SPEI or discharge data. Check previous steps.")
            return None, None
            
        # 4. Daten auf gemeinsame Jahre ausrichten
        common_years, idx_spei, idx_discharge = np.intersect1d(
            spei_detrended.season_year.values,
            discharge_detrended.season_year.values,
            return_indices=True
        )
        
        if len(common_years) < 10:
            logging.warning(f"Not enough common years ({len(common_years)}) for map calculation.")
            return None, None
            
        spei_common = spei_detrended.isel(season_year=idx_spei).load()
        discharge_common = discharge_detrended.isel(season_year=idx_discharge).load()

        # 5. Prädiktor (X) und Response (Y) definieren
        if analysis_type == 'regression':
            predictor_field = StatsAnalyzer.normalize(spei_common)
        else: 
            predictor_field = spei_common
        
        response_da = discharge_common

        # 6. Grid-weise Regression/Korrelation durchführen
        def regression_on_grid(predictor_ts, response_ts):
            # Diese Funktion wird jetzt für jede Zelle mit zwei 1D-Arrays aufgerufen.
            slope, _, r_value, p_value, _ = StatsAnalyzer.calculate_regression(predictor_ts, response_ts)
            result = r_value if analysis_type == 'correlation' else slope
            return np.array([result, p_value])

        regression_results = xr.apply_ufunc(
            regression_on_grid,
            predictor_field,
            response_da,
            input_core_dims=[['season_year'], ['season_year']],
            output_core_dims=[['outputs']],
            exclude_dims=set(('season_year',)),
            vectorize=True,  # This line is added
            dask="parallelized",
            output_dtypes=[predictor_field.dtype],
            dask_gufunc_kwargs={'output_sizes': {'outputs': 2}}
        ).compute()

        # 7. Ergebnisse aufteilen
        map_data = regression_results.isel(outputs=0)
        p_values = regression_results.isel(outputs=1)
        
        if analysis_type == 'correlation':
             map_data = map_data.clip(min=-1, max=1)

        logging.info(f"Successfully calculated {analysis_type} map.")
        return map_data, p_values
    
    @staticmethod
    def calculate_single_model_regression_maps(model_data, model_key, historical_period=(1995, 2014)):
        """Calculates regression maps for a single CMIP6 model."""
        logging.info(f"\nCalculating single-model regression maps for {model_key} for period {historical_period}...")
        if not model_data or not all(v in model_data for v in ['ua', 'pr', 'tas']):
            logging.warning(f"  Skipping {model_key} due to missing data.")
            return {}

        hist_start, hist_end = historical_period
        seasonal_means = {}
        try:
            for var in ['ua', 'pr', 'tas']:
                monthly_hist = model_data[var].sel(time=slice(str(hist_start), str(hist_end)))
                if monthly_hist.time.size == 0:
                    raise ValueError(f"No data for {var} in period {historical_period}")
                seasonal_full = DataProcessor.assign_season_to_dataarray(monthly_hist)
                seasonal_means[var] = DataProcessor.calculate_seasonal_means(seasonal_full).load()
        except Exception as e:
            logging.error(f"  Error processing data for {model_key}: {e}")
            return {}

        box_coords = (Config.BOX_LAT_MIN, Config.BOX_LAT_MAX, Config.BOX_LON_MIN, Config.BOX_LON_MAX)
        pr_box = DataProcessor.calculate_spatial_mean(seasonal_means['pr'], *box_coords)
        tas_box = DataProcessor.calculate_spatial_mean(seasonal_means['tas'], *box_coords)

        pr_box_detrended = DataProcessor.detrend_data(pr_box)
        tas_box_detrended = DataProcessor.detrend_data(tas_box)
        ua_detrended = DataProcessor.detrend_data(seasonal_means['ua'])

        regression_results = {}
        for season in ['Winter', 'Summer']:
            logging.info(f"  - Processing {season} for {model_key}")
            pr_idx_norm = StatsAnalyzer.normalize(DataProcessor.filter_by_season(pr_box_detrended, season))
            tas_idx_norm = StatsAnalyzer.normalize(DataProcessor.filter_by_season(tas_box_detrended, season))
            ua_season_detrended = DataProcessor.filter_by_season(ua_detrended, season)

            slopes_pr, p_values_pr = StorylineAnalyzer._calculate_regression_for_variable(pr_idx_norm, ua_season_detrended)
            slopes_tas, p_values_tas = StorylineAnalyzer._calculate_regression_for_variable(tas_idx_norm, ua_season_detrended)
            ua850_mean_orig = DataProcessor.filter_by_season(seasonal_means['ua'], season).mean(dim='season_year', skipna=True)

            regression_results[season] = {
                'slopes_pr': slopes_pr, 'p_values_pr': p_values_pr,
                'slopes_tas': slopes_tas, 'p_values_tas': p_values_tas,
                'ua850_mean': ua850_mean_orig.values if ua850_mean_orig is not None else None,
                'lons': ua_season_detrended.lon.values if 'lon' in ua_season_detrended.coords else None,
                'lats': ua_season_detrended.lat.values if 'lat' in ua_season_detrended.coords else None,
                'std_dev_pr': DataProcessor.filter_by_season(pr_box_detrended, season).std().item(),
                'std_dev_tas': DataProcessor.filter_by_season(tas_box_detrended, season).std().item(),
            }
        return regression_results
    
    @staticmethod
    def calculate_direct_discharge_impacts(cmip6_results, config):
        """
        Calculates the mean discharge change AND the mean standard deviation for each 2D storyline.
        The standard deviation is calculated from the absolute discharge values within the 30-year
        window for each model, and then averaged across the models in the storyline.
        """
        logging.info("Calculating direct storyline impacts for discharge (change and std dev)...")
        storyline_classification = cmip6_results.get('storyline_classification_2d')
        hist_means = cmip6_results.get('model_data_at_hist_reference')
        gwl_means = cmip6_results.get('model_data_at_gwl')
        
        # Get the timeseries data and GWL years needed for std dev calculation
        metric_timeseries = cmip6_results.get('model_metric_timeseries', {})
        gwl_years = cmip6_results.get('gwl_threshold_years', {})
        window = config.GWL_YEARS_WINDOW

        if not all([storyline_classification, hist_means, gwl_means, metric_timeseries, gwl_years]):
            logging.warning("Missing data for direct discharge impact calculation with std dev.")
            return {}

        direct_impacts = {gwl: {'DJF_discharge': {}, 'JJA_discharge': {}} for gwl in config.GLOBAL_WARMING_LEVELS}

        for gwl in config.GLOBAL_WARMING_LEVELS:
            for storyline_key, model_list in storyline_classification.get(gwl, {}).items():
                if not model_list:
                    continue
                
                season = 'DJF' if 'DJF' in storyline_key else 'JJA'
                impact_key = f"{season}_discharge"
                storyline_name = storyline_key.replace(f'{season}_', '')
                
                model_deltas = []
                model_std_devs = [] # List to store std devs from each model

                for model_run_key in model_list:
                    # Calculate the change (same as before)
                    hist_val = hist_means.get(model_run_key, {}).get(impact_key)
                    gwl_val = gwl_means.get(model_run_key, {}).get('gwl', {}).get(gwl, {}).get(impact_key)

                    if hist_val is not None and gwl_val is not None:
                        delta_absolute = gwl_val - hist_val
                        model_deltas.append(delta_absolute)

                    # Calculate the standard deviation for this model from its 30yr timeseries
                    threshold_year = gwl_years.get(model_run_key, {}).get(gwl)
                    discharge_ts = metric_timeseries.get(model_run_key, {}).get(impact_key)
                    
                    if threshold_year is not None and discharge_ts is not None:
                        start_year = threshold_year - window // 2
                        end_year = threshold_year + (window - 1) // 2
                        
                        ts_slice = discharge_ts.sel(season_year=slice(start_year, end_year))
                        
                        if ts_slice.season_year.size > 1:
                            std_dev = ts_slice.std(dim='season_year', skipna=True).item()
                            if not np.isnan(std_dev):
                                model_std_devs.append(std_dev)
                
                if model_deltas:
                    mean_change = np.mean(model_deltas)
                    mean_std_dev = np.mean(model_std_devs) if model_std_devs else np.nan
                    
                    direct_impacts[gwl][impact_key][storyline_name] = {
                        'total': mean_change,
                        'mean_std_dev': mean_std_dev # Store the mean std dev
                    }

        logging.info("Direct discharge impact calculation (change and std dev) finished.")
        return direct_impacts
    
    @staticmethod
    def analyze_extreme_discharge_frequency(historical_discharge_da, cmip6_discharge_data):
        """
        Analyzes and compares the frequency of extreme low-flow discharge events.

        Compares observed historical data against CMIP6 historical and future projections.
        Extreme events are defined based on the standard deviation of the historical observations.
        """
        logging.info("Analyzing frequency of extreme low-flow discharge events...")
        if historical_discharge_da is None or not cmip6_discharge_data:
            logging.warning("Skipping extreme discharge analysis: Missing historical or CMIP6 data.")
            return None

        # --- 1. Define thresholds from historical observations ---
        hist_mean = historical_discharge_da.mean().item()
        hist_std = historical_discharge_da.std().item()
        
        threshold_2std = hist_mean - 2 * hist_std
        threshold_3std = hist_mean - 3 * hist_std
        
        logging.info(f"  Historical Mean: {hist_mean:.2f} m³/s, Std Dev: {hist_std:.2f} m³/s")
        logging.info(f"  Threshold for < 2σ event: {threshold_2std:.2f} m³/s")
        logging.info(f"  Threshold for < 3σ event: {threshold_3std:.2f} m³/s")

        # --- 2. Calculate frequency in historical observations ---
        total_hist_months = historical_discharge_da.time.size
        count_hist_2std = (historical_discharge_da < threshold_2std).sum().item()
        count_hist_3std = (historical_discharge_da < threshold_3std).sum().item()
        
        freq_hist = {
            '2std': (count_hist_2std / total_hist_months) * 100,
            '3std': (count_hist_3std / total_hist_months) * 100
        }

        # --- 3. Calculate frequencies in CMIP6 models ---
        hist_period = (1995, 2014)
        future_period = (2070, 2099)
        
        cmip6_counts = {
            'historical': {'2std': [], '3std': [], 'total_months': []},
            'ssp245': {'2std': [], '3std': [], 'total_months': []},
            'ssp585': {'2std': [], '3std': [], 'total_months': []}
        }

        for model_scenario_key, ts in cmip6_discharge_data.items():
            scenario = model_scenario_key.split('_')[-1]
            
            # Historical period
            ts_hist = ts.sel(time=slice(str(hist_period[0]), str(hist_period[1])))
            if ts_hist.time.size > 0:
                cmip6_counts['historical']['2std'].append((ts_hist < threshold_2std).sum().item())
                cmip6_counts['historical']['3std'].append((ts_hist < threshold_3std).sum().item())
                cmip6_counts['historical']['total_months'].append(ts_hist.time.size)

            # Future period
            ts_future = ts.sel(time=slice(str(future_period[0]), str(future_period[1])))
            if ts_future.time.size > 0 and scenario in ['ssp245', 'ssp585']:
                cmip6_counts[scenario]['2std'].append((ts_future < threshold_2std).sum().item())
                cmip6_counts[scenario]['3std'].append((ts_future < threshold_3std).sum().item())
                cmip6_counts[scenario]['total_months'].append(ts_future.time.size)
        
        # --- 4. Aggregate CMIP6 results and calculate frequencies ---
        freq_cmip6 = {}
        for period, counts in cmip6_counts.items():
            if counts['total_months']:
                total_events_2std = sum(counts['2std'])
                total_events_3std = sum(counts['3std'])
                total_months = sum(counts['total_months'])
                
                freq_cmip6[period] = {
                    '2std': (total_events_2std / total_months) * 100 if total_months > 0 else 0,
                    '3std': (total_events_3std / total_months) * 100 if total_months > 0 else 0
                }
        
        # --- 5. Structure final results ---
        results = {
            'Historical Obs.': freq_hist,
            'CMIP6 Hist. (1995-2014)': freq_cmip6.get('historical', {'2std': 0, '3std': 0}),
            'CMIP6 SSP2-4.5 Future (2070-2099)': freq_cmip6.get('ssp245', {'2std': 0, '3std': 0}),
            'CMIP6 SSP5-8.5 Future (2070-2099)': freq_cmip6.get('ssp585', {'2std': 0, '3std': 0}),
            'thresholds': {'mean': hist_mean, 'std': hist_std}
        }
        
        return results

    @staticmethod
    def analyze_storyline_discharge_extremes(cmip6_results, historical_discharge_da, config, discharge_thresholds):
        """
        Analyzes the frequency and return period of low-flow discharge events
        for multiple statistical and fixed thresholds, AND calculates the change
        in interannual standard deviation.
        MODIFIED: Now returns individual model return periods AND model counts (X/Y) for each storyline.
        """
        logging.info("Analyzing storyline discharge return periods and standard deviation (with individual models)...")
        
        if historical_discharge_da is None or not discharge_thresholds:
            logging.error("Historical discharge data or fixed thresholds are missing. Aborting.")
            return None
        
        results = {}
        thresholds = {}

        storyline_order = [
            'MMM', 
            # 'Northward Shift Only', # <-- AUSKOMMENTIERT
            'Slow Jet & Northward Shift', 
            'Fast Jet & Northward Shift',
            # 'Southward Shift Only', # <-- AUSKOMMENTIERT
            'Slow Jet & Southward Shift', 
            'Fast Jet & Southward Shift',
            # 'Slow Jet Only',        # <-- AUSKOMMENTIERT
            # 'Fast Jet Only',        # <-- AUSKOMMENTIERT
            # 'Extreme NW',           # <-- AUSKOMMENTIERT
            # 'Extreme SE'            # <-- AUSKOMMENTIERT
        ]

        for season in ['Winter', 'Summer']:
            season_lower = season.lower()
            
            seasonal_data = DataProcessor.filter_by_season(
                DataProcessor.assign_season_to_dataarray(historical_discharge_da), season
            )
            if seasonal_data is None:
                logging.error(f"Could not filter historical discharge data for season: {season}")
                continue
            
            hist_seasonal = seasonal_data.groupby('season_year').mean('time')
            total_years = hist_seasonal.season_year.size
            if total_years == 0: continue

            hist_std_dev = hist_seasonal.std().item()
            
            mean = hist_seasonal.mean().item()
            std = hist_seasonal.std().item()
            
            moderate_fixed_threshold = discharge_thresholds.get(f'{season_lower}_lowflow_threshold_30')
            extreme_fixed_threshold = discharge_thresholds.get(f'{season_lower}_lowflow_threshold')

            event_definitions = {
                f'Moderate (-1σ)': {'val': mean - 1 * std},
                f'Severe (-2σ)': {'val': mean - 2 * std},
                f'Moderate Low-Flow (<{moderate_fixed_threshold} m³/s)': {'val': moderate_fixed_threshold},
                f'Extreme Low-Flow (<{extreme_fixed_threshold} m³/s)': {'val': extreme_fixed_threshold}
            }
            
            season_thresholds = {}
            for name, data in event_definitions.items():
                if data['val'] is None: continue
                
                hist_count = (hist_seasonal < data['val']).sum().item()
                hist_freq = hist_count / total_years if total_years > 0 else 0
                
                season_thresholds[name] = {
                    'val': data['val'],
                    'hist_return_period': 1 / hist_freq if hist_freq > 0 else np.inf
                }
            
            season_thresholds['historical_std_dev'] = hist_std_dev
            thresholds[season] = season_thresholds

        storyline_classification = cmip6_results.get('storyline_classification_2d')
        metric_timeseries = cmip6_results.get('model_metric_timeseries', {})
        gwl_years = cmip6_results.get('gwl_threshold_years', {})
        window = config.GWL_YEARS_WINDOW

        if not all([storyline_classification, metric_timeseries, gwl_years]):
            logging.error("Missing CMIP6 data for storyline discharge analysis.")
            return None

        for gwl in config.GLOBAL_WARMING_LEVELS:
            results[gwl] = {}
            for season in ['Winter', 'Summer']:
                if season not in thresholds: continue
                results[gwl][season] = {}
                season_label = 'DJF' if season == 'Winter' else 'JJA'
                impact_key = f"{season_label}_discharge"
                
                # Iteriere über die definierte Reihenfolge
                for storyline_name in storyline_order:
                    storyline_key = f"{season_label}_{storyline_name}"
                    model_list = storyline_classification.get(gwl, {}).get(storyline_key, [])
                    
                    if not model_list:
                        continue
                    
                    results[gwl][season][storyline_name] = {}
                    Y_total_in_storyline = len(model_list)
                    
                    model_std_devs = []
                    for model_run_key in model_list:
                        threshold_year = gwl_years.get(model_run_key, {}).get(gwl)
                        discharge_ts = metric_timeseries.get(model_run_key, {}).get(impact_key)
                        if threshold_year is None or discharge_ts is None: continue
                        
                        start_year, end_year = threshold_year - window // 2, threshold_year + (window - 1) // 2
                        ts_slice = discharge_ts.sel(season_year=slice(start_year, end_year))
                        
                        if ts_slice.season_year.size > 1:
                            std_dev = ts_slice.std(dim='season_year', skipna=True).item()
                            if not np.isnan(std_dev):
                                model_std_devs.append(std_dev)
                    
                    if model_std_devs:
                        results[gwl][season][storyline_name]['future_std_dev'] = np.mean(model_std_devs)

                    for event_name, data in thresholds[season].items():
                        if event_name == 'historical_std_dev': continue

                        model_return_periods = []
                        for model_run_key in model_list:
                            threshold_year = gwl_years.get(model_run_key, {}).get(gwl)
                            discharge_ts = metric_timeseries.get(model_run_key, {}).get(impact_key)
                            if threshold_year is None or discharge_ts is None: continue

                            start_year, end_year = threshold_year - window // 2, threshold_year + (window - 1) // 2
                            ts_slice = discharge_ts.sel(season_year=slice(start_year, end_year))

                            if ts_slice.season_year.size > 0:
                                event_count = (ts_slice < data['val']).sum().item()
                                years_analyzed = ts_slice.season_year.size
                                
                                future_freq_model = event_count / years_analyzed if years_analyzed > 0 else 0
                                future_return_period_model = 1 / future_freq_model if future_freq_model > 0 else np.inf
                                model_return_periods.append(future_return_period_model)
                        
                        if model_return_periods:
                            finite_periods = [p for p in model_return_periods if np.isfinite(p)]
                            X_finite_models = len(finite_periods)
                            mean_period = np.mean(finite_periods) if finite_periods else np.inf
                            
                            results[gwl][season][storyline_name][event_name] = {
                                'hist_return_period': data['hist_return_period'],
                                'future_return_periods_all_models': model_return_periods,
                                'future_return_period_mean': mean_period,
                                'model_count_X': X_finite_models,
                                'model_count_Y': Y_total_in_storyline
                            }

        results['thresholds'] = thresholds
        return results

    @staticmethod
    def calculate_storyline_spei_impacts(storyline_impacts, historical_monthly_data, config):
        """
        Calculates the future SPEI CHANGE for each storyline based on tas and pr changes.
        This version correctly applies seasonal deltas to a continuous monthly timeseries,
        calculates the future SPEI, and then subtracts the historical mean SPEI.
        """
        logging.info("Calculating future SPEI change impacts for each storyline...")
        spei_impacts = {gwl: {} for gwl in config.GLOBAL_WARMING_LEVELS}

        hist_pr_monthly = historical_monthly_data.get('pr_box_monthly')
        hist_tas_monthly = historical_monthly_data.get('tas_box_monthly')

        if hist_pr_monthly is None or hist_tas_monthly is None:
            logging.error("Cannot calculate SPEI impacts: Missing historical monthly data for the box.")
            return {}

        lat_center_of_box = (config.BOX_LAT_MIN + config.BOX_LAT_MAX) / 2

        # NEU: Historischen SPEI und dessen saisonale Mittelwerte berechnen (Baseline)
        hist_spei_ts_monthly = DataProcessor.calculate_spei(
            hist_pr_monthly, hist_tas_monthly, lat=lat_center_of_box, scale=4
        )
        if hist_spei_ts_monthly is None:
            logging.error("Failed to calculate historical SPEI baseline.")
            return {}
        
        hist_spei_ts_seasonal = DataProcessor.assign_season_to_dataarray(hist_spei_ts_monthly)
        hist_mean_spei = {
            'DJF': DataProcessor.filter_by_season(hist_spei_ts_seasonal, 'Winter').mean().item(),
            'JJA': DataProcessor.filter_by_season(hist_spei_ts_seasonal, 'Summer').mean().item()
        }
        logging.info(f"Historical mean SPEI calculated: DJF={hist_mean_spei['DJF']:.2f}, JJA={hist_mean_spei['JJA']:.2f}")

        # Stelle sicher, dass die monatlichen Daten eine 'season'-Koordinate haben
        hist_pr_monthly_seas = DataProcessor.assign_season_to_dataarray(hist_pr_monthly)
        hist_tas_monthly_seas = DataProcessor.assign_season_to_dataarray(hist_tas_monthly)

        for gwl, impacts in storyline_impacts.items():
            if gwl not in config.GLOBAL_WARMING_LEVELS:
                continue
            
            spei_impacts[gwl] = {'DJF_spei': {}, 'JJA_spei': {}}

            storylines_in_gwl = impacts.get('DJF_tas', {}) or impacts.get('JJA_tas', {})
            for storyline_name in storylines_in_gwl:
                
                future_tas_monthly = hist_tas_monthly_seas.copy(deep=True)
                future_pr_monthly = hist_pr_monthly_seas.copy(deep=True)

                for season_key, season_name in [('DJF', 'Winter'), ('JJA', 'Summer')]:
                    delta_tas = impacts.get(f'{season_key}_tas', {}).get(storyline_name, {}).get('total', 0)
                    delta_pr_percent = impacts.get(f'{season_key}_pr', {}).get(storyline_name, {}).get('total', 0)

                    future_tas_monthly = xr.where(
                        future_tas_monthly.season == season_name,
                        future_tas_monthly + delta_tas,
                        future_tas_monthly
                    )
                    future_pr_monthly = xr.where(
                        future_pr_monthly.season == season_name,
                        future_pr_monthly * (1 + delta_pr_percent / 100.0),
                        future_pr_monthly
                    )

                future_spei_ts_monthly = DataProcessor.calculate_spei(
                    future_pr_monthly, future_tas_monthly, lat=lat_center_of_box, scale=4
                )
                
                if future_spei_ts_monthly is not None:
                    future_spei_ts_seasonal = DataProcessor.assign_season_to_dataarray(future_spei_ts_monthly)
                    
                    for season_key, season_name in [('DJF', 'Winter'), ('JJA', 'Summer')]:
                        spei_season_filtered = DataProcessor.filter_by_season(future_spei_ts_seasonal, season_name)
                        if spei_season_filtered is not None:
                            mean_future_spei = spei_season_filtered.mean().item()
                            
                            # GEÄNDERT: Berechne die Differenz und speichere sie
                            spei_change = mean_future_spei - hist_mean_spei[season_key]
                            spei_impacts[gwl][f'{season_key}_spei'][storyline_name] = {'total': spei_change}

        logging.info("SPEI impact change calculation finished.")
        return spei_impacts
    
    @staticmethod
    def calculate_storyline_wind_change_maps(cmip6_results, config, historical_mmm_u850_by_season=None):
        """
        Calculates the 2D U850 wind change maps for each defined storyline.

        This version can accept a pre-calculated historical multi-model mean to use as
        a consistent baseline for all storylines.
        """
        logging.info("Calculating 2D U850 wind change maps for each storyline...")
        classification = cmip6_results.get('storyline_classification_2d')
        model_data_loaded = cmip6_results.get('cmip6_model_data_loaded')
        gwl_years = cmip6_results.get('gwl_threshold_years')

        if not all([classification, model_data_loaded, gwl_years]):
            logging.error("Cannot calculate wind change maps: Missing classification, model data, or GWL years.")
            return {}

        hist_period = (config.CMIP6_ANOMALY_REF_START, config.CMIP6_ANOMALY_REF_END)
        window = config.GWL_YEARS_WINDOW
        
        results = {gwl: {'DJF': {}, 'JJA': {}} for gwl in config.GLOBAL_WARMING_LEVELS}

        storyline_order = [
            'MMM', 
            # 'Northward Shift Only', # <-- AUSKOMMENTIERT
            'Slow Jet & Northward Shift', 
            'Fast Jet & Northward Shift',
            # 'Southward Shift Only', # <-- AUSKOMMENTIERT
            'Slow Jet & Southward Shift', 
            'Fast Jet & Southward Shift',
            # 'Slow Jet Only',        # <-- AUSKOMMENTIERT
            # 'Fast Jet Only',        # <-- AUSKOMMENTIERT
            # 'Extreme NW',           # <-- AUSKOMMENTIERT
            # 'Extreme SE'            # <-- AUSKOMMENTIERT
        ]

        for gwl in config.GLOBAL_WARMING_LEVELS:
            # Iteriere über die definierte Reihenfolge, um sicherzustellen, dass MMM zuerst kommt
            for storyline_name in storyline_order:
                season_keys = []
                if f'DJF_{storyline_name}' in classification.get(gwl, {}):
                    season_keys.append(('DJF', f'DJF_{storyline_name}'))
                if f'JJA_{storyline_name}' in classification.get(gwl, {}):
                    season_keys.append(('JJA', f'JJA_{storyline_name}'))

                for season, storyline_key in season_keys:
                    model_list = classification[gwl].get(storyline_key, [])
                    if not model_list:
                        continue
                
                    model_change_maps = []
                    model_hist_maps = [] # Still needed for the fallback case

                    for model_run_key in model_list:
                        ua_data = model_data_loaded.get(model_run_key, {}).get('ua')
                        threshold_year = gwl_years.get(model_run_key, {}).get(gwl)

                        if ua_data is None or threshold_year is None:
                            continue

                        ua_seasonal = DataProcessor.calculate_seasonal_means(
                            DataProcessor.assign_season_to_dataarray(ua_data)
                        )
                        
                        hist_ua_season = DataProcessor.filter_by_season(
                            ua_seasonal.sel(season_year=slice(hist_period[0], hist_period[1])),
                            'Winter' if season == 'DJF' else 'Summer'
                        )

                        start_year, end_year = threshold_year - window // 2, threshold_year + (window - 1) // 2
                        future_ua_season = DataProcessor.filter_by_season(
                            ua_seasonal.sel(season_year=slice(start_year, end_year)),
                            'Winter' if season == 'DJF' else 'Summer'
                        )

                        if hist_ua_season is None or future_ua_season is None or hist_ua_season.season_year.size == 0 or future_ua_season.season_year.size == 0:
                            continue

                        hist_map = hist_ua_season.mean(dim='season_year', skipna=True)
                        future_map = future_ua_season.mean(dim='season_year', skipna=True)
                        
                        change_map = future_map - hist_map
                        
                        model_change_maps.append(change_map)
                        model_hist_maps.append(hist_map)
                    
                    if model_change_maps:
                        aligned_changes = [da.reindex_like(model_change_maps[0], method='nearest') for da in model_change_maps]
                        mmm_change_map = xr.concat(aligned_changes, dim='model').mean(dim='model', skipna=True)
                        
                        # --- MODIFIED LOGIC ---
                        # Use the provided historical MMM if available, otherwise use the old method as a fallback.
                        if historical_mmm_u850_by_season and season in historical_mmm_u850_by_season:
                            mmm_hist_map = historical_mmm_u850_by_season[season]
                            logging.info(f"  -> Using consistent MMM historical baseline for GWL {gwl}°C, {season}, {storyline_name}")
                        else:
                            aligned_hists = [da.reindex_like(model_hist_maps[0], method='nearest') for da in model_hist_maps]
                            mmm_hist_map = xr.concat(aligned_hists, dim='model').mean(dim='model', skipna=True)
                        # --- END MODIFIED LOGIC ---
                        
                        results[gwl][season][storyline_name] = {
                            'mean_change_map': mmm_change_map.load(),
                            'historical_mean_map': mmm_hist_map.load()
                        }
                        logging.info(f"  -> Calculated change map for GWL {gwl}°C, {season}, {storyline_name} ({len(model_change_maps)} models)")

        return results
    
    @staticmethod
    def calculate_storyline_timeseries_correlation(cmip6_results, storyline_impacts, historical_monthly_data, config):
        """
        Calculates the correlation between the ensemble-mean future timeseries of
        SPEI change and discharge change for each storyline.
        
        CORRECTED VERSION: This function now calculates future SPEI and discharge on a per-model
        basis before creating the ensemble means to be correlated.
        """
        # --- DEBUG-BLOCK 1: START ---
        logging.info("\n--- DEBUG: Betrete calculate_storyline_timeseries_correlation ---")
        
        logging.info("Calculating storyline-based SPEI vs. Discharge timeseries correlations (Corrected Logic)...")
        correlations = {gwl: {'DJF': {}, 'JJA': {}} for gwl in config.GLOBAL_WARMING_LEVELS}

        # Dependencies
        classification = cmip6_results.get('storyline_classification_2d')
        all_deltas = cmip6_results.get('all_individual_model_deltas_for_plot')
        metric_timeseries = cmip6_results.get('model_metric_timeseries', {})
        gwl_years = cmip6_results.get('gwl_threshold_years', {})
        hist_pr_monthly = historical_monthly_data.get('pr_box_monthly')
        hist_tas_monthly = historical_monthly_data.get('tas_box_monthly')
        window = config.GWL_YEARS_WINDOW

        if not all(v is not None for v in [classification, all_deltas, metric_timeseries, gwl_years, hist_pr_monthly, hist_tas_monthly]):
            logging.error("Missing data for timeseries correlation calculation.")
            logging.info("--- DEBUG: Funktion wird wegen fehlender Eingangsdaten vorzeitig beendet. ---") # DEBUG
            return {}

        hist_pr_monthly_seas = DataProcessor.assign_season_to_dataarray(hist_pr_monthly)
        hist_tas_monthly_seas = DataProcessor.assign_season_to_dataarray(hist_tas_monthly)
        lat_center_of_box = (config.BOX_LAT_MIN + config.BOX_LAT_MAX) / 2

        for gwl in config.GLOBAL_WARMING_LEVELS:
            logging.info(f"\n--- DEBUG: Verarbeite GWL +{gwl}°C ---") # DEBUG
            storylines_in_gwl = classification.get(gwl, {})
            for storyline_key, model_list in storylines_in_gwl.items():
                if not model_list:
                    continue

                season_key = 'DJF' if 'DJF' in storyline_key else 'JJA'
                season_name = 'Winter' if season_key == 'DJF' else 'Summer'
                storyline_name = storyline_key.replace(f'{season_key}_', '')
                
                logging.info(f"  --- DEBUG: Analysiere Storyline '{storyline_name}' ({season_key}) mit {len(model_list)} Modellen ---") # DEBUG

                all_model_spei_ts = []
                all_model_discharge_ts = []

                for model_run_key in model_list:
                    # 1. Get INDIVIDUAL model delta_tas and delta_pr
                    delta_tas = all_deltas.get(f'{season_key}_tas', {}).get(gwl, {}).get(model_run_key)
                    delta_pr_percent = all_deltas.get(f'{season_key}_pr', {}).get(gwl, {}).get(model_run_key)

                    if delta_tas is None or delta_pr_percent is None:
                        logging.info(f"    - DEBUG: SKIP {model_run_key}, da delta_tas oder delta_pr fehlt.") # DEBUG
                        continue

                    # 2. Create future climate for THIS model
                    future_tas = xr.where(hist_tas_monthly_seas.season == season_name, hist_tas_monthly_seas + delta_tas, hist_tas_monthly_seas)
                    future_pr = xr.where(hist_tas_monthly_seas.season == season_name, hist_tas_monthly_seas * (1 + delta_pr_percent / 100.0), hist_tas_monthly_seas)
                    
                    # 3. Calculate future SPEI timeseries for THIS model
                    future_spei_monthly = DataProcessor.calculate_spei(future_pr, future_tas, lat=lat_center_of_box, scale=4)
                    if future_spei_monthly is None:
                        logging.info(f"    - DEBUG: SKIP {model_run_key}, da SPEI-Berechnung fehlschlug.") # DEBUG
                        continue
                    
                    future_spei_seasonal = DataProcessor.filter_by_season(DataProcessor.assign_season_to_dataarray(future_spei_monthly), season_name)
                    future_spei_ts = future_spei_seasonal.groupby('season_year').mean('time')

                    # 4. Get future Discharge timeseries for THIS model
                    discharge_ts_abs = metric_timeseries.get(model_run_key, {}).get(f'{season_key}_discharge')
                    if discharge_ts_abs is None:
                        logging.info(f"    - DEBUG: SKIP {model_run_key}, da Abfluss-Zeitreihe fehlt.") # DEBUG
                        continue

                    # 5. Slice both to the model's specific 30-year window
                    threshold_year = gwl_years.get(model_run_key, {}).get(gwl)
                    if threshold_year is None:
                        logging.info(f"    - DEBUG: SKIP {model_run_key}, da kein Schwellenjahr für GWL {gwl}°C gefunden wurde.") # DEBUG
                        continue
                    start_year, end_year = threshold_year - window // 2, threshold_year + (window - 1) // 2
                    
                    future_spei_window = future_spei_ts.sel(season_year=slice(start_year, end_year))
                    future_discharge_window = discharge_ts_abs.sel(season_year=slice(start_year, end_year))
                    
                    common_years = np.intersect1d(future_spei_window.season_year.values, future_discharge_window.season_year.values)
                    if len(common_years) < window * 0.8:
                        logging.info(f"    - DEBUG: SKIP {model_run_key}, zu wenig gemeinsame Jahre ({len(common_years)}) im Fenster.") # DEBUG
                        continue

                    # 6. Append INDIVIDUAL future timeseries to lists for ensembling
                    all_model_spei_ts.append(future_spei_window.sel(season_year=common_years).rename({'season_year': 'year'}))
                    all_model_discharge_ts.append(future_discharge_window.sel(season_year=common_years).rename({'season_year': 'year'}))

                # 7. Correlate the ENSEMBLE MEANS
                if len(all_model_spei_ts) > 1:
                    logging.info(f"    - DEBUG: Berechne Ensemble-Korrelation aus {len(all_model_spei_ts)} Modellen.") # DEBUG
                    ensemble_spei = xr.concat(all_model_spei_ts, dim='model').mean('model', skipna=True)
                    ensemble_discharge = xr.concat(all_model_discharge_ts, dim='model').mean('model', skipna=True)
                    
                    if ensemble_spei.size > 5 and ensemble_discharge.size > 5:
                        _, _, r_value, p_value, _ = StatsAnalyzer.calculate_regression(ensemble_spei.values, ensemble_discharge.values)
                        if not np.isnan(r_value):
                            correlations[gwl][season_key][storyline_name] = {'r': r_value, 'p': p_value}
                            logging.info(f"    --> DEBUG: ERGEBNIS für '{storyline_name}': r = {r_value:.2f}") # DEBUG
                else:
                    logging.info(f"    - DEBUG: SKIP Korrelation für '{storyline_name}', da nur {len(all_model_spei_ts)} Modelle erfolgreich verarbeitet wurden (benötigt > 1).") # DEBUG
        
        logging.info("Finished calculating storyline-based timeseries correlations.")
        
        # --- DEBUG-BLOCK 2: FINALE DATENSTRUKTUR LOGGEN ---
        logging.info("\n--- DEBUG: Finale 'correlations' Struktur vor dem Verlassen der Funktion: ---")
        try:
            # json.dumps ist eine gute Methode, um verschachtelte Dictionaries lesbar zu machen
            correlations_str = json.dumps(correlations, indent=2, default=str)
            for line in correlations_str.split('\n'):
                logging.info(line)
        except Exception as e:
            logging.error(f"Konnte 'correlations' nicht als JSON loggen: {e}")
        logging.info("-----------------------------------------------------------------------\n")
        # --- ENDE DEBUG-BLOCK 2 ---
        
        return correlations
    
    @staticmethod
    def calculate_storyline_impact_correlation(storyline_spei_impacts, direct_impacts_discharge, storyline_pr_tas_impacts, config):
        """
        Calculates the correlation between final impact values (e.g., SPEI vs Discharge, Precip vs Discharge) across all storylines.
        """
        logging.info("Calculating correlation between storyline impacts (SPEI vs Discharge, Precip vs Discharge)...")
        correlations = {gwl: {'DJF': {}, 'JJA': {}} for gwl in config.GLOBAL_WARMING_LEVELS}

        for gwl in config.GLOBAL_WARMING_LEVELS:
            for season_key in ['DJF', 'JJA']:
                
                # --- Correlation 1: SPEI vs. Discharge ---
                spei_key = f"{season_key}_spei"
                discharge_key = f"{season_key}_discharge"

                impacts_spei = storyline_spei_impacts.get(gwl, {}).get(spei_key, {})
                impacts_discharge = direct_impacts_discharge.get(gwl, {}).get(discharge_key, {})
                
                common_storylines_spei = sorted(list(set(impacts_spei.keys()) & set(impacts_discharge.keys())))
                
                if len(common_storylines_spei) >= 3:
                    spei_values = [impacts_spei[s]['total'] for s in common_storylines_spei]
                    discharge_values_spei = [impacts_discharge[s]['total'] for s in common_storylines_spei]
                    _, _, r_val, p_val, _ = StatsAnalyzer.calculate_regression(spei_values, discharge_values_spei)
                    if not np.isnan(r_val):
                        correlations[gwl][season_key]['spei_vs_discharge'] = {'r': r_val, 'p': p_val}
                        logging.info(f"  -> SPEI vs Discharge Corr for GWL +{gwl}°C {season_key}: r={r_val:.2f}")

                # --- NEU: Correlation 2: Precip vs. Discharge ---
                pr_key = f"{season_key}_pr"
                impacts_pr = storyline_pr_tas_impacts.get(gwl, {}).get(pr_key, {})
                
                common_storylines_pr = sorted(list(set(impacts_pr.keys()) & set(impacts_discharge.keys())))

                if len(common_storylines_pr) >= 3:
                    pr_values = [impacts_pr[s]['total'] for s in common_storylines_pr]
                    discharge_values_pr = [impacts_discharge[s]['total'] for s in common_storylines_pr]
                    _, _, r_val, p_val, _ = StatsAnalyzer.calculate_regression(pr_values, discharge_values_pr)
                    if not np.isnan(r_val):
                        correlations[gwl][season_key]['pr_vs_discharge'] = {'r': r_val, 'p': p_val}
                        logging.info(f"  -> Precip vs Discharge Corr for GWL +{gwl}°C {season_key}: r={r_val:.2f}")

        return correlations