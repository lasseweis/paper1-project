"""
Advanced analysis module for climate data.

This file contains the AdvancedAnalyzer class, which orchestrates more
complex, multi-step analyses. It combines functionalities from other modules
to produce high-level scientific results like regression maps,
correlation analyses, and comparisons between datasets.
"""
import numpy as np
import xarray as xr
import logging
import traceback
import os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

# Import local modules
from config import Config
from data_processing import DataProcessor
from stats_analyzer import StatsAnalyzer
from jet_analyzer import JetStreamAnalyzer

class AdvancedAnalyzer:
    """Advanced analysis methods for climate data."""

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

            slopes_pr, p_values_pr = AdvancedAnalyzer._calculate_regression_for_variable(pr_idx_norm, ua_season_detrended)
            slopes_tas, p_values_tas = AdvancedAnalyzer._calculate_regression_for_variable(tas_idx_norm, ua_season_detrended)
            
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
            futures = [executor.submit(AdvancedAnalyzer.calculate_regression_grid, index_common.values, field_common, [i], lon_indices, lat_dim, lon_dim) for i in lat_indices]
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

            slopes_pr, p_values_pr = AdvancedAnalyzer._calculate_regression_for_variable(pr_idx_norm, ua_season_detrended)
            slopes_tas, p_values_tas = AdvancedAnalyzer._calculate_regression_for_variable(tas_idx_norm, ua_season_detrended)
            
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
                slopes, p_values = AdvancedAnalyzer._calculate_regression_for_variable(jet_index_normalized, var_data_detrended)
                
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
            slopes, p_values = AdvancedAnalyzer._calculate_regression_for_variable(jet_index_common, var_field_common)

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
    def analyze_all_correlations_for_bar_chart(datasets_reanalysis, jet_data_reanalysis, discharge_data, season):
        """
        Analyzes all required correlations for the summary bar chart plot for a specific season.
        This method gathers data, calculates correlations, and returns a structured DataFrame.
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
        ]

        for dataset_key in [Config.DATASET_20CRV3, Config.DATASET_ERA5]:
            for config in plot_configs:
                # 1. Get the first timeseries (impact variable)
                var1_ts = None
                if config['var1_key'] in ['discharge', 'extreme_flow']:
                    if discharge_data:
                        var1_ts = discharge_data.get(f"{season_lower}_{config['var1_key']}")
                else:  # 'pr' or 'tas'
                    pr_tas_seasonal = datasets_reanalysis.get(f"{dataset_key}_{config['var1_key']}_box_mean")
                    if pr_tas_seasonal is not None:
                        var1_ts = DataProcessor.detrend_data(DataProcessor.filter_by_season(pr_tas_seasonal, season))

                # 2. Get the second timeseries (jet index)
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

                _, _, r_val, p_val, _ = StatsAnalyzer.calculate_regression(vals2, vals1) # Predictor (jet) is X, Impact is Y

                if not np.isnan(r_val):
                    # Create a descriptive label for the plot
                    base_label = config['title'].replace(f"{season} ", "").replace("vs", "vs.").replace("Jet", "Jet Index")
                    
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
        
        winter_correlations = AdvancedAnalyzer.analyze_amo_jet_correlations(
            jet_data=jet_data, amo_data=amo_data, season='Winter', window_size=window_size
        )
        
        summer_correlations = AdvancedAnalyzer.analyze_amo_jet_correlations(
            jet_data=jet_data, amo_data=amo_data, season='Summer', window_size=window_size
        )
        
        # This structure matches what the new plotting function expects
        return {
            'Winter': winter_correlations,
            'Summer': summer_correlations
        }