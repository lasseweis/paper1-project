"""
Main driver script for the climate analysis project.

This script initializes the environment, orchestrates the analysis workflow
by calling methods from the various specialized modules, and saves the
resulting plots and data.

To run the full analysis, execute this script from the command line:
$ python main.py
"""
import logging
import sys
import os 
import multiprocessing
import traceback
import matplotlib
import pandas as pd
import numpy as np
import xarray as xr 
from functools import lru_cache

# Set the backend for matplotlib to 'Agg' to prevent it from trying to open a GUI.
# This must be done before importing pyplot.
matplotlib.use('Agg')

# Import local modules
from config import Config
from data_processing import DataProcessor
from stats_analyzer import StatsAnalyzer
from jet_analyzer import JetStreamAnalyzer
from visualization import Visualizer
from storyline import StorylineAnalyzer


# --- Logging Configuration ---
class NoFindFontFilter(logging.Filter):
    """A custom filter to suppress 'findfont' messages from matplotlib."""
    def filter(self, record):
        return 'findfont: ' not in record.getMessage().lower()

log_filename = 'paper1_log.log'
logging.basicConfig(
    level=logging.INFO, # Changed to INFO for less verbose output, DEBUG is also fine
    format='%(asctime)s - %(levelname)s - [%(module)s.%(funcName)s] - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
matplotlib_logger = logging.getLogger('matplotlib')
matplotlib_logger.setLevel(logging.WARNING)
matplotlib_logger.addFilter(NoFindFontFilter())
logging.info("Logging initialized.")


class ClimateAnalysis:
    """Main class to orchestrate the climate data analysis workflow."""

    @staticmethod
    @lru_cache(maxsize=1)
    def process_20crv3_data():
        """Load and process all 20CRv3 climate data."""
        logging.info("Loading and processing 20CRv3 climate data...")
        try:
            pr_monthly = DataProcessor.process_ncfile(Config.PR_FILE_20CRV3, 'pr')
            tas_monthly = DataProcessor.process_ncfile(Config.TAS_FILE_20CRV3, 'tas')
            ua850_monthly = DataProcessor.process_ncfile(Config.UA_FILE_20CRV3, 'ua', 'ua', level_val=Config.WIND_LEVEL)
            
            if pr_monthly is None or tas_monthly is None or ua850_monthly is None:
                raise IOError("One or more 20CRv3 data files could not be processed.")

            pr_seasonal = DataProcessor.calculate_seasonal_means(DataProcessor.assign_season_to_dataarray(pr_monthly))
            tas_seasonal = DataProcessor.calculate_seasonal_means(DataProcessor.assign_season_to_dataarray(tas_monthly))
            ua850_seasonal = DataProcessor.calculate_seasonal_means(DataProcessor.assign_season_to_dataarray(ua850_monthly))

            pr_box_mean = DataProcessor.calculate_spatial_mean(pr_seasonal, Config.BOX_LAT_MIN, Config.BOX_LAT_MAX, Config.BOX_LON_MIN, Config.BOX_LON_MAX)
            tas_box_mean = DataProcessor.calculate_spatial_mean(tas_seasonal, Config.BOX_LAT_MIN, Config.BOX_LAT_MAX, Config.BOX_LON_MIN, Config.BOX_LON_MAX)

            return {
                '20CRv3_pr_monthly': pr_monthly,
                '20CRv3_tas_monthly': tas_monthly,
                '20CRv3_ua850_monthly': ua850_monthly,
                '20CRv3_pr_seasonal': pr_seasonal, 
                '20CRv3_tas_seasonal': tas_seasonal,
                '20CRv3_ua850_seasonal': ua850_seasonal, 
                '20CRv3_pr_box_mean': pr_box_mean,
                '20CRv3_tas_box_mean': tas_box_mean
            }
        except Exception as e:
            logging.error(f"Error in process_20crv3_data: {e}")
            return {}

    @staticmethod
    @lru_cache(maxsize=1)
    def process_era5_data():
        """Load and process all ERA5 climate data."""
        logging.info("Loading and processing ERA5 climate data...")
        try:
            pr_monthly = DataProcessor.process_era5_file(Config.ERA5_PR_FILE, 'pr')
            tas_monthly = DataProcessor.process_era5_file(Config.ERA5_TAS_FILE, 'tas')
            ua850_monthly = DataProcessor.process_era5_file(Config.ERA5_UA_FILE, 'u', 'ua', level_val=Config.WIND_LEVEL)

            if pr_monthly is None or tas_monthly is None or ua850_monthly is None:
                raise IOError("One or more ERA5 data files could not be processed.")

            pr_seasonal = DataProcessor.calculate_seasonal_means(DataProcessor.assign_season_to_dataarray(pr_monthly))
            tas_seasonal = DataProcessor.calculate_seasonal_means(DataProcessor.assign_season_to_dataarray(tas_monthly))
            ua850_seasonal = DataProcessor.calculate_seasonal_means(DataProcessor.assign_season_to_dataarray(ua850_monthly))
            
            pr_box_mean = DataProcessor.calculate_spatial_mean(pr_seasonal, Config.BOX_LAT_MIN, Config.BOX_LAT_MAX, Config.BOX_LON_MIN, Config.BOX_LON_MAX)
            tas_box_mean = DataProcessor.calculate_spatial_mean(tas_seasonal, Config.BOX_LAT_MIN, Config.BOX_LAT_MAX, Config.BOX_LON_MIN, Config.BOX_LON_MAX)

            return {
                'ERA5_pr_monthly': pr_monthly,
                'ERA5_tas_monthly': tas_monthly,
                'ERA5_ua850_monthly': ua850_monthly,
                'ERA5_pr_seasonal': pr_seasonal, 
                'ERA5_tas_seasonal': tas_seasonal,
                'ERA5_ua850_seasonal': ua850_seasonal, 
                'ERA5_pr_box_mean': pr_box_mean,
                'ERA5_tas_box_mean': tas_box_mean
            }
        except Exception as e:
            logging.error(f"Error in process_era5_data: {e}")

    @staticmethod
    def process_historical_discharge_data(qobs_daily_da):
        """
        Processes the QOBS DAILY data to create the required historical timeseries
        (detrended seasonal discharge, extreme flow) and adds hard-coded thresholds.
        REPLACES the old 'process_discharge_data' function.
        
        MODIFIED: Accepts daily data, resamples to monthly internally for monthly/seasonal
        calculations, and includes the original daily data in the result dict.
        """
        logging.info("Processing QOBS historical data (daily and monthly) for analysis...")
        if qobs_daily_da is None:
            logging.error("No QOBS data provided to process_historical_discharge_data.")
            return {}

        result = {}
        # --- NEW: Store the original daily data ---
        result['daily_historical_da'] = qobs_daily_da
        
        # --- NEW: Resample to monthly for all subsequent monthly/seasonal analysis ---
        qobs_monthly_da = qobs_daily_da.resample(time='MS').mean()
        result['monthly_historical_da'] = qobs_monthly_da # Store this as well

        # --- Berechne saisonale Mittelwerte (aus monatlichen Daten) ---
        da_with_seasons = DataProcessor.assign_season_to_dataarray(qobs_monthly_da)
        seasonal_means_ts = DataProcessor.calculate_seasonal_means(da_with_seasons)

        if seasonal_means_ts is not None:
            for season in ['Winter', 'Summer']:
                season_lower = season.lower()
                season_ts = DataProcessor.filter_by_season(seasonal_means_ts, season)
                if season_ts is not None and season_ts.size > 0:
                    # Speichere die detrendeten saisonalen Zeitreihen (für Korrelationen)
                    result[f'{season_lower}_discharge'] = DataProcessor.detrend_data(season_ts)
                    result[f'{season_lower}_mean'] = season_ts.mean().item()
        
        # --- Berechne "Extreme Flow" (basierend auf Perzentilen der monatlichen QOBS-Daten) ---
        high_flow_threshold = qobs_monthly_da.quantile(0.90).item()
        low_flow_threshold_overall = qobs_monthly_da.quantile(0.10).item()
        
        # Wandle in DataFrame um, um 'extreme_flow' zu berechnen (einfacher)
        df = qobs_monthly_da.to_dataframe(name='discharge')
        df['extreme_flow'] = np.select(
            [df['discharge'] > high_flow_threshold, df['discharge'] < low_flow_threshold_overall],
            [1, -1], default=0
        )
        
        da_extreme = df[['extreme_flow']].to_xarray()['extreme_flow']
        da_extreme_seasons = DataProcessor.assign_season_to_dataarray(da_extreme)
        seasonal_extreme_ts = DataProcessor.calculate_seasonal_means(da_extreme_seasons)
        
        if seasonal_extreme_ts is not None:
            for season in ['Winter', 'Summer']:
                season_data = DataProcessor.filter_by_season(seasonal_extreme_ts, season)
                if season_data is not None:
                    # Speichere die detrendeten extremen Zeitreihen (für Korrelationen)
                    result[f'{season.lower()}_extreme_flow'] = DataProcessor.detrend_data(season_data)

        # --- Füge die hard-gecodeten Schwellenwerte hinzu (wie zuvor) ---
        logging.info(f"Adding fixed low-flow thresholds: 1417 (30th), 1064 (10th), 970 (LNWL).")
        for season in ['winter', 'summer']:
            result[f'{season}_lowflow_threshold'] = 1064
            result[f'{season}_lowflow_threshold_30'] = 1417
            result[f'{season}_lowflow_lnwl'] = 970
        
        return result

    @staticmethod
    def load_amo_index(file_path):
        """Loads and processes the AMO index from a CSV file for winter and summer."""
        logging.info(f"Loading and processing AMO index from {file_path}...")
        try:
            amo_df = pd.read_csv(file_path, sep=",", header=0)
            amo_df.replace(-999, np.nan, inplace=True)
            amo_long = amo_df.melt(id_vars="Year", var_name="Month", value_name="AMO")
            month_mapping = {"Jan":1, "Feb":2, "Mar":3, "Apr":4, "May":5, "Jun":6,
                             "Jul":7, "Aug":8, "Sep":9, "Oct":10, "Nov":11, "Dec":12}
            amo_long["Month"] = amo_long["Month"].map(month_mapping)
            
            # Use a datetime index to leverage the robust DataProcessor methods
            amo_long['time'] = pd.to_datetime(dict(year=amo_long['Year'], month=amo_long['Month'], day=15))
            da = amo_long.set_index('time')[['AMO']].to_xarray()['AMO'].dropna(dim='time')

            da_with_seasons = DataProcessor.assign_season_to_dataarray(da)
            seasonal_means = DataProcessor.calculate_seasonal_means(da_with_seasons)

            result = {}
            if seasonal_means is not None:
                for season in ['Winter', 'Summer']:
                    season_lower = season.lower()
                    season_data = DataProcessor.filter_by_season(seasonal_means, season)
                    if season_data is not None:
                        result[f'amo_{season_lower}'] = season_data
                        result[f'amo_{season_lower}_detrended'] = DataProcessor.detrend_data(season_data)
            
            logging.info("AMO index processing finished successfully.")
            return result

        except Exception as e:
            logging.error(f"Error processing AMO index file: {e}")
            logging.error(traceback.format_exc())
            return {}
        
    @staticmethod
    def process_cmip6_discharge_data(config):
        """Helper to load all available CMIP6 discharge data."""
        logging.info("Processing all CMIP6 discharge data...")
        # Get all models that we want to analyze from the config
        models_to_include = list(config.REQUIRED_MODEL_SCENARIOS.keys())

        cmip6_discharge = DataProcessor.process_discharge_data(
            config.DISCHARGE_SSP245_FILE,
            config.DISCHARGE_SSP585_FILE,
            models_to_include
        )
        return cmip6_discharge
    
    @staticmethod
    def load_historical_qobs_from_csv(config):
        """
        Loads the historical QOBS (observations) timeseries from the CMIP6 discharge CSV file.
        This data is assumed to be daily and is returned as a daily xarray DataArray.
        This is used as the basis for historical percentile and EVA calculations.
        
        MODIFIED: Returns daily data, not resampled to monthly.
        """
        logging.info("Loading DAILY historical QOBS discharge data from CSV...")
        filepath = config.DISCHARGE_SSP245_FILE # We can use either file, QOBS is the same

        if not os.path.exists(filepath):
            logging.error(f"Cannot load QOBS data: File not found at {filepath}")
            return None

        try:
            df = pd.read_csv(filepath, sep=';', decimal=',', na_values=['-0,01'])

            # First column is the date
            df = df.rename(columns={df.columns[0]: 'date'})
            # --- START: KORREKTUR (Explizites Datumsformat) ---
            # Parse dates, explicitly providing the format found in the CSV
            df['time'] = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
            # --- ENDE: KORREKTUR ---
            df = df.set_index('time')

            if 'QOBS' not in df.columns:
                logging.error(f"QOBS column not found in {filepath}")
                return None

            # Select QOBS and filter for the historical period (1960-2021)
            qobs_series_daily = df['QOBS'].loc['1960-01-01':'2021-12-31'].dropna()

            if qobs_series_daily.empty:
                logging.error(f"No QOBS data found in the 1960-2021 period in {filepath}")
                return None

            # --- MODIFICATION: DO NOT RESAMPLE TO MONTHLY ---
            # qobs_series_monthly = qobs_series_daily.resample('MS').mean()
            
            # Convert to xarray DataArray
            da = qobs_series_daily.to_xarray()
            da.name = 'discharge'
            da.attrs['units'] = 'm3/s'
            da.attrs['long_name'] = 'Observed Danube River Discharge (QOBS 1960-2021, daily)'

            logging.info(f"Successfully loaded DAILY QOBS historical data from 1960-2021.")
            return da

        except Exception as e:
            logging.error(f"Error processing QOBS from {filepath}: {e}")
            logging.error(traceback.format_exc())
            return None

    @staticmethod
    def run_full_analysis():
        """
        Main static method to execute the entire analysis workflow.
        MODIFIED to loop through scenarios for CMIP6/Storyline analysis.
        MODIFIED to calculate and pass threshold data to impact plot.
        """
        logging.info("=====================================================")
        logging.info("=== STARTING FULL CLIMATE ANALYSIS WORKFLOW ===")
        logging.info("=====================================================\n")

        Visualizer.ensure_plot_dir_exists()
        regression_results = {}

        # Create an instance of the central analysis class
        storyline_analyzer = StorylineAnalyzer(config=Config)

        # --- PART 1: BASIC REANALYSIS CALCULATIONS (Done once) ---
        logging.info("\n--- Processing Reanalysis Datasets ---")
        datasets_reanalysis = {
            **ClimateAnalysis.process_20crv3_data(),
            **ClimateAnalysis.process_era5_data()
        }
        if not datasets_reanalysis:
            logging.critical("Failed to process reanalysis datasets. Aborting.")
            return None

        logging.info("\n--- Calculating SPEI for Reanalysis Datasets ---")
        for dset_key in [Config.DATASET_20CRV3, Config.DATASET_ERA5]:
            logging.info(f"  Calculating SPEI for {dset_key}...")
            pr_monthly_full = datasets_reanalysis.get(f'{dset_key}_pr_monthly')
            tas_monthly_full = datasets_reanalysis.get(f'{dset_key}_tas_monthly')
            if pr_monthly_full is not None and tas_monthly_full is not None:
                pr_box_monthly = DataProcessor.calculate_spatial_mean(pr_monthly_full, Config.BOX_LAT_MIN, Config.BOX_LAT_MAX, Config.BOX_LON_MIN, Config.BOX_LON_MAX)
                tas_box_monthly = DataProcessor.calculate_spatial_mean(tas_monthly_full, Config.BOX_LAT_MIN, Config.BOX_LAT_MAX, Config.BOX_LON_MIN, Config.BOX_LON_MAX)
                if pr_box_monthly is not None and tas_box_monthly is not None:
                    lat_center_of_box = (Config.BOX_LAT_MIN + Config.BOX_LAT_MAX) / 2
                    spei4 = DataProcessor.calculate_spei(pr_box_monthly, tas_box_monthly, lat=lat_center_of_box, scale=4)
                    if spei4 is not None:
                        # --- MODIFIKATION: Store monthly SPEI for later use ---
                        datasets_reanalysis[f'{dset_key}_spei4_monthly_box'] = spei4
                        # --- ENDE MODIFIKATION ---
                        spei4_seasonal = DataProcessor.assign_season_to_dataarray(spei4)
                        datasets_reanalysis[f'{dset_key}_spei4'] = spei4_seasonal # Seasonal means are still stored
                        logging.info(f"    Successfully calculated SPEI-4 for {dset_key}.")

        # --- START: ÄNDERUNG (Laden der Abflussdaten) ---
        # 1. Lade QOBS-Daten (1960-2021, TÄGLICH) aus der .csv-Datei
        qobs_historical_da = ClimateAnalysis.load_historical_qobs_from_csv(Config())
        
        # 2. Verarbeite QOBS-Daten, um historische Zeitreihen zu erstellen UND fixe Schwellenwerte hinzuzufügen
        # Diese 'discharge_data_loaded' wird für alle historischen Plots (Korrelationen) 
        # UND für die fixen Schwellenwerte (LNWL) in den Zukunfts-Plots verwendet.
        # WICHTIG: Dieses dict enthält jetzt 'daily_historical_da' UND 'monthly_historical_da'
        discharge_data_loaded = ClimateAnalysis.process_historical_discharge_data(qobs_historical_da)
        # --- ENDE: ÄNDERUNG ---
        
        amo_data_loaded = ClimateAnalysis.load_amo_index(Config.AMO_INDEX_FILE)

        logging.info("\n--- Calculating Base Reanalysis Jet Indices ---")
        jet_data_reanalysis = {}
        for dset_key in [Config.DATASET_20CRV3, Config.DATASET_ERA5]:
            ua850_seasonal = datasets_reanalysis[f'{dset_key}_ua850_seasonal']
            for season in ['Winter', 'Summer']:
                ua_season = DataProcessor.filter_by_season(ua850_seasonal, season)
                season_lower = season.lower()
                jet_speed = JetStreamAnalyzer.calculate_jet_speed_index(ua_season)
                if jet_speed is not None:
                    jet_data_reanalysis[f'{dset_key}_{season_lower}_speed_data'] = {'jet': DataProcessor.detrend_data(jet_speed)}
                jet_lat = JetStreamAnalyzer.calculate_jet_lat_index(ua_season)
                if jet_lat is not None:
                    jet_data_reanalysis[f'{dset_key}_{season_lower}_lat_data'] = {'jet': DataProcessor.detrend_data(jet_lat)}

        # Calculate multivariate betas once from reanalysis, as they are constant for all scenarios
        beta_obs_slopes = storyline_analyzer.calculate_reanalysis_betas(
            datasets_reanalysis,
            jet_data_reanalysis,
            dataset_key=Config.DATASET_ERA5
        )

        # --- REANALYSIS PLOTTING (Done once, independent of CMIP6 scenarios) ---
        logging.info("\n--- Checking for Reanalysis Regression Maps ---")
        regression_period = (1981, 2010)
        for dset_key in [Config.DATASET_20CRV3, Config.DATASET_ERA5]:
            regression_plot_filename = os.path.join(Config.PLOT_DIR, f'regression_maps_norm_{dset_key}.png')
            if not os.path.exists(regression_plot_filename):
                logging.info(f"Plot '{regression_plot_filename}' not found. Calculating data and creating plot...")
                results = StorylineAnalyzer.calculate_regression_maps(
                    datasets=datasets_reanalysis,
                    dataset_key=dset_key,
                    regression_period=regression_period
                )
                if results:
                    regression_results[dset_key] = results
                    Visualizer.plot_regression_analysis(results, dset_key)
                else:
                    logging.warning(f"Could not calculate regression for {dset_key}.")
            else:
                logging.info(f"Plot '{regression_plot_filename}' already exists. Skipping.")

        logging.info("\n\n--- Checking for Reanalysis Jet Index Comparison Timeseries ---")
        jet_indices_plot_filename = os.path.join(Config.PLOT_DIR, "jet_indices_comparison_seasonal_detrended.png")
        if not os.path.exists(jet_indices_plot_filename):
            if jet_data_reanalysis:
                logging.info(f"Plot '{jet_indices_plot_filename}' not found, creating...")
                Visualizer.plot_jet_indices_comparison(jet_data_reanalysis)
            else:
                logging.warning("Skipping jet index comparison plot, no data was generated.")
        else:
            logging.info(f"Plot '{jet_indices_plot_filename}' already exists. Skipping.")

        logging.info("\n\n--- Checking for Reanalysis Jet Impact Comparison Maps ---")
        for season in ['Winter', 'Summer']:
            jet_impact_plot_filename = os.path.join(Config.PLOT_DIR, f'jet_impact_regression_maps_{season.lower()}.png')
            if not os.path.exists(jet_impact_plot_filename):
                logging.info(f"Plot '{jet_impact_plot_filename}' not found. Calculating data and creating plot...")
                impact_20crv3 = StorylineAnalyzer.calculate_jet_impact_maps(datasets_reanalysis, jet_data_reanalysis, Config.DATASET_20CRV3, season)
                impact_era5 = StorylineAnalyzer.calculate_jet_impact_maps(datasets_reanalysis, jet_data_reanalysis, Config.DATASET_ERA5, season)
                if impact_20crv3.get(season) and impact_era5.get(season):
                    Visualizer.plot_jet_impact_comparison_maps(impact_20crv3.get(season), impact_era5.get(season), season)
                else:
                    logging.warning(f"Skipping combined jet impact plot for {season}, data missing.")
            else:
                logging.info(f"Plot '{jet_impact_plot_filename}' already exists. Skipping.")

        logging.info("\n\n--- Checking for Reanalysis Jet Correlation Maps ---")
        for season in ['Winter', 'Summer']:
            jet_corr_plot_filename = os.path.join(Config.PLOT_DIR, f'jet_correlation_maps_{season.lower()}.png')
            if not os.path.exists(jet_corr_plot_filename):
                logging.info(f"Plot '{jet_corr_plot_filename}' not found. Calculating data and creating plot...")
                corr_20crv3 = StorylineAnalyzer.calculate_jet_correlation_maps(datasets_reanalysis, jet_data_reanalysis, Config.DATASET_20CRV3, season)
                corr_era5 = StorylineAnalyzer.calculate_jet_correlation_maps(datasets_reanalysis, jet_data_reanalysis, Config.DATASET_ERA5, season)
                if corr_20crv3 and corr_era5:
                    Visualizer.plot_jet_correlation_maps(corr_20crv3, corr_era5, season)
                else:
                    logging.warning(f"Skipping jet correlation plot for {season}, data missing.")
            else:
                logging.info(f"Plot '{jet_corr_plot_filename}' already exists. Skipping.")

        logging.info("\n\n--- Checking for Correlation Timeseries & Bar Charts ---")
        for season in ['Winter', 'Summer']:
            season_lower = season.lower()
            corr_timeseries_filename = os.path.join(Config.PLOT_DIR, f'{season_lower}_correlations_comparison_detrended.png')
            if not os.path.exists(corr_timeseries_filename):
                logging.info(f"Plot '{corr_timeseries_filename}' not found, creating...")
                Visualizer.plot_correlation_timeseries_comparison(datasets_reanalysis, jet_data_reanalysis, discharge_data_loaded, season)
            else:
                logging.info(f"Plot '{corr_timeseries_filename}' already exists.")
            corr_barchart_filename = os.path.join(Config.PLOT_DIR, f'correlation_matrix_comparison_{season_lower}_detrended_grouped.png')
            if not os.path.exists(corr_barchart_filename):
                logging.info(f"Plot '{corr_barchart_filename}' not found. Calculating and creating plot...")
                correlation_data_for_bar_chart = StorylineAnalyzer.analyze_all_correlations_for_bar_chart(
                    datasets_reanalysis, jet_data_reanalysis, discharge_data_loaded, amo_data_loaded, season
                )
                if not correlation_data_for_bar_chart.empty:
                    Visualizer.plot_correlation_bar_chart(correlation_data_for_bar_chart, season)
            else:
                logging.info(f"Plot '{corr_barchart_filename}' already exists.")

        logging.info("\n\n--- Checking for AMO-Jet Correlation Plot ---")
        window_size_amo = 15
        amo_plot_filename = os.path.join(Config.PLOT_DIR, f'amo_jet_correlations_comparison_rolling_{window_size_amo}yr.png')
        if not os.path.exists(amo_plot_filename):
            logging.info(f"Plot '{amo_plot_filename}' not found. Calculating data and creating plot...")
            if amo_data_loaded:
                amo_correlation_data = StorylineAnalyzer.analyze_amo_jet_correlations_for_plot(jet_data_reanalysis, amo_data_loaded, window_size_amo)
                if amo_correlation_data and any(amo_correlation_data.values()):
                    Visualizer.plot_amo_jet_correlation_comparison(amo_correlation_data, window_size_amo)
            else:
                logging.warning("AMO data could not be loaded, skipping AMO-Jet correlation analysis.")
        else:
            logging.info(f"Plot '{amo_plot_filename}' already exists.")

        logging.info("\n\n--- Checking for Seasonal Drought Analysis Plot ---")
        drought_plot_filename = os.path.join(Config.PLOT_DIR, 'spei_drought_analysis_seasonal_comparison.png')
        if not os.path.exists(drought_plot_filename):
            logging.info(f"Plot '{drought_plot_filename}' not found, creating...")
            if any(f'{dset}_spei4' in datasets_reanalysis for dset in [Config.DATASET_20CRV3, Config.DATASET_ERA5]):
                Visualizer.plot_seasonal_drought_analysis(datasets_reanalysis, scale=4)
        else:
            logging.info(f"Plot '{drought_plot_filename}' already exists.")

        logging.info("\n\n--- Checking for Combined Spatial SPEI & Discharge Analysis Map ---")
        combined_spei_plot_filename = os.path.join(Config.PLOT_DIR, 'spatial_spei_discharge_analysis_era5_summer.png')
        if not os.path.exists(combined_spei_plot_filename):
            logging.info(f"Plot '{combined_spei_plot_filename}' not found. Calculating and creating plot...")
            spatial_spei_era5 = StorylineAnalyzer.calculate_spatial_spei(datasets_reanalysis, Config.DATASET_ERA5, scale=4)
            summer_discharge_ts = discharge_data_loaded.get('summer_discharge') # Use detrended later
            if spatial_spei_era5 is not None and summer_discharge_ts is not None:
                # Detrend here for the map calculation
                summer_discharge_detrended = DataProcessor.detrend_data(summer_discharge_ts)
                if summer_discharge_detrended is not None:
                    corr_map, p_vals_corr = StorylineAnalyzer.calculate_spei_on_discharge_map(spatial_spei_data=spatial_spei_era5, discharge_timeseries=summer_discharge_detrended, season='Summer', analysis_type='correlation')
                    regr_slopes, p_vals_regr = StorylineAnalyzer.calculate_spei_on_discharge_map(spatial_spei_data=spatial_spei_era5, discharge_timeseries=summer_discharge_detrended, season='Summer', analysis_type='regression')
                    Visualizer.plot_spatial_spei_analysis_maps(
                        spatial_spei_data=spatial_spei_era5, discharge_corr_map=corr_map, p_values_corr=p_vals_corr,
                        discharge_regr_slopes=regr_slopes, p_values_regr=p_vals_regr,
                        time_slice='2003-08-15', season='Summer', title_prefix='ERA5',
                        filename=os.path.basename(combined_spei_plot_filename)
                    )
        else:
            logging.info(f"Plot '{combined_spei_plot_filename}' already exists.")

        # --- SPECIAL CASE: Single CMIP6 Model Regression Plot (Done once) ---
        logging.info("\n\n--- Checking for Single CMIP6 Model Regression Maps ---")
        single_model_plot_filename = os.path.join(Config.PLOT_DIR, "regression_maps_norm_CMIP6_single_models.png")
        if not os.path.exists(single_model_plot_filename):
            logging.info(f"Plot '{single_model_plot_filename}' not found. Calculating data and creating plot...")
            try:
                # Run a preliminary analysis with the first scenario to get the loaded model data
                preliminary_cmip6_results = storyline_analyzer.analyze_cmip6_changes_at_gwl(scenario_to_process=Config.CMIP6_SCENARIOS[0])
                if preliminary_cmip6_results and 'cmip6_model_data_loaded' in preliminary_cmip6_results:
                    models_to_plot = ["ACCESS-CM2_ssp585", "MPI-ESM1-2-HR_ssp585", "IPSL-CM6A-LR_ssp585", "MIROC6_ssp585"]
                    single_model_regression_data = {}
                    for model_key in models_to_plot:
                        if model_key in preliminary_cmip6_results['cmip6_model_data_loaded']:
                            model_data = preliminary_cmip6_results['cmip6_model_data_loaded'][model_key]
                            results = StorylineAnalyzer.calculate_single_model_regression_maps(model_data, model_key, historical_period=(1995, 2014))
                            if results:
                                single_model_regression_data[model_key] = results
                        else:
                            logging.warning(f"Data for selected model {model_key} not found in preloaded data.")
                    if single_model_regression_data:
                        Visualizer.plot_cmip6_model_regression_analysis(single_model_regression_data, models_to_plot)
                else:
                    logging.warning("Skipping single CMIP6 model regression plot: Preliminary CMIP6 analysis failed.")
            except Exception as e:
                logging.error(f"Failed to run preliminary CMIP6 analysis for single model plots: {e}")
                logging.error(traceback.format_exc()) # Added traceback
        else:
            logging.info(f"Plot '{single_model_plot_filename}' already exists.")

        # =================================================================================
        # === START OF SCENARIO-SPECIFIC CMIP6 ANALYSIS LOOP ===
        # =================================================================================

        # Load all CMIP6 discharge data once before the loop
        cmip6_discharge_loaded = ClimateAnalysis.process_cmip6_discharge_data(Config()) # This uses the static method

        for scenario in Config.CMIP6_SCENARIOS:
            logging.info(f"\n\n{'='*25} STARTING CMIP6 ANALYSIS FOR SCENARIO: {scenario.upper()} {'='*25}\n")

            try:
                # --- PART 2: CMIP6 AND STORYLINE ANALYSIS (per scenario) ---
                cmip6_results = storyline_analyzer.analyze_cmip6_changes_at_gwl(scenario_to_process=scenario)

                if not cmip6_results:
                    logging.warning(f"CMIP6 analysis did not produce results for scenario {scenario}. Skipping.")
                    continue

                # --- PLOT: Jet Changes vs GWL (per scenario) ---
                gwl_plot_filename = os.path.join(Config.PLOT_DIR, f"cmip6_jet_changes_vs_gwl_{scenario}.png")
                if not os.path.exists(gwl_plot_filename):
                    logging.info(f"Plot '{gwl_plot_filename}' not found, creating...")
                    Visualizer.plot_jet_changes_vs_gwl(cmip6_results, scenario=scenario)
                else:
                    logging.info(f"Plot '{gwl_plot_filename}' already exists.")

                # --- PLOT: Jet Inter-relationship Scatter (per scenario) ---
                inter_rel_plot_filename = os.path.join(Config.PLOT_DIR, f"cmip6_jet_inter_relationship_scatter_quadrants_{scenario}.png") # Updated filename
                if not os.path.exists(inter_rel_plot_filename):
                     logging.info(f"Plot '{inter_rel_plot_filename}' not found, creating...") # Added logging
                     if cmip6_results.get('mmm_changes'): # Check added
                         Visualizer.plot_jet_inter_relationship_scatter_combined_gwl(cmip6_results=cmip6_results, scenario=scenario)
                     else:
                         logging.warning(f"Skipping inter-relationship scatter for {scenario}, missing MMM changes.") # Added warning
                else:
                    logging.info(f"Plot '{inter_rel_plot_filename}' already exists.")

                # --- START: NEUER PLOT ---
                # --- PLOT: Cross-Season Jet Relationship (nur für ssp585) ---
                if scenario == 'ssp585':
                    cross_season_plot_filename = os.path.join(Config.PLOT_DIR, f"cmip6_jet_cross_season_relationship_{scenario}.png")
                    if not os.path.exists(cross_season_plot_filename):
                        logging.info(f"Plot '{cross_season_plot_filename}' not found, creating...")
                        Visualizer.plot_jet_cross_season_relationship(cmip6_results, scenario)
                    else:
                        logging.info(f"Plot '{cross_season_plot_filename}' already exists.")
                # --- ENDE: NEUER PLOT ---

                # --- PLOT: Climate Evolution Timeseries (per scenario) ---
                evolution_plot_filename = os.path.join(Config.PLOT_DIR, f"climate_indices_evolution_{scenario}.png")
                if not os.path.exists(evolution_plot_filename):
                    logging.info(f"Plot '{evolution_plot_filename}' not found. Calculating and creating plot...")
                    cmip6_plot_data, reanalysis_plot_data = StorylineAnalyzer.analyze_timeseries_for_projection_plot(cmip6_results, datasets_reanalysis, Config())
                    if cmip6_plot_data and reanalysis_plot_data:
                        Visualizer.plot_climate_projection_timeseries(cmip6_plot_data, reanalysis_plot_data, Config(), filename=os.path.basename(evolution_plot_filename))
                    else:
                         logging.warning(f"Skipping climate evolution plot for {scenario}, data preparation failed.") # Added warning
                else:
                    logging.info(f"Plot '{evolution_plot_filename}' already exists.")

                # --- PLOT: Storyline U850 Wind Change Maps (per scenario) ---
                storyline_map_plot_filename = os.path.join(Config.PLOT_DIR, f"storyline_u850_change_maps_{scenario}.png")
                if not os.path.exists(storyline_map_plot_filename):
                    logging.info(f"Plot '{storyline_map_plot_filename}' not found. Calculating data and creating plot...") # Added logging
                    future_period = (2070, 2099)
                    hist_period = (Config.CMIP6_ANOMALY_REF_START, Config.CMIP6_ANOMALY_REF_END)
                    preloaded_data = cmip6_results.get('cmip6_model_data_loaded', {})
                    # Ensure models are extracted correctly for the current scenario
                    models_for_calc = sorted(list(set(key.split('_')[0] for key in preloaded_data.keys() if key.endswith(scenario))))

                    if models_for_calc:
                        u850_change_data = storyline_analyzer.calculate_cmip6_u850_change_fields(
                            models_to_run=models_for_calc, future_scenario=scenario,
                            future_period=future_period, historical_period=hist_period,
                            preloaded_cmip6_data=preloaded_data
                        )
                        if u850_change_data:
                            historical_mmm_u850_for_storylines = {}
                            if u850_change_data.get('DJF'):
                                historical_mmm_u850_for_storylines['DJF'] = u850_change_data['DJF'].get('u850_historical_mean_mmm')
                            if u850_change_data.get('JJA'):
                                historical_mmm_u850_for_storylines['JJA'] = u850_change_data['JJA'].get('u850_historical_mean_mmm')

                            # Check if both seasons have data before proceeding
                            if 'DJF' in historical_mmm_u850_for_storylines and 'JJA' in historical_mmm_u850_for_storylines:
                                storyline_change_maps = storyline_analyzer.calculate_storyline_wind_change_maps(
                                    cmip6_results=cmip6_results, config=Config(),
                                    historical_mmm_u850_by_season=historical_mmm_u850_for_storylines
                                )
                                if storyline_change_maps:
                                    Visualizer.plot_storyline_wind_change_maps(map_data=storyline_change_maps, config=Config(), scenario=scenario)
                                else:
                                     logging.warning(f"Skipping wind change map plot for {scenario}: Storyline map calculation failed.") # Added warning
                            else:
                                 logging.warning(f"Skipping wind change map plot for {scenario}: Missing historical MMM U850 data for DJF or JJA.") # Added warning
                        else:
                             logging.warning(f"Skipping wind change map plot for {scenario}: U850 change field calculation failed.") # Added warning
                    else:
                         logging.warning(f"Skipping wind change map plot for {scenario}: No models found for calculation.") # Added warning
                else:
                    logging.info(f"Plot '{storyline_map_plot_filename}' already exists.")


                # --- START: BLOCK MIT ANPASSUNGEN (NEUE PLOT-AUFRUFE) ---
                # --- PLOT: Storyline Discharge Return Periods (per scenario) ---
                return_period_plot_filename_low = os.path.join(Config.PLOT_DIR, f"storyline_discharge_return_period_LOW_FLOW_{scenario}.png")
                return_period_plot_filename_high = os.path.join(Config.PLOT_DIR, f"storyline_discharge_return_period_HIGH_FLOW_{scenario}.png")
                
                return_period_results_for_plot = None # Initialisieren

                # Check if EITHER plot is missing
                if not os.path.exists(return_period_plot_filename_low) or not os.path.exists(return_period_plot_filename_high):
                    logging.info(f"One or more EVA return period plots for {scenario} not found. Calculating data...")
                    # --- START: ÄNDERUNG ---
                    # historical_da = discharge_data_loaded.get('monthly_historical_da') # ALTE ZEILE
                    historical_da = discharge_data_loaded.get('daily_historical_da') # NEU: Verwende die TÄGLICHEN QOBS-Daten
                    # --- ENDE: ÄNDERUNG ---
                    
                    if historical_da is not None:
                        # Ergebnisse speichern
                        return_period_results_for_plot = storyline_analyzer.analyze_storyline_discharge_extremes(
                            cmip6_results=cmip6_results,
                            historical_discharge_da=historical_da, # <-- HIER TÄGLICHE QOBS übergeben
                            config=Config(),
                            discharge_thresholds=discharge_data_loaded # Die fixen werden hier noch übergeben
                        )
                        if return_period_results_for_plot:
                            # Hier wird die VISUALISIERUNG für die NEUEN Plots aufgerufen
                            if not os.path.exists(return_period_plot_filename_low):
                                Visualizer.plot_storyline_return_period_LOW_FLOW(return_period_results_for_plot, Config(), scenario=scenario)
                            else:
                                logging.info(f"Plot '{return_period_plot_filename_low}' already exists.")
                                
                            if not os.path.exists(return_period_plot_filename_high):
                                Visualizer.plot_storyline_return_period_HIGH_FLOW(return_period_results_for_plot, Config(), scenario=scenario)
                            else:
                                logging.info(f"Plot '{return_period_plot_filename_high}' already exists.")
                        else:
                            logging.warning(f"Could not calculate return period results for {scenario}.")
                    else:
                        logging.warning(f"DAILY QOBS historical discharge data not available for return period analysis in {scenario}.")
                else:
                    logging.info(f"Both EVA return period plots for {scenario} already exist.")
                    
                # Ergebnisse trotzdem berechnen, wenn Plot existiert, für den nächsten Plot
                if return_period_results_for_plot is None: # Nur berechnen, wenn nicht schon geschehen
                    # --- START: ÄNDERUNG ---
                    historical_da = discharge_data_loaded.get('daily_historical_da') # NEU: Verwende die TÄGLICHEN QOBS-Daten
                    # --- ENDE: ÄNDERUNG ---
                    if historical_da is not None:
                         logging.info(f"Calculating return period results for {scenario} (needed for impact plot)...")
                         return_period_results_for_plot = storyline_analyzer.analyze_storyline_discharge_extremes(
                             cmip6_results=cmip6_results,
                             historical_discharge_da=historical_da, # <-- HIER TÄGLICHE QOBS übergeben
                             config=Config(),
                             discharge_thresholds=discharge_data_loaded
                         )

                # --- PLOT: Storyline Impacts Bar Chart (per scenario) ---
                impacts_plot_filename = os.path.join(Config.PLOT_DIR, f"storyline_impacts_summary_4x2_boxplots_{scenario}.png")
                if not os.path.exists(impacts_plot_filename):
                    logging.info(f"Plot '{impacts_plot_filename}' not found, creating...")
                    if cmip6_results:
                        storyline_correlations = None # Optional

                        # threshold_data übergeben
                        # Wir extrahieren nur den 'thresholds'-Teil aus den Return-Period-Ergebnissen
                        threshold_data_for_plot = return_period_results_for_plot.get('thresholds') if return_period_results_for_plot else None

                        if threshold_data_for_plot: # Nur plotten, wenn Schwellenwerte vorhanden sind
                            Visualizer.plot_storyline_impact_barchart_with_discharge(
                                cmip6_results=cmip6_results,
                                threshold_data=threshold_data_for_plot, # <-- HIER WIRD ES ÜBERGEBEN
                                discharge_data_historical=discharge_data_loaded,
                                reanalysis_data=datasets_reanalysis,
                                config=Config(),
                                scenario=scenario,
                                storyline_correlations=storyline_correlations
                            )
                        else:
                            logging.warning(f"Skipping impact plot for {scenario} because threshold data could not be calculated.")
                    else:
                        logging.warning(f"Skipping impact plot for {scenario} because cmip6_results are missing.")
                else:
                    logging.info(f"Plot '{impacts_plot_filename}' already exists.")
                # --- ENDE: BLOCK MIT ANPASSUNGEN ---


                # --- PLOT: Model Fidelity and Scatter Plots (per scenario) ---
                if beta_obs_slopes:
                    fidelity_plot_filename = os.path.join(Config.PLOT_DIR, f"cmip6_fidelity_vs_future_temporal_slopes_{scenario}.png")
                    if not os.path.exists(fidelity_plot_filename):
                        logging.info(f"Plot '{fidelity_plot_filename}' not found, creating...")
                        historical_period_for_fidelity = (1981, 2010)
                        cmip6_historical_slopes = storyline_analyzer.calculate_historical_slopes_comparison(beta_obs_slopes=beta_obs_slopes, cmip6_data_loaded=cmip6_results.get('cmip6_model_data_loaded', {}), jet_data_reanalysis=jet_data_reanalysis, historical_period=historical_period_for_fidelity)
                        cmip6_future_temporal_slopes = storyline_analyzer.calculate_future_temporal_slopes(cmip6_results=cmip6_results, beta_keys=list(beta_obs_slopes.keys()), gwls_to_analyze=Config.GLOBAL_WARMING_LEVELS)
                        if cmip6_historical_slopes and cmip6_future_temporal_slopes:
                             Visualizer.plot_model_fidelity_comparison(cmip6_historical_slopes=cmip6_historical_slopes, cmip6_future_temporal_slopes=cmip6_future_temporal_slopes, beta_obs_slopes=beta_obs_slopes, historical_period=historical_period_for_fidelity, gwls_to_plot=Config.GLOBAL_WARMING_LEVELS, scenario=scenario)
                        else:
                             logging.warning(f"Skipping fidelity plot for {scenario} due to missing slope data.")
                    else:
                         logging.info(f"Plot '{fidelity_plot_filename}' already exists.")

                    for gwl in Config.GLOBAL_WARMING_LEVELS:
                        scatter_plot_filename = os.path.join(Config.PLOT_DIR, f"cmip6_scatter_comparison_gwl_{gwl:.1f}_{scenario}.png")
                        if not os.path.exists(scatter_plot_filename):
                            logging.info(f"Plot '{scatter_plot_filename}' not found, creating...")
                            Visualizer.plot_cmip6_scatter_comparison(cmip6_results=cmip6_results, beta_obs_slopes=beta_obs_slopes, gwl_to_plot=gwl, scenario=scenario)
                        else:
                             logging.info(f"Plot '{scatter_plot_filename}' already exists.")
                else:
                    logging.warning(f"Skipping fidelity and scatter plots for {scenario} because beta_obs_slopes are missing.")

                # --- Log summary for the scenario ---
                storyline_classification_2d = cmip6_results.get('storyline_classification_2d')
                if storyline_classification_2d:
                    logging.info(f"\n\n--- CMIP6 Model 2D Storyline Classification for {scenario.upper()} ---")
                    radius = Config.STORYLINE_INNER_RADIUS
                    logging.info(f"Models are classified based on zones relative to MMM.")
                    for gwl, storylines in sorted(storyline_classification_2d.items()):
                        if gwl not in Config.GLOBAL_WARMING_LEVELS: continue
                        logging.info(f"\nGWL +{gwl}°C:")
                        if not storylines:
                            logging.info("   No models were classified for this GWL.")
                            continue
                        for storyline_key, models in sorted(storylines.items()):
                            season = 'Winter (DJF)' if 'DJF' in storyline_key else 'Summer (JJA)'
                            # Remove season prefix for cleaner name
                            storyline_name = storyline_key.replace('DJF_', '').replace('JJA_', '')
                            if models:
                                logging.info(f"   - Storyline '{storyline_name}' ({season}):")
                                logging.info(f"     - Models ({len(models)}): {', '.join(sorted(models))}")
                            # Optional: Nur loggen, wenn Modelle vorhanden sind, um Log zu verkürzen
                            # else:
                            #     logging.info(f"   - Storyline '{storyline_name}' ({season}): No models classified.")
                    logging.info("-----------------------------------------------------------------")

            except Exception as e:
                logging.error(f"A critical error occurred during the CMIP6/Storyline analysis for scenario {scenario}: {e}")
                logging.error(traceback.format_exc())

            logging.info(f"\n{'='*25} FINISHED CMIP6 ANALYSIS FOR SCENARIO: {scenario.upper()} {'='*25}\n")

        # =================================================================================
        # === END OF SCENARIO-SPECIFIC LOOP ===
        # =================================================================================

        logging.info("\n\n=====================================================")
        logging.info("=== FULL ANALYSIS COMPLETED ===")
        logging.info(f"All plots saved to: {Config.PLOT_DIR}")
        try:
            logging.info(f"Log file saved to: {log_filename}")
        except NameError:
            logging.info("Log file location is not defined in this scope.")
        logging.info("=====================================================\n")

        return regression_results
    
def main():
    """Main entry point for the program."""
    logging.info("Initializing climate analysis tool...")
    try:
        analysis_results = ClimateAnalysis.run_full_analysis()
        return analysis_results
    except Exception as e:
        logging.critical(f"A critical error occurred in the main execution: {e}")
        logging.critical(traceback.format_exc())
        return None

if __name__ == "__main__":
    main()