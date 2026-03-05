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
    def load_historical_qobs_long_term(config):
        """
        Loads the long-term monthly discharge data (1893-2021) from the Excel file.
        Columns: Month (2), Year (3), Discharge (8).
        1-based indices in description correspond to 0-based: Month=1, Year=2, Vol=7.
        """
        logging.info("Loading LONG-TERM MONTHLY historical QOBS discharge data from Excel...")
        filepath = config.DISCHARGE_FILE

        if not os.path.exists(filepath):
            logging.error(f"Cannot load long-term QOBS data: File not found at {filepath}")
            return None

        try:
            # Assuming header is in the first row (index 0), data starts from row 2
            df = pd.read_excel(filepath, engine='openpyxl') 
            
            # Extract relevant columns by index (using .iloc)
            # Month is col 1 (2nd col), Year is col 2 (3rd col), Discharge is col 7 (8th col)
            df_subset = df.iloc[:, [1, 2, 7]].copy()
            df_subset.columns = ['month', 'year', 'discharge']
            
            # Create datetime index (setting day to 15 to represent the month)
            df_subset['time'] = pd.to_datetime(dict(year=df_subset['year'], month=df_subset['month'], day=15))
            df_subset = df_subset.set_index('time').sort_index()
            
            # Create xarray DataArray
            da = xr.DataArray(
                df_subset['discharge'].values,
                coords={'time': df_subset.index},
                dims='time', 
                name='discharge',
                attrs={'units': 'm3/s', 'long_name': 'Observed Danube Discharge (Monthly 1893-2021)'}
            )
            
            logging.info(f"Successfully loaded LONG-TERM QOBS data: {da.time.size} months ({da.time.dt.year.min().item()}-{da.time.dt.year.max().item()}).")
            return da

        except Exception as e:
            logging.error(f"Error processing long-term QOBS from {filepath}: {e}")
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
        
        # 2. NEU: Lade QOBS-Daten (1893-2021, MONATLICH) aus der .excel-Datei
        qobs_historical_long_term_da = ClimateAnalysis.load_historical_qobs_long_term(Config())

        # 3. Verarbeite QOBS-Daten, um historische Zeitreihen zu erstellen UND fixe Schwellenwerte hinzuzufügen
        # Diese 'discharge_data_loaded' wird für alle historischen Plots (Korrelationen) 
        # UND für die fixen Schwellenwerte (LNWL) in den Zukunfts-Plots verwendet.
        # WICHTIG: Dieses dict enthält jetzt 'daily_historical_da' UND 'monthly_historical_da'
        discharge_data_loaded = ClimateAnalysis.process_historical_discharge_data(qobs_historical_da)
        
        # 4. Füge die Langzeit-Daten zum geladenen Dict hinzu (falls erfolgreich geladen)
        if qobs_historical_long_term_da is not None:
             discharge_data_loaded['monthly_historical_long_term_da'] = qobs_historical_long_term_da
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
            if dset_key == Config.DATASET_ERA5:
                # regression_plot_filename = os.path.join(Config.PLOT_DIR, 'Figure1_regression_maps_ERA5.png') # Reverted overwrite
                pass
            # Check if we need to calculate results (either old plot OR new ERL plot missing)
            erl_fig1_filename = os.path.join(Config.PLOT_DIR, 'Figure1_regression_maps_ERA5.png')
            need_calc = not os.path.exists(regression_plot_filename)
            if dset_key == Config.DATASET_ERA5 and not os.path.exists(erl_fig1_filename):
                need_calc = True

            if need_calc:
                logging.info(f"Calculating regression data for {dset_key}...")
                results = StorylineAnalyzer.calculate_regression_maps(
                    datasets=datasets_reanalysis,
                    dataset_key=dset_key,
                    regression_period=regression_period
                )
                if results:
                    regression_results[dset_key] = results
                    
                    # Plot standard version if missing
                    if not os.path.exists(regression_plot_filename):
                        Visualizer.plot_regression_analysis(results, dset_key)
                    
                    # Plot ERL Figure 1 if missing (ERA5 only)
                    if dset_key == Config.DATASET_ERA5 and not os.path.exists(erl_fig1_filename):
                        Visualizer.plot_erl_figure1_regression_maps(results, dset_key)
                else:
                    logging.warning(f"Could not calculate regression for {dset_key}.")
            else:
                logging.info(f"Regression plots for {dset_key} already exist. Skipping.")

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

        # --- NEW: Danube Box Correlation Plot ---
        danube_corr_filename = os.path.join(Config.PLOT_DIR, "danube_box_correlation_comparison.png")
        if not os.path.exists(danube_corr_filename):
            logging.info(f"Plot '{danube_corr_filename}' not found. Creating Danube Flow vs Box Indices correlation plot...")
            if discharge_data_loaded:
                Visualizer.plot_danube_box_correlation(datasets_reanalysis, discharge_data_loaded)
            else:
                 logging.warning("Skipping Danube box correlation plot: No discharge data loaded.")
        else:
             logging.info(f"Plot '{danube_corr_filename}' already exists.")

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

        aggregated_metric_timeseries = {}

        for scenario in Config.CMIP6_SCENARIOS:
            logging.info(f"\n\n{'='*25} STARTING CMIP6 ANALYSIS FOR SCENARIO: {scenario.upper()} {'='*25}\n")

            try:
                # --- PART 2: CMIP6 AND STORYLINE ANALYSIS (per scenario) ---
                cmip6_results = storyline_analyzer.analyze_cmip6_changes_at_gwl(scenario_to_process=scenario)

                if not cmip6_results:
                    logging.warning(f"CMIP6 analysis did not produce results for scenario {scenario}. Skipping.")
                    continue

                if 'model_metric_timeseries' in cmip6_results:
                    aggregated_metric_timeseries.update(cmip6_results['model_metric_timeseries'])

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
                # --- PLOT: Climate Evolution Timeseries (per scenario) ---
                evolution_plot_filename = os.path.join(Config.PLOT_DIR, f"climate_indices_evolution_{scenario}.png")
                erl_fig2_filename = os.path.join(Config.PLOT_DIR, f"Figure2_climate_indices_evolution_{scenario}.png")
                
                need_calc_evo = not os.path.exists(evolution_plot_filename)
                if not os.path.exists(erl_fig2_filename):
                    need_calc_evo = True
                
                # Store globally for scenario scope
                cmip6_plot_data_stored = None
                reanalysis_plot_data_stored = None

                # Check if Final Figure 3 needs it
                need_calc_final_fig3 = False
                for gwl in Config.GLOBAL_WARMING_LEVELS:
                    final_fig3_fn = os.path.join(Config.PLOT_DIR, f"final_figure_3_{Config.COMPOSITE_EVENT_KEY}_{scenario}_gwl{gwl}.png")
                    if not os.path.exists(final_fig3_fn):
                        need_calc_final_fig3 = True
                        break
                
                if need_calc_final_fig3 or need_calc_evo:
                    logging.info(f"Calculating data for climate evolution and final fig 3 ({scenario})...")
                    cmip6_plot_data_stored, reanalysis_plot_data_stored = StorylineAnalyzer.analyze_timeseries_for_projection_plot(cmip6_results, datasets_reanalysis, Config())
                    
                    if cmip6_plot_data_stored and reanalysis_plot_data_stored:
                        # Plot standard version if missing
                        if not os.path.exists(evolution_plot_filename):
                            Visualizer.plot_climate_projection_timeseries(cmip6_plot_data_stored, reanalysis_plot_data_stored, Config(), filename=os.path.basename(evolution_plot_filename))
                        
                        # Plot ERL Figure 2 if missing
                        if not os.path.exists(erl_fig2_filename):
                            Visualizer.plot_erl_figure2_climate_projection_timeseries(cmip6_plot_data_stored, reanalysis_plot_data_stored, Config(), scenario=scenario)
                    else:
                         logging.warning(f"Skipping climate evolution plot for {scenario}, data preparation failed.")
                else:
                    logging.info(f"Climate evolution plots and final fig 3 base for {scenario} already exist. Skipping.")

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


                # --- START: MODIFIED BLOCK (v3 - New Plot Layout by Event) ---
                # --- PLOT: Storyline Discharge Return Periods (Rows=Half-Year, Cols=Event) (per scenario) ---
                # Define the new filename for the new plot layout
                return_period_plot_filename = os.path.join(Config.PLOT_DIR, f"storyline_discharge_return_period_BY_EVENT_{scenario}.png")
                
                return_period_results_for_plot = None # Initialize

                # Calculate the data (needed for this plot AND the next one)
                # Calculate the data (needed for this plot AND the next one)
                historical_da = discharge_data_loaded.get('daily_historical_da') # DAILY QOBS data
                historical_long_term_da = discharge_data_loaded.get('monthly_historical_long_term_da') # LONG-TERM MONTHLY data

                if historical_da is not None:
                    # --- NEW: Inject Extreme/Non-Extreme into storyline_classification_2d ---
                    # We inject them here so `analyze_storyline_discharge_extremes` calculates GEV return periods for them,
                    # which is then used by `plot_core_finding_gev_panel` as per user request.
                    storyline_classification_2d = cmip6_results.get('storyline_classification_2d')
                    if storyline_classification_2d:
                        for gwl in Config.GLOBAL_WARMING_LEVELS:
                            if gwl not in storyline_classification_2d:
                                storyline_classification_2d[gwl] = {}
                            
                            for season_name, season_prefix in [('Winter', 'DJF'), ('Summer', 'JJA')]:
                                event_key = Config.COMPOSITE_EVENT_KEY
                                res = storyline_analyzer.get_composite_extreme_models(cmip6_results, gwl, event_key, season_name)
                                if res[0] is not None:
                                    ext_models, non_ext_models, _ = res
                                    storyline_classification_2d[gwl][f"{season_prefix}_Extreme Models"] = ext_models
                                    storyline_classification_2d[gwl][f"{season_prefix}_Non-Extreme Models"] = non_ext_models
                                    logging.info(f"Injected {len(ext_models)} Extreme and {len(non_ext_models)} Non-Extreme models for {season_prefix} GWL {gwl}.")

                    logging.info(f"Calculating half-year EVA return period data for {scenario}...")
                    return_period_results_for_plot = storyline_analyzer.analyze_storyline_discharge_extremes(
                        cmip6_results=cmip6_results,
                        historical_discharge_da=historical_da, # <-- Pass DAILY QOBS here
                        config=Config(),
                        discharge_thresholds=discharge_data_loaded, # Pass fixed thresholds
                        historical_discharge_long_term_da=historical_long_term_da # <-- Pass NEW Long-Term Data
                    )
                else:
                    logging.warning(f"DAILY QOBS historical discharge data not available for return period analysis in {scenario}.")

                # Create the new plot if it's missing
                if True or not os.path.exists(return_period_plot_filename):
                    if return_period_results_for_plot:
                        # --- THIS CALLS THE MODIFIED FUNCTION ---
                        Visualizer.plot_storyline_return_period_half_year(return_period_results_for_plot, Config(), scenario=scenario)
                        
                        # --- NEW: VERIFICATION PLOT ---
                        Visualizer.plot_historical_seasonal_verification(return_period_results_for_plot, Config(), scenario=scenario)
                    else:
                        logging.warning(f"Could not calculate return period results, skipping plot for {scenario}.")
                else:
                    logging.info(f"Half-year return period plot '{return_period_plot_filename}' already exists.")

                # --- PLOT: Figure 3 (Core Finding GEV Panel) ---
                if return_period_results_for_plot:
                     fig3_filename = os.path.join(Config.PLOT_DIR, f"Figure3_core_finding_regime_shift_{scenario}.png")
                     if True or not os.path.exists(fig3_filename):
                         Visualizer.plot_core_finding_gev_panel(return_period_results_for_plot, Config(), scenario)
                         Visualizer.plot_final_figure_2_shift_and_verification(return_period_results_for_plot, Config(), scenario)
                     else:
                         logging.info(f"Figure 3 '{fig3_filename}' already exists.")

                # --- PLOT: Discharge Events Timeseries (30Q10 & Lowflow) ---
                discharge_extreme_plot_filename = os.path.join(Config.PLOT_DIR, f"storyline_discharge_events_extremes_{scenario}.png")
                
                if True or not os.path.exists(discharge_extreme_plot_filename):
                    logging.info(f"Discharge extreme events plot not found or requested to recreate. Creating...")
                    Visualizer.plot_discharge_events_extreme_timeseries(cmip6_results, discharge_data_loaded, Config(), scenario)
                else:
                    logging.info(f"Discharge extreme events plot already exists.")

                # --- PLOT: Z500 Composite Analysis (Extreme vs Non-Extreme) ---
                # Added Feb 2026
                composite_event_key = Config.COMPOSITE_EVENT_KEY
                # composite_quantile = Config.COMPOSITE_QUANTILE # Deprecated
                
                # Check for 30Q30/30Q10 data availability first to avoid useless calls
                # But calculate_z500... checks internally.
                
                if scenario == 'ssp585':
                    for gwl in Config.GLOBAL_WARMING_LEVELS:
                        for composite_season in ['Winter', 'Summer']:
                            composite_plot_filename = os.path.join(Config.PLOT_DIR, f"composite_analysis_z500_{composite_season.lower()}_{composite_event_key}_{scenario}_gwl{gwl}.png")
                            if not os.path.exists(composite_plot_filename):
                                logging.info(f"Running Z500 composite analysis for GWL +{gwl}°C, Season {composite_season}, Event {composite_event_key}...")
                                result_tuple = storyline_analyzer.calculate_z500_composites_for_extremes(
                                    cmip6_results, gwl=gwl, event_key=composite_event_key, season=composite_season
                                )
                                if result_tuple:
                                    composite_results, model_lists, model_rps, n_total_models = result_tuple
                                    if composite_results:
                                        Visualizer.plot_z500_composite_analysis_panel(
                                            composite_results, gwl, composite_event_key, scenario, composite_season,
                                            model_rps, model_lists, n_total_models
                                        )
                                    else:
                                        logging.warning(f"Z500 composite analysis returned empty results for GWL {gwl}, {composite_season}.")
                                else:
                                    logging.warning(f"Z500 composite analysis returned no results for GWL {gwl}, {composite_season}.")
                            else:
                                logging.info(f"Z500 composite plot for GWL {gwl}, {composite_season} already exists.")

                # --- PLOT: PSL Composite Analysis (Extreme vs Non-Extreme) ---
                # Added Feb 2026 - Mirrors Z500 composite but for sea level pressure
                for gwl in Config.GLOBAL_WARMING_LEVELS:
                    for composite_season in ['Winter', 'Summer']:
                        psl_composite_plot_filename = os.path.join(Config.PLOT_DIR, f"composite_analysis_psl_{composite_season.lower()}_{composite_event_key}_{scenario}_gwl{gwl}.png")
                        if not os.path.exists(psl_composite_plot_filename):
                            logging.info(f"Running PSL composite analysis for GWL +{gwl}°C, Season {composite_season}, Event {composite_event_key}...")
                            psl_result_tuple = storyline_analyzer.calculate_psl_composites_for_extremes(
                                cmip6_results, gwl=gwl, event_key=composite_event_key, season=composite_season
                            )
                            if psl_result_tuple:
                                psl_composite_results, psl_model_lists, psl_model_rps, psl_n_total_models = psl_result_tuple
                                if psl_composite_results:
                                    Visualizer.plot_psl_composite_analysis_panel(
                                        psl_composite_results, gwl, composite_event_key, scenario, composite_season,
                                        psl_model_rps, psl_model_lists, psl_n_total_models
                                    )
                                else:
                                    logging.warning(f"PSL composite analysis returned empty results for GWL {gwl}, {composite_season}.")
                            else:
                                logging.warning(f"PSL composite analysis returned no results for GWL {gwl}, {composite_season}.")
                        else:
                            logging.info(f"PSL composite plot for GWL {gwl}, {composite_season} already exists.")

                # --- PLOT: PR Composite Analysis (Extreme vs Non-Extreme) ---
                # Added Feb 2026 - Mirrors Z500 composite but for precipitation
                pr_stored_composites = {}  # Store results for combined plot
                for gwl in Config.GLOBAL_WARMING_LEVELS:
                    for composite_season in ['Winter', 'Summer']:
                        pr_composite_plot_filename = os.path.join(Config.PLOT_DIR, f"composite_analysis_pr_{composite_season.lower()}_{composite_event_key}_{scenario}_gwl{gwl}.png")
                        pr_combined_plot_filename = os.path.join(Config.PLOT_DIR, f"combined_diff_pr_{composite_event_key}_{scenario}_gwl{gwl}.png")
                        need_compute = not os.path.exists(pr_composite_plot_filename) or not os.path.exists(pr_combined_plot_filename)
                        if need_compute:
                            logging.info(f"Running PR composite analysis for GWL +{gwl}°C, Season {composite_season}, Event {composite_event_key}...")
                            pr_result_tuple = storyline_analyzer.calculate_pr_composites_for_extremes(
                                cmip6_results, gwl=gwl, event_key=composite_event_key, season=composite_season
                            )
                            if pr_result_tuple:
                                pr_composite_results, pr_model_lists, pr_model_rps, pr_n_total_models = pr_result_tuple
                                if pr_composite_results:
                                    pr_stored_composites[(gwl, composite_season)] = (pr_composite_results, pr_model_rps, pr_n_total_models)
                                    if not os.path.exists(pr_composite_plot_filename):
                                        Visualizer.plot_pr_composite_analysis_panel(
                                            pr_composite_results, gwl, composite_event_key, scenario, composite_season,
                                            pr_model_rps, pr_model_lists, pr_n_total_models
                                        )
                                else:
                                    logging.warning(f"PR composite analysis returned empty results for GWL {gwl}, {composite_season}.")
                            else:
                                logging.warning(f"PR composite analysis returned no results for GWL {gwl}, {composite_season}.")
                        else:
                            logging.info(f"PR composite plot for GWL {gwl}, {composite_season} already exists.")

                # --- Combined PR Difference Plots (Winter + Summer) ---
                pr_shared_diff_limit = None
                if pr_stored_composites:
                    all_diff_maps = []
                    for comp_data in pr_stored_composites.values():
                        comp = comp_data[0]
                        if comp:
                            for key in ['diff_ext_non_future', 'diff_ext_non_hist']:
                                m = comp.get(key)
                                if m is not None:
                                    all_diff_maps.append(m)
                    if all_diff_maps:
                        all_vals = np.concatenate([m.values.ravel() for m in all_diff_maps])
                        all_vals = all_vals[np.isfinite(all_vals)]
                        if len(all_vals) > 0:
                            import math
                            limit = np.percentile(np.abs(all_vals), 98)
                            if limit > 0:
                                pr_shared_diff_limit = math.ceil(limit)

                for gwl in Config.GLOBAL_WARMING_LEVELS:
                    pr_combined_fn = os.path.join(Config.PLOT_DIR, f"combined_diff_pr_{composite_event_key}_{scenario}_gwl{gwl}.png")
                    if not os.path.exists(pr_combined_fn):
                        w_data = pr_stored_composites.get((gwl, 'Winter'))
                        s_data = pr_stored_composites.get((gwl, 'Summer'))
                        if w_data and s_data:
                            Visualizer.plot_pr_combined_composite_diff_panel(
                                winter_composite=w_data[0], summer_composite=s_data[0],
                                gwl=gwl, event_key=composite_event_key, scenario=scenario,
                                winter_model_rps=w_data[1], summer_model_rps=s_data[1],
                                winter_n_total=w_data[2], summer_n_total=s_data[2],
                                fixed_diff_limit=pr_shared_diff_limit
                            )
                        else:
                            logging.warning(f"Cannot create combined PR diff plot for GWL {gwl}: missing season data.")
                    else:
                        logging.info(f"Combined PR diff plot for GWL {gwl} already exists.")

                # --- PLOT: UA Composite Analysis (Zonal Wind 850hPa) ---
                ua_stored_composites = {}  # Store results for combined plot
                for gwl in Config.GLOBAL_WARMING_LEVELS:
                    final_fig3_fn_check = os.path.join(Config.PLOT_DIR, f"final_figure_3_{composite_event_key}_{scenario}_gwl{gwl}.png")
                    for composite_season in ['Winter', 'Summer']:
                        ua_composite_plot_filename = os.path.join(Config.PLOT_DIR, f"composite_analysis_ua850_{composite_season.lower()}_{composite_event_key}_{scenario}_gwl{gwl}.png")
                        ua_combined_plot_filename = os.path.join(Config.PLOT_DIR, f"combined_diff_ua850_{composite_event_key}_{scenario}_gwl{gwl}.png")
                        need_compute = (not os.path.exists(ua_composite_plot_filename) or 
                                        not os.path.exists(ua_combined_plot_filename) or
                                        not os.path.exists(final_fig3_fn_check))
                        
                        # ALways calculate composites to store it in memory for the final fig 3 
                        # even if plots exist, if final_fig3 is missing OR we need it.
                        # Wait, we can just ALWAYS calculate it if ANY plot needs it, 
                        # OR if we just want to ensure we have the data.
                        
                        compute_data = need_compute
                        # Actually to be safe let's just compute to get the data if final fig 3 is needed
                        # wait, earlier we already changed need_compute to include final_fig3_fn_check. 
                        # So it DOES compute!
                        
                        if need_compute:
                            logging.info(f"Running UA composite analysis for GWL +{gwl}°C, Season {composite_season}, Event {composite_event_key}...")
                            ua_result_tuple = storyline_analyzer.calculate_ua_composites_for_extremes(
                                cmip6_results, gwl=gwl, event_key=composite_event_key, season=composite_season
                            )
                            if ua_result_tuple:
                                ua_composite_results, ua_model_lists, ua_model_rps, ua_n_total_models = ua_result_tuple
                                if ua_composite_results:
                                    ua_stored_composites[(gwl, composite_season)] = (ua_composite_results, ua_model_rps, ua_n_total_models)
                                    if not os.path.exists(ua_composite_plot_filename):
                                        Visualizer.plot_ua_composite_analysis_panel(
                                            ua_composite_results, gwl, composite_event_key, scenario, composite_season,
                                            ua_model_rps, ua_model_lists, ua_n_total_models
                                        )
                                else:
                                    logging.warning(f"UA composite analysis returned empty results for GWL {gwl}, {composite_season}.")
                            else:
                                logging.warning(f"UA composite analysis returned no results for GWL {gwl}, {composite_season}.")
                        else:
                            logging.info(f"UA composite plot for GWL {gwl}, {composite_season} already exists and data not needed.")
                            # Still need to load it if final fig 3 needs it? NO, need_compute is True if final fig 3 is missing.
                            # BUT WHAT IF final fig 3 exists? Then we don't need it.
                            # Wait, the warning was "Cannot create final figure 3 for GWL 2.0: missing season data."
                            # If need_compute was True, it should have populated ua_stored_composites.
                            # Let's just FORCE computation of the data (without plotting if plots exist).
                            ua_result_tuple = storyline_analyzer.calculate_ua_composites_for_extremes(
                                cmip6_results, gwl=gwl, event_key=composite_event_key, season=composite_season
                            )
                            if ua_result_tuple:
                                ua_composite_results, ua_model_lists, ua_model_rps, ua_n_total_models = ua_result_tuple
                                if ua_composite_results:
                                    ua_stored_composites[(gwl, composite_season)] = (ua_composite_results, ua_model_rps, ua_n_total_models)


                # --- Combined UA Difference Plots (Winter + Summer) ---
                ua_shared_diff_limit = None
                if ua_stored_composites:
                    all_diff_maps = []
                    for comp_data in ua_stored_composites.values():
                        comp = comp_data[0]
                        if comp:
                            for key in ['diff_ext_non_future', 'diff_ext_non_hist']:
                                m = comp.get(key)
                                if m is not None:
                                    all_diff_maps.append(m)
                    if all_diff_maps:
                        all_vals = np.concatenate([m.values.ravel() for m in all_diff_maps])
                        all_vals = all_vals[np.isfinite(all_vals)]
                        if len(all_vals) > 0:
                            import math
                            limit = np.percentile(np.abs(all_vals), 98)
                            if limit > 0:
                                ua_shared_diff_limit = math.ceil(limit)

                for gwl in Config.GLOBAL_WARMING_LEVELS:
                    ua_combined_fn = os.path.join(Config.PLOT_DIR, f"combined_diff_ua850_{composite_event_key}_{scenario}_gwl{gwl}.png")
                    if not os.path.exists(ua_combined_fn):
                        w_data = ua_stored_composites.get((gwl, 'Winter'))
                        s_data = ua_stored_composites.get((gwl, 'Summer'))
                        if w_data and s_data:
                            Visualizer.plot_ua_combined_composite_diff_panel(
                                winter_composite=w_data[0], summer_composite=s_data[0],
                                gwl=gwl, event_key=composite_event_key, scenario=scenario,
                                winter_model_rps=w_data[1], summer_model_rps=s_data[1],
                                winter_n_total=w_data[2], summer_n_total=s_data[2],
                                fixed_diff_limit=ua_shared_diff_limit
                            )
                        else:
                            logging.warning(f"Cannot create combined UA diff plot for GWL {gwl}: missing season data.")
                    else:
                        logging.info(f"Combined UA diff plot for GWL {gwl} already exists.")

                    # --- NEW: Final Figure 3 Plot ---
                    final_fig3_fn = os.path.join(Config.PLOT_DIR, f"final_figure_3_{composite_event_key}_{scenario}_gwl{gwl}.png")
                    if not os.path.exists(final_fig3_fn):
                        w_data = ua_stored_composites.get((gwl, 'Winter'))
                        s_data = ua_stored_composites.get((gwl, 'Summer'))
                        if w_data and s_data:
                            Visualizer.plot_final_figure_3_u850_and_indices(
                                winter_composite=w_data[0], summer_composite=s_data[0],
                                gwl=gwl, event_key=composite_event_key, scenario=scenario,
                                cmip6_plot_data=cmip6_plot_data_stored, reanalysis_plot_data=reanalysis_plot_data_stored, config=Config(),
                                winter_model_rps=w_data[1], summer_model_rps=s_data[1],
                                winter_n_total=w_data[2], summer_n_total=s_data[2],
                                fixed_diff_limit=ua_shared_diff_limit
                            )
                        else:
                            logging.warning(f"Cannot create final figure 3 for GWL {gwl}: missing season data.")
                    else:
                        logging.info(f"Final figure 3 for GWL {gwl} already exists.")

                # --- PLOT: TAS Composite Analysis (Surface Temperature) ---
                for gwl in Config.GLOBAL_WARMING_LEVELS:
                    for composite_season in ['Winter', 'Summer']:
                        tas_composite_plot_filename = os.path.join(Config.PLOT_DIR, f"composite_analysis_tas_{composite_season.lower()}_{composite_event_key}_{scenario}_gwl{gwl}.png")
                        if not os.path.exists(tas_composite_plot_filename):
                            logging.info(f"Running TAS composite analysis for GWL +{gwl}°C, Season {composite_season}, Event {composite_event_key}...")
                            tas_result_tuple = storyline_analyzer.calculate_tas_composites_for_extremes(
                                cmip6_results, gwl=gwl, event_key=composite_event_key, season=composite_season
                            )
                            if tas_result_tuple:
                                tas_composite_results, tas_model_lists, tas_model_rps, tas_n_total_models = tas_result_tuple
                                if tas_composite_results:
                                    Visualizer.plot_tas_composite_analysis_panel(
                                        tas_composite_results, gwl, composite_event_key, scenario, composite_season,
                                        tas_model_rps, tas_model_lists, tas_n_total_models
                                    )
                                else:
                                    logging.warning(f"TAS composite analysis returned empty results for GWL {gwl}, {composite_season}.")
                            else:
                                logging.warning(f"TAS composite analysis returned no results for GWL {gwl}, {composite_season}.")
                        else:
                            logging.info(f"TAS composite plot for GWL {gwl}, {composite_season} already exists.")

                # --- PLOT: Storyline Impacts Bar Chart (per scenario) ---
                # (This block remains the same, but it uses the 'return_period_results_for_plot' calculated above)
                impacts_plot_filename = os.path.join(Config.PLOT_DIR, f"storyline_impacts_summary_4x2_boxplots_{scenario}.png")
                if not os.path.exists(impacts_plot_filename):
                    logging.info(f"Plot '{impacts_plot_filename}' not found, creating...")
                    if cmip6_results:
                        storyline_correlations = None # Optional

                        # Re-map the new threshold structure to the old one expected by this specific plot function
                        threshold_data_for_plot = None
                        if return_period_results_for_plot and 'thresholds' in return_period_results_for_plot:
                             threshold_data_for_plot = {}
                             
                             # Map winter/summer thresholds to all relevant keys
                             winter_events = return_period_results_for_plot['thresholds'].get('winter', {})
                             summer_events = return_period_results_for_plot['thresholds'].get('summer', {})
                             
                             def map_events_to_old_struct(events_dict):
                                 old_struct = {}
                                 for k, v in events_dict.items():
                                     old_struct[v['name']] = v
                                 return old_struct

                             winter_events_old_struct = map_events_to_old_struct(winter_events)
                             summer_events_old_struct = map_events_to_old_struct(summer_events)
                             
                             keys_winter = ['DJF_discharge', 'Mar_discharge', 'Apr_discharge', 'May_discharge']
                             keys_summer = ['JJA_discharge', 'Sep_discharge', 'Oct_discharge', 'Nov_discharge']
                             
                             # (Note: 'historical_std_dev' is not used in the plotting function, so we can omit it)
                             for k in keys_winter:
                                 threshold_data_for_plot[k] = winter_events_old_struct
                             for k in keys_summer:
                                 threshold_data_for_plot[k] = summer_events_old_struct = map_events_to_old_struct(summer_events)

                        if threshold_data_for_plot: # Only plot if thresholds are available
                            Visualizer.plot_storyline_impact_barchart_with_discharge(
                                cmip6_results=cmip6_results,
                                threshold_data=threshold_data_for_plot, # <-- Pass re-mapped data
                                discharge_data_historical=discharge_data_loaded,
                                reanalysis_data=datasets_reanalysis,
                                config=Config(),
                                scenario=scenario,
                                storyline_correlations=storyline_correlations
                            )
                        else:
                            logging.warning(f"Skipping impact plot for {scenario} because threshold data could not be calculated or re-mapped.")
                    else:
                        logging.warning(f"Skipping impact plot for {scenario} because cmip6_results are missing.")
                else:
                    logging.info(f"Plot '{impacts_plot_filename}' already exists.")
                # --- END: MODIFIED BLOCK ---

                # --- PLOT: Figure 4 (Mechanism Drivers Panel) ---
                if scenario == 'ssp585' and cmip6_results:
                    fig4_filename = os.path.join(Config.PLOT_DIR, "Figure4_mechanism_drivers_summary_ssp585.png")
                    if not os.path.exists(fig4_filename):
                        storyline_impacts = StorylineAnalyzer.calculate_storyline_impacts(cmip6_results)
                        Visualizer.plot_mechanism_drivers_panel(cmip6_results, Config(), scenario)
                    else:
                         logging.info(f"Figure 4 '{fig4_filename}' already exists.")


                # --- PLOT: LNWL Monthly Distribution (NEW PLOT v2: Grid Plot) ---
                lnwl_dist_plot_filename = os.path.join(Config.PLOT_DIR, f"storyline_lnwl_monthly_distribution_{scenario}.png")
                if not os.path.exists(lnwl_dist_plot_filename):
                    logging.info(f"Plot '{lnwl_dist_plot_filename}' not found. Calculating and creating plot...")
                    
                    # Hole den LNWL-Schwellenwert (sollte 970 sein)
                    lnwl_threshold = discharge_data_loaded.get('winter_lowflow_lnwl') 
                    # Hole die vollen täglichen QOBS-Daten (geladen am Anfang von run_full_analysis)
                    hist_qobs_daily = discharge_data_loaded.get('daily_historical_da') 

                    if lnwl_threshold and hist_qobs_daily is not None and cmip6_results:
                        # Rrufe die neue Analysefunktion auf, die das volle cmip6_results-Dict benötigt
                        distribution_data = StorylineAnalyzer.analyze_lnwl_monthly_distribution_by_storyline(
                            cmip6_results=cmip6_results, # Enthält Klassifikation, Modelldaten, GWL-Jahre
                            qobs_historical_da=hist_qobs_daily,
                            lnwl_threshold=lnwl_threshold,
                            hist_period_qobs=(1995, 2014), # Referenzzeitraum für QOBS wie gewünscht
                            config=Config()
                        )
                        
                        if distribution_data:
                            # Rufe die neue Plot-Funktion auf
                            Visualizer.plot_storyline_lnwl_monthly_distribution(
                                distribution_data=distribution_data,
                                scenario=scenario,
                                config=Config(),
                                lnwl_threshold=lnwl_threshold
                            )
                        else:
                            logging.warning(f"Could not calculate LNWL distribution data for {scenario}.")
                    else:
                        logging.warning(f"Skipping LNWL distribution plot: Missing QOBS, CMIP6 results, or LNWL threshold.")
                else:
                    logging.info(f"Plot '{lnwl_dist_plot_filename}' already exists. Skipping.")
                # --- END: NEW PLOT v2 ---

                
                # --- START: NEUER PLOT (LNWL Aggregation Comparison & NEW FIGURE 4) ---
                lnwl_agg_plot_filename = os.path.join(Config.PLOT_DIR, f"storyline_lnwl_aggregation_comparison_{scenario}.png")
                erl_fig4_filename = os.path.join(Config.PLOT_DIR, f"Figure4_impact_navigation_lnwl_{scenario}.png") # NEW
                
                # Check if we need to calculate (either big comparison OR Fig 4 missing)
                if True or not os.path.exists(lnwl_agg_plot_filename) or not os.path.exists(erl_fig4_filename):
                    logging.info(f"Calculating LNWL aggregation data...")
                    
                    # Hole die TÄGLICHEN QOBS-Daten (am Anfang von run_full_analysis geladen)
                    historical_da_daily = discharge_data_loaded.get('daily_historical_da') 
                    
                    if historical_da_daily is not None and cmip6_results:
                        # Hole den LNWL-Schwellenwert
                        lnwl_threshold = discharge_data_loaded.get('winter_lowflow_lnwl', 970.0)
                        
                        # Rufe die Analysefunktion auf
                        lnwl_agg_results_for_plot = storyline_analyzer.analyze_storyline_lnwl_aggregation_metrics(
                            cmip6_results=cmip6_results, # Benötigt 'cmip6_model_data_loaded' mit TÄGLICHEN Daten
                            historical_discharge_da=historical_da_daily,
                            config=Config(),
                            lnwl_threshold=lnwl_threshold
                        )
                        
                        if lnwl_agg_results_for_plot:
                            # 1. Erstelle den großen Vergleichs-Plot (falls nicht existent)
                            if True or not os.path.exists(lnwl_agg_plot_filename):
                                Visualizer.plot_storyline_lnwl_aggregation_comparison(
                                    lnwl_agg_results_for_plot, 
                                    Config(), 
                                    scenario=scenario,
                                    lnwl_threshold=lnwl_threshold
                                )
                            
                            # 2. Erstelle ERL Figure 4
                            if True or not os.path.exists(erl_fig4_filename):
                                Visualizer.plot_erl_figure4_lnwl_summary(
                                    lnwl_agg_results_for_plot,
                                    Config(),
                                    scenario=scenario,
                                    lnwl_threshold=lnwl_threshold
                                )

                        else:
                            logging.warning(f"Could not calculate LNWL aggregation results, skipping plots for {scenario}.")
                    else:
                        logging.warning(f"Skipping LNWL aggregation plot: Missing DAILY QOBS or CMIP6 results.")
                else:
                    logging.info(f"LNWL plots already exists.")
                # --- ENDE: NEUER PLOT ---


                # --- PLOT: Figure 5 (Formerly Figure 4) (Mechanism Drivers Panel) ---
                if cmip6_results:
                    fig5_filename = os.path.join(Config.PLOT_DIR, f"Figure5_mechanism_drivers_summary_{scenario}.png")
                    if not os.path.exists(fig5_filename):
                        # Visualizer.plot_mechanism_drivers_panel now creates Figure 5
                        Visualizer.plot_mechanism_drivers_panel(cmip6_results, Config(), scenario)
                    else:
                         logging.info(f"Figure 5 '{fig5_filename}' already exists.")


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

        # --- PLOT: Discharge Events Timeseries (30Q10 & Lowflow) - COMBINED ---
        discharge_events_plot_filename = os.path.join(Config.PLOT_DIR, f"final_figure_1_storyline_discharge_events_combined.png")
        if True or not os.path.exists(discharge_events_plot_filename):
            logging.info(f"Discharge events combined plot not found or requested to recreate. Creating...")
            
            if aggregated_metric_timeseries:
                cmip6_results_combined = {'model_metric_timeseries': aggregated_metric_timeseries}
                Visualizer.plot_discharge_events_timeseries(cmip6_results_combined, cmip6_discharge_loaded, Config(), 'combined')
            else:
                logging.warning("No aggregated metric timeseries data found for final figure 1.")
        else:
            logging.info(f"Discharge events combined plot already exists.")

        # --- PLOT: Discharge Extreme Events Timeseries - COMBINED (SSP245 + SSP585) ---
        final_fig4_filename = os.path.join(Config.PLOT_DIR, "final_figure_4_storyline_discharge_events_extremes_combined.png")
        ssp245_extremes_path = os.path.join(Config.PLOT_DIR, "storyline_discharge_events_extremes_ssp245.png")
        ssp585_extremes_path = os.path.join(Config.PLOT_DIR, "storyline_discharge_events_extremes_ssp585.png")

        if not os.path.exists(final_fig4_filename):
            if os.path.exists(ssp245_extremes_path) and os.path.exists(ssp585_extremes_path):
                logging.info(f"Creating combined extremes plot: {final_fig4_filename}")
                try:
                    from PIL import Image as PILImage
                    img_top = PILImage.open(ssp245_extremes_path)
                    img_bot = PILImage.open(ssp585_extremes_path)
                    width = max(img_top.width, img_bot.width)
                    total_height = img_top.height + img_bot.height
                    combined = PILImage.new('RGB', (width, total_height), 'white')
                    combined.paste(img_top, (0, 0))
                    combined.paste(img_bot, (0, img_top.height))
                    combined.save(final_fig4_filename, dpi=(300, 300))
                    logging.info(f"Saved combined extremes plot: {final_fig4_filename}")
                except Exception as e:
                    logging.error(f"Failed to create combined extremes plot: {e}")
            else:
                logging.warning("Cannot create final_figure_4: one or both per-scenario extremes plots are missing.")
        else:
            logging.info(f"Combined extremes plot '{final_fig4_filename}' already exists.")

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