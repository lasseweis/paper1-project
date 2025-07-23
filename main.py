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
import array as xr 
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
            return {}
        
    @staticmethod
    def process_discharge_data(file_path):
        """Process discharge data from Excel file and compute metrics."""
        logging.info(f"Processing discharge data from {file_path}...")
        try:
            # Load data from Excel
            data = pd.read_excel(file_path, index_col=None, na_values=['NA'], usecols='A,B,C,H')
            df = pd.DataFrame({
                'year': data['year'], 'month': data['month'], 'discharge': data['Wien']
            }).dropna()

            # Calculate thresholds for extreme events
            high_flow_threshold = np.percentile(df['discharge'], 90)
            low_flow_threshold = np.percentile(df['discharge'], 10)
            df['extreme_flow'] = np.select(
                [df['discharge'] > high_flow_threshold, df['discharge'] < low_flow_threshold],
                [1, -1], default=0
            )

            # A datetime index is required for assign_season_to_dataarray
            df['time'] = pd.to_datetime(df[['year', 'month']].assign(day=15))
            df = df.set_index('time')

            result = {}
            for metric in ['discharge', 'extreme_flow']:
                # Create a DataArray for the current metric
                da = df[[metric]].to_xarray()[metric]

                # Assign seasons
                da_with_seasons = DataProcessor.assign_season_to_dataarray(da)

                # Calculate seasonal means
                seasonal_means = DataProcessor.calculate_seasonal_means(da_with_seasons)

                if seasonal_means is not None:
                    for season in ['Winter', 'Summer']:
                        season_lower = season.lower()
                        key = f'{season_lower}_{metric}'

                        # Filter by season and detrend
                        season_data = DataProcessor.filter_by_season(seasonal_means, season)
                        if season_data is not None:
                            result[key] = DataProcessor.detrend_data(season_data)
            return result
        except Exception as e:
            logging.error(f"Error processing discharge data: {e}")
            logging.error(traceback.format_exc())
            return {}

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
    def run_full_analysis():
        """Main static method to execute the entire analysis workflow."""
        logging.info("=====================================================")
        logging.info("=== STARTING FULL CLIMATE ANALYSIS WORKFLOW ===")
        logging.info("=====================================================\n")

        Visualizer.ensure_plot_dir_exists()
        cmip6_results = None
        regression_results = {} # Wird nur gefüllt, wenn die Plots tatsächlich erstellt werden.
        
        # Erstelle eine Instanz der zentralen Analyseklasse
        storyline_analyzer = StorylineAnalyzer(config=Config)


        # --- PART 1: GRUNDLAGEN-BERECHNUNGEN ---
        # Diese Basis-Daten und Jet-Indizes werden immer geladen/berechnet,
        # da sie für potenziell mehrere, unterschiedliche Analysen benötigt werden.
        logging.info("\n--- Processing Reanalysis Datasets ---")
        datasets_reanalysis = {
            **ClimateAnalysis.process_20crv3_data(),
            **ClimateAnalysis.process_era5_data()
        }
        if not datasets_reanalysis:
            logging.critical("Failed to process reanalysis datasets. Aborting.")
            return None

        # --- START: NEUER BLOCK für SPEI-Berechnung ---
        logging.info("\n--- Calculating SPEI for Reanalysis Datasets ---")
        for dset_key in [Config.DATASET_20CRV3, Config.DATASET_ERA5]:
            logging.info(f"  Calculating SPEI for {dset_key}...")
            
            # Lade die monatlichen Daten für das gesamte Gebiet
            pr_monthly_full = datasets_reanalysis.get(f'{dset_key}_pr_monthly')
            tas_monthly_full = datasets_reanalysis.get(f'{dset_key}_tas_monthly')

            if pr_monthly_full is not None and tas_monthly_full is not None:
                # Berechne den räumlichen Mittelwert über die Analyse-Box
                pr_box_monthly = DataProcessor.calculate_spatial_mean(pr_monthly_full, Config.BOX_LAT_MIN, Config.BOX_LAT_MAX, Config.BOX_LON_MIN, Config.BOX_LON_MAX)
                tas_box_monthly = DataProcessor.calculate_spatial_mean(tas_monthly_full, Config.BOX_LAT_MIN, Config.BOX_LAT_MAX, Config.BOX_LON_MIN, Config.BOX_LON_MAX)

                if pr_box_monthly is not None and tas_box_monthly is not None:
                    # Bestimme die zentrale geografische Breite der Box für die PET-Berechnung
                    lat_center_of_box = (Config.BOX_LAT_MIN + Config.BOX_LAT_MAX) / 2
                    
                    # Berechne den SPEI mit einer Zeitskala von 4 Monaten
                    spei4 = DataProcessor.calculate_spei(pr_box_monthly, tas_box_monthly, lat=lat_center_of_box, scale=4)
                    
                    if spei4 is not None:
                        # Für die weitere Analyse benötigen wir die saisonalen Koordinaten
                        spei4_seasonal = DataProcessor.assign_season_to_dataarray(spei4)
                        datasets_reanalysis[f'{dset_key}_spei4'] = spei4_seasonal
                        logging.info(f"    Successfully calculated SPEI-4 for {dset_key}.")
        # --- END: NEUER BLOCK ---
        
        discharge_data_loaded = ClimateAnalysis.process_discharge_data(Config.DISCHARGE_FILE)
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

        # --- AB HIER: PLOT-SPEZIFISCHE BERECHNUNG UND ERSTELLUNG (PRÜFUNG ZUERST) ---

        logging.info("\n--- Checking for Reanalysis Regression Maps ---")
        regression_period = (1981, 2010)
        for dset_key in [Config.DATASET_20CRV3, Config.DATASET_ERA5]:
            regression_plot_filename = os.path.join(Config.PLOT_DIR, f'regression_maps_norm_{dset_key}.png')
            if not os.path.exists(regression_plot_filename):
                logging.info(f"Plot '{regression_plot_filename}' nicht gefunden. Berechne Daten und erstelle Plot...")
                # BERECHNUNG JETZT INNERHALB DER IF-BEDINGUNG
                results = StorylineAnalyzer.calculate_regression_maps(
                    datasets=datasets_reanalysis,
                    dataset_key=dset_key,
                    regression_period=regression_period
                )
                if results:
                    regression_results[dset_key] = results
                    Visualizer.plot_regression_analysis(results, dset_key)
                else:
                    logging.warning(f"Konnte Regression für {dset_key} nicht berechnen.")
            else:
                logging.info(f"Plot '{regression_plot_filename}' existiert bereits. Überspringe Berechnung und Erstellung.")
        
        logging.info("\n\n--- Checking for Reanalysis Jet Index Comparison Timeseries ---")
        jet_indices_plot_filename = os.path.join(Config.PLOT_DIR, "jet_indices_comparison_seasonal_detrended.png")
        if not os.path.exists(jet_indices_plot_filename):
            if jet_data_reanalysis:
                logging.info(f"Plot '{jet_indices_plot_filename}' nicht gefunden, wird erstellt...")
                Visualizer.plot_jet_indices_comparison(jet_data_reanalysis)
            else:
                logging.warning("Skipping jet index comparison plot, no data was generated.")
        else:
            logging.info(f"Plot '{jet_indices_plot_filename}' existiert bereits. Überspringe Erstellung.")

        logging.info("\n\n--- Checking for Reanalysis Jet Impact Comparison Maps ---")
        for season in ['Winter', 'Summer']:
            jet_impact_plot_filename = os.path.join(Config.PLOT_DIR, f'jet_impact_regression_maps_{season.lower()}.png')
            if not os.path.exists(jet_impact_plot_filename):
                logging.info(f"Plot '{jet_impact_plot_filename}' nicht gefunden. Berechne Daten und erstelle Plot...")
                # BERECHNUNG JETZT INNERHALB DER IF-BEDINGUNG
                impact_20crv3 = StorylineAnalyzer.calculate_jet_impact_maps(datasets_reanalysis, jet_data_reanalysis, Config.DATASET_20CRV3, season)
                impact_era5 = StorylineAnalyzer.calculate_jet_impact_maps(datasets_reanalysis, jet_data_reanalysis, Config.DATASET_ERA5, season)

                data_20crv3 = impact_20crv3.get(season)
                data_era5 = impact_era5.get(season)
                
                if data_20crv3 and data_era5:
                    Visualizer.plot_jet_impact_comparison_maps(data_20crv3, data_era5, season)
                else:
                    logging.warning(f"Skipping combined jet impact plot for {season}, data missing for one or both reanalyses.")
            else:
                logging.info(f"Plot '{jet_impact_plot_filename}' existiert bereits. Überspringe Berechnung und Erstellung.")

        logging.info("\n\n--- Checking for Reanalysis Jet Correlation Maps ---")
        for season in ['Winter', 'Summer']:
            jet_corr_plot_filename = os.path.join(Config.PLOT_DIR, f'jet_correlation_maps_{season.lower()}.png')
            if not os.path.exists(jet_corr_plot_filename):
                logging.info(f"Plot '{jet_corr_plot_filename}' nicht gefunden. Berechne Daten und erstelle Plot...")
                # BERECHNUNG JETZT INNERHALB DER IF-BEDINGUNG
                corr_20crv3 = StorylineAnalyzer.calculate_jet_correlation_maps(datasets_reanalysis, jet_data_reanalysis, Config.DATASET_20CRV3, season)
                corr_era5 = StorylineAnalyzer.calculate_jet_correlation_maps(datasets_reanalysis, jet_data_reanalysis, Config.DATASET_ERA5, season)

                if corr_20crv3 and corr_era5:
                    Visualizer.plot_jet_correlation_maps(corr_20crv3, corr_era5, season)
                else:
                    logging.warning(f"Skipping jet correlation plot for {season}, data missing for one or both reanalyses.")
            else:
                logging.info(f"Plot '{jet_corr_plot_filename}' existiert bereits. Überspringe Berechnung und Erstellung.")

        logging.info("\n\n--- Checking for Correlation Timeseries & Bar Charts ---")
        for season in ['Winter', 'Summer']:
            season_lower = season.lower()
            
            corr_timeseries_filename = os.path.join(Config.PLOT_DIR, f'{season_lower}_correlations_comparison_detrended.png')
            if not os.path.exists(corr_timeseries_filename):
                logging.info(f"Plot '{corr_timeseries_filename}' nicht gefunden, wird erstellt...")
                Visualizer.plot_correlation_timeseries_comparison(datasets_reanalysis, jet_data_reanalysis, discharge_data_loaded, season)
            else:
                logging.info(f"Plot '{corr_timeseries_filename}' existiert bereits. Überspringe Erstellung.")

            corr_barchart_filename = os.path.join(Config.PLOT_DIR, f'correlation_matrix_comparison_{season_lower}_detrended_grouped.png')
            if not os.path.exists(corr_barchart_filename):
                logging.info(f"Plot '{corr_barchart_filename}' nicht gefunden. Berechne Daten und erstelle Plot...")
                # BERECHNUNG JETZT INNERHALB DER IF-BEDINGUNG
                correlation_data_for_bar_chart = StorylineAnalyzer.analyze_all_correlations_for_bar_chart(
                    datasets_reanalysis, jet_data_reanalysis, discharge_data_loaded, season
                )
                if not correlation_data_for_bar_chart.empty:
                    Visualizer.plot_correlation_bar_chart(correlation_data_for_bar_chart, season)
                else:
                    logging.warning(f"No correlation data generated for {season} bar chart, skipping plot.")
            else:
                logging.info(f"Plot '{corr_barchart_filename}' existiert bereits. Überspringe Berechnung und Erstellung.")

        logging.info("\n\n--- Checking for AMO-Jet Correlation Plot ---")
        window_size_amo = 15
        amo_plot_filename = os.path.join(Config.PLOT_DIR, f'amo_jet_correlations_comparison_rolling_{window_size_amo}yr.png')
        if not os.path.exists(amo_plot_filename):
            logging.info(f"Plot '{amo_plot_filename}' nicht gefunden. Berechne Daten und erstelle Plot...")
            if amo_data_loaded:
                # BERECHNUNG JETZT INNERHALB DER IF-BEDINGUNG
                amo_correlation_data = StorylineAnalyzer.analyze_amo_jet_correlations_for_plot(jet_data_reanalysis, amo_data_loaded, window_size_amo)
                if amo_correlation_data and any(amo_correlation_data.values()):
                    Visualizer.plot_amo_jet_correlation_comparison(amo_correlation_data, window_size_amo)
                else:
                    logging.warning("No AMO correlation data was generated, skipping the comparison plot.")
            else:
                logging.warning("AMO data could not be loaded, skipping AMO-Jet correlation analysis and plotting.")
        else:
            logging.info(f"Plot '{amo_plot_filename}' existiert bereits. Überspringe Berechnung und Erstellung.")

        # --- PART 2: CMIP6 AND STORYLINE ANALYSIS ---
        # Diese Haupt-Analyse wird immer ausgeführt, da ihre Ergebnisse (cmip6_results) 
        # für mehrere nachfolgende Plots und Auswertungen benötigt werden.
        logging.info("\n\n--- Analyzing CMIP6 Data and Storylines ---")
        try:
            cmip6_results = storyline_analyzer.analyze_cmip6_changes_at_gwl()
            
            if cmip6_results:
                gwl_plot_filename = os.path.join(Config.PLOT_DIR, "cmip6_jet_changes_vs_gwl.png")
                if not os.path.exists(gwl_plot_filename):
                    logging.info(f"Plot '{gwl_plot_filename}' nicht gefunden, wird erstellt...")
                    Visualizer.plot_jet_changes_vs_gwl(cmip6_results)
                else:
                    logging.info(f"Plot '{gwl_plot_filename}' existiert bereits. Überspringe Erstellung.")
            else:
                logging.warning("CMIP6 analysis did not produce results. Skipping subsequent plots.")
        except Exception as e:
            logging.error(f"A critical error occurred during the CMIP6/Storyline analysis phase: {e}")
            logging.error(traceback.format_exc())
            
        # --- PART 3: CLIMATE EVOLUTION TIMESERIES PLOT ---
        logging.info("\n\n--- Checking for Climate Indices Evolution Timeseries ---")
        evolution_plot_filename = os.path.join(Config.PLOT_DIR, "climate_indices_evolution.png")
        if not os.path.exists(evolution_plot_filename):
            logging.info(f"Plot '{evolution_plot_filename}' nicht gefunden. Berechne Daten und erstelle Plot...")
            if cmip6_results and datasets_reanalysis:
                # BERECHNUNG JETZT INNERHALB DER IF-BEDINGUNG
                cmip6_plot_data, reanalysis_plot_data = StorylineAnalyzer.analyze_timeseries_for_projection_plot(
                    cmip6_results, datasets_reanalysis, Config()
                )
                if cmip6_plot_data and reanalysis_plot_data:
                    Visualizer.plot_climate_projection_timeseries(
                        cmip6_plot_data, reanalysis_plot_data, Config(), filename=os.path.basename(evolution_plot_filename)
                    )
                else:
                    logging.warning("Skipping climate evolution plot: Data preparation returned empty or None.")
            else:
                logging.warning("Skipping climate evolution plot: Missing CMIP6 results or reanalysis data.")
        else:
            logging.info(f"Plot '{evolution_plot_filename}' existiert bereits. Überspringe Berechnung und Erstellung.")
            
        # --- PART 4: BETA-OBS CALCULATION, MODEL FIDELITY & CMIP6 SCATTER PLOTS ---
        logging.info("\n\n--- Calculating Betas, Checking Model Fidelity & Future Slopes, and Plotting Comparisons ---")
        
        if cmip6_results and datasets_reanalysis and jet_data_reanalysis:
            
            # 1. Calculate the beta_obs slopes from ERA5 reanalysis
            beta_obs_slopes = StorylineAnalyzer.calculate_reanalysis_betas(
                datasets_reanalysis, 
                jet_data_reanalysis, 
                dataset_key=Config.DATASET_ERA5
            )
            
            # 2. Check and create the comprehensive temporal slope comparison plot
            fidelity_future_plot_filename = os.path.join(Config.PLOT_DIR, "cmip6_fidelity_vs_future_temporal_slopes.png")
            if not os.path.exists(fidelity_future_plot_filename):
                if beta_obs_slopes:
                    logging.info(f"Plot '{fidelity_future_plot_filename}' not found, calculating data and plotting...")
                    
                    historical_period_for_fidelity = (1981, 2010)
                    cmip6_historical_slopes = StorylineAnalyzer.calculate_historical_slopes_comparison(
                        beta_obs_slopes=beta_obs_slopes,
                        cmip6_data_loaded=cmip6_results.get('cmip6_model_data_loaded', {}),
                        jet_data_reanalysis=jet_data_reanalysis,
                        historical_period=historical_period_for_fidelity
                    )

                    # NEU: Berechne die Verteilung der zukünftigen zeitlichen Steigungen
                    cmip6_future_temporal_slopes = StorylineAnalyzer.calculate_future_temporal_slopes(
                        cmip6_results=cmip6_results,
                        beta_keys=list(beta_obs_slopes.keys()),
                        gwls_to_analyze=Config.GLOBAL_WARMING_LEVELS
                    )

                    # Rufe die überarbeitete Plot-Funktion auf
                    Visualizer.plot_model_fidelity_comparison(
                        cmip6_historical_slopes=cmip6_historical_slopes,
                        cmip6_future_temporal_slopes=cmip6_future_temporal_slopes,
                        beta_obs_slopes=beta_obs_slopes,
                        historical_period=historical_period_for_fidelity,
                        gwls_to_plot=Config.GLOBAL_WARMING_LEVELS
                    )
                else:
                    logging.warning("Skipping temporal slope comparison plot: beta_obs slopes could not be calculated.")
            else:
                logging.info(f"Plot '{fidelity_future_plot_filename}' already exists. Skipping.")

            # (Optional) Die alten Scatter-Plots können weiterhin generiert werden, falls gewünscht
            if beta_obs_slopes:
                for gwl in Config.GLOBAL_WARMING_LEVELS:
                    scatter_plot_filename = os.path.join(Config.PLOT_DIR, f"cmip6_scatter_comparison_gwl_{gwl:.1f}_extended.png")
                    if not os.path.exists(scatter_plot_filename):
                        Visualizer.plot_cmip6_scatter_comparison(
                            cmip6_results=cmip6_results,
                            beta_obs_slopes=beta_obs_slopes,
                            gwl_to_plot=gwl
                        )
        else:
            logging.warning("Skipping all Fidelity/Scatter plots due to missing cmip6_results, reanalysis, or jet data.")
            
        # --- PART 5: JET INTER-RELATIONSHIP SCATTER PLOTS ---
        logging.info("\n\n--- Plotting CMIP6 Jet Inter-relationship Scatter Plots ---")
        if cmip6_results:
            # Define filename for the combined plot
            inter_rel_plot_filename = os.path.join(Config.PLOT_DIR, "cmip6_jet_inter_relationship_scatter_combined_gwl.png")
            if not os.path.exists(inter_rel_plot_filename):
                # Check if there is any data to plot
                if cmip6_results.get('mmm_changes'):
                     Visualizer.plot_jet_inter_relationship_scatter_combined_gwl(
                         cmip6_results=cmip6_results
                     )
                else:
                    logging.warning(f"Skipping combined jet inter-relationship scatter plot: No data found in cmip6_results.")
            else:
                logging.info(f"Plot '{inter_rel_plot_filename}' existiert bereits. Überspringe Erstellung.")
        else:
            logging.warning("Skipping CMIP6 jet inter-relationship scatter plots due to missing cmip6_results.")
            
        #CMIP6 MMM Analysis
        logging.info("\n\n--- Checking for CMIP6 U850 Change Maps ---")
        u850_change_plot_filename = os.path.join(Config.PLOT_DIR, "cmip6_mmm_u850_change_djf_jja.png")
        if not os.path.exists(u850_change_plot_filename):
            logging.info(f"Plot '{u850_change_plot_filename}' nicht gefunden. Berechne Daten und erstelle Plot...")
            if cmip6_results and 'cmip6_model_data_loaded' in cmip6_results:
                future_period = (2070, 2099)
                hist_period = (Config.CMIP6_ANOMALY_REF_START, Config.CMIP6_ANOMALY_REF_END)
                
                # Modelle aus den erfolgreich geladenen Daten ableiten
                preloaded_data = cmip6_results['cmip6_model_data_loaded']
                models_for_calc = sorted(list(set(key.split('_')[0] for key in preloaded_data.keys())))

                if models_for_calc:
                    u850_change_data = storyline_analyzer.calculate_cmip6_u850_change_fields(
                        models_to_run=models_for_calc,
                        future_scenario=Config.CMIP6_SCENARIOS[0],
                        future_period=future_period,
                        historical_period=hist_period,
                        preloaded_cmip6_data=preloaded_data
                    )
                    if u850_change_data:
                        Visualizer.plot_cmip6_u850_change_panel(
                            u850_change_data,
                            Config(),
                            future_period=future_period,
                            historical_period=hist_period,
                            filename=os.path.basename(u850_change_plot_filename)
                        )
                    else:
                        logging.warning("Berechnung der U850-Änderungsdaten fehlgeschlagen.")
                else:
                    logging.warning("Keine Modelle für die U850-Änderungsberechnung verfügbar.")
            else:
                logging.warning("Überspringe U850-Änderungsplot: CMIP6-Ergebnisse fehlen.")
        else:
            logging.info(f"Plot '{u850_change_plot_filename}' existiert bereits. Überspringe Erstellung.")

        # --- FINALE ZUSAMMENFASSUNGEN ---
        if cmip6_results:
            model_status = cmip6_results.get('model_run_status')
            if model_status:
                logging.info("\n\n--- CMIP6 Model Run Summary ---")
                failed_models = model_status.get('failed_models')
                if failed_models:
                    logging.info(f"\nModels that failed or were excluded ({len(failed_models)}):")
                    logging.info(f"  {', '.join(failed_models)}")
                else:
                    logging.info("\nAll attempted models were successfully included in the final analysis.")
                successful_models = model_status.get('successful_models_per_gwl')
                if successful_models:
                    logging.info("\nModels used for analysis per Global Warming Level:")
                    for gwl, models in sorted(successful_models.items()):
                        if gwl in Config.GLOBAL_WARMING_LEVELS:
                            logging.info(f"  - GWL {gwl}°C ({len(models)} models): {', '.join(models)}")
                logging.info("-----------------------------")

            storyline_classification = cmip6_results.get('storyline_classification')
            gwl_years_data = cmip6_results.get('gwl_threshold_years')

            if storyline_classification and gwl_years_data:
                logging.info("\n\n--- CMIP6 Model Storyline Classification ---")
                scenario = Config.CMIP6_SCENARIOS[0] if Config.CMIP6_SCENARIOS else ''

                for gwl, jet_indices in sorted(storyline_classification.items()):
                    if gwl not in Config.GLOBAL_WARMING_LEVELS:
                        continue
                    logging.info(f"\nGWL {gwl}°C:")
                    for jet_index, storylines in jet_indices.items():
                        tolerance = 0.25 if 'Speed' in jet_index else 0.35
                        logging.info(f"  Classification for '{jet_index}' (Tolerance: ±{tolerance}):")
                        for storyline_type, models in sorted(storylines.items()):
                            if models:
                                models_with_timings = []
                                for model_key in models:
                                    threshold_year = gwl_years_data.get(model_key, {}).get(gwl)
                                    
                                    if threshold_year:
                                        window = Config.GWL_YEARS_WINDOW
                                        start_year = threshold_year - window // 2
                                        end_year = threshold_year + (window - 1) // 2
                                        models_with_timings.append(f"{model_key} ({start_year}-{end_year})")
                                    else:
                                        models_with_timings.append(model_key)
                                
                                logging.info(f"    - {storyline_type}: {', '.join(models_with_timings)} ({len(models)})")
                            else:
                                logging.info(f"    - {storyline_type}: No models in range.")
                
                logging.info("\n------------------------------------------")
                logging.info("\nErläuterung zum Format:")
                logging.info("  - Jede Zeile listet die Modelle auf, deren projizierte Änderung einem bestimmten Storyline-Typ entspricht.")
                logging.info("  - Beispiel: 'Modell-XYZ_ssp585 (2040-2059)'")
                logging.info(f"    - 'Modell-XYZ_ssp585': Name des Klimamodells und das verwendete Szenario.")
                logging.info(f"    - '(2040-2059)': Der {Config.GWL_YEARS_WINDOW}-jährige Analysezeitraum. Dieser Zeitraum ist um das Jahr zentriert, in dem das jeweilige Modell zum ersten Mal das entsprechende Global Warming Level (GWL) erreicht.")
                logging.info("------------------------------------------")
        
        logging.info("\n\n=====================================================")
        logging.info("=== FULL ANALYSIS COMPLETED ===")
        logging.info(f"All plots saved to: {Config.PLOT_DIR}")
        logging.info(f"Log file saved to: {log_filename}")
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