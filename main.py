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
from advanced_analyzer import AdvancedAnalyzer
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
                '20CRv3_pr_seasonal': pr_seasonal, '20CRv3_tas_seasonal': tas_seasonal,
                '20CRv3_ua850_seasonal': ua850_seasonal, '20CRv3_pr_box_mean': pr_box_mean,
                '20CRv3_tas_box_mean': tas_box_mean, '20CRv3_tas_monthly': tas_monthly
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
                'ERA5_pr_seasonal': pr_seasonal, 'ERA5_tas_seasonal': tas_seasonal,
                'ERA5_ua850_seasonal': ua850_seasonal, 'ERA5_pr_box_mean': pr_box_mean,
                'ERA5_tas_box_mean': tas_box_mean, 'ERA5_tas_monthly': tas_monthly
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
        # This is a simplified placeholder. The full logic from your original script would be here.
        pass

    @staticmethod
    def run_full_analysis():
        """Main static method to execute the entire analysis workflow."""
        logging.info("=====================================================")
        logging.info("=== STARTING FULL CLIMATE ANALYSIS WORKFLOW ===")
        logging.info("=====================================================\n")

        Visualizer.ensure_plot_dir_exists()
        cmip6_results = None

        # --- PART 1: REANALYSIS DATA PROCESSING AND ANALYSIS ---
        logging.info("\n--- Processing Reanalysis Datasets ---")
        
        datasets_reanalysis = {
            **ClimateAnalysis.process_20crv3_data(),
            **ClimateAnalysis.process_era5_data()
        }
        
        if not datasets_reanalysis:
            logging.critical("Failed to process reanalysis datasets. Aborting.")
            return None

        logging.info("\n--- Calculating and Plotting Reanalysis Regression Maps ---")
        regression_results = {}
        regression_period = (1981, 2010) 

        for dset_key in [Config.DATASET_20CRV3, Config.DATASET_ERA5]:
            logging.info(f"\n--> Processing regression analysis for {dset_key}")
            
            results = AdvancedAnalyzer.calculate_regression_maps(
                datasets=datasets_reanalysis,
                dataset_key=dset_key,
                regression_period=regression_period
            )
            
            if results:
                regression_results[dset_key] = results
                logging.info(f"--> Plotting regression maps for {dset_key}")
                Visualizer.plot_regression_analysis(results, dset_key)
            else:
                logging.warning(f"Could not calculate or plot regression for {dset_key}. Results were empty.")
        
        logging.info("\n\n--- Analyzing Jet Impacts and Correlations for Reanalysis Data ---")
        
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

        # --- Plot der Jet-Index-Zeitreihen ---
        logging.info("\n\n--- Plotting Reanalysis Jet Index Comparison Timeseries ---")
        if jet_data_reanalysis:
            Visualizer.plot_jet_indices_comparison(jet_data_reanalysis)
        else:
            logging.warning("Skipping jet index comparison plot, no data was generated.")

        # --- Berechnung der Jet-Impact-Karten ---
        logging.info("\n\n--- Calculating Reanalysis Jet Impact Maps ---")
        jet_impact_all_results = {}
        for dset_key in [Config.DATASET_20CRV3, Config.DATASET_ERA5]:
            logging.info(f"\n--> Calculating jet impact maps for {dset_key}")
            
            dset_results = {}
            for season in ['Winter', 'Summer']:
                impact_maps = AdvancedAnalyzer.calculate_jet_impact_maps(
                    datasets=datasets_reanalysis,
                    jet_data=jet_data_reanalysis,
                    dataset_key=dset_key,
                    season=season
                )
                if impact_maps:
                    dset_results.update(impact_maps)
                    
            if dset_results:
                jet_impact_all_results[dset_key] = dset_results
            else:
                logging.warning(f"Could not calculate jet impact maps for {dset_key}.")

        # --- Plot der Jet-Impact-Vergleichskarten ---
        logging.info("\n\n--- Plotting Reanalysis Jet Impact Comparison Maps ---")
        for season in ['Winter', 'Summer']:
            data_20crv3 = jet_impact_all_results.get(Config.DATASET_20CRV3, {}).get(season)
            data_era5 = jet_impact_all_results.get(Config.DATASET_ERA5, {}).get(season)
            
            if data_20crv3 and data_era5:
                logging.info(f"--> Plotting combined jet impact maps for {season}")
                Visualizer.plot_jet_impact_comparison_maps(data_20crv3, data_era5, season)
            else:
                logging.warning(f"Skipping combined jet impact plot for {season}, data missing for one or both reanalyses.")

        # --- Berechnung und Plot der Jet-Korrelationskarten (räumlich) ---
        logging.info("\n\n--- Calculating and Plotting Reanalysis Jet Correlation Maps ---")
        jet_corr_results = {}
        for dset_key in [Config.DATASET_20CRV3, Config.DATASET_ERA5]:
            jet_corr_results[dset_key] = {}
            for season in ['Winter', 'Summer']:
                logging.info(f"--> Calculating jet correlation maps for {dset_key} ({season})")
                season_results = AdvancedAnalyzer.calculate_jet_correlation_maps(
                    datasets=datasets_reanalysis,
                    jet_data=jet_data_reanalysis,
                    dataset_key=dset_key,
                    season=season
                )
                if season_results:
                    jet_corr_results[dset_key][season] = season_results
        
        for season in ['Winter', 'Summer']:
            data_20crv3 = jet_corr_results.get(Config.DATASET_20CRV3, {}).get(season)
            data_era5 = jet_corr_results.get(Config.DATASET_ERA5, {}).get(season)
            if data_20crv3 and data_era5:
                logging.info(f"--> Plotting jet correlation maps for {season}")
                Visualizer.plot_jet_correlation_maps(data_20crv3, data_era5, season)
            else:
                logging.warning(f"Skipping jet correlation plot for {season}, data missing for one or both reanalyses.")

        # --- Aufruf für die Zeitreihen-Korrelationsplots, die die angehängten Bilder erzeugen ---
        logging.info("\n\n--- Plotting Reanalysis Correlation Timeseries Comparison ---")
        try:
            # Lade die Abflussdaten einmal, da sie für beide Saisons benötigt werden.
            # Die Methode muss in der ClimateAnalysis-Klasse existieren.
            discharge_data_loaded = ClimateAnalysis.process_discharge_data(Config.DISCHARGE_FILE)
            for season in ['Winter', 'Summer']:
                Visualizer.plot_correlation_timeseries_comparison(
                    datasets_reanalysis=datasets_reanalysis,
                    jet_data_reanalysis=jet_data_reanalysis,
                    discharge_data=discharge_data_loaded,
                    season=season
                )
        except Exception as e:
            logging.error(f"Fehler beim Erstellen der Korrelations-Zeitreihenplots: {e}")
            logging.error(traceback.format_exc())


        # --- PART 2: CMIP6 AND STORYLINE ANALYSIS ---
        logging.info("\n\n--- Analyzing CMIP6 Data and Storylines ---")
        try:
            storyline_analyzer = StorylineAnalyzer(config=Config)
            cmip6_results = storyline_analyzer.analyze_cmip6_changes_at_gwl()
            
            if cmip6_results:
                logging.info("--> Plotting CMIP6 jet changes vs. Global Warming Level...")
                Visualizer.plot_jet_changes_vs_gwl(cmip6_results)
            else:
                logging.warning("CMIP6 analysis did not produce results. Skipping subsequent plots.")
                
        except Exception as e:
            logging.error(f"A critical error occurred during the CMIP6/Storyline analysis phase: {e}")
            logging.error(traceback.format_exc())

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
                
                # *** HINZUGEFÜGTE ERLÄUTERUNG ***
                logging.info("\n------------------------------------------")
                logging.info("\nErläuterung zum Format:")
                logging.info("  - Jede Zeile listet die Modelle auf, deren projizierte Änderung einem bestimmten Storyline-Typ entspricht.")
                logging.info("  - Beispiel: 'Modell-XYZ_ssp585 (2040-2059)'")
                logging.info("    - 'Modell-XYZ_ssp585': Name des Klimamodells und das verwendete Szenario.")
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