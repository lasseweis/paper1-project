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

        # --- TEIL 1: DATENVERARBEITUNG UND ANALYSE DER REANALYSE-DATEN ---
        logging.info("\n--- Processing Reanalysis Datasets ---")
        
        datasets_reanalysis = {
            **ClimateAnalysis.process_20crv3_data(),
            **ClimateAnalysis.process_era5_data()
        }
        
        if not datasets_reanalysis:
            logging.critical("Failed to process reanalysis datasets. Aborting.")
            return None

        # --- HINZUGEFÜGTER ANALYSE- UND PLOTTING-TEIL ---
        logging.info("\n--- Calculating and Plotting Reanalysis Regression Maps ---")
        regression_results = {}
        # Definieren der Periode für die Regressionsanalyse
        regression_period = (1981, 2010) 

        for dset_key in [Config.DATASET_20CRV3, Config.DATASET_ERA5]:
            logging.info(f"\n--> Processing regression analysis for {dset_key}")
            
            # Aufruf der Regressionsberechnung aus dem AdvancedAnalyzer
            results = AdvancedAnalyzer.calculate_regression_maps(
                datasets=datasets_reanalysis,
                dataset_key=dset_key,
                regression_period=regression_period
            )
            
            if results:
                regression_results[dset_key] = results
                # Aufruf der Plot-Funktion aus dem Visualizer
                logging.info(f"--> Plotting regression maps for {dset_key}")
                Visualizer.plot_regression_analysis(results, dset_key)
            else:
                logging.warning(f"Could not calculate or plot regression for {dset_key}. Results were empty.")
        # --- ENDE DES HINZUGEFÜGTEN TEILS ---

        # Hier könnten zukünftige Analyseschritte folgen (z.B. für CMIP6)

        logging.info("\n\n=====================================================")
        logging.info("=== FULL ANALYSIS COMPLETED ===")
        logging.info(f"All plots saved to: {Config.PLOT_DIR}")
        logging.info(f"Log file saved to: {log_filename}")
        logging.info("=====================================================\n")
        
        return regression_results # Gibt die Ergebnisse zurück

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