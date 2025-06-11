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
        """Loads and processes the AMO index from a CSV file."""
        try:
            amo_df = pd.read_csv(file_path, header=0)
            amo_long = amo_df.melt(id_vars="Year", var_name="Month", value_name="AMO")
            month_map = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6, "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}
            amo_long["Month"] = amo_long["Month"].map(month_map)
            
            # Winter
            amo_winter_raw = amo_long[amo_long["Month"].isin([12, 1, 2])].copy()
            amo_winter_raw.loc[amo_winter_raw["Month"] == 12, "season_year"] = amo_winter_raw["Year"] + 1
            amo_winter_raw.loc[amo_winter_raw["Month"] != 12, "season_year"] = amo_winter_raw["Year"]
            amo_winter_mean = amo_winter_raw.groupby("season_year")["AMO"].mean()
            amo_winter_da = xr.DataArray(amo_winter_mean, name='AMO_winter')
            
            # Summer
            amo_summer_raw = amo_long[amo_long["Month"].isin([6, 7, 8])].copy()
            amo_summer_mean = amo_summer_raw.groupby("Year")["AMO"].mean()
            amo_summer_da = xr.DataArray(amo_summer_mean.rename_axis('season_year'), name='AMO_summer')
            
            return {
                'amo_winter_detrended': DataProcessor.detrend_data(amo_winter_da),
                'amo_summer_detrended': DataProcessor.detrend_data(amo_summer_da)
            }
        except Exception:
            return {}

    @staticmethod
    def run_full_analysis():
        """Main static method to execute the entire analysis workflow."""
        logging.info("=====================================================")
        logging.info("=== STARTING FULL CLIMATE ANALYSIS WORKFLOW ===")
        logging.info("=====================================================\n")

        Visualizer.ensure_plot_dir_exists()

        # --- PART 1: REANALYSIS DATA PROCESSING AND ANALYSIS ---
        logging.info("\n--- Processing Reanalysis Datasets ---")
        datasets_reanalysis = {
            **ClimateAnalysis.process_20crv3_data(),
            **ClimateAnalysis.process_era5_data()
        }
        if not datasets_reanalysis.get('20CRV3_pr_seasonal') or not datasets_reanalysis.get('ERA5_pr_seasonal'):
            logging.critical("Failed to process one or both reanalysis datasets. Aborting.")
            return None
        
        # Load external indices
        amo_data = ClimateAnalysis.load_amo_index(Config.AMO_INDEX_FILE)

        # --- Analyze Jet Indices, Correlations, etc. for Reanalysis ---
        jet_data_reanalysis = {}
        all_correlations_reanalysis = {}
        jet_impact_maps_reanalysis = {}
        beta_obs_slopes = {}

        for dataset_key in [Config.DATASET_20CRV3, Config.DATASET_ERA5]:
            logging.info(f"\n--- Analyzing Indices and Correlations for: {dataset_key} ---")
            
            jet_data_ds = AdvancedAnalyzer.analyze_jet_indices(datasets_reanalysis, dataset_key)
            if jet_data_ds: jet_data_reanalysis.update(jet_data_ds)
            
            correlations_ds = AdvancedAnalyzer.analyze_correlations(datasets_reanalysis, {}, jet_data_reanalysis, dataset_key)
            all_correlations_reanalysis[dataset_key] = correlations_ds

            if dataset_key == Config.DATASET_ERA5 and correlations_ds:
                logging.info("Extracting Beta_obs slopes from ERA5 correlations...")
                required_beta_keys = [
                    ('tas', 'winter', 'speed', 'DJF_JetSpeed_vs_tas'), ('tas', 'winter', 'lat', 'DJF_JetLat_vs_tas'),
                    ('pr', 'winter', 'speed', 'DJF_JetSpeed_vs_pr'), ('pr', 'winter', 'lat', 'DJF_JetLat_vs_pr'),
                    ('tas', 'summer', 'speed', 'JJA_JetSpeed_vs_tas'), ('tas', 'summer', 'lat', 'JJA_JetLat_vs_tas'),
                    ('pr', 'summer', 'speed', 'JJA_JetSpeed_vs_pr'), ('pr', 'summer', 'lat', 'JJA_JetLat_vs_pr')
                ]
                for var_type, season, jet_dim, beta_key_name in required_beta_keys:
                    slope = correlations_ds.get(var_type, {}).get(season, {}).get(jet_dim, {}).get('slope')
                    beta_obs_slopes[beta_key_name] = slope

            jet_impact_maps_reanalysis[dataset_key] = {
                'Winter': AdvancedAnalyzer.calculate_jet_impact_maps(datasets_reanalysis, jet_data_reanalysis, dataset_key, 'Winter'),
                'Summer': AdvancedAnalyzer.calculate_jet_impact_maps(datasets_reanalysis, jet_data_reanalysis, dataset_key, 'Summer')
            }
        
        # --- U850 vs Box-Index Regressionen für Reanalyse ---
        logging.info("\n--- Calculating Reanalysis Regression Maps ---")
        regression_period = (1981, 2010) 
        regression_results_reanalysis = {}
        for dset_key in [Config.DATASET_20CRV3, Config.DATASET_ERA5]:
            results = AdvancedAnalyzer.calculate_regression_maps(datasets_reanalysis, dset_key, regression_period)
            regression_results_reanalysis[dset_key] = results

        # --- VISUALIZATION (Reanalysis) ---
        logging.info("\n--- PLOTTING REANALYSIS RESULTS ---")
        Visualizer.plot_jet_indices_timeseries(jet_data_reanalysis)
        Visualizer.plot_seasonal_correlation_matrix(all_correlations_reanalysis.get('20CRV3',{}), all_correlations_reanalysis.get('ERA5',{}), 'Winter')
        Visualizer.plot_seasonal_correlation_matrix(all_correlations_reanalysis.get('20CRV3',{}), all_correlations_reanalysis.get('ERA5',{}), 'Summer')
        Visualizer.plot_jet_impact_maps(jet_impact_maps_reanalysis.get('20CRV3',{}).get('Winter',{}), jet_impact_maps_reanalysis.get('ERA5',{}).get('Winter',{}), 'Winter')
        Visualizer.plot_jet_impact_maps(jet_impact_maps_reanalysis.get('20CRV3',{}).get('Summer',{}), jet_impact_maps_reanalysis.get('ERA5',{}).get('Summer',{}), 'Summer')
        
        if regression_results_reanalysis.get(Config.DATASET_20CRV3):
             Visualizer.plot_regression_analysis(regression_results_reanalysis[Config.DATASET_20CRV3], Config.DATASET_20CRV3)
        if regression_results_reanalysis.get(Config.DATASET_ERA5):
             Visualizer.plot_regression_analysis(regression_results_reanalysis[Config.DATASET_ERA5], Config.DATASET_ERA5)


        # --- PART 2: CMIP6 AND STORYLINE ANALYSIS ---
        logging.info("\n\n--- Analyzing CMIP6 Data and Storylines ---")
        storyline_analyzer = StorylineAnalyzer(config=Config)
        cmip6_results = storyline_analyzer.analyze_cmip6_changes_at_gwl()

        if cmip6_results:
            logging.info("\n--- PLOTTING CMIP6 RESULTS ---")
            Visualizer.plot_jet_changes_vs_gwl(cmip6_results)
            logging.warning("\n>>> ACTION REQUIRED: Examine 'cmip6_jet_changes_vs_gwl.png' and update placeholders in Config.STORYLINE_JET_CHANGES if necessary! <<<")

            cmip6_hist_slopes = AdvancedAnalyzer.calculate_historical_slopes_comparison(beta_obs_slopes, cmip6_results['cmip6_model_data_loaded'], jet_data_reanalysis, regression_period)
            Visualizer.plot_beta_obs_comparison(beta_obs_slopes, cmip6_hist_slopes)

            storyline_impacts = storyline_analyzer.calculate_storyline_impacts(cmip6_results, beta_obs_slopes)
            if storyline_impacts:
                Visualizer.plot_storyline_impacts(storyline_impacts)
        else:
            logging.warning("CMIP6 analysis did not produce results. Skipping subsequent plots.")

        logging.info("\n\n=====================================================")
        logging.info("=== FULL ANALYSIS COMPLETED ===")
        logging.info(f"All plots saved to: {Config.PLOT_DIR}")
        logging.info(f"Log file saved to: {log_filename}")
        logging.info("=====================================================\n")
        
        return {} # Return empty dict for now

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