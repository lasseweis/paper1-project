import os
import sys
import logging
import xarray as xr
from config import Config
from storyline import StorylineAnalyzer
from main import ClimateAnalysis
from data_processing import DataProcessor

def run_limited():
    from visualization import Visualizer
    logging.info("Starting run_limited")
    
    config = Config()
    qobs_historical_da = ClimateAnalysis.load_historical_qobs_from_csv(config)
    discharge_data_loaded = ClimateAnalysis.process_historical_discharge_data(qobs_historical_da)

    storyline_analyzer = StorylineAnalyzer(config)
    for scenario in ["ssp585", "ssp245"]:
        logging.info(f"Running scenario {scenario}")
        cmip6_results = storyline_analyzer.analyze_cmip6_changes_at_gwl(scenario_to_process=scenario)
        
        storyline_classification_2d = cmip6_results.get('storyline_classification_2d')
        if storyline_classification_2d:
            for gwl in config.GLOBAL_WARMING_LEVELS:
                if gwl not in storyline_classification_2d:
                    storyline_classification_2d[gwl] = {}
                for season_name, season_prefix in [('Winter', 'DJF'), ('Summer', 'JJA')]:
                    event_key = config.COMPOSITE_EVENT_KEY
                    res = storyline_analyzer.get_composite_extreme_models(cmip6_results, gwl, event_key, season_name)
                    if res[0] is not None:
                        ext_models, non_ext_models, _ = res
                        storyline_classification_2d[gwl][f"{season_prefix}_Extreme Models"] = ext_models
                        storyline_classification_2d[gwl][f"{season_prefix}_Non-Extreme Models"] = non_ext_models

        logging.info(f"ABOUT TO CALL PLOT FOR {scenario}")
        Visualizer.plot_discharge_events_timeseries(cmip6_results, discharge_data_loaded, config, scenario)
        logging.info(f"DONE CALLING PLOT FOR {scenario}")

if __name__ == '__main__':
    run_limited()
