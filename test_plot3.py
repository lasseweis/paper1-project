import os
import sys
import logging
import pandas as pd
import numpy as np
from config import Config
from main import ClimateAnalysis
from data_processing import DataProcessor

def run_isolated_plot():
    from visualization import Visualizer
    logging.info("Starting run_isolated_plot")
    
    config = Config()
    qobs_historical_da = ClimateAnalysis.load_historical_qobs_from_csv(config)
    discharge_data_loaded = ClimateAnalysis.process_historical_discharge_data(qobs_historical_da)

    models_to_include = list(config.REQUIRED_MODEL_SCENARIOS.keys())
    cmip6_discharge = DataProcessor.process_discharge_data(
        config.DISCHARGE_SSP245_FILE, 
        config.DISCHARGE_SSP585_FILE, 
        models_to_include
    )
    
    metric_ts = {}
    for scenario in ['ssp585', 'ssp245']:
        for model in models_to_include:
            key = f"{model}_{scenario}"
            model_ds = cmip6_discharge.get(key)
            if model_ds is None: continue
            
            da = model_ds
            
            # Use real monthly DA for tests
            da_year = da.groupby('time.year').min('time')
            
            metric_ts[key] = {
                '30Q_low_full_year': da_year,
                '30Q_low_summer': da_year,
                '30Q_low_winter': da_year
            }
            
    classes = {}
    for scenario in ["ssp585", "ssp245"]:
        n_select = 14 if scenario == "ssp585" else config.COMPOSITE_N_MODELS
        
        # Test script mocks this per scenario to show realistic n=14
        classes_scenario = {}
        for gwl in config.GLOBAL_WARMING_LEVELS:
            classes_scenario[gwl] = {}
            for prefix in ['JJA', 'DJF']:
                classes_scenario[gwl][f"{prefix}_Extreme Models"] = [f"{m}_{scenario}" for m in list(models_to_include)[0:n_select]]
                classes_scenario[gwl][f"{prefix}_Non-Extreme Models"] = [f"{m}_{scenario}" for m in list(models_to_include)[-n_select:]]

        cmip6_results = {'model_metric_timeseries': metric_ts, 'storyline_classification_2d': classes_scenario}
        
        logging.info(f"Plotting for {scenario}...")
        try:
            Visualizer.plot_discharge_events_timeseries(cmip6_results, discharge_data_loaded, config, scenario)
            Visualizer.plot_discharge_events_extreme_timeseries(cmip6_results, discharge_data_loaded, config, scenario)
            logging.info(f"SUCCESS: Plot generated for {scenario}")
        except Exception as e:
            logging.error(f"FAILED for {scenario}: {e}")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    run_isolated_plot()
