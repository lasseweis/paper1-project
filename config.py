"""
Configuration module for the climate analysis project.

This file centralizes all paths, parameters, and settings
to make it easy to configure and run the analysis.
"""
import numpy as np
import os
import multiprocessing

class Config:
    """Configuration parameters for the analysis."""
    # Analysis time period - common period for comparison
    ANALYSIS_START_YEAR, ANALYSIS_END_YEAR = 1850, 2021

    # Base path for reanalysis and other data
    DATA_BASE_PATH = '/nas/home/vlw/Desktop/STREAM/STREAM/'

    # 20CRv3 data paths
    UA_FILE_20CRV3 = os.path.join(DATA_BASE_PATH, 'uwnd.mon.mean.nc')
    PR_FILE_20CRV3 = os.path.join(DATA_BASE_PATH, 'prate.mon.mean.nc')
    TAS_FILE_20CRV3 = os.path.join(DATA_BASE_PATH, 'air.2m.mon.mean.nc')

    # ERA5 data paths
    ERA5_BASE_PATH = '/data/reloclim/normal/ERA5_daily/'
    ERA5_UA_FILE = os.path.join(ERA5_BASE_PATH, 'ERA5_2p5_day_UA_19580101-20221231.nc')
    ERA5_PR_FILE = os.path.join(ERA5_BASE_PATH, 'ERA5_2p5cdo_day_PR_19500101-20221231.nc')
    ERA5_TAS_FILE = os.path.join(ERA5_BASE_PATH, 'ERA5_2p5cdo_day_TAS_19500101-20221231.nc')    
    
    #ERA5_UA_FILE = os.path.join(ERA5_BASE_PATH, 'ERA5_025_day_ua850_19500101-20211231.nc')
    #ERA5_PR_FILE = os.path.join(ERA5_BASE_PATH, 'ERA5_0p25_day_PR_19500101-20221231.nc')
    #ERA5_TAS_FILE = os.path.join(ERA5_BASE_PATH, 'ERA5_0p25_day_TAS_19500101-20221231.nc')

    # Index and discharge data paths
    AMO_INDEX_FILE = '/nas/home/vlw/Desktop/STREAM/CMIP6-datasets/amo_index.csv'
    DISCHARGE_FILE = os.path.join(DATA_BASE_PATH, 'danube_discharge_monthly_1893-2021.xlsx')
    
    # Danube CMIP6 Discharge Data Paths ---
    DISCHARGE_SSP245_FILE = os.path.join(DATA_BASE_PATH, 'CP16_4.5-Tabelle_1.csv')
    DISCHARGE_SSP585_FILE = os.path.join(DATA_BASE_PATH, 'CP16_8.5-Tabelle_1.csv')


    # Plot output directory
    PLOT_DIR = '/nas/home/vlw/Desktop/STREAM/paper1-plots'

    # --- Analysis Parameters ---
    WIND_LEVEL = 850  # 850 hPa pressure level

    # Analysis box coordinates (Central/Eastern Europe)
    BOX_LAT_MIN, BOX_LAT_MAX = 46.0, 51.0
    BOX_LON_MIN, BOX_LON_MAX = 8.0, 18.0

    # Jet index boxes
    JET_SPEED_BOX_LAT_MIN, JET_SPEED_BOX_LAT_MAX = 40.0, 60.0
    JET_SPEED_BOX_LON_MIN, JET_SPEED_BOX_LON_MAX = -20.0, 20.0
    JET_LAT_BOX_LAT_MIN, JET_LAT_BOX_LAT_MAX = 30.0, 70.0
    JET_LAT_BOX_LON_MIN, JET_LAT_BOX_LON_MAX = -20.0, 0.0

    # Base period for reanalysis anomaly calculation
    BASE_PERIOD_START_YEAR = 1981
    BASE_PERIOD_END_YEAR = 2010

    # --- CMIP6 Parameters ---
    CMIP6_DATA_BASE_PATH = '/data/reloclim/normal/CMIP6_STREAM/paper1-cmip-data'
    CMIP6_VAR_PATH = os.path.join(CMIP6_DATA_BASE_PATH, '{variable}_regrid')
    CMIP6_FILE_PATTERN = '{variable}_Amon_{model}_{experiment}_{member}_{grid}_*_regridded.nc'
    CMIP6_SCENARIOS = ['ssp585', 'ssp245']
    CMIP6_HISTORICAL_EXPERIMENT_NAME = 'historical'
    CMIP6_MEMBER_ID = '*'
    CMIP6_VARIABLES_TO_LOAD = ['ua', 'pr', 'tas', 'discharge']
    CMIP6_GLOBAL_TAS_VAR = 'tas'
    CMIP6_LEVEL = 850  # For U850

    # CMIP6 reference periods
    CMIP6_PRE_INDUSTRIAL_REF_START = 1850
    CMIP6_PRE_INDUSTRIAL_REF_END = 1900
    CMIP6_ANOMALY_REF_START = 1995  # Standard IPCC AR6 reference
    CMIP6_ANOMALY_REF_END = 2014

    # Global Warming Level (GWL) parameters
    GLOBAL_WARMING_LEVELS = [2.0, 3.0]
    GWL_FINE_STEPS_FOR_PLOT = np.arange(1.5, 3.51, 0.5).tolist() # Yields [1.5, 2.0, ..., 4.5]
    GWL_YEARS_WINDOW = 30
    GWL_TEMP_SMOOTHING_WINDOW = 20
    
    REQUIRED_MODEL_SCENARIOS = {
        "ACCESS-CM2": ["ssp245", "ssp585"],
        "ACCESS-ESM1-5": ["ssp245", "ssp585"],
        "BCC-CSM2-MR": ["ssp585"],
        "CESM2": ["ssp245", "ssp585"],
        "CESM2-WACCM": ["ssp245", "ssp585"],
        "CMCC-CM2-SR5": ["ssp245", "ssp585"],
        "CMCC-ESM2": ["ssp245", "ssp585"],
        "CNRM-CM6-1-HR": ["ssp585"],
        "CNRM-CM6-1": ["ssp585"],
        "CNRM-ESM2-1": ["ssp585"],
        "CanESM5": ["ssp245", "ssp585"],
        "E3SM-1-0": ["ssp585"],
        "EC-Earth3-CC": ["ssp245", "ssp585"],
        "EC-Earth3-Veg-LR": ["ssp245", "ssp585"],
        "EC-Earth3": ["ssp245", "ssp585"],
        "GFDL-ESM4": ["ssp245", "ssp585"],
        "HadGEM3-GC31-LL": ["ssp245", "ssp585"],
        "HadGEM3-GC31-MM": ["ssp585"],
        "IITM-ESM": ["ssp245", "ssp585"],
        "INM-CM4-8": ["ssp245", "ssp585"],
        "INM-CM5-0": ["ssp245", "ssp585"],
        "IPSL-CM6A-LR": ["ssp245", "ssp585"],
        "KACE-1-0-G": ["ssp245", "ssp585"],
        "KIOST-ESM": ["ssp245", "ssp585"],
        "MIROC-ES2L": ["ssp245", "ssp585"],
        "MIROC6": ["ssp245", "ssp585"],
        "MPI-ESM1-2-HR": ["ssp245", "ssp585"],
        "MPI-ESM1-2-LR": ["ssp245", "ssp585"],
        "MRI-ESM2-0": ["ssp245", "ssp585"],
        "NorESM2-MM": ["ssp245", "ssp585"],
        "UKESM1-0-LL": ["ssp245", "ssp585"],
    }

    # --- Storyline Definitions (PLACEHOLDERS!) ---
    # TODO: These values MUST be adapted after analyzing the CMIP6 jet changes!
    # Format: {'IndexName': {GWL: {'StorylineType': Value}}}
    STORYLINE_JET_CHANGES = {
        # Winter (DJF) Jet Speed [Unten Rechts]
        'DJF_JetSpeed': {
            2.0: {'Core Mean': 0.17, 'Core High': 0.6, 'Extreme Low': -0.7, 'Extreme High': 1.0},
            3.0: {'Core Mean': 0.26, 'Core High': 0.75, 'Extreme Low': -0.5, 'Extreme High': 1.2},
        },
        # Sommer (JJA) Jet Latitude [Oben Links]
        'JJA_JetLat': {
            2.0: {'Core Mean': 0.7, 'Core High': 1.5, 'Extreme Low': -1.0, 'Extreme High': 2.6},
            3.0: {'Core Mean': 1.1, 'Core High': 2.0, 'Extreme Low': -0.8, 'Extreme High': 3.0},
        },
        # Winter (DJF) Jet Latitude [Unten Links]
        'DJF_JetLat': {
            2.0: {'Core Mean': 0.0, 'Core High': 1.0, 'Extreme Low': -1.7, 'Extreme High': 2.0},
            3.0: {'Core Mean': 0.0, 'Core High': 1.0, 'Extreme Low': -2.0, 'Extreme High': 2.5},
        },
        # Sommer (JJA) Jet Speed [Oben Rechts]
        'JJA_JetSpeed': {
            2.0: {'Core Mean': -0.12, 'Core High': 0.1, 'Extreme Low': -0.5, 'Extreme High': 0.25},
            3.0: {'Core Mean': -0.13, 'Core High': 0.12, 'Extreme Low': -0.55, 'Extreme High': 0.3},
        }
    }

    # --- Plotting and Performance Parameters ---
    PLOT_MAP_EXTENT = [-80, 40, 0, 90]
    PLOT_COLORMAP = 'RdBu_r'
    PLOT_COLORBAR_LEVELS = 21
    DATASET_20CRV3 = "20CRv3"
    DATASET_ERA5 = "ERA5"
    N_PROCESSES = max(1, multiprocessing.cpu_count() - 4)

    # --- NEW: 2D Storyline Definitions (for Scatter Plots) ---
    # Radius for the new quadrant method (in units of standard deviation)
    # A value of 0.5 means models inside the inner oval are classified into that storyline zone.
    STORYLINE_INNER_RADIUS = 0.5 

    # Format: {'Season': {GWL: [List of Storyline Names]}}
    # The names are used to initialize the categories. The classification logic
    # dynamically assigns models to them.
    # ADDED THE FOUR "Only" aAial storylines to the definitions.
    STORYLINE_JET_CHANGES_2D = {
        'DJF': {
            2.0: [
                'Core Mean', 'Neutral',
                'Fast Jet & Northward Shift',
                'Slow Jet & Northward Shift',
                'Slow Jet & Southward Shift',
                'Fast Jet & Southward Shift',
                # --- AXIAL STORYLINES ---
                'Northward Shift Only',
                'Southward Shift Only',
                'Fast Jet Only',
                'Slow Jet Only',
            ],
            3.0: [
                'Core Mean', 'Neutral',
                'Fast Jet & Northward Shift',
                'Slow Jet & Northward Shift',
                'Slow Jet & Southward Shift',
                'Fast Jet & Southward Shift',
                # --- AXIAL STORYLINES ---
                'Northward Shift Only',
                'Southward Shift Only',
                'Fast Jet Only',
                'Slow Jet Only',
            ]
        },
        'JJA': {
            2.0: [
                'Core Mean', 'Neutral',
                'Fast Jet & Northward Shift',
                'Slow Jet & Northward Shift',
                'Slow Jet & Southward Shift',
                'Fast Jet & Southward Shift',
                # --- AXIAL STORYLINES ---
                'Northward Shift Only',
                'Southward Shift Only',
                'Fast Jet Only',
                'Slow Jet Only',
            ],
            3.0: [
                'Core Mean', 'Neutral',
                'Fast Jet & Northward Shift',
                'Slow Jet & Northward Shift',
                'Slow Jet & Southward Shift',
                'Fast Jet & Southward Shift',
                # --- AXIAL STORYLINES ---
                'Northward Shift Only',
                'Southward Shift Only',
                'Fast Jet Only',
                'Slow Jet Only',
            ]
        }
    }