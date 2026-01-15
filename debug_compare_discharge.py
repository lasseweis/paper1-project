
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import os
import sys
import logging

# Configure logging to stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import project modules
from config import Config
import numpy as np

# Copied from main.py to avoid ImportError (statsmodels)
def load_historical_qobs_from_csv(config):
    logging.info("Loading DAILY historical QOBS discharge data from CSV...")
    filepath = config.DISCHARGE_SSP245_FILE 
    if not os.path.exists(filepath):
        logging.error(f"Cannot load QOBS data: File not found at {filepath}")
        return None
    try:
        df = pd.read_csv(filepath, sep=';', decimal=',', na_values=['-0,01'])
        
        # Rename first column to date (as in main.py)
        df = df.rename(columns={df.columns[0]: 'date'})
        
        # Parse date with correct format
        df['time'] = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
        df = df.set_index('time').sort_index()
        
        if 'QOBS' not in df.columns:
             logging.error("QOBS column not found")
             return None

        # Filter for relevant period (1960 onwards) as in main.py, or just take all
        # main.py filters 1960-2021. I'll do the same to be consistent.
        # But for comparison with 1893 dataset, I might want more if available?
        # The file head shows it starts 1960. So filtering 1960-2021 is fine.
        qobs_series_daily = df['QOBS'].dropna() #.loc['1960-01-01':'2021-12-31']
        
        # Create DataArray
        da = xr.DataArray(
            qobs_series_daily.values,
            coords={'time': qobs_series_daily.index},
            dims='time', 
            name='discharge',
            attrs={'units': 'm3/s'}
        )
        return da
    except Exception as e:
        logging.error(f"Error loading daily csv: {e}")
        return None

def load_historical_qobs_long_term(config):
    logging.info("Loading LONG-TERM MONTHLY historical QOBS discharge data from Excel...")
    filepath = config.DISCHARGE_FILE
    if not os.path.exists(filepath):
        logging.error(f"Cannot load long-term QOBS data: File not found at {filepath}")
        return None
    try:
        df = pd.read_excel(filepath, engine='openpyxl') 
        # Extract columns: Month (col 1), Year (col 2), Discharge (col 7)
        df_subset = df.iloc[:, [1, 2, 7]].copy()
        df_subset.columns = ['month', 'year', 'discharge']
        df_subset['time'] = pd.to_datetime(dict(year=df_subset['year'], month=df_subset['month'], day=15))
        df_subset = df_subset.set_index('time').sort_index()
        da = xr.DataArray(
            df_subset['discharge'].values,
            coords={'time': df_subset.index},
            dims='time', 
            name='discharge'
        )
        return da
    except Exception as e:
        logging.error(f"Error loading monthly excel: {e}")
        return None

def compare_datasets():
    logging.info("Starting comparison of Daily vs Monthly discharge datasets...")

    # 1. Load Daily Data
    daily_da = load_historical_qobs_from_csv(Config())
    if daily_da is None:
        logging.error("Failed to load daily data.")
        return

    # 2. Load Monthly Long-Term Data
    monthly_da = load_historical_qobs_long_term(Config())
    if monthly_da is None:
        logging.error("Failed to load monthly data.")
        return

    # 3. Resample Daily to Monthly Mean
    logging.info("Resampling daily data to monthly means...")
    # Using 'MS' for Month Start to align, or match the time index of monthly_da
    # monthly_da uses day=15. daily_da has random days. 
    # Resample to monthly freq.
    daily_resampled = daily_da.resample(time='MS').mean()
    
    # Adjust time index of resampled data to match monthly_da (if needed for direct comparison)
    # monthly_da was set to day=15. Let's align both to Year-Month for intersection
    
    # Convert both to pandas series for easier alignment/plotting features
    s_daily_resampled = daily_resampled.to_series()
    s_monthly = monthly_da.to_series()

    # Align indexes to simple periods (YYYY-MM) to ensure matching
    s_daily_resampled.index = s_daily_resampled.index.to_period('M')
    s_monthly.index = s_monthly.index.to_period('M')

    # intersect
    common_index = s_daily_resampled.index.intersection(s_monthly.index)
    
    # Filter for the requested period (1960-2000) for the main comparison
    # But let's verify if they cover it.
    
    # Create a DataFrame for the common period
    df_compare = pd.DataFrame({
        'Daily_Resampled': s_daily_resampled.loc[common_index],
        'Monthly_Original': s_monthly.loc[common_index]
    }).dropna()

    # Filter for 1990-2000
    start_date = pd.Period('1990-01', freq='M')
    end_date = pd.Period('2000-12', freq='M')
    
    df_subset = df_compare.loc[start_date:end_date]
    
    if df_subset.empty:
        logging.warning("No overlapping data found in 1990-2000 range. Plotting full overlapping range instead.")
        df_subset = df_compare

    logging.info(f"Comparison period: {df_subset.index.min()} to {df_subset.index.max()} (N={len(df_subset)} months)")

    # 4. Plotting
    output_path = os.path.join(Config.PLOT_DIR, 'debug_discharge_comparison_1990-2000.png')
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 15), constrained_layout=True)
    
    # Plot 1: Time Series
    ax0 = axes[0]
    # convert period index back to timestamp for plotting (use start time)
    plot_index = df_subset.index.to_timestamp()
    
    ax0.plot(plot_index, df_subset['Daily_Resampled'], label='Daily Resampled (MS)', linewidth=1.5, alpha=0.8)
    ax0.plot(plot_index, df_subset['Monthly_Original'], label='Monthly Original (Long-term)', linewidth=1.5, alpha=0.8, linestyle='--')
    ax0.set_title(f'Discharge Time Series Comparison ({df_subset.index.min()} - {df_subset.index.max()})')
    ax0.set_ylabel('Discharge [m³/s]')
    ax0.legend()
    ax0.grid(True, alpha=0.3)

    # Plot 2: Difference
    ax1 = axes[1]
    diff = df_subset['Daily_Resampled'] - df_subset['Monthly_Original']
    ax1.plot(plot_index, diff, color='k', linewidth=1)
    ax1.axhline(0, color='r', linestyle=':', alpha=0.5)
    ax1.set_title('Difference (Daily Resampled - Monthly Original)')
    ax1.set_ylabel('Difference [m³/s]')
    ax1.grid(True, alpha=0.3)
    
    # Calculate stats
    mae = diff.abs().mean()
    rmse = (diff**2).mean()**0.5
    bias = diff.mean()
    ax1.text(0.02, 0.95, f'MAE: {mae:.2f}\nRMSE: {rmse:.2f}\nBias: {bias:.2f}', 
             transform=ax1.transAxes, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # Plot 3: Scatter
    ax2 = axes[2]
    ax2.scatter(df_subset['Monthly_Original'], df_subset['Daily_Resampled'], alpha=0.5, s=10)
    
    # 1:1 line
    min_val = min(df_subset.min().min(), 0)
    max_val = df_subset.max().max()
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1 Line')
    
    ax2.set_xlabel('Monthly Original [m³/s]')
    ax2.set_ylabel('Daily Resampled [m³/s]')
    ax2.set_title('Scatter Plot')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')

    plt.savefig(output_path, dpi=150)
    logging.info(f"Comparison plot saved to: {output_path}")

if __name__ == "__main__":
    compare_datasets()
