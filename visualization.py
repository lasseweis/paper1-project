"""
Visualization module for climate analysis results.

This module contains the Visualizer class, which bundles all plotting
functions. Each static method is responsible for creating a specific
figure, such as time series plots, regression maps, or correlation
matrices. It separates the analysis logic from the presentation.
"""
import os
import sys
import glob
import logging
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker
import matplotlib.gridspec
from matplotlib import gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import shapely.ops
import shapely.geometry
from scipy.stats import chi2, linregress
import json
import traceback
from typing import cast, Any

# Import local modules
from storyline import StorylineAnalyzer 
from config import Config
from stats_analyzer import StatsAnalyzer
from data_processing import DataProcessor
from jet_analyzer import JetStreamAnalyzer


class Visualizer:
    """A collection of static methods for plotting climate analysis results."""

    # --- Global Style Settings ---
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['figure.titlesize'] = 12
    
    # Unified Colors for Global Warming Levels
    GWL_COLORS = {2.0: '#0072B2', 3.0: '#D55E00'}

    @staticmethod
    def ensure_plot_dir_exists():
        """Ensure the plot directory from the config exists."""
        if not os.path.exists(Config.PLOT_DIR):
            os.makedirs(Config.PLOT_DIR, exist_ok=True)
            logging.info(f"Created plot directory: {Config.PLOT_DIR}")

    @staticmethod
    def _format_scenario_title(scenario):
        """Helper to format scenario string for titles (e.g., 'ssp585' -> 'SSP5-8.5')."""
        if scenario.lower() == 'ssp585': return 'SSP5-8.5'
        if scenario.lower() == 'ssp245': return 'SSP2-4.5'
        return scenario.upper()

    @staticmethod
    def plot_regression_map(ax, slopes, p_values, lons, lats, title, box_coords,
                          season_label, variable, ua_seasonal_mean=None,
                          show_jet_boxes=False, significance_level=0.05, std_dev_predictor=None,
                          stipple_skip=None, dataset_key=None, set_title_in_helper=True): # MODIFIED: Added dataset_key and set_title_in_helper
        """Helper function to create a single regression map panel."""
        ax.set_extent(Config.PLOT_MAP_EXTENT, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=0)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = gl.right_labels = False
        gl.xlabel_style = {'size': 8}; gl.ylabel_style = {'size': 8}
        
        cmap = Config.PLOT_COLORMAP
        vmin, vmax = (-2.0, 2.0) if variable == 'pr' else (-3.0, 3.0)
        label = f'U850 Slope (m/s per std. dev. of box {variable})'
        if std_dev_predictor is not None and not np.isnan(std_dev_predictor):
            unit = 'mm/day' if variable == 'pr' else '°C'
            label += f'\n(1 std. dev. = {std_dev_predictor:.2f} {unit})'

        if lons is None or lats is None or np.all(np.isnan(lons)) or np.all(np.isnan(lats)):
            ax.text(0.5, 0.5, "Coordinate Data Missing", transform=ax.transAxes, ha='center', va='center')
            ax.set_title(f"{title}\n{season_label}", fontsize=10)
            return None, None

        lons_plot, lats_plot = np.meshgrid(lons, lats) if lons.ndim == 1 else (lons, lats)
        
        cf = None
        if slopes is not None and not np.all(np.isnan(slopes)):
            cf = ax.pcolormesh(lons_plot, lats_plot, slopes, shading='auto',
                               cmap=cmap, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
        
        if ua_seasonal_mean is not None and not np.all(np.isnan(ua_seasonal_mean)):
            contour_levels = np.arange(4, 21, 4)
            cs = ax.contour(lons_plot, lats_plot, ua_seasonal_mean, levels=contour_levels, colors='black',
                            linewidths=0.8, transform=ccrs.PlateCarree())
            ax.clabel(cs, inline=True, fontsize=7, fmt='%d')

        if p_values is not None and slopes is not None:
            sig_mask = (p_values < significance_level) & np.isfinite(slopes)
            
            # --- MODIFIED BLOCK: DYNAMIC STIPPLE SKIP ---
            # Set default skip value
            skip_val = 2
            # If the dataset is ERA5, use a larger skip value
            if dataset_key == Config.DATASET_ERA5:
                skip_val = 7
            # Allow manual override from function call
            if stipple_skip is not None:
                skip_val = stipple_skip
            
            if skip_val > 1:
                points_to_plot_mask = np.zeros_like(sig_mask, dtype=bool)
                points_to_plot_mask[::skip_val, ::skip_val] = True
                final_mask = sig_mask & points_to_plot_mask
            else:
                final_mask = sig_mask
            # --- END MODIFIED BLOCK ---

            if np.any(final_mask):
                 ax.scatter(lons_plot[final_mask], lats_plot[final_mask], s=0.2, color='dimgray', marker='.',
                            alpha=0.6, transform=ccrs.PlateCarree())

        box_lon_min, box_lon_max, box_lat_min, box_lat_max = box_coords
        box = mpatches.Rectangle((box_lon_min, box_lat_min), box_lon_max - box_lon_min, box_lat_max - box_lat_min,
                                 fill=False, edgecolor='lime', linewidth=2, zorder=10, transform=ccrs.PlateCarree())
        ax.add_patch(box)
        
        if title: # Changed from set_title_in_helper
            ax.set_title(f"{title}\n{season_label}", fontsize=10)
        return cf, label

    @staticmethod
    def plot_regression_analysis(all_season_data, dataset_key):
        """
        Creates a panel plot for regression maps (U850 vs PR/TAS indices).
        MODIFIED for ERL Figure 1: 2x2 Layout (Winter/Summer x PR/TAS)
        """
        logging.info(f"Plotting U850 vs Box Index regression maps for {dataset_key}...")
        Visualizer.ensure_plot_dir_exists()

        if not isinstance(all_season_data, dict) or not all_season_data.get('Winter') or not all_season_data.get('Summer'):
             logging.warning(f"Skipping regression plot for {dataset_key}: Missing Winter or Summer data.")
             return

        fig = plt.figure(figsize=(15, 9))
        gs = gridspec.GridSpec(2, 3, width_ratios=[10, 10, 1], height_ratios=[1, 1], wspace=0.1, hspace=0.3)
        box_coords = [Config.BOX_LON_MIN, Config.BOX_LON_MAX, Config.BOX_LAT_MIN, Config.BOX_LAT_MAX]
        
        # Plot PR panels
        winter_pr_data = all_season_data['Winter']
        ax_pr_winter = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
        cf_pr, label_pr = Visualizer.plot_regression_map(ax_pr_winter, winter_pr_data.get('slopes_pr'), winter_pr_data.get('p_values_pr'), winter_pr_data.get('lons'), winter_pr_data.get('lats'), f"Winter (DJF) Precipitation", box_coords, "DJF", 'pr', ua_seasonal_mean=winter_pr_data.get('ua850_mean'), std_dev_predictor=winter_pr_data.get('std_dev_pr'), dataset_key=dataset_key)

        summer_pr_data = all_season_data['Summer']
        ax_pr_summer = fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree())
        Visualizer.plot_regression_map(ax_pr_summer, summer_pr_data.get('slopes_pr'), summer_pr_data.get('p_values_pr'), summer_pr_data.get('lons'), summer_pr_data.get('lats'), f"Summer (JJA) Precipitation", box_coords, "JJA", 'pr', ua_seasonal_mean=summer_pr_data.get('ua850_mean'), std_dev_predictor=summer_pr_data.get('std_dev_pr'), dataset_key=dataset_key)

        if cf_pr:
            cax_pr = fig.add_subplot(gs[0, 2]); fig.colorbar(cf_pr, cax=cax_pr, extend='both', label=label_pr)

        # Plot TAS panels
        winter_tas_data = all_season_data['Winter']
        ax_tas_winter = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
        cf_tas, label_tas = Visualizer.plot_regression_map(ax_tas_winter, winter_tas_data.get('slopes_tas'), winter_tas_data.get('p_values_tas'), winter_tas_data.get('lons'), winter_tas_data.get('lats'), f"Winter (DJF) Temperature", box_coords, "DJF", 'tas', ua_seasonal_mean=winter_tas_data.get('ua850_mean'), std_dev_predictor=winter_tas_data.get('std_dev_tas'), dataset_key=dataset_key)

        summer_tas_data = all_season_data['Summer']
        ax_tas_summer = fig.add_subplot(gs[1, 1], projection=ccrs.PlateCarree())
        Visualizer.plot_regression_map(ax_tas_summer, summer_tas_data.get('slopes_tas'), summer_tas_data.get('p_values_tas'), summer_tas_data.get('lons'), summer_tas_data.get('lats'), f"Summer (JJA) Temperature", box_coords, "JJA", 'tas', ua_seasonal_mean=summer_tas_data.get('ua850_mean'), std_dev_predictor=summer_tas_data.get('std_dev_tas'), dataset_key=dataset_key)

        if cf_tas:
            cax_tas = fig.add_subplot(gs[1, 2]); fig.colorbar(cf_tas, cax=cax_tas, extend='both', label=label_tas)
            
        plt.suptitle(f"{dataset_key}: U850 Regression onto Box Climate Indices (Detrended, Normalized Predictors)", fontsize=14, weight='bold')
        fig.tight_layout(rect=(0, 0, 0.95, 0.95))
        filename = os.path.join(Config.PLOT_DIR, f'regression_maps_norm_{dataset_key}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logging.info(f"Saved regression analysis plot to {filename}")

    @staticmethod
    def plot_erl_figure1_regression_maps(all_season_data, dataset_key):
        """
        Creates ERL Figure 1: Historical Jet Influence (Regression Maps).
        Layout: 2x2 Grid (Winter/Summer x Precip/Temp).
        MODIFIED: Now includes Jet Index Definition Boxes.
        """
        logging.info(f"Plotting ERL Figure 1 for {dataset_key}...")
        Visualizer.ensure_plot_dir_exists()

        if not isinstance(all_season_data, dict) or not all_season_data.get('Winter') or not all_season_data.get('Summer'):
             logging.warning(f"Skipping ERL Figure 1 for {dataset_key}: Missing Winter or Summer data.")
             return

        # --- MODIFIED LAYOUT: 2x2 Grid ---
        fig = plt.figure(figsize=(5.9, 8.5))
        gs = gridspec.GridSpec(2, 2, wspace=0.3, hspace=0.3, top=0.92, bottom=0.15, left=0.1, right=0.9)
        box_coords = [Config.BOX_LON_MIN, Config.BOX_LON_MAX, Config.BOX_LAT_MIN, Config.BOX_LAT_MAX]
        
        # --- Row 1: Winter (Left: PR, Right: TAS) ---
        winter_data = all_season_data['Winter']
        
        # Winter PR
        ax_pr_winter = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
        cf_pr, label_pr = Visualizer.plot_regression_map(
            ax_pr_winter, winter_data.get('slopes_pr'), winter_data.get('p_values_pr'), 
            winter_data.get('lons'), winter_data.get('lats'), 
            "", box_coords, "", 'pr', 
            ua_seasonal_mean=winter_data.get('ua850_mean'), 
            std_dev_predictor=winter_data.get('std_dev_pr'), dataset_key=dataset_key,
            set_title_in_helper=False
        )
        ax_pr_winter.set_title("a) Winter (DJF) Precipitation", loc='left', fontweight='bold')

        # Winter TAS
        ax_tas_winter = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
        cf_tas, label_tas = Visualizer.plot_regression_map(
            ax_tas_winter, winter_data.get('slopes_tas'), winter_data.get('p_values_tas'), 
            winter_data.get('lons'), winter_data.get('lats'), 
            "", box_coords, "", 'tas', 
            ua_seasonal_mean=winter_data.get('ua850_mean'), 
            std_dev_predictor=winter_data.get('std_dev_tas'), dataset_key=dataset_key,
            set_title_in_helper=False
        )
        ax_tas_winter.set_title("b) Winter (DJF) Temperature", loc='left', fontweight='bold')

        # --- Row 2: Summer (Left: PR, Right: TAS) ---
        summer_data = all_season_data['Summer']
        
        # Summer PR
        ax_pr_summer = fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree())
        Visualizer.plot_regression_map(
            ax_pr_summer, summer_data.get('slopes_pr'), summer_data.get('p_values_pr'), 
            summer_data.get('lons'), summer_data.get('lats'), 
            "", box_coords, "", 'pr', 
            ua_seasonal_mean=summer_data.get('ua850_mean'), 
            std_dev_predictor=summer_data.get('std_dev_pr'), dataset_key=dataset_key,
            set_title_in_helper=False
        )
        ax_pr_summer.set_title("c) Summer (JJA) Precipitation", loc='left', fontweight='bold')

        # Summer TAS
        ax_tas_summer = fig.add_subplot(gs[1, 1], projection=ccrs.PlateCarree())
        Visualizer.plot_regression_map(
            ax_tas_summer, summer_data.get('slopes_tas'), summer_data.get('p_values_tas'), 
            summer_data.get('lons'), summer_data.get('lats'), 
            "", box_coords, "", 'tas', 
            ua_seasonal_mean=summer_data.get('ua850_mean'), 
            std_dev_predictor=summer_data.get('std_dev_tas'), dataset_key=dataset_key,
            set_title_in_helper=False
        )
        ax_tas_summer.set_title("d) Summer (JJA) Temperature", loc='left', fontweight='bold')

        # --- NEW: Add Jet Definition Boxes to all subplots ---
        # Get coordinates from Config
        speed_box = [Config.JET_SPEED_BOX_LON_MIN, Config.JET_SPEED_BOX_LON_MAX,
                     Config.JET_SPEED_BOX_LAT_MIN, Config.JET_SPEED_BOX_LAT_MAX]
        lat_box = [Config.JET_LAT_BOX_LON_MIN, Config.JET_LAT_BOX_LON_MAX,
                   Config.JET_LAT_BOX_LAT_MIN, Config.JET_LAT_BOX_LAT_MAX]

        # Iterate over all axes to add the boxes
        for ax in [ax_pr_winter, ax_tas_winter, ax_pr_summer, ax_tas_summer]:
            # Jet Speed Box (Red, Solid)
            ax.add_patch(mpatches.Rectangle(
                (speed_box[0], speed_box[2]), speed_box[1] - speed_box[0], speed_box[3] - speed_box[2],
                fill=False, edgecolor='darkred', linewidth=1.5, linestyle='-', zorder=12, transform=ccrs.PlateCarree()
            ))
            # Jet Latitude Box (Blue, Dashed)
            ax.add_patch(mpatches.Rectangle(
                (lat_box[0], lat_box[2]), lat_box[1] - lat_box[0], lat_box[3] - lat_box[2],
                fill=False, edgecolor='darkblue', linewidth=1.5, linestyle='--', zorder=12, transform=ccrs.PlateCarree()
            ))
        # --- END NEW ---

        # --- Colorbars (Horizontal at bottom) ---
        # PR Colorbar (Left side)
        cax_pr = fig.add_axes((0.1, 0.04, 0.35, 0.02))
        if cf_pr is not None:
            fig.colorbar(cf_pr, cax=cax_pr, orientation='horizontal', label=label_pr, extend='both')
        
        # TAS Colorbar (Right side)
        cax_tas = fig.add_axes((0.55, 0.04, 0.35, 0.02))
        if cf_tas is not None:
            fig.colorbar(cf_tas, cax=cax_tas, orientation='horizontal', label=label_tas, extend='both')

        plt.suptitle(f"{dataset_key}: Historical Jet Influence on Local Climate", weight='bold') 
        
        fig.tight_layout(rect=(0, 0.1, 1, 0.95), h_pad=2.0, w_pad=2.0)
        filename = os.path.join(Config.PLOT_DIR, 'Figure1_regression_maps_ERA5.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logging.info(f"Saved ERL Figure 1 to {filename}")


    @staticmethod
    def plot_jet_correlation_maps(correlation_data_20crv3, correlation_data_era5, season):
        """Plots pre-calculated jet correlation/regression slope maps for a given season."""
        logging.info(f"Plotting jet correlation maps for {season}...")
        Visualizer.ensure_plot_dir_exists()

        if not correlation_data_20crv3 or not correlation_data_era5:
            logging.warning(f"Skipping jet correlation maps for {season} due to missing data for one or both datasets.")
            return

        plot_configs = {
            'jet_speed_tas': {'title': f'{season} Jet Speed vs. Temperature', 'cmap': 'coolwarm', 'vmin': -1.0, 'vmax': 1.0, 'base_label': 'TAS Slope (°C per std. dev. of Jet Speed)'},
            'jet_speed_pr':  {'title': f'{season} Jet Speed vs. Precipitation', 'cmap': 'BrBG', 'vmin': -0.5, 'vmax': 0.5, 'base_label': 'PR Slope (mm/day per std. dev. of Jet Speed)'},
            'jet_lat_tas':   {'title': f'{season} Jet Latitude vs. Temperature', 'cmap': 'coolwarm', 'vmin': -1.0, 'vmax': 1.0, 'base_label': 'TAS Slope (°C per std. dev. of Jet Lat.)'},
            'jet_lat_pr':    {'title': f'{season} Jet Latitude vs. Precipitation', 'cmap': 'BrBG', 'vmin': -0.5, 'vmax': 0.5, 'base_label': 'PR Slope (mm/day per std. dev. of Jet Lat.)'}
        }

        fig = plt.figure(figsize=(12, 18))
        gs = gridspec.GridSpec(len(plot_configs), 3, width_ratios=[10, 10, 1], wspace=0.1, hspace=0.3)

        cf = None
        row_idx = 0
        for key, config in plot_configs.items():
            if 'speed' in key:
                jet_box_coords = (Config.JET_SPEED_BOX_LON_MIN, Config.JET_SPEED_BOX_LON_MAX,
                                  Config.JET_SPEED_BOX_LAT_MIN, Config.JET_SPEED_BOX_LAT_MAX)
                jet_box_edgecolor = 'blue'
            elif 'lat' in key:
                jet_box_coords = (Config.JET_LAT_BOX_LON_MIN, Config.JET_LAT_BOX_LON_MAX,
                                  Config.JET_LAT_BOX_LAT_MIN, Config.JET_LAT_BOX_LAT_MAX)
                jet_box_edgecolor = 'red'
            else:
                jet_box_coords = None
                jet_box_edgecolor = 'black'
            
            analysis_box_coords = (Config.BOX_LON_MIN, Config.BOX_LON_MAX, 
                                   Config.BOX_LAT_MIN, Config.BOX_LAT_MAX)

            # --- Data for 20CRv3 ---
            data_20crv3 = correlation_data_20crv3.get(key)
            ax1 = cast(Any, fig.add_subplot(gs[row_idx, 0], projection=ccrs.PlateCarree()))
            ax1.set_title(f"20CRv3: {config['title']}", fontsize=10)
            
            if data_20crv3 and data_20crv3.get('slopes') is not None:
                lons, lats = data_20crv3['lons'], data_20crv3['lats']
                lons_plot, lats_plot = np.meshgrid(lons, lats) if lons.ndim == 1 else (lons, lats)
                
                ax1.set_extent(Config.PLOT_MAP_EXTENT, crs=ccrs.PlateCarree())
                ax1.add_feature(cfeature.COASTLINE, linewidth=0.5); ax1.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
                gl = ax1.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
                gl.top_labels = gl.right_labels = False
                gl.xlabel_style = {'size': 8}; gl.ylabel_style = {'size': 8}

                cf = ax1.pcolormesh(lons_plot, lats_plot, data_20crv3['slopes'], shading='auto',
                                    cmap=config['cmap'], vmin=config['vmin'], vmax=config['vmax'],
                                    transform=ccrs.PlateCarree())
                
                sig_mask = (data_20crv3['p_values'] < 0.05) & np.isfinite(data_20crv3['slopes'])
                # MODIFIED: Stipple skip logic for 20CRv3 (default is 2)
                stipple_skip_20crv3 = 2
                points_to_plot_mask_20crv3 = np.zeros_like(sig_mask, dtype=bool)
                points_to_plot_mask_20crv3[::stipple_skip_20crv3, ::stipple_skip_20crv3] = True
                final_mask_20crv3 = sig_mask & points_to_plot_mask_20crv3
                if np.any(final_mask_20crv3):
                    ax1.scatter(lons_plot[final_mask_20crv3], lats_plot[final_mask_20crv3], s=0.5, color='dimgray', marker='.',
                                alpha=0.4, transform=ccrs.PlateCarree())
                
                if jet_box_coords:
                    lon_min, lon_max, lat_min, lat_max = jet_box_coords
                    ax1.add_patch(mpatches.Rectangle((lon_min, lat_min), lon_max - lon_min, lat_max - lat_min,
                                  fill=False, edgecolor=jet_box_edgecolor, linewidth=1.5, linestyle='--',
                                  zorder=10, transform=ccrs.PlateCarree()))
                
                lon_min, lon_max, lat_min, lat_max = analysis_box_coords
                ax1.add_patch(mpatches.Rectangle((lon_min, lat_min), lon_max - lon_min, lat_max - lat_min,
                                             fill=False, edgecolor='lime', linewidth=2, linestyle='-',
                                             zorder=10, transform=ccrs.PlateCarree()))
            else:
                ax1.text(0.5, 0.5, "Data not available", transform=ax1.transAxes, ha='center', va='center')

            # --- Data for ERA5 ---
            data_era5 = correlation_data_era5.get(key)
            ax2 = cast(Any, fig.add_subplot(gs[row_idx, 1], projection=ccrs.PlateCarree()))
            ax2.set_title(f"ERA5: {config['title']}", fontsize=10)

            if data_era5 and data_era5.get('slopes') is not None:
                lons, lats = data_era5['lons'], data_era5['lats']
                lons_plot, lats_plot = np.meshgrid(lons, lats) if lons.ndim == 1 else (lons, lats)

                ax2.set_extent(Config.PLOT_MAP_EXTENT, crs=ccrs.PlateCarree())
                ax2.add_feature(cfeature.COASTLINE, linewidth=0.5); ax2.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
                gl = ax2.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
                gl.top_labels = gl.right_labels = False; gl.left_labels = False
                gl.xlabel_style = {'size': 8}; gl.ylabel_style = {'size': 8}

                cf = ax2.pcolormesh(lons_plot, lats_plot, data_era5['slopes'], shading='auto',
                                    cmap=config['cmap'], vmin=config['vmin'], vmax=config['vmax'],
                                    transform=ccrs.PlateCarree())
                
                sig_mask = (data_era5['p_values'] < 0.05) & np.isfinite(data_era5['slopes'])
                # MODIFIED: Stipple skip logic for ERA5 (set to 8)
                stipple_skip_era5 = 7
                points_to_plot_mask_era5 = np.zeros_like(sig_mask, dtype=bool)
                points_to_plot_mask_era5[::stipple_skip_era5, ::stipple_skip_era5] = True
                final_mask_era5 = sig_mask & points_to_plot_mask_era5
                if np.any(final_mask_era5):
                    ax2.scatter(lons_plot[final_mask_era5], lats_plot[final_mask_era5], s=0.5, color='dimgray', marker='.',
                                alpha=0.4, transform=ccrs.PlateCarree())
                                
                if jet_box_coords:
                    lon_min, lon_max, lat_min, lat_max = jet_box_coords
                    ax2.add_patch(mpatches.Rectangle((lon_min, lat_min), lon_max - lon_min, lat_max - lat_min,
                                                 fill=False, edgecolor=jet_box_edgecolor, linewidth=1.5, linestyle='--',
                                                 zorder=10, transform=ccrs.PlateCarree()))
                
                lon_min, lon_max, lat_min, lat_max = analysis_box_coords
                ax2.add_patch(mpatches.Rectangle((lon_min, lat_min), lon_max - lon_min, lat_max - lat_min,
                                             fill=False, edgecolor='lime', linewidth=2, linestyle='-',
                                             zorder=10, transform=ccrs.PlateCarree()))
            else:
                ax2.text(0.5, 0.5, "Data not available", transform=ax2.transAxes, ha='center', va='center')

            # --- Colorbar ---
            cax = fig.add_subplot(gs[row_idx, 2])
            
            final_label = config['base_label']
            std_dev_val = data_era5.get('std_dev_jet') if data_era5 else data_20crv3.get('std_dev_jet')
            if std_dev_val is not None and not np.isnan(std_dev_val):
                unit = "m/s" if "Speed" in config['title'] else "°Lat"
                final_label += f'\n(1 std. dev. = {std_dev_val:.2f} {unit})'
            
            if cf is not None:
                fig.colorbar(cf, cax=cax, extend='both', label=final_label)

            row_idx += 1
        
        plt.suptitle(f"Correlation Maps of Jet Variations and Climate ({season}, Detrended)", fontsize=16, weight='bold')
        fig.tight_layout(rect=(0, 0, 0.95, 0.96))
        filename = os.path.join(Config.PLOT_DIR, f'jet_correlation_maps_{season.lower()}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)

    @staticmethod
    def plot_jet_changes_vs_gwl(cmip6_results, scenario, filename=None):
        """
        Plots CMIP6 jet index changes vs GWL, with percentile spread.
        MODIFIED to create a 2x2 grid for Winter/Summer and Speed/Latitude.
        NOW INCLUDES tolerance error bars for storylines.
        """
        logging.info("Plotting Jet Changes vs GWL (2x2 layout with Storyline Tolerances)...")
        Visualizer.ensure_plot_dir_exists()
        scenario_title = Visualizer._format_scenario_title(scenario)

        if not cmip6_results or 'all_individual_model_deltas_for_plot' not in cmip6_results:
            logging.warning("Cannot plot jet_changes_vs_gwl: Missing CMIP6 analysis results.")
            return

        all_deltas = cmip6_results['all_individual_model_deltas_for_plot']
        mmm_changes = cmip6_results.get('mmm_changes', {})
        
        jet_indices_to_plot = ['JJA_JetLat', 'JJA_JetSpeed', 'DJF_JetLat', 'DJF_JetSpeed']
        
        fig, axs = plt.subplots(2, 2, figsize=(15, 12), sharex=True, squeeze=False)
        axs = axs.flatten()

        # --- Legend Setup ---
        master_legend_handles = [
            plt.Line2D([0], [0], color='darkgray', lw=0.8, marker='.', markersize=4, linestyle='-'),
            mpatches.Patch(color='lightcoral', alpha=0.4),
            plt.Line2D([0], [0], color='black', lw=2.5, marker='o', markersize=7, linestyle='-')
        ]
        master_legend_labels = [
            'Individual CMIP6 Models',
            '100% Model Spread',
            'Multi-Model Mean'
        ]

        storyline_styles = {
            'MMM':    {'color': '#1f77b4', 'marker': 'X', 's': 60}, 
            'Core High':    {'color': '#ff7f0e', 'marker': 'X', 's': 60},
            'Extreme Low':  {'color': '#2ca02c', 'marker': 'P', 's': 70},
            'Extreme High': {'color': '#d62728', 'marker': 'P', 's': 70}
        }

        used_storyline_types = set()
        for jet_idx in Config.STORYLINE_JET_CHANGES:
            for gwl in Config.STORYLINE_JET_CHANGES.get(jet_idx, {}):
                for storyline_type in Config.STORYLINE_JET_CHANGES[jet_idx][gwl]:
                    used_storyline_types.add(storyline_type)

        for storyline_type in storyline_styles:
            if storyline_type in used_storyline_types:
                style = storyline_styles[storyline_type]
                master_legend_handles.append(
                    plt.Line2D([0], [0], marker=style['marker'], color='w',
                            markerfacecolor=style['color'], markeredgecolor='k', markersize=9)
                )
                master_legend_labels.append(f'Storyline: {storyline_type}')
        
        # NEW: Add legend entry for tolerance bars
        master_legend_handles.append(plt.Line2D([0], [0], color='gray', lw=1.5, ls='-', marker='_'))
        master_legend_labels.append('Storyline Classification Tolerance')
        # --- End Legend Setup ---
        
        for i, jet_idx in enumerate(jet_indices_to_plot):
            ax = axs[i]
            
            deltas_by_gwl = all_deltas.get(jet_idx)
            if not deltas_by_gwl:
                ax.text(0.5, 0.5, f"No data for\n{jet_idx}", ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'Projected Change in {jet_idx.replace("_", " ")}')
                continue

            gwls_fine = sorted(deltas_by_gwl.keys())

            model_runs = {}
            for gwl, model_deltas_dict in deltas_by_gwl.items():
                for model_key, delta_val in model_deltas_dict.items():
                    if model_key not in model_runs:
                        model_runs[model_key] = []
                    model_runs[model_key].append((gwl, delta_val))

            for model_key, run_data in model_runs.items():
                run_data.sort()
                gwls_sorted = [d[0] for d in run_data]
                values_sorted = [d[1] for d in run_data]
                if len(gwls_sorted) > 1:
                    ax.plot(gwls_sorted, values_sorted, marker='.', linestyle='-', color='darkgray', alpha=0.5, lw=0.8)

            delta_values_per_gwl = [list(deltas_by_gwl[gwl].values()) for gwl in gwls_fine]
            p10 = [np.nanmin(d) if d else np.nan for d in delta_values_per_gwl]
            p90 = [np.nanmax(d) if d else np.nan for d in delta_values_per_gwl]
            
            valid_indices = [j for j, (p10_val, p90_val) in enumerate(zip(p10, p90)) if not np.isnan(p10_val) and not np.isnan(p90_val)]
            if valid_indices:
                gwls_plot = [gwls_fine[j] for j in valid_indices]
                p10_plot = [p10[j] for j in valid_indices]
                p90_plot = [p90[j] for j in valid_indices]
                ax.fill_between(gwls_plot, p10_plot, p90_plot, color='lightcoral', alpha=0.4)
                    
            gwls_main = sorted(mmm_changes.keys())
            mmm_values = [mmm_changes[gwl].get(jet_idx, np.nan) for gwl in gwls_main]
            ax.plot(gwls_main, mmm_values, marker='o', linestyle='-', color='black', lw=2.5, markersize=7)

            # --- MODIFIED BLOCK: PLOT STORYLINES WITH ERROR BARS ---
            for gwl, storylines in Config.STORYLINE_JET_CHANGES.get(jet_idx, {}).items():
                # Define tolerance based on the jet index name
                tolerance = 0.25 if 'Speed' in jet_idx else 0.35
                
                for name, value in storylines.items():
                    if name in storyline_styles:
                        style = storyline_styles[name]
                        # Use ax.errorbar instead of ax.scatter
                        ax.errorbar(
                            x=gwl, 
                            y=value,
                            yerr=tolerance,  # The vertical error bar
                            marker=style['marker'],
                            color=style['color'],
                            markersize=float(style['s']) / 7, # Adjust markersize, as 's' in scatter scales differently
                            markeredgecolor='black',
                            markeredgewidth=0.8,
                            linestyle='none', # No connecting line
                            capsize=4,       # Size of the caps at the end of the error bars
                            elinewidth=1.5,  # Thickness of the error bars
                            ecolor='gray',   # Color of the error bars
                            zorder=5
                        )
            # --- END OF MODIFIED BLOCK ---
            
            season_title = "Summer (JJA)" if "JJA" in jet_idx else "Winter (DJF)"
            index_type_title = "Jet Latitude" if "Lat" in jet_idx else "Jet Speed"
            
            ylabel = f'Change in {index_type_title}'
            if '_pr' in jet_idx: ylabel += ' (%)'
            elif 'Lat' in jet_idx: ylabel += ' (°Lat)'
            elif 'Speed' in jet_idx: ylabel += ' (m/s)'
            ax.set_ylabel(ylabel)
            
            if i >= 2:
                ax.set_xlabel('Global Warming Level (°C)')
                
            ax.set_title(f'Projected Change in {season_title} {index_type_title}')
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.axhline(0, color='grey', lw=0.8)
        
        fig.legend(handles=master_legend_handles, labels=master_legend_labels,
                loc='lower center',
                bbox_to_anchor=(0.5, -0.02),
                ncol=4, # Increased to 4 columns for the new legend entry
                fontsize=11,
                frameon=True)

        fig.tight_layout(rect=(0, 0.06, 1, 0.95))
        
        # Use the scenario argument for a dynamic title and filename
        fig.suptitle(f"CMIP6 Projected Jet Changes vs. Global Warming Level ({scenario_title})", fontsize=16, weight='bold')
        if filename is None:
            filename = f"cmip6_jet_changes_vs_gwl_{scenario}.png"
        filepath = os.path.join(Config.PLOT_DIR, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logging.info(f"Saved jet_changes_vs_gwl plot to {filepath}")

    @staticmethod
    def plot_jet_impact_comparison_maps(impact_data_20crv3, impact_data_era5, season):
        """
        Creates a comparison plot (8 subplots) for jet impact regressions for a given season.
        Compares 20CRv3 and ERA5 side-by-side for different jet indices and variables.
        NOW WITH CORRECTED STIPPLING.
        """
        logging.info(f"Plotting combined jet impact regression maps for {season}...")
        Visualizer.ensure_plot_dir_exists()

        if not impact_data_20crv3 or not impact_data_era5:
            logging.warning(f"Skipping combined jet impact maps for {season} due to missing data.")
            return

        plot_configs = {
            'jet_speed_tas': {'title': 'Jet Speed vs. Temperature', 'cmap': 'coolwarm', 'vmin': -1.0, 'vmax': 1.0, 'base_label': 'TAS Slope (°C per std. dev. of Jet Speed)'},
            'jet_speed_pr':  {'title': 'Jet Speed vs. Precipitation', 'cmap': 'BrBG', 'vmin': -0.5, 'vmax': 0.5, 'base_label': 'PR Slope (mm/day per std. dev. of Jet Speed)'},
            'jet_lat_tas':   {'title': 'Jet Latitude vs. Temperature', 'cmap': 'coolwarm', 'vmin': -1.0, 'vmax': 1.0, 'base_label': 'TAS Slope (°C per std. dev. of Jet Lat.)'},
            'jet_lat_pr':    {'title': 'Jet Latitude vs. Precipitation', 'cmap': 'BrBG', 'vmin': -0.5, 'vmax': 0.5, 'base_label': 'PR Slope (mm/day per std. dev. of Jet Lat.)'}
        }

        fig = plt.figure(figsize=(12, 18))
        gs = gridspec.GridSpec(len(plot_configs), 3, width_ratios=[10, 10, 1], wspace=0.1, hspace=0.3)

        row_idx = 0
        for key, config in plot_configs.items():
            # Define jet and analysis box properties
            if 'speed' in key:
                jet_box_coords = (Config.JET_SPEED_BOX_LON_MIN, Config.JET_SPEED_BOX_LON_MAX,
                                Config.JET_SPEED_BOX_LAT_MIN, Config.JET_SPEED_BOX_LAT_MAX)
                jet_box_edgecolor = 'blue'
            elif 'lat' in key:
                jet_box_coords = (Config.JET_LAT_BOX_LON_MIN, Config.JET_LAT_BOX_LON_MAX,
                                Config.JET_LAT_BOX_LAT_MIN, Config.JET_LAT_BOX_LAT_MAX)
                jet_box_edgecolor = 'red'
            else:
                jet_box_coords = None
                jet_box_edgecolor = 'black'
            
            analysis_box_coords = (Config.BOX_LON_MIN, Config.BOX_LON_MAX, 
                                Config.BOX_LAT_MIN, Config.BOX_LAT_MAX)

            # --- Subplot for 20CRv3 ---
            data_20crv3 = impact_data_20crv3.get(key)
            ax1 = cast(Any, fig.add_subplot(gs[row_idx, 0], projection=ccrs.PlateCarree()))
            ax1.set_title(f"20CRv3: {config['title']}", fontsize=10)
            
            cf = None # Initialize cf to handle cases where data is missing
            if data_20crv3 and data_20crv3.get('slopes') is not None:
                lons, lats = data_20crv3['lons'], data_20crv3['lats']
                lons_plot, lats_plot = np.meshgrid(lons, lats) if lons.ndim == 1 else (lons, lats)
                
                ax1.set_extent(Config.PLOT_MAP_EXTENT, crs=ccrs.PlateCarree())
                ax1.add_feature(cfeature.COASTLINE, linewidth=0.5); ax1.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
                gl = ax1.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
                gl.top_labels = gl.right_labels = False
                gl.xlabel_style = {'size': 8}; gl.ylabel_style = {'size': 8}

                cf = ax1.pcolormesh(lons_plot, lats_plot, data_20crv3['slopes'], shading='auto',
                                    cmap=config['cmap'], vmin=config['vmin'], vmax=config['vmax'],
                                    transform=ccrs.PlateCarree())
                
                # --- START OF CHANGE: 20CRv3 Stippling ---
                if 'p_values' in data_20crv3:
                    sig_mask = (data_20crv3['p_values'] < 0.05) & np.isfinite(data_20crv3['slopes'])
                    stipple_skip_20crv3 = 2 # Set skip value for 20CRv3
                    points_to_plot_mask = np.zeros_like(sig_mask, dtype=bool)
                    points_to_plot_mask[::stipple_skip_20crv3, ::stipple_skip_20crv3] = True
                    final_mask = sig_mask & points_to_plot_mask
                    if np.any(final_mask):
                        ax1.scatter(lons_plot[final_mask], lats_plot[final_mask], s=0.5, color='dimgray', marker='.',
                                    alpha=0.4, transform=ccrs.PlateCarree())
                # --- END OF CHANGE ---
                
                # Draw boxes
                if jet_box_coords is not None:
                    lon_min, lon_max, lat_min, lat_max = jet_box_coords
                    ax1.add_patch(mpatches.Rectangle((lon_min, lat_min), lon_max - lon_min, lat_max - lat_min,
                                    fill=False, edgecolor=jet_box_edgecolor, linewidth=1.5, linestyle='--',
                                    zorder=10, transform=ccrs.PlateCarree()))
                lon_min, lon_max, lat_min, lat_max = analysis_box_coords
                ax1.add_patch(mpatches.Rectangle((lon_min, lat_min), lon_max - lon_min, lat_max - lat_min,
                                            fill=False, edgecolor='lime', linewidth=2, linestyle='-',
                                            zorder=10, transform=ccrs.PlateCarree()))
            else:
                ax1.text(0.5, 0.5, "Data not available", transform=ax1.transAxes, ha='center', va='center')

            # --- Subplot for ERA5 ---
            data_era5 = impact_data_era5.get(key)
            ax2 = cast(Any, fig.add_subplot(gs[row_idx, 1], projection=ccrs.PlateCarree()))
            ax2.set_title(f"ERA5: {config['title']}", fontsize=10)

            if data_era5 and data_era5.get('slopes') is not None:
                lons, lats = data_era5['lons'], data_era5['lats']
                lons_plot, lats_plot = np.meshgrid(lons, lats) if lons.ndim == 1 else (lons, lats)

                ax2.set_extent(Config.PLOT_MAP_EXTENT, crs=ccrs.PlateCarree())
                ax2.add_feature(cfeature.COASTLINE, linewidth=0.5); ax2.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
                gl = ax2.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
                gl.top_labels = gl.right_labels = False; gl.left_labels = False
                gl.xlabel_style = {'size': 8}; gl.ylabel_style = {'size': 8}

                cf_era5 = ax2.pcolormesh(lons_plot, lats_plot, data_era5['slopes'], shading='auto',
                                    cmap=config['cmap'], vmin=config['vmin'], vmax=config['vmax'],
                                    transform=ccrs.PlateCarree())
                if cf is None: cf = cf_era5

                # --- START OF CHANGE: ERA5 Stippling ---
                if 'p_values' in data_era5:
                    sig_mask = (data_era5['p_values'] < 0.05) & np.isfinite(data_era5['slopes'])
                    stipple_skip_era5 = 7 # Set skip value for ERA5
                    points_to_plot_mask = np.zeros_like(sig_mask, dtype=bool)
                    points_to_plot_mask[::stipple_skip_era5, ::stipple_skip_era5] = True
                    final_mask = sig_mask & points_to_plot_mask
                    if np.any(final_mask):
                        ax2.scatter(lons_plot[final_mask], lats_plot[final_mask], s=0.5, color='dimgray', marker='.',
                                    alpha=0.4, transform=ccrs.PlateCarree())
                # --- END OF CHANGE ---
                
                # Draw boxes
                if jet_box_coords is not None:
                    lon_min, lon_max, lat_min, lat_max = jet_box_coords
                    ax2.add_patch(mpatches.Rectangle((lon_min, lat_min), lon_max - lon_min, lat_max - lat_min,
                                                fill=False, edgecolor=jet_box_edgecolor, linewidth=1.5, linestyle='--',
                                                zorder=10, transform=ccrs.PlateCarree()))
                lon_min, lon_max, lat_min, lat_max = analysis_box_coords
                ax2.add_patch(mpatches.Rectangle((lon_min, lat_min), lon_max - lon_min, lat_max - lat_min,
                                            fill=False, edgecolor='lime', linewidth=2, linestyle='-',
                                            zorder=10, transform=ccrs.PlateCarree()))
            else:
                ax2.text(0.5, 0.5, "Data not available", transform=ax2.transAxes, ha='center', va='center')

            # --- Colorbar ---
            cax = fig.add_subplot(gs[row_idx, 2])
            
            final_label = config['base_label']
            std_dev_val = data_era5.get('std_dev_jet') if data_era5 else data_20crv3.get('std_dev_jet')
            if std_dev_val is not None and not np.isnan(std_dev_val):
                unit = "m/s" if "Speed" in config['title'] else "°Lat"
                final_label += f'\n(1 std. dev. = {std_dev_val:.2f} {unit})'
            
            if cf is not None:
                fig.colorbar(cf, cax=cax, extend='both', label=final_label)

            row_idx += 1
        
        plt.suptitle(f"Impact Maps of Jet Variations on Local Climate ({season}, Detrended)", fontsize=16, weight='bold')
        fig.tight_layout(rect=(0, 0, 0.95, 0.96))
        filename = os.path.join(Config.PLOT_DIR, f'jet_impact_regression_maps_{season.lower()}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)

    @staticmethod
    def plot_jet_indices_comparison(jet_data_reanalysis, filename="jet_indices_comparison_seasonal_detrended.png"):
        """
        Plots a comparison of detrended jet speed and latitude indices for Winter and Summer.
        Compares 20CRv3 and ERA5 reanalysis datasets and calculates the correlation
        between them in their overlapping period.
        """
        logging.info("Plotting comparison of detrended jet indices (Speed vs. Latitude)...")
        Visualizer.ensure_plot_dir_exists()

        fig, axs = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
        plot_configs = [
            {'ax': axs[0, 0], 'season': 'Winter', 'index_type': 'speed', 'ylabel': 'Jet Speed Anomaly (m/s)'},
            {'ax': axs[0, 1], 'season': 'Summer', 'index_type': 'speed', 'ylabel': 'Jet Speed Anomaly (m/s)'},
            {'ax': axs[1, 0], 'season': 'Winter', 'index_type': 'lat', 'ylabel': 'Jet Latitude Anomaly (°N)'},
            {'ax': axs[1, 1], 'season': 'Summer', 'index_type': 'lat', 'ylabel': 'Jet Latitude Anomaly (°N)'},
        ]
        
        all_years = []
        
        for config in plot_configs:
            ax: Any = config['ax']
            season_lower = str(config['season']).lower()
            
            # Store data for correlation calculation
            ts_data = {}
            
            for dataset_key, color in [(Config.DATASET_20CRV3, 'royalblue'), (Config.DATASET_ERA5, 'crimson')]:
                data_key = f"{dataset_key}_{season_lower}_{config['index_type']}_data"
                jet_bundle = jet_data_reanalysis.get(data_key)
                
                if jet_bundle and 'jet' in jet_bundle and jet_bundle['jet'] is not None:
                    jet_ts = jet_bundle['jet']
                    if 'season_year' in jet_ts.coords and jet_ts.size > 0:
                        years = jet_ts.season_year.values
                        values = jet_ts.values
                        all_years.extend(years)
                        
                        # Store for correlation
                        ts_data[dataset_key] = {'years': years, 'values': values}
                        
                        # Plot the timeseries
                        ax.plot(years, values, '-', color=color, linewidth=1.5, label=dataset_key)
            
            # --- Calculate and plot correlation for the overlap ---
            if Config.DATASET_20CRV3 in ts_data and Config.DATASET_ERA5 in ts_data:
                data_20crv3 = ts_data[Config.DATASET_20CRV3]
                data_era5 = ts_data[Config.DATASET_ERA5]
                
                # Find common years
                common_years, idx1, idx2 = np.intersect1d(data_20crv3['years'], data_era5['years'], return_indices=True)
                
                if len(common_years) > 5: # Require at least 5 years of overlap
                    values1 = data_20crv3['values'][idx1]
                    values2 = data_era5['values'][idx2]
                    
                    # Calculate regression to get r and p values
                    slope, intercept, r_value, p_value, std_err = StatsAnalyzer.calculate_regression(values1, values2)
                    
                    if not np.isnan(r_value):
                        # Add significance stars to p-value
                        if p_value < 0.01:
                            p_str = "***"
                        elif p_value < 0.05:
                            p_str = "**"
                        elif p_value < 0.1:
                            p_str = "*"
                        else:
                            p_str = ""
                            
                        corr_text = f"Overlap Corr: r = {r_value:.2f}{p_str}"
                        # Add text to the plot
                        ax.text(0.04, 0.92, corr_text, transform=ax.transAxes, fontsize=10,
                                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))

            ax.set_title(f"{config['season']} Jet {config['index_type'].capitalize()} Index (Detrended)")
            ax.set_ylabel(config['ylabel'])
            ax.grid(True, linestyle=':', alpha=0.7)
            ax.legend(loc='lower left')

        if all_years:
            fig.suptitle(f"Jet Stream Indices Comparison (Detrended)\n({int(min(all_years))}-{int(max(all_years))})", fontsize=16, weight='bold')
        else:
            fig.suptitle("Jet Stream Indices Comparison (Detrended)", fontsize=16, weight='bold')
            
        # Add X-axis labels to bottom plots
        axs[1, 0].set_xlabel("Year")
        axs[1, 1].set_xlabel("Year")

        # [KORREKTUR] fig.tight_layout() anstelle von plt.tight_layout()
        fig.tight_layout()
        filepath = os.path.join(Config.PLOT_DIR, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logging.info(f"Saved detrended jet indices comparison plot to {filepath}")
        
    @staticmethod
    def plot_correlation_timeseries_comparison(datasets_reanalysis, jet_data_reanalysis, discharge_data, season):
        """
        Erstellt einen Vergleichsplot mit Zeitreihen von Korrelationen zwischen verschiedenen Klimaindizes für eine bestimmte Saison.
        Diese Funktion ist eine Adaption der alten `plot_seasonal_correlations`-Funktion und ist für die neue Datenstruktur ausgelegt.
        """
        logging.info(f"Erstelle Plot für Zeitreihen-Korrelationen für {season}...")
        Visualizer.ensure_plot_dir_exists()

        season_lower = season.lower()
        if season_lower not in ['winter', 'summer']:
            logging.error(f"Ungültige Saison '{season}' für Korrelationsplot übergeben.")
            return

        # Konfiguration der Subplots, passend zu Ihren Beispielbildern
        plot_configs = [
            {'title': f'{season} Temp vs Jet Speed', 'var1_key': 'tas', 'var2_key': 'speed'},
            {'title': f'{season} Precip vs Jet Lat', 'var1_key': 'pr',  'var2_key': 'lat'},
            {'title': f'{season} Temp vs Jet Lat', 'var1_key': 'tas', 'var2_key': 'lat'},
            {'title': f'{season} Discharge vs Jet Speed',  'var1_key': 'discharge', 'var2_key': 'speed'},
            {'title': f'{season} Extreme Flow vs Jet Speed', 'var1_key': 'extreme_flow', 'var2_key': 'speed'},
            {'title': f'{season} Precip vs Jet Speed', 'var1_key': 'pr', 'var2_key': 'speed'},
        ]

        fig, axs = plt.subplots(2, 3, figsize=(18, 9))
        axs = axs.flatten()

        for i, config in enumerate(plot_configs):
            ax = axs[i]
            ax.set_title(config['title'])
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.set_xlabel("Year")
            if i % 3 == 0:  # Y-Achsen-Label nur für die linke Spalte
                ax.set_ylabel("Normalized Value (Detrended)")

            # Iteriere durch die Datensätze (20CRv3 und ERA5)
            for dataset_key, color in [(Config.DATASET_20CRV3, 'royalblue'), (Config.DATASET_ERA5, 'crimson')]:
                # --- Daten für Variable 1 holen ---
                var1_ts = None
                if config['var1_key'] in ['discharge', 'extreme_flow']:
                    # Abflussdaten sind für beide Reanalyse-Datensätze gleich
                    if discharge_data:
                        var1_ts = discharge_data.get(f"{season_lower}_{config['var1_key']}")
                else:  # pr oder tas
                    pr_tas_seasonal = datasets_reanalysis.get(f"{dataset_key}_{config['var1_key']}_box_mean")
                    if pr_tas_seasonal is not None:
                        var1_ts = DataProcessor.detrend_data(DataProcessor.filter_by_season(pr_tas_seasonal, season))

                # --- Daten für Variable 2 (Jet Index) holen ---
                jet_data_key = f"{dataset_key}_{season_lower}_{config['var2_key']}_data"
                var2_ts = jet_data_reanalysis.get(jet_data_key, {}).get('jet')

                if var1_ts is None or var2_ts is None or var1_ts.size == 0 or var2_ts.size == 0:
                    logging.debug(f"Daten für '{config['title']}' im Datensatz '{dataset_key}' nicht komplett. Überspringe.")
                    continue

                # --- Gemeinsame Jahre finden und Daten für die Korrelation vorbereiten ---
                common_years, idx1, idx2 = np.intersect1d(var1_ts.season_year.values, var2_ts.season_year.values, return_indices=True)
                if len(common_years) < 5:
                    continue

                vals1 = var1_ts.values[idx1]
                vals2 = var2_ts.values[idx2]

                # --- Normalisieren und Plotten ---
                vals1_norm = StatsAnalyzer.normalize(vals1)
                vals2_norm = StatsAnalyzer.normalize(vals2)

                var1_label = config['var1_key'].replace('_', ' ').title()
                var2_label = f"Jet {config['var2_key'].title()} Index"

                line1, = ax.plot(common_years, vals1_norm, '-', color=color, linewidth=1.2, alpha=0.9)
                line2, = ax.plot(common_years, vals2_norm, '--', color=color, linewidth=1.5)

                # Legende dynamisch erstellen, um Duplikate zu vermeiden
                if i == 0: # Beispielhaft Legende im ersten Plot hinzufügen
                    if dataset_key == Config.DATASET_20CRV3:
                        ax.legend([line1, line2], [var1_label, var2_label], loc='upper left', ncol=1, fontsize=7)

                # --- Korrelation berechnen und anzeigen ---
                _, _, r_val, p_val, _ = StatsAnalyzer.calculate_regression(vals2, vals1) # Jet (var2) als Prädiktor (X), Impakt (var1) als Y
                if not np.isnan(r_val):
                    p_str = ""
                    if p_val < 0.01: p_str = "***"
                    elif p_val < 0.05: p_str = "**"
                    elif p_val < 0.1: p_str = "*"

                    text_y = 0.95 if dataset_key == Config.DATASET_20CRV3 else 0.85
                    ax.text(0.03, text_y, f"{dataset_key}: r={r_val:.2f}{p_str}", transform=ax.transAxes,
                            fontsize=9, color=color, weight='bold', bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))

        fig.suptitle(f"{season} Correlations with Jet Stream Indices: 20CRv3 vs ERA5 (Detrended)", fontsize=16, weight='bold')
        # [KORREKTUR] fig.tight_layout() anstelle von plt.tight_layout()
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        filename = os.path.join(Config.PLOT_DIR, f'{season_lower}_correlations_comparison_detrended.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logging.info(f"Korrelations-Zeitreihenplot für {season} gespeichert unter: {filename}")

    @staticmethod
    def plot_danube_box_correlation(datasets_reanalysis, discharge_data):
        """
        Creates a correlation plot between Danube flow (obs) and TAS/PR indices in the analysis box.
        Layout: 2x2 Grid (Winter/Summer x Temperature/Precipitation).
        """
        logging.info("Creating Danube Flow vs TAS/PR Correlation Plot...")
        Visualizer.ensure_plot_dir_exists()

        seasons = ['Winter', 'Summer']
        variables = ['tas', 'pr']
        
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        
        for row_idx, var_key in enumerate(variables):
            for col_idx, season in enumerate(seasons):
                ax = axs[row_idx, col_idx]
                season_lower = season.lower()
                
                # Get Discharge Data (QOBS)
                discharge_ts = discharge_data.get(f'{season_lower}_discharge')
                if discharge_ts is None:
                    logging.warning(f"No discharge data found for {season}. Skipping subplot.")
                    continue

                # Get Climate Index Data (ERA5)
                dataset_key = Config.DATASET_ERA5
                climate_ts_full = datasets_reanalysis.get(f"{dataset_key}_{var_key}_box_mean")
                
                if climate_ts_full is None:
                    logging.warning(f"No {var_key} data found for {dataset_key}. Skipping subplot.")
                    continue
                    
                # Filter season and detrend
                climate_ts_season = DataProcessor.filter_by_season(climate_ts_full, season)
                climate_ts_detrended = DataProcessor.detrend_data(climate_ts_season)
                
                if climate_ts_detrended is None:
                    logging.warning(f"Could not detrend {var_key} for {season}. Skipping.")
                    continue

                # Intersect Years
                common_years, idx_d, idx_c = np.intersect1d(
                    discharge_ts.season_year.values, 
                    climate_ts_detrended.season_year.values, 
                    return_indices=True
                )
                
                if len(common_years) < 10:
                    logging.warning(f"Not enough overlapping years for {season} {var_key}. Skipping.")
                    continue
                    
                val_discharge = discharge_ts.values[idx_d]
                val_climate = climate_ts_detrended.values[idx_c]
                
                # Normalize
                val_d_norm = StatsAnalyzer.normalize(val_discharge)
                val_c_norm = StatsAnalyzer.normalize(val_climate)
                
                # Calculate Correlation
                _, _, r_val, p_val, _ = StatsAnalyzer.calculate_regression(val_climate, val_discharge)
                
                # Plot
                color_discharge = 'black'
                color_climate = 'red' if var_key == 'tas' else 'blue'
                label_climate = 'Temperature' if var_key == 'tas' else 'Precipitation'
                
                ax.plot(common_years, val_d_norm, color=color_discharge, label='Danube Discharge (Obs)', linewidth=1.5)
                ax.plot(common_years, val_c_norm, color=color_climate, label=f'Box {label_climate} (ERA5)', linestyle='--', linewidth=1.2)
                
                # Style
                ax.set_title(f"{season} {label_climate} vs Discharge")
                ax.grid(True, linestyle=':', alpha=0.6)
                if row_idx == 1:
                    ax.set_xlabel("Year")
                if col_idx == 0:
                    ax.set_ylabel("Normalized Anomalies")
                    
                # Annotate Correlation
                p_str = ""
                if p_val < 0.01: p_str = "***"
                elif p_val < 0.05: p_str = "**"
                elif p_val < 0.1: p_str = "*"
                
                ax.text(0.05, 0.90, f"r = {r_val:.2f}{p_str}", transform=ax.transAxes, 
                        fontsize=12, fontweight='bold', 
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', boxstyle='round,pad=0.3'))
                
            # Remove individual legend
            # ax.legend(loc='lower left', fontsize=9)

        # Add shared legend at the bottom
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='black', lw=1.5, label='Danube Discharge (Obs)'),
            Line2D([0], [0], color='red', lw=1.2, linestyle='--', label='Box Temperature (ERA5)'),
            Line2D([0], [0], color='blue', lw=1.2, linestyle='--', label='Box Precipitation (ERA5)')
        ]
        
        fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.02), 
                   ncol=3, fontsize=10, frameon=False)

        fig.suptitle("Correlation: Danube Flow (Obs) vs. Analysis Box Climate Indices (ERA5) [Detrended]", fontsize=16)
        # Adjust layout to make room for legend at bottom
        plt.tight_layout(rect=(0, 0.08, 1, 0.95))
        
        filename = os.path.join(Config.PLOT_DIR, "danube_box_correlation_comparison.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logging.info(f"Saved Danube box correlation plot to {filename}")

    @staticmethod
    def plot_correlation_bar_chart(correlation_df, season):
        """
        Creates a visually improved, grouped horizontal bar chart of correlation coefficients.
        This version addresses feedback by grouping bars by analysis type for clarity
        and using color to distinguish datasets.
        """
        if correlation_df.empty:
            logging.warning(f"Cannot plot correlation bar chart for {season}: DataFrame is empty.")
            return

        logging.info(f"Plotting improved correlation bar chart for {season}...")
        Visualizer.ensure_plot_dir_exists()

        # --- 1. Datenvorbereitung ---
        df = correlation_df.copy()
        df['abs_correlation'] = df['correlation'].abs()
        
        # Bestimme die Reihenfolge der Gruppen auf der Y-Achse nach der mittleren Korrelationsstärke
        group_order = df.groupby('base_label')['abs_correlation'].mean().sort_values(ascending=True).index
        df['sort_order'] = pd.Categorical(df['base_label'], categories=group_order, ordered=True)
        df = df.sort_values('sort_order')

        unique_labels = df['base_label'].unique()
        y_pos = np.arange(len(unique_labels)) # Position für jede Gruppe
        bar_height = 0.35  # Höhe jedes einzelnen Balkens
        
        # --- 2. Plot-Setup ---
        # Dynamische Höhe basierend auf der Anzahl der Analysen
        fig_height = len(unique_labels) * 0.7 + 2 
        fig, ax = plt.subplots(figsize=(12, fig_height))
        
        dataset_colors = {Config.DATASET_20CRV3: 'royalblue', Config.DATASET_ERA5: 'crimson'}

        # --- 3. Balken und Text-Labels plotten ---
        for i, label in enumerate(unique_labels):
            group_data = df[df['base_label'] == label]

            # Funktion zum Hinzufügen von Text-Labels
            def add_value_label(dataset_name, y_position, color):
                data = group_data[group_data['dataset'] == dataset_name]
                if not data.empty:
                    corr = data['correlation'].iloc[0]
                    p_val = data['p_value'].iloc[0]
                    
                    # Balken zeichnen
                    ax.barh(y_position, corr, height=bar_height, color=color, 
                            edgecolor='black', linewidth=0.5, label=dataset_name)
                    
                    # Signifikanz-Sterne
                    stars = ""
                    if p_val < 0.001: stars = "***"
                    elif p_val < 0.01: stars = "**"
                    elif p_val < 0.05: stars = "*"
                    
                    # Text-Positionierung
                    ha = 'left' if corr >= 0 else 'right'
                    offset = 0.01
                    x_pos = corr + offset if corr >= 0 else corr - offset
                    
                    ax.text(x_pos, y_position, f" {corr:.2f}{stars}", 
                            ha=ha, va='center', fontsize=9, weight='bold', color=color)

            # Balken für 20CRv3 (oben in der Gruppe)
            add_value_label(Config.DATASET_20CRV3, y_pos[i] + bar_height / 2, dataset_colors[Config.DATASET_20CRV3])
            
            # Balken für ERA5 (unten in der Gruppe)
            add_value_label(Config.DATASET_ERA5, y_pos[i] - bar_height / 2, dataset_colors[Config.DATASET_ERA5])

        # --- 4. Achsen, Legende und finale Formatierung ---
        ax.set_yticks(y_pos)
        ax.set_yticklabels(unique_labels, fontsize=11)
        ax.set_xlabel('Correlation Coefficient (r)', fontsize=12)
        ax.set_title(f'{season} Correlation Analysis: 20CRv3 vs ERA5 (Detrended)', fontsize=16, weight='bold', pad=20)

        # Gitternetz und Achsenlinien
        ax.grid(axis='x', linestyle=':', alpha=0.7)
        ax.axvline(0, color='black', linewidth=0.8)
        
        # X-Achsen-Limits anpassen für mehr Platz
        current_xlim = ax.get_xlim()
        new_lim = max(abs(l) for l in current_xlim) * 1.15
        ax.set_xlim(-new_lim, new_lim)

        # Legende erstellen (ohne Duplikate)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='lower right', fontsize=10, title="Datasets", title_fontsize=11)
        
        # Signifikanz-Erklärung
        plt.figtext(0.5, 0.01, "* p < 0.05, ** p < 0.01, *** p < 0.001", ha="center", fontsize=10)

        # [KORREKTUR] fig.tight_layout() anstelle von plt.tight_layout()
        fig.tight_layout(rect=(0, 0.05, 1, 0.95))
        filename = os.path.join(Config.PLOT_DIR, f'correlation_matrix_comparison_{season.lower()}_detrended_grouped.png')
        plt.savefig(filename, dpi=300)
        plt.close(fig)
        logging.info(f"Saved improved correlation bar chart for {season} to {filename}")

    @staticmethod
    def plot_amo_jet_correlation_comparison(correlation_data, window_size=15):
        """
        Creates a 2x2 comparison plot of AMO vs Jet Indices for Winter and Summer.
        """
        logging.info(f"Plotting 2x2 AMO-Jet correlation comparison ({window_size}-yr rolling mean)...")
        Visualizer.ensure_plot_dir_exists()

        if not correlation_data or not any(correlation_data.values()):
            logging.warning("Cannot plot AMO-Jet comparison: correlation_data is empty.")
            return

        fig, axs = plt.subplots(2, 2, figsize=(18, 10), sharex=True, sharey=True)
        axs = axs.flatten()
        
        plot_configs = [
            {'ax_idx': 0, 'season': 'Winter', 'jet_type': 'speed', 'title': 'AMO vs Jet Speed Index (Winter)'},
            {'ax_idx': 1, 'season': 'Winter', 'jet_type': 'lat', 'title': 'AMO vs Jet Latitude Index (Winter)'},
            {'ax_idx': 2, 'season': 'Summer', 'jet_type': 'speed', 'title': 'AMO vs Jet Speed Index (Summer)'},
            {'ax_idx': 3, 'season': 'Summer', 'jet_type': 'lat', 'title': 'AMO vs Jet Latitude Index (Summer)'},
        ]

        colors = {Config.DATASET_20CRV3: 'royalblue', Config.DATASET_ERA5: 'crimson', 'AMO': 'black'}
        # Map keys from our plot config to the keys in the data dictionary
        jet_type_map = {'speed': 'speed', 'lat': 'latitude'}

        for config in plot_configs:
            ax_idx = int(config['ax_idx'])
            jet_type = str(config['jet_type'])
            season_key = str(config['season'])
            title_key = str(config['title'])
            
            ax = axs[ax_idx]
            season_data = correlation_data.get(season_key)
            
            ax.set_title(f"{title_key}, {window_size}-yr mean")
            ax.grid(True, linestyle=':', alpha=0.6)

            if not season_data:
                ax.text(0.5, 0.5, "Data Not Available", transform=ax.transAxes, ha='center', va='center')
                continue

            plotted_amo = False
            for dataset_key in [Config.DATASET_20CRV3, Config.DATASET_ERA5]:
                # The data structure from analyze_amo_jet_correlations is {'20CRv3': {'speed': {...}, 'latitude': {...}}}
                mapped_jet_key = jet_type_map.get(jet_type, jet_type)
                jet_data = season_data.get(dataset_key, {}).get(mapped_jet_key)
                
                if jet_data and 'amo_values' in jet_data and 'jet_values' in jet_data:
                    # Normalize data for consistent plotting scale
                    amo_norm = StatsAnalyzer.normalize(jet_data['amo_values'])
                    jet_norm = StatsAnalyzer.normalize(jet_data['jet_values'])
                    years = jet_data['common_years']

                    # Plot AMO series (only once per subplot)
                    if not plotted_amo:
                        ax.plot(years, amo_norm, '-', color=colors['AMO'], linewidth=1.5, label=f'AMO ({window_size}yr)')
                        plotted_amo = True

                    # Plot jet series
                    label = f"{dataset_key} Jet {jet_type.capitalize()} ({window_size}yr)"
                    ax.plot(years, jet_norm, '--', color=colors[dataset_key], linewidth=2.0, label=label)

                    # Add correlation text
                    r_val, p_val = jet_data.get('r_value'), jet_data.get('p_value')
                    if r_val is not None:
                        stars = ""
                        if p_val is not None:
                            if p_val < 0.01: stars = "***"
                            elif p_val < 0.05: stars = "**"
                            elif p_val < 0.1: stars = "*"
                        
                        text_y = 0.95 if dataset_key == Config.DATASET_20CRV3 else 0.85
                        ax.text(0.03, text_y, f"{dataset_key}: r={r_val:.2f}{stars}",
                                transform=ax.transAxes, fontsize=10, color=colors[dataset_key],
                                weight='bold', bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))

            # Set labels and legends
            if config['ax_idx'] in [0, 2]:
                ax.set_ylabel("Normalized Value (Smoothed, Detrended)")
            if config['ax_idx'] in [2, 3]:
                ax.set_xlabel("Year")
            
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                by_label = dict(zip(labels, handles))
                ax.legend(by_label.values(), by_label.keys(), loc='lower left', fontsize=8)

        fig.suptitle('Relationship Between AMO Index and Jet Stream Indices (Detrended)', fontsize=16, weight='bold')
        # [KORREKTUR] fig.tight_layout() anstelle von plt.tight_layout()
        fig.tight_layout(rect=(0, 0.03, 1, 0.96))
        filename = os.path.join(Config.PLOT_DIR, f'amo_jet_correlations_comparison_rolling_{window_size}yr.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logging.info(f"Saved AMO vs Jet correlation comparison plot to {filename}")

    @staticmethod
    def plot_climate_projection_timeseries(cmip6_plot_data, reanalysis_plot_data, config, filename="climate_indices_evolution.png", window_size=20):
        """
        Plots CMIP6 and Reanalysis changes over time, showing the evolution of key climate indices.
        Original 2x2 layout for backward compatibility.
        """
        logging.info(f"Plotting climate projection timeseries comparison to {filename}...")
        Visualizer.ensure_plot_dir_exists()

        # Original Layout: 2x2 Grid
        # Row 1: Summer Jet Lat, Summer Jet Speed
        # Row 2: Winter Jet Lat, Winter Jet Speed
        # (Note: Global Temp was likely a separate plot or not in this specific function originally, 
        # but to be safe, we'll just plot the 4 jet indices as that's what the data structure suggests)
        
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        axs = axs.flatten()

        plot_configs = [
            {'key': 'JJA_JetLat',   'ax': axs[0], 'title': 'Summer (JJA) Jet Latitude', 'ylabel': 'Latitude Anomaly (°)'},
            {'key': 'JJA_JetSpeed', 'ax': axs[1], 'title': 'Summer (JJA) Jet Speed',    'ylabel': 'Speed Anomaly (m/s)'},
            {'key': 'DJF_JetLat',   'ax': axs[2], 'title': 'Winter (DJF) Jet Latitude', 'ylabel': 'Latitude Anomaly (°)'},
            {'key': 'DJF_JetSpeed', 'ax': axs[3], 'title': 'Winter (DJF) Jet Speed',    'ylabel': 'Speed Anomaly (m/s)'},
        ]

        for p_config in plot_configs:
            ax: Any = p_config['ax']
            key = str(p_config['key'])
            title_str = str(p_config['title'])
            ylabel_str = str(p_config['ylabel'])

            if cmip6_plot_data.get(key) and cmip6_plot_data[key]['members']:
                for member_jet in cmip6_plot_data[key]['members']:
                    ax.plot(member_jet.season_year, member_jet, color='grey', alpha=0.3, linewidth=0.7)
            if cmip6_plot_data.get(key) and cmip6_plot_data[key]['mmm'] is not None:
                ax.plot(cmip6_plot_data[key]['mmm'].season_year, cmip6_plot_data[key]['mmm'], color='black', linewidth=2.5, label='CMIP6 MMM')
            
            if reanalysis_plot_data.get(key) and reanalysis_plot_data[key].get('20CRv3') is not None:
                reanalysis_20crv3 = reanalysis_plot_data[key]['20CRv3']
                ax.plot(reanalysis_20crv3.season_year, reanalysis_20crv3, color='darkorange', linewidth=2, label='20CRv3')
            if reanalysis_plot_data.get(key) and reanalysis_plot_data[key].get('ERA5') is not None:
                reanalysis_era5 = reanalysis_plot_data[key]['ERA5']
                ax.plot(reanalysis_era5.season_year, reanalysis_era5, color='purple', linewidth=2, label='ERA5')

            ax.set_title(title_str, fontsize=12, weight='bold', loc='left')
            ax.set_ylabel(ylabel_str, fontsize=10)
            ax.grid(True, linestyle=':', alpha=0.7)
            ax.set_xlim(1850, 2100)
            ax.axhline(0, color='black', linewidth=0.5)
            if ax in [axs[2], axs[3]]: # Bottom row
                ax.set_xlabel('Year', fontsize=10)

        fig.suptitle(f'Evolution of Key Climate Indices ({window_size}-Year Rolling Mean)', fontsize=16, weight='bold')
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        filepath = os.path.join(config.PLOT_DIR, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logging.info(f"Saved climate projection timeseries plot to {filepath}")

    @staticmethod
    def plot_final_figure_5_climate_projection_timeseries(cmip6_plot_data, reanalysis_plot_data, config, scenario='ssp585', window_size=20):
        """
        Creates Final Figure 5: Future Dynamical Uncertainty (Climate Indices Evolution).
        Layout: 3 Rows (Global Temp, Summer Jet, Winter Jet).
        - Unified Y-AXIS LIMITS per variable type (Lat/Speed).
        - Legend closer to plots, no frame.
        """
        filename = f"final_figure_5_climate_indices_evolution_{scenario}.png"
        logging.info(f"Plotting Final Figure 5 to {filename}...")
        Visualizer.ensure_plot_dir_exists()

        # --- MODIFIED LAYOUT: 3 Rows ---
        fig = plt.figure(figsize=(5.9, 10.0))
        
        # Add Main Title with Scenario
        scenario_title = Visualizer._format_scenario_title(scenario)
        # [MOVED/REMOVED] Title set at the end

        gs = gridspec.GridSpec(3, 2, height_ratios=[0.6, 1, 1], hspace=0.5, top=0.88)

        # --- (a) Global Temperature Anomaly ---
        ax_a = fig.add_subplot(gs[0, :]) # Span both columns
        if cmip6_plot_data.get('Global_Tas') and cmip6_plot_data['Global_Tas']['members']:
            for member_tas in cmip6_plot_data['Global_Tas']['members']:
                ax_a.plot(member_tas.year, member_tas, color='grey', alpha=0.3, linewidth=0.7, linestyle='-')
        if cmip6_plot_data.get('Global_Tas') and cmip6_plot_data['Global_Tas']['mmm'] is not None:
            ax_a.plot(cmip6_plot_data['Global_Tas']['mmm'].year, cmip6_plot_data['Global_Tas']['mmm'], 
                      color='black', linewidth=2.5, linestyle='-', label='MMM')
        
        # Reanalysis for Global Temp
        if reanalysis_plot_data.get('Global_Tas'):
            if reanalysis_plot_data['Global_Tas'].get('20CRv3') is not None:
                reanalysis_20crv3_tas = reanalysis_plot_data['Global_Tas']['20CRv3']
                ax_a.plot(reanalysis_20crv3_tas.year, reanalysis_20crv3_tas, color='darkorange', linewidth=1.5, linestyle='-', label='20CRv3')
            if reanalysis_plot_data['Global_Tas'].get('ERA5') is not None:
                reanalysis_era5_tas = reanalysis_plot_data['Global_Tas']['ERA5']
                ax_a.plot(reanalysis_era5_tas.year, reanalysis_era5_tas, color='purple', linewidth=1.5, linestyle='-', label='ERA5')
        
        ax_a.set_title('(a) Global Temperature Change (°C)', weight='bold', loc='left')
        ax_a.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=8))
        ax_a.grid(True, linestyle=':', alpha=0.6)
        ax_a.set_xlim(1850, 2100)
        ax_a.axhline(0, color='black', linewidth=0.5)

        # --- Helper to Calculate Unified Y-Limits ---
        def get_unified_limits(keys):
            min_val, max_val = np.inf, -np.inf
            has_data = False
            for key in keys:
                # Check CMIP6 Members
                if cmip6_plot_data.get(key) and cmip6_plot_data[key]['members']:
                    for m in cmip6_plot_data[key]['members']:
                        if m is not None and m.size > 0:
                            vals = m.values if hasattr(m, 'values') else m
                            min_val = min(min_val, np.nanmin(vals))
                            max_val = max(max_val, np.nanmax(vals))
                            has_data = True
                # Check CMIP6 MMM
                if cmip6_plot_data.get(key) and cmip6_plot_data[key]['mmm'] is not None:
                    mmm = cmip6_plot_data[key]['mmm']
                    if mmm.size > 0:
                        vals = mmm.values if hasattr(mmm, 'values') else mmm
                        min_val = min(min_val, np.nanmin(vals))
                        max_val = max(max_val, np.nanmax(vals))
                        has_data = True
                # Check Reanalysis
                if reanalysis_plot_data.get(key):
                    for dset in reanalysis_plot_data[key]:
                        data = reanalysis_plot_data[key][dset]
                        if data is not None and data.size > 0:
                            vals = data.values if hasattr(data, 'values') else data
                            min_val = min(min_val, np.nanmin(vals))
                            max_val = max(max_val, np.nanmax(vals))
                            has_data = True
            
            if not has_data or np.isinf(min_val) or np.isinf(max_val):
                return None
            
            # Add 5% padding
            range_val = max_val - min_val
            if range_val == 0: range_val = 1.0
            return (min_val - 0.05 * range_val, max_val + 0.05 * range_val)

        # Calculate limits for Latitude and Speed
        lat_ylim = get_unified_limits(['JJA_JetLat', 'DJF_JetLat'])
        speed_ylim = get_unified_limits(['JJA_JetSpeed', 'DJF_JetSpeed'])

        # --- Configuration for Jet Indices ---
        plot_configs = [
            {'key': 'JJA_JetLat',   'ax': fig.add_subplot(gs[1, 0]), 'title': '(b) Summer Jet Latitude', 'ylabel': 'Lat. Anom. (°)', 'ylim': lat_ylim},
            {'key': 'JJA_JetSpeed', 'ax': fig.add_subplot(gs[1, 1]), 'title': '(c) Summer Jet Speed',    'ylabel': 'Speed Anom. (m/s)', 'ylim': speed_ylim},
            {'key': 'DJF_JetLat',   'ax': fig.add_subplot(gs[2, 0]), 'title': '(d) Winter Jet Latitude', 'ylabel': 'Lat. Anom. (°)', 'ylim': lat_ylim},
            {'key': 'DJF_JetSpeed', 'ax': fig.add_subplot(gs[2, 1]), 'title': '(e) Winter Jet Speed',    'ylabel': 'Speed Anom. (m/s)', 'ylim': speed_ylim},
        ]

        for p_config in plot_configs:
            ax: Any = p_config['ax']
            key = str(p_config['key'])
            title_str = str(p_config['title'])
            ylabel_str = str(p_config['ylabel'])

            if cmip6_plot_data.get(key) and cmip6_plot_data[key]['members']:
                for member_jet in cmip6_plot_data[key]['members']:
                    ax.plot(member_jet.season_year, member_jet, color='grey', alpha=0.3, linewidth=0.7, linestyle='-')
            if cmip6_plot_data.get(key) and cmip6_plot_data[key]['mmm'] is not None:
                ax.plot(cmip6_plot_data[key]['mmm'].season_year, cmip6_plot_data[key]['mmm'], 
                        color='black', linewidth=2.5, linestyle='-', label='MMM')
            
            if reanalysis_plot_data.get(key) and reanalysis_plot_data[key].get('20CRv3') is not None:
                reanalysis_20crv3 = reanalysis_plot_data[key]['20CRv3']
                ax.plot(reanalysis_20crv3.season_year, reanalysis_20crv3, color='darkorange', linewidth=1.5, linestyle='-', label='20CRv3')
            if reanalysis_plot_data.get(key) and reanalysis_plot_data[key].get('ERA5') is not None:
                reanalysis_era5 = reanalysis_plot_data[key]['ERA5']
                ax.plot(reanalysis_era5.season_year, reanalysis_era5, color='purple', linewidth=1.5, linestyle='-', label='ERA5')

            ax.set_title(title_str, weight='bold', loc='left')
            ax.set_ylabel(ylabel_str)
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.set_xlim(1850, 2100)
            ax.axhline(0, color='black', linewidth=0.5)

            # Apply unified Y-limits if available
            if p_config.get('ylim'):
                ax.set_ylim(p_config['ylim'])

            if p_config['ax'] in [plot_configs[2]['ax'], plot_configs[3]['ax']]: # Bottom row
                ax.set_xlabel('Year')

        # Shared Legend at the bottom
        handles, labels = ax_a.get_legend_handles_labels()
        
        final_handles = []
        final_labels = []
        
        # Order: Multi-Model Mean, CMIP6 Models, 20CRv3, ERA5
        if 'Multi-Model Mean' in labels:
            idx = labels.index('Multi-Model Mean')
            final_handles.append(handles[idx])
            final_labels.append(labels[idx])
        
        final_handles.append(plt.Line2D([0], [0], color='grey', linewidth=0.7, alpha=0.5, linestyle='-', label='CMIP6 Models'))
        final_labels.append('CMIP6 Models')
        
        if '20CRv3' in labels:
            idx = labels.index('20CRv3')
            final_handles.append(handles[idx])
            final_labels.append(labels[idx])
        if 'ERA5' in labels:
            idx = labels.index('ERA5')
            final_handles.append(handles[idx])
            final_labels.append(labels[idx])

        # KORREKTUR: Legende etwas nach unten verschoben und ohne Rahmen
        fig.legend(final_handles, final_labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.02), frameon=False)

        main_title = f'Climate Indices Evolution ({window_size}-yr mean) — {scenario_title}'
        ref_text = "All changes relative to 1850–1900 mean"
        fig.suptitle(f'{main_title}\n{ref_text}', fontsize=15, weight='bold', y=0.98)
        
        # Layout angepasst für engere Legende und mehr Platz am unteren Rand (rect[1] erhöht)
        fig.tight_layout(rect=(0, 0.08, 1, 0.93), h_pad=3.0, w_pad=2.5)
        
        filepath = os.path.join(config.PLOT_DIR, filename)
        plt.savefig(filepath, dpi=600, bbox_inches='tight')
        pdf_filepath = os.path.join(config.PLOT_DIR, filename.replace('.png', '.pdf'))
        plt.savefig(pdf_filepath, bbox_inches='tight')
        plt.close(fig)
        logging.info(f"Saved Final Figure 5 to {filepath} and {pdf_filepath}")
        
    @staticmethod
    def _plot_single_scatter_panel(ax, cmip6_results, beta_obs_slopes, gwl_to_plot,
                                   jet_key, impact_key, beta_key, title):
        """Helper function to draw one panel of the CMIP6 scatter comparison plot."""
        
        all_deltas = cmip6_results['all_individual_model_deltas_for_plot']
        mmm_changes = cmip6_results['mmm_changes']
        
        # Extract delta values for the specific jet and impact variable
        jet_deltas = all_deltas.get(jet_key, {}).get(gwl_to_plot, {})
        impact_deltas = all_deltas.get(impact_key, {}).get(gwl_to_plot, {})
        
        if not jet_deltas or not impact_deltas:
            ax.text(0.5, 0.5, "Data Missing", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title, fontsize=10)
            return

        # Align data using model keys
        models = sorted(list(set(jet_deltas.keys()) & set(impact_deltas.keys())))
        jet_vals = np.array([jet_deltas[m] for m in models])
        impact_vals = np.array([impact_deltas[m] for m in models])

        # Plot scatter of individual models
        ax.scatter(jet_vals, impact_vals, color='grey', alpha=0.6, s=25, label=f'CMIP6 Models (N={len(models)})')

        # Plot CMIP6 inter-model regression fit
        slope_cmip6, intercept_cmip6, _, _, _ = StatsAnalyzer.calculate_regression(jet_vals, impact_vals)
        x_fit = np.array(ax.get_xlim())
        if not np.isnan(slope_cmip6):
            y_fit = intercept_cmip6 + slope_cmip6 * x_fit
            ax.plot(x_fit, y_fit, color='black', linestyle='-', linewidth=2, label=f'CMIP6 Fit (Slope={slope_cmip6:.2f})')

        # Plot observed IAV slope (beta_obs) anchored at the MMM
        beta_obs = beta_obs_slopes.get(beta_key)
        delta_jet_mmm = mmm_changes.get(gwl_to_plot, {}).get(jet_key)
        delta_impact_mmm = mmm_changes.get(gwl_to_plot, {}).get(impact_key)

        if beta_obs is not None and delta_jet_mmm is not None and delta_impact_mmm is not None:
            intercept_iav = delta_impact_mmm - beta_obs * delta_jet_mmm
            y_fit_iav = intercept_iav + beta_obs * x_fit
            # --- START DER ÄNDERUNG ---
            # Hier wird die Legende von "Obs." auf "ERA5" geändert
            ax.plot(x_fit, y_fit_iav, color='red', linestyle='--', linewidth=2, label=f'ERA5 IAV Slope ($\\beta_{{obs}}$={beta_obs:.2f})')
            # --- ENDE DER ÄNDERUNG ---

        # Formatting
        # --- START DER ÄNDERUNG ---
        # Hier werden die Achsenbeschriftungen präzisiert
        x_label = f'Change in {jet_key.replace("_", " ")}'
        # Hier wird die Referenz zur "Box" in der Y-Achsen-Beschriftung hinzugefügt
        y_label = f'Change in {impact_key.replace("_", " ")} over Box'
        
        if "Lat" in jet_key: x_label += ' (°Lat)'
        elif "Speed" in jet_key: x_label += ' (m/s)'
        
        if "_pr" in impact_key: y_label += ' (%)'
        elif "_tas" in impact_key: y_label += ' (°C)'
        # --- ENDE DER ÄNDERUNG ---
        
        ax.set_xlabel(x_label, fontsize=9)
        ax.set_ylabel(y_label, fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.axhline(0, color='grey', lw=0.7); ax.axvline(0, color='grey', lw=0.7)
        ax.legend(fontsize=8)

    @staticmethod
    def plot_cmip6_scatter_comparison(cmip6_results, beta_obs_slopes, gwl_to_plot, scenario):
        """
        Creates a 2x4 subplot figure comparing CMIP6 projected changes for all 8 combinations
        of jet indices and impact variables at a specific Global Warming Level.
        MODIFIED: Accepts a scenario parameter for filename and title.
        """
        if not cmip6_results or not beta_obs_slopes:
            logging.warning(f"Cannot plot CMIP6 scatter comparison for {scenario}: Missing results or beta slopes.")
            return
            
        logging.info(f"Plotting expanded 2x4 CMIP6 scatter comparison for {gwl_to_plot}°C GWL (Scenario: {scenario})...")
        Visualizer.ensure_plot_dir_exists()

        # Change subplot layout to 2x4 and adjust figsize to be wider
        fig, axs = plt.subplots(2, 4, figsize=(24, 12))
        
        # Define all 8 plot configurations
        plot_configs = [
            # Row 1: Temperature Impacts
            {'ax': axs[0, 0], 'jet_key': 'DJF_JetSpeed', 'impact_key': 'DJF_tas', 'beta_key': 'DJF_JetSpeed_vs_tas', 'title': 'Winter Temp vs. Jet Speed'},
            {'ax': axs[0, 1], 'jet_key': 'DJF_JetLat',   'impact_key': 'DJF_tas', 'beta_key': 'DJF_JetLat_vs_tas',   'title': 'Winter Temp vs. Jet Latitude'},
            {'ax': axs[0, 2], 'jet_key': 'JJA_JetSpeed', 'impact_key': 'JJA_tas', 'beta_key': 'JJA_JetSpeed_vs_tas', 'title': 'Summer Temp vs. Jet Speed'},
            {'ax': axs[0, 3], 'jet_key': 'JJA_JetLat',   'impact_key': 'JJA_tas', 'beta_key': 'JJA_JetLat_vs_tas',   'title': 'Summer Temp vs. Jet Latitude'},
            # Row 2: Precipitation Impacts
            {'ax': axs[1, 0], 'jet_key': 'DJF_JetSpeed', 'impact_key': 'DJF_pr',  'beta_key': 'DJF_JetSpeed_vs_pr',  'title': 'Winter Precip vs. Jet Speed'},
            {'ax': axs[1, 1], 'jet_key': 'DJF_JetLat',   'impact_key': 'DJF_pr',  'beta_key': 'DJF_JetLat_vs_pr',    'title': 'Winter Precip vs. Jet Latitude'},
            {'ax': axs[1, 2], 'jet_key': 'JJA_JetSpeed', 'impact_key': 'JJA_pr',  'beta_key': 'JJA_JetSpeed_vs_pr',  'title': 'Summer Precip vs. Jet Speed'},
            {'ax': axs[1, 3], 'jet_key': 'JJA_JetLat',   'impact_key': 'JJA_pr',  'beta_key': 'JJA_JetLat_vs_pr',    'title': 'Summer Precip vs. Jet Latitude'},
        ]

        for config in plot_configs:
            # The helper function _plot_single_scatter_panel is generic and can be reused
            Visualizer._plot_single_scatter_panel(
                ax=config['ax'],
                cmip6_results=cmip6_results,
                beta_obs_slopes=beta_obs_slopes,
                gwl_to_plot=gwl_to_plot,
                jet_key=config['jet_key'],
                impact_key=config['impact_key'],
                beta_key=config['beta_key'],
                title=config['title']
            )
        
        ref_period_changes = f"{Config.CMIP6_ANOMALY_REF_START}-{Config.CMIP6_ANOMALY_REF_END}"
        ref_period_gwl = f"{Config.CMIP6_PRE_INDUSTRIAL_REF_START}-{Config.CMIP6_PRE_INDUSTRIAL_REF_END}"
        # MODIFIED: Add scenario to the title
        fig.suptitle(f"CMIP6 Projected Changes at {gwl_to_plot}°C GWL for {scenario.upper()}\n"
                     f"(Changes relative to {ref_period_changes}; GWL defined relative to {ref_period_gwl})",
                     fontsize=16, weight='bold') 
        
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        # MODIFIED: Add scenario to the filename to make it unique
        filename = os.path.join(Config.PLOT_DIR, f"cmip6_scatter_comparison_gwl_{gwl_to_plot:.1f}_{scenario}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logging.info(f"Saved expanded CMIP6 scatter comparison plot for {scenario} to {filename}")

    @staticmethod
    def _plot_single_jet_relationship_panel(ax, cmip6_results, gwl_to_plot, x_jet_key, y_jet_key, title,
                                        inner_radius=None):
        """
        Helper-Funktion zum Zeichnen eines Panels, das die Beziehung zwischen Jet-Indizes darstellt.
        Stellt jetzt die Mittelwerte der Extrem-Quadranten (Kreuze), die Klassifikations-Zonen
        der axialen Storylines (Ellipsen) und die beiden Eck-Extreme (Sterne) dar.
        """
        all_deltas = cmip6_results.get('all_individual_model_deltas_for_plot', {})
        storyline_classification = cmip6_results.get('storyline_classification_2d', {})
        gwl_classification = storyline_classification.get(gwl_to_plot, {})
        season_prefix = x_jet_key.split('_')[0]

        x_deltas = all_deltas.get(x_jet_key, {}).get(gwl_to_plot, {})
        y_deltas = all_deltas.get(y_jet_key, {}).get(gwl_to_plot, {})
        if not x_deltas or not y_deltas:
            ax.text(0.5, 0.5, "Data Missing", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title, fontsize=10); return

        common_models = sorted(list(set(x_deltas.keys()) & set(y_deltas.keys())))
        x_vals = np.array([x_deltas[m] for m in common_models])
        y_vals = np.array([y_deltas[m] for m in common_models])

        ax.scatter(x_vals, y_vals, color='teal', alpha=0.5, s=25, label=f'CMIP6 Models (N={len(common_models)})')
        
        slope, intercept, r_value, p_value, _ = StatsAnalyzer.calculate_regression(x_vals, y_vals)
        if not np.isnan(slope):
            x_fit = np.array(ax.get_xlim()); y_fit = intercept + slope * x_fit
            p_str = " (p<0.05)" if p_value < 0.05 else ""
            ax.plot(x_fit, y_fit, color='black', linestyle='--', linewidth=1.5, label=f'Fit (r={r_value:.2f}{p_str})')

        mmm_x = np.mean(x_vals); mmm_y = np.mean(y_vals)
        std_dev_x = np.std(x_vals); std_dev_y = np.std(y_vals)
        ax.scatter(mmm_x, mmm_y, color='red', marker='X', s=120, zorder=10, edgecolor='black', linewidth=1.5, label='MMM')
        
        ax.axhline(mmm_y, color='dimgrey', linestyle='-', linewidth=1.0, zorder=6)
        ax.axvline(mmm_x, color='dimgrey', linestyle='-', linewidth=1.0, zorder=6)

        if inner_radius:
            #inner_ellipse = mpatches.Ellipse(xy=(mmm_x, mmm_y), width=2*inner_radius*std_dev_x, height=2*inner_radius*std_dev_y, angle=0, edgecolor='black', facecolor='grey', alpha=0.2, linewidth=1.0, zorder=5, label='Axial Storyline Zone')
            #ax.add_patch(inner_ellipse)
            t_50 = np.sqrt(chi2.ppf(0.5, 2) / 2)
            outer_ellipse = mpatches.Ellipse(xy=(mmm_x, mmm_y), width=2*t_50*std_dev_x, height=2*t_50*std_dev_y, angle=0, edgecolor='black', facecolor='none', linestyle='--', linewidth=1.5, zorder=5, label='50% Confidence Region')
            ax.add_patch(outer_ellipse)

        storyline_means_to_plot = {
            'Fast Jet & Northward Shift': '#d62728', 'Slow Jet & Northward Shift': '#ff7f0e',
            'Slow Jet & Southward Shift': '#1f77b4', 'Fast Jet & Southward Shift': '#2ca02c'
        }
        for storyline_name, color in storyline_means_to_plot.items():
            storyline_key = f"{season_prefix}_{storyline_name}"
            models_in_storyline = gwl_classification.get(storyline_key, [])
            if models_in_storyline:
                mean_x = np.mean([x_deltas[m] for m in models_in_storyline])
                mean_y = np.mean([y_deltas[m] for m in models_in_storyline])
                ax.scatter(mean_x, mean_y, color=color, marker='X', s=100, zorder=9,
                        edgecolor='black', linewidth=1.0, label=f'Mean: {storyline_name}')
        
        # --- NEW BLOCK: Plot single models for the new extreme storylines ---
        # --- AUSKOMMENTIERT START ---
        # extreme_storylines_to_plot = {
        #     'Extreme NW': {'color': '#8c564b', 'marker': '*'}, # brown star
        #     'Extreme SE': {'color': '#e377c2', 'marker': '*'}  # pink star
        # }
        # for storyline_name, style in extreme_storylines_to_plot.items():
        #     storyline_key = f"{season_prefix}_{storyline_name}"
        #     model_in_storyline = gwl_classification.get(storyline_key) # Should be a list with one model
        #     if model_in_storyline:
        #         model_key = model_in_storyline[0]
        #         x_coord = x_deltas.get(model_key)
        #         y_coord = y_deltas.get(model_key)
        #         if x_coord is not None and y_coord is not None:
        #             ax.scatter(x_coord, y_coord, color=style['color'], marker=style['marker'], s=250, zorder=12,
        #                     edgecolor='black', linewidth=1.2, label=f'Model: {storyline_name}')
        # --- AUSKOMMENTIERT ENDE ---

        # --- AUSKOMMENTIERT START ---
        # axial_colors = {'Northward': '#1f77b4', 'Southward': '#ff7f0e', 'Fast': '#2ca02c', 'Slow': '#d62728'}
        # extreme_storyline_means = {}
        # extreme_types = [k.replace(f'{season_prefix}_', '') for k in gwl_classification if 'Shift' in k and 'Only' not in k and 'Extreme' not in k]
        # for storyline_name in extreme_types:
        #     models_in_storyline = gwl_classification.get(f"{season_prefix}_{storyline_name}", [])
        #     if models_in_storyline:
        #         mean_x = np.mean([x_deltas[m] for m in models_in_storyline])
        #         mean_y = np.mean([y_deltas[m] for m in models_in_storyline])
        #         extreme_storyline_means[storyline_name] = {'speed': mean_x, 'lat': mean_y}
        # 
        # if extreme_storyline_means:
        #     max_lat_storyline = max(extreme_storyline_means, key=lambda k: extreme_storyline_means[k]['lat'])
        #     min_lat_storyline = min(extreme_storyline_means, key=lambda k: extreme_storyline_means[k]['lat'])
        #     max_speed_storyline = max(extreme_storyline_means, key=lambda k: extreme_storyline_means[k]['speed'])
        #     min_speed_storyline = min(extreme_storyline_means, key=lambda k: extreme_storyline_means[k]['speed'])
        #     axial_centers = {
        #         'Northward Shift Only': {'center': (0, extreme_storyline_means[max_lat_storyline]['lat']), 'color': axial_colors['Northward']},
        #         'Southward Shift Only': {'center': (0, extreme_storyline_means[min_lat_storyline]['lat']), 'color': axial_colors['Southward']},
        #         'Fast Jet Only': {'center': (extreme_storyline_means[max_speed_storyline]['speed'], 0), 'color': axial_colors['Fast']},
        #         'Slow Jet Only': {'center': (extreme_storyline_means[min_speed_storyline]['speed'], 0), 'color': axial_colors['Slow']},
        #     }
        #     for name, data in axial_centers.items():
        #         center_x, center_y = data['center']
        #         ax.scatter(center_x, center_y, marker='D', s=80, color=data['color'], edgecolor='black', zorder=11, label=f'Center: {name}')
        #         axial_ellipse = mpatches.Ellipse(xy=(center_x, center_y), width=2*inner_radius*std_dev_x, height=2*inner_radius*std_dev_y, angle=0, edgecolor=data['color'], facecolor='none', linestyle=':', linewidth=2.0, zorder=10)
        #         ax.add_patch(axial_ellipse)
        # --- AUSKOMMENTIERT ENDE ---
        
        def get_axis_label(key):
            label = f'Change in {key.replace("_", " ")}';
            if "Lat" in key: label += ' (°Lat)';
            elif "Speed" in key: label += ' (m/s)';
            return label
        ax.set_xlabel(get_axis_label(x_jet_key), fontsize=9)
        ax.set_ylabel(get_axis_label(y_jet_key), fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.grid(True, linestyle=':', alpha=0.6)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=2, fontsize=8)
        
    @staticmethod
    def plot_jet_inter_relationship_scatter_combined_gwl(cmip6_results, scenario):
        """
        Erstellt einen kombinierten Scatter-Plot, der die Beziehung zwischen den Jet-Indizes
        in den CMIP6-Modellen für alle GWLs, getrennt nach Jahreszeiten, darstellt.
        NEU: Verwendet die Quadranten-Visualisierungsmethode.
        MODIFIED: Accepts a scenario parameter for filename and title.
        """
        if not cmip6_results or 'all_individual_model_deltas_for_plot' not in cmip6_results:
            logging.warning(f"Cannot plot combined jet inter-relationship scatter for {scenario}: Missing CMIP6 results.")
            return

        gwls_to_plot = Config.GLOBAL_WARMING_LEVELS
        if not gwls_to_plot:
            logging.warning(f"No global warming levels to plot for {scenario}. Skipping jet inter-relationship plot.")
            return

        inner_radius = Config.STORYLINE_INNER_RADIUS

        logging.info(f"Plotting seasonal CMIP6 jet inter-relationship scatter for scenario {scenario}, GWLs: {gwls_to_plot}...")
        Visualizer.ensure_plot_dir_exists()

        fig, axs = plt.subplots(len(gwls_to_plot), 2, figsize=(14, 6.5 * len(gwls_to_plot)), squeeze=False)

        for i, gwl in enumerate(gwls_to_plot):
            Visualizer._plot_single_jet_relationship_panel(
                ax=axs[i, 0], cmip6_results=cmip6_results, gwl_to_plot=gwl,
                x_jet_key='DJF_JetSpeed', y_jet_key='DJF_JetLat',
                title=f'Winter: Speed vs. Latitude ({gwl}°C GWL)',
                inner_radius=inner_radius
            )
            Visualizer._plot_single_jet_relationship_panel(
                ax=axs[i, 1], cmip6_results=cmip6_results, gwl_to_plot=gwl,
                x_jet_key='JJA_JetSpeed', y_jet_key='JJA_JetLat',
                title=f'Summer: Speed vs. Latitude ({gwl}°C GWL)',
                inner_radius=inner_radius
            )

        ref_period = f"{Config.CMIP6_ANOMALY_REF_START}-{Config.CMIP6_ANOMALY_REF_END}"
        fig.suptitle(f"CMIP6 Jet Index Inter-relationships by Season for {scenario.upper()}\n"
                     f"(Changes relative to {ref_period})",
                     fontsize=16, weight='bold')

        fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=5.0)
        
        filename = os.path.join(Config.PLOT_DIR, f"cmip6_jet_inter_relationship_scatter_quadrants_{scenario}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logging.info(f"Saved seasonal CMIP6 jet inter-relationship scatter plot for {scenario} to {filename}")

    @staticmethod
    def plot_u850_change_map(ax, u850_change_data, historical_mean_contours,
                            lons, lats, title, season_label,
                            cmap='RdBu_r', vmin=-2, vmax=2, cbar_label='U850 Change (m/s)',
                            contour_levels=np.arange(4, 21, 4)):
        """
        Plots a map of U850 change with historical mean contours.
        """
        if u850_change_data is None:
            ax.text(0.5, 0.5, "Data Missing", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title + f"\n{season_label}\n(Data Missing)", fontsize=10)
            return None

        ax.set_extent(Config.PLOT_MAP_EXTENT, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=0)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = gl.right_labels = False
        gl.xlabel_style = {'size': 8}; gl.ylabel_style = {'size': 8}

        if lons.ndim == 1:
            lons_plot, lats_plot = np.meshgrid(lons, lats)
        else:
            lons_plot, lats_plot = lons, lats

        cf = ax.pcolormesh(lons_plot, lats_plot, u850_change_data, shading='auto',
                        cmap=cmap, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())

        if historical_mean_contours is not None:
            try:
                cs = ax.contour(lons_plot, lats_plot, historical_mean_contours,
                                levels=contour_levels, colors='black',
                                linewidths=0.8, transform=ccrs.PlateCarree())
                ax.clabel(cs, inline=True, fontsize=7, fmt='%d')
            except Exception as e_contour:
                logging.warning(f"Could not plot contours for {title} {season_label}: {e_contour}")

        ax.set_title(f"{title}\n{season_label}", fontsize=10)
        return cf, cbar_label

    @staticmethod
    def plot_cmip6_u850_change_panel(u850_change_results, config,
                                    future_period=(2070,2099), historical_period=(1995,2014),
                                    filename="cmip6_u850_change_djf_jja.png"):
        """
        Creates a panel plot for CMIP6 MMM U850 changes (DJF & JJA).
        """
        logging.info(f"Plotting CMIP6 MMM U850 change panel to {filename}...")
        Visualizer.ensure_plot_dir_exists()

        if u850_change_results is None or \
        u850_change_results.get('DJF') is None or \
        u850_change_results.get('JJA') is None:
            logging.warning("Cannot plot U850 change panel: Missing DJF or JJA data.")
            return

        fig = plt.figure(figsize=(12, 6))
        gs = gridspec.GridSpec(1, 3, width_ratios=[10, 10, 1], wspace=0.1)
        cf_ref = None

        # DJF Plot
        ax_djf = cast(Any, fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree()))
        djf_data = u850_change_results['DJF']
        if djf_data and djf_data.get('u850_change_mmm') is not None:
            res_djf = Visualizer.plot_u850_change_map(
                ax_djf,
                djf_data['u850_change_mmm'].data,
                djf_data['u850_historical_mean_mmm'].data,
                djf_data['u850_change_mmm'].lon.values,
                djf_data['u850_change_mmm'].lat.values,
                "CMIP6 MMM U850 Change", "DJF"
            )
            if res_djf is not None:
                cf_djf, _ = res_djf
                if cf_djf: cf_ref = cf_djf
        else:
            ax_djf.text(0.5,0.5, "DJF Data Missing", transform=ax_djf.transAxes, ha='center', va='center')
            ax_djf.set_title("CMIP6 MMM U850 Change\nDJF\n(Data Missing)")

        # JJA Plot
        ax_jja = cast(Any, fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree()))
        jja_data = u850_change_results['JJA']
        if jja_data and jja_data.get('u850_change_mmm') is not None:
            res_jja = Visualizer.plot_u850_change_map(
                ax_jja,
                jja_data['u850_change_mmm'].data,
                jja_data['u850_historical_mean_mmm'].data,
                jja_data['u850_change_mmm'].lon.values,
                jja_data['u850_change_mmm'].lat.values,
                "CMIP6 MMM U850 Change", "JJA"
            )
            if res_jja is not None:
                cf_jja, label_jja = res_jja
                if cf_jja and cf_ref is None: cf_ref = cf_jja
        else:
            ax_jja.text(0.5,0.5, "JJA Data Missing", transform=ax_jja.transAxes, ha='center', va='center')
            ax_jja.set_title("CMIP6 MMM U850 Change\nJJA\n(Data Missing)")

        # Colorbar
        if cf_ref is not None:
            cax = fig.add_subplot(gs[0, 2])
            cbar = fig.colorbar(cf_ref, cax=cax, extend='both')
            cbar.set_label('U850 Change (m/s)', fontsize=9)
            cbar.ax.tick_params(labelsize=8)
        
        plt.suptitle(f"CMIP6 MMM U850 Change ({future_period[0]}-{future_period[1]} minus {historical_period[0]}-{historical_period[1]})", fontsize=14, weight='bold')
        # [KORREKTUR] fig.tight_layout() anstelle von plt.tight_layout()
        fig.tight_layout(rect=(0, 0, 0.95, 0.95))
        filepath = os.path.join(config.PLOT_DIR, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        logging.info(f"Saved CMIP6 U850 change panel to {filepath}")
        plt.close(fig)
        
    @staticmethod
    def plot_model_fidelity_comparison(cmip6_historical_slopes, cmip6_future_temporal_slopes, beta_obs_slopes,
                                     historical_period, gwls_to_plot, scenario):
        """
        Creates a comprehensive, publication-quality plot comparing the distribution
        of temporal jet-impact slopes from historical simulations and future projections.
        This provides a direct comparison of model fidelity vs. future relationship stability.
        """
        if not cmip6_historical_slopes or not beta_obs_slopes or not cmip6_future_temporal_slopes:
            logging.warning("Cannot plot comprehensive model fidelity comparison: Missing data.")
            return

        logging.info("Plotting comprehensive Model Fidelity vs. Future Temporal Slopes...")
        Visualizer.ensure_plot_dir_exists()

        valid_keys = sorted([k for k in beta_obs_slopes if k in cmip6_historical_slopes and cmip6_historical_slopes[k]])
        if not valid_keys:
            logging.warning("No common valid keys for fidelity plot.")
            return

        # --- 1. Setup Aesthetics ---
        color_cmip_hist = '#a6cee3'
        color_cmip_future_2c = '#fdbf6f'
        color_cmip_future_3c = '#ff7f00'
        color_cmip_median = 'black'
        color_era5_obs = '#e31a1c'

        # --- 2. Create Figure Layout ---
        n_keys = len(valid_keys)
        n_cols = 1 + len(gwls_to_plot)
        fig, axs = plt.subplots(n_keys, n_cols, figsize=(4 * n_cols, 3.5 * n_keys),
                                sharey='row', squeeze=False)
        fig.subplots_adjust(wspace=0.1, hspace=0.35)

        # --- 3. Plotting Loop ---
        for i, key in enumerate(valid_keys):
            # Column 1: Historical Fidelity
            ax_hist = axs[i, 0]
            model_slopes_hist = cmip6_historical_slopes[key]
            obs_slope = beta_obs_slopes[key]

            bp_hist = ax_hist.boxplot(model_slopes_hist, vert=True, patch_artist=True, widths=0.7,
                                      boxprops=dict(facecolor=color_cmip_hist, color='black', linewidth=1),
                                      medianprops=dict(color=color_cmip_median, linewidth=2.5),
                                      showfliers=False)
            
            ax_hist.axhline(obs_slope, color=color_era5_obs, linestyle='-', linewidth=2.5, zorder=10)

            # Columns 2, 3...: Future Projections
            for j, gwl in enumerate(gwls_to_plot):
                ax_future = axs[i, j + 1]
                future_slopes = cmip6_future_temporal_slopes.get(key, {}).get(gwl, [])
                
                if not future_slopes:
                    ax_future.text(0.5, 0.5, "Data\nMissing", ha='center', va='center', transform=ax_future.transAxes)
                else:
                    future_color = color_cmip_future_2c if gwl == 2.0 else color_cmip_future_3c
                    bp_future = ax_future.boxplot(future_slopes, vert=True, patch_artist=True, widths=0.7,
                                                  boxprops=dict(facecolor=future_color, color='black', linewidth=1),
                                                  medianprops=dict(color=color_cmip_median, linewidth=2.5),
                                                  showfliers=False)
                
                ax_future.axhline(obs_slope, color=color_era5_obs, linestyle='-', linewidth=2.5, zorder=10)
        
        # --- 4. Final Formatting & Labeling ---
        for i, key in enumerate(valid_keys):
            title_parts = key.replace('_vs_', ' vs. ').replace('_', ' ')
            # Geänderte Y-Achsen-Beschriftung für mehr Klarheit
            axs[i, 0].set_ylabel(f'Regression Slope for\n{title_parts}', fontsize=9, weight='bold')

            all_row_data = list(cmip6_historical_slopes.get(key, []))
            for j, gwl in enumerate(gwls_to_plot):
                 all_row_data.extend(cmip6_future_temporal_slopes.get(key, {}).get(gwl, []))
            
            if all_row_data:
                y_min = min(all_row_data)
                y_max = max(all_row_data)
                y_range = y_max - y_min
                axs[i, 0].set_ylim(y_min - y_range * 0.1, y_max + y_range * 0.1)

            for j in range(n_cols):
                axs[i, j].grid(axis='y', linestyle=':', alpha=0.7)
                axs[i, j].set_xticks([])

        # Column titles
        axs[0, 0].set_title(f'Historical\n({historical_period[0]}-{historical_period[1]})', fontsize=11, weight='bold')
        for j, gwl in enumerate(gwls_to_plot):
            axs[0, j + 1].set_title(f'Future Projection\n(+{gwl}°C GWL)', fontsize=11, weight='bold')
        
        # --- 5. Centralized Legend ---
        hist_patch = mpatches.Patch(color=color_cmip_hist, ec='black', label=f'CMIP6 Historical')
        future_2c_patch = mpatches.Patch(color=color_cmip_future_2c, ec='black', label=f'CMIP6 Future (+2°C)')
        future_3c_patch = mpatches.Patch(color=color_cmip_future_3c, ec='black', label=f'CMIP6 Future (+3°C)')
        median_line = plt.Line2D([0], [0], color=color_cmip_median, lw=2.5, label='CMIP6 Median')
        era5_line = plt.Line2D([0], [0], color=color_era5_obs, lw=2.5, ls='-', label='ERA5 Historical')
        
        fig.legend(handles=[hist_patch, future_2c_patch, future_3c_patch, median_line, era5_line],
                   loc='lower center', bbox_to_anchor=(0.5, -0.02), ncol=5, fontsize=9, frameon=False)
        
        # Geänderte Überschrift für mehr Klarheit
        fig.suptitle('Comparison of Jet-Impact Regression Slopes: Historical vs. Future', fontsize=16, weight='bold')
        fig.tight_layout(rect=(0, 0.08, 1, 0.93))
        
        filename = os.path.join(Config.PLOT_DIR, f"cmip6_fidelity_vs_future_temporal_slopes_{scenario}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logging.info(f"Saved comprehensive temporal slope comparison plot to {filename}")
        
    @staticmethod
    def plot_seasonal_drought_analysis(datasets_reanalysis, scale=4):
        """
        Creates a 2x2 summary plot of seasonal drought characteristics from SPEI.
        """
        logging.info(f"Plotting seasonal SPEI-{scale} drought analysis...")
        Visualizer.ensure_plot_dir_exists()
        
        fig, axs = plt.subplots(2, 2, figsize=(18, 12), sharex=True)
        threshold = -1.0

        for row, dataset_key in enumerate([Config.DATASET_20CRV3, Config.DATASET_ERA5]):
            spei_data = datasets_reanalysis.get(f'{dataset_key}_spei{scale}')
            if spei_data is None:
                for col in range(2):
                    axs[row, col].text(0.5, 0.5, f"SPEI Data for {dataset_key}\nnot available", 
                                       ha='center', va='center', transform=axs[row, col].transAxes)
                continue

            for col, season in enumerate(['Winter', 'Summer']):
                ax = axs[row, col]
                # Filter the data for the correct season
                seasonal_filtered = DataProcessor.filter_by_season(spei_data, season)
                
                # Check if data exists after filtering
                if seasonal_filtered is None or seasonal_filtered.time.size < 2:
                    ax.text(0.5, 0.5, "Not enough data", ha='center', va='center', transform=ax.transAxes)
                    continue
                
                # FIX: Group by 'season_year' to make it the primary dimension for the plot.
                # This resolves the "Dimension not found" error.
                seasonal_spei = seasonal_filtered.groupby('season_year').mean(dim='time')
                
                # Now 'season_year' is the dimension, and we can safely use it.
                years = seasonal_spei.season_year.values
                values = seasonal_spei.values

                # Plot SPEI time series
                ax.plot(years, values, color='darkblue', lw=0.8, label=f'SPEI-{scale}')
                
                # Highlight drought periods
                ax.fill_between(years, values, threshold, where=(values < threshold),
                                color='red', alpha=0.3, interpolate=True)
                
                # Calculate and plot linear trend
                slope, intercept, r_val, p_val, std_err = StatsAnalyzer.calculate_regression(years, values)
                trend_line = intercept + slope * years
                ax.plot(years, trend_line, 'k--', lw=1.5, label=f'Trend (p={p_val:.3f})')
                
                # Get drought statistics
                # The stats function expects a 'time' dimension, so we rename 'season_year' just for this call.
                stats = StatsAnalyzer.analyze_drought_characteristics(seasonal_spei.rename({'season_year': 'time'}), threshold)
                
                # Create text box with stats
                stats_text = (
                    f"Drought Stats (SPEI < {threshold}):\n"
                    f"---------------------------------\n"
                    f"Number of Events: {stats.get('number_of_events', 'N/A')}\n"
                    f"Mean Duration: {stats.get('mean_duration', 0):.1f} years\n"
                    f"Longest Duration: {stats.get('longest_duration', 0)} years\n"
                    f"Mean Intensity: {stats.get('mean_intensity', 0):.2f}\n"
                    f"Peak Intensity: {stats.get('peak_intensity', 0):.2f}"
                )
                
                ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, fontsize=9,
                        verticalalignment='bottom', bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8))

                ax.set_title(f"{dataset_key} - {season} (DJF)" if season == 'Winter' else f"{dataset_key} - {season} (JJA)")
                ax.set_ylabel(f"SPEI-{scale}")
                ax.grid(True, linestyle=':', alpha=0.6)
                ax.axhline(threshold, color='red', linestyle=':', lw=1.0)
                ax.legend(loc='upper right')

        for ax in axs[1, :]:
            ax.set_xlabel("Year")
            
        fig.suptitle(f'Seasonal Drought Analysis Comparison (SPEI-{scale})', fontsize=16, weight='bold')
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        filename = os.path.join(Config.PLOT_DIR, 'spei_drought_analysis_seasonal_comparison.png')
        plt.savefig(filename, dpi=300)
        plt.close(fig)

    @staticmethod
    def plot_spatial_spei_analysis_maps(
        spatial_spei_data,
        discharge_corr_map,
        p_values_corr,
        discharge_regr_slopes,
        p_values_regr,
        time_slice,
        season,
        title_prefix,
        filename="spei_discharge_analysis.png"
    ):
        """
        Plots a 1x3 panel:
        1. Spatial SPEI for a specific time, cropped to the analysis box.
        2. Correlation map showing where SPEI is linked to discharge.
        3. Regression map showing where SPEI has the strongest influence on discharge.
        """
        logging.info(f"Plotting combined SPEI and Discharge analysis maps for {season}...")
        Visualizer.ensure_plot_dir_exists()

        fig, axs = plt.subplots(1, 3, figsize=(24, 7), subplot_kw={'projection': ccrs.PlateCarree()})
        plt.subplots_adjust(wspace=0.15, hspace=0.2)

        # --- General Settings ---
        extent_box = [Config.BOX_LON_MIN - 2, Config.BOX_LON_MAX + 2, Config.BOX_LAT_MIN - 2, Config.BOX_LAT_MAX + 2]
        analysis_box_rect = mpatches.Rectangle(
            (Config.BOX_LON_MIN, Config.BOX_LAT_MIN), Config.BOX_LON_MAX - Config.BOX_LON_MIN, Config.BOX_LAT_MAX - Config.BOX_LAT_MIN,
            fill=False, edgecolor='lime', linewidth=2.5, zorder=10
        )
        
        # MODIFIED: Determine stipple skip value based on dataset
        stipple_skip = 7 if title_prefix == Config.DATASET_ERA5 else 2

        def setup_map_ax(ax):
            ax.set_extent(extent_box, crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
            ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.8)
            gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
            gl.top_labels = gl.right_labels = False
            ax.add_patch(mpatches.Rectangle((analysis_box_rect.get_x(), analysis_box_rect.get_y()), analysis_box_rect.get_width(), analysis_box_rect.get_height(), fill=False, edgecolor='lime', linewidth=2.5, zorder=10))
            return gl

        # --- Panel 1: SPEI Map ---
        ax1 = axs[0]
        gl1 = setup_map_ax(ax1)
        try:
            time_stamp = pd.to_datetime(time_slice)
            spei_slice = spatial_spei_data.sel(time=time_stamp, method='nearest')
            lons, lats = np.meshgrid(spei_slice.lon, spei_slice.lat)
            cf1 = ax1.pcolormesh(lons, lats, spei_slice.values, cmap='BrBG', vmin=-2.5, vmax=2.5, transform=ccrs.PlateCarree(), shading='auto')
            cbar1 = fig.colorbar(cf1, ax=ax1, orientation='vertical', pad=0.02, aspect=25, extend='both', shrink=0.8)
            cbar1.set_label('SPEI Value')
            ax1.set_title(f"a) SPEI-4 on {time_stamp.strftime('%Y-%m-%d')}", fontsize=12)
        except Exception as e:
            ax1.text(0.5, 0.5, "SPEI Data Error", transform=ax1.transAxes, ha='center', va='center')
            logging.error(f"Could not create SPEI slice panel: {e}")

        # --- Panel 2: Correlation Map ---
        ax2 = axs[1]
        gl2 = setup_map_ax(ax2)
        gl2.left_labels = False
        if discharge_corr_map is not None:
            lons, lats = np.meshgrid(discharge_corr_map.lon, discharge_corr_map.lat)
            cf2 = ax2.pcolormesh(lons, lats, discharge_corr_map.values, cmap='PiYG', vmin=-0.7, vmax=0.7, transform=ccrs.PlateCarree(), shading='auto')
            if p_values_corr is not None:
                sig_mask = (p_values_corr < 0.05) & np.isfinite(discharge_corr_map)
                # MODIFIED: Apply stipple skip logic
                points_to_plot_mask = np.zeros_like(sig_mask, dtype=bool)
                points_to_plot_mask[::stipple_skip, ::stipple_skip] = True
                final_mask = sig_mask & points_to_plot_mask
                ax2.scatter(lons[final_mask], lats[final_mask], s=2, color='black', marker='.', alpha=0.7, transform=ccrs.PlateCarree())

            cbar2 = fig.colorbar(cf2, ax=ax2, orientation='vertical', pad=0.02, aspect=25, extend='both', shrink=0.8)
            cbar2.set_label('Correlation Coefficient (r)')
            ax2.set_title(f"b) Correlation: Local SPEI vs. Discharge ({season})", fontsize=12)
        else:
            ax2.text(0.5, 0.5, "Correlation Data Error", transform=ax2.transAxes, ha='center', va='center')

        # --- Panel 3: Regression Map ---
        ax3 = axs[2]
        gl3 = setup_map_ax(ax3)
        gl3.left_labels = False
        if discharge_regr_slopes is not None:
            lons, lats = np.meshgrid(discharge_regr_slopes.lon, discharge_regr_slopes.lat)
            vmax = np.nanpercentile(np.abs(discharge_regr_slopes), 98) # Dynamic limit for better colors
            cf3 = ax3.pcolormesh(lons, lats, discharge_regr_slopes.values, cmap='RdYlBu', vmin=-vmax, vmax=vmax, transform=ccrs.PlateCarree(), shading='auto')
            if p_values_regr is not None:
                sig_mask = (p_values_regr < 0.05) & np.isfinite(discharge_regr_slopes)
                # MODIFIED: Apply stipple skip logic
                points_to_plot_mask = np.zeros_like(sig_mask, dtype=bool)
                points_to_plot_mask[::stipple_skip, ::stipple_skip] = True
                final_mask = sig_mask & points_to_plot_mask
                ax3.scatter(lons[final_mask], lats[final_mask], s=2, color='black', marker='.', alpha=0.7, transform=ccrs.PlateCarree())

            cbar3 = fig.colorbar(cf3, ax=ax3, orientation='vertical', pad=0.02, aspect=25, extend='both', shrink=0.8)
            cbar3.set_label('Discharge Change [m³/s] per Std.Dev. of SPEI')
            ax3.set_title(f"c) Influence: Local SPEI on Discharge ({season})", fontsize=12)
        else:
            ax3.text(0.5, 0.5, "Regression Data Error", transform=ax3.transAxes, ha='center', va='center')
        
        fig.suptitle(f"{title_prefix}: Spatial Drought Analysis and Hydrological Link ({season})", fontsize=16, weight='bold')
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        
        filepath = os.path.join(Config.PLOT_DIR, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logging.info(f"Saved combined SPEI/Discharge analysis plot to {filepath}")
        
    @staticmethod
    def plot_cmip6_model_regression_analysis(all_model_data, model_keys, dataset_key_prefix="CMIP6"):
        """Creates a 4x4 panel plot for regression maps of individual models."""
        logging.info(f"Plotting single-model U850 vs Box Index regression maps for {model_keys}...")
        Visualizer.ensure_plot_dir_exists()

        if not all_model_data or not all(m in all_model_data for m in model_keys):
            logging.warning("Skipping single-model regression plot: Data for one or more models is missing.")
            return

        # Create a 4x5 grid: 4 rows for models, 4 columns for plots, 1 for the colorbars
        fig = plt.figure(figsize=(22, 18))
        gs = gridspec.GridSpec(4, 5, width_ratios=[10, 10, 10, 10, 1], height_ratios=[1, 1, 1, 1], wspace=0.3, hspace=0.4)
        box_coords = [Config.BOX_LON_MIN, Config.BOX_LON_MAX, Config.BOX_LAT_MIN, Config.BOX_LAT_MAX]

        # Global references for the colorbars
        cf_pr_ref, cf_tas_ref = None, None
        label_pr_ref, label_tas_ref = "", ""

        for i, model_key in enumerate(model_keys):
            model_data = all_model_data.get(model_key, {})
            if not model_data:
                for j in range(4):
                    ax = fig.add_subplot(gs[i, j])
                    ax.text(0.5, 0.5, f"Data for {model_key}\nnot available", ha='center', va='center', transform=ax.transAxes)
                    ax.set_xticks([])
                    ax.set_yticks([])
                continue

            # --- Precipitation Plots (PR) ---
            winter_data = model_data.get('Winter', {})
            ax_pr_winter = fig.add_subplot(gs[i, 0], projection=ccrs.PlateCarree())
            cf_pr, label_pr = Visualizer.plot_regression_map(
                ax_pr_winter, winter_data.get('slopes_pr'), winter_data.get('p_values_pr'),
                winter_data.get('lons'), winter_data.get('lats'),
                f"{model_key.split('_')[0]}", box_coords, "DJF PR", 'pr',
                ua_seasonal_mean=winter_data.get('ua850_mean'),
                std_dev_predictor=winter_data.get('std_dev_pr'), dataset_key=model_key, stipple_skip=1
            )
            if cf_pr is not None: cf_pr_ref, label_pr_ref = cf_pr, label_pr

            summer_data = model_data.get('Summer', {})
            ax_pr_summer = fig.add_subplot(gs[i, 1], projection=ccrs.PlateCarree())
            cf_pr, _ = Visualizer.plot_regression_map(
                ax_pr_summer, summer_data.get('slopes_pr'), summer_data.get('p_values_pr'),
                summer_data.get('lons'), summer_data.get('lats'),
                f"{model_key.split('_')[0]}", box_coords, "JJA PR", 'pr',
                ua_seasonal_mean=summer_data.get('ua850_mean'),
                std_dev_predictor=summer_data.get('std_dev_pr'), dataset_key=model_key, stipple_skip=1
            )
            if cf_pr is not None and cf_pr_ref is None: cf_pr_ref = cf_pr


            # --- Temperature Plots (TAS) ---
            ax_tas_winter = fig.add_subplot(gs[i, 2], projection=ccrs.PlateCarree())
            cf_tas, label_tas = Visualizer.plot_regression_map(
                ax_tas_winter, winter_data.get('slopes_tas'), winter_data.get('p_values_tas'),
                winter_data.get('lons'), winter_data.get('lats'),
                f"{model_key.split('_')[0]}", box_coords, "DJF TAS", 'tas',
                ua_seasonal_mean=winter_data.get('ua850_mean'),
                std_dev_predictor=winter_data.get('std_dev_tas'), dataset_key=model_key, stipple_skip=1
            )
            if cf_tas is not None: cf_tas_ref, label_tas_ref = cf_tas, label_tas

            ax_tas_summer = fig.add_subplot(gs[i, 3], projection=ccrs.PlateCarree())
            cf_tas, _ = Visualizer.plot_regression_map(
                ax_tas_summer, summer_data.get('slopes_tas'), summer_data.get('p_values_tas'),
                summer_data.get('lons'), summer_data.get('lats'),
                f"{model_key.split('_')[0]}", box_coords, "JJA TAS", 'tas',
                ua_seasonal_mean=summer_data.get('ua850_mean'),
                std_dev_predictor=summer_data.get('std_dev_tas'), dataset_key=model_key, stipple_skip=1
            )
            if cf_tas is not None and cf_tas_ref is None: cf_tas_ref = cf_tas

        # Add shared colorbars
        if cf_pr_ref:
            cax_pr = fig.add_subplot(gs[:2, 4])
            fig.colorbar(cf_pr_ref, cax=cax_pr, extend='both', label=label_pr_ref)
        if cf_tas_ref:
            cax_tas = fig.add_subplot(gs[2:, 4])
            fig.colorbar(cf_tas_ref, cax=cax_tas, extend='both', label=label_tas_ref)

        plt.suptitle("CMIP6 Single Model U850 Regression onto Box Climate Indices (1995-2014)", fontsize=18, weight='bold')
        fig.tight_layout(rect=(0, 0, 0.95, 0.96))
        filename = os.path.join(Config.PLOT_DIR, f'regression_maps_norm_{dataset_key_prefix}_single_models.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)

    @staticmethod
    def plot_storyline_impact_barchart(storyline_impacts, config):
        """
        Creates a publication-quality 2x2 vertical bar chart to visualize
        the final impacts for different storylines and warming levels, inspired by
        the provided visual example.

        Parameters:
        -----------
        storyline_impacts : dict
            Nested dictionary with final calculated impacts.
            Expected format: {gwl: {impact_var: {storyline: {'total': v, ...}}}}
        config : Config
            The project configuration object.
        """
        logging.info("Plotting final storyline impacts as a vertical bar chart...")
        Visualizer.ensure_plot_dir_exists()

        if not storyline_impacts or not any(storyline_impacts.values()):
            logging.warning("Cannot plot storyline impacts: Input data is empty.")
            return

        # --- 1. Plotting Setup & Aesthetics ---
        plt.style.use('seaborn-v0_8-whitegrid')
        matplotlib.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
            'axes.edgecolor': 'black', 'axes.linewidth': 1,
            'xtick.color': 'black', 'ytick.color': 'black',
            'grid.color': 'grey', 'grid.linestyle': ':', 'grid.linewidth': 0.5,
            'axes.grid': True, 'axes.grid.axis': 'y'
        })

        fig, axs = plt.subplots(2, 2, figsize=(16, 12), sharey='row')

        gwls_to_plot = config.GLOBAL_WARMING_LEVELS
        gwl_colors = {gwls_to_plot[0]: '#4575b4', gwls_to_plot[1]: '#d73027'}

        # Re-ordered grid for better logical flow (Temp top, Precip bottom)
        plot_grid = {
            (0, 0): {'key': 'DJF_tas', 'title': 'a) Winter (DJF) Temperature'},
            (0, 1): {'key': 'JJA_tas', 'title': 'b) Summer (JJA) Temperature'},
            (1, 0): {'key': 'DJF_pr', 'title': 'c) Winter (DJF) Precipitation'},
            (1, 1): {'key': 'JJA_pr', 'title': 'd) Summer (JJA) Precipitation'}
        }

        # --- 2. Data Processing and Plotting Loop ---
        for (row, col), plot_info in plot_grid.items():
            ax = axs[row, col]
            impact_key = plot_info['key']
            season = impact_key.split('_')[0]
            
            # Get ordered storyline names from config to ensure consistency
            storyline_names_ordered = list(config.STORYLINE_JET_CHANGES_2D.get(season, {}).get(gwls_to_plot[0], {}).keys())
            
            # Prepare data for plotting
            plot_data = {}
            for gwl in gwls_to_plot:
                impacts = []
                for name in storyline_names_ordered:
                    total_impact = storyline_impacts.get(gwl, {}).get(impact_key, {}).get(name, {}).get('total', np.nan)
                    impacts.append(total_impact)
                plot_data[gwl] = impacts
            
            df = pd.DataFrame(plot_data, index=storyline_names_ordered)

            # Plotting parameters
            x_pos = np.arange(len(storyline_names_ordered))
            bar_width = 0.35
            
            # Plot bars for each GWL
            rects1 = ax.bar(x_pos - bar_width/2, df[gwls_to_plot[0]], bar_width, 
                            label=f'+{gwls_to_plot[0]}°C GWL', color=gwl_colors[gwls_to_plot[0]], zorder=10)
            rects2 = ax.bar(x_pos + bar_width/2, df[gwls_to_plot[1]], bar_width, 
                            label=f'+{gwls_to_plot[1]}°C GWL', color=gwl_colors[gwls_to_plot[1]], zorder=10)

            # --- 3. Subplot Formatting ---
            ax.set_title(plot_info['title'], loc='left', fontsize=14, weight='bold')
            ax.axhline(0, color='black', linestyle='-', linewidth=0.8, zorder=1)
            
            # Y-axis label (only for the left column)
            unit = '(°C)' if 'tas' in impact_key else '(%)'
            if col == 0:
                ax.set_ylabel(f'Projected Change {unit}', fontsize=12)

            # X-axis ticks and labels
            ax.set_xticks(x_pos)
            ax.set_xticklabels([name.replace(' & ', ' &\n').replace(' (MMM)','') for name in storyline_names_ordered], 
                            rotation=45, ha="right", fontsize=11)
            
            # Add value labels on top of bars
            for rects in [rects1, rects2]:
                for rect in rects:
                    height = rect.get_height()
                    if np.isnan(height): continue
                    
                    # Determine text position based on bar height
                    offset = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02
                    text_pos = height + offset if height >= 0 else height - offset
                    va = 'bottom' if height >= 0 else 'top'

                    # KORREKTUR: Formatierung auf zwei Nachkommastellen geändert
                    ax.annotate(f'{height:.2f}',
                                xy=(rect.get_x() + rect.get_width() / 2, text_pos),
                                ha='center', va=va, fontsize=9)
                                
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        # --- 4. Final Figure Formatting ---
        handles, labels = axs[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=2, fontsize=12, frameon=False)
        
        main_title = "Projected Climate Impacts for Jet Stream Storylines"
        subtitle = (f"Impacts calculated for the Central European analysis box\n"
                    f"({config.BOX_LAT_MIN}°N-{config.BOX_LAT_MAX}°N, {config.BOX_LON_MIN}°E-{config.BOX_LON_MAX}°E)")
        
        fig.suptitle(f"{main_title}\n{subtitle}", fontsize=16, weight='bold', y=0.99)
        
        fig.tight_layout(rect=(0, 0.02, 1, 0.93))
        
        filename = os.path.join(config.PLOT_DIR, "storyline_impacts_summary_vertical.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logging.info(f"Saved vertical storyline impacts summary plot to {filename}")

    @staticmethod
    def plot_storyline_impact_barchart_with_discharge(cmip6_results, threshold_data, discharge_data_historical, reanalysis_data, config, scenario, storyline_correlations=None):
        """
        Creates a 7x2 plot to visualize storyline impacts for Temp, Precip, SPEI,
        seasonal Discharge, and lagged monthly Discharge.
        MODIFIED: Applies offset to TAS to show warming relative to 1850-1900.
        """
        logging.info(f"Plotting EXTENDED storyline impacts (7x2 grid with percentile thresholds) for {scenario}...")
        Visualizer.ensure_plot_dir_exists()

        if not cmip6_results:
            logging.warning(f"Cannot plot impacts for {scenario}: Input data is empty.")
            return
        if not threshold_data:
            logging.warning(f"Cannot plot impacts for {scenario}: Threshold data is missing.")
            return

        # --- 1. Datenextraktion und -aufbereitung ---
        all_deltas = cmip6_results.get('all_individual_model_deltas_for_plot', {})
        classification = cmip6_results.get('storyline_classification_2d', {})
        # NEU: Zugriff auf Zeitreihen für Offset
        metric_timeseries = cmip6_results.get('model_metric_timeseries', {})
        
        # Referenzperioden
        ref_1850_1900 = (config.CMIP6_PRE_INDUSTRIAL_REF_START, config.CMIP6_PRE_INDUSTRIAL_REF_END)
        ref_1995_2014 = (config.CMIP6_ANOMALY_REF_START, config.CMIP6_ANOMALY_REF_END)

        pr_box_monthly_hist = DataProcessor.calculate_spatial_mean(reanalysis_data.get('ERA5_pr_monthly'), config.BOX_LAT_MIN, config.BOX_LAT_MAX, config.BOX_LON_MIN, config.BOX_LON_MAX)
        tas_box_monthly_hist = DataProcessor.calculate_spatial_mean(reanalysis_data.get('ERA5_tas_monthly'), config.BOX_LAT_MIN, config.BOX_LAT_MAX, config.BOX_LON_MIN, config.BOX_LON_MAX)

        spei_impacts = {}
        if pr_box_monthly_hist is not None and tas_box_monthly_hist is not None:
             temp_storyline_impacts = StorylineAnalyzer.calculate_storyline_impacts(cmip6_results)
             spei_impacts = StorylineAnalyzer.calculate_storyline_spei_impacts(
                 storyline_impacts=temp_storyline_impacts,
                 historical_monthly_data={
                     'pr_box_monthly': pr_box_monthly_hist,
                     'tas_box_monthly': tas_box_monthly_hist
                 },
                 config=config
             )

        gwls_to_plot = config.GLOBAL_WARMING_LEVELS
        
        plot_data_list = []
        impact_keys_for_boxplot = [
            'DJF_tas', 'JJA_tas', 'Annual_tas',
            'DJF_pr', 'JJA_pr', 'Annual_pr',
            'DJF_discharge', 'JJA_discharge', 'Annual_discharge'
        ]

        for gwl in gwls_to_plot:
            storylines_in_gwl = classification.get(gwl, {})
            for storyline_key, model_list in storylines_in_gwl.items():
                if not model_list: continue
                season_prefix = storyline_key.split('_')[0]
                storyline_name = storyline_key.replace(f'{season_prefix}_', '')
                
                if season_prefix == 'DJF':
                    # Add Annual keys here to ensure they are processed exactly once per model/storyline combination
                    relevant_impact_keys = [k for k in impact_keys_for_boxplot if 'DJF' in k or 'Annual' in k]
                elif season_prefix == 'JJA':
                    relevant_impact_keys = [k for k in impact_keys_for_boxplot if 'JJA' in k]
                else:
                    relevant_impact_keys = []
                
                for impact_key in relevant_impact_keys:
                    if impact_key not in impact_keys_for_boxplot: continue
                    
                    # Ist es Temperatur?
                    is_temp = 'tas' in impact_key

                    for model_run_key in model_list:
                        delta_val = all_deltas.get(impact_key, {}).get(gwl, {}).get(model_run_key)
                        
                        # --- NEU: Offset-Berechnung für TAS ---
                        if is_temp and delta_val is not None and np.isfinite(delta_val):
                             ts = metric_timeseries.get(model_run_key, {}).get(impact_key)
                             if ts is not None:
                                 # Wichtig: Prüfen ob Zeitdimension existiert und nicht leer ist
                                 try:
                                     mean_pi = ts.sel(season_year=slice(*ref_1850_1900)).mean().item()
                                     mean_ref = ts.sel(season_year=slice(*ref_1995_2014)).mean().item()
                                     if not np.isnan(mean_pi) and not np.isnan(mean_ref):
                                         offset = mean_ref - mean_pi
                                         delta_val += offset
                                 except Exception:
                                     pass # Fallback auf normales Delta bei Datenfehlern
                        # --- ENDE NEU ---

                        if delta_val is not None and np.isfinite(delta_val):
                            plot_data_list.append({
                                'gwl': f'+{gwl}°C', 'storyline': storyline_name,
                                'impact_key': impact_key, 'value': delta_val
                            })
        df_plot = pd.DataFrame(plot_data_list)

        # --- 2. Plot-Setup (7x2 Grid) ---
        plt.style.use('seaborn-v0_8-whitegrid')
        # --- 2. Plot-Setup (4x3 Grid) ---
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, axs = plt.subplots(3, 3, figsize=(24, 18))
        gwl_colors = {f'+{gwls_to_plot[0]}°C': '#4575b4', f'+{gwls_to_plot[1]}°C': '#d73027'}
        
        plot_grid = {
            (0, 0): {'key': 'Annual_tas', 'title': 'a) Annual Temperature (MMM)'},
            (0, 1): {'key': 'DJF_tas', 'title': 'b) Winter (DJF) Temperature'},
            (0, 2): {'key': 'JJA_tas', 'title': 'c) Summer (JJA) Temperature'},
            
            (1, 0): {'key': 'Annual_pr', 'title': 'd) Annual Precipitation (MMM)'},
            (1, 1): {'key': 'DJF_pr', 'title': 'e) Winter (DJF) Precipitation'},
            (1, 2): {'key': 'JJA_pr', 'title': 'f) Summer (JJA) Precipitation'},
            
            (2, 0): {'key': 'Annual_discharge', 'title': 'g) Annual Discharge (MMM)'},
            (2, 1): {'key': 'DJF_discharge', 'title': 'h) Winter (DJF) Discharge'},
            (2, 2): {'key': 'JJA_discharge', 'title': 'i) Summer (JJA) Discharge'}
        }
        
        storyline_display_order = [
            'MMM', 'Slow Jet & Northward Shift', 'Fast Jet & Northward Shift',
            'Slow Jet & Southward Shift', 'Fast Jet & Southward Shift',
        ]
        
        # --- NEW: Calculate Unified Y-Axis Limits ---
        def get_global_limits(df, keys):
            subset = df[df['impact_key'].isin(keys)]
            if subset.empty: return None, None
            v_min, v_max = subset['value'].min(), subset['value'].max()
            margin = (v_max - v_min) * 0.1 if v_max != v_min else abs(v_max) * 0.1
            if margin == 0: margin = 0.5
            return v_min - margin, v_max + margin

        tas_keys = ['DJF_tas', 'JJA_tas', 'Annual_tas']
        pr_keys = ['DJF_pr', 'JJA_pr', 'Annual_pr']
        dis_keys = ['DJF_discharge', 'JJA_discharge', 'Annual_discharge']

        tas_lims = get_global_limits(df_plot, tas_keys)
        pr_lims = get_global_limits(df_plot, pr_keys)
        dis_lims = get_global_limits(df_plot, dis_keys)
        # --- END NEW ---

        # --- 3. Plotting-Schleife ---
        for (row, col), plot_info in plot_grid.items():
            ax = axs[row, col]
            impact_key = plot_info['key']
            is_spei_plot = 'spei' in impact_key
            is_annual_col = (col == 0) # Annual is now the first column
            
            # Use 'MMM' only for the Annual column, else use full order
            current_order = ['MMM'] if is_annual_col else storyline_display_order

            # Fallback für SPEI (wie zuvor)
            if is_spei_plot:
                df_spei_list = []
                for gwl_float in gwls_to_plot:
                    impacts = spei_impacts.get(gwl_float, {}).get(impact_key, {})
                    for storyline_name in storyline_display_order:
                        if storyline_name in impacts:
                            df_spei_list.append({
                                'gwl': f'+{gwl_float}°C', 'storyline': storyline_name,
                                'value': impacts[storyline_name].get('total')
                            })
                if not df_spei_list:
                    ax.text(0.5, 0.5, "SPEI Data N/A", ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(plot_info['title'], loc='left', fontsize=14, weight='bold')
                    continue
                
                # Filter for MMM if Annual column
                if is_annual_col:
                    df_spei_list = [d for d in df_spei_list if d['storyline'] == 'MMM']
                
                df_spei = pd.pivot_table(pd.DataFrame(df_spei_list),
                                        index='storyline', columns='gwl', values='value').reindex(current_order).dropna(how='all')
                if not df_spei.empty:
                    n_gwls = len(df_spei.columns)
                    total_bar_width = 0.8; bar_width = total_bar_width / n_gwls
                    x_pos = np.arange(len(df_spei.index))
                    for i, gwl_label in enumerate(df_spei.columns):
                        offset = (i - (n_gwls - 1) / 2) * bar_width
                        positions = x_pos + offset
                        color = gwl_colors.get(gwl_label, f'C{i}')
                        ax.bar(positions, df_spei[gwl_label], width=bar_width, label=gwl_label, color=color)
                    ax.set_xticks(x_pos)
                    xtick_labels_spei = [name.replace(' & ', ' &\n').replace(' (MMM)','') for name in df_spei.index]
                    ax.set_xticklabels(xtick_labels_spei, rotation=45, ha="right", fontsize=11)
                    ax.set_xticklabels(xtick_labels_spei, rotation=45, ha="right", fontsize=11)
                    # ax.legend(title="GWL") # Removed per user request

            # Boxplots + Stripplots für tas, pr, discharge
            else:
                if df_plot.empty:
                    ax.text(0.5, 0.5, "Data N/A", ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(plot_info['title'], loc='left', fontsize=14, weight='bold')
                    continue
                data_subset = df_plot[df_plot['impact_key'] == impact_key]
                if is_annual_col:
                    data_subset = data_subset[data_subset['storyline'] == 'MMM']

                if data_subset.empty:
                    ax.text(0.5, 0.5, "Data N/A", ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(plot_info['title'], loc='left', fontsize=14, weight='bold')
                    continue

                sns.boxplot(data=data_subset, x='storyline', y='value', hue='gwl', ax=ax,
                            order=current_order, palette=gwl_colors,
                            linewidth=1.2, showfliers=False, boxprops={'alpha': 0.7}, legend=False)
                sns.stripplot(data=data_subset, x='storyline', y='value', hue='gwl', ax=ax,
                            order=current_order, palette=gwl_colors,
                            dodge=True, jitter=0.15, size=4, edgecolor='gray', linewidth=0.5, legend=False)
                # Ensure no legend remains
                if ax.get_legend() is not None: ax.get_legend().remove()

            # --- 4. Formatierung ---
            ax.set_title(plot_info['title'], loc='left', fontsize=14, weight='bold')
            ax.axhline(0, color='black', linestyle='-', linewidth=0.8, zorder=1)
            ax.set_xlabel('')
            
            unit = ''
            ylabel_text = 'Projected Change' # Default
            
            if 'tas' in impact_key: 
                unit = '(°C)'
                ylabel_text = 'Warming vs. 1850-1900' # <-- ANGEPASST
            elif 'pr' in impact_key: unit = '(%)'
            elif 'discharge' in impact_key: unit = '(m³/s)'
            elif is_spei_plot: unit = '(Std. Dev.)'
            
            if col == 0: ax.set_ylabel(f'{ylabel_text} {unit}', fontsize=12)
            else: ax.set_ylabel('')
            
            if row < 2 or is_spei_plot:
                ax.set_xticklabels([])
            else:
                xtick_labels = [name.replace(' & ', ' &\n').replace(' (MMM)','') for name in current_order]
                ax.set_xticklabels(xtick_labels, rotation=45, ha="right", fontsize=11)
            
            # --- Threshold Lines für Discharge ---
            if 'discharge' in impact_key:
                if dis_lims[0] is not None: ax.set_ylim(dis_lims) # Apply unified limits
                key_thresholds = threshold_data.get(impact_key, {})
                hist_mean_specific = np.nan 
                hist_discharge_monthly = DataProcessor.assign_season_to_dataarray(discharge_data_historical.get('monthly_historical_da'))
                if hist_discharge_monthly is not None:
                     if impact_key in ['DJF_discharge', 'JJA_discharge']:
                         season_name = 'Winter' if 'DJF' in impact_key else 'Summer'
                         hist_ts_mean = DataProcessor.calculate_seasonal_means(hist_discharge_monthly)
                         if hist_ts_mean is not None:
                             hist_ts_filtered = DataProcessor.filter_by_season(hist_ts_mean, season_name)
                             if hist_ts_filtered is not None:
                                 hist_mean_specific = hist_ts_filtered.mean().item()
                     elif 'Annual' in impact_key:
                         hist_mean_specific = hist_discharge_monthly.mean().item()
                     else: 
                         month_num = int(impact_key[0:impact_key.find('_')].replace('Mar','3').replace('Apr','4').replace('May','5').replace('Sep','9').replace('Oct','10').replace('Nov','11'))
                         hist_ts_monthly_filtered = hist_discharge_monthly.where(hist_discharge_monthly.time.dt.month == month_num, drop=True)
                         if hist_ts_monthly_filtered is not None and hist_ts_monthly_filtered.size > 0:
                             hist_mean_specific = hist_ts_monthly_filtered.mean().item()

                if not np.isnan(hist_mean_specific):
                    lnwl_event_name = [k for k in key_thresholds if 'LNWL' in k]
                    if lnwl_event_name:
                         lnwl_val = key_thresholds[lnwl_event_name[0]].get('threshold_m3s') 
                         if lnwl_val is not None:
                             # ax.axhline(lnwl_val - hist_mean_specific, color='red', linestyle='-.', linewidth=2.5, zorder=5) # REMOVED LNWL LINE PER USER REQUEST
                             pass
                    low_extreme_event_name = [k for k in key_thresholds if '<1%' in k]
                    if low_extreme_event_name:
                         low_extreme_val = key_thresholds[low_extreme_event_name[0]].get('threshold_m3s')
                         if low_extreme_val is not None:
                             ax.axhline(low_extreme_val - hist_mean_specific, color='darkviolet', linestyle=(0, (3, 5)), linewidth=2.5, zorder=5) 
                    high_extreme_event_name = [k for k in key_thresholds if '>99%' in k]
                    if high_extreme_event_name:
                         high_extreme_val = key_thresholds[high_extreme_event_name[0]].get('threshold_m3s') 
                         if high_extreme_val is not None:
                             ax.axhline(high_extreme_val - hist_mean_specific, color='deepskyblue', linestyle=(0, (5, 5)), linewidth=2.5, zorder=5)

            # Apply unified limits for TAS and PR
            if 'tas' in impact_key and tas_lims[0] is not None: ax.set_ylim(tas_lims)
            elif 'pr' in impact_key and pr_lims[0] is not None: ax.set_ylim(pr_lims)

        # --- 5. Legende und Finale Formatierung ---
        handles, labels = axs[0, 0].get_legend_handles_labels()
        unique_labels_map = {}
        for l, h in zip(labels, handles):
            if l in gwl_colors and l not in unique_labels_map:
                unique_labels_map[l] = h
        
        djf_thresholds = threshold_data.get('DJF_discharge', {})
        
        # lnwl_key = [k for k in djf_thresholds if 'LNWL' in k]
        # if lnwl_key:
        #      lnwl_val_str = f"{djf_thresholds[lnwl_key[0]].get('threshold_m3s', '?'):.0f}" if isinstance(djf_thresholds[lnwl_key[0]].get('threshold_m3s'), (int, float)) else "?"
        #      unique_labels_map[f'LNWL (~{lnwl_val_str} m³/s)'] = plt.Line2D([0], [0], color='red', linestyle='-.', linewidth=2.5)
        
        # --- Add GWL Legend Entries Manually ---
        unique_labels_map[f'+{gwls_to_plot[0]}°C GWL'] = plt.Rectangle((0,0),1,1, color='#4575b4')
        unique_labels_map[f'+{gwls_to_plot[1]}°C GWL'] = plt.Rectangle((0,0),1,1, color='#d73027')

        low_extreme_key = [k for k in djf_thresholds if '<1%' in k]
        if low_extreme_key:
             low_val_str = f"{djf_thresholds[low_extreme_key[0]].get('threshold_m3s', '?'):.0f}" if isinstance(djf_thresholds[low_extreme_key[0]].get('threshold_m3s'), (int, float)) else "?"
             unique_labels_map[f'Extr. Low (<1%, ~{low_val_str} m³/s)'] = plt.Line2D([0], [0], color='darkviolet', linestyle=(0, (3, 5)), linewidth=2.5)

        high_extreme_key = [k for k in djf_thresholds if '>99%' in k]
        if high_extreme_key:
             high_val_str = f"{djf_thresholds[high_extreme_key[0]].get('threshold_m3s', '?'):.0f}" if isinstance(djf_thresholds[high_extreme_key[0]].get('threshold_m3s'), (int, float)) else "?"
             unique_labels_map[f'Extr. High (>99%, ~{high_val_str} m³/s)'] = plt.Line2D([0], [0], color='deepskyblue', linestyle=(0, (5, 5)), linewidth=2.5)

        fig.legend(unique_labels_map.values(), unique_labels_map.keys(), loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=3, fontsize=12, frameon=False)
        
        main_title = f"Projected Impacts for Jet Stream Storylines ({scenario.upper()})"
        # Angepasster Untertitel, um die gemischten Referenzen zu erklären
        ref_period_text = f"Warming relative to 1850-1900; Other changes relative to 1995-2014"
        fig.suptitle(f"{main_title}\n{ref_period_text}", fontsize=18, weight='bold', y=0.99)
        
        plt.subplots_adjust(left=0.07, right=0.98, top=0.92, bottom=0.18, hspace=0.55, wspace=0.25)
        
        filename = os.path.join(config.PLOT_DIR, f"storyline_impacts_summary_4x2_boxplots_{scenario}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logging.info(f"Saved EXTENDED 7x2 storyline impacts boxplot summary (TAS rel 1850) for {scenario} to {filename}")
        
    @staticmethod
    def plot_extreme_discharge_frequency_comparison(frequency_data, config):
        """
        Creates a grouped bar chart to compare the frequency of extreme low-flow events.
        """
        if not frequency_data:
            logging.warning("Cannot plot extreme discharge frequency: Input data is empty.")
            return

        logging.info("Plotting comparison of extreme discharge event frequency...")
        Visualizer.ensure_plot_dir_exists()

        labels = ['Events < Mean - 2σ', 'Events < Mean - 3σ']
        categories = list(frequency_data.keys())
        # Remove thresholds from categories to plot
        categories.remove('thresholds')

        data_2std = [frequency_data[cat]['2std'] for cat in categories]
        data_3std = [frequency_data[cat]['3std'] for cat in categories]

        df = pd.DataFrame({
            'Below Mean - 2σ': data_2std,
            'Below Mean - 3σ': data_3std,
        }, index=categories)

        # --- Plotting ---
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(12, 8))
        
        df.plot(kind='bar', ax=ax, width=0.8, 
                color={'Below Mean - 2σ': 'skyblue', 'Below Mean - 3σ': 'darkblue'})

        # --- Formatting ---
        ax.set_ylabel('Frequency of Months (%)', fontsize=12)
        ax.set_xticklabels(df.index, rotation=25, ha="right", fontsize=11)
        ax.grid(axis='x', linestyle='none')
        ax.grid(axis='y', linestyle=':', alpha=0.7)
        
        # Add value labels on top of bars
        for container in ax.containers:
            ax.bar_label(cast(Any, container), fmt='%.2f%%', fontsize=9, padding=3)

        # Adjust ylim for padding
        ax.set_ylim(top=ax.get_ylim()[1] * 1.15)
        
        ax.legend(title='Event Threshold', fontsize=11)
        
        hist_mean = frequency_data['thresholds']['mean']
        hist_std = frequency_data['thresholds']['std']
        
        title = "Change in Frequency of Extreme Low-Flow Events for the Danube"
        subtitle = (f"Events are defined by thresholds from historical observations (Mean: {hist_mean:.0f}, σ: {hist_std:.0f} m³/s)")
        
        ax.set_title(f"{title}\n{subtitle}", fontsize=14, weight='bold')

        fig.tight_layout()
        filename = os.path.join(config.PLOT_DIR, "extreme_discharge_frequency_comparison.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logging.info(f"Saved extreme discharge frequency comparison plot to {filename}")

    @staticmethod
    def plot_storyline_return_period_by_event(results, config, scenario):
        """
        Creates a plot showing the change in return periods, organized with
        Half-Year as rows and Event Type as columns.

        NEW LAYOUT (v3b - English):
        - Rows: Winter, Summer
        - Columns: Events (e.g., 7Q10 Low, 7Q10 High, LNWL)
        - Subplots show boxplots for all storylines (vertical).
        - Subplot titles include the specific threshold value.
        """
        if not results or not config or 'thresholds' not in results or 'data' not in results:
            logging.warning(f"Cannot plot return period change by event for {scenario}: Missing results or thresholds.")
            return

        logging.info(f"Plotting storyline return period BY EVENT (Rows=HalfYear, Cols=Event) for {scenario}...")
        Visualizer.ensure_plot_dir_exists()
        
        gwls_to_plot = config.GLOBAL_WARMING_LEVELS
        
        # --- 1. Define Event and Plot Structure ---
        # Get all event keys from thresholds (e.g., '7Q10_low', 'LNWL', '7Q10_high')
        all_event_keys = list(results['thresholds']['winter'].keys())
        all_event_keys.extend(list(results['thresholds']['summer'].keys()))
        unique_event_keys = sorted(list(set(all_event_keys)))

        # Separate Low-Flow and High-Flow events
        low_flow_events = sorted([k for k in unique_event_keys if results['thresholds']['winter'].get(k, {}).get('type') == 'low' or results['thresholds']['summer'].get(k, {}).get('type') == 'low'])
        high_flow_events = sorted([k for k in unique_event_keys if results['thresholds']['winter'].get(k, {}).get('type') == 'high' or results['thresholds']['summer'].get(k, {}).get('type') == 'high'])
        
        # Define plot order
        event_plot_order = low_flow_events + high_flow_events
        num_cols = len(event_plot_order)
        num_rows = 2 # One row for Winter, one for Summer
        half_year_order = ['winter', 'summer']
        
        if num_cols == 0:
            logging.warning(f"No valid EVA events found to plot for {scenario}.")
            return

        fig, axs = plt.subplots(num_rows, num_cols, figsize=(5.5 * num_cols, 7 * num_rows), squeeze=False, sharey=True)
        plt.style.use('seaborn-v0_8-whitegrid')

        storyline_order = [
            'MMM',
            'Slow Jet & Northward Shift',
            'Fast Jet & Northward Shift',
            'Slow Jet & Southward Shift',
            'Fast Jet & Southward Shift',
        ]
        
        x_ticks = np.arange(len(storyline_order))
        x_tick_labels = [s.replace(' & ', ' &\n') for s in storyline_order] # English labels

        gwl_colors = {f'+{gwls_to_plot[0]}°C': '#ff7f0e', f'+{gwls_to_plot[1]}°C': '#d62728'}
        
        # --- 2. Prepare Data for Plotting ---
        plot_data_list = []
        for gwl in gwls_to_plot:
            for half_year in half_year_order:
                for storyline in storyline_order:
                    for event_key in event_plot_order:
                        event_data = results['data'].get(gwl, {}).get(half_year, {}).get(storyline, {}).get(event_key)
                        if event_data and 'future_return_periods_all_models' in event_data:
                            for model_period in event_data['future_return_periods_all_models']:
                                if np.isfinite(model_period):
                                    plot_data_list.append({
                                        'half_year': half_year,
                                        'event': event_key,
                                        'storyline': storyline,
                                        'gwl': f'+{gwl}°C',
                                        'return_period': model_period,
                                    })
        
        if not plot_data_list:
            logging.warning(f"No finite return period data to plot for {scenario}.")
            plt.close(fig)
            return

        df_plot = pd.DataFrame(plot_data_list)
        max_return_period_for_plot = 1000
        # Clip data for plotting (prevents log-scale issues with > 1000yr)
        df_plot_clipped = df_plot[df_plot['return_period'] <= max_return_period_for_plot].copy()
        
        # --- 3. Plotting Loop ---
        for row, half_year in enumerate(half_year_order):
            for col, event_key in enumerate(event_plot_order):
                ax = axs[row, col]
                
                # --- Get Thresholds for Title (as requested) ---
                threshold_data = results.get('thresholds', {}).get(half_year, {}).get(event_key, {})
                event_name = threshold_data.get('name', event_key).replace("Low-Flow", "Low Flow").replace("High-Flow", "High Flow")
                hist_val_q = threshold_data.get('threshold_m3s')
                op = '<' if threshold_data.get('type') == 'low' else '>'
                
                title = f"{event_name}"
                # if hist_val_q:
                #     # Add specific threshold value to title
                #     title += f"\n(Threshold: {op} {hist_val_q:.0f} m³/s)"
                ax.set_title(title, fontsize=11, weight='bold')
                
                # Filter data for this subplot
                data_subset = df_plot_clipped[(df_plot_clipped['half_year'] == half_year) & (df_plot_clipped['event'] == event_key)]
                
                if not data_subset.empty:
                    # Plot HORIZONTAL boxplots (Style matching Figure 3)
                    sns.boxplot(data=data_subset, y='storyline', x='return_period', hue='gwl',
                                order=storyline_order, palette=gwl_colors,
                                ax=ax, linewidth=1.0, width=0.7, showfliers=False, orient='h',
                                boxprops={'alpha': 0.4})
                    sns.stripplot(data=data_subset, y='storyline', x='return_period', hue='gwl',
                                order=storyline_order, palette=gwl_colors,
                                ax=ax, dodge=True, jitter=0.15, size=2, 
                                edgecolor='gray', linewidth=0.5, alpha=0.5, orient='h')
                else:
                    ax.text(0.5, 0.5, "Data N/A", ha='center', va='center', transform=ax.transAxes)

                # Plot Historical Return Period as a VERTICAL line (since plot is horizontal)
                hist_period = threshold_data.get('hist_return_period')
                if hist_period is not None and np.isfinite(hist_period):
                    ax.axvline(x=hist_period, color='skyblue', linestyle='--', linewidth=1.5, zorder=5)

                # --- Axis Formatting ---
                ax.set_xscale('log')
                ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
                ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
                ax.set_xticks([1, 2, 5, 10, 20, 50, 100, 500, 1000])
                ax.set_xlim(left=0.8, right=max_return_period_for_plot * 2.5) # X-lim (formerly Y-lim)

                ax.grid(axis='y', linestyle='none')
                ax.grid(axis='x', linestyle=':', which='both')
                
                # ax.set_xticks(x_ticks) # No longer needed for X-axis in horizontal mode
                
                # --- Labeling Logic ---
                # X-Axis Labels (Return Period) only on bottom row
                if row == num_rows - 1:
                     ax.set_xlabel('Return Period (Years)', fontsize=10)
                     ax.tick_params(axis='x', which='both', labelbottom=True)
                     plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
                else:
                     ax.set_xlabel('')
                     ax.tick_params(axis='x', which='both', labelbottom=False)

                # Y-Axis Labels (Storylines) only on first column
                if col == 0:
                    ax.set_yticklabels(x_tick_labels, fontsize=9) # Use the formatted names
                    season_title = "Winter\n(Nov - Apr)" if half_year == 'winter' else "Summer\n(May - Oct)"
                    ax.set_ylabel(f"{season_title}", fontsize=11, weight='bold')
                else:
                    ax.set_yticklabels([])
                    ax.set_ylabel('')

                if ax.get_legend() is not None: ax.get_legend().remove()
                
                # Add n=X/Y annotations (aligned to the right side now)
                for i, storyline in enumerate(storyline_order):
                    # y_base corresponds to the storyline index
                    y_base = i 
                    for j, gwl in enumerate(gwls_to_plot):
                        y_offset = -0.2 + (j * 0.4) # GWL bar offset
                        gwl_label = f'+{gwl}°C'
                        event_data_gwl = results['data'].get(gwl, {}).get(half_year, {}).get(storyline, {}).get(event_key)
                        if event_data_gwl and 'model_count_X' in event_data_gwl:
                            X, Y = event_data_gwl['model_count_X'], event_data_gwl['model_count_Y']
                            # Place text at the right edge of the plot area
                            ax.text(0.98, y_base + y_offset, f"n={X}/{Y}", transform=ax.get_yaxis_transform(),
                                    horizontalalignment='right', fontsize=7, weight='bold', color=gwl_colors[gwl_label],
                                    bbox=dict(facecolor='white', alpha=0.6, pad=0.1, edgecolor='none'))

        # --- 4. Final Figure Formatting ---
        legend_handles = [
            plt.Line2D([0], [0], color='skyblue', linestyle='--', linewidth=3, label='Historical Return Period'),
            mpatches.Patch(color=gwl_colors[f'+{gwls_to_plot[0]}°C'], label=f'Future (+{gwls_to_plot[0]}°C)'),
            mpatches.Patch(color=gwl_colors[f'+{gwls_to_plot[1]}°C'], label=f'Future (+{gwls_to_plot[1]}°C)')
        ]
        fig.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.5, 0.01), ncol=3, fontsize=12)
        
        fig.suptitle(f"Change in Return Period of Discharge Events for {scenario.upper()} (Half-Year Analysis)",
                    fontsize=16, weight='bold')
        
        # Adjust layout
        bottom_margin = 0.12 if num_cols > 4 else 0.15
        fig.tight_layout(rect=(0.05, bottom_margin, 0.98, 0.95), h_pad=2.5, w_pad=2.0)
        
        # Use a new filename to avoid overwriting the old plot
        filename = os.path.join(config.PLOT_DIR, f"storyline_discharge_return_period_BY_EVENT_{scenario}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logging.info(f"Saved new return period boxplot (by event) to {filename}")


    @staticmethod
    def plot_storyline_return_period_half_year(results, config, scenario):
        """
        Creates a plot showing the change in return periods, organized with
        Low-Flow and High-Flow sections, each having a Winter, Summer, and Full Year row.
        
        --- MODIFIKATION (14.11.2025) ---
        - Ändert die Annotation von 'n=X/Y' zu 'N=Pool (Fin: %)'
        """
        if not results or not config or 'thresholds' not in results or 'data' not in results:
            logging.warning(f"Cannot plot return period change by event for {scenario}: Missing results or thresholds.")
            return

        logging.info(f"Plotting POOLED GEV return period (6-row, 9-col, Full-Year-MMM) for {scenario}...")
        Visualizer.ensure_plot_dir_exists()
        
        # 'config' ist hier das Config-Objekt
        gwls_to_plot = config.GLOBAL_WARMING_LEVELS
        
        # --- 1. Define Event and Plot Structure ---
        all_event_keys = list(results['thresholds'].get('winter', 
                                results['thresholds'].get('summer', 
                                    results['thresholds'].get('full_year', {}))).keys())
        unique_event_keys = sorted(list(set(all_event_keys)))

        def get_event_type(key):
            for hy in ['winter', 'summer', 'full_year']:
                if key in results['thresholds'][hy]:
                    return results['thresholds'][hy][key].get('type')
            return 'unknown'

        low_flow_events = sorted([k for k in unique_event_keys if get_event_type(k) == 'low'])
        high_flow_events = sorted([k for k in unique_event_keys if get_event_type(k) == 'high'])
        
        event_plot_order_keys_low = [
            '7Q10_low', '7Q30_low', 
            '30Q10_low', '30Q30_low'
        ] 
        event_plot_order_keys_high = [
            '7Q10_high', '7Q30_high', 
            '30Q10_high', '30Q30_high'
        ]

        def get_ordered_events(base_list, available_keys):
            ordered_list = [key for key in base_list if key in available_keys]
            ordered_list.extend([k for k in available_keys if k not in base_list and k != 'LNWL'])
            return ordered_list

        low_flow_events_ordered = get_ordered_events(event_plot_order_keys_low, low_flow_events)
        high_flow_events_ordered = get_ordered_events(event_plot_order_keys_high, high_flow_events)

        num_cols_low = len(low_flow_events_ordered)
        num_cols_high = len(high_flow_events_ordered)
        num_cols = max(num_cols_low, num_cols_high, 1)
        
        num_rows = 6 
        
        if num_cols == 0:
            logging.warning(f"No valid EVA events (excl. LNWL) found to plot for {scenario}.")
            return

        fig, axs = plt.subplots(
            num_rows, num_cols, 
            figsize=(5.5 * num_cols, 33), 
            squeeze=False, 
            sharey=False, 
            sharex=False
        )
        
        plt.style.use('seaborn-v0_8-whitegrid')

        storyline_order = [
            'MMM',
            'Slow Jet & Northward Shift',
            'Fast Jet & Northward Shift',
            'Slow Jet & Southward Shift',
            'Fast Jet & Southward Shift',
        ]
        storyline_order_mmm_only = ['MMM']
        
        num_storylines = len(storyline_order)
        num_storylines = len(storyline_order)
        # Use standard limits (0 at bottom) - we will control order via the list
        y_limits = (-0.5, num_storylines - 0.5)
        y_ticks = np.arange(len(storyline_order))
        y_tick_labels = [s.replace(' & ', ' &\n') for s in storyline_order]
        y_tick_labels_mmm_only = ['MMM'] + [''] * (num_storylines - 1)


        gwl_colors = {f'+{gwls_to_plot[0]}°C': '#ff7f0e', f'+{gwls_to_plot[1]}°C': '#d62728'}
        gwl_markers = {f'+{gwls_to_plot[0]}°C': 'o', f'+{gwls_to_plot[1]}°C': 'X'}
        
        # --- 2. Prepare Data for Plotting (v4.0) ---
        plot_data_list = []
        max_val_in_plot = 0
        
        for gwl in gwls_to_plot:
            gwl_label = f'+{gwl}°C'
            
            for half_year in ['winter', 'summer', 'full_year']:
                for storyline in storyline_order:
                    storyline_display = 'Multi-Model Mean' if storyline == 'MMM' else storyline

                    all_events_for_plotting = low_flow_events_ordered + high_flow_events_ordered
                    for event_key in all_events_for_plotting:
                        if event_key == 'LNWL': continue
                        
                        event_data = results['data'].get(gwl, {}).get(half_year, {}).get(storyline, {}).get(event_key)
                        
                        if event_data:
                            # --- MODIFIKATION: Extract ALL model return periods for boxplots ---
                            if 'future_return_periods_all_models' in event_data:
                                rps = event_data['future_return_periods_all_models']
                                rps = [rp for rp in rps if np.isfinite(rp)]
                                
                                if rps:
                                    # Extract CIs (scalars) for this event/storyline
                                    ci_low = event_data.get('future_return_period_ci_low', np.nan)
                                    ci_high = event_data.get('future_return_period_ci_high', np.nan)
                                    
                                    for rp in rps:
                                        if rp > max_val_in_plot: max_val_in_plot = rp
                                        plot_data_list.append({
                                            'half_year': half_year,
                                            'event': event_key,
                                            'storyline': storyline, 
                                            'storyline_display': storyline_display,
                                            'gwl': gwl_label,
                                            'return_period': rp,
                                            'ci_low': ci_low,   # RESTORED KEY
                                            'ci_high': ci_high  # RESTORED KEY
                                        })
        
        if not plot_data_list:
            logging.warning(f"No finite return period data (excl. LNWL) to plot for {scenario}.")
            plt.close(fig)
            return

        df_plot = pd.DataFrame(plot_data_list)
        df_plot_clipped = df_plot.copy()
        
        column_x_limits_lowflow = {c: [] for c in range(num_cols)}
        column_x_limits_highflow = {c: [] for c in range(num_cols)}

        # Defined order for display (matches Figure 3)
        storyline_display_order = [
            'Multi-Model Mean',
            'Slow Jet & Northward Shift',
            'Fast Jet & Northward Shift',
            'Slow Jet & Southward Shift',
            'Fast Jet & Southward Shift',
        ]

        storyline_display_order_mmm = ['Multi-Model Mean']

        # --- 3. Plotting Loop (Reorganized) - PASS 1 ---
        row_configs = [
            {'title_prefix': 'Low-Flow', 'half_year': 'winter', 'events': low_flow_events_ordered, 'mmm_only': False, 'limit_dict': column_x_limits_lowflow, 'season_label': "Winter\n(Nov - Apr)"}, # <<< KORRIGIERT >>>
            {'title_prefix': 'Low-Flow', 'half_year': 'summer', 'events': low_flow_events_ordered, 'mmm_only': False, 'limit_dict': column_x_limits_lowflow, 'season_label': "Summer\n(May - Oct)"}, # <<< KORRIGIERT >>>
            {'title_prefix': 'Low-Flow', 'half_year': 'full_year', 'events': low_flow_events_ordered, 'mmm_only': True, 'limit_dict': column_x_limits_lowflow, 'season_label': "Full Year\n(Jan - Dec)\n(MMM Only)"},
            {'title_prefix': 'High-Flow', 'half_year': 'winter', 'events': high_flow_events_ordered, 'mmm_only': False, 'limit_dict': column_x_limits_highflow, 'season_label': "Winter\n(Nov - Apr)"}, # <<< KORRIGIERT >>>
            {'title_prefix': 'High-Flow', 'half_year': 'summer', 'events': high_flow_events_ordered, 'mmm_only': False, 'limit_dict': column_x_limits_highflow, 'season_label': "Summer\n(May - Oct)"}, # <<< KORRIGIERT >>>
            {'title_prefix': 'High-Flow', 'half_year': 'full_year', 'events': high_flow_events_ordered, 'mmm_only': True, 'limit_dict': column_x_limits_highflow, 'season_label': "Full Year\n(Jan - Dec)\n(MMM Only)"}
        ]
        
        for row, row_config in enumerate(row_configs):
            
            ax_row_start = axs[row, 0]
            half_year = str(row_config['half_year'])
            event_list = cast(list, row_config['events'])
            mmm_only = bool(row_config['mmm_only'])
            limit_dict = cast(dict, row_config['limit_dict'])
            
            current_storyline_order = storyline_display_order_mmm if mmm_only else storyline_display_order
            current_y_tick_labels = [s.replace(' & ', ' &\n') for s in current_storyline_order]
            
            for col, event_key in enumerate(event_list):
                ax = axs[row, col]

                if col > 0:
                    ax.sharey(ax_row_start)
                
                threshold_data = results.get('thresholds', {}).get(half_year, {}).get(event_key, {})
                event_name = threshold_data.get('name', event_key).replace("Low-Flow", "Low Flow").replace("High-Flow", "High Flow")
                hist_val_q = threshold_data.get('threshold_m3s')
                op = '<' if threshold_data.get('type') == 'low' else '>'
                
                title = f"{event_name}"
                if '(<' in title:
                    title = title.split('(<')[0].strip()
                
                if row == 2 or row == 5: # Full Year (italic)
                     ax.set_title(title, fontsize=11, weight='normal', style='italic')
                else: # Winter & Summer (bold)
                    ax.set_title(title, fontsize=11, weight='bold')

                data_subset = df_plot_clipped[(df_plot_clipped['half_year'] == half_year) & (df_plot_clipped['event'] == event_key)]
                
                if mmm_only:
                    data_subset = data_subset[data_subset['storyline'] == 'MMM']
                
                all_data_for_lims = []

                if not data_subset.empty:
                    # --- START: MODIFIKATION (Seaborn Boxplots + Stripplots) ---
                    sns.boxplot(
                        data=data_subset, 
                        y='storyline_display', 
                        x='return_period', 
                        hue='gwl', 
                        ax=ax,
                        order=current_storyline_order, 
                        palette=gwl_colors,
                        showfliers=False, 
                        linewidth=1.0, 
                        width=0.7, 
                        orient='h',
                        boxprops={'alpha': 0.4}
                    )
                    
                    sns.stripplot(
                        data=data_subset, 
                        y='storyline_display', 
                        x='return_period', 
                        hue='gwl', 
                        ax=ax,
                        order=current_storyline_order, 
                        palette=gwl_colors,
                        dodge=True, 
                        jitter=0.15, 
                        size=6, 
                        alpha=0.6, 
                        legend=False, 
                        orient='h'
                    )
                    # --- ENDE: MODIFIKATION ---
                    
                    # Daten für die Achsenlimits sammeln
                    all_data_for_lims.extend(data_subset['return_period'].dropna().values)

                
                else:
                    if not mmm_only: 
                        ax.text(0.5, 0.5, "Data N/A", ha='center', va='center', transform=ax.transAxes, color='gray')

                # Historische Jährlichkeit (T_hist) plotten
                hist_period = threshold_data.get('hist_return_period')
                if hist_period is not None and np.isfinite(hist_period):
                    ax.axvline(x=hist_period, color='skyblue', linestyle='--', linewidth=3, zorder=5)
                    all_data_for_lims.append(hist_period)

                ax.set_xscale('linear')
                ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
                ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
                
                if all_data_for_lims:
                    min_val = np.min([v for v in all_data_for_lims if v > 0]) # Log-Skala mag keine 0
                    max_val = np.max(all_data_for_lims)
                    x_min_limit = max(0.8, min_val * 0.8) 
                    x_max_limit = max_val * 1.5          
                    if (max_val / min_val) < 5: 
                        x_max_limit = max(x_max_limit, min_val * 5)
                    limit_dict[col].append((x_min_limit, x_max_limit))
                else:
                    limit_dict[col].append((0.8, 100))
                
                ax.grid(axis='y', linestyle='none')
                ax.grid(axis='x', linestyle=':', which='both')
                
                ax.set_xlabel('Return Period (Years)', fontsize=11)
                ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, pos: f'{x:.0f}'))
                ax.tick_params(axis='x', labelbottom=True)
                
                ax.set_yticks(np.arange(len(current_y_tick_labels)))
                ax.set_yticklabels(current_y_tick_labels, fontsize=9)
                
                # Set Y-limits (standard, no inversion)
                ax.set_ylim(y_limits) 
                
                
                if col == 0: 
                    ax.set_ylabel(row_config['season_label'], fontsize=11, weight='bold', labelpad=15)
                else:
                    ax.set_ylabel('')
                    plt.setp(ax.get_yticklabels(), visible=False)
                    plt.setp(ax.get_yticklines(), visible=False)

                if ax.get_legend() is not None: ax.get_legend().remove()
                
                # Annotationen (z.B. N=120 (Fin: 10%))
                for i, storyline in enumerate(current_storyline_order):
                    y_base = y_ticks[i]
                    for j, gwl in enumerate(gwls_to_plot):
                        gwl_label = f'+{gwl}°C'
                        y_offset = -0.15 + (j * 0.3) # Y-Versatz für Annotationen
                        event_data_gwl = results['data'].get(gwl, {}).get(half_year, {}).get(storyline, {}).get(event_key)
                        
                        # --- START: MODIFIZIERTE ANNOTATION ---
                        # Explicitly handle MMM mapping if needed
                        storyline_key = 'MMM' if storyline == 'Multi-Model Mean' else storyline
                        
                        # Fallback for MMM if not found directly
                        data_source = results['data'].get(gwl, {}).get(half_year, {}).get(storyline_key, {}).get(event_key)
                        if data_source is None and storyline == 'MMM': # Try direct 'MMM' key if 'Multi-Model Mean' failed
                             data_source = results['data'].get(gwl, {}).get(half_year, {}).get('MMM', {}).get(event_key)

                        if data_source and 'model_count_X' in data_source and 'model_count_Y' in data_source:
                            X = data_source['model_count_X']
                            Y = data_source['model_count_Y']
                            
                            text_to_display = f"n={X}/{Y}"
                            
                            # Ensure color exists
                            text_color = gwl_colors.get(gwl_label, 'black')
                            
                            ax.text(0.98, y_base + y_offset, text_to_display, 
                                    transform=ax.get_yaxis_transform(), 
                                    horizontalalignment='right', fontsize=7, weight='bold', 
                                    color=text_color,
                                    bbox=dict(facecolor='white', alpha=0.6, pad=0.1, edgecolor='none'))
                        # --- ENDE: MODIFIZIERTE ANNOTATION ---
        
        # --- Unbenutzte Achsen ausschalten ---
        for r_idx, row_config in enumerate(row_configs):
            events = row_config.get('events', [])
            num_events_in_row = len(events) if isinstance(events, (list, tuple, dict, str)) else 0
            for c_idx in range(num_events_in_row, num_cols):
                axs[r_idx, c_idx].axis('off')
        
        # --- Zweiter Durchlauf (PASS 2) - Anwenden der X-Limits (GETRENNT) ---
        logging.info("Applying shared column X-limits (Low-Flow and High-Flow separately)...")
        
        for col, limits_list in column_x_limits_lowflow.items():
            if limits_list: 
                final_min_lim_low = min(l[0] for l in limits_list)
                final_max_lim_low = max(l[1] for l in limits_list)
                
                for row in [0, 1, 2]:
                    if col < axs.shape[1]: 
                        axs[row, col].set_xlim(left=0, right=35)
                        if True: pass # axs[row, col].set_xticks([1, 2, 5, 10, 20, 50]) 

        for col, limits_list in column_x_limits_highflow.items():
            if limits_list: 
                final_min_lim_high = min(l[0] for l in limits_list)
                final_max_lim_high = max(l[1] for l in limits_list)
                
                for row in [3, 4, 5]:
                     if col < axs.shape[1]:
                        axs[row, col].set_xlim(left=0, right=35)
                        if True: pass # axs[row, col].set_xticks([1, 2, 5, 10, 20, 50]) 

        # --- 4. Final Figure Formatting ---
        legend_handles = [
            plt.Line2D([0], [0], color='skyblue', linestyle='--', linewidth=3, label='Historical Return Period'),
            plt.Line2D([0], [0], marker=gwl_markers[f'+{gwls_to_plot[0]}°C'], color=gwl_colors[f'+{gwls_to_plot[0]}°C'], 
                       label=f'Future (+{gwls_to_plot[0]}°C)', linestyle='None', markersize=10, markeredgecolor='black', markeredgewidth=0.5),
            plt.Line2D([0], [0], marker=gwl_markers[f'+{gwls_to_plot[1]}°C'], color=gwl_colors[f'+{gwls_to_plot[1]}°C'],
                       label=f'Future (+{gwls_to_plot[1]}°C)', linestyle='None', markersize=10, markeredgecolor='black', markeredgewidth=0.5)
        ]
        fig.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.5, 0.01), ncol=3, fontsize=12)
        
        fig.suptitle(f"Change in Return Period of Discharge Events for {scenario.upper()} (Half-Year & Full-Year Analysis, Empirical)",
                    fontsize=16, weight='bold', y=0.99)
        
        try:
            ax_pos_low_start = axs[0, 0].get_position()
            ax_pos_low_end = axs[0, max(num_cols_low-1, 0)].get_position()
            mid_pos_low = (ax_pos_low_start.x0 + ax_pos_low_end.x1) / 2
            y_pos_low_title = (axs[0,0].get_position().y1 + fig.get_window_extent().height / fig.dpi / fig.get_figheight()) * 0.5 + 0.48
            y_pos_low_title = min(y_pos_low_title, 0.96) 
            fig.text(mid_pos_low, y_pos_low_title, 'Low-Flow Events', ha='center', va='center', fontsize=14, weight='bold')

            ax_pos_high_start = axs[3, 0].get_position()
            ax_pos_high_end = axs[3, max(num_cols_high-1, 0)].get_position()
            mid_pos_high = (ax_pos_high_start.x0 + ax_pos_high_end.x1) / 2
            y_pos_high_title = (axs[2,0].get_position().y0 + axs[3,0].get_position().y1) / 2
            fig.text(mid_pos_high, y_pos_high_title, 'High-Flow Events', ha='center', va='center', fontsize=14, weight='bold')
        except Exception:
             fig.text(0.5, 0.96, 'Low-Flow Events (Rows 1-3)', ha='center', va='center', fontsize=14, weight='bold')
             fig.text(0.5, 0.51, 'High-Flow Events (Rows 4-6)', ha='center', va='center', fontsize=14, weight='bold')
        
        fig.tight_layout(rect=(0.05, 0.05, 0.98, 0.94), h_pad=3.0, w_pad=2.0)
        
        filename = os.path.join(config.PLOT_DIR, f"storyline_discharge_return_period_BY_EVENT_{scenario}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logging.info(f"Saved REORGANIZED return period plot (6-row, Empirical, Full-Year-MMM) to {filename}")
        
    @staticmethod
    def plot_storyline_wind_change_maps(map_data, config, scenario, filename="storyline_u850_change_maps.png"):
        """
        Creates a panel plot of 2D maps showing U850 wind changes for each storyline.
        MODIFIED: Accepts a scenario parameter for filename and title.
        """
        logging.info(f"Plotting U850 wind change maps for each storyline for scenario {scenario}...")
        Visualizer.ensure_plot_dir_exists()

        if not map_data:
            logging.warning(f"Cannot plot wind change maps for {scenario}: Input data is empty.")
            return

        storyline_order = [
            'MMM', 
            # 'Northward Shift Only', # <-- AUSKOMMENTIERT
            'Slow Jet & Northward Shift', 
            'Fast Jet & Northward Shift',
            # 'Southward Shift Only', # <-- AUSKOMMENTIERT
            'Slow Jet & Southward Shift', 
            'Fast Jet & Southward Shift',
            # 'Slow Jet Only',        # <-- AUSKOMMENTIERT
            # 'Fast Jet Only',        # <-- AUSKOMMENTIERT
            # 'Extreme NW',           # <-- AUSKOMMENTIERT
            # 'Extreme SE'            # <-- AUSKOMMENTIERT
        ]
        
        gwls = config.GLOBAL_WARMING_LEVELS
        seasons = ['DJF', 'JJA']
        
        num_rows = len(gwls) * len(seasons)
        num_cols = len(storyline_order)
        
        fig = plt.figure(figsize=(5 * num_cols, 5 * num_rows))
        gs = matplotlib.gridspec.GridSpec(num_rows, num_cols + 1, width_ratios=[10]*num_cols + [0.5], wspace=0.15, hspace=0.4)
        
        cf_ref = None

        row_idx = 0
        for gwl in gwls:
            for season in seasons:
                season_full = "Winter" if season == "DJF" else "Summer"
                for col_idx, storyline_name in enumerate(storyline_order):
                    ax = cast(Any, fig.add_subplot(gs[row_idx, col_idx], projection=ccrs.PlateCarree()))
                    
                    storyline_title_formatted = storyline_name.replace(" & ", " &\n")
                    main_title_part = f'GWL {gwl}°C, {season_full} ({season})'

                    data = map_data.get(gwl, {}).get(season, {}).get(storyline_name)
                    
                    if data:
                        change_map = data['mean_change_map']
                        hist_map = data['historical_mean_map']
                        
                        plot_res = Visualizer.plot_u850_change_map(
                            ax, u850_change_data=change_map.data, 
                            historical_mean_contours=hist_map.data,
                            lons=change_map.lon.values, lats=change_map.lat.values,
                            title=main_title_part, 
                            season_label=storyline_title_formatted,
                            vmin=-2.5, vmax=2.5
                        )
                        if plot_res is not None:
                            cf, _ = plot_res
                            cf_ref = cf
                    else:
                        ax.set_title(f'GWL {gwl}°C, {season}\n{storyline_name}', fontsize=10)
                        ax.text(0.5, 0.5, "N/A", ha='center', va='center', transform=ax.transAxes)
                        ax.set_xticks([])
                        ax.set_yticks([])
                        
                row_idx += 1

        if cf_ref:
            cax = fig.add_subplot(gs[:, -1])
            cbar = fig.colorbar(cf_ref, cax=cax, extend='both')
            cbar.set_label('U850 Change (m/s)', fontsize=12)

        contour_handle = plt.Line2D([0], [0], color='black', lw=0.8, label='Historical Mean U850 (m/s)')
        fig.legend(handles=[contour_handle], loc='lower center', bbox_to_anchor=(0.5, 0.01),
                   ncol=1, fontsize=12, frameon=True, edgecolor='gray')

        hist_period_text = f"Historical Reference: {config.CMIP6_ANOMALY_REF_START}-{config.CMIP6_ANOMALY_REF_END}"
        fig.suptitle(f'Storyline-Based U850 Zonal Wind Change for {scenario.upper()}\n({hist_period_text})', fontsize=18, weight='bold')
                
        fig.tight_layout(rect=(0.01, 0.04, 0.95, 0.96))
        
        # MODIFIED: Filename now includes scenario
        filename_out = os.path.join(config.PLOT_DIR, f"storyline_u850_change_maps_{scenario}.png")
        plt.savefig(filename_out, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
    @staticmethod
    def plot_jet_cross_season_relationship(cmip6_results, scenario):
        """
        Creates a scatter plot to analyze the relationship between Summer Jet Latitude
        and Winter Jet Speed changes for different Global Warming Levels.
        """
        logging.info(f"Plotting cross-season jet relationship for {scenario}...")
        Visualizer.ensure_plot_dir_exists()

        all_deltas = cmip6_results.get('all_individual_model_deltas_for_plot')
        if not all_deltas:
            logging.warning("Cannot plot cross-season jet relationship: Missing delta values.")
            return

        gwls_to_plot = [2.0, 3.0]
        fig, axs = plt.subplots(1, 2, figsize=(16, 7), sharey=True, squeeze=False)
        axs = axs.flatten()

        # Define the variables for x and y axes
        x_jet_key = 'JJA_JetLat'
        y_jet_key = 'DJF_JetSpeed'

        for i, gwl in enumerate(gwls_to_plot):
            ax = axs[i]
            
            # Extract delta values for the specific variables and GWL
            x_deltas = all_deltas.get(x_jet_key, {}).get(gwl, {})
            y_deltas = all_deltas.get(y_jet_key, {}).get(gwl, {})

            if not x_deltas or not y_deltas:
                ax.text(0.5, 0.5, "Data Missing", ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"GWL {gwl}°C", fontsize=12)
                continue

            # Align data using common model keys
            common_models = sorted(list(set(x_deltas.keys()) & set(y_deltas.keys())))
            if not common_models:
                ax.text(0.5, 0.5, "No Common Models", ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"GWL {gwl}°C", fontsize=12)
                continue
                
            x_vals = np.array([x_deltas[m] for m in common_models])
            y_vals = np.array([y_deltas[m] for m in common_models])

            # Plot scatter of individual models
            ax.scatter(x_vals, y_vals, color='gray', alpha=0.7, s=30, label=f'CMIP6 Models (N={len(common_models)})')

            # Calculate and plot the linear regression fit
            slope, intercept, r_value, p_value, _ = StatsAnalyzer.calculate_regression(x_vals, y_vals)
            
            if not np.isnan(slope):
                x_fit = np.array(ax.get_xlim())
                y_fit = intercept + slope * x_fit
                
                # Add significance stars to p-value
                p_str = ""
                if p_value < 0.01: p_str = "**"
                elif p_value < 0.05: p_str = "*"

                fit_label = f'Fit (r={r_value:.2f}{p_str})'
                ax.plot(x_fit, y_fit, color='red', linestyle='--', linewidth=2, label=fit_label)

            # --- Formatting ---
            ax.set_xlabel('Change in Summer Jet Latitude (°Lat)', fontsize=11)
            if i == 0:
                ax.set_ylabel('Change in Winter Jet Speed (m/s)', fontsize=11)
            
            ax.set_title(f"GWL {gwl}°C", fontsize=14, weight='bold')
            ax.grid(True, linestyle=':', alpha=0.7)
            ax.axhline(0, color='black', lw=0.8, linestyle='-')
            ax.axvline(0, color='black', lw=0.8, linestyle='-')
            ax.legend(fontsize=10)

        ref_period_changes = f"{Config.CMIP6_ANOMALY_REF_START}-{Config.CMIP6_ANOMALY_REF_END}"
        fig.suptitle(f"Cross-Season Jet Relationship for {scenario.upper()}\n(Changes relative to {ref_period_changes})",
                     fontsize=16, weight='bold')
        
        fig.tight_layout(rect=(0.02, 0.02, 1, 0.93))
        
        filename = os.path.join(Config.PLOT_DIR, f"cmip6_jet_cross_season_relationship_{scenario}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        
    @staticmethod
    def plot_storyline_lnwl_monthly_distribution(distribution_data, scenario, config, lnwl_threshold):
        """
        Zeichnet ein Grid-Plot (GWL x Storyline) der monatlichen LNWL-Verteilung.
        Jeder Subplot vergleicht QOBS (Baseline) mit der CMIP6-Storyline-Zukunft.
        """
        logging.info(f"Zeichne monatliches LNWL-Verteilungs-Grid für {scenario}...")
        Visualizer.ensure_plot_dir_exists()

        fig = None
        try:
            # --- 1. Daten und Plot-Struktur vorbereiten ---
            qobs_baseline_data = distribution_data.get('qobs_baseline')
            storyline_data = distribution_data.get('storylines')

            if not qobs_baseline_data or not storyline_data:
                logging.warning("LNWL-Verteilungsdaten unvollständig. Plot wird übersprungen.")
                return

            # Plot-Reihenfolge definieren
            # (Diese muss mit der Reihenfolge in config.py übereinstimmen)
            storyline_plot_order = [
                'MMM', 
                'Slow Jet & Northward Shift', 
                'Fast Jet & Northward Shift',
                'Slow Jet & Southward Shift', 
                'Fast Jet & Southward Shift'
            ]
            gwls = config.GLOBAL_WARMING_LEVELS
            
            month_order = [7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6]
            month_labels = ['Jul', 'Aug', 'Sep', 'Okt', 'Nov', 'Dez', 'Jan', 'Feb', 'Mrz', 'Apr', 'Mai', 'Jun']
            
            # QOBS-Daten (Baseline) vorbereiten
            qobs_hist_period = "1995-2014"
            df_qobs = pd.DataFrame.from_dict(qobs_baseline_data, orient='index', columns=[f'QOBS ({qobs_hist_period})'])
            df_qobs = df_qobs.reindex(month_order)

            # --- 2. Plot-Grid erstellen ---
            nrows = len(gwls)
            ncols = len(storyline_plot_order)
            fig = None
            
            fig, axs = plt.subplots(
                nrows, ncols, 
                figsize=(ncols * 5, nrows * 4.5), 
                sharex=True, sharey=True,
                squeeze=False # Stellt sicher, dass axs immer ein 2D-Array ist
            )
            
            plt.style.use('seaborn-v0_8-whitegrid')
            bar_width = 0.4
            x_pos = np.arange(len(month_labels))

            # --- 3. Durch Subplots iterieren ---
            for row, gwl in enumerate(gwls):
                for col, storyline_name in enumerate(storyline_plot_order):
                    ax = axs[row, col]
                    
                    # Zukunftsdaten für diesen Subplot holen
                    future_data = storyline_data.get(gwl, {}).get(storyline_name)
                    if future_data is None:
                        future_data = {m: 0 for m in range(1, 13)} # Fallback
                    
                    col_name_future = f"Future ({storyline_name})"
                    df_future = pd.DataFrame.from_dict(future_data, orient='index', columns=[col_name_future])
                    df_future = df_future.reindex(month_order)
                    
                    # QOBS-Baseline plotten
                    ax.bar(x_pos - bar_width/2, df_qobs[f'QOBS ({qobs_hist_period})'], bar_width, 
                           label=f'QOBS ({qobs_hist_period})', color='darkblue', alpha=0.9)
                    
                    # Storyline-Zukunft plotten
                    ax.bar(x_pos + bar_width/2, df_future[col_name_future], bar_width, 
                           label=col_name_future, color='crimson', alpha=0.9)

                    # --- 4. Formatierung pro Subplot ---
                    ax.grid(axis='y', linestyle=':', alpha=0.7)
                    ax.grid(axis='x', linestyle='none')
                    ax.set_xticks(x_pos)
                    
                    # X-Achsen-Beschriftung (nur in der untersten Reihe)
                    if row == nrows - 1:
                        ax.set_xticklabels(month_labels, rotation=90, fontsize=10)
                    
                    # Spalten-Titel (Storyline-Namen, nur in der obersten Reihe)
                    if row == 0:
                        ax.set_title(storyline_name.replace(' & ', ' &\n'), fontsize=12, weight='bold')
                    
                    # Reihen-Titel (GWL, nur in der ersten Spalte)
                    if col == 0:
                        ax.set_ylabel(f'GWL {gwl}°C\n(% aller LNWL-Tage)', fontsize=11, weight='bold')
            
            # --- 5. Finale Formatierung der Gesamt-Figur ---
            # Y-Achse für alle Subplots setzen
            max_y = max(df_qobs.max().max(), max(storyline_data[gwl][sn][m] for gwl in gwls for sn in storyline_plot_order if storyline_data[gwl].get(sn) for m in range(1,13)))
            axs[0, 0].set_ylim(top=max_y * 1.15) 
            axs[0, 0].set_xticks(x_pos) # Stellt sicher, dass Ticks gesetzt sind, auch wenn sharex=True

            # Legende
            handles, labels = axs[0, 0].get_legend_handles_labels()
            # Wir brauchen nur eine Legende für QOBS vs. Future
            simple_labels = [f'QOBS ({qobs_hist_period})', f'CMIP6 Zukunft (pro Storyline)']
            simple_handles = [
                mpatches.Patch(color='darkblue', label=simple_labels[0]),
                mpatches.Patch(color='crimson', label=simple_labels[1])
            ]
            fig.legend(handles=simple_handles, labels=simple_labels, 
                       loc='lower center', bbox_to_anchor=(0.5, 0.01), 
                       ncol=2, fontsize=12, frameon=True)
            
            # Haupttitel
            title = f"Monatliche Verteilung von Niedrigwasser-Ereignissen (Abfluss < {lnwl_threshold:.0f} m³/s) für {scenario.upper()}"
            fig.suptitle(title, fontsize=16, weight='bold', y=0.99)
            
            fig.tight_layout(rect=(0.03, 0.05, 1, 0.95), h_pad=2.0, w_pad=0.5)
            
            filename = os.path.join(config.PLOT_DIR, f"storyline_lnwl_monthly_distribution_{scenario}.png")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close(fig)
            logging.info(f"Monatliches LNWL-Grid-Plot gespeichert: {filename}")

        except Exception as e:
            logging.error(f"Fehler beim Zeichnen des LNWL-Grid-Plots: {e}")
            logging.error(traceback.format_exc())
            if fig is not None:
                plt.close(fig)
                
    @staticmethod
    def plot_storyline_lnwl_aggregation_comparison(results, config, scenario, lnwl_threshold=970.0):
        """
        Creates a high-impact 3x4 plot (added Full Year row) showing the change in 
        return periods for the LNWL threshold (< 970 m³/s) across four different 
        time aggregations (Daily, 7-Day, 30-Day, 3-Month).
        
        Layout:
        - Rows: Winter, Summer, Full Year (NEW)
        - Columns: Daily, 7-Day, 30-Day (MODIFIED), 3-Month Minimums
        - Plot Type: Horizontal Boxplots (Y-axis=Storylines, X-axis=Return Period)
        - Aesthetics: English labels, T_hist in legend, X-axis label on all plots,
                      DYNAMIC X-AXIS scaling per subplot.
        
        --- MODIFIED (Nov 5, 2025) ---
        - Changed 'Q_monthly_low' to 'Q_30day_low' and title to '30-Day Minimum'.
        - Changed grid to 3 rows (num_rows=3) to include 'full_year'.
        - Updated 'half_year_order' list to include 'full_year'.
        - Updated 'figsize' to be taller (14 instead of 10).
        - Updated logic for x-axis labels to only show on the new bottom row (row == 2).
        - Updated season_title logic to include 'Full Year'.
        
        --- USER-MODIFIKATION (Nov 6, 2025) v2 ---
        - sharex=False: X-Achsen werden manuell pro Spalte synchronisiert.
        - Logik in zwei Durchgängen:
          1. Plotten und Sammeln der "idealen Zoom-Limits" für jeden Subplot.
          2. Finden der weitesten Spanne (min/max) pro Spalte und Anwenden auf alle Plots der Spalte.
        - X-Achsen-Beschriftung: Wird nun auf allen Subplots angezeigt (User-Wunsch).
        
        --- USER-MODIFIKATION (Nov 6, 2025) v3 ---
        - Logik hinzugefügt, um in der "Full Year"-Zeile (row 2) NUR die 'MMM'-Storyline
          zu plotten, da die anderen Storylines saisonal (DJF/JJA) definiert sind.
        --- ENDE USER-MODIFIKATION ---
        """
        if not results or not config or 'thresholds' not in results or 'data' not in results:
            logging.warning(f"Cannot plot LNWL aggregation comparison for {scenario}: Missing results.")
            return

        logging.info(f"Plotting LNWL Aggregation Comparison (3x4 grid, 30-Day) for {scenario}...")
        Visualizer.ensure_plot_dir_exists()
        
        gwls_to_plot = config.GLOBAL_WARMING_LEVELS
        
        # --- 1. Define Event and Plot Structure (NOW 4 COLS, ENGLISH NAMES) ---
        event_plot_order = [
            ('Q_daily_low', 'Daily lowflow'),
            ('Q_7day_low', '7-Day lowflow'),
            ('Q_30day_low', '30-Day lowflow'),
            ('Q_3month_low', '3-Month lowflow')
        ]
        
        num_cols = len(event_plot_order) # Should be 4
        
        num_rows = 3 # Winter, Summer, Full Year
        half_year_order = ['winter', 'summer', 'full_year']
        
        fig, axs = plt.subplots(
            num_rows, num_cols, 
            figsize=(7 * num_cols, 14), 
            squeeze=False, 
            sharey=True,  # Y-Achse (Storylines) wird geteilt
            sharex=False  # --- MODIFIKATION v2: Muss False sein für manuelle Steuerung ---
        )
        plt.style.use('seaborn-v0_8-whitegrid')

        # Y-Axis-Setup (Storylines)
        storyline_order = [
            'MMM',
            'Slow Jet & Northward Shift',
            'Fast Jet & Northward Shift',
            'Slow Jet & Southward Shift',
            'Fast Jet & Southward Shift',
        ]
        
        num_storylines = len(storyline_order)
        y_limits = (num_storylines - 0.5, -0.5) # For inverted Y-axis (MMM top)
        y_ticks = np.arange(len(storyline_order))
        y_tick_labels = [s.replace(' & ', ' &\n') for s in storyline_order] # English labels

        gwl_colors = {f'+{gwls_to_plot[0]}°C': '#ff7f0e', f'+{gwls_to_plot[1]}°C': '#d62728'} # Orange/Red
        
        # --- 2. Daten für Plotting vorbereiten ---
        plot_data_list = []
        for gwl in gwls_to_plot:
            for half_year in half_year_order:
                for storyline in storyline_order:
                    for event_key, _ in event_plot_order:
                        event_data = results['data'].get(gwl, {}).get(half_year, {}).get(storyline, {}).get(event_key)
                        if event_data and 'future_return_periods_all_models' in event_data:
                            for model_period in event_data['future_return_periods_all_models']:
                                if np.isfinite(model_period):
                                    plot_data_list.append({
                                        'half_year': half_year,
                                        'event': event_key,
                                        'storyline': storyline,
                                        'gwl': f'+{gwl}°C',
                                        'return_period': model_period,
                                    })
        
        if not plot_data_list:
            logging.warning(f"No finite LNWL aggregation data to plot for {scenario}.")
            plt.close(fig)
            return

        df_plot = pd.DataFrame(plot_data_list)
        
        # --- MODIFIKATION v2: Speicher für die X-Achsen-Limits pro Spalte ---
        column_x_limits = {c: [] for c in range(num_cols)}
        
        # --- 3. Plotting-Schleife (Pass 1: Plotten & Limits sammeln) ---
        for row, half_year in enumerate(half_year_order):
            for col, (event_key, event_title) in enumerate(event_plot_order):
                ax = axs[row, col]
                ax.set_ylim(y_limits)
                ax.invert_yaxis() # Show MMM at the top
                
                # --- Titel für Subplots (Spalten-Titel) ---
                if row == 0: # Nur in der obersten Zeile
                    ax.set_title(event_title, fontsize=12, weight='bold')
                
                # Filter data for this subplot
                data_subset = df_plot[(df_plot['half_year'] == half_year) & (df_plot['event'] == event_key)]
                
                legend_handles = [] # For this specific subplot's legend
                all_data_for_lims = [] # Wichtig: für jeden Plot neu initialisieren
                
                # --- MODIFIKATION v3: Nur MMM für 'full_year' plotten ---
                if half_year == 'full_year':
                    plot_data_for_ax = data_subset[data_subset['storyline'] == 'MMM']
                else:
                    plot_data_for_ax = data_subset
                # --- ENDE MODIFIKATION v3 ---

                if not plot_data_for_ax.empty: # <-- Geändert zu plot_data_for_ax
                    # Plot Horizontal Boxplots
                    sns.boxplot(data=plot_data_for_ax, y='storyline', x='return_period', hue='gwl',
                                order=storyline_order, palette=gwl_colors,
                                ax=ax, linewidth=1.2, showfliers=False, orient='h',
                                boxprops={'alpha': 0.85})
                    sns.stripplot(data=plot_data_for_ax, y='storyline', x='return_period', hue='gwl',
                                order=storyline_order, palette=gwl_colors,
                                ax=ax, dodge=True, jitter=0.15, size=4, 
                                edgecolor='gray', linewidth=0.5, alpha=0.9, orient='h',
                                legend=False)
                    
                    all_data_for_lims.extend(plot_data_for_ax['return_period'].dropna().values) # <-- Geändert
                else:
                    ax.text(0.5, 0.5, "Data N/A", ha='center', va='center', transform=ax.transAxes, color='gray')

                # Plot Historical Return Period (T_hist) als vertikale Linie
                threshold_data = results.get('thresholds', {}).get(half_year, {}).get(event_key, {})
                hist_period = threshold_data.get('hist_return_period')
                if hist_period is not None and np.isfinite(hist_period):
                    hist_label = f'Hist. T = {hist_period:.1f} yrs'
                    line = ax.axvline(x=hist_period, color='skyblue', linestyle='--', linewidth=3, zorder=5, label=hist_label)
                    legend_handles.append(line)
                    all_data_for_lims.append(hist_period)

                # --- Achsen-Formatierung ---
                ax.set_xscale('log')
                
                # Setze Ticks und Formatter
                ax.set_xticks([1, 2, 5, 10, 20, 50, 100, 150, 200, 500])
                ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, pos: f'{x:.0f}'))
                ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
                
                # --- MODIFIKATION v2: Limits berechnen und speichern, statt sie zu setzen ---
                if all_data_for_lims:
                    min_val = np.min(all_data_for_lims)
                    max_val = np.max(all_data_for_lims)
                    x_min_limit = max(0.8, min_val * 0.8) # "Zoom In" Minimum
                    x_max_limit = max_val * 1.5           # "Zoom In" Maximum
                    # Diese Logik stellt sicher, dass der "Zoom" nicht zu extrem ist
                    if (max_val / min_val) < 5: 
                        x_max_limit = max(x_max_limit, min_val * 5)
                    
                    # Speichere die berechneten Limits für die Spalte
                    column_x_limits[col].append((x_min_limit, x_max_limit))
                else:
                    # Speichere Fallback-Limits
                    column_x_limits[col].append((0.8, 100))
                # --- ENDE MODIFIKATION v2 ---

                ax.grid(axis='y', linestyle='none')
                ax.grid(axis='x', linestyle=':', which='both')
                
                # --- MODIFIKATION v2 (User-Wunsch): X-Achsen-Beschriftung auf ALLEN Subplots ---
                ax.set_xlabel('Return Period (Years)', fontsize=11)
                ax.xaxis.set_tick_params(labelbottom=True)
                # --- ENDE MODIFIKATION v2 ---
                
                # Y-Achsen-Label (nur in der ersten Spalte)
                if col == 0:
                    if half_year == 'winter':
                        season_title = "Winter Half-Year\n(Nov - Apr)" # <<< KORRIGIERT >>>
                    elif half_year == 'summer':
                        season_title = "Summer Half-Year\n(May - Oct)" # <<< KORRIGIERT >>>
                    else: # 'full_year'
                        season_title = "Full Year\n(Jan - Dec)"
                    ax.set_ylabel(season_title, fontsize=12, weight='bold', labelpad=15)
                
                # Legende (IN JEDEM PLOT)
                gwl_patches = [mpatches.Patch(color=gwl_colors[gwl_label], label=gwl_label) for gwl_label in gwl_colors]
                all_handles = gwl_patches + legend_handles
                all_handles.sort(key=lambda x: "Hist." in x.get_label()) 
                ax.legend(handles=all_handles, loc='upper right', fontsize='small', frameon=True, facecolor='white', framealpha=0.8)
                
                # n=X/Y Annotationen (Anzahl Modelle)
                for i, storyline in enumerate(storyline_order):
                    
                    # --- MODIFIKATION v3: Annotationen für 'full_year' auf 'MMM' beschränken ---
                    if half_year == 'full_year' and storyline != 'MMM':
                        continue
                    # --- ENDE MODIFIKATION v3 ---

                    y_base = y_ticks[i]
                    for j, gwl in enumerate(gwls_to_plot):
                        y_offset = -0.2 + (j * 0.4) # Position für GWL bar
                        gwl_label = f'+{gwl}°C'
                        event_data_gwl = results['data'].get(gwl, {}).get(half_year, {}).get(storyline, {}).get(event_key)
                        if event_data_gwl and 'model_count_X' in event_data_gwl:
                            X, Y = event_data_gwl['model_count_X'], event_data_gwl['model_count_Y']
                            ax.text(0.98, y_base + y_offset, f"n={X}/{Y}", 
                                    transform=ax.get_yaxis_transform(),
                                    horizontalalignment='right', fontsize=7, weight='bold', 
                                    color=gwl_colors[gwl_label],
                                    bbox=dict(facecolor='white', alpha=0.6, pad=0.1, edgecolor='none'))

        # --- 4. Zweiter Durchlauf: Gesammelte X-Limits anwenden (Pass 2) ---
        logging.info("Applying shared column X-limits based on widest 'zoomed' range...")
        for col, limits_list in column_x_limits.items():
            if limits_list: # Stellen sicher, dass die Liste nicht leer ist
                # Finde das absolute Minimum und Maximum aus allen Limits dieser Spalte
                final_min_lim = min(l[0] for l in limits_list)
                final_max_lim = max(l[1] for l in limits_list)
                
                # Wende dieses finale Limit auf alle Zeilen in dieser Spalte an
                for row in range(num_rows):
                    axs[row, col].set_xlim(left=final_min_lim, right=final_max_lim)
        # --- ENDE MODIFIKATION v2 ---

        # --- 5. Finale Formatierung der Gesamt-Figur ---
        fig.suptitle(f"Change in Return Period of Low Navigable Water Level (LNWL < {lnwl_threshold:.0f} m³/s) for {scenario.upper()}",
                    fontsize=16, weight='bold', y=0.99)
        
        fig.tight_layout(rect=(0.05, 0.05, 0.98, 0.95), h_pad=3.0, w_pad=2.5)
        
        filename = os.path.join(config.PLOT_DIR, f"storyline_lnwl_aggregation_comparison_{scenario}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logging.info(f"Saved LNWL aggregation comparison plot (3x4 grid, 30-Day, FullYear) to {filename}")

    @staticmethod
    def plot_final_figure_2_shift_and_verification(return_period_results, config, scenario):
        """
        Reconstructs Figure 2 showing Historical Event Counts and Future Counts in a combined 90°-rotated 1x2 subplot layout.
        The x-axis represents the evolution from Historical to +2.0°C and +3.0°C GWL, and the y-axis represents the Event Counts.
        """
        logging.info(f"Plotting Final Figure 2 (Combined Shift and Verification) for {scenario}...")
        Visualizer.ensure_plot_dir_exists()

        if not return_period_results or 'data' not in return_period_results or 'historical_verification' not in return_period_results:
            logging.warning("Missing data for Final Figure 2.")
            return

        hist_data = return_period_results.get('historical_verification', {})
        future_data = return_period_results.get('data', {})
        
        # GWL selection based on scenario
        gwls_to_plot = sorted([g for g in future_data.keys() if g in config.GLOBAL_WARMING_LEVELS])
        if scenario.lower() == 'ssp245':
            gwls_to_plot = [g for g in gwls_to_plot if g == 2.0]
            
        target_event_substring = "30Q10"
        winter_keys = list(return_period_results['thresholds'].get('winter', {}).keys())
        summer_keys = list(return_period_results['thresholds'].get('summer', {}).keys())
        
        low_key_winter = next((k for k in winter_keys if target_event_substring in k and 'low' in k.lower()), None)
        low_key_summer = next((k for k in summer_keys if target_event_substring in k and 'low' in k.lower()), None)

        if not low_key_winter or not low_key_summer:
            logging.warning("30Q10 event keys not found for Final Figure 2.")
            return

        # --- PREPARE HISTORICAL COUNTS MAPS ---
        summer_hist_counts = hist_data[low_key_summer].get('summer_counts', [])
        summer_hist_years = hist_data[low_key_summer].get('summer_years', [])
        summer_hist_keys = hist_data[low_key_summer].get('historical_keys', [])
        summer_hist_map = {k: c * float(config.GWL_YEARS_WINDOW) / y for k, c, y in zip(summer_hist_keys, summer_hist_counts, summer_hist_years) if y > 0}

        winter_hist_counts = hist_data[low_key_winter].get('winter_counts', [])
        winter_hist_years = hist_data[low_key_winter].get('winter_years', [])
        winter_hist_keys = hist_data[low_key_winter].get('historical_keys', [])
        winter_hist_map = {k: c * float(config.GWL_YEARS_WINDOW) / y for k, c, y in zip(winter_hist_keys, winter_hist_counts, winter_hist_years) if y > 0}

        # --- SETUP SUBPLOTS GRID (1x2, side-by-side) ---
        fig, axs = plt.subplots(1, 2, figsize=(10.0, 5.5))
        
        # Season labels & titles
        seasons = ['summer', 'winter']
        season_names = {'summer': 'Summer Half-Year', 'winter': 'Winter Half-Year'}
        event_keys = {'summer': low_key_summer, 'winter': low_key_winter}
        hist_maps = {'summer': summer_hist_map, 'winter': winter_hist_map}
        
        gwl_display_order = ['Historical'] + [f'+{gwl}°C GWL' for gwl in gwls_to_plot]

        # Get the number of extreme models from the future data
        n_extreme = 14
        for gwl in gwls_to_plot:
            try:
                ext_models = future_data[gwl]['summer']['Extreme Models'][low_key_summer].get('future_keys_all_models', [])
                if ext_models:
                    n_extreme = len(ext_models)
                    break
            except KeyError:
                try:
                    ext_models = future_data[gwl]['winter']['Extreme Models'][low_key_winter].get('future_keys_all_models', [])
                    if ext_models:
                        n_extreme = len(ext_models)
                        break
                except KeyError:
                    continue

        for col, season in enumerate(seasons):
            ax = axs[col]
            event_key = event_keys[season]
            hist_map = hist_maps[season]
            
            # Construct unified records for this season
            records = []
            
            # Add historical records
            for k, C_hist in hist_map.items():
                records.append({
                    'GWL': 'Historical',
                    'Counts': C_hist,
                    'Category': 'Historical',
                    'Color': 'gray',
                    'Model': k
                })
            
            # Add future records
            for gwl in gwls_to_plot:
                gwl_label = f'+{gwl}°C GWL'
                try:
                    future_node = future_data[gwl][season]['MMM'][event_key]
                    fut_counts = future_node.get('future_counts_all_models', [])
                    fut_keys = future_node.get('future_keys_all_models', [])
                except KeyError:
                    continue
                
                try:
                    extreme_keys = future_data[gwl][season]['Extreme Models'][event_key].get('future_keys_all_models', [])
                except KeyError:
                    extreme_keys = []
                try:
                    non_extreme_keys = future_data[gwl][season]['Non-Extreme Models'][event_key].get('future_keys_all_models', [])
                except KeyError:
                    non_extreme_keys = []
                
                for k, C_fut in zip(fut_keys, fut_counts):
                    C_hist_scaled = hist_map.get(k, np.nan)
                    if np.isnan(C_hist_scaled):
                        continue
                    
                    if k in extreme_keys:
                        category = 'High-Freq.'
                        color = '#b2182b'
                    elif k in non_extreme_keys:
                        category = 'Low-Freq.'
                        color = '#2166ac'
                    else:
                        category = 'Other'
                        color = 'gray'
                        
                    records.append({
                        'GWL': gwl_label,
                        'Counts': C_fut,
                        'Category': category,
                        'Color': color,
                        'Model': k
                    })
            
            df = pd.DataFrame(records)
            
            if df.empty:
                ax.text(0.5, 0.5, "No Data", ha='center', va='center', transform=ax.transAxes)
                continue
                
            # Vertical Boxplot (orient='v', stage on x-axis, counts on y-axis)
            sns.boxplot(data=df, x='GWL', y='Counts', ax=ax,
                        order=gwl_display_order, color='lightgray',
                        showfliers=False, linewidth=1.0, width=0.5, orient='v',
                        boxprops={'alpha': 0.7}, medianprops={'color': 'black', 'linewidth': 2.5})
            
            # Stripplot with custom markers and ordered horizontal positioning (dodge/swarm)
            x_ticks_pos = np.arange(len(gwl_display_order))
            for idx_gwl, gwl_label in enumerate(gwl_display_order):
                key_df = df[df['GWL'] == gwl_label]
                
                # Group by the Counts value rounded to 4 decimal places
                grouped = key_df.groupby(key_df['Counts'].round(4))
                for counts_val, group in grouped:
                    # Sort the group by Category and Model to keep it ordered
                    category_order = {'High-Freq.': 0, 'Other': 1, 'Low-Freq.': 2, 'Historical': 3}
                    sorted_group = group.copy()
                    sorted_group['sort_key'] = sorted_group['Category'].map(category_order).fillna(4)
                    sorted_group = sorted_group.sort_values(by=['sort_key', 'Model'])
                    
                    N = len(sorted_group)
                    if N == 1:
                        offsets = [0.0]
                    else:
                        max_spread = 0.42
                        # We want a preferred spacing of 0.05, but if N is large we reduce spacing
                        # so that total spread does not exceed max_spread
                        dx = max_spread / (N - 1)
                        if dx > 0.05:
                            dx = 0.05
                        
                        # Calculate symmetric offsets centered at 0
                        offsets = [(i - (N - 1) / 2.0) * dx for i in range(N)]
                    
                    for idx_item, (_, row) in enumerate(sorted_group.iterrows()):
                        x_pos = x_ticks_pos[idx_gwl] + offsets[idx_item]
                        if row['Category'] == 'High-Freq.':
                            marker_style = '^'
                            color = '#b2182b'
                            zorder = 4
                            size = 7
                            alpha = 0.9
                        elif row['Category'] == 'Low-Freq.':
                            marker_style = 'v'
                            color = '#2166ac'
                            zorder = 4
                            size = 7
                            alpha = 0.9
                        elif row['Category'] == 'Historical':
                            marker_style = 'o'
                            color = 'black'
                            zorder = 3
                            size = 6.5
                            alpha = 0.6
                        else:
                            marker_style = 'o'
                            color = 'gray'
                            zorder = 3
                            size = 5
                            alpha = 0.4
                        ax.plot(x_pos, row['Counts'], marker=marker_style, color=color,
                                markersize=size, alpha=alpha, linestyle='None', zorder=zorder)
            
            # Draw horizontal median lines for Extreme (red) and Non-Extreme (blue) groups
            for idx_gwl, gwl_label in enumerate(gwl_display_order):
                x_pos_center = x_ticks_pos[idx_gwl]
                for cat, cat_color in [('High-Freq.', '#b2182b'), ('Low-Freq.', '#2166ac')]:
                    cat_changes = df[(df['GWL'] == gwl_label) & (df['Category'] == cat)]['Counts'].values
                    if len(cat_changes) > 0:
                        median_change = np.nanmedian(cat_changes)
                        if np.isfinite(median_change):
                            ax.hlines(y=median_change, xmin=x_pos_center - 0.25, xmax=x_pos_center + 0.25,
                                      colors=cat_color, linestyles='-', linewidth=2.5, zorder=5)
            
            # Horizontal reference line at historical median across the entire subplot
            scaled_counts = list(hist_map.values())
            hist_median = np.nanmedian(scaled_counts) if scaled_counts else np.nan
            if np.isfinite(hist_median):
                ax.axhline(hist_median, color='black', linestyle='--', linewidth=1.2, zorder=2)
            
            panel_letter = 'a' if col == 0 else 'b'
            ax.set_title(f"({panel_letter}) {season_names[season]}", weight='bold', loc='left', fontsize=12)
            ax.set_xlabel("")
            if col == 0:
                ax.set_ylabel("Event Counts per 31 years", fontsize=11)
            else:
                ax.set_ylabel("")
            ax.set_ylim(-0.5, 14)
            ax.set_yticks(range(0, 15, 2))
            ax.grid(True, which='major', axis='y', linestyle=':', alpha=0.7)
            
            ax.set_xticks(range(len(gwl_display_order)))
            ax.set_xticklabels(gwl_display_order, fontsize=10)
            ax.tick_params(axis='x', which='major', labelsize=10)
            ax.tick_params(axis='y', which='major', labelsize=10)
        
        # Overall figure titles & layouts
        scenario_title = Visualizer._format_scenario_title(scenario)
        fig.suptitle(f"Historical and Future Counts of 30Q10 Low-Flow Events - {scenario_title}",
                     fontsize=16, weight='bold', y=0.97)
        
        plt.tight_layout(rect=(0.02, 0.08, 0.98, 0.92), h_pad=2.0, w_pad=2.0)
        
        # Legend at the bottom (ordered for Column-Major legend layout with ncol=4)
        from matplotlib.lines import Line2D
        legend_handles = [
            Line2D([0], [0], marker='^', color='w', markerfacecolor='#b2182b', label=f'High-Freq. (Top {n_extreme})', markersize=8),
            Line2D([0], [0], color='#b2182b', lw=2.0, label='Median (High-Freq.)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='black', alpha=0.6, label='Historical Models (31y eq.)', markersize=6.5),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', alpha=0.4, label='Other Future Models', markersize=5),
            Line2D([0], [0], marker='v', color='w', markerfacecolor='#2166ac', label=f'Low-Freq. (Bottom {n_extreme})', markersize=8),
            Line2D([0], [0], color='#2166ac', lw=2.0, label='Median (Low-Freq.)'),
            Line2D([0], [0], color='black', linestyle='--', lw=1.2, label='Historical Median'),
            Line2D([0], [0], color='black', lw=2.5, label='Multi-Model Median')
        ]
        fig.legend(handles=legend_handles, loc='lower center', ncol=4, fontsize=9.0, frameon=False, bbox_to_anchor=(0.5, 0.015))
        
        filename = os.path.join(config.PLOT_DIR, f"final_figure_2_regime_shift_and_verification_{scenario}.png")
        plt.savefig(filename, dpi=600, bbox_inches='tight')
        pdf_filename = os.path.join(config.PLOT_DIR, f"final_figure_2_regime_shift_and_verification_{scenario}.pdf")
        plt.savefig(pdf_filename, bbox_inches='tight')
        plt.close(fig)
        logging.info(f"Saved Final Figure 2 to {filename} and {pdf_filename}")

    @staticmethod
    def plot_core_finding_gev_panel(return_period_results, config, scenario):
        """
        Creates ERL Figure 3: Core Finding - Regime Shift in Extremes (30Q100).
        Layout: 2x2 Grid, but each plot uses a BROKEN X-AXIS (Left: Normal, Right: Extreme).
        INCLUDES: Broken Axis Fixes (No duplicate Y-labels, No overlapping X-labels on top row).
        """
        logging.info(f"Plotting Figure 3 (Core Finding Empirical Panel) with BROKEN AXIS for {scenario}...")
        Visualizer.ensure_plot_dir_exists()

        if not return_period_results or 'data' not in return_period_results:
            logging.warning("Missing data for Figure 3.")
            return

        # --- 1. Identify Events (30Q10) ---
        target_event_substring = "30Q10"
        winter_keys = list(return_period_results['thresholds']['winter'].keys())
        summer_keys = list(return_period_results['thresholds']['summer'].keys())
        
        low_key_winter = next((k for k in winter_keys if target_event_substring in k and 'low' in k.lower()), None)
        high_key_winter = next((k for k in winter_keys if target_event_substring in k and 'high' in k.lower()), None)
        low_key_summer = next((k for k in summer_keys if target_event_substring in k and 'low' in k.lower()), None)
        high_key_summer = next((k for k in summer_keys if target_event_substring in k and 'high' in k.lower()), None)
        
        # Config for the 2 logical plots (Low Flow ONLY)
        plot_configs = [
            {'half_year': 'summer', 'event_key': low_key_summer,  'base_title': 'a) Summer Half-Year: 30-Day Low Flow', 'row': 0, 'col_group': 0},
            {'half_year': 'winter', 'event_key': low_key_winter,  'base_title': 'b) Winter Half-Year: 30-Day Low Flow', 'row': 0, 'col_group': 1},
        ]

        # --- SETUP FIGURE (Standard 1x2) ---
        fig, axs = plt.subplots(1, 2, figsize=(16, 5))
        axs = axs.flatten()
        
        # --- MODIFIED: Correct spelling for SSP5-8.5 ---
        scenario_title = Visualizer._format_scenario_title(scenario)
        fig.suptitle(f"Shift in Return Periods of Extremes (30Q10) - {scenario_title}", fontsize=16, weight='bold', y=0.98)
        
        gwls_to_plot = config.GLOBAL_WARMING_LEVELS
        # MODIFIED: Add " GWL" to keys
        gwl_colors = {f'+{gwl}°C GWL': Visualizer.GWL_COLORS[gwl] for gwl in gwls_to_plot}
        
        storyline_data_keys = [
            'MMM', 'Extreme Models', 'Non-Extreme Models'
        ]
        storyline_display_order = [
            'Multi-Model Mean', 'High-Frequency', 'Low-Frequency'
        ]

        # Loop through the 4 logical plots
        for i, cfg in enumerate(plot_configs):
            ax = axs[i]
            event_key = cfg['event_key']
            half_year = cfg['half_year']
            
            # Title Construction
            thresh_meta = return_period_results['thresholds'][half_year][event_key]
            full_title = cfg['base_title']
            
            ax.set_title(full_title, loc='left', fontsize=11, weight='bold')

            # --- DATA COLLECTION ---
            plot_data = []
            mean_ci_data = []
            hist_rp = thresh_meta.get('hist_return_period')
            
            for storyline_key in storyline_data_keys:
                if storyline_key == 'MMM':
                    display_name = 'Multi-Model Mean'
                elif storyline_key == 'Extreme Models':
                    display_name = 'High-Frequency'
                elif storyline_key == 'Non-Extreme Models':
                    display_name = 'Low-Frequency'
                else:
                    display_name = storyline_key
                for gwl in gwls_to_plot:
                    gwl_label = f'+{gwl}°C GWL'
                    try:
                        event_data = return_period_results['data'][gwl][half_year][storyline_key][event_key]
                        if event_data:
                            if 'future_return_periods_all_models' in event_data:
                                rps = event_data['future_return_periods_all_models']
                                rps = [rp for rp in rps if np.isfinite(rp)]
                                for rp in rps:
                                    plot_data.append({'Storyline': display_name, 'GWL': gwl_label, 'Plot Pos': rp})
                            
                            rp_mean = event_data.get('future_return_period_mean', np.nan)
                            if np.isfinite(rp_mean):
                                mean_ci_data.append({
                                    'Storyline': display_name, 'GWL': gwl_label, 'Mean_Plot': rp_mean,
                                    'Low_Plot': np.nan,
                                    'High_Plot': np.nan
                                })
                    except KeyError: continue
            
            df = pd.DataFrame(plot_data)
            
            if df.empty:
                ax.text(0.5, 0.5, "No Data", ha='center', va='center', transform=ax.transAxes)
                continue

            # --- PLOTTING ---
            # Boxplots
            sns.boxplot(data=df, y='Storyline', x='Plot Pos', hue='GWL', ax=ax,
                        order=storyline_display_order, palette=gwl_colors,
                        showfliers=False, linewidth=1.0, width=0.7, orient='h',
                        boxprops={'alpha': 0.4})
            # Stripplots
            sns.stripplot(data=df, y='Storyline', x='Plot Pos', hue='GWL', ax=ax,
                          order=storyline_display_order, palette=gwl_colors,
                          dodge=True, jitter=0.15, size=6, alpha=0.6, legend=False, orient='h')
            
            # --- N=X/Y Annotations ---
            y_ticks_pos = np.arange(len(storyline_display_order))
            
            for i_story, storyline_display in enumerate(storyline_display_order):
                storyline_key = 'MMM' if storyline_display == 'Multi-Model Mean' else storyline_display
                y_base = y_ticks_pos[i_story]
                
                for j, gwl in enumerate(gwls_to_plot):
                    gwl_label = f'+{gwl}°C GWL'
                    y_offset = -0.15 + (j * 0.25)
                    
                    try:
                        event_data_gwl = return_period_results['data'][gwl][half_year][storyline_key][event_key]
                        if event_data_gwl and 'model_count_X' in event_data_gwl and 'model_count_Y' in event_data_gwl:
                            X = event_data_gwl['model_count_X']
                            Y = event_data_gwl['model_count_Y']
                            text_to_display = f"n={X}/{Y}"
                            
                            ax.text(0.98, y_base + y_offset, text_to_display, 
                                    transform=ax.get_yaxis_transform(), 
                                    horizontalalignment='right', verticalalignment='center',
                                    fontsize=7, weight='bold', 
                                    color=gwl_colors[gwl_label],
                                    bbox=dict(facecolor='white', alpha=0.6, pad=0.1, edgecolor='none'))
                    except Exception:
                        pass
            
            # Historical Line
            if hist_rp:
                ax.axvline(hist_rp, color='black', linestyle='--', linewidth=1.5)
            
            # Remove Legend from individual subplots
            if ax.get_legend(): ax.get_legend().remove()

            # --- AXIS LIMITS & SCALES ---
            ax.set_xscale('linear')
            ax.set_xlim(0, 35) # Fixed limit 0-35 as requested
            ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            
            ax.grid(True, which='major', axis='x', linestyle=':', alpha=0.7)
            
            # X-Axis Labels & Ticks
            ax.tick_params(axis='x', which='both', bottom=True, labelbottom=True)
            
            # Set Label for the only row
            ax.set_xlabel("Return Period (Years)", fontsize=10)
            
            # Y-Axis Labels Logic
            ax.set_ylabel('')
            if i % 2 == 0: # Left Column
                # Add newline to labels with '&'
                labels = [l.replace(' & ', ' &\n') for l in storyline_display_order]
                ax.set_yticks(range(len(labels)))
                ax.set_yticklabels(labels, fontsize=10)
            else: # Right Column
                ax.set_yticks([])
                ax.set_yticklabels([])

            # ax.invert_yaxis() # Removed: seaborn places index 0 (MMM) at the top by default

        # Shared Legend
        handles = []
        handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', alpha=0.5, label='Models'))
        handles.append(plt.Line2D([0], [0], color='black', linestyle='--', linewidth=1.5, label='Historical Return Period'))
        
        # MODIFIED: Legend matches new keys with "GWL"
        for gwl_label, color in gwl_colors.items():
            handles.append(mpatches.Patch(color=color, label=gwl_label))
            
        fig.legend(handles=handles, loc='lower center', ncol=5, bbox_to_anchor=(0.5, 0.02), frameon=False)

        plt.tight_layout(rect=(0, 0.05, 1, 0.96)) 
        
        filename = os.path.join(config.PLOT_DIR, f"Figure3_core_finding_regime_shift_{scenario}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logging.info(f"Saved Figure 3 to {filename}")

    @staticmethod
    def plot_mechanism_drivers_panel(cmip6_results, config, scenario):
        """
        Creates ERL Figure 5 (was Fig 4): Mechanism - Drivers (Temp & Precip) as BOXPLOTS.
        ROTATED: Variable changes on X-axis, Storylines on Y-axis.
        MODIFIED: Applies offset to TAS to show warming relative to 1850-1900.
        MODIFIED: Unified X-axes per column, Analysis Box Reference.
        MODIFIED: VISUAL LEGEND REMOVED and frameon=False applied.
        MODIFIED: Added 'Individual Models' and 'Median' to legend.
        """
        logging.info(f"Plotting Figure 5 (Mechanism Drivers Panel) with Boxplots (Horizontal) for {scenario}...")
        Visualizer.ensure_plot_dir_exists()

        if not cmip6_results:
            logging.warning("Missing data for Figure 5.")
            return

        # Extract required data structures
        all_deltas = cmip6_results.get('all_individual_model_deltas_for_plot', {})
        classification = cmip6_results.get('storyline_classification_2d', {})
        
        # NEU: Zugriff auf Zeitreihen für Offset
        metric_timeseries = cmip6_results.get('model_metric_timeseries', {})
        
        # Referenzperioden
        ref_1850_1900 = (config.CMIP6_PRE_INDUSTRIAL_REF_START, config.CMIP6_PRE_INDUSTRIAL_REF_END)
        ref_1995_2014 = (config.CMIP6_ANOMALY_REF_START, config.CMIP6_ANOMALY_REF_END)
        
        if not all_deltas or not classification:
            logging.warning("Missing delta or classification data for Figure 5.")
            return

        fig, axs = plt.subplots(2, 2, figsize=(16, 13))
        axs = axs.flatten()
        
        gwls_to_plot = config.GLOBAL_WARMING_LEVELS
        # MODIFIED: Add " GWL" to keys
        gwl_colors = {f'+{gwl}°C GWL': Visualizer.GWL_COLORS[gwl] for gwl in gwls_to_plot}
        
        # --- MODIFIED: Updated Titles ---
        plot_configs = [
            {'key': 'JJA_tas', 'title': 'a) Local summer temperature change', 'unit': '°C', 'ax': axs[0], 'type': 'TAS'},
            {'key': 'JJA_pr',  'title': 'b) Local summer precipitation change', 'unit': '%',  'ax': axs[1], 'type': 'PR'},
            {'key': 'DJF_tas', 'title': 'c) Local winter temperature change', 'unit': '°C', 'ax': axs[2], 'type': 'TAS'},
            {'key': 'DJF_pr',  'title': 'd) Local winter precipitation change', 'unit': '%',  'ax': axs[3], 'type': 'PR'},
        ]

        storyline_data_keys = [
            'MMM', 'Slow Jet & Northward Shift', 'Fast Jet & Northward Shift',
            'Slow Jet & Southward Shift', 'Fast Jet & Southward Shift',
        ]
        storyline_display_order = [
            'Multi-Model Mean', 'Slow Jet & Northward Shift', 'Fast Jet & Northward Shift',
            'Slow Jet & Southward Shift', 'Fast Jet & Southward Shift',
        ]
        
        # --- MODIFIED: New Title, No Subtitle ---
        scenario_title = Visualizer._format_scenario_title(scenario)
        main_title = f"Local drivers of change - {scenario_title}"
        fig.suptitle(f"{main_title}", fontsize=16, weight='bold', y=0.98)

        # Containers for collecting data limits
        all_tas_values = []
        all_pr_values = []

        # First Pass: Collect Data
        plot_data_storage = {}

        for i, p_conf in enumerate(plot_configs):
            key = p_conf['key']
            season_prefix = key.split('_')[0] 
            is_temp = 'tas' in key
            
            current_plot_data = []
            
            for gwl in gwls_to_plot:
                gwl_label = f'+{gwl}°C GWL' # MODIFIED: Matches new color keys
                
                for storyline_key_short in storyline_data_keys:
                    full_storyline_key = f"{season_prefix}_{storyline_key_short}"
                    models = classification.get(gwl, {}).get(full_storyline_key, [])
                    display_name = 'Multi-Model Mean' if storyline_key_short == 'MMM' else storyline_key_short
                    
                    for model in models:
                         val = all_deltas.get(key, {}).get(gwl, {}).get(model)
                         
                         # --- Offset-Berechnung für TAS ---
                         if is_temp and val is not None and np.isfinite(val):
                             ts = metric_timeseries.get(model, {}).get(key)
                             if ts is not None:
                                 try:
                                     mean_pi = ts.sel(season_year=slice(*ref_1850_1900)).mean().item()
                                     mean_ref = ts.sel(season_year=slice(*ref_1995_2014)).mean().item()
                                     
                                     if not np.isnan(mean_pi) and not np.isnan(mean_ref):
                                         # Offset: Wie viel wärmer war 1995-2014 im Vergleich zu 1850?
                                         offset = mean_ref - mean_pi
                                         val += offset
                                 except Exception:
                                     pass 
                         
                         if val is not None and np.isfinite(val):
                             current_plot_data.append({
                                 'Storyline': display_name,
                                 'GWL': gwl_label,
                                 'Change': val
                             })
                             if is_temp:
                                 all_tas_values.append(val)
                             else:
                                 all_pr_values.append(val)
            
            plot_data_storage[i] = pd.DataFrame(current_plot_data)

        # Calculate common limits
        tas_min = min(all_tas_values) if all_tas_values else 0
        tas_max = max(all_tas_values) if all_tas_values else 0
        pr_min = min(all_pr_values) if all_pr_values else 0
        pr_max = max(all_pr_values) if all_pr_values else 0
        
        # Add some padding
        tas_pad = (tas_max - tas_min) * 0.05
        pr_pad = (pr_max - pr_min) * 0.05
        tas_lims = (tas_min - tas_pad, tas_max + tas_pad)
        pr_lims = (pr_min - pr_pad, pr_max + pr_pad)

        # Second Pass: Plotting
        for i, p_conf in enumerate(plot_configs):
            ax: Any = p_conf['ax']
            df = plot_data_storage[i]
            
            if df.empty:
                ax.text(0.5, 0.5, "No Data", ha='center', va='center')
            else:
                sns.boxplot(data=df, y='Storyline', x='Change', hue='GWL', ax=ax,
                            order=storyline_display_order, palette=gwl_colors,
                            showfliers=False, linewidth=1.2, width=0.7, orient='h')
                
                sns.stripplot(data=df, y='Storyline', x='Change', hue='GWL', ax=ax,
                              order=storyline_display_order, palette=gwl_colors,
                              dodge=True, jitter=0.15, size=3, alpha=0.6, legend=False,
                              edgecolor='gray', linewidth=0.5, orient='h')

            # Formatting
            ax.set_title(p_conf['title'], weight='bold', loc='left', fontsize=12)
            ax.set_ylabel('')
            
            # --- MODIFIED: Label and Limits ---
            ax.set_xlabel('') # Reset first
            
            # Apply Unified Limit
            if p_conf['type'] == 'TAS':
                ax.set_xlim(tas_lims) 
            else:
                ax.set_xlim(pr_lims) 
            
            # Apply X-Label only for bottom row (indices 2 and 3)
            if i >= 2:
                if p_conf['type'] == 'TAS':
                    ax.set_xlabel(f"Warming rel. to 1850-1900 ({p_conf['unit']})", fontsize=10)
                else:
                    ax.set_xlabel(f"Change rel. to 1995-2014 ({p_conf['unit']})", fontsize=10)
            
            # Ensure tick labels are visible on all plots
            ax.tick_params(axis='x', labelbottom=True)

            ax.axvline(0, color='black', linewidth=0.8, linestyle='-') 
            ax.grid(True, axis='x', linestyle=':', alpha=0.7)
            
            # Y-Achsen-Labels nur links
            labels = [l.replace(' & ', ' &\n') for l in storyline_display_order]
            if i % 2 == 0:
                ax.set_yticklabels(labels, fontsize=10)
            else:
                ax.set_yticklabels([]) 

            ax.invert_yaxis()
            
            if ax.get_legend(): ax.get_legend().remove()

        # --- LEGEND SECTION ---
        handles, labels = axs[0].get_legend_handles_labels()
        # Sicherstellen, dass die Reihenfolge der GWLs stimmt
        unique_handles_labels = dict(zip(labels, handles))
        sorted_labels = sorted([l for l in unique_handles_labels.keys() if "GWL" in l])
        
        final_handles = [unique_handles_labels[l] for l in sorted_labels]
        final_labels = sorted_labels

        # --- NEU: Hinzufügen von 'Individual Models' und 'Median' ---
        final_handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                                        markeredgecolor='gray', markersize=6, label='Individual Models'))
        final_labels.append('Individual Models')

        final_handles.append(plt.Line2D([0], [0], marker='|', color='black', linestyle='None', 
                                        markersize=10, markeredgewidth=1.5, label='Median'))
        final_labels.append('Median')
        
        # KORREKTUR: ncol auf 4 erhöht, um die neuen Einträge unterzubringen
        fig.legend(final_handles, final_labels, loc='lower center', bbox_to_anchor=(0.5, 0.02), 
                   ncol=4, fontsize=12, frameon=False)

        plt.tight_layout(rect=(0, 0.08, 1, 0.95)) # Angepasstes Layout ohne die schematische Legende
        
        # --- CHANGED FILENAME TO FIGURE 5 ---
        filename = os.path.join(config.PLOT_DIR, f"Figure5_mechanism_drivers_summary_{scenario}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logging.info(f"Saved Figure 5 (formerly Figure 4) to {filename}")

    @staticmethod
    def plot_erl_figure4_lnwl_summary(results, config, scenario, lnwl_threshold=970.0):
        """
        Creates ERL Figure 4: Projected impact on inland navigation (LNWL).
        Layout: 1x2 Grid (Summer 30-Day, Summer 3-Month).
        """
        logging.info(f"Plotting ERL Figure 4 (Summer LNWL Impact) for {scenario}...")
        Visualizer.ensure_plot_dir_exists()

        if not results or 'data' not in results:
            logging.warning(f"Cannot plot Figure 4 for {scenario}: Missing LNWL data.")
            return

        # Config
        gwls_to_plot = config.GLOBAL_WARMING_LEVELS
        gwl_colors = {f'+{gwl}°C GWL': Visualizer.GWL_COLORS[gwl] for gwl in gwls_to_plot}
        
        # Plot structure: Only Summer, Only Long Duration events
        plot_configs = [
            {'event': 'Q_30day_low',  'title': 'a) Summer: 30-Day Low Water Events'},
            {'event': 'Q_3month_low', 'title': 'b) Summer: 3-Month Low Water Events'}
        ]
        
        half_year = 'summer'
        storyline_order = [
            'MMM', 'Slow Jet & Northward Shift', 'Fast Jet & Northward Shift',
            'Slow Jet & Southward Shift', 'Fast Jet & Southward Shift',
        ]
        
        fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
        
        # Prepare Data
        for i, p_conf in enumerate(plot_configs):
            ax = axs[i]
            event_key = p_conf['event']
            
            # Collect data for this panel
            plot_data_list = []
            for gwl in gwls_to_plot:
                for storyline in storyline_order:
                    event_data = results['data'].get(gwl, {}).get(half_year, {}).get(storyline, {}).get(event_key)
                    if event_data and 'future_return_periods_all_models' in event_data:
                        for model_period in event_data['future_return_periods_all_models']:
                            if np.isfinite(model_period):
                                plot_data_list.append({
                                    'storyline': storyline,
                                    'GWL': f'+{gwl}°C GWL',
                                    'return_period': model_period,
                                })
            
            if not plot_data_list:
                ax.text(0.5, 0.5, "Data Missing", ha='center', va='center', transform=ax.transAxes)
                continue
                
            df = pd.DataFrame(plot_data_list)
            
            # Plot Boxplots
            sns.boxplot(data=df, y='storyline', x='return_period', hue='GWL',
                        order=storyline_order, palette=gwl_colors,
                        ax=ax, linewidth=1.2, showfliers=False, orient='h',
                        boxprops={'alpha': 0.7})
            
            sns.stripplot(data=df, y='storyline', x='return_period', hue='GWL',
                          order=storyline_order, palette=gwl_colors,
                          ax=ax, dodge=True, jitter=0.15, size=4,
                          edgecolor='gray', linewidth=0.5, alpha=0.8, orient='h')
            
            # Historical Reference Line
            threshold_data = results.get('thresholds', {}).get(half_year, {}).get(event_key, {})
            hist_period = threshold_data.get('hist_return_period')
            if hist_period and np.isfinite(hist_period):
                ax.axvline(x=hist_period, color='skyblue', linestyle='--', linewidth=2.5, zorder=0, label='Historical Return Period')
                # Add text annotation for hist value
                ax.text(hist_period, -0.6, f'Hist: {hist_period:.1f} yrs', color='#1f77b4', 
                        ha='center', fontsize=9, weight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))

            # Formatting
            ax.set_title(p_conf['title'], weight='bold', loc='left', fontsize=12)
            ax.set_xscale('log')
            
            # Specific Ticks for readability
            ax.set_xticks([1, 2, 5, 10, 20, 50, 100])
            ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax.set_xlim(0.8, 150) # Zoom to relevant range
            
            ax.set_xlabel('Return Period (Years)', fontsize=11)
            ax.grid(True, axis='x', linestyle=':', which='both')
            
            if i == 0:
                labels = [s.replace(' & ', ' &\n') for s in storyline_order]
                ax.set_yticklabels(labels, fontsize=10)
                ax.set_ylabel('')
            else:
                ax.set_ylabel('')
            
            ax.invert_yaxis() # MMM top
            if ax.get_legend(): ax.get_legend().remove()

        # Shared Legend
        handles = []
        handles.append(plt.Line2D([0], [0], color='skyblue', linestyle='--', linewidth=2.5, label='Historical Return Period'))
        for gwl_label, color in gwl_colors.items():
            handles.append(mpatches.Patch(color=color, label=gwl_label))
        
        fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=3, fontsize=12, frameon=False)
        
        scenario_title = Visualizer._format_scenario_title(scenario)
        fig.suptitle(f"Projected impact on inland navigation reliability (LNWL < {lnwl_threshold:.0f} m³/s) - {scenario_title}", 
                     fontsize=16, weight='bold', y=0.98)
        
        plt.tight_layout(rect=(0.02, 0.08, 0.98, 0.94))
        
        filename = os.path.join(config.PLOT_DIR, f"Figure4_impact_navigation_lnwl_{scenario}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logging.info(f"Saved ERL Figure 4 to {filename}")

    @staticmethod
    def plot_historical_seasonal_verification(results, config, scenario):
        """
        Plots a bar chart verifying the seasonal return periods.
        Row 1: Historical Verification (based on Annual Threshold).
        Row 2+: Future Projections at GWLs (Annual Threshold vs Future Seasonal Distributions).
        """
        try:
            if not results or 'historical_verification' not in results:
                logging.warning("No historical verification data found.")
                return

            logging.info("Plotting seasonal verification bar chart (Historical + Future GWLs)...")
            Visualizer.ensure_plot_dir_exists()
            
            hist_data = results['historical_verification']
            future_data = results.get('data', {})
            
            # 1. Determine Structure (Rows = Hist + GWLs)
            gwls_to_plot = sorted([g for g in future_data.keys() if g in config.GLOBAL_WARMING_LEVELS])
            if scenario.lower() == 'ssp245':
                gwls_to_plot = [g for g in gwls_to_plot if g == 2.0]
                
            n_rows = 1 + len(gwls_to_plot) # Historical + each GWL
            
            # 2. Filter Keys (Exclude Q30 as per previous request)
            low_keys = [k for k in sorted(hist_data.keys()) 
                        if hist_data[k]['type'] == 'low' 
                        and 'LNWL' not in k 
                        and 'Q30' not in k]
                        
            high_keys = [k for k in sorted(hist_data.keys()) 
                         if hist_data[k]['type'] == 'high'
                         and 'Q30' not in k]
            
            # Setup Figure: Rows = Scenarios (Hist, +2C, +3C), Cols = Event Types (Low, High)
            fig, axs = plt.subplots(n_rows, 2, figsize=(14, 5 * n_rows), squeeze=False)
            
            # Helper to get median
            def get_median(lst):
                clean = [x for x in lst if np.isfinite(x)]
                if not clean: return np.nan
                return np.median(clean)
            
            # --- LOOP THROUGH ROWS (Scenarios) ---
            scenario_labels = ['Historical (1960-2014)'] + [f'Future GWL +{g}°C' for g in gwls_to_plot]
            
            for r in range(n_rows):
                is_hist = (r == 0)
                current_gwl = gwls_to_plot[r-1] if not is_hist else None
                row_label = scenario_labels[r]
                
                # --- LOOP THROUGH COLS (Low/High) ---
                for c, (keys, type_title) in enumerate([(low_keys, 'Low-Flow Events'), (high_keys, 'High-Flow Events')]):
                    ax = axs[r, c]
                    if not keys: 
                        ax.set_visible(False)
                        continue
                    
                    x = np.arange(len(keys))
                    width = 0.25
                    
                    # --- PREPARE DATA ---
                    annual_targets = []
                    winter_periods = [] # List of LISTS (for points) or median
                    summer_periods = [] 
                    
                    winter_medians = []
                    summer_medians = []
                    
                    winter_counts = [] # tuples (X, Y)
                    summer_counts = []
                    
                    w_color = Visualizer.GWL_COLORS[2.0] # Reuse blue
                    s_color = '#ff7f0e'

                    for k in keys:
                        # Target is always the definition (e.g. 10yr) which comes from historical analysis
                        target_T = hist_data[k]['target_T']
                        annual_targets.append(target_T)
                        
                        if is_hist:
                            # HISTORICAL DATA
                            w_vals = hist_data[k]['winter_periods']
                            s_vals = hist_data[k]['summer_periods']
                            
                            winter_periods.append(w_vals)
                            summer_periods.append(s_vals)
                            winter_medians.append(get_median(w_vals))
                            summer_medians.append(get_median(s_vals))
                            
                            winter_counts.append((len([x for x in w_vals if np.isfinite(x)]), len(w_vals)))
                            summer_counts.append((len([x for x in s_vals if np.isfinite(x)]), len(s_vals)))
                            
                        else:
                            # FUTURE DATA (MMM)
                            try:
                                # Note: storyline.py outputs 'winter' and 'summer' keys in 'data' dictionary
                                # See line 2429 in storyline.py: results['data'][gwl] = {'winter': {}, 'summer': {}, ...}
                                w_data_node = future_data[current_gwl]['winter']['MMM'][k]
                                w_vals = w_data_node['future_return_periods_all_models']
                                winter_periods.append(w_vals)
                                winter_medians.append(get_median(w_vals))
                                winter_counts.append((w_data_node['model_count_X'], w_data_node['model_count_Y']))
                            except (KeyError, TypeError) as e:
                                # logging.warning(f"Missing Future Data [{current_gwl}][Winter][{k}]: {e}")
                                winter_periods.append([])
                                winter_medians.append(np.nan)
                                winter_counts.append((0,0))

                            try:
                                s_data_node = future_data[current_gwl]['summer']['MMM'][k]
                                s_vals = s_data_node['future_return_periods_all_models']
                                summer_periods.append(s_vals)
                                summer_medians.append(get_median(s_vals))
                                summer_counts.append((s_data_node['model_count_X'], s_data_node['model_count_Y']))
                            except (KeyError, TypeError) as e:
                                # logging.warning(f"Missing Future Data [{current_gwl}][Summer][{k}]: {e}")
                                summer_periods.append([])
                                summer_medians.append(np.nan)
                                summer_counts.append((0,0))
                                
                            w_color = Visualizer.GWL_COLORS[2.0]
                            s_color = '#ff7f0e'

                    # --- PLOT BARS ---
                    # Reference Annual
                    rects1 = ax.bar(x - width, annual_targets, width, label='Annual Target T', color='black', alpha=0.7)
                    
                    # Winter Median
                    label_w = 'Winter Median T' if is_hist else f'Future Winter Median T (+{current_gwl}°C)' 
                    rects2 = ax.bar(x, winter_medians, width, label=label_w, color=w_color, alpha=0.9 if is_hist else 0.7)
                    
                    # Summer Median
                    label_s = 'Summer Median T' if is_hist else f'Future Summer Median T (+{current_gwl}°C)'
                    rects3 = ax.bar(x + width, summer_medians, width, label=label_s, color=s_color, alpha=0.9 if is_hist else 0.7)
                    
                    # --- FORMATTING ---
                    ax.set_ylabel('Return Period (Years)')
                    ax.set_title(f'{row_label} - {type_title}', weight='bold')
                    ax.set_xticks(x)
                    ax.set_xticklabels(keys)
                    ax.grid(axis='y', linestyle='--', alpha=0.5)
                    
                    # Legend (only nice to have on first row or managed globally?)
                    # Individual subplot legends are useful here as labels change slightly
                    ax.legend(loc='upper left', fontsize='small')

                    # --- ANNOTATIONS ---
                    def autolabel(rects, counts):
                        for i, rect in enumerate(rects):
                            height = rect.get_height()
                            
                            X, Y = 0, 0
                            if i < len(counts):
                                X, Y = counts[i]
                            
                            label_text = f'{height:.1f}'
                            if Y > 0:
                                label_text += f"\n(n={X}/{Y})"
                            
                            if np.isfinite(height):
                                ax.annotate(label_text,
                                            xy=(rect.get_x() + rect.get_width() / 2, height),
                                            xytext=(0, 3), 
                                            textcoords="offset points",
                                            ha='center', va='bottom', rotation=90, fontsize=7)
                            else:
                                ax.annotate('Inf',
                                            xy=(rect.get_x() + rect.get_width() / 2, 0),
                                            xytext=(0, 3), 
                                            textcoords="offset points",
                                            ha='center', va='bottom', fontsize=7, color='red', rotation=90)

                    autolabel(rects2, winter_counts)
                    autolabel(rects3, summer_counts)
                    
                    # --- SCATTER POINTS ---
                    for j, sub_vals in enumerate(winter_periods):
                        clean = [v for v in sub_vals if np.isfinite(v)]
                        if clean:
                            jitter = np.random.uniform(-0.05, 0.05, size=len(clean))
                            ax.scatter(np.full_like(clean, x[j]) + jitter, clean, 
                                     color='navy', s=8, alpha=0.4, zorder=5, marker='o')

                    for j, sub_vals in enumerate(summer_periods):
                        clean = [v for v in sub_vals if np.isfinite(v)]
                        if clean:
                            jitter = np.random.uniform(-0.05, 0.05, size=len(clean))
                            ax.scatter(np.full_like(clean, x[j] + width) + jitter, clean, 
                                     color='brown', s=8, alpha=0.4, zorder=5, marker='o')

            plt.tight_layout()
            filename = os.path.join(config.PLOT_DIR, f'historical_seasonal_verification_{scenario}.png')
            plt.savefig(filename, dpi=150)
            plt.close(fig)
            logging.info(f"Saved seasonal verification plot to {filename}")

        except Exception as e:
            logging.error(f"Failed to plot historical seasonal verification: {e}")
            logging.exception("Traceback:")

    @staticmethod
    def _plot_composite_3x3_panel(composite_results, gwl, event_key, scenario, season,
                                      var_label, diff_unit, cmap_diff='PuOr',
                                      contour_fmt='%.0f', model_rps=None, model_lists=None, 
                                      n_total_models=None, map_extent=None):
        """
        Generic 3×3 composite panel for any variable.
        
        Rows: Future | Historical | Diff (Future − Historical)
        Columns: Extreme | Non-Extreme | Diff (Extreme − Non-Extreme)
        
        - Row 0, Row 1: contour maps of absolute values (shared levels across all 4 absolute panels)
        - Column 2 (all rows): filled difference maps with significance stippling
        - Row 2, Cols 0-1: filled difference maps with significance stippling
        - Row 2, Col 2: left empty (or could be double-difference)
        """
        logging.info(f"Plotting {var_label} composite 3×3 for {season}, GWL {gwl}°C...")
        Visualizer.ensure_plot_dir_exists()
        
        if not composite_results:
            logging.warning(f"No {var_label} composite results to plot.")
            return

        extent = [-105, 40, 0, 90]
        
        fig = plt.figure(figsize=(18, 15))
        gs = gridspec.GridSpec(3, 3, wspace=0.12, hspace=0.2)
        
        # Unpack results
        fut_ext = composite_results.get('future_extreme_mean')
        fut_non = composite_results.get('future_non_extreme_mean')
        hist_ext = composite_results.get('hist_extreme_mean')
        hist_non = composite_results.get('hist_non_extreme_mean')
        
        diff_ext_non_fut = composite_results.get('diff_ext_non_future')
        diff_ext_non_hist = composite_results.get('diff_ext_non_hist')
        diff_fut_hist_ext = composite_results.get('diff_fut_hist_ext')
        diff_fut_hist_non = composite_results.get('diff_fut_hist_non')
        
        sig_ext_non_fut = composite_results.get('sig_mask_ext_non_future')
        sig_ext_non_hist = composite_results.get('sig_mask_ext_non_hist')
        sig_fut_hist_ext = composite_results.get('sig_mask_fut_hist_ext')
        sig_fut_hist_non = composite_results.get('sig_mask_fut_hist_non')
        
        hist_climatology = composite_results.get('hist_climatology_mean')

        # --- Shared contour levels for absolute value panels (Row 0, 1, Cols 0, 1) ---
        abs_maps = [m for m in [fut_ext, fut_non, hist_ext, hist_non] if m is not None]
        if abs_maps:
            all_abs_vals = np.concatenate([m.values.ravel() for m in abs_maps])
            all_abs_vals = all_abs_vals[np.isfinite(all_abs_vals)]
            contour_levels = np.linspace(np.percentile(all_abs_vals, 2), np.percentile(all_abs_vals, 98), 15)
        else:
            contour_levels = None
        
        # --- Shared color limits for difference panels (Dynamic 98th Percentile) ---
        diff_maps = [m for m in [diff_ext_non_fut, diff_ext_non_hist, diff_fut_hist_ext, diff_fut_hist_non] if m is not None]
        
        # Default values
        diff_limit = 1.0
        mask_threshold = 0.0
        diff_levels = None
        diff_norm = None

        if diff_maps:
            # Concatenate all data to find the global range for this plot
            all_diff_vals = np.concatenate([m.values.ravel() for m in diff_maps])
            all_diff_vals = all_diff_vals[np.isfinite(all_diff_vals)]
            
            if len(all_diff_vals) > 0:
                # Calculate 98th percentile of absolute differences
                limit = np.percentile(np.abs(all_diff_vals), 98)
                
                # Round up to nearest integer for clean edge values
                if limit > 0:
                    import math
                    diff_limit = math.ceil(limit)
                else:
                    diff_limit = 1.0
                
                # Define discrete levels (12 color intervals)
                diff_levels = np.round(np.linspace(-diff_limit, diff_limit, 13), 1)
                
                # Mask threshold (5% of range, similar to previous logic)
                mask_threshold = diff_limit * 0.05
        
        logging.info(f"  Using dynamic difference color limit (98th percentile): +/- {diff_limit} {diff_unit}")
        logging.info(f"  Masking threshold: < +/- {mask_threshold}")

        def _add_map_features(ax):
            if map_extent:
                ax.set_extent(map_extent, crs=ccrs.PlateCarree())
            else:
                ax.set_extent(extent, crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linewidth=0.5, alpha=0.5)
            
            # Add analysis box
            lon_min, lon_max = Config.BOX_LON_MIN, Config.BOX_LON_MAX
            lat_min, lat_max = Config.BOX_LAT_MIN, Config.BOX_LAT_MAX
            analysis_box = mpatches.Rectangle((lon_min, lat_min), lon_max - lon_min, lat_max - lat_min,
                                              fill=False, edgecolor='magenta', linewidth=2.0, transform=ccrs.PlateCarree(), zorder=10)
            ax.add_patch(analysis_box)
        
        def _plot_contour(ax, data_map, title, contours_to_use=None):
             _add_map_features(ax)
             levels = contours_to_use if contours_to_use is not None else contour_levels
             if data_map is not None and levels is not None:
                 cs = ax.contour(data_map.lon, data_map.lat, data_map, levels=levels,
                                 colors='black', linewidths=0.8, transform=ccrs.PlateCarree())
                 ax.clabel(cs, inline=True, fontsize=6, fmt=contour_fmt)
             ax.set_title(title, fontsize=10)
        
        def _plot_diff(ax, diff_map, sig_mask, title, contour_map=None):
            _add_map_features(ax)
            cf = None
            if diff_map is not None:
                import matplotlib.colors as mcolors
                
                custom_cmap = None
                
                if var_label in ['Z500', 'PSL']:
                    colors = ['#542788', '#ffffff', '#b2182b'] 
                    custom_cmap = mcolors.LinearSegmentedColormap.from_list('PuWhRd', colors)
                    
                elif var_label == 'PR':
                    colors = ['#1a9850', '#ffffff', '#b2182b'] 
                    custom_cmap = mcolors.LinearSegmentedColormap.from_list('GnWhRd', colors)
                
                else:
                    try:
                        custom_cmap = plt.get_cmap(cmap_diff)
                    except:
                        custom_cmap = matplotlib.cm.get_cmap(cmap_diff)

                # Use discrete norm if levels are defined
                norm = None
                if diff_levels is not None:
                    norm = mcolors.BoundaryNorm(diff_levels, ncolors=256) # Use 256 to allow gradient within norm or just map?
                    # For discrete colors matching levels exactly, we usually want ncolors=len(levels)-1
                    # But if we use a continuous colormap with BoundaryNorm, it picks colors.
                    # Let's use clean discrete mapping.
                    norm = mcolors.BoundaryNorm(diff_levels, ncolors=custom_cmap.N, clip=False)

                # --- PLOT DIFFERENCE MAP ---
                if norm is not None:
                    cf = ax.pcolormesh(diff_map.lon, diff_map.lat, diff_map, cmap=custom_cmap,
                                       norm=norm, transform=ccrs.PlateCarree())
                else:
                    cf = ax.pcolormesh(diff_map.lon, diff_map.lat, diff_map, cmap=custom_cmap,
                                       vmin=-diff_limit, vmax=diff_limit, transform=ccrs.PlateCarree())
                
                # --- ADDED: Reference Climatology Contours (skip for PR) ---
                if contour_map is not None and contour_levels is not None and var_label != 'PR':
                    ref_levels = contour_levels[::2]
                    cs = ax.contour(contour_map.lon, contour_map.lat, contour_map, levels=ref_levels,
                                    colors='gray', linewidths=1.2, alpha=0.9, transform=ccrs.PlateCarree())
                    ax.clabel(cs, inline=True, fontsize=6, fmt=contour_fmt, colors='gray')

                if sig_mask is not None:
                    skip = 4
                    lons_mesh, lats_mesh = np.meshgrid(diff_map.lon, diff_map.lat)
                    mask_sub = sig_mask[::skip, ::skip]
                    lons_sub = lons_mesh[::skip, ::skip]
                    lats_sub = lats_mesh[::skip, ::skip]
                    ax.scatter(lons_sub[mask_sub], lats_sub[mask_sub], s=1, color='black', 
                              alpha=0.5, transform=ccrs.PlateCarree())
            ax.set_title(title, fontsize=10)
            return cf
        
        # --- Row 0: Future ---
        ax00 = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
        _plot_contour(ax00, fut_ext, "Future – Extreme")
        
        ax01 = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
        _plot_contour(ax01, fut_non, "Future – Non-Extreme")
        
        ax02 = fig.add_subplot(gs[0, 2], projection=ccrs.PlateCarree())
        # Add Reference Contours to Difference Plot
        cf_r0 = _plot_diff(ax02, diff_ext_non_fut, sig_ext_non_fut, "Future: Ext − Non", contour_map=hist_climatology)
        
        # --- Row 1: Historical ---
        ax10 = fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree())
        _plot_contour(ax10, hist_ext, "Historical – Extreme")
        
        ax11 = fig.add_subplot(gs[1, 1], projection=ccrs.PlateCarree())
        _plot_contour(ax11, hist_non, "Historical – Non-Extreme")
        
        ax12 = fig.add_subplot(gs[1, 2], projection=ccrs.PlateCarree())
        # Add Reference Contours to Difference Plot
        cf_r1 = _plot_diff(ax12, diff_ext_non_hist, sig_ext_non_hist, "Historical: Ext − Non", contour_map=hist_climatology)
        
        # --- Row 2: Difference (Future − Historical) ---
        ax20 = fig.add_subplot(gs[2, 0], projection=ccrs.PlateCarree())
        cf_r2a = _plot_diff(ax20, diff_fut_hist_ext, sig_fut_hist_ext, "Δ(Fut−Hist) – Extreme", contour_map=hist_climatology)
        
        ax21 = fig.add_subplot(gs[2, 1], projection=ccrs.PlateCarree())
        cf_r2b = _plot_diff(ax21, diff_fut_hist_non, sig_fut_hist_non, "Δ(Fut−Hist) – Non-Extreme", contour_map=hist_climatology)
        
        # Row 2, Col 2: Return Period Boxplot
        ax22 = fig.add_subplot(gs[2, 2])
        
        if model_rps:
            # Get actually used models from composites (those that didn't fail during loading)
            used_ext_keys = composite_results.get('used_extreme_models', [])
            used_non_keys = composite_results.get('used_non_extreme_models', [])
            
            records = []
            for m_key, rp in model_rps.items():
                if np.isinf(rp):
                    rp = 35  # Clamp for display
                category = 'Other'
                color = 'gray'
                marker = 'o'
                size = 5
                alpha = 0.4
                zorder = 1
                
                if m_key in used_ext_keys:
                    category = 'Extreme (used)'
                    color = '#b2182b'
                    marker = 'D'
                    size = 7
                    alpha = 1.0
                    zorder = 3
                elif m_key in used_non_keys:
                    category = 'Non-Extreme (used)'
                    color = '#2166ac'
                    marker = 'D'
                    size = 7
                    alpha = 1.0
                    zorder = 3
                    
                records.append({'Model': m_key, 'Return Period': rp, 'Category': category,
                               'Color': color, 'Marker': marker, 'Size': size, 'Alpha': alpha,
                               'Z': zorder, 'DummyY': 0})
            
            df_rp = pd.DataFrame(records)
            
            import seaborn as sns
            sns.boxplot(data=df_rp, x='Return Period', y='DummyY', ax=ax22,
                        color='lightgray', width=0.3, showfliers=False, orient='h')
            
            # Plot other models
            other = df_rp[df_rp['Category'] == 'Other']
            if not other.empty:
                sns.stripplot(data=other, x='Return Period', y='DummyY', ax=ax22,
                              color='gray', alpha=0.4, size=5, jitter=True, orient='h')
            
            # Plot key models (actually used)
            key = df_rp[df_rp['Category'] != 'Other']
            if not key.empty:
                for _, row in key.iterrows():
                    y_pos = np.random.uniform(-0.05, 0.05)
                    ax22.plot(row['Return Period'], y_pos, marker='D', color=row['Color'],
                             markersize=7, alpha=1.0, linestyle='None', zorder=3)
            
            ax22.set_xlim(0, 35)
            ax22.set_xticks([0, 5, 10, 15, 20, 25, 30, 35])
            ax22.set_xticklabels(['0', '5', '10', '15', '20', '25', '30', '∞'])
            ax22.set_yticks([])
            ax22.set_ylabel('')
            ax22.invert_yaxis()
            ax22.grid(axis='x', linestyle=':', alpha=0.7)
            ax22.set_xlabel('Return Period (Years)', fontsize=10)
            ax22.set_title('Model Selection (Return Period)', fontsize=10)
            
            # Annotation & Legend
            n_total = n_total_models if n_total_models else len(model_rps)
            ax22.text(0.95, 0.9, f"n={len(model_rps)}/{n_total}", transform=ax22.transAxes,
                     ha='right', fontsize=9, fontweight='bold')
            
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='D', color='w', markerfacecolor='#b2182b',
                       label=f'High-Frequency (N={len(used_ext_keys)})', markersize=7),
                Line2D([0], [0], marker='D', color='w', markerfacecolor='#2166ac',
                       label=f'Low-Frequency (N={len(used_non_keys)})', markersize=7),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                       label='Other Models', markersize=5, alpha=0.5),
            ]
            ax22.legend(handles=legend_elements, loc='lower right', fontsize=7, frameon=True, framealpha=0.8)
        else:
            ax22.axis('off')
        
        # Row labels
        fig.text(0.02, 0.78, "Future", fontsize=14, fontweight='bold', rotation=90, va='center')
        fig.text(0.02, 0.50, "Historical", fontsize=14, fontweight='bold', rotation=90, va='center')
        fig.text(0.02, 0.22, "Δ (Fut−Hist)", fontsize=14, fontweight='bold', rotation=90, va='center')
        
        # Colorbar for difference maps
        ref_cf = cf_r0 or cf_r1 or cf_r2a or cf_r2b
        if ref_cf:
            cax = fig.add_axes((0.25, 0.04, 0.50, 0.015))
            fig.colorbar(ref_cf, cax=cax, orientation='horizontal', label=f'Difference ({diff_unit})', extend='both')
        
        plt.subplots_adjust(bottom=0.10, left=0.06)
        plt.suptitle(f"{var_label} Composite ({season} half-year): High vs Low-Frequency ({event_key})\n"
                     f"GWL {gwl}°C | {scenario.upper()}",
                     fontsize=14, weight='bold', y=0.97)
        
        filename = f"composite_analysis_{var_label.lower()}_{season.lower()}_{event_key}_{scenario}_gwl{gwl}.png"
        filepath = os.path.join(Config.PLOT_DIR, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logging.info(f"Saved {var_label} composite plot to {filepath}")

    @staticmethod
    def plot_z500_composite_analysis_panel(composite_results, gwl, event_key, scenario, season,
                                           model_rps=None, model_lists=None, n_total_models=None):
        """Plots a 3×3 composite panel for Z500 (one season per plot)."""
        Visualizer._plot_composite_3x3_panel(
            composite_results, gwl, event_key, scenario, season,
            var_label='Z500', diff_unit='m', cmap_diff='PuOr', contour_fmt='%.0f',
            model_rps=model_rps, model_lists=model_lists, n_total_models=n_total_models
        )

    @staticmethod
    def plot_psl_composite_analysis_panel(composite_results, gwl, event_key, scenario, season,
                                           model_rps=None, model_lists=None, n_total_models=None):
        """Plots a 3×3 composite panel for PSL (one season per plot)."""
        Visualizer._plot_composite_3x3_panel(
            composite_results, gwl, event_key, scenario, season,
            var_label='PSL', diff_unit='hPa', cmap_diff='PuOr', contour_fmt='%.0f',
            model_rps=model_rps, model_lists=model_lists, n_total_models=n_total_models
        )

    @staticmethod
    def plot_pr_composite_analysis_panel(composite_results, gwl, event_key, scenario, season,
                                          model_rps=None, model_lists=None, n_total_models=None):
        """Plots a 3×3 composite panel for PR (one season per plot)."""
        # Central Europe Box from Config
        # [LonMin, LonMax, LatMin, LatMax]
        # Adding a small buffer around the box for better visualization
        buffer = 5.0
        ce_extent = [
            Config.BOX_LON_MIN - buffer, Config.BOX_LON_MAX + buffer,
            Config.BOX_LAT_MIN - buffer, Config.BOX_LAT_MAX + buffer
        ]
        
        Visualizer._plot_composite_3x3_panel(
            composite_results, gwl, event_key, scenario, season,
            var_label='PR', diff_unit='mm/day', cmap_diff='BrBG', contour_fmt='%.1f',
            model_rps=model_rps, model_lists=model_lists, n_total_models=n_total_models,
            map_extent=ce_extent # <--- Zoomed in
        )

    @staticmethod
    def plot_ua_composite_analysis_panel(composite_results, gwl, event_key, scenario, season,
                                          model_rps=None, model_lists=None, n_total_models=None):
        """Plots a 3×3 composite panel for UA (Zonal Wind)."""
        # Apply Greenland Mask for UA850
        if composite_results:
            mask_lon_min, mask_lon_max = Config.GREENLAND_LON_MIN, Config.GREENLAND_LON_MAX
            mask_lat_min, mask_lat_max = Config.GREENLAND_LAT_MIN, Config.GREENLAND_LAT_MAX
            
            for key, data_array in composite_results.items():
                if isinstance(data_array, xr.DataArray):
                    # Create a mask for Greenland
                    # Note: Longitudes might be 0-360 or -180 to 180. Check coordinate system.
                    # Assuming standard -180 to 180 or 0 to 360, we handle both if needed, but config is -75 to -10.
                    # If data is 0-360, -75 is 285.
                    
                    # Check longitude convention of data
                    if data_array.lon.min() >= 0:
                        # 0 to 360 convention
                        m_lon_min = mask_lon_min + 360
                        m_lon_max = mask_lon_max + 360
                    else:
                        m_lon_min = mask_lon_min
                        m_lon_max = mask_lon_max
                        
                    # Apply mask: set values inside the box to NaN
                    mask = (data_array.lon >= m_lon_min) & (data_array.lon <= m_lon_max) & \
                           (data_array.lat >= mask_lat_min) & (data_array.lat <= mask_lat_max)
                    
                    composite_results[key] = data_array.where(~mask)

        Visualizer._plot_composite_3x3_panel(
            composite_results, gwl, event_key, scenario, season,
            var_label='UA850', diff_unit='m/s', cmap_diff='RdBu_r', contour_fmt='%.1f',
            model_rps=model_rps, model_lists=model_lists, n_total_models=n_total_models
        )

    @staticmethod
    def plot_tas_composite_analysis_panel(composite_results, gwl, event_key, scenario, season,
                                          model_rps=None, model_lists=None, n_total_models=None):
        """Plots a 3×3 composite panel for TAS (Temperature)."""
        Visualizer._plot_composite_3x3_panel(
            composite_results, gwl, event_key, scenario, season,
            var_label='TAS', diff_unit='°C', cmap_diff='RdBu_r', contour_fmt='%.1f',
            model_rps=model_rps, model_lists=model_lists, n_total_models=n_total_models
        )

    # =========================================================================
    # Combined Composite Difference Panel (Winter + Summer side-by-side)
    # =========================================================================
    @staticmethod
    def plot_combined_composite_diff_panel(
        winter_composite, summer_composite,
        gwl, event_key, scenario,
        var_label, diff_unit, cmap_diff='RdBu_r', contour_fmt='%.1f',
        winter_model_rps=None, summer_model_rps=None,
        winter_n_total=None, summer_n_total=None,
        map_extent=None, fixed_diff_limit=None
    ):
        """
        Combined difference-column plot for two seasons (Winter + Summer).

        Replicates exactly the 3rd column of the original 3×3 composite plot:
          Row 0: Future: Ext − Non  (difference map)
          Row 1: Historical: Ext − Non  (difference map)
          Row 2: Return Period boxplot
        Laid out as 2 columns: Winter (left) | Summer (right).
        """
        logging.info(f"Plotting combined composite diff panel for {var_label}, GWL {gwl}°C, {scenario}...")
        Visualizer.ensure_plot_dir_exists()

        if not winter_composite and not summer_composite:
            logging.warning(f"No composite data for combined diff panel ({var_label}).")
            return

        default_extent = [-105, 40, 0, 90]
        extent = map_extent if map_extent else default_extent

        # --- Collect difference maps across both seasons for shared color limits ---
        all_diff_maps = []
        for comp in [winter_composite, summer_composite]:
            if comp is None:
                continue
            for key in ['diff_ext_non_future', 'diff_ext_non_hist']:
                m = comp.get(key)
                if m is not None:
                    all_diff_maps.append(m)

        # Calculate shared difference limits (98th percentile)
        diff_limit = fixed_diff_limit if fixed_diff_limit is not None else 1.0
        diff_levels = None
        
        if fixed_diff_limit is not None:
            # If a fixed limit is provided, use it to generate the levels
            diff_levels = np.round(np.linspace(-diff_limit, diff_limit, 13), 1)
        elif all_diff_maps:
            # Otherwise, calculate dynamically from the provided maps
            all_vals = np.concatenate([m.values.ravel() for m in all_diff_maps])
            all_vals = all_vals[np.isfinite(all_vals)]
            if len(all_vals) > 0:
                import math
                limit = np.percentile(np.abs(all_vals), 98)
                if limit > 0:
                    diff_limit = math.ceil(limit)
                diff_levels = np.round(np.linspace(-diff_limit, diff_limit, 13), 1)

        # Shared contour levels for reference climatology
        all_abs_maps = []
        for comp in [winter_composite, summer_composite]:
            if comp is None:
                continue
            for key in ['fut_extreme_mean', 'future_non_extreme_mean',
                        'hist_extreme_mean', 'hist_non_extreme_mean']:
                m = comp.get(key)
                if m is not None:
                    all_abs_maps.append(m)

        contour_levels = None
        if all_abs_maps:
            abs_vals = np.concatenate([m.values.ravel() for m in all_abs_maps])
            abs_vals = abs_vals[np.isfinite(abs_vals)]
            if len(abs_vals) > 0:
                contour_levels = np.linspace(np.percentile(abs_vals, 2),
                                             np.percentile(abs_vals, 98), 15)

        logging.info(f"  Combined diff limit: +/- {diff_limit} {diff_unit}")

        # --- Setup figure: 3 rows × 2 columns ---
        # Rows 0-1: map projections, Row 2: regular axes for boxplots
        fig = plt.figure(figsize=(12, 13))
        gs = gridspec.GridSpec(3, 2, wspace=0.08, hspace=0.25,
                               height_ratios=[1, 1, 0.5])

        # --- Colormap selection (same logic as _plot_composite_3x3_panel) ---
        import matplotlib.colors as mcolors
        custom_cmap = None
        if var_label in ['Z500', 'PSL']:
            colors = ['#542788', '#ffffff', '#b2182b']
            custom_cmap = mcolors.LinearSegmentedColormap.from_list('PuWhRd', colors)
        elif var_label == 'PR':
            colors = ['#1a9850', '#ffffff', '#b2182b']
            custom_cmap = mcolors.LinearSegmentedColormap.from_list('GnWhRd', colors)
        else:
            try:
                custom_cmap = plt.get_cmap(cmap_diff)
            except:
                custom_cmap = matplotlib.cm.get_cmap(cmap_diff)

        norm = None
        if diff_levels is not None:
            norm = mcolors.BoundaryNorm(diff_levels, ncolors=custom_cmap.N, clip=False)

        def _add_map_features(ax):
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linewidth=0.5, alpha=0.5)
            
            # Add analysis box
            lon_min, lon_max = Config.BOX_LON_MIN, Config.BOX_LON_MAX
            lat_min, lat_max = Config.BOX_LAT_MIN, Config.BOX_LAT_MAX
            analysis_box = mpatches.Rectangle((lon_min, lat_min), lon_max - lon_min, lat_max - lat_min,
                                              fill=False, edgecolor='magenta', linewidth=2.0, transform=ccrs.PlateCarree(), zorder=10)
            ax.add_patch(analysis_box)

        def _plot_diff(ax, diff_map, sig_mask, title, contour_map=None):
            _add_map_features(ax)
            cf = None
            if diff_map is not None:
                if norm is not None:
                    cf = ax.pcolormesh(diff_map.lon, diff_map.lat, diff_map,
                                       cmap=custom_cmap, norm=norm,
                                       transform=ccrs.PlateCarree())
                else:
                    cf = ax.pcolormesh(diff_map.lon, diff_map.lat, diff_map,
                                       cmap=custom_cmap,
                                       vmin=-diff_limit, vmax=diff_limit,
                                       transform=ccrs.PlateCarree())

                # Reference climatology contours (skip for PR)
                if contour_map is not None and contour_levels is not None and var_label != 'PR':
                    ref_levels = contour_levels[::2]
                    cs = ax.contour(contour_map.lon, contour_map.lat, contour_map,
                                    levels=ref_levels, colors='gray', linewidths=1.2,
                                    alpha=0.9, transform=ccrs.PlateCarree())
                    ax.clabel(cs, inline=True, fontsize=6, fmt=contour_fmt, colors='gray')

                # Significance stippling
                if sig_mask is not None:
                    skip = 4
                    lons_mesh, lats_mesh = np.meshgrid(diff_map.lon, diff_map.lat)
                    mask_sub = sig_mask[::skip, ::skip]
                    lons_sub = lons_mesh[::skip, ::skip]
                    lats_sub = lats_mesh[::skip, ::skip]
                    ax.scatter(lons_sub[mask_sub], lats_sub[mask_sub], s=1,
                               color='black', alpha=0.5, transform=ccrs.PlateCarree())
            ax.set_title(title, fontsize=10)
            return cf

        def _plot_rp_boxplot(ax, model_rps, composite_results, season_label, n_total_models_val):
            """Draw return period boxplot in a regular (non-map) axes."""
            if not model_rps:
                ax.text(0.5, 0.5, "No RP Data", ha='center', va='center',
                        transform=ax.transAxes, fontsize=10)
                ax.set_title(f'{season_label}: Model Selection', fontsize=10)
                return

            used_ext_keys = composite_results.get('used_extreme_models', [])
            used_non_keys = composite_results.get('used_non_extreme_models', [])

            records = []
            for m_key, rp in model_rps.items():
                if np.isinf(rp):
                    rp = 35
                category = 'Other'
                color = 'gray'
                if m_key in used_ext_keys:
                    category = 'Extreme (used)'
                    color = '#b2182b'
                elif m_key in used_non_keys:
                    category = 'Non-Extreme (used)'
                    color = '#2166ac'
                records.append({'Model': m_key, 'Return Period': rp,
                                'Category': category, 'Color': color, 'DummyY': 0})

            df_rp = pd.DataFrame(records)

            import seaborn as sns
            sns.boxplot(data=df_rp, x='Return Period', y='DummyY', ax=ax,
                        color='lightgray', width=0.3, showfliers=False, orient='h')

            other = df_rp[df_rp['Category'] == 'Other']
            if not other.empty:
                sns.stripplot(data=other, x='Return Period', y='DummyY', ax=ax,
                              color='gray', alpha=0.4, size=5, jitter=True, orient='h')

            key_df = df_rp[df_rp['Category'] != 'Other']
            if not key_df.empty:
                for _, row in key_df.iterrows():
                    y_pos = np.random.uniform(-0.05, 0.05)
                    ax.plot(row['Return Period'], y_pos, marker='D',
                            color=row['Color'], markersize=7, alpha=1.0,
                            linestyle='None', zorder=3)

            ax.set_xlim(0, 35)
            ax.set_xticks([0, 5, 10, 15, 20, 25, 30, 35])
            ax.set_xticklabels(['0', '5', '10', '15', '20', '25', '30', '∞'])
            ax.set_yticks([])
            ax.set_ylabel('')
            ax.invert_yaxis()
            ax.grid(axis='x', linestyle=':', alpha=0.7)
            ax.set_xlabel('Return Period (Years)', fontsize=9)
            ax.set_title(f'{season_label}: Model Selection (Return Period)', fontsize=10)

            n_total = n_total_models_val if n_total_models_val else len(model_rps)
            ax.text(0.95, 0.85, f"n={len(model_rps)}/{n_total}",
                    transform=ax.transAxes, ha='right', fontsize=8, fontweight='bold')

            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='D', color='w', markerfacecolor='#b2182b',
                       label=f'High-Frequency (N={len(used_ext_keys)})', markersize=6),
                Line2D([0], [0], marker='D', color='w', markerfacecolor='#2166ac',
                       label=f'Low-Frequency (N={len(used_non_keys)})', markersize=6),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                       label='Other Models', markersize=5, alpha=0.5),
            ]
            ax.legend(handles=legend_elements, loc='lower right', fontsize=6,
                      frameon=True, framealpha=0.8)

        # --- Row content: only the 3rd column of the original 3×3 ---
        row_configs = [
            {'key_diff': 'diff_ext_non_future', 'key_sig': 'sig_mask_ext_non_future',
             'title': 'Future: High − Low'},
            {'key_diff': 'diff_ext_non_hist', 'key_sig': 'sig_mask_ext_non_hist',
             'title': 'Historical: Ext − Non'},
        ]

        ref_cf = None
        season_data = [
            ('Winter', winter_composite, winter_model_rps, winter_n_total),
            ('Summer', summer_composite, summer_model_rps, summer_n_total),
        ]

        for col_idx, (season_label, comp, model_rps_dict, n_total_val) in enumerate(season_data):
            if comp is None:
                for row_idx in range(2):
                    ax = fig.add_subplot(gs[row_idx, col_idx], projection=ccrs.PlateCarree())
                    _add_map_features(ax)
                    ax.set_title(f"{season_label}: No Data", fontsize=10)
                ax_rp = fig.add_subplot(gs[2, col_idx])
                ax_rp.text(0.5, 0.5, "No Data", ha='center', va='center',
                           transform=ax_rp.transAxes)
                ax_rp.set_title(f'{season_label}: Model Selection', fontsize=10)
                continue

            hist_clim = comp.get('hist_climatology_mean')

            # Map rows (0-1): difference maps
            for row_idx, rc in enumerate(row_configs):
                ax = fig.add_subplot(gs[row_idx, col_idx], projection=ccrs.PlateCarree())
                diff_map = comp.get(rc['key_diff'])
                sig_mask = comp.get(rc['key_sig'])
                title = f"{season_label}: {rc['title']}"
                cf = _plot_diff(ax, diff_map, sig_mask, title, contour_map=hist_clim)
                if cf is not None:
                    ref_cf = cf

            # Row 2: Return Period boxplot
            ax_rp = fig.add_subplot(gs[2, col_idx])
            _plot_rp_boxplot(ax_rp, model_rps_dict, comp, season_label, n_total_val)

        # --- Shared colorbar ---
        if ref_cf:
            cax = fig.add_axes((0.15, 0.06, 0.70, 0.015))
            fig.colorbar(ref_cf, cax=cax, orientation='horizontal',
                         label=f'Difference ({diff_unit})', extend='both')

        scenario_title = Visualizer._format_scenario_title(scenario)
        plt.suptitle(
            f"{var_label} Composite Differences: Winter vs Summer ({event_key})\n"
            f"GWL {gwl}°C | {scenario_title}",
            fontsize=14, weight='bold', y=0.98
        )
        plt.subplots_adjust(bottom=0.10, left=0.05, right=0.95, top=0.94)

        filename = f"combined_diff_{var_label.lower()}_{event_key}_{scenario}_gwl{gwl}.png"
        filepath = os.path.join(Config.PLOT_DIR, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logging.info(f"Saved combined diff panel to {filepath}")

    @staticmethod
    def plot_ua_combined_composite_diff_panel(
        winter_composite, summer_composite,
        gwl, event_key, scenario,
        winter_model_rps=None, summer_model_rps=None,
        winter_n_total=None, summer_n_total=None,
        fixed_diff_limit=None
    ):
        """Combined diff panel for UA850, with Greenland masking applied."""
        # Apply Greenland mask to both composites
        for comp in [winter_composite, summer_composite]:
            if comp is None:
                continue
            mask_lon_min = Config.GREENLAND_LON_MIN
            mask_lon_max = Config.GREENLAND_LON_MAX
            mask_lat_min = Config.GREENLAND_LAT_MIN
            mask_lat_max = Config.GREENLAND_LAT_MAX
            for key, data_array in comp.items():
                if isinstance(data_array, xr.DataArray):
                    if data_array.lon.min() >= 0:
                        m_lon_min = mask_lon_min + 360
                        m_lon_max = mask_lon_max + 360
                    else:
                        m_lon_min = mask_lon_min
                        m_lon_max = mask_lon_max
                    mask = ((data_array.lon >= m_lon_min) & (data_array.lon <= m_lon_max) &
                            (data_array.lat >= mask_lat_min) & (data_array.lat <= mask_lat_max))
                    comp[key] = data_array.where(~mask)

        Visualizer.plot_combined_composite_diff_panel(
            winter_composite, summer_composite,
            gwl, event_key, scenario,
            var_label='UA850', diff_unit='m/s', cmap_diff='RdBu_r', contour_fmt='%.1f',
            winter_model_rps=winter_model_rps, summer_model_rps=summer_model_rps,
            winter_n_total=winter_n_total, summer_n_total=summer_n_total,
            fixed_diff_limit=fixed_diff_limit
        )

    @staticmethod
    def plot_pr_combined_composite_diff_panel(
        winter_composite, summer_composite,
        gwl, event_key, scenario,
        winter_model_rps=None, summer_model_rps=None,
        winter_n_total=None, summer_n_total=None,
        fixed_diff_limit=None
    ):
        """Combined diff panel for PR, with Central Europe zoom."""
        buffer = 5.0
        ce_extent = [
            Config.BOX_LON_MIN - buffer, Config.BOX_LON_MAX + buffer,
            Config.BOX_LAT_MIN - buffer, Config.BOX_LAT_MAX + buffer
        ]
        Visualizer.plot_combined_composite_diff_panel(
            winter_composite, summer_composite,
            gwl, event_key, scenario,
            var_label='PR', diff_unit='mm/day', cmap_diff='BrBG', contour_fmt='%.1f',
            winter_model_rps=winter_model_rps, summer_model_rps=summer_model_rps,
            winter_n_total=winter_n_total, summer_n_total=summer_n_total,
            map_extent=ce_extent,
            fixed_diff_limit=fixed_diff_limit
        )

    @staticmethod
    def plot_discharge_events_timeseries(cmip6_results, discharge_data_loaded, config, scenario):
        """
        Plots the MMM and spread of annual minimum 30-day discharge.
        Creates a single plot showing MMM and the 10th-90th percentile spread along with LNWL crossing events.
        """
        logging.info(f"Plotting discharge events timeseries for {scenario}...")
        Visualizer.ensure_plot_dir_exists()
        
        # 1. Extract timeseries
        metric_timeseries = cmip6_results.get('model_metric_timeseries', {})
        if not metric_timeseries:
            logging.warning("No model metric timeseries available.")
            return

        all_models_data = []
        
        def get_scenario_data(target_scenario):
            scen_models_data = []
            for key, ts_dict in metric_timeseries.items():
                if not key.endswith(target_scenario):
                    continue
                if '30Q_low_full_year' in ts_dict:
                    da = ts_dict['30Q_low_full_year']
                    if da is not None:
                         try:
                             df = da.to_dataframe(name='discharge')
                         except ValueError:
                             df = da.to_dataframe()
                             if len(df.columns) == 1:
                                 df.columns = ['discharge']
                         if 'year' in df.index.names:
                             df = df.reset_index()
                         df['model_key'] = key
                         df['model'] = key.split('_')[0]
                         scen_models_data.append(df)
            
            if not scen_models_data:
                return None, 0, [], [], []
            
            df_all = pd.concat(scen_models_data, ignore_index=True)
            df_stats = df_all.groupby('year')['discharge'].agg(['mean', 'min', 'max']).reset_index()
            df_stats.columns = ['year', 'mmm_raw', 'p2_5', 'p97_5']

            df_stats['mmm'] = df_stats['mmm_raw'].rolling(window=5, center=True).mean()
            df_stats['p2_5'] = df_stats['p2_5'].rolling(window=5, center=True).mean()
            df_stats['p97_5'] = df_stats['p97_5'].rolling(window=5, center=True).mean()

            # Calculate individual member trends over 2015-2100 (m3/s per decade)
            ind_trends = []
            ind_pvals = []
            ind_models = []
            for df_m in scen_models_data:
                m_key = df_m['model_key'].iloc[0] if 'model_key' in df_m.columns else df_m['model'].iloc[0]
                sub = df_m[(df_m['year'] >= 2015) & (df_m['year'] <= 2100)].dropna(subset=['discharge'])
                if len(sub) > 1:
                    slope, _, _, p_val, _ = linregress(sub['year'], sub['discharge'])
                    ind_trends.append(slope * 10.0)
                    ind_pvals.append(p_val)
                    ind_models.append(m_key)

            return df_stats, len(scen_models_data), ind_trends, ind_pvals, ind_models

        df_stats_ssp585, n_ssp585, trends_ssp585, pvals_ssp585, models_ssp585 = get_scenario_data('ssp585')
        df_stats_ssp245, n_ssp245, trends_ssp245, pvals_ssp245, models_ssp245 = get_scenario_data('ssp245')
        
        if df_stats_ssp585 is None and df_stats_ssp245 is None:
            logging.warning("No 30Q_low_full_year data found for either scenario.")
            return

        # Calculate MMM trends for 2015-2100
        mmm_trend_245, p_mmm_245 = 0.0, 1.0
        if df_stats_ssp245 is not None and not df_stats_ssp245.empty:
            sub_245 = df_stats_ssp245[(df_stats_ssp245['year'] >= 2015) & (df_stats_ssp245['year'] <= 2100)]
            if len(sub_245) > 1:
                sl245, _, _, p_mmm_245, _ = linregress(sub_245['year'], sub_245['mmm_raw'])
                mmm_trend_245 = sl245 * 10.0

        mmm_trend_585, p_mmm_585 = 0.0, 1.0
        if df_stats_ssp585 is not None and not df_stats_ssp585.empty:
            sub_585 = df_stats_ssp585[(df_stats_ssp585['year'] >= 2015) & (df_stats_ssp585['year'] <= 2100)]
            if len(sub_585) > 1:
                sl585, _, _, p_mmm_585, _ = linregress(sub_585['year'], sub_585['mmm_raw'])
                mmm_trend_585 = sl585 * 10.0

        # Print summary statistics clearly to stdout and log
        summary_lines = [
            "\n" + "=" * 80,
            "CMIP6 INDIVIDUAL MODEL TREND SUMMARY STATISTICS (2015–2100)",
            "Metric: 30-day Minimum Discharge Trend (m³/s per decade)",
            "=" * 80
        ]
        for scen_label, n_mod, tr_list, pv_list, m_tr, p_m in [
            ("SSP2-4.5", n_ssp245, trends_ssp245, pvals_ssp245, mmm_trend_245, p_mmm_245),
            ("SSP5-8.5", n_ssp585, trends_ssp585, pvals_ssp585, mmm_trend_585, p_mmm_585)
        ]:
            if n_mod > 0 and len(tr_list) > 0:
                arr_tr = np.array(tr_list)
                arr_pv = np.array(pv_list)
                min_t, max_t = np.min(arr_tr), np.max(arr_tr)
                n_neg = np.sum(arr_tr < 0)
                pct_neg = (n_neg / len(arr_tr)) * 100.0
                n_sig_neg = np.sum((arr_tr < 0) & (arr_pv < 0.05))
                pct_sig_neg = (n_sig_neg / len(arr_tr)) * 100.0
                p_m_str = "p < 0.001" if p_m < 0.001 else f"p = {p_m:.4f}"

                summary_lines.append(f"\n--- {scen_label} (n = {len(arr_tr)}) ---")
                summary_lines.append(f"  * Range of individual model trends (min, max): {min_t:+.2f} to {max_t:+.2f} m³/s per decade")
                summary_lines.append(f"  * Percentage of models with negative trend (< 0 m³/s/dec): {pct_neg:.1f}% ({n_neg}/{len(arr_tr)})")
                summary_lines.append(f"  * Percentage of models with sig. negative trend (p < 0.05): {pct_sig_neg:.1f}% ({n_sig_neg}/{len(arr_tr)})")
                summary_lines.append(f"  * Multi-model mean (MMM) trend: {m_tr:+.2f} m³/s per decade ({p_m_str})")
        summary_lines.append("=" * 80 + "\n")

        summary_output = "\n".join(summary_lines)
        print(summary_output)
        sys.stdout.flush()
        logging.info(summary_output)

        # 2. Get lowflow Threshold (LNWL 970 m3/s)
        threshold_lowflow = 970.0
        if discharge_data_loaded:
            threshold_lowflow = discharge_data_loaded.get('winter_lowflow_lnwl', 970.0)

        # Create figure with 3 subplots (map on left, timeseries in middle, trend distribution on right)
        fig = plt.figure(figsize=(15.5, 5.2))
        gs = gridspec.GridSpec(1, 3, width_ratios=[1.0, 1.25, 0.55], wspace=0.25)
        
        # Define shape CRS: Lambert Azimuthal Equal Area based on zones.prj
        shape_crs = ccrs.LambertAzimuthalEqualArea(
            central_longitude=20.0, 
            central_latitude=55.0, 
            globe=ccrs.Globe(ellipse=None, semimajor_axis=6370997.0, semiminor_axis=6370997.0)
        )
        
        map_aspect = 0.5216  # Fallback aspect ratio if shapefile loading fails
        
        # --- Top Subplot: Map ---
        ax_map = cast(Any, fig.add_subplot(gs[0], projection=ccrs.PlateCarree()))
        
        # Add high-resolution satellite background 
        import cartopy.io.img_tiles as cimgt
        request = cimgt.GoogleTiles(style='satellite')
        ax_map.add_image(request, 6)  # type: ignore # zoom level 6 is usually good for a region like a large river basin
        
        ax_map.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='white', zorder=5)
        ax_map.add_feature(cfeature.BORDERS, linestyle='-', linewidth=0.8, edgecolor='white', zorder=5)
        
        try:
            shapefile_path = '/nas/home/vlw/Desktop/STREAM/hydro-units-files/zones.shp'
            reader = shpreader.Reader(shapefile_path)
            geometries = list(reader.geometries())
            
            # Merge all individual sub-shapes to get only the outer boundary
            import shapely.ops
            merged_geom = shapely.ops.unary_union(geometries)
            
            # Plot only the outer boundary
            ax_map.add_geometries([merged_geom], crs=shape_crs, edgecolor='red', facecolor='none', linewidth=1.5, zorder=10)
            
            # Extract bounds from geometries
            bounds = [
                min(g.bounds[0] for g in geometries),
                min(g.bounds[1] for g in geometries),
                max(g.bounds[2] for g in geometries),
                max(g.bounds[3] for g in geometries)
            ]
            
            # Transform bounds to PlateCarree to set extent properly
            # bounds are [minx, miny, maxx, maxy] in shape_crs
            import shapely.geometry as sgeom
            box = sgeom.box(bounds[0], bounds[1], bounds[2], bounds[3])
            projected_box = ccrs.PlateCarree().project_geometry(box, shape_crs)
            p_bounds = projected_box.bounds
            
            # Allow some padding (roughly 2 degrees)
            buffer_lon = 3.3
            buffer_lat = 2.0
            ax_map.set_extent([p_bounds[0] - buffer_lon, p_bounds[2] + buffer_lon,
                               p_bounds[1] - buffer_lat, p_bounds[3] + buffer_lat], crs=ccrs.PlateCarree())
            ax_map.set_title("(a) Upper Danube Basin", fontsize=12, weight='bold', loc='left')
            
            # Calculate dynamic aspect ratio for PlateCarree map
            lon_min = p_bounds[0] - buffer_lon
            lon_max = p_bounds[2] + buffer_lon
            lat_min = p_bounds[1] - buffer_lat
            lat_max = p_bounds[3] + buffer_lat
            # Match equirectangular aspect ratio of PlateCarree axes to ensure identical size
            map_aspect = (lat_max - lat_min) / (lon_max - lon_min)
            
            # --- Add Rivers (Clipped to Basin) and Labels ---
            try:
                # Project the basin geometry to PlateCarree for intersection with Cartopy features
                projected_basin = ccrs.PlateCarree().project_geometry(merged_geom, shape_crs)
                
                def plot_scaled_rivers(name):
                    shp = shpreader.natural_earth(resolution='10m', category='physical', name=name)
                    reader = shpreader.Reader(shp)
                    for rec in reader.records():
                        geom = rec.geometry
                        if geom.intersects(projected_basin):
                            clipped = geom.intersection(projected_basin)
                            # Use strokeweig to determine thickness (Danube is typically higher, e.g. ~2.0+, minor streams ~0.3)
                            sw = rec.attributes.get('strokeweig', 0.8)
                            try:
                                lw = max(0.8, float(sw) * 2.0)
                            except:
                                lw = 1.6
                            ax_map.add_geometries([clipped], crs=ccrs.PlateCarree(), 
                                                  edgecolor='dodgerblue', facecolor='none', 
                                                  linewidth=lw, zorder=6)
                
                plot_scaled_rivers('rivers_lake_centerlines')
                plot_scaled_rivers('rivers_europe')
    
                import matplotlib.patheffects as pe
                
                # We need the transform object for annotate
                transform = ccrs.PlateCarree()._as_mpl_transform(ax_map)

                ann_dan = ax_map.annotate('Danube', xy=(14.8, 48.25), xycoords=transform,
                                xytext=(12.8, 49.3), textcoords=transform,
                                arrowprops=dict(arrowstyle="->", color="black", lw=2),
                                color='black', fontsize=12, weight='bold', style='italic', zorder=12,
                                path_effects=[pe.withStroke(linewidth=5, foreground="white")])
                if ann_dan.arrow_patch:
                    ann_dan.arrow_patch.set_path_effects([pe.withStroke(linewidth=5, foreground="white")])
                                
                # Add Korneuburg gauge
                korn_lon, korn_lat = 16.326, 48.344
                ax_map.scatter(korn_lon, korn_lat, color='red', s=60, zorder=13, transform=ccrs.PlateCarree(), 
                               edgecolor='black', linewidth=1.0)
                
                ann_kor = ax_map.annotate('Korneuburg', xy=(korn_lon, korn_lat), xycoords=transform,
                                xytext=(korn_lon - 1.0, korn_lat + 1.2), textcoords=transform,
                                arrowprops=dict(arrowstyle="->", color="black", lw=2),
                                color='black', fontsize=12, weight='bold', zorder=13,
                                path_effects=[pe.withStroke(linewidth=5, foreground="white")])
                if ann_kor.arrow_patch:
                    ann_kor.arrow_patch.set_path_effects([pe.withStroke(linewidth=5, foreground="white")])
            except Exception as e:
                logging.error(f"Failed to add clipped rivers: {e}")

        except Exception as e:
            logging.error(f"Failed to plot shapefile map: {e}")
            ax_map.set_title("Study Region (Shapefile error)", weight='bold', loc='left')
        # --- Bottom Subplot: Timeseries ---
        ax = fig.add_subplot(gs[1])
        
        # Colors: SSP5-8.5 (blueish), SSP2-4.5 (orangeish)
        # We use standard color for SSP5-8.5 since it was midnightblue previously.
        if df_stats_ssp585 is not None:
            ax.fill_between(df_stats_ssp585['year'], df_stats_ssp585['p2_5'], df_stats_ssp585['p97_5'], 
                            color='royalblue', alpha=0.3, label='SSP5-8.5 100% Model Spread')
            ax.plot(df_stats_ssp585['year'], df_stats_ssp585['mmm'], 
                    color='midnightblue', linewidth=2, linestyle='-', label=f'SSP5-8.5 MMM (n={n_ssp585})')

        if df_stats_ssp245 is not None:
            ax.fill_between(df_stats_ssp245['year'], df_stats_ssp245['p2_5'], df_stats_ssp245['p97_5'], 
                            color='darkorange', alpha=0.3, label='SSP2-4.5 100% Model Spread')
            ax.plot(df_stats_ssp245['year'], df_stats_ssp245['mmm'], 
                    color='darkorange', linewidth=2, linestyle='-.', label=f'SSP2-4.5 MMM (n={n_ssp245})')

        # Linear trend calculation and plotting for historical (1960-2014) & projected (2015-2100)
        def _compute_and_plot_trend(df_stats, start_yr, end_yr, color, linestyle=':'):
            if df_stats is None or df_stats.empty:
                return None
            col = 'mmm_raw' if 'mmm_raw' in df_stats.columns else 'mmm'
            mask = (df_stats['year'] >= start_yr) & (df_stats['year'] <= end_yr) & (~df_stats[col].isna())
            sub = df_stats[mask]
            if len(sub) < 2:
                return None
            slope, intercept, r_val, p_val, std_err = linregress(sub['year'], sub[col])
            x_vals = sub['year'].values
            y_vals = slope * x_vals + intercept
            ax.plot(x_vals, y_vals, color=color, linestyle=linestyle, linewidth=1.8, alpha=0.9, label='_nolegend_')
            
            trend_dec = slope * 10.0
            p_str = "p < 0.01" if p_val < 0.01 else f"p = {p_val:.2f}"
            return f"{trend_dec:+.1f} m³/s/dec ({p_str})"

        t_hist_245 = _compute_and_plot_trend(df_stats_ssp245, 1960, 2014, 'darkorange', ':')
        t_proj_245 = _compute_and_plot_trend(df_stats_ssp245, 2015, 2100, 'darkorange', ':')
        t_hist_585 = _compute_and_plot_trend(df_stats_ssp585, 1960, 2014, 'midnightblue', ':')
        t_proj_585 = _compute_and_plot_trend(df_stats_ssp585, 2015, 2100, 'midnightblue', ':')

        trend_lines_text = []
        if t_hist_245 or t_proj_245 or t_hist_585 or t_proj_585:
            trend_lines_text.append("Linear Trends (dotted):")
            if t_hist_245 or t_hist_585:
                hist_parts = []
                if t_hist_245:
                    hist_parts.append(f"SSP2-4.5: {t_hist_245}")
                if t_hist_585:
                    hist_parts.append(f"SSP5-8.5: {t_hist_585}")
                trend_lines_text.append("Hist. (1960–2014):  " + " | ".join(hist_parts))
            if t_proj_245 or t_proj_585:
                proj_parts = []
                if t_proj_245:
                    proj_parts.append(f"SSP2-4.5: {t_proj_245}")
                if t_proj_585:
                    proj_parts.append(f"SSP5-8.5: {t_proj_585}")
                trend_lines_text.append("Proj. (2015–2100):  " + " | ".join(proj_parts))
            
            ax.text(0.03, 0.05, "\n".join(trend_lines_text), transform=ax.transAxes,
                    fontsize=7.5, verticalalignment='bottom', horizontalalignment='left',
                    bbox=dict(boxstyle='round,pad=0.35', facecolor='white', alpha=0.9, edgecolor='lightgray'))

        ax.set_title('(b) Annual Min. 30-Day Discharge', weight='bold', loc='left')
        ax.set_ylabel('Discharge (m³/s)')
        ax.set_xlabel('Year')
        ax.grid(True, linestyle=':', alpha=0.7)
        
        # Limit the x-axis properly
        min_year = 1960
        max_year = 2100
        ax.set_xlim(min_year, max_year)
        
        # --- Right Subplot: 2015–2100 Discharge Trends Distribution (Panel c) ---
        ax_c = fig.add_subplot(gs[2])
        
        ax_c.axhline(0, color='gray', linestyle='--', linewidth=1.0, alpha=0.7, zorder=1)

        if len(trends_ssp245) > 0:
            bp245 = ax_c.boxplot([trends_ssp245], positions=[1], widths=0.4, patch_artist=True,
                                 boxprops=dict(facecolor='darkorange', alpha=0.25, edgecolor='darkorange', linewidth=1.2),
                                 medianprops=dict(color='darkorange', linewidth=2.0),
                                 whiskerprops=dict(color='darkorange', linewidth=1.2),
                                 capprops=dict(color='darkorange', linewidth=1.2),
                                 flierprops=dict(marker='', color='none'))
            np.random.seed(42)
            jitter245 = 1 + np.random.uniform(-0.08, 0.08, size=len(trends_ssp245))
            ax_c.scatter(jitter245, trends_ssp245, color='darkorange', alpha=0.7, s=30, zorder=3, edgecolors='none')
            ax_c.scatter([1], [mmm_trend_245], color='crimson', marker='D', s=60, zorder=4, edgecolor='black', linewidth=0.8)

        if len(trends_ssp585) > 0:
            bp585 = ax_c.boxplot([trends_ssp585], positions=[2], widths=0.4, patch_artist=True,
                                 boxprops=dict(facecolor='royalblue', alpha=0.25, edgecolor='midnightblue', linewidth=1.2),
                                 medianprops=dict(color='midnightblue', linewidth=2.0),
                                 whiskerprops=dict(color='midnightblue', linewidth=1.2),
                                 capprops=dict(color='midnightblue', linewidth=1.2),
                                 flierprops=dict(marker='', color='none'))
            np.random.seed(43)
            jitter585 = 2 + np.random.uniform(-0.08, 0.08, size=len(trends_ssp585))
            ax_c.scatter(jitter585, trends_ssp585, color='midnightblue', alpha=0.7, s=30, zorder=3, edgecolors='none')
            ax_c.scatter([2], [mmm_trend_585], color='crimson', marker='D', s=60, zorder=4, edgecolor='black', linewidth=0.8)

        # Legend items for Panel c
        ax_c.scatter([], [], color='gray', marker='o', s=30, label='Model Member')
        ax_c.scatter([], [], color='crimson', marker='D', s=50, edgecolor='black', linewidth=0.8, label='MMM Trend')

        ax_c.set_xticks([1, 2])
        ax_c.set_xticklabels(['SSP2-4.5', 'SSP5-8.5'], weight='bold')
        ax_c.set_xlim(0.4, 2.6)
        ax_c.set_ylabel('30-Day Min. Discharge Trend\n(m³/s per decade)')
        ax_c.set_title('(c) 2015–2100 Trends', weight='bold', loc='left')
        ax_c.grid(True, linestyle=':', alpha=0.7)
        ax_c.legend(loc='upper right', frameon=True, fontsize=7.5, facecolor='white', framealpha=0.9)

        plt.suptitle("Study Region and Projected 30-Day Minimum Discharge", weight='bold', y=0.97, fontsize=15)
        fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.18, wspace=0.28)
        
        # Center the figure legend at the bottom of the entire figure
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.01), ncol=4, frameon=False, fontsize=9.0)
        filename = f"final_figure_1_storyline_discharge_events_{scenario}.png"
        filepath = os.path.join(config.PLOT_DIR, filename)
        plt.savefig(filepath, dpi=600, bbox_inches='tight')
        pdf_filepath = os.path.join(config.PLOT_DIR, f"final_figure_1_storyline_discharge_events_{scenario}.pdf")
        plt.savefig(pdf_filepath, bbox_inches='tight')
        plt.close(fig)
        logging.info(f"Saved discharge events timeseries plot to {filepath} and {pdf_filepath}")

    @staticmethod
    def plot_discharge_events_extreme_timeseries(cmip6_results, discharge_data_loaded, config, scenario, target_gwl=None):
        """
        Plots a 2x2 grid.
        Row 1: Extreme Models (Summer left, Winter right)
        Row 2: Non-Extreme Models (Summer left, Winter right)
        Adding LNWL (970) as threshold.
        """
        logging.info(f"Plotting extreme discharge events timeseries for {scenario}...")
        Visualizer.ensure_plot_dir_exists()
        scenario_title = Visualizer._format_scenario_title(scenario)
        
        metric_timeseries = cmip6_results.get('model_metric_timeseries', {})
        if not metric_timeseries:
            logging.warning("No model metric timeseries available.")
            return

        storyline_classification_2d = cmip6_results.get('storyline_classification_2d', {})
        extreme_models = {'Summer': [], 'Winter': []}
        non_extreme_models = {'Summer': [], 'Winter': []}

        if storyline_classification_2d:
            gwls_present = [gwl for gwl in config.GLOBAL_WARMING_LEVELS if gwl in storyline_classification_2d]
            if gwls_present:
                if target_gwl is None:
                    target_gwl = 3.0 if 3.0 in gwls_present else max(gwls_present)
                
                # Check if target_gwl exists in classification
                if target_gwl in storyline_classification_2d:
                    extreme_models['Summer'] = storyline_classification_2d[target_gwl].get('JJA_Extreme Models', [])
                    non_extreme_models['Summer'] = storyline_classification_2d[target_gwl].get('JJA_Non-Extreme Models', [])
                    extreme_models['Winter'] = storyline_classification_2d[target_gwl].get('DJF_Extreme Models', [])
                    non_extreme_models['Winter'] = storyline_classification_2d[target_gwl].get('DJF_Non-Extreme Models', [])
                else:
                    logging.warning(f"Target GWL {target_gwl} not found in classification data.")

        data_by_season = {'Summer': [], 'Winter': []}
        season_keys = {'Summer': '30Q_low_summer', 'Winter': '30Q_low_winter'}

        for season, metric_key in season_keys.items():
            for key, ts_dict in metric_timeseries.items():
                if not key.endswith(scenario):
                    continue
                if metric_key in ts_dict:
                    da = ts_dict[metric_key]
                    if da is not None:
                        try:
                            df = da.to_dataframe(name='discharge')
                        except ValueError:
                            df = da.to_dataframe()
                            if len(df.columns) == 1:
                                df.columns = ['discharge']
                            
                        if 'year' in df.index.names:
                            df = df.reset_index()
                        
                        model_name = key.split('_')[0] 
                        df['model'] = model_name
                        data_by_season[season].append(df)
        
        threshold_lowflow = 970.0
        if discharge_data_loaded:
            threshold_lowflow = discharge_data_loaded.get('winter_lowflow_lnwl', 970.0)

        fig, axs = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)
        
        def _plot_combined_panel(ax, df_all, ext_list, non_ext_list, title):
            if df_all.empty:
                return
            
            def _plot_group(model_list, label_prefix, color_line):
                target_names = [m.split('_')[0] for m in model_list]
                df_target = df_all[df_all['model'].isin(target_names)]

                if not df_target.empty:
                    df_target_stats = df_target.groupby('year')['discharge'].agg(['mean', 'min', 'max']).reset_index()
                    df_target_stats.columns = ['year', 'mmm', 'p10', 'p90']

                    # Apply 5-year moving average
                    df_target_stats['mmm'] = df_target_stats['mmm'].rolling(window=5, center=True).mean()
                    df_target_stats['p10'] = df_target_stats['p10'].rolling(window=5, center=True).mean()
                    df_target_stats['p90'] = df_target_stats['p90'].rolling(window=5, center=True).mean()

                    ax.fill_between(df_target_stats['year'], df_target_stats['p10'], df_target_stats['p90'], color=color_line, alpha=0.3, label=f'{label_prefix} 100% Model Spread')
                    ax.plot(df_target_stats['year'], df_target_stats['mmm'], color=color_line, linewidth=2, label=f'{label_prefix} MMM (5-yr MA, n={len(target_names)})')

            _plot_group(ext_list, 'Increasing Freq.', '#b2182b')
            _plot_group(non_ext_list, 'Decreasing Freq.', '#2166ac')
            
            ax.set_title(title, fontsize=12, weight='bold')
            ax.grid(True, linestyle=':', alpha=0.7)
            ax.set_xlim(2015, 2100)

        # Plot Summer (Col 0)
        df_summer = pd.concat(data_by_season['Summer'], ignore_index=True) if data_by_season['Summer'] else pd.DataFrame()
        _plot_combined_panel(axs[0], df_summer, extreme_models['Summer'], non_extreme_models['Summer'], f'Summer Half-Year ({scenario_title})')

        # Plot Winter (Col 1)
        df_winter = pd.concat(data_by_season['Winter'], ignore_index=True) if data_by_season['Winter'] else pd.DataFrame()
        _plot_combined_panel(axs[1], df_winter, extreme_models['Winter'], non_extreme_models['Winter'], f'Winter Half-Year ({scenario_title})')

        axs[0].set_ylabel('Discharge (m³/s)', fontsize=10)
        axs[1].set_ylabel('Discharge (m³/s)', fontsize=10)
        axs[1].tick_params(labelleft=True)
        axs[0].set_xlabel('Year', fontsize=10)
        axs[1].set_xlabel('Year', fontsize=10)
        
        # Limit y-axis as requested
        axs[0].set_ylim(top=1750)

        handles, labels = axs[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.02), frameon=False, fontsize=10)

        plt.suptitle(f'Annual Minimum 30-Day Discharge ({scenario_title})', fontsize=16, weight='bold')
        fig.tight_layout()
        fig.subplots_adjust(top=0.88, bottom=0.15)
        filename = f"storyline_discharge_events_extremes_{scenario}.png"
        filepath = os.path.join(config.PLOT_DIR, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logging.info(f"Saved extreme discharge events timeseries plot to {filepath}")


    @staticmethod
    def plot_final_figure_4_combined(cmip6_results, discharge_data_loaded, pr_stored_composites, tas_stored_composites, config, scenario, target_gwl):
        """
        Creates a combined Figure 4:
        Row 0: Extreme vs Non-Extreme Danube Basin precipitation timeseries (Summer, Winter)
        Row 1: Future Precipitation Difference maps (Summer, Winter)
        Row 2: Extreme vs Non-Extreme Danube Basin temperature timeseries (Summer, Winter)
        Row 3: Future Temperature Difference maps (Summer, Winter)
        Row 4: Extreme vs Non-Extreme discharge timeseries (Summer, Winter)
        """
        logging.info(f"Plotting combined final figure 4 for GWL {target_gwl}°C, {scenario}...")
        Visualizer.ensure_plot_dir_exists()
        
        # --- 1. PREPARE TIME SERIES DATA (DISCHARGE, PR, TAS) ---
        metric_timeseries = cmip6_results.get('model_metric_timeseries', {})
        storyline_classification_2d = cmip6_results.get('storyline_classification_2d', {})
        extreme_models = {'Summer': [], 'Winter': []}
        non_extreme_models = {'Summer': [], 'Winter': []}

        if target_gwl in storyline_classification_2d:
            extreme_models['Summer'] = storyline_classification_2d[target_gwl].get('JJA_Extreme Models', [])
            non_extreme_models['Summer'] = storyline_classification_2d[target_gwl].get('JJA_Non-Extreme Models', [])
            extreme_models['Winter'] = storyline_classification_2d[target_gwl].get('DJF_Extreme Models', [])
            non_extreme_models['Winter'] = storyline_classification_2d[target_gwl].get('DJF_Non-Extreme Models', [])

        # Discharge timeseries
        data_by_season = {'Summer': [], 'Winter': []}
        season_keys = {'Summer': '30Q_low_summer', 'Winter': '30Q_low_winter'}

        for sn, mk in season_keys.items():
            for key, ts_dict in metric_timeseries.items():
                if not key.endswith(scenario): continue
                if mk in ts_dict:
                    da = ts_dict[mk]
                    if da is not None:
                        try:
                            df = da.to_dataframe(name='discharge')
                        except:
                            df = da.to_dataframe()
                            if len(df.columns) == 1: df.columns = ['discharge']
                        if 'year' in df.index.names: df = df.reset_index()
                        model_name = key.split('_')[0] 
                        df['model'] = model_name
                        data_by_season[sn].append(df)

        # PR and TAS timeseries and spatial catchment maps loaded from bias-adjusted daily data
        catchment_dirs = [
            '/nas/home/vlw/Desktop/STREAM/final-bias-adjusted-data',
            '/nas/home/vlw/Desktop/STREAM/copernicus-final-adjusted-data',
            '/nas/home/vlw/Desktop/STREAM/in-catchment-data',
            '/nas/home/vlw/Desktop/STREAM/copernicus-in-catchment'
        ]

        def find_catchment_files(model, scn):
            for c_dir in catchment_dirs:
                ba_pr1 = os.path.join(c_dir, f"MONTHLY_*_{model}_*pr_{scn}_count-*.csv")
                ba_pr2 = os.path.join(c_dir, f"MONTHLY_*_{model}_pr_{scn}_count-*.csv")
                f_pr_ba = sorted(glob.glob(ba_pr1) + glob.glob(ba_pr2))
                
                ba_tas1 = os.path.join(c_dir, f"MONTHLY_*_{model}_*tas_{scn}_count-*.csv")
                ba_tas2 = os.path.join(c_dir, f"MONTHLY_*_{model}_tas_{scn}_count-*.csv")
                f_tas_ba = sorted(glob.glob(ba_tas1) + glob.glob(ba_tas2))
                
                if f_pr_ba and f_tas_ba:
                    return f_pr_ba[0], f_tas_ba[0]
                
                p1_pr = os.path.join(c_dir, f"{model}_pr_{scn}_*_in-catchment-units.csv")
                p2_pr = os.path.join(c_dir, f"{model}_*_pr_{scn}_*_in-catchment-units.csv")
                f_pr = sorted(glob.glob(p1_pr) + glob.glob(p2_pr))
                
                p1_tas = os.path.join(c_dir, f"{model}_tas_{scn}_*_in-catchment-units.csv")
                p2_tas = os.path.join(c_dir, f"{model}_*_tas_{scn}_*_in-catchment-units.csv")
                f_tas = sorted(glob.glob(p1_tas) + glob.glob(p2_tas))
                
                if f_pr and f_tas:
                    return f_pr[0], f_tas[0]
            return None, None

        all_model_keys = set()
        if extreme_models['Summer']: all_model_keys.update(extreme_models['Summer'])
        if non_extreme_models['Summer']: all_model_keys.update(non_extreme_models['Summer'])
        if extreme_models['Winter']: all_model_keys.update(extreme_models['Winter'])
        if non_extreme_models['Winter']: all_model_keys.update(non_extreme_models['Winter'])
        all_models = list(set([m.split('_')[0] for m in all_model_keys]))
        if not all_models:
            all_models = list(set([k.split('_')[0] for k in metric_timeseries.keys() if k.endswith(scenario)]))

        gwl_thresh = cmip6_results.get('gwl_threshold_years', cmip6_results.get('gwl_years', {}))

        pr_ts_by_season = {'Summer': [], 'Winter': []}
        tas_ts_by_season = {'Summer': [], 'Winter': []}
        pr_catchment_gwl = {'Summer': {}, 'Winter': {}}
        tas_catchment_gwl = {'Summer': {}, 'Winter': {}}

        for model in all_models:
            pr_file, tas_file = find_catchment_files(model, scenario)
            if not pr_file or not tas_file:
                continue
            try:
                df_pr = pd.read_csv(pr_file, sep='\t', skiprows=1)
                df_tas = pd.read_csv(tas_file, sep='\t', skiprows=1)
                
                p_cols = [c for c in df_pr.columns if c.startswith('P_')]
                t_cols = [c for c in df_tas.columns if c.startswith('T_')]
                
                df_pr['pr_mean'] = df_pr[p_cols].mean(axis=1)
                df_tas['tas_mean'] = df_tas[t_cols].mean(axis=1)
                
                df = pd.merge(df_pr[['year', 'month', 'day', 'pr_mean'] + p_cols],
                              df_tas[['year', 'month', 'day', 'tas_mean'] + t_cols],
                              on=['year', 'month', 'day'])
                              
                if df['tas_mean'].mean() > 100:
                    df['tas_mean'] -= 273.15
                    df[t_cols] -= 273.15
                    
                pr_s_list, pr_w_list = [], []
                tas_s_list, tas_w_list = [], []
                
                min_yr = int(df['year'].min())
                max_yr = int(df['year'].max())
                
                for y in range(max(1960, min_yr), max_yr + 1):
                    sub_s = df[(df['year'] == y) & (df['month'].isin([5, 6, 7, 8, 9, 10]))]
                    if not sub_s.empty:
                        pr_s_list.append({'year': y, 'precip': sub_s['pr_mean'].mean(), 'model': model})
                        tas_s_list.append({'year': y, 'tas': sub_s['tas_mean'].mean(), 'model': model})
                        
                    sub_w = df[((df['year'] == y - 1) & (df['month'].isin([11, 12]))) | ((df['year'] == y) & (df['month'].isin([1, 2, 3, 4])))]
                    if not sub_w.empty:
                        pr_w_list.append({'year': y, 'precip': sub_w['pr_mean'].mean(), 'model': model})
                        tas_w_list.append({'year': y, 'tas': sub_w['tas_mean'].mean(), 'model': model})
                        
                if pr_s_list: pr_ts_by_season['Summer'].append(pd.DataFrame(pr_s_list))
                if pr_w_list: pr_ts_by_season['Winter'].append(pd.DataFrame(pr_w_list))
                if tas_s_list: tas_ts_by_season['Summer'].append(pd.DataFrame(tas_s_list))
                if tas_w_list: tas_ts_by_season['Winter'].append(pd.DataFrame(tas_w_list))

                # Spatial GWL calculations
                m_gwl_dict = gwl_thresh.get(model) or gwl_thresh.get(f"{model}_{scenario}")
                gwl_yr = None
                if isinstance(m_gwl_dict, dict):
                    gwl_yr = m_gwl_dict.get(target_gwl)
                elif isinstance(m_gwl_dict, (int, float, np.integer)):
                    gwl_yr = m_gwl_dict
                    
                if gwl_yr is not None and np.isfinite(gwl_yr):
                    gwl_yr = int(gwl_yr)
                    w_start, w_end = max(1960, gwl_yr - 15), min(2099, gwl_yr + 15)
                    
                    df_gwl_s = df[(df['year'] >= w_start) & (df['year'] <= w_end) & (df['month'].isin([5, 6, 7, 8, 9, 10]))]
                    if not df_gwl_s.empty:
                        pr_catchment_gwl['Summer'][model] = df_gwl_s[p_cols].mean(axis=0).values
                        tas_catchment_gwl['Summer'][model] = df_gwl_s[t_cols].mean(axis=0).values
                        
                    df_gwl_w = df[(df['year'] >= w_start) & (df['year'] <= w_end) & (((df['year'] >= w_start - 1) & (df['month'].isin([11, 12]))) | ((df['year'] <= w_end) & (df['month'].isin([1, 2, 3, 4]))))]
                    if not df_gwl_w.empty:
                        pr_catchment_gwl['Winter'][model] = df_gwl_w[p_cols].mean(axis=0).values
                        tas_catchment_gwl['Winter'][model] = df_gwl_w[t_cols].mean(axis=0).values
                        
            except Exception as e:
                logging.error(f"Error loading catchment file for {model} in Fig 4: {e}")

        def compute_catchment_diff(catchment_dict, high_list, low_list):
            high_models = [m.split('_')[0] for m in high_list]
            low_models = [m.split('_')[0] for m in low_list]
            
            h_arrs = [catchment_dict[m] for m in high_models if m in catchment_dict]
            l_arrs = [catchment_dict[m] for m in low_models if m in catchment_dict]
            
            if h_arrs and l_arrs:
                mmm_h = np.mean(h_arrs, axis=0)
                mmm_l = np.mean(l_arrs, axis=0)
                return mmm_h - mmm_l
            return None

        diff_pr_s = compute_catchment_diff(pr_catchment_gwl['Summer'], extreme_models['Summer'], non_extreme_models['Summer'])
        diff_pr_w = compute_catchment_diff(pr_catchment_gwl['Winter'], extreme_models['Winter'], non_extreme_models['Winter'])

        diff_tas_s = compute_catchment_diff(tas_catchment_gwl['Summer'], extreme_models['Summer'], non_extreme_models['Summer'])
        diff_tas_w = compute_catchment_diff(tas_catchment_gwl['Winter'], extreme_models['Winter'], non_extreme_models['Winter'])
        
        # --- 3. SET UP FIGURE ---
        fig = plt.figure(figsize=(9, 17.5))
        gs = gridspec.GridSpec(5, 2, height_ratios=[1, 1.15, 1, 1.15, 1], hspace=0.55, wspace=0.15)
        
        # Helper to convert raw values to percentage change relative to 1985-2014 climatology per model
        def _convert_to_pct_change(df_all, val_col='discharge', hist_start=1985, hist_end=2014):
            if df_all.empty: return df_all
            df_out = []
            for model_name, df_m in df_all.groupby('model'):
                hist_mask = (df_m['year'] >= hist_start) & (df_m['year'] <= hist_end)
                hist_vals = df_m[hist_mask][val_col].dropna()
                if len(hist_vals) < 3:
                    hist_vals = df_m[df_m['year'] <= 2014][val_col].dropna()
                
                if not hist_vals.empty:
                    v_hist = hist_vals.mean()
                    if v_hist != 0 and np.isfinite(v_hist):
                        df_m_copy = df_m.copy()
                        df_m_copy[val_col] = ((df_m_copy[val_col] - v_hist) / v_hist) * 100.0
                        df_out.append(df_m_copy)
                    else:
                        df_out.append(df_m)
                else:
                    df_out.append(df_m)
            return pd.concat(df_out, ignore_index=True) if df_out else pd.DataFrame()

        # Helper to convert raw values to absolute change (°C) relative to baseline per model
        def _convert_to_abs_change(df_all, val_col='tas', hist_start=1850, hist_end=1900):
            if df_all.empty: return df_all
            df_out = []
            for model_name, df_m in df_all.groupby('model'):
                hist_mask = (df_m['year'] >= hist_start) & (df_m['year'] <= hist_end)
                hist_vals = df_m[hist_mask][val_col].dropna()
                if len(hist_vals) < 3:
                    hist_vals = df_m[df_m['year'] <= 2014][val_col].dropna()
                
                if not hist_vals.empty:
                    v_hist = hist_vals.mean()
                    if np.isfinite(v_hist):
                        df_m_copy = df_m.copy()
                        df_m_copy[val_col] = df_m_copy[val_col] - v_hist
                        df_out.append(df_m_copy)
                    else:
                        df_out.append(df_m)
                else:
                    df_out.append(df_m)
            return pd.concat(df_out, ignore_index=True) if df_out else pd.DataFrame()

        # Helper for plotting time series (% change)
        def _p_ts(ax, df_all, ext_list, non_ext_list, title, val_col='discharge', unit_str='%', text_y_ext=0.12, text_y_non=0.04, y_lim=None):
            if df_all.empty: return ([], [])
            ax.axhline(0, color='gray', linestyle='--', alpha=0.6, linewidth=1.0)
            def _p_grp(ml, lbl, clr, l_style='-', text_y=0.04):
                tn = [m.split('_')[0] for m in ml]
                df_t = df_all[df_all['model'].isin(tn)]
                if not df_t.empty:
                    stats = df_t.groupby('year')[val_col].agg(['mean', 'min', 'max']).reset_index()
                    stats.columns = ['year', 'mmm', 'p2_5', 'p97_5']
                    stats['mmm'] = stats['mmm'].rolling(window=5, center=True).mean()
                    stats['p2_5'] = stats['p2_5'].rolling(window=5, center=True).mean()
                    stats['p97_5'] = stats['p97_5'].rolling(window=5, center=True).mean()
                    
                    period_mask = (stats['year'] >= 2015) & (stats['year'] <= 2100)
                    valid = (~np.isnan(stats['mmm'])) & period_mask
                    x_vals = stats['year'][valid]
                    y_vals = stats['mmm'][valid]

                    if len(x_vals) > 1:
                        slope, intercept, _, p_val, _ = linregress(x_vals, y_vals)
                        trend_line = slope * x_vals + intercept
                        ax.plot(x_vals, trend_line, color=clr, linestyle=':', alpha=0.9, linewidth=1.5)
                        
                        trend_per_decade = slope * 10
                        if p_val < 0.01:
                            p_str = "p<0.01"
                        else:
                            p_str = f"p={p_val:.2f}"
                        label_suffix = f" (trend: {trend_per_decade:+.2f}{unit_str}/dec, {p_str})"
                    else:
                        label_suffix = ""

                    ax.fill_between(stats['year'], stats['p2_5'], stats['p97_5'], color=clr, alpha=0.3, label=f'{lbl} 100% Model Spread')
                    ax.plot(stats['year'], stats['mmm'], color=clr, linewidth=2.5, linestyle=l_style, label=f'{lbl} MMM (n={len(tn)}){label_suffix}')
                    
                    if label_suffix:
                        clean_suffix = label_suffix.strip(" ()").replace("trend: ", "Trend: ")
                        ax.text(0.04, text_y, clean_suffix, transform=ax.transAxes, ha='left', va='bottom', color=clr,
                                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0.3))

            _p_grp(ext_list, 'High-Freq.', '#b2182b', '--', text_y=text_y_ext)
            _p_grp(non_ext_list, 'Low-Freq.', '#2166ac', '-.', text_y=text_y_non)
            ax.set_title(title, weight='bold', loc='left', fontsize=12)
            ax.grid(True, linestyle=':', alpha=0.7)
            ax.set_xlim(2015, 2100)
            if y_lim is not None:
                ax.set_ylim(y_lim)
            
            return ax.get_legend_handles_labels()

        # Helper for maps (Catchment-specific polygons matching Danube hydro-units, as in Figure 6)
        shapefile_path = '/nas/home/vlw/Desktop/STREAM/hydro-units-files/zones.shp'
        shape_crs = ccrs.LambertAzimuthalEqualArea(central_longitude=20.0, central_latitude=55.0, globe=ccrs.Globe(semimajor_axis=6370997.0, semiminor_axis=6370997.0))
        buf = 5.0
        ex = [config.BOX_LON_MIN - buf, config.BOX_LON_MAX + buf, config.BOX_LAT_MIN - buf, config.BOX_LAT_MAX + buf]
        import matplotlib.colors as mcolors

        def _p_map(ax, diff_array, title, cmap, norm):
            ax.set_extent(ex, crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='black', zorder=5)
            ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='black', zorder=5)
            ax.add_patch(mpatches.Rectangle((config.BOX_LON_MIN, config.BOX_LAT_MIN), config.BOX_LON_MAX-config.BOX_LON_MIN, config.BOX_LAT_MAX-config.BOX_LAT_MIN, fill=False, edgecolor='magenta', linewidth=1.5, linestyle='--', transform=ccrs.PlateCarree(), zorder=10))

            if diff_array is not None and len(diff_array) == 61:
                try:
                    import pyproj
                    from shapely.ops import transform as shapely_transform
                    import cartopy.io.shapereader as shpreader
                    import shapely.ops
                    
                    reader = shpreader.Reader(shapefile_path)
                    geometries = list(reader.geometries())
                    
                    proj_laea = pyproj.CRS.from_proj4('+proj=laea +lat_0=55 +lon_0=20 +x_0=0 +y_0=0 +a=6370997 +b=6370997 +units=m +no_defs')
                    proj_wgs84 = pyproj.CRS.from_epsg(4326)
                    transformer = pyproj.Transformer.from_crs(proj_laea, proj_wgs84, always_xy=True)
                    
                    geoms_wgs84 = [shapely_transform(transformer.transform, g) for g in geometries]
                    
                    for g_wgs84, val in zip(geoms_wgs84, diff_array):
                        if np.isfinite(val):
                            color = cmap(norm(val))
                            ax.add_geometries([g_wgs84], crs=ccrs.PlateCarree(), facecolor=color, edgecolor='none', linewidth=0, zorder=4)
                    
                    merged_geom = shapely.ops.unary_union(geometries)
                    ax.add_geometries([merged_geom], crs=shape_crs, edgecolor='black', facecolor='none', linewidth=1.2, zorder=6)
                except Exception as e:
                    logging.warning(f"Could not plot catchment spatial map in Fig 4: {e}")
            ax.set_title(title, weight='bold', fontsize=12, loc='left')

        # --- ROW 0: PRECIPITATION TIME SERIES ---
        ax_prts_s = fig.add_subplot(gs[0, 0])
        df_pr_s = pd.concat(pr_ts_by_season['Summer'], ignore_index=True) if pr_ts_by_season['Summer'] else pd.DataFrame()
        df_pr_s_pct = _convert_to_pct_change(df_pr_s, val_col='precip')
        _p_ts(ax_prts_s, df_pr_s_pct, extreme_models['Summer'], non_extreme_models['Summer'], f'(a) Summer Half Year', val_col='precip', unit_str='%')
        ax_prts_s.set_ylabel('Precipitation Change (%)')

        ax_prts_w = fig.add_subplot(gs[0, 1])
        df_pr_w = pd.concat(pr_ts_by_season['Winter'], ignore_index=True) if pr_ts_by_season['Winter'] else pd.DataFrame()
        df_pr_w_pct = _convert_to_pct_change(df_pr_w, val_col='precip')
        handles, labels = _p_ts(ax_prts_w, df_pr_w_pct, extreme_models['Winter'], non_extreme_models['Winter'], f'(b) Winter Half Year', val_col='precip', unit_str='%')
        ax_prts_w.tick_params(labelleft=False)

        pr_ymin = min(ax_prts_s.get_ylim()[0], ax_prts_w.get_ylim()[0])
        pr_ymax = max(ax_prts_s.get_ylim()[1], ax_prts_w.get_ylim()[1])
        ax_prts_s.set_ylim(pr_ymin, pr_ymax)
        ax_prts_w.set_ylim(pr_ymin, pr_ymax)

        # --- ROW 1: PR COMPOSITES (Danube Catchments) ---
        if diff_pr_s is not None or diff_pr_w is not None:
            max_pr_val = 0.0
            if diff_pr_s is not None: max_pr_val = max(max_pr_val, np.nanmax(np.abs(diff_pr_s)))
            if diff_pr_w is not None: max_pr_val = max(max_pr_val, np.nanmax(np.abs(diff_pr_w)))
            d_lim_pr = max(0.2, np.ceil(max_pr_val * 10) / 10.0)
        else:
            d_lim_pr = 1.0

        try:
            cmap_pr = plt.get_cmap('BrBG', 10)
        except:
            cmap_pr = matplotlib.cm.get_cmap('BrBG', 10)
            
        levs_pr = np.linspace(-d_lim_pr, d_lim_pr, 11)
        norm_pr = mcolors.BoundaryNorm(levs_pr, ncolors=cmap_pr.N, clip=False)

        ax_pr_s = fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree())
        _p_map(ax_pr_s, diff_pr_s, "", cmap_pr, norm_pr)
        ax_pr_s.set_title("(c) Summer Half Year", weight='bold', loc='left', fontsize=12)

        ax_pr_w = fig.add_subplot(gs[1, 1], projection=ccrs.PlateCarree())
        _p_map(ax_pr_w, diff_pr_w, "", cmap_pr, norm_pr)
        ax_pr_w.set_title("(d) Winter Half Year", weight='bold', loc='left', fontsize=12)

        # --- ROW 2: TEMPERATURE TIME SERIES (°C change vs 1985-2014) ---
        ax_tasts_s = fig.add_subplot(gs[2, 0])
        df_tas_s = pd.concat(tas_ts_by_season['Summer'], ignore_index=True) if tas_ts_by_season['Summer'] else pd.DataFrame()
        df_tas_s_abs = _convert_to_abs_change(df_tas_s, val_col='tas', hist_start=1850, hist_end=1900)
        _p_ts(ax_tasts_s, df_tas_s_abs, extreme_models['Summer'], non_extreme_models['Summer'], f'(e) Summer Half Year', val_col='tas', unit_str='°C')
        ax_tasts_s.set_ylabel('Temperature Change (°C)')

        ax_tasts_w = fig.add_subplot(gs[2, 1])
        df_tas_w = pd.concat(tas_ts_by_season['Winter'], ignore_index=True) if tas_ts_by_season['Winter'] else pd.DataFrame()
        df_tas_w_abs = _convert_to_abs_change(df_tas_w, val_col='tas', hist_start=1850, hist_end=1900)
        _p_ts(ax_tasts_w, df_tas_w_abs, extreme_models['Winter'], non_extreme_models['Winter'], f'(f) Winter Half Year', val_col='tas', unit_str='°C')
        ax_tasts_w.tick_params(labelleft=False)

        tas_ymin = min(ax_tasts_s.get_ylim()[0], ax_tasts_w.get_ylim()[0])
        tas_ymax = max(ax_tasts_s.get_ylim()[1], ax_tasts_w.get_ylim()[1])
        ax_tasts_s.set_ylim(tas_ymin, tas_ymax)
        ax_tasts_w.set_ylim(tas_ymin, tas_ymax)

        # --- ROW 3: TAS COMPOSITES (Danube Catchments) ---
        if diff_tas_s is not None or diff_tas_w is not None:
            max_tas_val = 0.0
            if diff_tas_s is not None: max_tas_val = max(max_tas_val, np.nanmax(np.abs(diff_tas_s)))
            if diff_tas_w is not None: max_tas_val = max(max_tas_val, np.nanmax(np.abs(diff_tas_w)))
            d_lim_tas = max(0.5, np.ceil(max_tas_val * 2) / 2.0)
        else:
            d_lim_tas = 1.5

        try:
            cmap_tas = plt.get_cmap('RdBu_r', 10)
        except:
            cmap_tas = matplotlib.cm.get_cmap('RdBu_r', 10)

        levs_tas = np.linspace(-d_lim_tas, d_lim_tas, 11)
        norm_tas = mcolors.BoundaryNorm(levs_tas, ncolors=cmap_tas.N, clip=False)

        ax_tas_s = fig.add_subplot(gs[3, 0], projection=ccrs.PlateCarree())
        _p_map(ax_tas_s, diff_tas_s, "", cmap_tas, norm_tas)
        ax_tas_s.set_title("(g) Summer Half Year", weight='bold', loc='left', fontsize=12)

        ax_tas_w = fig.add_subplot(gs[3, 1], projection=ccrs.PlateCarree())
        _p_map(ax_tas_w, diff_tas_w, "", cmap_tas, norm_tas)
        ax_tas_w.set_title("(h) Winter Half Year", weight='bold', loc='left', fontsize=12)

        # --- ROW 4: DISCHARGE TIME SERIES (bottom) ---
        ax_ds_s = fig.add_subplot(gs[4, 0])
        df_s = pd.concat(data_by_season['Summer'], ignore_index=True) if data_by_season['Summer'] else pd.DataFrame()
        df_s_pct = _convert_to_pct_change(df_s, val_col='discharge')
        _p_ts(ax_ds_s, df_s_pct, extreme_models['Summer'], non_extreme_models['Summer'], f'(i) Summer Half Year', val_col='discharge', unit_str='%')
        ax_ds_s.set_ylabel('Discharge Change (%)')

        ax_ds_w = fig.add_subplot(gs[4, 1])
        df_w = pd.concat(data_by_season['Winter'], ignore_index=True) if data_by_season['Winter'] else pd.DataFrame()
        df_w_pct = _convert_to_pct_change(df_w, val_col='discharge')
        _p_ts(ax_ds_w, df_w_pct, extreme_models['Winter'], non_extreme_models['Winter'], f'(j) Winter Half Year', val_col='discharge', unit_str='%')
        ax_ds_w.tick_params(labelleft=False)

        ds_ymin = min(ax_ds_s.get_ylim()[0], ax_ds_w.get_ylim()[0])
        ds_ymax = max(ax_ds_s.get_ylim()[1], ax_ds_w.get_ylim()[1])
        ax_ds_s.set_ylim(ds_ymin, ds_ymax)
        ax_ds_w.set_ylim(ds_ymin, ds_ymax)

        if handles:
            clean_labels = [l.split(' (trend:')[0] if 'MMM' in l else l for l in labels]
            fig.legend(handles, clean_labels, loc='lower center', bbox_to_anchor=(0.5, 0.01), ncol=2, frameon=False, fontsize=9)

        s_t = Visualizer._format_scenario_title(scenario)
        plt.suptitle(f'Danube Basin Climate Drivers & Minimum 30-Day Discharge Trend\n{s_t} | GWL {target_gwl}°C', weight='bold', y=0.985, fontsize=15)
        fig.tight_layout(rect=(0, 0.03, 1, 0.965), h_pad=1.2, w_pad=1.0)
        
        # Dynamically position map colorbars below row 1 (PR maps) and row 3 (TAS maps)
        pos_pr = ax_pr_s.get_position()
        cb_ax_pr = fig.add_axes((0.15, pos_pr.y0 - 0.025, 0.7, 0.008))
        fig.colorbar(ScalarMappable(norm=norm_pr, cmap=cmap_pr), cax=cb_ax_pr, orientation='horizontal', label='Precip. Diff. (mm/day)', extend='both')

        pos_tas = ax_tas_s.get_position()
        cb_ax_tas = fig.add_axes((0.15, pos_tas.y0 - 0.025, 0.7, 0.008))
        fig.colorbar(ScalarMappable(norm=norm_tas, cmap=cmap_tas), cax=cb_ax_tas, orientation='horizontal', label='Temp. Diff. (°C)', extend='both')

        # Add left-aligned section titles above each row
        nudge = 0.015
        fig.text(ax_prts_s.get_position().x0, ax_prts_s.get_position().y1 + nudge, "Danube Basin Precipitation Trend", ha='left', va='bottom', fontsize=11, weight='bold')
        fig.text(ax_pr_s.get_position().x0, ax_pr_s.get_position().y1 + nudge, "Precipitation Difference (High-Freq. - Low-Freq.)", ha='left', va='bottom', fontsize=11, weight='bold')
        fig.text(ax_tasts_s.get_position().x0, ax_tasts_s.get_position().y1 + nudge, "Danube Basin Temperature Trend", ha='left', va='bottom', fontsize=11, weight='bold')
        fig.text(ax_tas_s.get_position().x0, ax_tas_s.get_position().y1 + nudge, "Temperature Difference (High-Freq. - Low-Freq.)", ha='left', va='bottom', fontsize=11, weight='bold')
        fig.text(ax_ds_s.get_position().x0, ax_ds_s.get_position().y1 + nudge, "30-day Minimum Discharge Trend", ha='left', va='bottom', fontsize=11, weight='bold')

        path = os.path.join(config.PLOT_DIR, f"final_figure_4_combined_{scenario}_gwl{target_gwl}.png")
        plt.savefig(path, dpi=600, bbox_inches='tight')
        pdf_path = os.path.join(config.PLOT_DIR, f"final_figure_4_combined_{scenario}_gwl{target_gwl}.pdf")
        plt.savefig(pdf_path, bbox_inches='tight')
        plt.close(fig)
        logging.info(f"Saved combined final figure 4 to {path} and {pdf_path}")


    @staticmethod
    def plot_final_figure_3_u850_and_indices(
        winter_composite, summer_composite,
        gwl, event_key, scenario,
        cmip6_plot_data, reanalysis_plot_data, config,
        winter_model_rps=None, summer_model_rps=None,
        winter_n_total=None, summer_n_total=None,
        fixed_diff_limit=None, gwl_years=None
    ):
        """
        Creates final_figure_3:
        Top: UA850 difference maps (Future: Ext - Non) and Return Period Boxplots (2x2)
        Bottom: Subplots b (Summer Jet Latitude) and e (Winter Jet Speed) from Figure 2 (1x2)
        """
        logging.info(f"Plotting final figure 3 for GWL {gwl}°C, {scenario}...")
        Visualizer.ensure_plot_dir_exists()

        if not winter_composite and not summer_composite:
            logging.warning("No composite data for final figure 3.")
            return

        mask_lon_min = Config.GREENLAND_LON_MIN
        mask_lon_max = Config.GREENLAND_LON_MAX
        mask_lat_min = Config.GREENLAND_LAT_MIN
        mask_lat_max = Config.GREENLAND_LAT_MAX
        for comp in [winter_composite, summer_composite]:
            if comp is None: continue
            for key_arr, data_array in comp.items():
                if isinstance(data_array, xr.DataArray):
                    if data_array.lon.min() >= 0:
                        m_lon_min, m_lon_max = mask_lon_min + 360, mask_lon_max + 360
                    else:
                        m_lon_min, m_lon_max = mask_lon_min, mask_lon_max
                    mask = ((data_array.lon >= m_lon_min) & (data_array.lon <= m_lon_max) &
                            (data_array.lat >= mask_lat_min) & (data_array.lat <= mask_lat_max))
                    comp[key_arr] = data_array.where(~mask)

        all_diff_maps = []
        for comp in [winter_composite, summer_composite]:
            if comp is None: continue
            for key in ['diff_ext_non_future']: 
                m = comp.get(key)
                if m is not None: all_diff_maps.append(m)

        diff_limit = fixed_diff_limit if fixed_diff_limit is not None else 1.0
        diff_levels = None
        if fixed_diff_limit is not None:
            diff_levels = np.round(np.linspace(-diff_limit, diff_limit, 13), 1)
        elif all_diff_maps:
            all_vals = np.concatenate([m.values.ravel() for m in all_diff_maps])
            all_vals = all_vals[np.isfinite(all_vals)]
            if len(all_vals) > 0:
                import math
                limit = np.percentile(np.abs(all_vals), 98)
                if limit > 0: diff_limit = math.ceil(limit)
                diff_levels = np.round(np.linspace(-diff_limit, diff_limit, 13), 1)

        all_abs_maps = []
        for comp in [winter_composite, summer_composite]:
            if comp is None: continue
            for key in ['fut_extreme_mean', 'future_non_extreme_mean']:
                m = comp.get(key)
                if m is not None: all_abs_maps.append(m)
        contour_levels = None
        if all_abs_maps:
            abs_vals = np.concatenate([m.values.ravel() for m in all_abs_maps])
            abs_vals = abs_vals[np.isfinite(abs_vals)]
            if len(abs_vals) > 0:
                contour_levels = np.linspace(np.percentile(abs_vals, 2), np.percentile(abs_vals, 98), 15)

        fig = plt.figure(figsize=(5.9, 10.0))
        # Split layout for better control of spaces and subtitles
        gs_top = gridspec.GridSpec(1, 2, top=0.91, bottom=0.73, wspace=0.10, left=0.12, right=0.95)
        gs_cbar = gridspec.GridSpec(1, 1, top=0.725, bottom=0.705, left=0.235, right=0.835)
        gs_bottom = gridspec.GridSpec(2, 2, top=0.55, bottom=0.13, wspace=0.35, hspace=0.55, left=0.12, right=0.95)

        import matplotlib.colors as mcolors
        import matplotlib.patheffects as pe
        try:
            custom_cmap = plt.get_cmap('RdBu_r')
        except:
            custom_cmap = matplotlib.cm.get_cmap('RdBu_r')
        norm = mcolors.BoundaryNorm(diff_levels, ncolors=custom_cmap.N, clip=False) if diff_levels is not None else None

        extent = [-105, 40, 0, 90]
        
        def _add_map_features(ax):
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linewidth=0.5, alpha=0.5)
            # Danube Box (original)
            ax.add_patch(mpatches.Rectangle((Config.BOX_LON_MIN, Config.BOX_LAT_MIN), 
                                              Config.BOX_LON_MAX - Config.BOX_LON_MIN, 
                                              Config.BOX_LAT_MAX - Config.BOX_LAT_MIN,
                                              fill=False, edgecolor='magenta', linewidth=1.5, linestyle='-', transform=ccrs.PlateCarree(), zorder=10,
                                              path_effects=[pe.withStroke(linewidth=3, foreground="white")]))
            # Jet Speed Box (Red)
            ax.add_patch(mpatches.Rectangle((Config.JET_SPEED_BOX_LON_MIN, Config.JET_SPEED_BOX_LAT_MIN), 
                                              Config.JET_SPEED_BOX_LON_MAX - Config.JET_SPEED_BOX_LON_MIN, 
                                              Config.JET_SPEED_BOX_LAT_MAX - Config.JET_SPEED_BOX_LAT_MIN,
                                              fill=False, edgecolor='red', linewidth=1.2, transform=ccrs.PlateCarree(), zorder=11,
                                              path_effects=[pe.withStroke(linewidth=2.5, foreground="white")]))
            # Jet Lat Box (Blue)
            ax.add_patch(mpatches.Rectangle((Config.JET_LAT_BOX_LON_MIN, Config.JET_LAT_BOX_LAT_MIN), 
                                              Config.JET_LAT_BOX_LON_MAX - Config.JET_LAT_BOX_LON_MIN, 
                                              Config.JET_LAT_BOX_LAT_MAX - Config.JET_LAT_BOX_LAT_MIN,
                                              fill=False, edgecolor='blue', linewidth=1.2, transform=ccrs.PlateCarree(), zorder=11,
                                              path_effects=[pe.withStroke(linewidth=2.5, foreground="white")]))

        def _plot_diff(ax, diff_map, sig_mask, title, contour_map=None):
            _add_map_features(ax)
            cf = None
            if diff_map is not None:
                if norm is not None:
                    cf = ax.pcolormesh(diff_map.lon, diff_map.lat, diff_map, cmap=custom_cmap, norm=norm, transform=ccrs.PlateCarree())
                else:
                    cf = ax.pcolormesh(diff_map.lon, diff_map.lat, diff_map, cmap=custom_cmap, vmin=-diff_limit, vmax=diff_limit, transform=ccrs.PlateCarree())
                if contour_map is not None and contour_levels is not None:
                    ref_levels = contour_levels[::2]
                    cs = ax.contour(contour_map.lon, contour_map.lat, contour_map, levels=ref_levels, colors='gray', linewidths=1.2, alpha=0.9, transform=ccrs.PlateCarree())
                    ax.clabel(cs, inline=True, fontsize=6, fmt="%.1f", colors='gray')
                if sig_mask is not None:
                    skip = 4
                    lons_mesh, lats_mesh = np.meshgrid(diff_map.lon, diff_map.lat)
                    mask_sub = sig_mask[::skip, ::skip]
                    lons_sub = lons_mesh[::skip, ::skip]
                    lats_sub = lats_mesh[::skip, ::skip]
                    ax.scatter(lons_sub[mask_sub], lats_sub[mask_sub], s=1, color='black', alpha=0.5, transform=ccrs.PlateCarree(), zorder=20)
            ax.set_title(title, weight='bold', loc='left')
            return cf

        season_data = [
            ('Summer', summer_composite, summer_model_rps, summer_n_total),
            ('Winter', winter_composite, winter_model_rps, winter_n_total),
        ]

        ref_cf = None
        for col_idx, (season_label, comp, model_rps_dict, n_total_val) in enumerate(season_data):
            if comp is None:
                ax = fig.add_subplot(gs_top[0, col_idx], projection=ccrs.PlateCarree())
                _add_map_features(ax)
                ax.set_title(f"{season_label} Half-Year: No Data", loc='left')
                continue

            hist_clim = comp.get('hist_climatology_mean')
            
            ax = fig.add_subplot(gs_top[0, col_idx], projection=ccrs.PlateCarree())
            diff_map = comp.get('diff_ext_non_future')
            sig_mask = comp.get('sig_mask_ext_non_future')
            title = f"{'(a)' if col_idx==0 else '(b)'} {season_label} Half-Year"
            cf = _plot_diff(ax, diff_map, sig_mask, title, contour_map=hist_clim)
            if cf: ref_cf = cf

        # Colorbar in its own dedicated row
        if ref_cf:
            cax = fig.add_subplot(gs_cbar[0, 0])
            fig.colorbar(ref_cf, cax=cax, orientation='horizontal', label='Difference (m/s)', extend='both')

        def get_unified_limits(keys):
            min_val, max_val = np.inf, -np.inf
            has_data = False
            for key in keys:
                if cmip6_plot_data and cmip6_plot_data.get(key) and cmip6_plot_data[key]['members']:
                    for m in cmip6_plot_data[key]['members']:
                        if m is not None and m.size > 0:
                            vals = m.values if hasattr(m, 'values') else m
                            min_val = min(min_val, np.nanmin(vals))
                            max_val = max(max_val, np.nanmax(vals))
                            has_data = True
                if cmip6_plot_data and cmip6_plot_data.get(key) and cmip6_plot_data[key].get('mmm') is not None:
                    mmm = cmip6_plot_data[key]['mmm']
                    if mmm.size > 0:
                        vals = mmm.values if hasattr(mmm, 'values') else mmm
                        min_val = min(min_val, np.nanmin(vals))
                        max_val = max(max_val, np.nanmax(vals))
                        has_data = True
                if reanalysis_plot_data and reanalysis_plot_data.get(key):
                    for dset in reanalysis_plot_data[key]:
                        data = reanalysis_plot_data[key][dset]
                        if data is not None and data.size > 0:
                            vals = data.values if hasattr(data, 'values') else data
                            min_val = min(min_val, np.nanmin(vals))
                            max_val = max(max_val, np.nanmax(vals))
                            has_data = True
            if not has_data or np.isinf(min_val) or np.isinf(max_val): return None
            range_val = max_val - min_val
            if range_val == 0: range_val = 1.0
            return (min_val - 0.05 * range_val, max_val + 0.05 * range_val)

        lat_ylim = (45, 56.2)
        speed_ylim = (1.5, 9)
        
        plot_configs = [
            {'key': 'Hydro_Summer_JetLat',   'ax': fig.add_subplot(gs_bottom[0, 0]), 'title': '(c) Summer Half-Year\n    Jet Latitude (\u00b0N)', 'ylabel': '', 'ylim': lat_ylim, 'yticks': [45, 50, 55]},
            {'key': 'Hydro_Winter_JetLat',   'ax': fig.add_subplot(gs_bottom[0, 1]), 'title': '(d) Winter Half-Year\n    Jet Latitude (\u00b0N)', 'ylabel': '', 'ylim': lat_ylim, 'yticks': [45, 50, 55]},
            {'key': 'Hydro_Summer_JetSpeed', 'ax': fig.add_subplot(gs_bottom[1, 0]), 'title': '(e) Summer Half-Year\n    Jet Speed (m/s)',    'ylabel': '', 'ylim': speed_ylim, 'yticks': [2, 4, 6, 8]},
            {'key': 'Hydro_Winter_JetSpeed', 'ax': fig.add_subplot(gs_bottom[1, 1]), 'title': '(f) Winter Half-Year\n    Jet Speed (m/s)',    'ylabel': '', 'ylim': speed_ylim, 'yticks': [2, 4, 6, 8]},
        ]

        label_suffix = ""
        global_legend_handles = []
        global_legend_labels = []

        for p_config in plot_configs:
            key = str(p_config['key'])
            ax: Any = p_config['ax']
            if cmip6_plot_data.get(key) and reanalysis_plot_data:
                current_handles = []
                current_labels = []

                if cmip6_plot_data.get(key) and cmip6_plot_data[key]['members']:
                    # Determine extreme/non-extreme for the models
                    model_rps = summer_model_rps if 'Summer' in key else winter_model_rps
                    extreme_models_set = set()
                    non_extreme_models_set = set()
                    if model_rps:
                        sorted_models = sorted(model_rps.items(), key=lambda item: item[1])
                        n_select = config.COMPOSITE_N_MODELS
                        if sorted_models and '_ssp585' in sorted_models[0][0]:
                            n_select = 10
                        if n_select * 2 > len(sorted_models):
                            n_select = len(sorted_models) // 2
                        if n_select < 1: n_select = 1
                            
                        if 'low' in event_key:
                            extreme_models_set = set([m[0] for m in sorted_models[:n_select]])
                            non_extreme_models_set = set([m[0] for m in sorted_models[-n_select:]])
                        else:
                            extreme_models_set = set([m[0] for m in sorted_models[-n_select:]])
                            non_extreme_models_set = set([m[0] for m in sorted_models[:n_select]])

                    extreme_members = []
                    non_extreme_members = []
                    
                    for idx, member_jet in enumerate(cmip6_plot_data[key]['members']):
                        model_name = member_jet.attrs.get('model_key', '')
                        if model_name in extreme_models_set:
                            extreme_members.append(member_jet)
                        elif model_name in non_extreme_models_set:
                            non_extreme_members.append(member_jet)

                    # Plot 100% Model Spread for ALL CMIP6 Models
                    all_members = cmip6_plot_data[key]['members']
                    if all_members:
                        try:
                            aligned_all = xr.align(*all_members, join='outer')
                            stacked_all = xr.concat(aligned_all, dim='model')
                            
                            p2_5 = stacked_all.min(dim='model', skipna=True)
                            p97_5 = stacked_all.max(dim='model', skipna=True)
                            
                            ax.fill_between(p2_5.season_year, p2_5, p97_5, color='grey', alpha=0.2, zorder=1)
                            
                            label = 'CMIP6 100% Model Spread'
                            patch = mpatches.Patch(color='grey', alpha=0.3)
                            current_handles.append(patch)
                            current_labels.append(label)
                        except Exception as e:
                            logging.warning(f"Could not compute 100% model range for {key}: {e}")

                    # Plot Extreme Models Mean
                    # Calculate storyline group mean anomalies directly from individual model anomalies
                    # This avoids the "smearing" effect of calculating a non-linear jet index on averaged wind fields
                    ext_vals = []
                    for member_jet in extreme_members:
                        model_name = member_jet.attrs.get('model_key', '')
                        gwl_yr = gwl_years.get(model_name, {}).get(gwl) if gwl_years is not None else None
                        if gwl_yr is not None and gwl_yr in member_jet.season_year.values:
                            val = float(member_jet.sel(season_year=gwl_yr).values)
                            ext_vals.append(val)
                    ext_val = np.nanmean(ext_vals) if ext_vals else None

                    non_ext_vals = []
                    for member_jet in non_extreme_members:
                        model_name = member_jet.attrs.get('model_key', '')
                        gwl_yr = gwl_years.get(model_name, {}).get(gwl) if gwl_years is not None else None
                        if gwl_yr is not None and gwl_yr in member_jet.season_year.values:
                            val = float(member_jet.sel(season_year=gwl_yr).values)
                            non_ext_vals.append(val)
                    non_ext_val = np.nanmean(non_ext_vals) if non_ext_vals else None

                    # Extreme Models Composite Line
                    if ext_val is not None:
                        try:
                            label = 'High-Freq.' if 'low' in event_key else 'Low-Freq.'
                            # Get crossing range
                            ext_crossing_years = [gwl_years[m][gwl] for m in extreme_models_set if gwl_years is not None and m in gwl_years and gwl in gwl_years[m] and gwl_years[m][gwl]]
                            if ext_crossing_years:
                                t_min, t_max = min(ext_crossing_years), max(ext_crossing_years)
                                ax.hlines(y=ext_val, xmin=t_min, xmax=t_max, color='#b2182b', alpha=0.9, linewidth=4, linestyles='dashed', zorder=6)
                                from matplotlib.lines import Line2D
                                proxy = Line2D([0], [0], color='#b2182b', linewidth=4, alpha=0.9, linestyle='--')
                                current_handles.append(proxy)
                                current_labels.append(label)
                        except Exception as e:
                            logging.warning(f"Could not plot composite line for {key} extreme models: {e}")

                    # Non-Extreme Models Composite Line
                    if non_ext_val is not None:
                        try:
                            label = 'Low-Freq.' if 'low' in event_key else 'High-Freq.'
                            # Get crossing range
                            non_ext_crossing_years = [gwl_years[m][gwl] for m in non_extreme_models_set if gwl_years is not None and m in gwl_years and gwl in gwl_years[m] and gwl_years[m][gwl]]
                            if non_ext_crossing_years:
                                t_min, t_max = min(non_ext_crossing_years), max(non_ext_crossing_years)
                                ax.hlines(y=non_ext_val, xmin=t_min, xmax=t_max, color='#2166ac', alpha=0.9, linewidth=4, linestyles='dashdot', zorder=6)
                                from matplotlib.lines import Line2D
                                proxy = Line2D([0], [0], color='#2166ac', linewidth=4, alpha=0.9, linestyle='-.')
                                current_handles.append(proxy)
                                current_labels.append(label)
                        except Exception as e:
                            logging.warning(f"Could not plot composite line for {key} non-extreme models: {e}")
                
                if cmip6_plot_data.get(key) and cmip6_plot_data[key].get('mmm') is not None:
                    mmm_ts = cmip6_plot_data[key]['mmm']
                    
                    # Calculate trend and p-value on 2015-2100 values to match the subplot extent
                    period_mask = (mmm_ts.season_year >= 2015) & (mmm_ts.season_year <= 2100)
                    valid = (~np.isnan(mmm_ts.values)) & period_mask
                    x_vals = mmm_ts.season_year.values[valid]
                    y_vals = mmm_ts.values[valid]

                    label_suffix = ""
                    if len(x_vals) > 1:
                        slope, intercept, _, p_val, _ = linregress(x_vals, y_vals)
                        trend_line = slope * x_vals + intercept
                        ax.plot(x_vals, trend_line, color='black', linestyle=':', alpha=0.8, linewidth=1.5, zorder=5)
                        
                        unit = "\u00b0N" if 'JetLat' in key else "m/s"
                        trend_per_decade = slope * 10
                        if p_val < 0.01:
                            p_str = "p<0.01"
                        else:
                            p_str = f"p={p_val:.2f}"
                        label_suffix = f" (trend: {trend_per_decade:+.2f}{unit}/dec, {p_str})"

                    line, = ax.plot(mmm_ts.season_year, mmm_ts, color='black', linewidth=2.5, label=f'CMIP6 MMM{label_suffix}', zorder=6)
                    current_handles.append(line)
                    current_labels.append(f'CMIP6 MMM{label_suffix}')

                ax.set_title(p_config['title'], weight='bold', loc='left')
                ax.set_ylabel(p_config['ylabel'])
                ax.grid(True, linestyle=':', alpha=0.6)
                ax.set_xlim(2015, 2100)
                ax.set_xlabel('Year')
                
                # Plot GWL Crossing Range
                if gwl_years:
                    crossing_years = []
                    for model, thresholds in gwl_years.items():
                        if gwl in thresholds and thresholds[gwl] is not None:
                            crossing_years.append(thresholds[gwl])
                    
                    if crossing_years:
                        min_yr = min(crossing_years)
                        max_yr = max(crossing_years)
                        
                        # Add axvspan for the range
                        ax.axvspan(min_yr, max_yr, color='gold', alpha=0.2, zorder=0)
                        
                        gwl_patch = mpatches.Patch(color='gold', alpha=0.3)
                        current_handles.append(gwl_patch)
                        current_labels.append('GWL Crossing Range')

                if p_config.get('ylim'):
                    ax.set_ylim(p_config['ylim'])
                if p_config.get('yticks'):
                    ax.set_yticks(p_config['yticks'])
                
                # Plot isolated trend text natively in the subplot and forward handles
                if label_suffix:
                    clean_suffix = label_suffix.strip(" ()").replace("trend: ", "Trend: ")
                    ax.text(0.96, 0.04, clean_suffix, transform=ax.transAxes, ha='right', va='bottom', 
                            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0.3))
                global_legend_handles = current_handles
                global_legend_labels = current_labels
            else:
                ax.text(0.5, 0.5, "No Timeseries Data", ha='center', va='center')
                ax.set_title(p_config['title'], weight='bold', loc='left')

        scenario_title = Visualizer._format_scenario_title(scenario)
        plt.suptitle(f'Projections of Zonal Wind Differences and Jet Stream Indices\n(GWL {gwl}°C, {scenario_title})', fontsize=16, weight='bold', y=0.99)
        # Add subtitles for the row sections
        fig.text(0.12, 0.93, 'Zonal Wind (u850) Differences (High-Freq. \u2212 Low-Freq.)', ha='left', va='center', fontsize=12, weight='bold')
        fig.text(0.12, 0.61, 'Jet Stream Indices', ha='left', va='center', fontsize=12, weight='bold')

        if global_legend_handles:
            clean_labels = [l.split(' (trend:')[0] if 'MMM' in l else l for l in global_legend_labels]
            fig.legend(global_legend_handles, clean_labels, loc='lower center', bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=False)

        filename_out = f"final_figure_3_{event_key}_{scenario}_gwl{gwl}.png"
        filepath = os.path.join(Config.PLOT_DIR, filename_out)
        plt.savefig(filepath, dpi=600, bbox_inches='tight')
        pdf_filepath = os.path.join(Config.PLOT_DIR, f"final_figure_3_{event_key}_{scenario}_gwl{gwl}.pdf")
        plt.savefig(pdf_filepath, bbox_inches='tight')
        plt.close(fig)
        logging.info(f"Saved final figure 3 to {filepath} and {pdf_filepath}")

    @staticmethod
    def plot_final_figure_6_subseasonal_metrics(panels_data, config, scenario='ssp585', target_gwl=3.0):
        """
        Creates Final Figure 6: Sub-seasonal Metrics (CDD, CDD Duration Spectrum, Dry-Spell Temp, Wet-Day Temp, and PR Intensity Spectrum)
        16-Panel Layout (8x2 Grid):
          Row 0: (a) Summer CDD Trend (2015-2099)                            | (b) Winter CDD Trend (2015-2099)
          Row 1: (c) Summer CDD Difference Map (High - Low @ GWL 3.0°C)     | (d) Winter CDD Difference Map
          Row 2: (e) Summer CDD Duration Distribution (GWL 3.0°C)           | (f) Winter CDD Duration Distribution
          Row 3: (g) Summer Dry-Spell (CDD ≥7d) Temp Trend (2015-2099)     | (h) Winter Dry-Spell Temp Trend
          Row 4: (i) Summer Dry-Spell Temp Difference Map                   | (j) Winter Dry-Spell Temp Difference Map
          Row 5: (k) Summer Wet-Day Temp Trend (2015-2099)                  | (l) Winter Wet-Day Temp Trend (2015-2099)
          Row 6: (m) Summer Wet-Day Temp Difference Map (High - Low)        | (n) Winter Wet-Day Temp Difference Map
          Row 7: (o) Summer Wet-Day PR Intensity Spectrum (GWL 3.0°C)       | (p) Winter Wet-Day PR Intensity Spectrum
        """
        filename_png = "final_figure_6_subseasonal_metrics.png"
        filename_pdf = "final_figure_6_subseasonal_metrics.pdf"
        filepath_png = os.path.join(config.PLOT_DIR, filename_png)
        filepath_pdf = os.path.join(config.PLOT_DIR, filename_pdf)
        
        logging.info(f"Plotting 16-Panel Final Figure 6 to {filepath_png}...")
        Visualizer.ensure_plot_dir_exists()
        
        if not panels_data or 'years' not in panels_data:
            logging.error("Cannot plot Final Figure 6: Missing subseasonal panels data.")
            return

        years = panels_data['years']
        spatial_diffs = panels_data.get('spatial_diffs', {})
        cdd_dur_dist = panels_data.get('cdd_dur_dist', {})
        pr_intensity_dist = panels_data.get('pr_intensity_dist', {})

        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        import cartopy.io.shapereader as shpreader
        import matplotlib.colors as mcolors
        import matplotlib.patches as mpatches
        from matplotlib.cm import ScalarMappable
        import matplotlib.gridspec as gridspec
        import scipy.stats as stats
        import shapely.ops

        fig = plt.figure(figsize=(14, 33))
        gs = gridspec.GridSpec(8, 2, figure=fig, hspace=0.42, wspace=0.20)

        high_color = '#d62728' # Crimson Red
        low_color  = '#1f77b4' # Navy / Steel Blue

        # Helper for time series subplots
        def _plot_ts(ax, key, title, ylabel, unit_str, is_bottom_ts=False, y_text_pos=0.95):
            data = panels_data.get(key, {})
            if not data or data.get('high_mmm') is None or data.get('low_mmm') is None:
                ax.text(0.5, 0.5, "Data Unavailable", ha='center', va='center')
                ax.set_title(title, weight='bold', loc='left', fontsize=11)
                return
                
            h_mmm = data['high_mmm'].values
            h_min = data['high_min'].values
            h_max = data['high_max'].values

            l_mmm = data['low_mmm'].values
            l_min = data['low_min'].values
            l_max = data['low_max'].values

            # 1. Shading 100% Model Spread
            ax.fill_between(years, h_min, h_max, color=high_color, alpha=0.15, linewidth=0, label='High-Freq. Spread')
            ax.fill_between(years, l_min, l_max, color=low_color,  alpha=0.15, linewidth=0, label='Low-Freq. Spread')

            # 2. Lines for MMM
            ax.plot(years, h_mmm, color=high_color, linestyle='--', linewidth=2.0, label='High-Freq. Group (MMM)')
            ax.plot(years, l_mmm, color=low_color,  linestyle='-',  linewidth=2.0, label='Low-Freq. Group (MMM)')

            # 3. Calculate and Plot Linear Regression Trends
            mask_h = np.isfinite(years) & np.isfinite(h_mmm)
            mask_l = np.isfinite(years) & np.isfinite(l_mmm)

            slope_h, inter_h, r_h, p_h, std_h = stats.linregress(years[mask_h], h_mmm[mask_h])
            slope_l, inter_l, r_l, p_l, std_l = stats.linregress(years[mask_l], l_mmm[mask_l])

            trend_h_line = inter_h + slope_h * years
            trend_l_line = inter_l + slope_l * years

            ax.plot(years, trend_h_line, color=high_color, linestyle=':', linewidth=1.5)
            ax.plot(years, trend_l_line, color=low_color,  linestyle=':', linewidth=1.5)

            slope_h_dec = slope_h * 10.0
            slope_l_dec = slope_l * 10.0

            p_h_str = f"{p_h:.3f}" if p_h >= 0.001 else "< 0.001"
            p_l_str = f"{p_l:.3f}" if p_l >= 0.001 else "< 0.001"

            trend_box_text = (
                f"High-Freq. Trend: {slope_h_dec:+.2f} {unit_str}/dec (p = {p_h_str})\n"
                f"Low-Freq. Trend:  {slope_l_dec:+.2f} {unit_str}/dec (p = {p_l_str})"
            )

            ax.text(0.03, y_text_pos, trend_box_text, transform=ax.transAxes,
                    fontsize=9.0, verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85, edgecolor='gray', linewidth=0.5))

            ax.set_title(title, weight='bold', fontsize=11, loc='left')
            ax.set_ylabel(ylabel, fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.set_xlim(2015, 2100)
            if is_bottom_ts:
                ax.set_xlabel('Year', fontsize=10)

        # Helper for duration distribution subplots
        def _plot_dur_dist(ax, season_key, title):
            if not cdd_dur_dist or 'durations' not in cdd_dur_dist:
                ax.text(0.5, 0.5, "Data Unavailable", ha='center', va='center')
                ax.set_title(title, weight='bold', loc='left', fontsize=11)
                return
            
            durs = cdd_dur_dist['durations']
            high_data = cdd_dur_dist.get(f"{season_key}_high", {})
            low_data  = cdd_dur_dist.get(f"{season_key}_low", {})

            if not high_data or not low_data:
                ax.text(0.5, 0.5, "Data Unavailable", ha='center', va='center')
                ax.set_title(title, weight='bold', loc='left', fontsize=11)
                return

            h_mmm = high_data['mmm']
            h_min = high_data['min']
            h_max = high_data['max']

            l_mmm = low_data['mmm']
            l_min = low_data['min']
            l_max = low_data['max']

            ax.fill_between(durs, h_min, h_max, color=high_color, alpha=0.15, linewidth=0, label='High-Freq. Spread')
            ax.fill_between(durs, l_min, l_max, color=low_color,  alpha=0.15, linewidth=0, label='Low-Freq. Spread')

            ax.plot(durs, h_mmm, color=high_color, linestyle='--', linewidth=2.0, marker='o', markersize=3, label='High-Freq. Group (MMM)')
            ax.plot(durs, l_mmm, color=low_color,  linestyle='-',  linewidth=2.0, marker='s', markersize=3, label='Low-Freq. Group (MMM)')

            ax.set_title(title, weight='bold', fontsize=11, loc='left')
            ax.set_xlabel('CDD Duration [days]', fontsize=10)
            ax.set_ylabel('Total Events (31-yr Window)', fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.set_xlim(6.5, 35.5)
            ax.set_xticks([7, 10, 15, 20, 25, 30, 35])
            ax.set_xticklabels(['7', '10', '15', '20', '25', '30', '35+'])

        # Helper for PR intensity distribution subplots
        def _plot_pr_dist(ax, season_key, title):
            if not pr_intensity_dist or 'bin_centers' not in pr_intensity_dist:
                ax.text(0.5, 0.5, "Data Unavailable", ha='center', va='center')
                ax.set_title(title, weight='bold', loc='left', fontsize=11)
                return
            
            centers = pr_intensity_dist['bin_centers']
            high_data = pr_intensity_dist.get(f"{season_key}_high", {})
            low_data  = pr_intensity_dist.get(f"{season_key}_low", {})

            if not high_data or not low_data:
                ax.text(0.5, 0.5, "Data Unavailable", ha='center', va='center')
                ax.set_title(title, weight='bold', loc='left', fontsize=11)
                return

            h_mmm = high_data['mmm']
            h_min = high_data['min']
            h_max = high_data['max']

            l_mmm = low_data['mmm']
            l_min = low_data['min']
            l_max = low_data['max']

            ax.fill_between(centers, h_min, h_max, color=high_color, alpha=0.15, linewidth=0, label='High-Freq. Spread')
            ax.fill_between(centers, l_min, l_max, color=low_color,  alpha=0.15, linewidth=0, label='Low-Freq. Spread')

            ax.plot(centers, h_mmm, color=high_color, linestyle='--', linewidth=2.0, marker='o', markersize=3, label='High-Freq. Group (MMM)')
            ax.plot(centers, l_mmm, color=low_color,  linestyle='-',  linewidth=2.0, marker='s', markersize=3, label='Low-Freq. Group (MMM)')

            ax.set_title(title, weight='bold', fontsize=11, loc='left')
            ax.set_xlabel('Precipitation Intensity [mm/day]', fontsize=10)
            ax.set_ylabel('Total Wet Days (31-yr Window)', fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.set_xlim(1, 25)
            ax.set_xticks(range(1, 26, 4))
            ax.set_xticklabels(['1', '5', '9', '13', '17', '21', '25+'])

        # Helper for spatial difference maps (Catchment-specific polygons matching Danube hydro-units)
        shapefile_path = '/nas/home/vlw/Desktop/STREAM/hydro-units-files/zones.shp'
        shape_crs = ccrs.LambertAzimuthalEqualArea(central_longitude=20.0, central_latitude=55.0, globe=ccrs.Globe(semimajor_axis=6370997.0, semiminor_axis=6370997.0))
        buf = 5.0
        ex = [config.BOX_LON_MIN - buf, config.BOX_LON_MAX + buf, config.BOX_LAT_MIN - buf, config.BOX_LAT_MAX + buf]

        def _plot_spatial_map(ax, diff_array, title, cmap, norm):
            ax.set_extent(ex, crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='black', zorder=5)
            ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='black', zorder=5)
            ax.add_patch(mpatches.Rectangle((config.BOX_LON_MIN, config.BOX_LAT_MIN),
                                           config.BOX_LON_MAX-config.BOX_LON_MIN, config.BOX_LAT_MAX-config.BOX_LAT_MIN,
                                           fill=False, edgecolor='magenta', linewidth=1.5, linestyle='--',
                                           transform=ccrs.PlateCarree(), zorder=10))

            if diff_array is not None and len(diff_array) == 61:
                try:
                    import pyproj
                    from shapely.ops import transform as shapely_transform
                    reader = shpreader.Reader(shapefile_path)
                    geometries = list(reader.geometries())
                    
                    proj_laea = pyproj.CRS.from_proj4('+proj=laea +lat_0=55 +lon_0=20 +x_0=0 +y_0=0 +a=6370997 +b=6370997 +units=m +no_defs')
                    proj_wgs84 = pyproj.CRS.from_epsg(4326)
                    transformer = pyproj.Transformer.from_crs(proj_laea, proj_wgs84, always_xy=True)
                    
                    # 1. Reproject 61 catchment geometries to WGS84 Lat/Lon
                    geoms_wgs84 = [shapely_transform(transformer.transform, g) for g in geometries]
                    
                    # 2. Color each catchment polygon with its zone difference value
                    for g_wgs84, val in zip(geoms_wgs84, diff_array):
                        if np.isfinite(val):
                            color = cmap(norm(val))
                            ax.add_geometries([g_wgs84], crs=ccrs.PlateCarree(), facecolor=color, edgecolor='none', linewidth=0, zorder=4)
                    
                    # 3. Overlay Danube basin boundary outline
                    merged_geom = shapely.ops.unary_union(geometries)
                    ax.add_geometries([merged_geom], crs=shape_crs, edgecolor='black', facecolor='none', linewidth=1.2, zorder=6)
                except Exception as e:
                    logging.warning(f"Could not plot catchment spatial map: {e}")
            
            ax.set_title(title, weight='bold', fontsize=11, loc='left')

        # --- ROW 0: CDD Time Series ---
        ax_a = fig.add_subplot(gs[0, 0])
        _plot_ts(ax_a, 'summer_cdd', '(a) Summer CDD Trend (May–Oct)', 'CDD [days]', 'days', y_text_pos=0.95)

        ax_b = fig.add_subplot(gs[0, 1])
        _plot_ts(ax_b, 'winter_cdd', '(b) Winter CDD Trend (Nov–Apr)', 'CDD [days]', 'days', y_text_pos=0.95)

        # --- ROW 1: CDD Spatial Difference Maps ---
        d_cdd_s = spatial_diffs.get('summer_cdd_diff')
        d_cdd_w = spatial_diffs.get('winter_cdd_diff')

        d_lim_cdd = 6.0
        cmap_cdd = plt.get_cmap('BrBG', 10)
        levs_cdd = np.linspace(-d_lim_cdd, d_lim_cdd, 11)
        norm_cdd = mcolors.BoundaryNorm(levs_cdd, ncolors=cmap_cdd.N, clip=False)

        ax_c = fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree())
        _plot_spatial_map(ax_c, d_cdd_s, '(c) Summer CDD Difference (High - Low)', cmap_cdd, norm_cdd)

        ax_d = fig.add_subplot(gs[1, 1], projection=ccrs.PlateCarree())
        _plot_spatial_map(ax_d, d_cdd_w, '(d) Winter CDD Difference (High - Low)', cmap_cdd, norm_cdd)

        # --- ROW 2: CDD Event Duration Spectrum (GWL 3.0°C Window) ---
        ax_e = fig.add_subplot(gs[2, 0])
        _plot_dur_dist(ax_e, 'summer', '(e) Summer CDD Event Frequency by Duration (GWL 3.0°C)')

        ax_f = fig.add_subplot(gs[2, 1])
        _plot_dur_dist(ax_f, 'winter', '(f) Winter CDD Event Frequency by Duration (GWL 3.0°C)')

        # --- ROW 3: CDD (≥7d) Temperature Time Series ---
        ax_g = fig.add_subplot(gs[3, 0])
        _plot_ts(ax_g, 'summer_cdd_tas', '(g) Summer Dry-Spell (CDD ≥7d) Temp Trend', 'CDD Temp [°C]', '°C', y_text_pos=0.95)

        ax_h = fig.add_subplot(gs[3, 1])
        _plot_ts(ax_h, 'winter_cdd_tas', '(h) Winter Dry-Spell (CDD ≥7d) Temp Trend', 'CDD Temp [°C]', '°C', y_text_pos=0.95)

        # --- ROW 4: CDD (≥7d) Temperature Spatial Difference Maps ---
        d_cdd_tas_s = spatial_diffs.get('summer_cdd_tas_diff')
        d_cdd_tas_w = spatial_diffs.get('winter_cdd_tas_diff')

        d_lim_cdd_tas = 1.5
        cmap_cdd_tas = plt.get_cmap('RdBu_r', 10)
        levs_cdd_tas = np.linspace(-d_lim_cdd_tas, d_lim_cdd_tas, 11)
        norm_cdd_tas = mcolors.BoundaryNorm(levs_cdd_tas, ncolors=cmap_cdd_tas.N, clip=False)

        ax_i = fig.add_subplot(gs[4, 0], projection=ccrs.PlateCarree())
        _plot_spatial_map(ax_i, d_cdd_tas_s, '(i) Summer Dry-Spell Temp Difference', cmap_cdd_tas, norm_cdd_tas)

        ax_j = fig.add_subplot(gs[4, 1], projection=ccrs.PlateCarree())
        _plot_spatial_map(ax_j, d_cdd_tas_w, '(j) Winter Dry-Spell Temp Difference', cmap_cdd_tas, norm_cdd_tas)

        # --- ROW 5: Wet-Day Temperature Time Series ---
        ax_k = fig.add_subplot(gs[5, 0])
        _plot_ts(ax_k, 'summer_wet_tas', '(k) Summer Wet-Day Temp Trend (May–Oct)', 'Wet-Day Temp [°C]', '°C', y_text_pos=0.95)

        ax_l = fig.add_subplot(gs[5, 1])
        _plot_ts(ax_l, 'winter_wet_tas', '(l) Winter Wet-Day Temp Trend (Nov–Apr)', 'Wet-Day Temp [°C]', '°C', y_text_pos=0.95)

        # --- ROW 6: Wet-Day Temperature Spatial Difference Maps ---
        d_wet_s = spatial_diffs.get('summer_wet_tas_diff')
        d_wet_w = spatial_diffs.get('winter_wet_tas_diff')

        d_lim_wet = 1.2
        cmap_wet = plt.get_cmap('RdBu_r', 10)
        levs_wet = np.linspace(-d_lim_wet, d_lim_wet, 11)
        norm_wet = mcolors.BoundaryNorm(levs_wet, ncolors=cmap_wet.N, clip=False)

        ax_m = fig.add_subplot(gs[6, 0], projection=ccrs.PlateCarree())
        _plot_spatial_map(ax_m, d_wet_s, '(m) Summer Wet-Day Temp Difference', cmap_wet, norm_wet)

        ax_n = fig.add_subplot(gs[6, 1], projection=ccrs.PlateCarree())
        _plot_spatial_map(ax_n, d_wet_w, '(n) Winter Wet-Day Temp Difference', cmap_wet, norm_wet)

        # --- ROW 7: Wet-Day Precipitation Intensity Spectrum (GWL 3.0°C Window) ---
        ax_o = fig.add_subplot(gs[7, 0])
        _plot_pr_dist(ax_o, 'summer', '(o) Summer Wet-Day PR Intensity Spectrum (GWL 3.0°C)')

        ax_p = fig.add_subplot(gs[7, 1])
        _plot_pr_dist(ax_p, 'winter', '(p) Winter Wet-Day PR Intensity Spectrum (GWL 3.0°C)')

        # Shared Legend for Time Series (High-Freq vs Low-Freq)
        handles, labels = ax_a.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.005), ncol=4, frameon=True, fontsize=10)

        # Main Title
        scenario_str = Visualizer._format_scenario_title(scenario)
        plt.suptitle(f'Sub-seasonal Drivers (CDD, CDD Duration, Wet-Day Temp & PR Intensity) Comparison: High- vs. Low-Frequency Model Groups\n(Danube Catchment, {scenario_str}, GWL {target_gwl:.1f}°C)',
                     fontsize=14, weight='bold', y=0.993)

        fig.tight_layout(rect=(0, 0.02, 1, 0.98), h_pad=1.4, w_pad=1.0)

        # Add horizontal colorbars below Row 1 (CDD Maps), Row 4 (CDD Temp Maps), and Row 6 (Wet-Day Temp Maps)
        pos_c = ax_c.get_position()
        cb_ax_cdd = fig.add_axes((0.18, pos_c.y0 - 0.015, 0.64, 0.006))
        fig.colorbar(ScalarMappable(norm=norm_cdd, cmap=cmap_cdd), cax=cb_ax_cdd, orientation='horizontal',
                     label='CDD Difference [days] (High-Freq. minus Low-Freq.)', extend='both')

        pos_i = ax_i.get_position()
        cb_ax_cdd_tas = fig.add_axes((0.18, pos_i.y0 - 0.015, 0.64, 0.006))
        fig.colorbar(ScalarMappable(norm=norm_cdd_tas, cmap=cmap_cdd_tas), cax=cb_ax_cdd_tas, orientation='horizontal',
                     label='Dry-Spell (CDD ≥7d) Temp. Difference [°C] (High-Freq. minus Low-Freq.)', extend='both')

        pos_m = ax_m.get_position()
        cb_ax_wet = fig.add_axes((0.18, pos_m.y0 - 0.015, 0.64, 0.006))
        fig.colorbar(ScalarMappable(norm=norm_wet, cmap=cmap_wet), cax=cb_ax_wet, orientation='horizontal',
                     label='Wet-Day Temp. Difference [°C] (High-Freq. minus Low-Freq.)', extend='both')

        # Add section titles above each row
        nudge = 0.008
        fig.text(ax_a.get_position().x0, ax_a.get_position().y1 + nudge, "Danube Basin Consecutive Dry Days (CDD) Trend (2015–2099)", ha='left', va='bottom', fontsize=11, weight='bold')
        fig.text(ax_c.get_position().x0, ax_c.get_position().y1 + nudge, f"CDD Spatial Difference at GWL {target_gwl:.1f}°C (High-Freq. minus Low-Freq.)", ha='left', va='bottom', fontsize=11, weight='bold')
        fig.text(ax_e.get_position().x0, ax_e.get_position().y1 + nudge, f"CDD Event Duration Spectrum at GWL {target_gwl:.1f}°C (High-Freq. vs Low-Freq.)", ha='left', va='bottom', fontsize=11, weight='bold')
        fig.text(ax_g.get_position().x0, ax_g.get_position().y1 + nudge, "Danube Basin Dry-Spell (CDD ≥7d) Temperature Trend (2015–2099)", ha='left', va='bottom', fontsize=11, weight='bold')
        fig.text(ax_i.get_position().x0, ax_i.get_position().y1 + nudge, f"Dry-Spell (CDD ≥7d) Temp Spatial Difference at GWL {target_gwl:.1f}°C (High-Freq. minus Low-Freq.)", ha='left', va='bottom', fontsize=11, weight='bold')
        fig.text(ax_k.get_position().x0, ax_k.get_position().y1 + nudge, "Danube Basin Wet-Day Temperature Trend (2015–2099)", ha='left', va='bottom', fontsize=11, weight='bold')
        fig.text(ax_m.get_position().x0, ax_m.get_position().y1 + nudge, f"Wet-Day Temperature Spatial Difference at GWL {target_gwl:.1f}°C (High-Freq. minus Low-Freq.)", ha='left', va='bottom', fontsize=11, weight='bold')
        fig.text(ax_o.get_position().x0, ax_o.get_position().y1 + nudge, f"Wet-Day Precipitation Intensity Spectrum at GWL {target_gwl:.1f}°C (High-Freq. vs Low-Freq.)", ha='left', va='bottom', fontsize=11, weight='bold')

        plt.savefig(filepath_png, dpi=300, bbox_inches='tight')
        plt.savefig(filepath_pdf, bbox_inches='tight')
        plt.close(fig)
        logging.info(f"Successfully generated 16-Panel Final Figure 6: {filepath_png} and {filepath_pdf}")

    @staticmethod
    def plot_final_figure_7_summer_winter_lowflow_clustering(cmip6_results, config, scenario='ssp585', target_gwl=3.0, return_period_results=None):
        """
        Creates Final Figure 7: Timeline Analysis of Summer (Red) vs. Winter (Blue) 30Q10 Low-Flow Events
        across a 31-Year Window around GWL (+3.0°C) per CMIP6 Model, presented in 2 Subplots:
          Subplot 1 (Top): Grouped by Summer Model Group Assignment (JJA 30Q10_low Extreme vs Non-Extreme)
          Subplot 2 (Bottom): Grouped by Winter Model Group Assignment (DJF 30Q10_low Extreme vs Non-Extreme)
        """
        filename_png = "final_figure_7_summer_winter_lowflow_clustering.png"
        filename_pdf = "final_figure_7_summer_winter_lowflow_clustering.pdf"
        filepath_png = os.path.join(config.PLOT_DIR, filename_png)
        filepath_pdf = os.path.join(config.PLOT_DIR, filename_pdf)
        
        logging.info(f"Plotting 2-Subplot Final Figure 7 to {filepath_png}...")
        Visualizer.ensure_plot_dir_exists()
        
        if not cmip6_results:
            logging.error("Cannot plot Final Figure 7: Missing cmip6_results.")
            return

        metric_timeseries = cmip6_results.get('model_metric_timeseries', {})
        gwl_years_dict = cmip6_results.get('gwl_threshold_years', cmip6_results.get('gwl_years', {}))

        if not metric_timeseries:
            logging.error("Cannot plot Final Figure 7: Missing model_metric_timeseries.")
            return

        # 1. Extract Summer & Winter Extreme/Non-Extreme model lists matching Figure 2
        ext_w, non_w, ext_s, non_s = [], [], [], []

        if return_period_results and 'data' in return_period_results and target_gwl in return_period_results['data']:
            try:
                gwl_node = return_period_results['data'][target_gwl]
                ext_w = gwl_node.get('winter', {}).get('Extreme Models', {}).get('30Q10_low', {}).get('future_keys_all_models', [])
                non_w = gwl_node.get('winter', {}).get('Non-Extreme Models', {}).get('30Q10_low', {}).get('future_keys_all_models', [])
                ext_s = gwl_node.get('summer', {}).get('Extreme Models', {}).get('30Q10_low', {}).get('future_keys_all_models', [])
                non_s = gwl_node.get('summer', {}).get('Non-Extreme Models', {}).get('30Q10_low', {}).get('future_keys_all_models', [])
            except Exception:
                pass

        if not ext_w or not non_w or not ext_s or not non_s:
            try:
                from storyline import StorylineAnalyzer
                analyzer = StorylineAnalyzer(config)
                ext_w_calc, non_w_calc, _ = analyzer.get_composite_extreme_models(cmip6_results, target_gwl, '30Q10_low', 'Winter')
                ext_s_calc, non_s_calc, _ = analyzer.get_composite_extreme_models(cmip6_results, target_gwl, '30Q10_low', 'Summer')
                if ext_w_calc: ext_w = ext_w_calc
                if non_w_calc: non_w = non_w_calc
                if ext_s_calc: ext_s = ext_s_calc
                if non_s_calc: non_s = non_s_calc
            except Exception as e:
                logging.warning(f"Could not calculate composite extreme models: {e}")

        def is_match(m_key, clean_name, target_list):
            if not target_list: return False
            return (m_key in target_list) or (clean_name in target_list) or any(
                m_key.startswith(x) or x.startswith(clean_name) or clean_name.startswith(x.split('_')[0])
                for x in target_list
            )

        # Identify all model keys for the requested scenario
        model_keys = sorted([k for k in metric_timeseries.keys() if k.endswith(scenario)])
        if not model_keys:
            model_keys = sorted(list(metric_timeseries.keys()))

        base_model_records = []

        for m_key in model_keys:
            ts_dict = metric_timeseries.get(m_key, {})
            ts_summer = ts_dict.get('30Q_low_summer')
            ts_winter = ts_dict.get('30Q_low_winter')
            ts_annual = ts_dict.get('30Q_low_full_year')

            if ts_summer is None or ts_winter is None or ts_annual is None:
                continue

            clean_name = m_key.replace(f"_{scenario}", "")

            # Historical 30Q10 threshold (1960 - 2014)
            try:
                hist_slice = ts_annual.sel(year=slice(1960, 2014))
                if hist_slice.year.size < 10:
                    hist_slice = ts_annual.where(ts_annual.year < 2015, drop=True)
            except Exception:
                hist_slice = ts_annual.where(ts_annual.year < 2015, drop=True)

            hist_vals = hist_slice.values
            hist_vals = hist_vals[np.isfinite(hist_vals)]
            if len(hist_vals) < 10:
                continue

            thresh_30q10 = np.quantile(hist_vals, 0.10)

            # Determine GWL year
            gwl_yr = None
            m_info = gwl_years_dict.get(m_key) or gwl_years_dict.get(clean_name)
            if isinstance(m_info, dict):
                gwl_yr = m_info.get(target_gwl)
            elif isinstance(m_info, (int, float, np.integer)):
                gwl_yr = m_info

            if gwl_yr is None or not np.isfinite(gwl_yr):
                continue

            gwl_yr = int(gwl_yr)
            window_half = config.GWL_YEARS_WINDOW // 2 if hasattr(config, 'GWL_YEARS_WINDOW') else 15
            rel_years = np.arange(-window_half, window_half + 1)
            cal_years = gwl_yr + rel_years

            summer_rel = []
            winter_rel = []
            same_year_cooccur_rel = []
            preceded_winter_rel = []

            for ry, cy in zip(rel_years, cal_years):
                is_s = False
                is_w = False

                if cy in ts_summer.year.values:
                    v_s = ts_summer.sel(year=cy).item()
                    if np.isfinite(v_s) and v_s < thresh_30q10:
                        is_s = True

                if cy in ts_winter.year.values:
                    v_w = ts_winter.sel(year=cy).item()
                    if np.isfinite(v_w) and v_w < thresh_30q10:
                        is_w = True

                if is_s:
                    summer_rel.append(ry)
                if is_w:
                    winter_rel.append(ry)
                if is_s and is_w:
                    same_year_cooccur_rel.append(ry)
                    preceded_winter_rel.append(ry)

            # Classifications
            is_high_s = is_match(m_key, clean_name, ext_s)
            is_low_s  = is_match(m_key, clean_name, non_s)
            group_s   = 'High-Freq.' if is_high_s else ('Low-Freq.' if is_low_s else 'Other')

            is_high_w = is_match(m_key, clean_name, ext_w)
            is_low_w  = is_match(m_key, clean_name, non_w)
            group_w   = 'High-Freq.' if is_high_w else ('Low-Freq.' if is_low_w else 'Other')

            base_model_records.append({
                'key': m_key,
                'name': clean_name,
                'group_s': group_s,
                'group_w': group_w,
                'gwl_year': gwl_yr,
                'thresh_30q10': thresh_30q10,
                'rel_years': rel_years,
                'summer_rel': summer_rel,
                'winter_rel': winter_rel,
                'same_year_cooccur_rel': same_year_cooccur_rel,
                'preceded_winter_rel': preceded_winter_rel,
                'n_summer': len(summer_rel),
                'n_winter': len(winter_rel),
                'n_preceded_winter': len(preceded_winter_rel)
            })

        if not base_model_records:
            logging.error("Cannot plot Final Figure 7: No valid model records found.")
            return

        import matplotlib.gridspec as gridspec
        import matplotlib.patches as mpatches
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors

        n_models = len(base_model_records)

        # Set up 2-row layout (Row 1: Summer Grouping, Row 2: Winter Grouping)
        fig = plt.figure(figsize=(17, max(18, 0.46 * n_models * 2 + 4.5)))
        gs_outer = gridspec.GridSpec(2, 1, figure=fig, hspace=0.20)

        red_color = '#d62728'   # Summer
        blue_color = '#1f77b4'  # Winter
        gold_color = '#ff7f0e'  # Co-occurrence highlight

        high_grp_color = '#b2182b' # Dark Crimson for High-Freq label
        low_grp_color  = '#2166ac' # Dark Blue for Low-Freq label
        mod_grp_color  = '#4d4d4d' # Charcoal for Moderate label

        # Helper function to plot a single subplot section (Row)
        def _plot_section(gs_spec, season_mode, section_title, panel_letters):
            grp_key = 'group_s' if season_mode == 'summer' else 'group_w'
            
            high_group = [r for r in base_model_records if r[grp_key] == 'High-Freq.']
            low_group  = [r for r in base_model_records if r[grp_key] == 'Low-Freq.']
            mod_group  = [r for r in base_model_records if r[grp_key] == 'Other']

            high_group.sort(key=lambda x: (x['n_winter'], x['n_summer']), reverse=True)
            low_group.sort(key=lambda x: (x['n_winter'], x['n_summer']), reverse=True)
            mod_group.sort(key=lambda x: (x['n_winter'], x['n_summer']), reverse=True)

            ordered_records = []
            if high_group: ordered_records.extend(high_group)
            if mod_group: ordered_records.extend(mod_group)
            if low_group: ordered_records.extend(low_group)

            display_records = list(reversed(ordered_records))
            n_display = len(display_records)

            gs_inner = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_spec, width_ratios=[3.8, 1.2], wspace=0.15)
            ax_main = fig.add_subplot(gs_inner[0, 0])
            ax_stat = fig.add_subplot(gs_inner[0, 1], sharey=ax_main)

            y_labels = []
            y_label_colors = []

            for i, rec in enumerate(display_records):
                y_pos = i
                grp = rec[grp_key]

                # Reference baseline dotted line
                ax_main.plot([-15.5, 15.5], [y_pos, y_pos], color='gray', linestyle=':', linewidth=0.5, zorder=2, alpha=0.4)

                s_set = set(rec['summer_rel'])
                w_set = set(rec['winter_rel'])
                co_set = set(rec['same_year_cooccur_rel'])

                for ry in rec['rel_years']:
                    in_s = ry in s_set
                    in_w = ry in w_set
                    in_co = ry in co_set

                    if in_co:
                        ax_main.plot([ry - 0.18, ry + 0.18], [y_pos, y_pos], color=gold_color, linewidth=5.0, alpha=0.95, zorder=3)
                        ax_main.scatter(ry - 0.18, y_pos, color=red_color, s=65, marker='o', zorder=4, edgecolor='black', linewidth=0.8)
                        ax_main.scatter(ry + 0.18, y_pos, color=blue_color, s=65, marker='o', zorder=4, edgecolor='black', linewidth=0.8)
                    else:
                        if in_s:
                            ax_main.scatter(ry - 0.12, y_pos, color=red_color, s=55, marker='o', zorder=4, edgecolor='black', linewidth=0.7)
                        if in_w:
                            ax_main.scatter(ry + 0.12, y_pos, color=blue_color, s=55, marker='o', zorder=4, edgecolor='black', linewidth=0.7)

                grp_tag = "High-Freq." if grp == "High-Freq." else ("Low-Freq." if grp == "Low-Freq." else "Other")
                y_labels.append(f"{rec['name']} ({rec['gwl_year']})  [{grp_tag}]")
                
                if grp == "High-Freq.":
                    y_label_colors.append(high_grp_color)
                elif grp == "Low-Freq.":
                    y_label_colors.append(low_grp_color)
                else:
                    y_label_colors.append(mod_grp_color)

            # Division lines
            for idx in range(1, n_display):
                if display_records[idx][grp_key] != display_records[idx-1][grp_key]:
                    sep_y = idx - 0.5
                    ax_main.axhline(sep_y, color='black', linestyle='--', linewidth=1.2, zorder=5)
                    ax_stat.axhline(sep_y, color='black', linestyle='--', linewidth=1.2, zorder=5)

            ax_main.axvline(0, color='gray', linestyle='--', linewidth=1.5, zorder=5)

            ax_main.set_xlim(-16, 16)
            ax_main.set_xticks(np.arange(-15, 16, 5))
            ax_main.set_xticklabels([f"{x:+d}" if x != 0 else "0 (GWL)" for x in np.arange(-15, 16, 5)], fontsize=10, weight='bold')
            ax_main.set_xlabel("Years Relative to GWL +3.0°C Central Year", fontsize=10.5, weight='bold')
            
            ax_main.set_yticks(np.arange(n_display))
            ax_main.set_yticklabels(y_labels, fontsize=8.5)
            
            for tick_label, color in zip(ax_main.get_yticklabels(), y_label_colors):
                tick_label.set_color(color)
                tick_label.set_fontweight('bold')

            ax_main.set_ylim(-0.8, n_display - 0.2)
            ax_main.grid(True, axis='x', linestyle=':', alpha=0.6)
            ax_main.set_title(f"({panel_letters[0]}) {section_title} - 31-Year Timelines", fontsize=11, weight='bold', loc='left')

            # Bar Chart
            y_positions = np.arange(n_display)
            total_w = [r['n_winter'] for r in display_records]
            preceded_w = [r['n_preceded_winter'] for r in display_records]
            independent_w = [tw - pw for tw, pw in zip(total_w, preceded_w)]

            ax_stat.barh(y_positions, preceded_w, color=gold_color, edgecolor='darkorange', height=0.55, zorder=3)
            ax_stat.barh(y_positions, independent_w, left=preceded_w, color=blue_color, edgecolor='darkblue', height=0.55, zorder=3)

            # Annotate each model bar with percentage
            max_w = max(total_w) if total_w else 1
            model_pcts = []
            for i, (tw, pw) in enumerate(zip(total_w, preceded_w)):
                if tw > 0:
                    pct_val = (pw / tw) * 100
                    model_pcts.append(pct_val)
                    txt = f"{pct_val:.0f}%"
                    ax_stat.text(tw + 0.15, i, txt, va='center', ha='left', fontsize=8.0, weight='bold', color='#d95f02' if pw > 0 else '#555555')
                else:
                    ax_stat.text(0.15, i, "0%", va='center', ha='left', fontsize=8.0, color='gray')

            sec_tot = sum(total_w)
            sec_pre = sum(preceded_w)
            sec_pct = (sec_pre / sec_tot * 100) if sec_tot > 0 else 0
            mean_model_pct = np.mean(model_pcts) if model_pcts else 0

            # Section Group Stats
            h_recs = [r for r in display_records if r[grp_key] == 'High-Freq.']
            l_recs = [r for r in display_records if r[grp_key] == 'Low-Freq.']

            h_w_tot, h_w_pre = sum(r['n_winter'] for r in h_recs), sum(r['n_preceded_winter'] for r in h_recs)
            h_pct = (h_w_pre / h_w_tot * 100) if h_w_tot > 0 else 0

            l_w_tot, l_w_pre = sum(r['n_winter'] for r in l_recs), sum(r['n_preceded_winter'] for r in l_recs)
            l_pct = (l_w_pre / l_w_tot * 100) if l_w_tot > 0 else 0

            ax_stat.set_xlabel("Winter 30Q10 Event Count", fontsize=10.5, weight='bold')
            ax_stat.set_title(f"({panel_letters[1]}) Driver Breakdown (High-Freq: {h_pct:.1f}%  |  Low-Freq: {l_pct:.1f}%)", fontsize=10.0, weight='bold', loc='left')
            ax_stat.grid(True, axis='x', linestyle=':', alpha=0.6)
            ax_stat.set_xlim(0, max(max_w + 2.2, 5.0))
            ax_stat.set_ylim(-0.8, n_display - 0.2)
            plt.setp(ax_stat.get_yticklabels(), visible=False)

            return h_w_pre, h_w_tot, h_pct, l_w_pre, l_w_tot, l_pct, sec_pre, sec_tot, sec_pct, mean_model_pct

        # Plot Subplot 1 (Top): Summer Grouping
        h_pre_s, h_tot_s, h_pct_s, l_pre_s, l_tot_s, l_pct_s, sec_pre_s, sec_tot_s, sec_pct_s, mean_m_pct_s = _plot_section(
            gs_outer[0], 'summer', 'Subplot 1: Grouped by Summer Model Group Assignment (JJA 30Q10)', ['a1', 'a2']
        )

        # Plot Subplot 2 (Bottom): Winter Grouping
        h_pre_w, h_tot_w, h_pct_w, l_pre_w, l_tot_w, l_pct_w, sec_pre_w, sec_tot_w, sec_pct_w, mean_m_pct_w = _plot_section(
            gs_outer[1], 'winter', 'Subplot 2: Grouped by Winter Model Group Assignment (DJF 30Q10)', ['b1', 'b2']
        )

        # Shared Legend Handles
        leg_red = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=red_color, markeredgecolor='black', markersize=8.5, label='Summer 30Q10 Event')
        leg_blue = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=blue_color, markeredgecolor='black', markersize=8.5, label='Winter 30Q10 Event')
        leg_gold = mpatches.Patch(color=gold_color, label='Co-occurring / Summer-Preceded Event')
        leg_high = mpatches.Patch(color=high_grp_color, label='High-Freq. Model Group')
        leg_low = mpatches.Patch(color=low_grp_color, label='Low-Freq. Model Group')

        fig.legend(handles=[leg_red, leg_blue, leg_gold, leg_high, leg_low], loc='lower center', bbox_to_anchor=(0.5, 0.005), ncol=5, frameon=True, fontsize=9.5)

        # Overall Title
        scenario_str = Visualizer._format_scenario_title(scenario)
        plt.suptitle(f"Final Figure 7: Co-occurrence of Summer & Winter 30Q10 Low-Flow Events @ GWL +{target_gwl:.1f}°C ({scenario_str})\n"
                     f"Summer Grouping: Overall Mean {sec_pct_s:.1f}% Winter Events Summer-Preceded (High-Freq {h_pct_s:.1f}%, Low-Freq {l_pct_s:.1f}%)  |  "
                     f"Winter Grouping: Overall Mean {sec_pct_w:.1f}% Winter Events Summer-Preceded (High-Freq {h_pct_w:.1f}%, Low-Freq {l_pct_w:.1f}%)",
                     fontsize=12.0, weight='bold', y=0.99)

        fig.tight_layout(rect=(0, 0.04, 1, 0.97))

        plt.savefig(filepath_png, dpi=300, bbox_inches='tight')
        plt.savefig(filepath_pdf, bbox_inches='tight')

        plt.close(fig)
        logging.info(f"Successfully generated 2-Subplot Final Figure 7: {filepath_png} and {filepath_pdf}")

    @staticmethod
    def plot_final_figure_8_winter_lagged_correlation(cmip6_results, config, scenario='ssp585', target_gwl=3.0, return_period_results=None):
        """
        Creates Final Figure 8: Lagged Correlation Analysis (Lags 1-6 Months) for Winter & Summer Streamflow
        vs. Preceding Precipitation (P) & Temperature (T), comparing High-Frequency (Extreme) vs Low-Frequency (Non-Extreme)
        Storyline Model Clusters across all subplots.
        """
        filename_png = "final_figure_8_winter_pr_discharge_lagged_correlation.png"
        filename_pdf = "final_figure_8_winter_pr_discharge_lagged_correlation.pdf"
        filepath_png = os.path.join(config.PLOT_DIR, filename_png)
        filepath_pdf = os.path.join(config.PLOT_DIR, filename_pdf)

        logging.info(f"Plotting 4-Panel Final Figure 8 to {filepath_png}...")
        Visualizer.ensure_plot_dir_exists()

        import pandas as pd
        import glob
        from scipy.stats import spearmanr

        if not cmip6_results:
            logging.error("Cannot plot Final Figure 8: Missing cmip6_results.")
            return

        metric_timeseries = cmip6_results.get('model_metric_timeseries', {})
        if not metric_timeseries:
            logging.error("Cannot plot Final Figure 8: Missing model_metric_timeseries.")
            return

        # 1. Identify Model Clusters (Extreme vs Non-Extreme for Winter & Summer)
        ext_w, non_w, ext_s, non_s = [], [], [], []
        if return_period_results and 'data' in return_period_results and target_gwl in return_period_results['data']:
            try:
                gwl_node = return_period_results['data'][target_gwl]
                ext_w = gwl_node.get('winter', {}).get('Extreme Models', {}).get('30Q10_low', {}).get('future_keys_all_models', [])
                non_w = gwl_node.get('winter', {}).get('Non-Extreme Models', {}).get('30Q10_low', {}).get('future_keys_all_models', [])
                ext_s = gwl_node.get('summer', {}).get('Extreme Models', {}).get('30Q10_low', {}).get('future_keys_all_models', [])
                non_s = gwl_node.get('summer', {}).get('Non-Extreme Models', {}).get('30Q10_low', {}).get('future_keys_all_models', [])
            except Exception:
                pass

        storyline_classification_2d = cmip6_results.get('storyline_classification_2d', {}) if cmip6_results else {}
        if target_gwl in storyline_classification_2d:
            if not ext_w: ext_w = storyline_classification_2d[target_gwl].get('DJF_Extreme Models', storyline_classification_2d[target_gwl].get('winter_Extreme Models', []))
            if not non_w: non_w = storyline_classification_2d[target_gwl].get('DJF_Non-Extreme Models', storyline_classification_2d[target_gwl].get('winter_Non-Extreme Models', []))
            if not ext_s: ext_s = storyline_classification_2d[target_gwl].get('JJA_Extreme Models', storyline_classification_2d[target_gwl].get('summer_Extreme Models', []))
            if not non_s: non_s = storyline_classification_2d[target_gwl].get('JJA_Non-Extreme Models', storyline_classification_2d[target_gwl].get('summer_Non-Extreme Models', []))

        if not ext_w or not non_w or not ext_s or not non_s:
            try:
                from storyline import StorylineAnalyzer
                analyzer = StorylineAnalyzer(config)
                ext_w_calc, non_w_calc, _ = analyzer.get_composite_extreme_models(cmip6_results, target_gwl, '30Q10_low', 'Winter')
                ext_s_calc, non_s_calc, _ = analyzer.get_composite_extreme_models(cmip6_results, target_gwl, '30Q10_low', 'Summer')
                if not ext_w and ext_w_calc: ext_w = ext_w_calc
                if not non_w and non_w_calc: non_w = non_w_calc
                if not ext_s and ext_s_calc: ext_s = ext_s_calc
                if not non_s and non_s_calc: non_s = non_s_calc
            except Exception as e:
                logging.warning(f"Could not calculate composite extreme models for Fig 8: {e}")

        def is_in_list(m_key, clean_name, target_list):
            if not target_list: return False
            import re
            def clean_model_id(s):
                if not s: return ''
                s = str(s).strip()
                for scn in ['ssp585', 'ssp245', 'ssp126', 'historical']:
                    if s.endswith(f'_{scn}'):
                        s = s[:-len(scn)-1]
                s = re.sub(r'_r\d+i\d+p\d+f\d+$', '', s)
                return s.strip()

            c_key = clean_model_id(m_key)
            c_name = clean_model_id(clean_name)
            raw_targets = set(target_list)
            clean_t = {clean_model_id(x) for x in target_list}
            return (m_key in raw_targets) or (clean_name in raw_targets) or (c_key in clean_t) or (c_name in clean_t)

        # Prepare Fallback Data Loaders if pr_box_full or discharge_monthly_full are missing in metric_timeseries
        discharge_filepath = getattr(config, f"DISCHARGE_{scenario.upper()}_FILE", None)
        if not discharge_filepath or not os.path.exists(discharge_filepath):
            discharge_filepath = os.path.join(config.DATA_BASE_PATH, f"CP65_{'8.5' if scenario=='ssp585' else '4.5'}-Tabelle_1.csv")

        df_q_raw = None
        if os.path.exists(discharge_filepath):
            try:
                df_q_raw = pd.read_csv(discharge_filepath, sep=';', decimal=',', na_values=['-0,01'])
                date_col = df_q_raw.columns[0]
                df_q_raw = df_q_raw.rename(columns={date_col: 'date'})
                df_q_raw['time'] = pd.to_datetime(df_q_raw['date'])
                df_q_raw['year'] = df_q_raw['time'].dt.year
                df_q_raw['month'] = df_q_raw['time'].dt.month
                df_q_raw['day'] = df_q_raw['time'].dt.day
                for col in df_q_raw.columns:
                    if col not in ['date', 'time', 'year', 'month', 'day']:
                        df_q_raw[col] = pd.to_numeric(df_q_raw[col], errors='coerce')
            except Exception as e:
                logging.warning(f"Failed to load discharge fallback CSV in Fig 8: {e}")

        catchment_dirs = [
            '/nas/home/vlw/Desktop/STREAM/final-bias-adjusted-data',
            '/nas/home/vlw/Desktop/STREAM/copernicus-final-adjusted-data',
            '/nas/home/vlw/Desktop/STREAM/in-catchment-data',
            '/nas/home/vlw/Desktop/STREAM/copernicus-in-catchment'
        ]

        def find_catchment_files(model_name, scn):
            for c_dir in catchment_dirs:
                ba_pr1 = os.path.join(c_dir, f"MONTHLY_*_{model_name}_*pr_{scn}_count-*.csv")
                ba_pr2 = os.path.join(c_dir, f"MONTHLY_*_{model_name}_pr_{scn}_count-*.csv")
                f_pr_ba = sorted(glob.glob(ba_pr1) + glob.glob(ba_pr2))
                
                ba_tas1 = os.path.join(c_dir, f"MONTHLY_*_{model_name}_*tas_{scn}_count-*.csv")
                ba_tas2 = os.path.join(c_dir, f"MONTHLY_*_{model_name}_tas_{scn}_count-*.csv")
                f_tas_ba = sorted(glob.glob(ba_tas1) + glob.glob(ba_tas2))
                
                if f_pr_ba and f_tas_ba:
                    return f_pr_ba[0], f_tas_ba[0]
                
                p1_pr = os.path.join(c_dir, f"{model_name}_pr_{scn}_*_in-catchment-units.csv")
                p2_pr = os.path.join(c_dir, f"{model_name}_*_pr_{scn}_*_in-catchment-units.csv")
                f_pr = sorted(glob.glob(p1_pr) + glob.glob(p2_pr))
                
                p1_tas = os.path.join(c_dir, f"{model_name}_tas_{scn}_*_in-catchment-units.csv")
                p2_tas = os.path.join(c_dir, f"{model_name}_*_tas_{scn}_*_in-catchment-units.csv")
                f_tas = sorted(glob.glob(p1_tas) + glob.glob(p2_tas))
                
                if f_pr and f_tas:
                    return f_pr[0], f_tas[0]
            return None, None

        model_keys = sorted([k for k in metric_timeseries.keys() if k.endswith(scenario)])
        if not model_keys and df_q_raw is not None:
            model_keys = [c for c in df_q_raw.columns if c not in ['date', 'time', 'year', 'month', 'day', 'QOBS', 'QSIM']]
        elif not model_keys:
            model_keys = sorted(list(metric_timeseries.keys()))

        gwl_years_dict = cmip6_results.get('gwl_threshold_years', cmip6_results.get('gwl_years', {}))

        model_results = []

        def standardize_monthly_da(da, name='val'):
            if da is None:
                return None
            try:
                years = da.time.dt.year.values
                months = da.time.dt.month.values
                new_times = pd.to_datetime([f"{int(y):04d}-{int(m):02d}-01" for y, m in zip(years, months)])
                da_new = da.copy(deep=False)
                da_new['time'] = new_times
                da_new.name = name
                return da_new
            except Exception as e:
                logging.warning(f"Error standardizing monthly time coordinate: {e}")
                return da

        for m_key in model_keys:
            ts_dict = metric_timeseries.get(m_key, {})
            clean_name = m_key.replace(f"_{scenario}", "")

            q_full = standardize_monthly_da(ts_dict.get('discharge_monthly_full'), 'discharge')

            # Fallback calculation if key missing in ts_dict
            if q_full is None and df_q_raw is not None and clean_name in df_q_raw.columns:
                try:
                    df_qm = df_q_raw[['time', clean_name]].dropna()
                    if not df_qm.empty:
                        s_q = df_qm.set_index('time')[clean_name].resample('MS').mean()
                        da_q = s_q.to_xarray()
                        q_full = standardize_monthly_da(da_q, 'discharge')
                except Exception as e:
                    logging.warning(f"Error computing fallback discharge for {clean_name}: {e}")

            # Prioritize loading bias-adjusted catchment data for PR and TAS
            pr_full, tas_full = None, None
            pr_file, tas_file = find_catchment_files(clean_name, scenario)
            if pr_file and tas_file:
                try:
                    df_pr = pd.read_csv(pr_file, sep='\t', skiprows=1)
                    df_tas = pd.read_csv(tas_file, sep='\t', skiprows=1)
                    p_cols = [c for c in df_pr.columns if c.startswith('P_')]
                    t_cols = [c for c in df_tas.columns if c.startswith('T_')]
                    df_pr['pr_mean'] = df_pr[p_cols].mean(axis=1)
                    df_tas['tas_mean'] = df_tas[t_cols].mean(axis=1)
                    
                    df_pt = pd.merge(df_pr[['year', 'month', 'day', 'pr_mean']], df_tas[['year', 'month', 'day', 'tas_mean']], on=['year', 'month', 'day'])
                    if df_pt['tas_mean'].mean() > 100:
                        df_pt['tas_mean'] -= 273.15
                    df_pt['time'] = pd.to_datetime(df_pt[['year', 'month', 'day']])
                    
                    s_pr = df_pt.set_index('time')['pr_mean'].resample('MS').mean()
                    da_pr = s_pr.to_xarray()
                    pr_full = standardize_monthly_da(da_pr, 'pr')
                    
                    s_tas = df_pt.set_index('time')['tas_mean'].resample('MS').mean()
                    da_tas = s_tas.to_xarray()
                    tas_full = standardize_monthly_da(da_tas, 'tas')
                except Exception as e:
                    logging.warning(f"Error computing bias-adjusted PR/TAS for {clean_name}: {e}")

            # Fallback to pr_box_full and tas_box_full from grid data if catchment data unavailable
            if pr_full is None:
                pr_full = standardize_monthly_da(ts_dict.get('pr_box_full'), 'pr')
            if tas_full is None:
                tas_full = standardize_monthly_da(ts_dict.get('tas_box_full'), 'tas')

            if pr_full is None or q_full is None:
                continue

            # Align times
            try:
                common_times = np.intersect1d(pr_full.time.values, q_full.time.values)
                if tas_full is not None:
                    common_times = np.intersect1d(common_times, tas_full.time.values)
                
                if len(common_times) < 36: # at least 3 years
                    continue

                pr_da = pr_full.sel(time=common_times)
                q_da = q_full.sel(time=common_times)
                tas_da = tas_full.sel(time=common_times) if tas_full is not None else None
            except Exception as e:
                logging.warning(f"Error aligning times for {m_key}: {e}")
                continue

            # Determine GWL 3.0 window (31 years)
            gwl_yr = None
            m_info = gwl_years_dict.get(m_key) or gwl_years_dict.get(clean_name)
            if isinstance(m_info, dict):
                gwl_yr = m_info.get(target_gwl)
            elif isinstance(m_info, (int, float, np.integer)):
                gwl_yr = m_info
            if gwl_yr is None:
                gwl_yr = 2050 # default midpoint

            start_yr = max(int(q_da.time.dt.year.min()), int(gwl_yr) - 15)
            end_yr = min(int(q_da.time.dt.year.max()), int(gwl_yr) + 15)

            # Filter to 31-year GWL window
            try:
                q_sub = q_da.sel(time=slice(f"{start_yr}-01-01", f"{end_yr}-12-31"))
                pr_sub = pr_da.sel(time=slice(f"{start_yr - 1}-01-01", f"{end_yr}-12-31")) # include preceding year for lags
                tas_sub = tas_da.sel(time=slice(f"{start_yr - 1}-01-01", f"{end_yr}-12-31")) if tas_da is not None else None
            except Exception as e:
                logging.warning(f"Error slicing GWL window for {m_key}: {e}")
                continue

            if q_sub.size < 24:
                continue

            # Function to compute lagged correlation for target months
            def calc_season_lags(target_months):
                q_times = q_sub.time.values
                q_months = q_sub.time.dt.month.values
                q_vals = q_sub.values

                mask = np.isin(q_months, target_months)
                q_sel_vals = q_vals[mask]
                q_sel_times = q_times[mask]

                if len(q_sel_vals) < 10:
                    return np.full(6, np.nan), np.full(6, np.nan), np.nan, np.nan, np.nan

                pr_lag_vals = {lag: [] for lag in range(1, 7)}
                tas_lag_vals = {lag: [] for lag in range(1, 7)}

                pr_accum_1 = []
                pr_accum_3 = []
                pr_accum_6 = []

                for t_val in q_sel_times:
                    dt_t = pd.Timestamp(t_val)
                    for lag in range(1, 7):
                        prev_date = dt_t - pd.DateOffset(months=lag)
                        target_str = f"{prev_date.year:04d}-{prev_date.month:02d}-01"
                        try:
                            p_match = pr_sub.sel(time=target_str, method='nearest')
                            val_p = float(p_match.values.item() if hasattr(p_match.values, 'item') else p_match.values)
                            pr_lag_vals[lag].append(val_p)
                        except Exception:
                            pr_lag_vals[lag].append(np.nan)

                        if tas_sub is not None:
                            try:
                                t_match = tas_sub.sel(time=target_str, method='nearest')
                                val_t = float(t_match.values.item() if hasattr(t_match.values, 'item') else t_match.values)
                                tas_lag_vals[lag].append(val_t)
                            except Exception:
                                tas_lag_vals[lag].append(np.nan)

                    pr_accum_1.append(pr_lag_vals[1][-1])
                    pr_accum_3.append(sum([pr_lag_vals[k][-1] for k in [1, 2, 3] if np.isfinite(pr_lag_vals[k][-1])]))
                    pr_accum_6.append(sum([pr_lag_vals[k][-1] for k in range(1, 7) if np.isfinite(pr_lag_vals[k][-1])]))

                r_pr_vec = []
                r_tas_vec = []
                for lag in range(1, 7):
                    pv = np.array(pr_lag_vals[lag])
                    valid = np.isfinite(pv) & np.isfinite(q_sel_vals)
                    if np.sum(valid) > 10 and np.std(pv[valid]) > 1e-6 and np.std(q_sel_vals[valid]) > 1e-6:
                        rho_p, _ = spearmanr(pv[valid], q_sel_vals[valid])
                    else:
                        rho_p = np.nan
                    r_pr_vec.append(rho_p)

                    if tas_sub is not None:
                        tv = np.array(tas_lag_vals[lag])
                        valid_t = np.isfinite(tv) & np.isfinite(q_sel_vals)
                        if np.sum(valid_t) > 10 and np.std(tv[valid_t]) > 1e-6 and np.std(q_sel_vals[valid_t]) > 1e-6:
                            rho_t, _ = spearmanr(tv[valid_t], q_sel_vals[valid_t])
                        else:
                            rho_t = np.nan
                        r_tas_vec.append(rho_t)
                    else:
                        r_tas_vec.append(np.nan)

                v1 = np.isfinite(pr_accum_1) & np.isfinite(q_sel_vals)
                v3 = np.isfinite(pr_accum_3) & np.isfinite(q_sel_vals)
                v6 = np.isfinite(pr_accum_6) & np.isfinite(q_sel_vals)

                rho_acc1 = spearmanr(np.array(pr_accum_1)[v1], q_sel_vals[v1])[0] if np.sum(v1) > 10 and np.std(np.array(pr_accum_1)[v1]) > 1e-6 else np.nan
                rho_acc3 = spearmanr(np.array(pr_accum_3)[v3], q_sel_vals[v3])[0] if np.sum(v3) > 10 and np.std(np.array(pr_accum_3)[v3]) > 1e-6 else np.nan
                rho_acc6 = spearmanr(np.array(pr_accum_6)[v6], q_sel_vals[v6])[0] if np.sum(v6) > 10 and np.std(np.array(pr_accum_6)[v6]) > 1e-6 else np.nan

                return np.array(r_pr_vec), np.array(r_tas_vec), rho_acc1, rho_acc3, rho_acc6

            # Winter: DJF [12, 1, 2]
            w_r_pr, w_r_tas, w_acc1, w_acc3, w_acc6 = calc_season_lags([12, 1, 2])
            # Summer: JJA [6, 7, 8]
            s_r_pr, s_r_tas, s_acc1, s_acc3, s_acc6 = calc_season_lags([6, 7, 8])

            if np.all(np.isnan(w_r_pr)):
                continue

            is_extreme_w = is_in_list(m_key, clean_name, ext_w)
            is_non_extreme_w = is_in_list(m_key, clean_name, non_w)
            is_extreme_s = is_in_list(m_key, clean_name, ext_s)
            is_non_extreme_s = is_in_list(m_key, clean_name, non_s)

            model_results.append({
                'key': m_key,
                'name': clean_name,
                'is_extreme_w': is_extreme_w,
                'is_non_extreme_w': is_non_extreme_w,
                'is_extreme_s': is_extreme_s,
                'is_non_extreme_s': is_non_extreme_s,
                'w_r_pr': w_r_pr,
                'w_r_tas': w_r_tas,
                'w_acc': [w_acc1, w_acc3, w_acc6],
                's_r_pr': s_r_pr,
                's_r_tas': s_r_tas,
                's_acc': [s_acc1, s_acc3, s_acc6]
            })

        if not model_results:
            logging.error("Cannot plot Final Figure 8: No valid model lagged correlations computed.")
            return

        import matplotlib.pyplot as plt

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17.5, 5.5))

        # --- Colors ---
        w_high_col = '#08306b' # Winter High-Freq (Dark Blue)
        w_low_col  = '#41b6c4' # Winter Low-Freq (Cyan-Blue)
        s_high_col = '#b2182b' # Summer High-Freq (Dark Red)
        s_low_col  = '#f4a582' # Summer Low-Freq (Coral-Orange)

        t_high_col = '#e66101' # Temperature High-Freq (Dark Orange)
        t_low_col  = '#5e3c99' # Temperature Low-Freq (Purple)

        x_lags = np.arange(1, 7)

        w_ext_models = [r for r in model_results if r['is_extreme_w']]
        w_non_models = [r for r in model_results if r.get('is_non_extreme_w', False)]
        s_ext_models = [r for r in model_results if r['is_extreme_s']]
        s_non_models = [r for r in model_results if r.get('is_non_extreme_s', False)]

        def get_mat_median_quantiles(models_list, key_name):
            if not models_list: return None, None, None
            mat = np.array([r[key_name] for r in models_list if not np.all(np.isnan(r[key_name]))])
            if len(mat) == 0: return None, None, None
            med = np.nanmedian(mat, axis=0)
            q25 = np.nanpercentile(mat, 25, axis=0)
            q75 = np.nanpercentile(mat, 75, axis=0)
            return med, q25, q75

        # Compute medians and quantiles for Winter & Summer P and T vs Q
        w_ext_med, w_ext_q25, w_ext_q75 = get_mat_median_quantiles(w_ext_models, 'w_r_pr')
        w_non_med, w_non_q25, w_non_q75 = get_mat_median_quantiles(w_non_models, 'w_r_pr')
        w_ext_tas_med, w_ext_tas_q25, w_ext_tas_q75 = get_mat_median_quantiles(w_ext_models, 'w_r_tas')
        w_non_tas_med, w_non_tas_q25, w_non_tas_q75 = get_mat_median_quantiles(w_non_models, 'w_r_tas')

        s_ext_med, s_ext_q25, s_ext_q75 = get_mat_median_quantiles(s_ext_models, 's_r_pr')
        s_non_med, s_non_q25, s_non_q75 = get_mat_median_quantiles(s_non_models, 's_r_pr')
        s_ext_tas_med, s_ext_tas_q25, s_ext_tas_q75 = get_mat_median_quantiles(s_ext_models, 's_r_tas')
        s_non_tas_med, s_non_tas_q25, s_non_tas_q75 = get_mat_median_quantiles(s_non_models, 's_r_tas')

        # --- SUBPLOT A: Winter Lowflow Storylines (High- vs Low-Frequency Drivers) ---
        if w_ext_med is not None:
            ax1.plot(x_lags, w_ext_med, color=s_high_col, linewidth=2.5, marker='^', label=f'High-Freq. Models: Q vs P (N={len(w_ext_models)})')
            ax1.fill_between(x_lags, w_ext_q25, w_ext_q75, color=s_high_col, alpha=0.18)
        if w_non_med is not None:
            ax1.plot(x_lags, w_non_med, color=w_high_col, linewidth=2.2, marker='v', linestyle='-.', label=f'Low-Freq. Models: Q vs P (N={len(w_non_models)})')
            ax1.fill_between(x_lags, w_non_q25, w_non_q75, color=w_high_col, alpha=0.15)

        if w_ext_tas_med is not None:
            ax1.plot(x_lags, w_ext_tas_med, color=t_high_col, linewidth=2.2, marker='d', linestyle=':', label='High-Freq. Models: Q vs T')
            ax1.fill_between(x_lags, w_ext_tas_q25, w_ext_tas_q75, color=t_high_col, alpha=0.15)
        if w_non_tas_med is not None:
            ax1.plot(x_lags, w_non_tas_med, color=t_low_col, linewidth=2.0, marker='d', linestyle='--', label='Low-Freq. Models: Q vs T')
            ax1.fill_between(x_lags, w_non_tas_q25, w_non_tas_q75, color=t_low_col, alpha=0.12)

        ax1.axhline(0, color='gray', linestyle=':', linewidth=1.2)
        ax1.set_xlim(0.8, 6.2)
        ax1.set_xticks(x_lags)
        ax1.set_xticklabels([f"Lag {k}\n({k} Mo. Prior)" for k in x_lags], fontsize=8.5)
        ax1.set_ylabel("Spearman Rank Correlation (ρ with Discharge Q)", fontsize=10, weight='bold')
        ax1.set_title("(a) Winter Lowflows: Discharge (Q) Correlation with Preceding P & T", fontsize=9.5, weight='bold', loc='left')
        ax1.legend(loc='upper right', fontsize=8.0, frameon=True)
        ax1.grid(True, linestyle=':', alpha=0.6)

        # --- SUBPLOT B: Summer Lowflow Storylines (High- vs Low-Frequency Drivers) ---
        if s_ext_med is not None:
            ax2.plot(x_lags, s_ext_med, color=s_high_col, linewidth=2.5, marker='^', label=f'High-Freq. Models: Q vs P (N={len(s_ext_models)})')
            ax2.fill_between(x_lags, s_ext_q25, s_ext_q75, color=s_high_col, alpha=0.18)
        if s_non_med is not None:
            ax2.plot(x_lags, s_non_med, color=w_high_col, linewidth=2.2, marker='v', linestyle='-.', label=f'Low-Freq. Models: Q vs P (N={len(s_non_models)})')
            ax2.fill_between(x_lags, s_non_q25, s_non_q75, color=w_high_col, alpha=0.15)

        if s_ext_tas_med is not None:
            ax2.plot(x_lags, s_ext_tas_med, color=t_high_col, linewidth=2.2, marker='d', linestyle=':', label='High-Freq. Models: Q vs T')
            ax2.fill_between(x_lags, s_ext_tas_q25, s_ext_tas_q75, color=t_high_col, alpha=0.15)
        if s_non_tas_med is not None:
            ax2.plot(x_lags, s_non_tas_med, color=t_low_col, linewidth=2.0, marker='d', linestyle='--', label='Low-Freq. Models: Q vs T')
            ax2.fill_between(x_lags, s_non_tas_q25, s_non_tas_q75, color=t_low_col, alpha=0.12)

        ax2.axhline(0, color='gray', linestyle=':', linewidth=1.2)
        ax2.set_xlim(0.8, 6.2)
        ax2.set_xticks(x_lags)
        ax2.set_xticklabels([f"Lag {k}\n({k} Mo. Prior)" for k in x_lags], fontsize=8.5)
        ax2.set_ylabel("Spearman Rank Correlation (ρ with Discharge Q)", fontsize=10, weight='bold')
        ax2.set_title("(b) Summer Lowflows: Discharge (Q) Correlation with Preceding P & T", fontsize=9.5, weight='bold', loc='left')
        ax2.legend(loc='upper right', fontsize=8.0, frameon=True)
        ax2.grid(True, linestyle=':', alpha=0.6)

        # --- SUBPLOT C: Cumulative Antecedent Moisture Memory Across Storylines ---
        w_ext_acc = np.nanmedian(np.array([r['w_acc'] for r in w_ext_models if not np.all(np.isnan(r['w_acc']))]), axis=0) if w_ext_models else [np.nan]*3
        w_non_acc = np.nanmedian(np.array([r['w_acc'] for r in w_non_models if not np.all(np.isnan(r['w_acc']))]), axis=0) if w_non_models else [np.nan]*3
        s_ext_acc = np.nanmedian(np.array([r['s_acc'] for r in s_ext_models if not np.all(np.isnan(r['s_acc']))]), axis=0) if s_ext_models else [np.nan]*3
        s_non_acc = np.nanmedian(np.array([r['s_acc'] for r in s_non_models if not np.all(np.isnan(r['s_acc']))]), axis=0) if s_non_models else [np.nan]*3

        x_bars = np.arange(3)
        width = 0.18

        rects1 = ax3.bar(x_bars - 1.5*width, w_ext_acc, width, label='Winter High-Freq.', color=w_high_col, edgecolor='black', alpha=0.85)
        rects2 = ax3.bar(x_bars - 0.5*width, w_non_acc, width, label='Winter Low-Freq.', color=w_low_col, edgecolor='black', alpha=0.85)
        rects3 = ax3.bar(x_bars + 0.5*width, s_ext_acc, width, label='Summer High-Freq.', color=s_high_col, edgecolor='black', alpha=0.85)
        rects4 = ax3.bar(x_bars + 1.5*width, s_non_acc, width, label='Summer Low-Freq.', color=s_low_col, edgecolor='black', alpha=0.85)

        for rect_group, col in [(rects1, w_high_col), (rects2, w_low_col), (rects3, s_high_col), (rects4, s_low_col)]:
            for r in rect_group:
                h = r.get_height()
                if np.isfinite(h):
                    ax3.text(r.get_x() + r.get_width()/2., h + (0.015 if h>=0 else -0.04), f"{h:+.2f}", ha='center', va='bottom' if h>=0 else 'top', fontsize=7.5, weight='bold', color=col)

        ax3.axhline(0, color='gray', linestyle=':', linewidth=1.2)
        ax3.set_xticks(x_bars)
        ax3.set_xticklabels(['1-Month Antecedent\n(Lag 1 Sum)', '3-Month Antecedent\n(Lag 1-3 Sum)', '6-Month Antecedent\n(Lag 1-6 Sum)'], fontsize=8.5, weight='bold')
        ax3.set_ylabel("Spearman Rank Correlation (ρ with Discharge Q)", fontsize=10, weight='bold')
        ax3.set_title("(c) Cumulative Antecedent P Memory vs. Discharge Q", fontsize=9.5, weight='bold', loc='left')
        ax3.legend(loc='upper left', fontsize=8.0, frameon=True, ncol=2)
        ax3.grid(True, axis='y', linestyle=':', alpha=0.6)

        # Set consistent Y-limits across subplots for easy comparison
        all_r_vals = np.concatenate([
            [r['w_r_pr'] for r in model_results],
            [r['s_r_pr'] for r in model_results]
        ])
        all_r_vals = all_r_vals[np.isfinite(all_r_vals)]
        if len(all_r_vals) > 0:
            ymin = min(-0.18, np.min(all_r_vals) - 0.05)
            ymax = max(0.68, np.max(all_r_vals) + 0.08)
            ax1.set_ylim(ymin, ymax)
            ax2.set_ylim(ymin, ymax)
            ax3.set_ylim(ymin, ymax)

        # Main Title
        scenario_str = Visualizer._format_scenario_title(scenario)
        fig.suptitle(f"Final Figure 8: Lagged Correlation & Storyline Cluster Comparison @ GWL +{target_gwl:.1f}°C ({scenario_str})\n"
                     f"Streamflow / Discharge (Q) Response to Preceding Precipitation (P) & Temperature (T) Drivers",
                     fontsize=11.5, weight='bold', y=0.98)

        fig.tight_layout(rect=(0, 0.02, 1, 0.93))

        plt.savefig(filepath_png, dpi=300, bbox_inches='tight')
        plt.savefig(filepath_pdf, bbox_inches='tight')
        plt.close(fig)
        logging.info(f"Successfully generated 3-Subplot Final Figure 8: {filepath_png} and {filepath_pdf}")

    @staticmethod
    def plot_final_figure_9_daily_discharge_variance_and_extreme_pr_response(
        cmip6_results, discharge_data_loaded, config, scenario='ssp585', target_gwl=3.0, return_period_results=None
    ):
        """
        Creates Final Figure 9: Daily Discharge Variance & Lagged Extreme Precipitation Response Analysis
        Compares High-Frequency (Extreme) vs Low-Frequency (Non-Extreme) Storyline Model Clusters against Historical Baseline.
        """
        filename_png = f"final_figure_9_daily_discharge_variance_{scenario}_gwl{target_gwl}.png"
        filename_pdf = f"final_figure_9_daily_discharge_variance_{scenario}_gwl{target_gwl}.pdf"
        filepath_png = os.path.join(config.PLOT_DIR, filename_png)
        filepath_pdf = os.path.join(config.PLOT_DIR, filename_pdf)

        logging.info(f"Plotting 4-Panel Final Figure 9 to {filepath_png}...")
        Visualizer.ensure_plot_dir_exists()

        # 1. Identify Model Clusters (Extreme vs Non-Extreme Models)
        storyline_classification_2d = cmip6_results.get('storyline_classification_2d', {}) if cmip6_results else {}
        extreme_models = {'Summer': [], 'Winter': []}
        non_extreme_models = {'Summer': [], 'Winter': []}

        if target_gwl in storyline_classification_2d:
            extreme_models['Summer'] = storyline_classification_2d[target_gwl].get('JJA_Extreme Models', storyline_classification_2d[target_gwl].get('summer_Extreme Models', []))
            non_extreme_models['Summer'] = storyline_classification_2d[target_gwl].get('JJA_Non-Extreme Models', storyline_classification_2d[target_gwl].get('summer_Non-Extreme Models', []))
            extreme_models['Winter'] = storyline_classification_2d[target_gwl].get('DJF_Extreme Models', storyline_classification_2d[target_gwl].get('winter_Extreme Models', []))
            non_extreme_models['Winter'] = storyline_classification_2d[target_gwl].get('DJF_Non-Extreme Models', storyline_classification_2d[target_gwl].get('winter_Non-Extreme Models', []))

        if (not extreme_models['Summer'] or not non_extreme_models['Summer']) and return_period_results and 'data' in return_period_results:
            try:
                gwl_node = return_period_results['data'].get(target_gwl, {})
                if not extreme_models['Summer']:
                    extreme_models['Summer'] = gwl_node.get('summer', {}).get('Extreme Models', {}).get('30Q10_low', {}).get('future_keys_all_models', [])
                if not non_extreme_models['Summer']:
                    non_extreme_models['Summer'] = gwl_node.get('summer', {}).get('Non-Extreme Models', {}).get('30Q10_low', {}).get('future_keys_all_models', [])
                if not extreme_models['Winter']:
                    extreme_models['Winter'] = gwl_node.get('winter', {}).get('Extreme Models', {}).get('30Q10_low', {}).get('future_keys_all_models', [])
                if not non_extreme_models['Winter']:
                    non_extreme_models['Winter'] = gwl_node.get('winter', {}).get('Non-Extreme Models', {}).get('30Q10_low', {}).get('future_keys_all_models', [])
            except Exception:
                pass

        if not extreme_models['Summer'] or not non_extreme_models['Summer']:
            try:
                from storyline import StorylineAnalyzer
                analyzer = StorylineAnalyzer(config)
                ext_s, non_s, _ = analyzer.get_composite_extreme_models(cmip6_results, target_gwl, '30Q10_low', 'Summer')
                ext_w, non_w, _ = analyzer.get_composite_extreme_models(cmip6_results, target_gwl, '30Q10_low', 'Winter')
                if ext_s: extreme_models['Summer'] = ext_s
                if non_s: non_extreme_models['Summer'] = non_s
                if ext_w: extreme_models['Winter'] = ext_w
                if non_w: non_extreme_models['Winter'] = non_w
            except Exception as e:
                logging.warning(f"Could not calculate composite extreme models for Fig 9: {e}")

        def is_in_list(m_key, clean_name, target_list):
            if not target_list: return False
            import re
            def clean_model_id(s):
                if not s: return ''
                s = str(s).strip()
                for scn in ['ssp585', 'ssp245', 'ssp126', 'historical']:
                    if s.endswith(f'_{scn}'):
                        s = s[:-len(scn)-1]
                s = re.sub(r'_r\d+i\d+p\d+f\d+$', '', s)
                return s.strip()

            c_key = clean_model_id(m_key)
            c_name = clean_model_id(clean_name)
            raw_targets = set(target_list)
            clean_t = {clean_model_id(x) for x in target_list}
            return (m_key in raw_targets) or (clean_name in raw_targets) or (c_key in clean_t) or (c_name in clean_t)

        # Load Daily Discharge Data
        discharge_filepath = getattr(config, f"DISCHARGE_{scenario.upper()}_FILE", None)
        if not discharge_filepath or not os.path.exists(discharge_filepath):
            discharge_filepath = os.path.join(config.DATA_BASE_PATH, f"CP65_{'8.5' if scenario=='ssp585' else '4.5'}-Tabelle_1.csv")

        if not os.path.exists(discharge_filepath):
            logging.error(f"Cannot plot Final Figure 9: Discharge file {discharge_filepath} not found.")
            return

        try:
            df_q_raw = pd.read_csv(discharge_filepath, sep=';', decimal=',', na_values=['-0,01'])
            date_col = df_q_raw.columns[0]
            df_q_raw = df_q_raw.rename(columns={date_col: 'date'})
            df_q_raw['time'] = pd.to_datetime(df_q_raw['date'])
            df_q_raw['year'] = df_q_raw['time'].dt.year
            df_q_raw['month'] = df_q_raw['time'].dt.month
            df_q_raw['day'] = df_q_raw['time'].dt.day
            
            for col in df_q_raw.columns:
                if col not in ['date', 'time', 'year', 'month', 'day']:
                    df_q_raw[col] = pd.to_numeric(df_q_raw[col], errors='coerce')
        except Exception as e:
            logging.error(f"Error reading discharge CSV in Fig 9: {e}")
            return

        catchment_dirs = [
            '/nas/home/vlw/Desktop/STREAM/final-bias-adjusted-data',
            '/nas/home/vlw/Desktop/STREAM/copernicus-final-adjusted-data',
            '/nas/home/vlw/Desktop/STREAM/in-catchment-data',
            '/nas/home/vlw/Desktop/STREAM/copernicus-in-catchment'
        ]

        def find_catchment_files(model_name, scn):
            for c_dir in catchment_dirs:
                ba_pr1 = os.path.join(c_dir, f"MONTHLY_*_{model_name}_*pr_{scn}_count-*.csv")
                ba_pr2 = os.path.join(c_dir, f"MONTHLY_*_{model_name}_pr_{scn}_count-*.csv")
                f_pr_ba = sorted(glob.glob(ba_pr1) + glob.glob(ba_pr2))
                
                ba_tas1 = os.path.join(c_dir, f"MONTHLY_*_{model_name}_*tas_{scn}_count-*.csv")
                ba_tas2 = os.path.join(c_dir, f"MONTHLY_*_{model_name}_tas_{scn}_count-*.csv")
                f_tas_ba = sorted(glob.glob(ba_tas1) + glob.glob(ba_tas2))
                
                if f_pr_ba and f_tas_ba:
                    return f_pr_ba[0], f_tas_ba[0]
                
                p1_pr = os.path.join(c_dir, f"{model_name}_pr_{scn}_*_in-catchment-units.csv")
                p2_pr = os.path.join(c_dir, f"{model_name}_*_pr_{scn}_*_in-catchment-units.csv")
                f_pr = sorted(glob.glob(p1_pr) + glob.glob(p2_pr))
                
                p1_tas = os.path.join(c_dir, f"{model_name}_tas_{scn}_*_in-catchment-units.csv")
                p2_tas = os.path.join(c_dir, f"{model_name}_*_tas_{scn}_*_in-catchment-units.csv")
                f_tas = sorted(glob.glob(p1_tas) + glob.glob(p2_tas))
                
                if f_pr and f_tas:
                    return f_pr[0], f_tas[0]
            return None, None

        gwl_thresh = cmip6_results.get('gwl_threshold_years', cmip6_results.get('gwl_years', {}))

        available_q_models = [c for c in df_q_raw.columns if c not in ['date', 'time', 'year', 'month', 'day', 'QOBS', 'QSIM']]

        monthly_volatility_hist = {m: [] for m in range(1, 13)}
        monthly_volatility_gwl_all = {m: [] for m in range(1, 13)}
        monthly_volatility_gwl_ext = {m: [] for m in range(1, 13)}
        monthly_volatility_gwl_non = {m: [] for m in range(1, 13)}

        tas_volatility_points_hist = []
        tas_volatility_points_gwl_ext = []
        tas_volatility_points_gwl_non = []

        max_lag = 28
        lags_array = np.arange(-5, max_lag + 1)
        lagged_vol_hist = []
        lagged_vol_gwl_ext = []
        lagged_vol_gwl_non = []

        for model in available_q_models:
            pr_file, tas_file = find_catchment_files(model, scenario)
            if not pr_file or not tas_file:
                continue

            try:
                df_pr = pd.read_csv(pr_file, sep='\t', skiprows=1)
                df_tas = pd.read_csv(tas_file, sep='\t', skiprows=1)
                
                p_cols = [c for c in df_pr.columns if c.startswith('P_')]
                t_cols = [c for c in df_tas.columns if c.startswith('T_')]
                
                df_pr['pr_mean'] = df_pr[p_cols].mean(axis=1)
                df_tas['tas_mean'] = df_tas[t_cols].mean(axis=1)
                
                df_merged = pd.merge(
                    df_pr[['year', 'month', 'day', 'pr_mean']],
                    df_tas[['year', 'month', 'day', 'tas_mean']],
                    on=['year', 'month', 'day']
                )

                if df_merged['tas_mean'].mean() > 100:
                    df_merged['tas_mean'] -= 273.15

                df_m = pd.merge(
                    df_merged,
                    df_q_raw[['year', 'month', 'day', model]].rename(columns={model: 'Q'}),
                    on=['year', 'month', 'day']
                ).sort_values(by=['year', 'month', 'day']).reset_index(drop=True)

                df_m['Q'] = df_m['Q'].interpolate(method='linear')
                df_m['dQ'] = df_m['Q'].diff().abs()
                df_m['tas_ant30'] = df_m['tas_mean'].rolling(30, min_periods=15).mean()

                m_gwl_dict = gwl_thresh.get(model) or gwl_thresh.get(f"{model}_{scenario}")
                gwl_yr = None
                if isinstance(m_gwl_dict, dict):
                    gwl_yr = m_gwl_dict.get(target_gwl)
                elif isinstance(m_gwl_dict, (int, float, np.integer)):
                    gwl_yr = m_gwl_dict

                mask_hist = (df_m['year'] >= 1985) & (df_m['year'] <= 2014)
                if gwl_yr is not None and np.isfinite(gwl_yr):
                    gwl_yr = int(gwl_yr)
                    mask_gwl = (df_m['year'] >= max(1960, gwl_yr - 15)) & (df_m['year'] <= min(2099, gwl_yr + 15))
                else:
                    mask_gwl = (df_m['year'] >= 2070) & (df_m['year'] <= 2099)

                is_ext_s = is_in_list(model, model, extreme_models['Summer'])
                is_non_s = is_in_list(model, model, non_extreme_models['Summer'])
                is_ext_w = is_in_list(model, model, extreme_models['Winter'])
                is_non_w = is_in_list(model, model, non_extreme_models['Winter'])

                for m in range(1, 13):
                    val_h = df_m[mask_hist & (df_m['month'] == m)]['dQ'].mean()
                    val_g = df_m[mask_gwl & (df_m['month'] == m)]['dQ'].mean()
                    if np.isfinite(val_h): monthly_volatility_hist[m].append(val_h)
                    if np.isfinite(val_g): 
                        monthly_volatility_gwl_all[m].append(val_g)
                        is_ext_m = is_ext_s if m in [5, 6, 7, 8, 9, 10] else is_ext_w
                        is_non_m = is_non_s if m in [5, 6, 7, 8, 9, 10] else is_non_w
                        if is_ext_m:
                            monthly_volatility_gwl_ext[m].append(val_g)
                        elif is_non_m:
                            monthly_volatility_gwl_non[m].append(val_g)

                sub_h = df_m[mask_hist & (df_m['month'].isin([5, 6, 7, 8, 9, 10]))].dropna(subset=['tas_ant30', 'dQ'])
                sub_g = df_m[mask_gwl & (df_m['month'].isin([5, 6, 7, 8, 9, 10]))].dropna(subset=['tas_ant30', 'dQ'])
                
                if not sub_h.empty:
                    tas_volatility_points_hist.append(sub_h[['tas_ant30', 'dQ']])
                if not sub_g.empty:
                    if is_ext_s:
                        tas_volatility_points_gwl_ext.append(sub_g[['tas_ant30', 'dQ']])
                    elif is_non_s:
                        tas_volatility_points_gwl_non.append(sub_g[['tas_ant30', 'dQ']])

                def extract_event_composites(df_sub):
                    if df_sub.empty or len(df_sub) < 50: return None
                    pr_p95 = df_sub['pr_mean'].quantile(0.95)
                    event_indices = df_sub[(df_sub['pr_mean'] >= pr_p95) & (df_sub['pr_mean'] > df_sub['pr_mean'].shift(1, fill_value=0))].index
                    
                    event_profiles = []
                    for idx in event_indices:
                        if idx - 5 >= df_sub.index[0] and idx + max_lag <= df_sub.index[-1]:
                            slice_dQ = df_sub.loc[idx - 5 : idx + max_lag, 'dQ'].values
                            if len(slice_dQ) == len(lags_array) and not np.isnan(slice_dQ).any():
                                event_profiles.append(slice_dQ)
                    if event_profiles:
                        return np.mean(event_profiles, axis=0)
                    return None

                comp_h = extract_event_composites(df_m[mask_hist].reset_index(drop=True))
                comp_g = extract_event_composites(df_m[mask_gwl].reset_index(drop=True))

                if comp_h is not None: lagged_vol_hist.append(comp_h)
                if comp_g is not None:
                    if is_ext_s or is_ext_w:
                        lagged_vol_gwl_ext.append(comp_g)
                    elif is_non_s or is_non_w:
                        lagged_vol_gwl_non.append(comp_g)

            except Exception as e:
                logging.warning(f"Error processing model {model} in Fig 9: {e}")

        if not monthly_volatility_hist:
            logging.error("Cannot plot Final Figure 9: No valid model data processed.")
            return

        fig = plt.figure(figsize=(16, 12), dpi=300)
        gs = gridspec.GridSpec(2, 2, hspace=0.32, wspace=0.25)

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])

        high_freq_col = '#b2182b' # Dark Red
        low_freq_col  = '#2166ac' # Dark Blue
        base_col      = '#1f77b4' # Blue

        months_x = np.arange(1, 13)
        month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        mean_h_m = [np.mean(monthly_volatility_hist[int(m)]) if monthly_volatility_hist[int(m)] else np.nan for m in months_x]
        mean_g_ext = [np.mean(monthly_volatility_gwl_ext[int(m)]) if monthly_volatility_gwl_ext[int(m)] else np.nan for m in months_x]
        mean_g_non = [np.mean(monthly_volatility_gwl_non[int(m)]) if monthly_volatility_gwl_non[int(m)] else np.nan for m in months_x]

        ax1.plot(months_x, mean_h_m, 'o-', color=base_col, linewidth=2.2, label='Baseline (1985–2014)')
        if any(np.isfinite(mean_g_ext)):
            ax1.plot(months_x, mean_g_ext, '^-', color=high_freq_col, linewidth=2.5, label=f'GWL +{target_gwl:.1f}°C (High-Freq. Models)')
        if any(np.isfinite(mean_g_non)):
            ax1.plot(months_x, mean_g_non, 'v-.', color=low_freq_col, linewidth=2.2, label=f'GWL +{target_gwl:.1f}°C (Low-Freq. Models)')

        ax1.set_xticks(months_x)
        ax1.set_xticklabels(month_labels, fontsize=9.5, weight='bold')
        ax1.set_ylabel("Daily Discharge Volatility $|dQ/dt|$ ($m^3/s / day$)", fontsize=10.5, weight='bold')
        ax1.set_title("(a) Seasonal Cycle of Daily Discharge Volatility", fontsize=11, weight='bold', loc='left')
        ax1.grid(True, linestyle=':', alpha=0.6)
        ax1.legend(loc='upper left', fontsize=9.0, frameon=True)

        if tas_volatility_points_hist:
            df_tas_vol_h = pd.concat(tas_volatility_points_hist, ignore_index=True)
            bins = np.arange(5, 30, 2.5)
            df_tas_vol_h['bin'] = pd.cut(df_tas_vol_h['tas_ant30'], bins=bins)
            binned_h = df_tas_vol_h.groupby('bin', observed=False)['dQ'].agg(['mean', 'sem']).reset_index()
            bin_centers = [b.mid for b in binned_h['bin']]

            ax2.plot(bin_centers, binned_h['mean'], 'o-', color=base_col, linewidth=2.2, label='Baseline (1985–2014)')
            ax2.fill_between(bin_centers, binned_h['mean'] - 1.96*binned_h['sem'], binned_h['mean'] + 1.96*binned_h['sem'], color=base_col, alpha=0.15)

            if tas_volatility_points_gwl_ext:
                df_tas_vol_ext = pd.concat(tas_volatility_points_gwl_ext, ignore_index=True)
                df_tas_vol_ext['bin'] = pd.cut(df_tas_vol_ext['tas_ant30'], bins=bins)
                binned_ext = df_tas_vol_ext.groupby('bin', observed=False)['dQ'].agg(['mean', 'sem']).reset_index()
                bin_centers_ext = [b.mid for b in binned_ext['bin']]
                ax2.plot(bin_centers_ext, binned_ext['mean'], '^-', color=high_freq_col, linewidth=2.5, label=f'GWL +{target_gwl:.1f}°C (High-Freq. Models)')
                ax2.fill_between(bin_centers_ext, binned_ext['mean'] - 1.96*binned_ext['sem'], binned_ext['mean'] + 1.96*binned_ext['sem'], color=high_freq_col, alpha=0.2)

            if tas_volatility_points_gwl_non:
                df_tas_vol_non = pd.concat(tas_volatility_points_gwl_non, ignore_index=True)
                df_tas_vol_non['bin'] = pd.cut(df_tas_vol_non['tas_ant30'], bins=bins)
                binned_non = df_tas_vol_non.groupby('bin', observed=False)['dQ'].agg(['mean', 'sem']).reset_index()
                bin_centers_non = [b.mid for b in binned_non['bin']]
                ax2.plot(bin_centers_non, binned_non['mean'], 'v-.', color=low_freq_col, linewidth=2.2, label=f'GWL +{target_gwl:.1f}°C (Low-Freq. Models)')
                ax2.fill_between(bin_centers_non, binned_non['mean'] - 1.96*binned_non['sem'], binned_non['mean'] + 1.96*binned_non['sem'], color=low_freq_col, alpha=0.15)

            ax2.set_xlabel("Antecedent 30-Day Catchment Temperature ($°C$)", fontsize=10.5, weight='bold')
            ax2.set_ylabel("Daily Discharge Volatility $|dQ/dt|$ ($m^3/s / day$)", fontsize=10.5, weight='bold')
            ax2.set_title("(b) Discharge Volatility vs. Antecedent Temperature (Dry Soil Proxy)", fontsize=11, weight='bold', loc='left')
            ax2.grid(True, linestyle=':', alpha=0.6)
            ax2.legend(loc='upper left', fontsize=9.0, frameon=True)

        if lagged_vol_hist:
            arr_h = np.array(lagged_vol_hist)
            m_h = np.mean(arr_h, axis=0)
            sem_h = np.std(arr_h, axis=0) / np.sqrt(len(arr_h)) if len(arr_h) > 1 else np.zeros_like(m_h)
            ax3.plot(lags_array, m_h, 'o-', color=base_col, linewidth=2.2, label='Baseline (1985–2014)')
            ax3.fill_between(lags_array, m_h - sem_h, m_h + sem_h, color=base_col, alpha=0.15)

        if lagged_vol_gwl_ext:
            arr_ext = np.array(lagged_vol_gwl_ext)
            m_ext = np.mean(arr_ext, axis=0)
            sem_ext = np.std(arr_ext, axis=0) / np.sqrt(len(arr_ext)) if len(arr_ext) > 1 else np.zeros_like(m_ext)
            ax3.plot(lags_array, m_ext, '^-', color=high_freq_col, linewidth=2.5, label=f'GWL +{target_gwl:.1f}°C (High-Freq. Models)')
            ax3.fill_between(lags_array, m_ext - sem_ext, m_ext + sem_ext, color=high_freq_col, alpha=0.2)

        if lagged_vol_gwl_non:
            arr_non = np.array(lagged_vol_gwl_non)
            m_non = np.mean(arr_non, axis=0)
            sem_non = np.std(arr_non, axis=0) / np.sqrt(len(arr_non)) if len(arr_non) > 1 else np.zeros_like(m_non)
            ax3.plot(lags_array, m_non, 'v-.', color=low_freq_col, linewidth=2.2, label=f'GWL +{target_gwl:.1f}°C (Low-Freq. Models)')
            ax3.fill_between(lags_array, m_non - sem_non, m_non + sem_non, color=low_freq_col, alpha=0.15)

        ax3.axvline(0, color='gray', linestyle='--', label='Extreme PR Event (Day 0)')
        ax3.set_xlabel("Lag after Extreme Precipitation Event (Days)", fontsize=10.5, weight='bold')
        ax3.set_ylabel("Composite Discharge Volatility $|dQ/dt|$ ($m^3/s / day$)", fontsize=10.5, weight='bold')
        ax3.set_title("(c) Lagged Response of Discharge Volatility (Days 0 to +28)", fontsize=11, weight='bold', loc='left')
        ax3.grid(True, linestyle=':', alpha=0.6)
        ax3.legend(loc='upper right', fontsize=8.5, frameon=True)

        weeks = ['Week 1\n(Days 1–7)', 'Week 2\n(Days 8–14)', 'Week 3\n(Days 15–21)', 'Week 4\n(Days 22–28)']
        x_w = np.arange(len(weeks))
        width = 0.25

        def get_weekly_means(arr_list):
            if not arr_list: return [0, 0, 0, 0], [0, 0, 0, 0]
            arr = np.array(arr_list)
            w1 = np.mean(arr[:, 6:13], axis=1)
            w2 = np.mean(arr[:, 13:20], axis=1)
            w3 = np.mean(arr[:, 20:27], axis=1)
            w4 = np.mean(arr[:, 27:34], axis=1)
            means = [np.mean(w1), np.mean(w2), np.mean(w3), np.mean(w4)]
            sems = [np.std(w1)/np.sqrt(len(w1)) if len(w1)>1 else 0,
                    np.std(w2)/np.sqrt(len(w2)) if len(w2)>1 else 0,
                    np.std(w3)/np.sqrt(len(w3)) if len(w3)>1 else 0,
                    np.std(w4)/np.sqrt(len(w4)) if len(w4)>1 else 0]
            return means, sems

        m_w_h, s_w_h = get_weekly_means(lagged_vol_hist)
        m_w_g_ext, s_w_g_ext = get_weekly_means(lagged_vol_gwl_ext)
        m_w_g_non, s_w_g_non = get_weekly_means(lagged_vol_gwl_non)

        ax4.bar(x_w - width, m_w_h, width, yerr=s_w_h, capsize=4, color=base_col, label='Baseline (1985–2014)', alpha=0.85)
        if any(m > 0 for m in m_w_g_ext):
            ax4.bar(x_w, m_w_g_ext, width, yerr=s_w_g_ext, capsize=4, color=high_freq_col, label=f'GWL +{target_gwl:.1f}°C (High-Freq. Models)', alpha=0.85)
        if any(m > 0 for m in m_w_g_non):
            ax4.bar(x_w + width, m_w_g_non, width, yerr=s_w_g_non, capsize=4, color=low_freq_col, label=f'GWL +{target_gwl:.1f}°C (Low-Freq. Models)', alpha=0.85)

        ax4.set_xticks(x_w)
        ax4.set_xticklabels(weeks, fontsize=9.5, weight='bold')
        ax4.set_ylabel("Mean Daily Discharge Volatility ($m^3/s / day$)", fontsize=10.5, weight='bold')
        ax4.set_title("(d) Weekly Post-Extreme Precipitation Discharge Volatility", fontsize=11, weight='bold', loc='left')
        ax4.grid(True, axis='y', linestyle=':', alpha=0.6)
        ax4.legend(loc='upper right', fontsize=8.5, frameon=True)

        scenario_str = Visualizer._format_scenario_title(scenario)
        fig.suptitle(
            f"Final Figure 9: Daily Discharge Variance & Extreme Precipitation Response @ GWL +{target_gwl:.1f}°C ({scenario_str})\n"
            f"High-Frequency vs. Low-Frequency Storyline Model Breakdown (Soil Desiccation & Post-Precipitation Multi-Week Volatility)",
            fontsize=12.5, weight='bold', y=0.99
        )

        fig.tight_layout(rect=(0, 0.02, 1, 0.95))

        plt.savefig(filepath_png, dpi=300, bbox_inches='tight')
        plt.savefig(filepath_pdf, bbox_inches='tight')
        plt.close(fig)
        logging.info(f"Successfully generated 4-Panel Final Figure 9: {filepath_png} and {filepath_pdf}")

    @staticmethod
    def plot_final_figure_10_seasonal_cycles(
        cmip6_results, discharge_data_loaded, config, scenario='ssp585', target_gwl=3.0, return_period_results=None
    ):
        """
        Creates Final Figure 10: Seasonal Cycles (PR, TAS, Discharge, U850 for u>0)
        at GWL +3.0°C under specified scenario (e.g., ssp585).
        
        Subplots arranged in 2 sections (2 rows x 4 columns):
          Row 1: Grouped by Summer Storyline Classification (High-Freq vs Low-Freq)
          Row 2: Grouped by Winter Storyline Classification (High-Freq vs Low-Freq)
        
        Year Filtering:
          - High-Frequency models: filter to low-flow cluster event years (summer low-flow followed by winter low-flow).
          - Low-Frequency models: filter to the same relative years as identified from HF cluster events.
        """
        filename_png = f"final_figure_10_seasonal_cycles_{scenario}_gwl{target_gwl:.1f}.png"
        filename_pdf = f"final_figure_10_seasonal_cycles_{scenario}_gwl{target_gwl:.1f}.pdf"
        filepath_png = os.path.join(config.PLOT_DIR, filename_png)
        filepath_pdf = os.path.join(config.PLOT_DIR, filename_pdf)

        logging.info(f"Plotting 8-Panel Final Figure 10 (Seasonal Cycles) to {filepath_png}...")
        Visualizer.ensure_plot_dir_exists()

        import pandas as pd
        import glob
        import xarray as xr

        if not cmip6_results:
            logging.error("Cannot plot Final Figure 10: Missing cmip6_results.")
            return

        metric_timeseries = cmip6_results.get('model_metric_timeseries', {})
        gwl_years_dict = cmip6_results.get('gwl_threshold_years', cmip6_results.get('gwl_years', {}))

        if not metric_timeseries:
            logging.error("Cannot plot Final Figure 10: Missing model_metric_timeseries.")
            return

        # 1. Identify Model Clusters (Extreme vs Non-Extreme for Winter & Summer)
        ext_w, non_w, ext_s, non_s = [], [], [], []
        if return_period_results and 'data' in return_period_results and target_gwl in return_period_results['data']:
            try:
                gwl_node = return_period_results['data'][target_gwl]
                ext_w = gwl_node.get('winter', {}).get('Extreme Models', {}).get('30Q10_low', {}).get('future_keys_all_models', [])
                non_w = gwl_node.get('winter', {}).get('Non-Extreme Models', {}).get('30Q10_low', {}).get('future_keys_all_models', [])
                ext_s = gwl_node.get('summer', {}).get('Extreme Models', {}).get('30Q10_low', {}).get('future_keys_all_models', [])
                non_s = gwl_node.get('summer', {}).get('Non-Extreme Models', {}).get('30Q10_low', {}).get('future_keys_all_models', [])
            except Exception:
                pass

        storyline_classification_2d = cmip6_results.get('storyline_classification_2d', {}) if cmip6_results else {}
        if target_gwl in storyline_classification_2d:
            if not ext_w: ext_w = storyline_classification_2d[target_gwl].get('DJF_Extreme Models', storyline_classification_2d[target_gwl].get('winter_Extreme Models', []))
            if not non_w: non_w = storyline_classification_2d[target_gwl].get('DJF_Non-Extreme Models', storyline_classification_2d[target_gwl].get('winter_Non-Extreme Models', []))
            if not ext_s: ext_s = storyline_classification_2d[target_gwl].get('JJA_Extreme Models', storyline_classification_2d[target_gwl].get('summer_Extreme Models', []))
            if not non_s: non_s = storyline_classification_2d[target_gwl].get('JJA_Non-Extreme Models', storyline_classification_2d[target_gwl].get('summer_Non-Extreme Models', []))

        if not ext_w or not non_w or not ext_s or not non_s:
            try:
                from storyline import StorylineAnalyzer
                analyzer = StorylineAnalyzer(config)
                ext_w_calc, non_w_calc, _ = analyzer.get_composite_extreme_models(cmip6_results, target_gwl, '30Q10_low', 'Winter')
                ext_s_calc, non_s_calc, _ = analyzer.get_composite_extreme_models(cmip6_results, target_gwl, '30Q10_low', 'Summer')
                if not ext_w and ext_w_calc: ext_w = ext_w_calc
                if not non_w and non_w_calc: non_w = non_w_calc
                if not ext_s and ext_s_calc: ext_s = ext_s_calc
                if not non_s and non_s_calc: non_s = non_s_calc
            except Exception as e:
                logging.warning(f"Could not calculate composite extreme models for Fig 10: {e}")

        def is_in_list(m_key, clean_name, target_list):
            if not target_list: return False
            import re
            def clean_model_id(s):
                if not s: return ''
                s = str(s).strip()
                for scn in ['ssp585', 'ssp245', 'ssp126', 'historical']:
                    if s.endswith(f'_{scn}'):
                        s = s[:-len(scn)-1]
                s = re.sub(r'_r\d+i\d+p\d+f\d+$', '', s)
                return s.strip()

            c_key = clean_model_id(m_key)
            c_name = clean_model_id(clean_name)
            raw_targets = set(target_list)
            clean_t = {clean_model_id(x) for x in target_list}
            return (m_key in raw_targets) or (clean_name in raw_targets) or (c_key in clean_t) or (c_name in clean_t)

        # 2. Discharge Data Pre-loading
        discharge_filepath = getattr(config, f"DISCHARGE_{scenario.upper()}_FILE", None)
        if not discharge_filepath or not os.path.exists(discharge_filepath):
            discharge_filepath = os.path.join(config.DATA_BASE_PATH, f"CP65_{'8.5' if scenario=='ssp585' else '4.5'}-Tabelle_1.csv")

        df_q_raw = None
        if os.path.exists(discharge_filepath):
            try:
                df_q_raw = pd.read_csv(discharge_filepath, sep=';', decimal=',', na_values=['-0,01'])
                date_col = df_q_raw.columns[0]
                df_q_raw = df_q_raw.rename(columns={date_col: 'date'})
                df_q_raw['time'] = pd.to_datetime(df_q_raw['date'])
                df_q_raw['year'] = df_q_raw['time'].dt.year
                df_q_raw['month'] = df_q_raw['time'].dt.month
                df_q_raw['day'] = df_q_raw['time'].dt.day
                for col in df_q_raw.columns:
                    if col not in ['date', 'time', 'year', 'month', 'day']:
                        df_q_raw[col] = pd.to_numeric(df_q_raw[col], errors='coerce')
            except Exception as e:
                logging.warning(f"Failed to load discharge CSV in Fig 10: {e}")

        catchment_dirs = [
            '/nas/home/vlw/Desktop/STREAM/final-bias-adjusted-data',
            '/nas/home/vlw/Desktop/STREAM/copernicus-final-adjusted-data',
            '/nas/home/vlw/Desktop/STREAM/in-catchment-data',
            '/nas/home/vlw/Desktop/STREAM/copernicus-in-catchment'
        ]

        def find_catchment_files(model_name, scn):
            for c_dir in catchment_dirs:
                ba_pr1 = os.path.join(c_dir, f"MONTHLY_*_{model_name}_*pr_{scn}_count-*.csv")
                ba_pr2 = os.path.join(c_dir, f"MONTHLY_*_{model_name}_pr_{scn}_count-*.csv")
                f_pr_ba = sorted(glob.glob(ba_pr1) + glob.glob(ba_pr2))
                
                ba_tas1 = os.path.join(c_dir, f"MONTHLY_*_{model_name}_*tas_{scn}_count-*.csv")
                ba_tas2 = os.path.join(c_dir, f"MONTHLY_*_{model_name}_tas_{scn}_count-*.csv")
                f_tas_ba = sorted(glob.glob(ba_tas1) + glob.glob(ba_tas2))
                
                if f_pr_ba and f_tas_ba:
                    return f_pr_ba[0], f_tas_ba[0]
                
                p1_pr = os.path.join(c_dir, f"{model_name}_pr_{scn}_*_in-catchment-units.csv")
                p2_pr = os.path.join(c_dir, f"{model_name}_*_pr_{scn}_*_in-catchment-units.csv")
                f_pr = sorted(glob.glob(p1_pr) + glob.glob(p2_pr))
                
                p1_tas = os.path.join(c_dir, f"{model_name}_tas_{scn}_*_in-catchment-units.csv")
                p2_tas = os.path.join(c_dir, f"{model_name}_*_tas_{scn}_*_in-catchment-units.csv")
                f_tas = sorted(glob.glob(p1_tas) + glob.glob(p2_tas))
                
                if f_pr and f_tas:
                    return f_pr[0], f_tas[0]
            return None, None

        model_keys = sorted([k for k in metric_timeseries.keys() if k.endswith(scenario)])
        if not model_keys and df_q_raw is not None:
            model_keys = [c for c in df_q_raw.columns if c not in ['date', 'time', 'year', 'month', 'day', 'QOBS', 'QSIM']]
        elif not model_keys:
            model_keys = sorted(list(metric_timeseries.keys()))

        # Build list of per-model record metadata & data
        model_records = []

        from storyline import StorylineAnalyzer
        analyzer_inst = StorylineAnalyzer(config)

        for m_key in model_keys:
            ts_dict = metric_timeseries.get(m_key, {})
            clean_name = m_key.replace(f"_{scenario}", "")

            ts_summer = ts_dict.get('30Q_low_summer')
            ts_winter = ts_dict.get('30Q_low_winter')
            ts_annual = ts_dict.get('30Q_low_full_year')

            if ts_summer is None or ts_winter is None or ts_annual is None:
                continue

            # Historical 30Q10 threshold
            try:
                hist_slice = ts_annual.sel(year=slice(1960, 2014))
                if hist_slice.year.size < 10:
                    hist_slice = ts_annual.where(ts_annual.year < 2015, drop=True)
            except Exception:
                hist_slice = ts_annual.where(ts_annual.year < 2015, drop=True)

            hist_vals = hist_slice.values
            hist_vals = hist_vals[np.isfinite(hist_vals)]
            if len(hist_vals) < 10:
                continue
            thresh_30q10 = np.quantile(hist_vals, 0.10)

            # GWL year
            gwl_yr = None
            m_info = gwl_years_dict.get(m_key) or gwl_years_dict.get(clean_name)
            if isinstance(m_info, dict):
                gwl_yr = m_info.get(target_gwl)
            elif isinstance(m_info, (int, float, np.integer)):
                gwl_yr = m_info

            if gwl_yr is None or not np.isfinite(gwl_yr):
                continue
            gwl_yr = int(gwl_yr)

            window_half = config.GWL_YEARS_WINDOW // 2 if hasattr(config, 'GWL_YEARS_WINDOW') else 15
            rel_years = np.arange(-window_half, window_half + 1)
            cal_years = gwl_yr + rel_years

            # Find cluster event relative years (summer low-flow followed by winter low-flow)
            cluster_rel_years = []
            for ry, cy in zip(rel_years, cal_years):
                is_s, is_w = False, False
                if cy in ts_summer.year.values:
                    v_s = ts_summer.sel(year=cy).item()
                    if np.isfinite(v_s) and v_s < thresh_30q10:
                        is_s = True
                if cy in ts_winter.year.values:
                    v_w = ts_winter.sel(year=cy).item()
                    if np.isfinite(v_w) and v_w < thresh_30q10:
                        is_w = True
                if is_s and is_w:
                    cluster_rel_years.append(ry)

            # Load monthly PR & TAS
            pr_df, tas_df = None, None
            pr_file, tas_file = find_catchment_files(clean_name, scenario)
            if pr_file and tas_file:
                try:
                    df_pr_raw = pd.read_csv(pr_file, sep='\t', skiprows=1)
                    df_tas_raw = pd.read_csv(tas_file, sep='\t', skiprows=1)
                    p_cols = [c for c in df_pr_raw.columns if c.startswith('P_')]
                    t_cols = [c for c in df_tas_raw.columns if c.startswith('T_')]
                    df_pr_raw['pr_mean'] = df_pr_raw[p_cols].mean(axis=1)
                    df_tas_raw['tas_mean'] = df_tas_raw[t_cols].mean(axis=1)

                    df_pt = pd.merge(df_pr_raw[['year', 'month', 'day', 'pr_mean']], df_tas_raw[['year', 'month', 'day', 'tas_mean']], on=['year', 'month', 'day'])
                    if df_pt['tas_mean'].mean() > 100:
                        df_pt['tas_mean'] -= 273.15
                    pr_df = df_pt.groupby(['year', 'month'])['pr_mean'].mean().reset_index()
                    tas_df = df_pt.groupby(['year', 'month'])['tas_mean'].mean().reset_index()
                except Exception as e:
                    logging.warning(f"Error loading catchment files for {clean_name}: {e}")

            # Fallback for PR and TAS if catchment files unavailable
            if pr_df is None:
                da_pr_box = ts_dict.get('pr_box_full')
                if da_pr_box is not None:
                    pr_df = pd.DataFrame({
                        'year': da_pr_box.time.dt.year.values,
                        'month': da_pr_box.time.dt.month.values,
                        'pr_mean': da_pr_box.values
                    })
            if tas_df is None:
                da_tas_box = ts_dict.get('tas_box_full')
                if da_tas_box is not None:
                    tas_df = pd.DataFrame({
                        'year': da_tas_box.time.dt.year.values,
                        'month': da_tas_box.time.dt.month.values,
                        'tas_mean': da_tas_box.values
                    })

            # Load monthly Discharge
            q_df = None
            if df_q_raw is not None and clean_name in df_q_raw.columns:
                q_sub = df_q_raw[['year', 'month', clean_name]].dropna()
                q_df = q_sub.groupby(['year', 'month'])[clean_name].mean().reset_index()
                q_df = q_df.rename(columns={clean_name: 'q_mean'})
            elif ts_dict.get('discharge_monthly_full') is not None:
                da_q = ts_dict.get('discharge_monthly_full')
                q_df = pd.DataFrame({
                    'year': da_q.time.dt.year.values,
                    'month': da_q.time.dt.month.values,
                    'q_mean': da_q.values
                })

            # Load monthly U850 (u > 0)
            u850_df = None
            try:
                preloaded_ua = cmip6_results.get('preloaded_cmip6_data', {}).get(f"{clean_name}_{scenario}", {}).get('ua')
                if preloaded_ua is None:
                    preloaded_ua = analyzer_inst._load_and_preprocess_model_data(clean_name, [scenario], 'ua')
                if preloaded_ua is not None:
                    # Spatial box crop
                    lats = preloaded_ua.lat.values
                    lons = preloaded_ua.lon.values
                    lat_mask = (lats >= config.BOX_LAT_MIN) & (lats <= config.BOX_LAT_MAX)
                    lon_mask = (lons >= config.BOX_LON_MIN) & (lons <= config.BOX_LON_MAX)
                    
                    ua_sub = preloaded_ua.isel(lat=lat_mask, lon=lon_mask)
                    # Filter u > 0
                    ua_pos = ua_sub.where(ua_sub > 0)
                    ua_box_mean = ua_pos.mean(dim=[d for d in ua_pos.dims if d not in ['time']], skipna=True)
                    
                    u850_df = pd.DataFrame({
                        'year': ua_box_mean.time.dt.year.values,
                        'month': ua_box_mean.time.dt.month.values,
                        'u850_mean': ua_box_mean.values
                    })
            except Exception as e:
                logging.warning(f"Could not calculate U850 for {clean_name}: {e}")

            # Calculate event timing (exact months of 30Q10 low-flow event minimums)
            s_event_months = []
            w_event_months = []
            if df_q_raw is not None and clean_name in df_q_raw.columns:
                try:
                    df_m_q = df_q_raw[['time', 'year', clean_name]].dropna()
                    s_q_m = df_m_q.set_index('time')[clean_name]
                    q_30d_m = s_q_m.rolling(30, center=True, min_periods=15).mean()

                    for ry in cluster_rel_years:
                        cy = gwl_yr + ry
                        # Summer event timing (JJA)
                        try:
                            sub_s = q_30d_m.loc[f'{cy}-06-01':f'{cy}-09-30']
                            if not sub_s.empty and sub_s.min() < thresh_30q10:
                                t_min_s = sub_s.idxmin()
                                m_val = t_min_s.month + (t_min_s.day - 1) / 31.0
                                s_event_months.append(m_val)
                        except Exception:
                            pass

                        # Winter event timing (DJF)
                        try:
                            sub_w = q_30d_m.loc[f'{cy-1}-12-01':f'{cy}-03-15']
                            if not sub_w.empty and sub_w.min() < thresh_30q10:
                                t_min_w = sub_w.idxmin()
                                m_val = (t_min_w.month if t_min_w.month != 12 else 0) + (t_min_w.day - 1) / 31.0
                                if m_val < 0.5:
                                    m_val = 12.0 + (t_min_w.day - 1) / 31.0
                                w_event_months.append(m_val)
                        except Exception:
                            pass
                except Exception as e:
                    logging.warning(f"Could not calculate event timing for {clean_name}: {e}")

            # Determine Group Classification
            is_high_s = is_in_list(m_key, clean_name, ext_s)
            is_low_s  = is_in_list(m_key, clean_name, non_s)
            grp_s = 'High-Freq.' if is_high_s else ('Low-Freq.' if is_low_s else 'Other')

            is_high_w = is_in_list(m_key, clean_name, ext_w)
            is_low_w  = is_in_list(m_key, clean_name, non_w)
            grp_w = 'High-Freq.' if is_high_w else ('Low-Freq.' if is_low_w else 'Other')

            model_records.append({
                'key': m_key,
                'name': clean_name,
                'gwl_year': gwl_yr,
                'grp_s': grp_s,
                'grp_w': grp_w,
                'cluster_rel_years': cluster_rel_years,
                's_event_months': s_event_months,
                'w_event_months': w_event_months,
                'pr_df': pr_df,
                'tas_df': tas_df,
                'q_df': q_df,
                'u850_df': u850_df
            })

        if not model_records:
            logging.error("Cannot plot Final Figure 10: No valid model records.")
            return

        # Setup figure layout (4 rows x 4 columns: larger size, sharex=False so all subplots show X labels)
        fig, axes = plt.subplots(4, 4, figsize=(20, 20), sharex=False)
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        month_indices = np.arange(1, 13)

        high_col = '#b2182b' # Dark Crimson
        low_col  = '#2166ac' # Dark Blue
        diff_col = '#7a0177' # Deep Purple for difference

        def calc_seasonal_cycle_for_models(rec_list, target_rel_years_dict, var_col, df_key):
            """
            Computes average 12-month seasonal cycle across models in rec_list.
            """
            model_cycles = []
            for rec in rec_list:
                df = rec.get(df_key)
                if df is None or df.empty or var_col not in df.columns:
                    continue
                gwl_yr = rec['gwl_year']
                rel_yrs = target_rel_years_dict.get(rec['name'], rec['cluster_rel_years'])
                if not rel_yrs:
                    rel_yrs = np.arange(-15, 16)
                
                sel_cal_years = [gwl_yr + ry for ry in rel_yrs]
                df_sub = df[df['year'].isin(sel_cal_years)]
                if df_sub.empty:
                    continue
                
                cycle = df_sub.groupby('month')[var_col].mean()
                cycle = cycle.reindex(month_indices).values
                if np.all(np.isfinite(cycle)):
                    model_cycles.append(cycle)
            
            if not model_cycles:
                return None, None, None
            arr = np.array(model_cycles)
            mean_cycle = np.nanmean(arr, axis=0)
            std_cycle = np.nanstd(arr, axis=0)
            return mean_cycle, std_cycle, arr

        sections = [
            ('summer', 'Summer Storyline Classification', axes[0], axes[1],
             ['(a)', '(b)', '(c)', '(d)'], ['(e)', '(f)', '(g)', '(h)']),
            ('winter', 'Winter Storyline Classification', axes[2], axes[3],
             ['(i)', '(j)', '(k)', '(l)'], ['(m)', '(n)', '(o)', '(p)'])
        ]

        var_configs = [
            ('pr_mean', 'pr_df', 'Precipitation', 'Precipitation (mm/day)', '$\\Delta$ Precipitation (mm/day)'),
            ('tas_mean', 'tas_df', 'Temperature', 'Temperature (°C)', '$\\Delta$ Temperature (°C)'),
            ('u850_mean', 'u850_df', 'U850 Wind Speed (u>0)', 'U850 Wind Speed (m/s)', '$\\Delta$ U850 Wind Speed (m/s)'),
            ('q_mean', 'q_df', 'Discharge', 'Discharge ($m^3/s$)', '$\\Delta$ Discharge ($m^3/s$)')
        ]

        for s_idx, (season_mode, s_title, row_cycles_axes, row_diff_axes, letters_cyc, letters_diff) in enumerate(sections):
            grp_key = 'grp_s' if season_mode == 'summer' else 'grp_w'
            hf_recs = [r for r in model_records if r[grp_key] == 'High-Freq.']
            lf_recs = [r for r in model_records if r[grp_key] == 'Low-Freq.']

            # Extract HF cluster event relative years across HF models
            all_hf_rel_years = sorted(list(set([ry for r in hf_recs for ry in r['cluster_rel_years']])))
            if not all_hf_rel_years:
                all_hf_rel_years = list(np.arange(-15, 16))

            hf_target_years = {r['name']: r['cluster_rel_years'] if r['cluster_rel_years'] else all_hf_rel_years for r in hf_recs}
            lf_target_years = {r['name']: all_hf_rel_years for r in lf_recs}

            # Collect 30Q10 low-flow event months across HF models
            hf_s_event_months = [m for r in hf_recs for m in r['s_event_months']]
            hf_w_event_months = [m for r in hf_recs for m in r['w_event_months']]

            # Subheader banners across the top of each row section
            row_cycles_axes[0].annotate(
                f"SECTION {s_idx+1}A: {s_title.upper()} — MEAN SEASONAL CYCLES",
                xy=(0.0, 1.28), xycoords='axes fraction', fontsize=11.5, weight='bold', color='#111111',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#e6f2ff' if s_idx==0 else '#e6ffe6', edgecolor='none', alpha=0.9)
            )
            row_diff_axes[0].annotate(
                f"SECTION {s_idx+1}B: {s_title.upper()} — ABSOLUTE DIFFERENCE (HIGH-FREQ. − LOW-FREQ.)",
                xy=(0.0, 1.28), xycoords='axes fraction', fontsize=11.5, weight='bold', color='#5c007a',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#f3e6ff', edgecolor='none', alpha=0.9)
            )

            for v_idx, (var_col, df_key, var_label, y_label_cyc, y_label_diff) in enumerate(var_configs):
                ax_cyc = row_cycles_axes[v_idx]
                ax_diff = row_diff_axes[v_idx]

                letter_cyc = letters_cyc[v_idx]
                letter_diff = letters_diff[v_idx]

                m_hf, std_hf, _ = calc_seasonal_cycle_for_models(hf_recs, hf_target_years, var_col, df_key)
                m_lf, std_lf, _ = calc_seasonal_cycle_for_models(lf_recs, lf_target_years, var_col, df_key)

                # 1. Plot Seasonal Cycles
                if m_hf is not None:
                    ax_cyc.plot(month_indices, m_hf, color=high_col, linewidth=2.4, marker='o', markersize=5, label=f'High-Freq. ({len(hf_recs)} models)')
                    ax_cyc.fill_between(month_indices, m_hf - std_hf, m_hf + std_hf, color=high_col, alpha=0.18)

                if m_lf is not None:
                    ax_cyc.plot(month_indices, m_lf, color=low_col, linewidth=2.4, marker='s', markersize=5, linestyle='--', label=f'Low-Freq. ({len(lf_recs)} models)')
                    ax_cyc.fill_between(month_indices, m_lf - std_lf, m_lf + std_lf, color=low_col, alpha=0.18)

                # Plot 30Q10 Low-Flow Event Timing Scatter Dots along bottom of x-axis
                y_min_c, y_max_c = ax_cyc.get_ylim()
                y_rng_c = y_max_c - y_min_c
                y_pos_s = y_min_c + y_rng_c * 0.05
                y_pos_w = y_min_c + y_rng_c * 0.09

                np.random.seed(42)
                if hf_s_event_months:
                    jit_s = np.random.uniform(-y_rng_c * 0.012, y_rng_c * 0.012, size=len(hf_s_event_months))
                    ax_cyc.scatter(hf_s_event_months, y_pos_s + jit_s, color='#d62728', marker='o', s=32,
                                   edgecolor='black', linewidth=0.5, zorder=6, alpha=0.85, label='Summer 30Q10 Event')

                if hf_w_event_months:
                    jit_w = np.random.uniform(-y_rng_c * 0.012, y_rng_c * 0.012, size=len(hf_w_event_months))
                    ax_cyc.scatter(hf_w_event_months, y_pos_w + jit_w, color='#1f77b4', marker='o', s=32,
                                   edgecolor='black', linewidth=0.5, zorder=6, alpha=0.85, label='Winter 30Q10 Event')

                ax_cyc.set_xticks(month_indices)
                ax_cyc.set_xticklabels(month_names, fontsize=9.5, weight='bold')
                ax_cyc.tick_params(axis='x', labelbottom=True, labelsize=9.5)
                ax_cyc.set_xlabel("Month", fontsize=9.5, weight='bold')
                ax_cyc.set_ylabel(y_label_cyc, fontsize=9.5, weight='bold')
                ax_cyc.set_title(f"{letter_cyc} {var_label}", fontsize=11.0, weight='bold', loc='left', pad=6)
                ax_cyc.grid(True, linestyle=':', alpha=0.6)
                ax_cyc.legend(loc='best', fontsize=8.0, frameon=True)

                # 2. Plot Absolute Difference (High - Low)
                if m_hf is not None and m_lf is not None:
                    diff_vals = m_hf - m_lf
                    ax_diff.axhline(0, color='gray', linestyle='--', linewidth=1.1, alpha=0.7)
                    ax_diff.plot(month_indices, diff_vals, color=diff_col, linewidth=2.4, marker='d', markersize=5, label='Abs. Diff. (High − Low)')
                    
                    ax_diff.fill_between(month_indices, 0, diff_vals, where=(diff_vals >= 0), color='#d7191c', alpha=0.22, interpolate=True)
                    ax_diff.fill_between(month_indices, 0, diff_vals, where=(diff_vals < 0), color='#2b83ba', alpha=0.22, interpolate=True)

                y_min_d, y_max_d = ax_diff.get_ylim()
                y_rng_d = y_max_d - y_min_d
                y_pos_sd = y_min_d + y_rng_d * 0.05
                y_pos_wd = y_min_d + y_rng_d * 0.09

                if hf_s_event_months:
                    jit_sd = np.random.uniform(-y_rng_d * 0.012, y_rng_d * 0.012, size=len(hf_s_event_months))
                    ax_diff.scatter(hf_s_event_months, y_pos_sd + jit_sd, color='#d62728', marker='o', s=32,
                                    edgecolor='black', linewidth=0.5, zorder=6, alpha=0.85, label='Summer 30Q10 Event')

                if hf_w_event_months:
                    jit_wd = np.random.uniform(-y_rng_d * 0.012, y_rng_d * 0.012, size=len(hf_w_event_months))
                    ax_diff.scatter(hf_w_event_months, y_pos_wd + jit_wd, color='#1f77b4', marker='o', s=32,
                                    edgecolor='black', linewidth=0.5, zorder=6, alpha=0.85, label='Winter 30Q10 Event')

                ax_diff.set_xticks(month_indices)
                ax_diff.set_xticklabels(month_names, fontsize=9.5, weight='bold')
                ax_diff.tick_params(axis='x', labelbottom=True, labelsize=9.5)
                ax_diff.set_xlabel("Month", fontsize=9.5, weight='bold')
                ax_diff.set_ylabel(y_label_diff, fontsize=9.5, weight='bold')
                ax_diff.set_title(f"{letter_diff} $\\Delta$ {var_label} (High − Low)", fontsize=11.0, weight='bold', loc='left', pad=6)
                ax_diff.grid(True, linestyle=':', alpha=0.6)
                ax_diff.legend(loc='best', fontsize=8.0, frameon=True)

        fig.suptitle(
            f"Final Figure 10: GWL +{target_gwl:.1f}°C Seasonal Cycles & Absolute Differences ({Visualizer._format_scenario_title(scenario)})\n"
            f"Comparing High- vs. Low-Frequency Storylines Filtered for Low-Flow Cluster Event Years (30Q10 Event Timing Dots)",
            fontsize=13.0, weight='bold', y=0.996
        )

        fig.tight_layout(rect=(0, 0.01, 1, 0.97))
        fig.subplots_adjust(hspace=0.58, wspace=0.30)

        plt.savefig(filepath_png, dpi=300, bbox_inches='tight')
        plt.savefig(filepath_pdf, bbox_inches='tight')
        plt.close(fig)
        logging.info(f"Successfully generated 16-Panel Final Figure 10: {filepath_png} and {filepath_pdf}")





