"""
Visualization module for climate analysis results.

This module contains the Visualizer class, which bundles all plotting
functions. Each static method is responsible for creating a specific
figure, such as time series plots, regression maps, or correlation
matrices. It separates the analysis logic from the presentation.
"""
import os
import logging
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.stats import chi2
import json
import seaborn as sns
import traceback

# Import local modules
from storyline import StorylineAnalyzer 
from config import Config
from stats_analyzer import StatsAnalyzer
from data_processing import DataProcessor


class Visualizer:
    """A collection of static methods for plotting climate analysis results."""

    @staticmethod
    def ensure_plot_dir_exists():
        """Ensure the plot directory from the config exists."""
        if not os.path.exists(Config.PLOT_DIR):
            os.makedirs(Config.PLOT_DIR, exist_ok=True)
            logging.info(f"Created plot directory: {Config.PLOT_DIR}")

    @staticmethod
    def plot_regression_map(ax, slopes, p_values, lons, lats, title, box_coords,
                          season_label, variable, ua_seasonal_mean=None,
                          show_jet_boxes=False, significance_level=0.05, std_dev_predictor=None,
                          stipple_skip=None, dataset_key=None): # MODIFIED: Added dataset_key
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
        
        ax.set_title(f"{title}\n{season_label}", fontsize=10)
        return cf, label

    @staticmethod
    def plot_regression_analysis(all_season_data, dataset_key):
        """Creates a panel plot for regression maps (U850 vs PR/TAS indices)."""
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
        # MODIFIED: Pass dataset_key to the plotting function
        cf_pr, label_pr = Visualizer.plot_regression_map(ax_pr_winter, winter_pr_data.get('slopes_pr'), winter_pr_data.get('p_values_pr'), winter_pr_data.get('lons'), winter_pr_data.get('lats'), f"{dataset_key}: U850 vs PR Box Index", box_coords, "DJF", 'pr', ua_seasonal_mean=winter_pr_data.get('ua850_mean'), std_dev_predictor=winter_pr_data.get('std_dev_pr'), dataset_key=dataset_key)

        summer_pr_data = all_season_data['Summer']
        ax_pr_summer = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
        # MODIFIED: Pass dataset_key to the plotting function
        Visualizer.plot_regression_map(ax_pr_summer, summer_pr_data.get('slopes_pr'), summer_pr_data.get('p_values_pr'), summer_pr_data.get('lons'), summer_pr_data.get('lats'), f"{dataset_key}: U850 vs PR Box Index", box_coords, "JJA", 'pr', ua_seasonal_mean=summer_pr_data.get('ua850_mean'), std_dev_predictor=summer_pr_data.get('std_dev_pr'), dataset_key=dataset_key)

        if cf_pr:
            cax_pr = fig.add_subplot(gs[0, 2]); fig.colorbar(cf_pr, cax=cax_pr, extend='both', label=label_pr)

        # Plot TAS panels
        winter_tas_data = all_season_data['Winter']
        ax_tas_winter = fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree())
        # MODIFIED: Pass dataset_key to the plotting function
        cf_tas, label_tas = Visualizer.plot_regression_map(ax_tas_winter, winter_tas_data.get('slopes_tas'), winter_tas_data.get('p_values_tas'), winter_tas_data.get('lons'), winter_tas_data.get('lats'), f"{dataset_key}: U850 vs TAS Box Index", box_coords, "DJF", 'tas', ua_seasonal_mean=winter_tas_data.get('ua850_mean'), std_dev_predictor=winter_tas_data.get('std_dev_tas'), dataset_key=dataset_key)

        summer_tas_data = all_season_data['Summer']
        ax_tas_summer = fig.add_subplot(gs[1, 1], projection=ccrs.PlateCarree())
        # MODIFIED: Pass dataset_key to the plotting function
        Visualizer.plot_regression_map(ax_tas_summer, summer_tas_data.get('slopes_tas'), summer_tas_data.get('p_values_tas'), summer_tas_data.get('lons'), summer_tas_data.get('lats'), f"{dataset_key}: U850 vs TAS Box Index", box_coords, "JJA", 'tas', ua_seasonal_mean=summer_tas_data.get('ua850_mean'), std_dev_predictor=summer_tas_data.get('std_dev_tas'), dataset_key=dataset_key)

        if cf_tas:
            cax_tas = fig.add_subplot(gs[1, 2]); fig.colorbar(cf_tas, cax=cax_tas, extend='both', label=label_tas)
            
        plt.suptitle(f"{dataset_key}: U850 Regression onto Box Climate Indices (Detrended, Normalized Predictors)", fontsize=14, weight='bold')
        fig.tight_layout(rect=[0, 0, 0.95, 0.95])
        filename = os.path.join(Config.PLOT_DIR, f'regression_maps_norm_{dataset_key}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)

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
            
            analysis_box_coords = (Config.BOX_LON_MIN, Config.BOX_LON_MAX, 
                                   Config.BOX_LAT_MIN, Config.BOX_LAT_MAX)

            # --- Data for 20CRv3 ---
            data_20crv3 = correlation_data_20crv3.get(key)
            ax1 = fig.add_subplot(gs[row_idx, 0], projection=ccrs.PlateCarree())
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
            ax2 = fig.add_subplot(gs[row_idx, 1], projection=ccrs.PlateCarree())
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
            
            fig.colorbar(cf, cax=cax, extend='both', label=final_label)

            row_idx += 1
        
        plt.suptitle(f"Correlation Maps of Jet Variations and Climate ({season}, Detrended)", fontsize=16, weight='bold')
        fig.tight_layout(rect=[0, 0, 0.95, 0.96])
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
            '10-90th Percentile Spread',
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
            p10 = [np.percentile(d, 10) if d else np.nan for d in delta_values_per_gwl]
            p90 = [np.percentile(d, 90) if d else np.nan for d in delta_values_per_gwl]
            
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
                            markersize=style['s'] / 7, # Adjust markersize, as 's' in scatter scales differently
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

        fig.tight_layout(rect=[0, 0.06, 1, 0.95])
        
        # Use the scenario argument for a dynamic title and filename
        fig.suptitle(f"CMIP6 Projected Jet Changes vs. Global Warming Level ({scenario.upper()})", fontsize=16, weight='bold')
        if filename is None:
            filename = f"cmip6_jet_changes_vs_gwl_{scenario}.png"
        filepath = os.path.join(Config.PLOT_DIR, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)

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
            
            analysis_box_coords = (Config.BOX_LON_MIN, Config.BOX_LON_MAX, 
                                Config.BOX_LAT_MIN, Config.BOX_LAT_MAX)

            # --- Subplot for 20CRv3 ---
            data_20crv3 = impact_data_20crv3.get(key)
            ax1 = fig.add_subplot(gs[row_idx, 0], projection=ccrs.PlateCarree())
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
            ax2 = fig.add_subplot(gs[row_idx, 1], projection=ccrs.PlateCarree())
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
            if cf:
                cax = fig.add_subplot(gs[row_idx, 2])
                final_label = config['base_label']
                std_dev_val = data_era5.get('std_dev_predictor') if data_era5 and data_era5.get('std_dev_predictor') is not None else (data_20crv3.get('std_dev_predictor') if data_20crv3 else None)

                if std_dev_val is not None and not np.isnan(std_dev_val):
                    unit = "m/s" if "Speed" in config['title'] else "°Lat"
                    final_label += f'\n(1 std. dev. = {std_dev_val:.2f} {unit})'
                
                fig.colorbar(cf, cax=cax, extend='both', label=final_label)

            row_idx += 1
        
        plt.suptitle(f"Regression of Climate Variables on Jet Variations ({season}, Detrended)", fontsize=16, weight='bold')
        fig.tight_layout(rect=[0, 0, 0.95, 0.96])
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
            ax = config['ax']
            season_lower = config['season'].lower()
            
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
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        filename = os.path.join(Config.PLOT_DIR, f'{season_lower}_correlations_comparison_detrended.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logging.info(f"Korrelations-Zeitreihenplot für {season} gespeichert unter: {filename}")

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
        fig.tight_layout(rect=[0, 0.05, 1, 0.95])
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
            ax = axs[config['ax_idx']]
            season_data = correlation_data.get(config['season'])
            
            ax.set_title(f"{config['title']}, {window_size}-yr mean")
            ax.grid(True, linestyle=':', alpha=0.6)

            if not season_data:
                ax.text(0.5, 0.5, "Data Not Available", transform=ax.transAxes, ha='center', va='center')
                continue

            plotted_amo = False
            for dataset_key in [Config.DATASET_20CRV3, Config.DATASET_ERA5]:
                # The data structure from analyze_amo_jet_correlations is {'20CRv3': {'speed': {...}, 'latitude': {...}}}
                jet_data = season_data.get(dataset_key, {}).get(jet_type_map[config['jet_type']])
                
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
                    label = f"{dataset_key} Jet {config['jet_type'].capitalize()} ({window_size}yr)"
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
        fig.tight_layout(rect=[0, 0.03, 1, 0.96])
        filename = os.path.join(Config.PLOT_DIR, f'amo_jet_correlations_comparison_rolling_{window_size}yr.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logging.info(f"Saved AMO vs Jet correlation comparison plot to {filename}")

    @staticmethod
    def plot_climate_projection_timeseries(cmip6_plot_data, reanalysis_plot_data, config, filename="climate_indices_evolution.png"):
        """
        Plots CMIP6 and Reanalysis changes over time, showing the evolution of key climate indices.
        MODIFIZIERT, um ein 2x3-Raster mit globaler Temperatur und allen vier Jet-Indizes zu erstellen.
        """
        filename = "climate_indices_evolution.png"
        logging.info(f"Plotting climate projection timeseries comparison to {filename} (2x3 layout)...")
        Visualizer.ensure_plot_dir_exists()

        # --- MODIFIKATION START ---
        # Ändern des Layouts auf 2x3 Subplots
        fig, axs = plt.subplots(2, 3, figsize=(21, 12), sharex=True)
        axs_flat = axs.flatten()
        # --- MODIFIKATION ENDE ---

        # --- (a) Globale Temperatur-Anomalie ---
        ax_a = axs_flat[0]
        if cmip6_plot_data['Global_Tas']['members']:
            for member_tas in cmip6_plot_data['Global_Tas']['members']:
                ax_a.plot(member_tas.year, member_tas, color='grey', alpha=0.3, linewidth=0.7)
        if cmip6_plot_data['Global_Tas']['mmm'] is not None:
            ax_a.plot(cmip6_plot_data['Global_Tas']['mmm'].year, cmip6_plot_data['Global_Tas']['mmm'], color='black', linewidth=2.5, label='CMIP6 MMM')
        
        # --- START: NEUER PLOT-CODE ---
        # Zeichne Reanalyse-Daten für globale Temperatur
        if reanalysis_plot_data.get('Global_Tas'):
            if reanalysis_plot_data['Global_Tas'].get('20CRv3') is not None:
                reanalysis_20crv3_tas = reanalysis_plot_data['Global_Tas']['20CRv3']
                ax_a.plot(reanalysis_20crv3_tas.year, reanalysis_20crv3_tas, color='darkorange', linewidth=2, label='20CRv3')
            if reanalysis_plot_data['Global_Tas'].get('ERA5') is not None:
                reanalysis_era5_tas = reanalysis_plot_data['Global_Tas']['ERA5']
                ax_a.plot(reanalysis_era5_tas.year, reanalysis_era5_tas, color='purple', linewidth=2, label='ERA5')
        # --- ENDE: NEUER PLOT-CODE ---
        
        ax_a.set_title('(a) Global Temperature Change', fontsize=12)
        ax_a.set_ylabel(f'Temp. Anomaly (°C rel. to {config.CMIP6_PRE_INDUSTRIAL_REF_START}-{config.CMIP6_PRE_INDUSTRIAL_REF_END})', fontsize=10)
        ax_a.legend(fontsize=9)

        # --- Konfiguration für die vier Jet-Index-Plots ---
        plot_configs = [
            {'key': 'JJA_JetLat', 'ax': axs_flat[1], 'title': '(b) Summer (JJA) Jet Latitude Change', 'ylabel': 'Latitude Anomaly (°)'},
            {'key': 'JJA_JetSpeed', 'ax': axs_flat[2], 'title': '(c) Summer (JJA) Jet Speed Change', 'ylabel': 'Speed Anomaly (m/s)'},
            {'key': 'DJF_JetLat', 'ax': axs_flat[4], 'title': '(e) Winter (DJF) Jet Latitude Change', 'ylabel': 'Latitude Anomaly (°)'},
            {'key': 'DJF_JetSpeed', 'ax': axs_flat[5], 'title': '(f) Winter (DJF) Jet Speed Change', 'ylabel': 'Speed Anomaly (m/s)'},
        ]

        for p_config in plot_configs:
            ax = p_config['ax']
            key = p_config['key']

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

            ax.set_title(p_config['title'], fontsize=12)
            ax.set_ylabel(p_config['ylabel'], fontsize=10)
            ax.legend(fontsize=9)

        # --- Finale Formatierung für alle Achsen ---
        for ax in axs_flat:
            ax.grid(True, linestyle=':', alpha=0.7)
            ax.set_xlim(1850, 2100)
            ax.axhline(0, color='black', linewidth=0.5)
        
        # X-Achsen-Label nur für die untere Reihe
        for ax in axs[1, :]:
                ax.set_xlabel('Year', fontsize=10)

        # --- MODIFIKATION START ---
        # Leeren Subplot in der Mitte ausblenden
        axs_flat[3].axis('off')
        # --- MODIFIKATION ENDE ---

        fig.suptitle('Evolution of Key Climate Indices (20-Year Rolling Mean Anomaly)', fontsize=16, weight='bold')
        # [KORREKTUR] fig.tight_layout() anstelle von plt.tight_layout()
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        filepath = os.path.join(config.PLOT_DIR, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logging.info(f"Saved climate projection timeseries plot to {filepath}")
        
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
        if not np.isnan(slope_cmip6):
            x_fit = np.array(ax.get_xlim())
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
        
        fig.tight_layout(rect=[0, 0, 1, 0.94])
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
        ax.scatter(mmm_x, mmm_y, color='red', marker='X', s=120, zorder=10, edgecolor='black', linewidth=1.5, label='Multi-Model Mean')
        
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

        fig.tight_layout(rect=[0, 0, 1, 0.95], h_pad=5.0)
        
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
        ax_djf = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
        djf_data = u850_change_results['DJF']
        if djf_data and djf_data.get('u850_change_mmm') is not None:
            cf_djf, _ = Visualizer.plot_u850_change_map(
                ax_djf,
                djf_data['u850_change_mmm'].data,
                djf_data['u850_historical_mean_mmm'].data,
                djf_data['u850_change_mmm'].lon.values,
                djf_data['u850_change_mmm'].lat.values,
                "CMIP6 MMM U850 Change", "DJF"
            )
            if cf_djf: cf_ref = cf_djf
        else:
            ax_djf.text(0.5,0.5, "DJF Data Missing", transform=ax_djf.transAxes, ha='center', va='center')
            ax_djf.set_title("CMIP6 MMM U850 Change\nDJF\n(Data Missing)")

        # JJA Plot
        ax_jja = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
        jja_data = u850_change_results['JJA']
        if jja_data and jja_data.get('u850_change_mmm') is not None:
            cf_jja, label_jja = Visualizer.plot_u850_change_map(
                ax_jja,
                jja_data['u850_change_mmm'].data,
                jja_data['u850_historical_mean_mmm'].data,
                jja_data['u850_change_mmm'].lon.values,
                jja_data['u850_change_mmm'].lat.values,
                "CMIP6 MMM U850 Change", "JJA"
            )
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
        fig.tight_layout(rect=[0, 0, 0.95, 0.95])
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

            all_row_data = model_slopes_hist
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
        fig.tight_layout(rect=[0, 0.08, 1, 0.93])
        
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
        fig.tight_layout(rect=[0, 0, 1, 0.96])
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
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        
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
        fig.tight_layout(rect=[0, 0, 0.95, 0.96])
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
        
        fig.tight_layout(rect=[0, 0.02, 1, 0.93])
        
        filename = os.path.join(config.PLOT_DIR, "storyline_impacts_summary_vertical.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logging.info(f"Saved vertical storyline impacts summary plot to {filename}")

    @staticmethod
    # --- MODIFIKATION: Argument threshold_data hinzugefügt ---
    def plot_storyline_impact_barchart_with_discharge(cmip6_results, threshold_data, discharge_data_historical, reanalysis_data, config, scenario, storyline_correlations=None):
        """
        Creates a 7x2 plot to visualize storyline impacts for Temp, Precip, SPEI,
        seasonal Discharge, and lagged monthly Discharge.
        Uses boxplots with stripplots where model data is available (tas, pr, discharge),
        and a bar chart fallback for derived metrics (SPEI).
        MODIFIED: Accepts threshold_data containing percentile and other thresholds.
        MODIFIED: Draws lines for LNWL, 1% Low-Flow, and 99% High-Flow.
        """
        logging.info(f"Plotting EXTENDED storyline impacts (7x2 grid with percentile thresholds) for {scenario}...")
        Visualizer.ensure_plot_dir_exists()

        if not cmip6_results:
            logging.warning(f"Cannot plot impacts for {scenario}: Input data is empty.")
            return
        # --- MODIFIKATION: Prüfung für threshold_data ---
        if not threshold_data:
            logging.warning(f"Cannot plot impacts for {scenario}: Threshold data is missing.")
            return

        # --- 1. Datenextraktion und -aufbereitung (wie zuvor) ---
        all_deltas = cmip6_results.get('all_individual_model_deltas_for_plot', {})
        classification = cmip6_results.get('storyline_classification_2d', {})

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
            'DJF_tas', 'JJA_tas', 'DJF_pr', 'JJA_pr',
            'DJF_discharge', 'JJA_discharge',
            'Mar_discharge', 'Apr_discharge', 'May_discharge',
            'Sep_discharge', 'Oct_discharge', 'Nov_discharge'
        ]

        for gwl in gwls_to_plot:
            storylines_in_gwl = classification.get(gwl, {})
            for storyline_key, model_list in storylines_in_gwl.items():
                if not model_list: continue
                season_prefix = storyline_key.split('_')[0]
                storyline_name = storyline_key.replace(f'{season_prefix}_', '')
                if season_prefix == 'DJF':
                    relevant_impact_keys = [k for k in impact_keys_for_boxplot if 'DJF' in k or k.startswith('Mar') or k.startswith('Apr') or k.startswith('May')]
                else: # JJA
                    relevant_impact_keys = [k for k in impact_keys_for_boxplot if 'JJA' in k or k.startswith('Sep') or k.startswith('Oct') or k.startswith('Nov')]
                for impact_key in relevant_impact_keys:
                    if impact_key not in impact_keys_for_boxplot: continue
                    for model_run_key in model_list:
                        delta_val = all_deltas.get(impact_key, {}).get(gwl, {}).get(model_run_key)
                        if delta_val is not None and np.isfinite(delta_val):
                            plot_data_list.append({
                                'gwl': f'+{gwl}°C', 'storyline': storyline_name,
                                'impact_key': impact_key, 'value': delta_val
                            })
        df_plot = pd.DataFrame(plot_data_list)

        # --- 2. Plot-Setup (7x2 Grid, wie zuvor) ---
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, axs = plt.subplots(7, 2, figsize=(20, 38))
        gwl_colors = {f'+{gwls_to_plot[0]}°C': '#4575b4', f'+{gwls_to_plot[1]}°C': '#d73027'}
        
        plot_grid = {
            (0, 0): {'key': 'DJF_tas', 'title': 'a) Winter (DJF) Temperature'},
            (0, 1): {'key': 'JJA_tas', 'title': 'b) Summer (JJA) Temperature'},
            (1, 0): {'key': 'DJF_pr', 'title': 'c) Winter (DJF) Precipitation'},
            (1, 1): {'key': 'JJA_pr', 'title': 'd) Summer (JJA) Precipitation'},
            (2, 0): {'key': 'DJF_spei', 'title': 'e) Winter (DJF) SPEI-4'},
            (2, 1): {'key': 'JJA_spei', 'title': 'f) Summer (JJA) SPEI-4'},
            (3, 0): {'key': 'DJF_discharge', 'title': 'g) Winter (DJF) Discharge'},
            (3, 1): {'key': 'JJA_discharge', 'title': 'h) Summer (JJA) Discharge'},
            (4, 0): {'key': 'Mar_discharge', 'title': 'i) March Discharge (lagged)'},
            (4, 1): {'key': 'Sep_discharge', 'title': 'j) September Discharge (lagged)'},
            (5, 0): {'key': 'Apr_discharge', 'title': 'k) April Discharge (lagged)'},
            (5, 1): {'key': 'Oct_discharge', 'title': 'l) October Discharge (lagged)'},
            (6, 0): {'key': 'May_discharge', 'title': 'm) May Discharge (lagged)'},
            (6, 1): {'key': 'Nov_discharge', 'title': 'n) November Discharge (lagged)'}
        }
        
        storyline_display_order = [
            'MMM', 'Slow Jet & Northward Shift', 'Fast Jet & Northward Shift',
            'Slow Jet & Southward Shift', 'Fast Jet & Southward Shift',
        ]
        
        # --- 3. Plotting-Schleife (wie zuvor, mit Anpassungen für Schwellenlinien) ---
        for (row, col), plot_info in plot_grid.items():
            ax = axs[row, col]
            impact_key = plot_info['key']
            is_spei_plot = 'spei' in impact_key

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
                df_spei = pd.pivot_table(pd.DataFrame(df_spei_list),
                                        index='storyline', columns='gwl', values='value').reindex(storyline_display_order).dropna(how='all')
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
                    ax.legend(title="GWL")

            # Boxplots + Stripplots für tas, pr, discharge (wie zuvor)
            else:
                if df_plot.empty:
                    ax.text(0.5, 0.5, "Data N/A", ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(plot_info['title'], loc='left', fontsize=14, weight='bold')
                    continue
                data_subset = df_plot[df_plot['impact_key'] == impact_key]
                if data_subset.empty:
                    ax.text(0.5, 0.5, "Data N/A", ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(plot_info['title'], loc='left', fontsize=14, weight='bold')
                    continue

                sns.boxplot(data=data_subset, x='storyline', y='value', hue='gwl', ax=ax,
                            order=storyline_display_order, palette=gwl_colors,
                            linewidth=1.2, showfliers=False, boxprops={'alpha': 0.7})
                sns.stripplot(data=data_subset, x='storyline', y='value', hue='gwl', ax=ax,
                            order=storyline_display_order, palette=gwl_colors,
                            dodge=True, jitter=0.15, size=4, edgecolor='gray', linewidth=0.5)
                handles, labels = ax.get_legend_handles_labels()
                unique_handles = []; unique_labels = []; seen_labels = set()
                for h, l in zip(handles, labels):
                    if l not in seen_labels and l in gwl_colors:
                         unique_handles.append(h); unique_labels.append(l); seen_labels.add(l)
                if unique_handles: ax.legend(unique_handles, unique_labels, title='GWL')
                else:
                     if ax.get_legend() is not None: ax.get_legend().remove()

            # --- 4. Formatierung (wie zuvor, mit Anpassung für Schwellenlinien) ---
            ax.set_title(plot_info['title'], loc='left', fontsize=14, weight='bold')
            ax.axhline(0, color='black', linestyle='-', linewidth=0.8, zorder=1)
            ax.set_xlabel('')
            
            unit = ''
            if 'tas' in impact_key: unit = '(°C)'
            elif 'pr' in impact_key: unit = '(%)'
            elif 'discharge' in impact_key: unit = '(m³/s)'
            elif is_spei_plot: unit = '(Std. Dev.)'
            
            if col == 0: ax.set_ylabel(f'Projected Change {unit}', fontsize=12)
            else: ax.set_ylabel('')
            
            if row < 6 or is_spei_plot:
                ax.set_xticklabels([])
            else:
                xtick_labels = [name.replace(' & ', ' &\n').replace(' (MMM)','') for name in storyline_display_order]
                ax.set_xticklabels(xtick_labels, rotation=45, ha="right", fontsize=11)
            
            # --- START: MODIFIKATION (Threshold Lines) ---
            if 'discharge' in impact_key:
                # Get the specific thresholds calculated for THIS impact_key (e.g., 'Mar_discharge')
                key_thresholds = threshold_data.get(impact_key, {})
                
                # We need the historical mean *for this specific key* to plot change relative to 0
                # Retrieve it by recalculating from the historical discharge data passed in
                hist_mean_specific = np.nan # Default
                hist_discharge_monthly = DataProcessor.assign_season_to_dataarray(discharge_data_historical.get('monthly_historical_da'))
                if hist_discharge_monthly is not None:
                     if impact_key in ['DJF_discharge', 'JJA_discharge']:
                         season_name = 'Winter' if 'DJF' in impact_key else 'Summer'
                         hist_ts_mean = DataProcessor.calculate_seasonal_means(hist_discharge_monthly)
                         if hist_ts_mean is not None:
                             hist_ts_filtered = DataProcessor.filter_by_season(hist_ts_mean, season_name)
                             if hist_ts_filtered is not None:
                                 hist_mean_specific = hist_ts_filtered.mean().item()
                     else: # Monthly key
                         month_num = int(impact_key[0:impact_key.find('_')].replace('Mar','3').replace('Apr','4').replace('May','5').replace('Sep','9').replace('Oct','10').replace('Nov','11'))
                         hist_ts_monthly_filtered = hist_discharge_monthly.where(hist_discharge_monthly.time.dt.month == month_num, drop=True)
                         if hist_ts_monthly_filtered is not None and hist_ts_monthly_filtered.size > 0:
                             hist_mean_specific = hist_ts_monthly_filtered.mean().item()

                if not np.isnan(hist_mean_specific):
                    # LNWL (fixed)
                    lnwl_event_name = [k for k in key_thresholds if 'LNWL' in k]
                    if lnwl_event_name:
                         lnwl_val = key_thresholds[lnwl_event_name[0]].get('val')
                         if lnwl_val is not None:
                             ax.axhline(lnwl_val - hist_mean_specific, color='red', linestyle='-.', linewidth=2.5, zorder=5)

                    # Extreme Low-Flow (1%)
                    low_extreme_event_name = [k for k in key_thresholds if '<1%' in k]
                    if low_extreme_event_name:
                         low_extreme_val = key_thresholds[low_extreme_event_name[0]].get('val')
                         if low_extreme_val is not None:
                             ax.axhline(low_extreme_val - hist_mean_specific, color='darkviolet', linestyle=(0, (3, 5)), linewidth=2.5, zorder=5) # Dashed-dot

                    # Extreme High-Flow (99%)
                    high_extreme_event_name = [k for k in key_thresholds if '>99%' in k]
                    if high_extreme_event_name:
                         high_extreme_val = key_thresholds[high_extreme_event_name[0]].get('val')
                         if high_extreme_val is not None:
                             ax.axhline(high_extreme_val - hist_mean_specific, color='deepskyblue', linestyle=(0, (5, 5)), linewidth=2.5, zorder=5) # Dashed
            # --- ENDE: MODIFIKATION ---

        # --- 5. Finale Formatierung der gesamten Figur ---
        handles, labels = axs[0, 0].get_legend_handles_labels() # Get GWL labels
        unique_labels_map = {}
        for l, h in zip(labels, handles):
            if l in gwl_colors and l not in unique_labels_map:
                unique_labels_map[l] = h
        
        # Add threshold lines to legend dynamically based on what was plotted
        # Need to access one of the threshold dicts to get names and values (use DJF as example)
        djf_thresholds = threshold_data.get('DJF_discharge', {})
        
        lnwl_key = [k for k in djf_thresholds if 'LNWL' in k]
        if lnwl_key:
             lnwl_val_str = f"{djf_thresholds[lnwl_key[0]].get('val', '?'):.0f}"
             unique_labels_map[f'Low Navigable (LNWL, ~{lnwl_val_str} m³/s)'] = plt.Line2D([0], [0], color='red', linestyle='-.', linewidth=2.5)

        low_extreme_key = [k for k in djf_thresholds if '<1%' in k]
        if low_extreme_key:
             low_val_str = f"{djf_thresholds[low_extreme_key[0]].get('val', '?'):.0f}"
             unique_labels_map[f'Extreme Low-Flow (<1%, ~{low_val_str} m³/s)'] = plt.Line2D([0], [0], color='darkviolet', linestyle=(0, (3, 5)), linewidth=2.5)

        high_extreme_key = [k for k in djf_thresholds if '>99%' in k]
        if high_extreme_key:
             high_val_str = f"{djf_thresholds[high_extreme_key[0]].get('val', '?'):.0f}"
             unique_labels_map[f'Extreme High-Flow (>99%, ~{high_val_str} m³/s)'] = plt.Line2D([0], [0], color='deepskyblue', linestyle=(0, (5, 5)), linewidth=2.5)

        # Place legend below the plot
        fig.legend(unique_labels_map.values(), unique_labels_map.keys(), loc='lower center', bbox_to_anchor=(0.5, 0.06), ncol=3, fontsize=12, frameon=False)
        
        main_title = f"Projected Changes for Jet Stream Storylines ({scenario.upper()})"
        ref_period_text = f"Changes relative to the {config.CMIP6_ANOMALY_REF_START}-{config.CMIP6_ANOMALY_REF_END} reference period"
        fig.suptitle(f"{main_title}\n{ref_period_text}", fontsize=18, weight='bold', y=0.99)
        
        # Adjust layout
        plt.subplots_adjust(left=0.07, right=0.98, top=0.965, bottom=0.12, hspace=0.45, wspace=0.15)
        
        # Keep the original filename
        filename = os.path.join(config.PLOT_DIR, f"storyline_impacts_summary_4x2_boxplots_{scenario}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logging.info(f"Saved EXTENDED 7x2 storyline impacts boxplot summary (with percentile lines) for {scenario} to {filename}")
        
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
            ax.bar_label(container, fmt='%.2f%%', fontsize=9, padding=3)

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
                if hist_val_q:
                    # Add specific threshold value to title
                    title += f"\n(Threshold: {op} {hist_val_q:.0f} m³/s)"
                ax.set_title(title, fontsize=11, weight='bold')
                
                # Filter data for this subplot
                data_subset = df_plot_clipped[(df_plot_clipped['half_year'] == half_year) & (df_plot_clipped['event'] == event_key)]
                
                if not data_subset.empty:
                    # Plot vertical boxplots
                    sns.boxplot(data=data_subset, x='storyline', y='return_period', hue='gwl',
                                order=storyline_order, palette=gwl_colors,
                                ax=ax, linewidth=1.2, showfliers=False, orient='v',
                                boxprops={'alpha': 0.85})
                    sns.stripplot(data=data_subset, x='storyline', y='return_period', hue='gwl',
                                order=storyline_order, palette=gwl_colors,
                                ax=ax, dodge=True, jitter=0.15, size=4, 
                                edgecolor='gray', linewidth=0.5, alpha=0.9, orient='v')
                else:
                    ax.text(0.5, 0.5, "Data N/A", ha='center', va='center', transform=ax.transAxes)

                # Plot Historical Return Period as a horizontal line
                hist_period = threshold_data.get('hist_return_period')
                if hist_period is not None and np.isfinite(hist_period):
                    ax.axhline(y=hist_period, color='skyblue', linestyle='--', linewidth=3, zorder=5)

                # --- Axis Formatting ---
                ax.set_yscale('log')
                ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
                ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
                ax.set_yticks([1, 2, 5, 10, 20, 50, 100, 500, 1000])
                ax.set_ylim(bottom=0.8, top=max_return_period_for_plot * 2.5) # Y-lim

                ax.grid(axis='x', linestyle='none')
                ax.grid(axis='y', linestyle=':', which='both')
                
                ax.set_xticks(x_ticks)
                if row == num_rows - 1: # X-labels only on the bottom row
                    ax.set_xticklabels(x_tick_labels, rotation=45, ha='right', fontsize=9)
                else:
                    ax.set_xticklabels([])
                ax.set_xlabel('')

                # Add Row Titles (Half-Year) to the first column
                if col == 0:
                    season_title = "Winter\n(Dec - May)" if half_year == 'winter' else "Summer\n(Jun - Nov)"
                    ax.set_ylabel(f"{season_title}\n\nReturn Period (Years)", fontsize=11, weight='bold')
                else:
                    ax.set_ylabel('')

                if ax.get_legend() is not None: ax.get_legend().remove()
                
                # Add n=X/Y annotations
                for i, storyline in enumerate(storyline_order):
                    x_base = x_ticks[i]
                    for j, gwl in enumerate(gwls_to_plot):
                        x_offset = -0.2 + (j * 0.4) # Position for GWL bar
                        gwl_label = f'+{gwl}°C'
                        event_data_gwl = results['data'].get(gwl, {}).get(half_year, {}).get(storyline, {}).get(event_key)
                        if event_data_gwl and 'model_count_X' in event_data_gwl:
                            X, Y = event_data_gwl['model_count_X'], event_data_gwl['model_count_Y']
                            # Place text at the bottom of the plot
                            ax.text(x_base + x_offset, 0.02, f"n={X}/{Y}", transform=ax.get_xaxis_transform(),
                                    horizontalalignment='center', fontsize=7, weight='bold', color=gwl_colors[gwl_label],
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
        fig.tight_layout(rect=[0.05, bottom_margin, 0.98, 0.95], h_pad=2.5, w_pad=2.0)
        
        # Use a new filename to avoid overwriting the old plot
        filename = os.path.join(config.PLOT_DIR, f"storyline_discharge_return_period_BY_EVENT_{scenario}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logging.info(f"Saved new return period boxplot (by event) to {filename}")


    @staticmethod
    def plot_storyline_return_period_half_year(results, config, scenario):
        """
        Creates a plot showing the change in return periods, organized with
        Low-Flow and High-Flow sections, each having a Winter and Summer row.

        MODIFIED LAYOUT (v4.2 - Pooled GEV Plot, LNWL removed):
        - Creates the same 4-row layout as v4.1.
        - REPLACES Boxplots/Stripplots with ax.scatter() to show the
          single, robust, pooled GEV result for each storyline.
        - Places annotation (n=X/Y) on the right edge of the plot.
        - MODIFIED: The 'LNWL' event column has been removed from this plot.
        
        --- MODIFIKATION (User-Wunsch, 06.11.2025): ---
        - sharex=False: X-Achsen werden spaltenweise synchronisiert (alle 1Q10-Plots teilen sich eine Achse, etc.)
        - Verwendet eine Zwei-Durchlauf-Logik (ähnlich wie 'plot_storyline_lnwl_aggregation_comparison'),
          um zuerst die "idealen" Zoom-Limits für jede Spalte zu finden und sie dann anzuwenden.
        """
        if not results or not config or 'thresholds' not in results or 'data' not in results:
            logging.warning(f"Cannot plot return period change by event for {scenario}: Missing results or thresholds.")
            return

        logging.info(f"Plotting POOLED GEV return period (4-row, LNWL removed, SPALTEN-ZOOM) for {scenario}...")
        Visualizer.ensure_plot_dir_exists()
        
        gwls_to_plot = config.GLOBAL_WARMING_LEVELS
        
        # --- 1. Define Event and Plot Structure ---
        all_event_keys = list(results['thresholds']['winter'].keys())
        all_event_keys.extend(list(results['thresholds']['summer'].keys()))
        unique_event_keys = sorted(list(set(all_event_keys)))

        low_flow_events = sorted([k for k in unique_event_keys if results['thresholds']['winter'].get(k, {}).get('type') == 'low' or results['thresholds']['summer'].get(k, {}).get('type') == 'low'])
        high_flow_events = sorted([k for k in unique_event_keys if results['thresholds']['winter'].get(k, {}).get('type') == 'high' or results['thresholds']['summer'].get(k, {}).get('type') == 'high'])
        
        event_plot_order_keys_low = ['1Q10_low', '7Q10_low', '7Q50_low', '7Q100_low'] 
        event_plot_order_keys_high = ['1Q10_high', '7Q10_high', '7Q50_high', '7Q100_high']

        def get_ordered_events(base_list, available_keys):
            ordered_list = [key for key in base_list if key in available_keys]
            return ordered_list

        low_flow_events_ordered = get_ordered_events(event_plot_order_keys_low, low_flow_events)
        high_flow_events_ordered = get_ordered_events(event_plot_order_keys_high, high_flow_events)

        num_cols_low = len(low_flow_events_ordered)
        num_cols_high = len(high_flow_events_ordered)
        num_cols = max(num_cols_low, num_cols_high)
        
        num_rows = 4 # Row 0: Winter Low, Row 1: Summer Low, Row 2: Winter High, Row 3: Summer High
        
        if num_cols == 0:
            logging.warning(f"No valid EVA events found to plot for {scenario}.")
            return

        # --- MODIFIKATION: sharex=False ---
        fig, axs = plt.subplots(
            num_rows, num_cols, 
            figsize=(6 * num_cols, 22), 
            squeeze=False, 
            sharey=True, # Y-Achse (Storylines) wird geteilt
            sharex=False  # X-Achsen werden manuell pro Spalte gesteuert
        )
        plt.style.use('seaborn-v0_8-whitegrid')
        # --- ENDE MODIFIKATION ---

        # Y-Axis setup for Storylines
        storyline_order = [
            'MMM',
            'Slow Jet & Northward Shift',
            'Fast Jet & Northward Shift',
            'Slow Jet & Southward Shift',
            'Fast Jet & Southward Shift',
        ]
        
        num_storylines = len(storyline_order)
        y_limits = (num_storylines - 0.5, -0.5)
        y_ticks = np.arange(len(storyline_order))
        y_tick_labels = [s.replace(' & ', ' &\n') for s in storyline_order]

        gwl_colors = {f'+{gwls_to_plot[0]}°C': '#ff7f0e', f'+{gwls_to_plot[1]}°C': '#d62728'}
        gwl_markers = {f'+{gwls_to_plot[0]}°C': 'o', f'+{gwls_to_plot[1]}°C': 'X'}
        
        # --- 2. Prepare Data for Plotting (v4.0) ---
        # --- MODIFIKATION: Maximales Limit für Zoom entfernt (wird dynamisch) ---
        # max_return_period_for_plot = 150 
        
        plot_data_list = []
        for gwl in gwls_to_plot:
            gwl_label = f'+{gwl}°C'
            for half_year in ['winter', 'summer']:
                for storyline in storyline_order:
                    all_events_for_plotting = low_flow_events_ordered + high_flow_events_ordered
                    for event_key in all_events_for_plotting:
                        event_data = results['data'].get(gwl, {}).get(half_year, {}).get(storyline, {}).get(event_key)
                        if event_data and 'future_return_period_mean' in event_data:
                            period = event_data['future_return_period_mean']
                            
                            if np.isfinite(period):
                                plot_data_list.append({
                                    'half_year': half_year,
                                    'event': event_key,
                                    'storyline': storyline,
                                    'gwl': gwl_label,
                                    'return_period': period
                                })
        
        if not plot_data_list:
            logging.warning(f"No finite return period data (excl. LNWL) to plot for {scenario}.")
            plt.close(fig)
            return

        df_plot = pd.DataFrame(plot_data_list)
        # --- MODIFIKATION: Clipping entfernt ---
        # df_plot_clipped = df_plot[df_plot['return_period'] <= max_return_period_for_plot].copy()
        df_plot_clipped = df_plot.copy() # Wir verwenden jetzt alle Daten
        
        # --- MODIFIKATION: Speicher für die X-Achsen-Limits pro Spalte ---
        column_x_limits = {c: [] for c in range(num_cols)}

        # --- 3. Plotting Loop (Reorganized) - PASS 1 ---
        
        plot_blocks = [
            {'title': 'Low-Flow Events', 'events': low_flow_events_ordered, 'base_row': 0},
            {'title': 'High-Flow Events', 'events': high_flow_events_ordered, 'base_row': 2}
        ]
        
        for block in plot_blocks:
            base_row = block['base_row']
            event_list = block['events']
            
            for row_offset, half_year in enumerate(['winter', 'summer']):
                row = base_row + row_offset
                
                # --- MODIFIKATION: Verwende enumerate(event_list) für den Spaltenindex ---
                for col, event_key in enumerate(event_list):
                    ax = axs[row, col]
                    ax.set_ylim(y_limits)
                    ax.invert_yaxis()
                    
                    threshold_data = results.get('thresholds', {}).get(half_year, {}).get(event_key, {})
                    event_name = threshold_data.get('name', event_key).replace("Low-Flow", "Low Flow").replace("High-Flow", "High Flow")
                    hist_val_q = threshold_data.get('threshold_m3s')
                    op = '<' if threshold_data.get('type') == 'low' else '>'
                    
                    title = f"{event_name}" 
                    if hist_val_q and "(<" not in event_name and "(>" not in event_name and "m³/s" not in event_name:
                        title += f"\n(Threshold: {op} {hist_val_q:.0f} m³/s)"
                    ax.set_title(title, fontsize=11, weight='bold')

                    data_subset = df_plot_clipped[(df_plot_clipped['half_year'] == half_year) & (df_plot_clipped['event'] == event_key)]
                    
                    # --- MODIFIKATION: Liste zum Sammeln der Daten für die Achsen-Limits ---
                    all_data_for_lims = []

                    if not data_subset.empty:
                        for gwl_label, gwl_group in data_subset.groupby('gwl'):
                            y_values = gwl_group['storyline'].map(dict(zip(storyline_order, y_ticks)))
                            ax.scatter(
                                x=gwl_group['return_period'],
                                y=y_values,
                                color=gwl_colors[gwl_label],
                                marker=gwl_markers[gwl_label],
                                s=100, 
                                label=gwl_label,
                                edgecolor='black',
                                linewidth=0.5,
                                zorder=10
                            )
                        # --- MODIFIKATION: Daten für Limits sammeln ---
                        all_data_for_lims.extend(data_subset['return_period'].dropna().values)
                    
                    else:
                        ax.set_title(title, fontsize=11, weight='bold', color='gray') 
                        ax.text(0.5, 0.5, "Data N/A", ha='center', va='center', transform=ax.transAxes, color='gray')

                    hist_period = threshold_data.get('hist_return_period')
                    if hist_period is not None and np.isfinite(hist_period):
                        ax.axvline(x=hist_period, color='skyblue', linestyle='--', linewidth=3, zorder=5)
                        # --- MODIFIKATION: Daten für Limits sammeln ---
                        all_data_for_lims.append(hist_period)

                    # --- Axis Formatting (PASS 1) ---
                    ax.set_xscale('log')
                    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
                    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
                    
                    # --- MODIFIKATION: Dynamische X-Limits berechnen und speichern ---
                    if all_data_for_lims:
                        min_val = np.min(all_data_for_lims)
                        max_val = np.max(all_data_for_lims)
                        
                        # Definiere einen "zoomed" Bereich
                        x_min_limit = max(0.8, min_val * 0.8) 
                        x_max_limit = max_val * 1.5          
                        
                        if (max_val / min_val) < 5: 
                            x_max_limit = max(x_max_limit, min_val * 5)
                        
                        # Speichere die berechneten Limits für diese Spalte
                        column_x_limits[col].append((x_min_limit, x_max_limit))
                    else:
                        column_x_limits[col].append((0.8, 100)) # Fallback
                    # --- ENDE MODIFIKATION ---
                    
                    # --- MODIFIKATION: Statisches set_xlim und set_xticks entfernt ---
                    # ax.set_xticks([1, 2, 5, 10, 20, 50, 100, 150])
                    # ax.set_xlim(left=0.8, right=160) 
                    
                    ax.grid(axis='y', linestyle='none')
                    ax.grid(axis='x', linestyle=':', which='both')
                    
                    # --- MODIFIKATION: X-Achsen-Beschriftung für alle Plots ---
                    ax.set_xlabel('Return Period (Years)', fontsize=11)
                    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, pos: f'{x:.0f}'))
                    ax.tick_params(axis='x', labelbottom=True)
                    # --- ENDE MODIFIKATION ---
                    
                    ax.set_yticks(y_ticks)
                    if col == 0: 
                        ax.set_yticklabels(y_tick_labels, fontsize=9)
                        season_title = "Winter\n(Dec - May)" if half_year == 'winter' else "Summer\n(Jun - Nov)"
                        ax.set_ylabel(season_title, fontsize=11, weight='bold', labelpad=15)
                    else:
                        ax.set_yticklabels([])
                        ax.set_ylabel('')

                    if ax.get_legend() is not None: ax.get_legend().remove()
                    
                    # n=X/Y Annotationen (wie zuvor)
                    for i, storyline in enumerate(storyline_order):
                        y_base = y_ticks[i]
                        for j, gwl in enumerate(gwls_to_plot):
                            gwl_label = f'+{gwl}°C'
                            y_offset = -0.2 + (j * 0.4) 
                            event_data_gwl = results['data'].get(gwl, {}).get(half_year, {}).get(storyline, {}).get(event_key)
                            if event_data_gwl and 'model_count_X' in event_data_gwl:
                                X, Y = event_data_gwl['model_count_X'], event_data_gwl['model_count_Y']
                                text_to_display = f"n={X}/{Y}"
                                ax.text(0.98, y_base + y_offset, text_to_display, 
                                        transform=ax.get_yaxis_transform(), 
                                        horizontalalignment='right', fontsize=7, weight='bold', 
                                        color=gwl_colors[gwl_label],
                                        bbox=dict(facecolor='white', alpha=0.6, pad=0.1, edgecolor='none'))
        
        # --- MODIFIKATION: Unbenutzte Achsen ausschalten (angepasste Logik) ---
        for r in [0, 1]:
            for c in range(num_cols_low, num_cols):
                axs[r, c].axis('off')
        for r in [2, 3]:
            for c in range(num_cols_high, num_cols):
                axs[r, c].axis('off')
        
        # --- MODIFIKATION: Zweiter Durchlauf (PASS 2) - Anwenden der X-Limits ---
        logging.info("Applying shared column X-limits for return period plot...")
        for col, limits_list in column_x_limits.items():
            if limits_list: 
                # Finde den weitesten Bereich, der für diese Spalte benötigt wird
                final_min_lim = min(l[0] for l in limits_list)
                final_max_lim = max(l[1] for l in limits_list)
                
                # Wende dieses Limit auf alle Zeilen in dieser Spalte an
                for row in range(num_rows):
                    axs[row, col].set_xlim(left=final_min_lim, right=final_max_lim)
                    # Setze die Ticks basierend auf dem finalen Limit
                    if final_max_lim <= 50:
                        axs[row, col].set_xticks([1, 2, 5, 10, 20, 50])
                    elif final_max_lim <= 200:
                        axs[row, col].set_xticks([1, 2, 5, 10, 20, 50, 100, 150, 200])
                    else:
                        axs[row, col].set_xticks([1, 5, 10, 50, 100, 200, 500, 1000]) # Fallback für große Werte
        # --- ENDE MODIFIKATION ---


        # --- 4. Final Figure Formatting ---
        legend_handles = [
            plt.Line2D([0], [0], color='skyblue', linestyle='--', linewidth=3, label='Historical Return Period'),
            plt.Line2D([0], [0], marker=gwl_markers[f'+{gwls_to_plot[0]}°C'], color=gwl_colors[f'+{gwls_to_plot[0]}°C'], 
                       label=f'Future (+{gwls_to_plot[0]}°C)', linestyle='None', markersize=10, markeredgecolor='black', markeredgewidth=0.5),
            plt.Line2D([0], [0], marker=gwl_markers[f'+{gwls_to_plot[1]}°C'], color=gwl_colors[f'+{gwls_to_plot[1]}°C'],
                       label=f'Future (+{gwls_to_plot[1]}°C)', linestyle='None', markersize=10, markeredgecolor='black', markeredgewidth=0.5)
        ]
        fig.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.5, 0.01), ncol=3, fontsize=12)
        
        fig.suptitle(f"Change in Return Period of Discharge Events for {scenario.upper()} (Half-Year Analysis, Pooled GEV)",
                    fontsize=16, weight='bold', y=0.99)
        
        fig.text(0.5, 0.94, 'Low-Flow Events', ha='center', va='center', fontsize=14, weight='bold')
        fig.text(0.5, 0.505, 'High-Flow Events', ha='center', va='center', fontsize=14, weight='bold')
        
        fig.tight_layout(rect=[0.05, 0.05, 0.98, 0.92], h_pad=8.0, w_pad=2.0)
        
        filename = os.path.join(config.PLOT_DIR, f"storyline_discharge_return_period_BY_EVENT_{scenario}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logging.info(f"Saved REORGANIZED return period plot (4-row, Pooled GEV Points, LNWL removed, SPALTEN-ZOOM) to {filename}")
        
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
                    ax = fig.add_subplot(gs[row_idx, col_idx], projection=ccrs.PlateCarree())
                    
                    storyline_title_formatted = storyline_name.replace(" & ", " &\n")
                    main_title_part = f'GWL +{gwl}°C, {season_full} ({season})'

                    data = map_data.get(gwl, {}).get(season, {}).get(storyline_name)
                    
                    if data:
                        change_map = data['mean_change_map']
                        hist_map = data['historical_mean_map']
                        
                        cf, _ = Visualizer.plot_u850_change_map(
                            ax, u850_change_data=change_map.data, 
                            historical_mean_contours=hist_map.data,
                            lons=change_map.lon.values, lats=change_map.lat.values,
                            title=main_title_part, 
                            season_label=storyline_title_formatted,
                            vmin=-2.5, vmax=2.5
                        )
                        if cf is not None:
                            cf_ref = cf
                    else:
                        ax.set_title(f'GWL +{gwl}°C, {season}\n{storyline_name}', fontsize=10)
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
                
        fig.tight_layout(rect=[0.01, 0.04, 0.95, 0.96])
        
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
                ax.set_title(f"GWL +{gwl}°C", fontsize=12)
                continue

            # Align data using common model keys
            common_models = sorted(list(set(x_deltas.keys()) & set(y_deltas.keys())))
            if not common_models:
                ax.text(0.5, 0.5, "No Common Models", ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"GWL +{gwl}°C", fontsize=12)
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
            
            ax.set_title(f"GWL +{gwl}°C", fontsize=14, weight='bold')
            ax.grid(True, linestyle=':', alpha=0.7)
            ax.axhline(0, color='black', lw=0.8, linestyle='-')
            ax.axvline(0, color='black', lw=0.8, linestyle='-')
            ax.legend(fontsize=10)

        ref_period_changes = f"{Config.CMIP6_ANOMALY_REF_START}-{Config.CMIP6_ANOMALY_REF_END}"
        fig.suptitle(f"Cross-Season Jet Relationship for {scenario.upper()}\n(Changes relative to {ref_period_changes})",
                     fontsize=16, weight='bold')
        
        fig.tight_layout(rect=[0.02, 0.02, 1, 0.93])
        
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
                        ax.set_ylabel(f'GWL +{gwl}°C\n(% aller LNWL-Tage)', fontsize=11, weight='bold')
            
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
            
            fig.tight_layout(rect=[0.03, 0.05, 1, 0.95], h_pad=2.0, w_pad=0.5)
            
            filename = os.path.join(config.PLOT_DIR, f"storyline_lnwl_monthly_distribution_{scenario}.png")
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close(fig)
            logging.info(f"Monatliches LNWL-Grid-Plot gespeichert: {filename}")

        except Exception as e:
            logging.error(f"Fehler beim Zeichnen des LNWL-Grid-Plots: {e}")
            logging.error(traceback.format_exc())
            if 'fig' in locals():
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
            ('Q_daily_low', 'Daily Minimum'),
            ('Q_7day_low', '7-Day Minimum'),
            ('Q_30day_low', '30-Day Minimum'),     # <-- MODIFIED
            ('Q_3month_low', '3-Month Minimum')
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
                        season_title = "Winter Half-Year\n(Dec - May)"
                    elif half_year == 'summer':
                        season_title = "Summer Half-Year\n(Jun - Nov)"
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
        
        fig.tight_layout(rect=[0.05, 0.05, 0.98, 0.95], h_pad=3.0, w_pad=2.5)
        
        filename = os.path.join(config.PLOT_DIR, f"storyline_lnwl_aggregation_comparison_{scenario}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logging.info(f"Saved LNWL aggregation comparison plot (3x4 grid, 30-Day, FullYear) to {filename}")