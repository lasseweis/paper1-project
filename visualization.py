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
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Import local modules
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
                          show_jet_boxes=False, significance_level=0.05, std_dev_predictor=None):
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
            contour_levels = np.arange(-20, 21, 4)
            cs = ax.contour(lons_plot, lats_plot, ua_seasonal_mean, levels=contour_levels, colors='black',
                            linewidths=0.8, transform=ccrs.PlateCarree())
            ax.clabel(cs, inline=True, fontsize=7, fmt='%d')

        if p_values is not None and slopes is not None:
            sig_mask = (p_values < significance_level) & np.isfinite(slopes)
            if np.any(sig_mask):
                 ax.scatter(lons_plot[sig_mask], lats_plot[sig_mask], s=1, color='dimgray', marker='.',
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
        cf_pr, label_pr = Visualizer.plot_regression_map(ax_pr_winter, winter_pr_data.get('slopes_pr'), winter_pr_data.get('p_values_pr'), winter_pr_data.get('lons'), winter_pr_data.get('lats'), f"{dataset_key}: U850 vs PR Box Index", box_coords, "DJF", 'pr', ua_seasonal_mean=winter_pr_data.get('ua850_mean'), std_dev_predictor=winter_pr_data.get('std_dev_pr'))

        summer_pr_data = all_season_data['Summer']
        ax_pr_summer = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
        Visualizer.plot_regression_map(ax_pr_summer, summer_pr_data.get('slopes_pr'), summer_pr_data.get('p_values_pr'), summer_pr_data.get('lons'), summer_pr_data.get('lats'), f"{dataset_key}: U850 vs PR Box Index", box_coords, "JJA", 'pr', ua_seasonal_mean=summer_pr_data.get('ua850_mean'), std_dev_predictor=summer_pr_data.get('std_dev_pr'))

        if cf_pr:
            cax_pr = fig.add_subplot(gs[0, 2]); fig.colorbar(cf_pr, cax=cax_pr, extend='both', label=label_pr)

        # Plot TAS panels
        winter_tas_data = all_season_data['Winter']
        ax_tas_winter = fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree())
        cf_tas, label_tas = Visualizer.plot_regression_map(ax_tas_winter, winter_tas_data.get('slopes_tas'), winter_tas_data.get('p_values_tas'), winter_tas_data.get('lons'), winter_tas_data.get('lats'), f"{dataset_key}: U850 vs TAS Box Index", box_coords, "DJF", 'tas', ua_seasonal_mean=winter_tas_data.get('ua850_mean'), std_dev_predictor=winter_tas_data.get('std_dev_tas'))

        summer_tas_data = all_season_data['Summer']
        ax_tas_summer = fig.add_subplot(gs[1, 1], projection=ccrs.PlateCarree())
        Visualizer.plot_regression_map(ax_tas_summer, summer_tas_data.get('slopes_tas'), summer_tas_data.get('p_values_tas'), summer_tas_data.get('lons'), summer_tas_data.get('lats'), f"{dataset_key}: U850 vs TAS Box Index", box_coords, "JJA", 'tas', ua_seasonal_mean=summer_tas_data.get('ua850_mean'), std_dev_predictor=summer_tas_data.get('std_dev_tas'))

        if cf_tas:
            cax_tas = fig.add_subplot(gs[1, 2]); fig.colorbar(cf_tas, cax=cax_tas, extend='both', label=label_tas)
            
        plt.suptitle(f"{dataset_key}: U850 Regression onto Box Climate Indices (Detrended, Normalized Predictors)", fontsize=14, weight='bold')
        plt.tight_layout(rect=[0, 0, 0.95, 0.95])
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
                if np.any(sig_mask):
                    ax1.scatter(lons_plot[sig_mask], lats_plot[sig_mask], s=0.5, color='dimgray', marker='.',
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
                if np.any(sig_mask):
                    ax2.scatter(lons_plot[sig_mask], lats_plot[sig_mask], s=0.5, color='dimgray', marker='.',
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
        plt.tight_layout(rect=[0, 0, 0.95, 0.96])
        filename = os.path.join(Config.PLOT_DIR, f'jet_correlation_maps_{season.lower()}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)

    @staticmethod
    def plot_jet_changes_vs_gwl(cmip6_results, filename="cmip6_jet_changes_vs_gwl.png"):
        """
        Plots CMIP6 jet index changes vs GWL, with percentile spread.
        MODIFIZIERT, um eine gemeinsame Legende und spezifische Storyline-Marker zu verwenden,
        ähnlich dem Stil des ursprünglichen Codes.
        """
        logging.info("Plotting Jet Changes vs GWL (with shared legend and custom markers)...")
        Visualizer.ensure_plot_dir_exists()

        if not cmip6_results or 'all_individual_model_deltas_for_plot' not in cmip6_results:
            logging.warning("Cannot plot jet_changes_vs_gwl: Missing CMIP6 analysis results.")
            return

        all_deltas = cmip6_results['all_individual_model_deltas_for_plot']
        mmm_changes = cmip6_results.get('mmm_changes', {})
        jet_indices_to_plot = list(Config.STORYLINE_JET_CHANGES.keys())
        
        n_indices = len(jet_indices_to_plot)
        if n_indices == 0: return

        fig, axs = plt.subplots(1, n_indices, figsize=(7 * n_indices, 6.5), sharey=False, squeeze=False)
        axs = axs.flatten()

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
            'Core Mean':    {'color': '#1f77b4', 'marker': 'X', 's': 60}, 
            'Core High':    {'color': '#ff7f0e', 'marker': 'X', 's': 60},
            'Extreme Low':  {'color': '#2ca02c', 'marker': 'P', 's': 70},
            'Extreme High': {'color': '#d62728', 'marker': 'P', 's': 70}
        }

        used_storyline_types = set()
        for jet_idx in jet_indices_to_plot:
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

        for i, jet_idx in enumerate(jet_indices_to_plot):
            ax = axs[i]
            
            deltas_by_gwl = all_deltas[jet_idx]
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
            
            valid_indices = [i for i, (p10_val, p90_val) in enumerate(zip(p10, p90)) if not np.isnan(p10_val) and not np.isnan(p90_val)]
            if valid_indices:
                gwls_plot = [gwls_fine[i] for i in valid_indices]
                p10_plot = [p10[i] for i in valid_indices]
                p90_plot = [p90[i] for i in valid_indices]
                ax.fill_between(gwls_plot, p10_plot, p90_plot, color='lightcoral', alpha=0.4)
                    
            gwls_main = sorted(mmm_changes.keys())
            mmm_values = [mmm_changes[gwl].get(jet_idx, np.nan) for gwl in gwls_main]
            ax.plot(gwls_main, mmm_values, marker='o', linestyle='-', color='black', lw=2.5, markersize=7)

            for gwl, storylines in Config.STORYLINE_JET_CHANGES.get(jet_idx, {}).items():
                for name, value in storylines.items():
                    if name in storyline_styles:
                        style = storyline_styles[name]
                        ax.scatter(gwl, value,
                                   marker=style['marker'],
                                   color=style['color'],
                                   s=style['s'],
                                   edgecolor='black',
                                   linewidth=0.8,
                                   zorder=5)

            ylabel = f'Change in {jet_idx.replace("_", " ")}'
            if '_pr' in jet_idx: ylabel += ' (%)'
            elif 'Lat' in jet_idx: ylabel += ' (°Lat)'
            elif 'Speed' in jet_idx: ylabel += ' (m/s)'
            ax.set_ylabel(ylabel)
            ax.set_xlabel('Global Warming Level (°C)')
            ax.set_title(f'Projected Change in {jet_idx.replace("_", " ")}')
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.axhline(0, color='grey', lw=0.8)
        
        fig.legend(handles=master_legend_handles, labels=master_legend_labels,
                   loc='lower center',
                   bbox_to_anchor=(0.5, -0.01),
                   ncol=4,
                   fontsize=10,
                   frameon=True)

        plt.tight_layout(rect=[0, 0.08, 1, 0.95])
        fig.suptitle("CMIP6 Projected Jet Changes vs. Global Warming Level", fontsize=16, weight='bold')
        filepath = os.path.join(Config.PLOT_DIR, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)

    @staticmethod
    def plot_jet_impact_comparison_maps(impact_data_20crv3, impact_data_era5, season):
        """
        Creates a comparison plot (8 subplots) for jet impact regressions for a given season.
        Compares 20CRv3 and ERA5 side-by-side for different jet indices and variables.
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
                
                if 'p_values' in data_20crv3:
                    sig_mask = (data_20crv3['p_values'] < 0.05) & np.isfinite(data_20crv3['slopes'])
                    if np.any(sig_mask):
                        ax1.scatter(lons_plot[sig_mask], lats_plot[sig_mask], s=0.5, color='dimgray', marker='.',
                                    alpha=0.4, transform=ccrs.PlateCarree())
                
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

                # Use the same mappable for the colorbar, even if it's from the other plot
                cf_era5 = ax2.pcolormesh(lons_plot, lats_plot, data_era5['slopes'], shading='auto',
                                    cmap=config['cmap'], vmin=config['vmin'], vmax=config['vmax'],
                                    transform=ccrs.PlateCarree())
                if cf is None: cf = cf_era5 # Fallback if 20CRv3 data was missing

                if 'p_values' in data_era5:
                    sig_mask = (data_era5['p_values'] < 0.05) & np.isfinite(data_era5['slopes'])
                    if np.any(sig_mask):
                        ax2.scatter(lons_plot[sig_mask], lats_plot[sig_mask], s=0.5, color='dimgray', marker='.',
                                    alpha=0.4, transform=ccrs.PlateCarree())
                
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
        plt.tight_layout(rect=[0, 0, 0.95, 0.96])
        # [MODIFIED] Changed the filename to match the user's request
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

        fig, axs = plt.subplots(2, 2, figsize=(16, 10), sharex=True, tight_layout=True)
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

        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
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
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        filename = os.path.join(Config.PLOT_DIR, f'amo_jet_correlations_comparison_rolling_{window_size}yr.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logging.info(f"Saved AMO vs Jet correlation comparison plot to {filename}")
        
    @staticmethod
    def plot_climate_projection_timeseries(cmip6_plot_data, reanalysis_plot_data, config, filename="climate_indices_evolution.png"):
        """
        Plots CMIP6 and Reanalysis changes over time, showing the evolution of key climate indices.
        This plot is inspired by Harvey et al. (2023), Fig. 6.
        """
        filename = "climate_indices_evolution.png"
        logging.info(f"Plotting climate projection timeseries comparison to {filename}...")
        Visualizer.ensure_plot_dir_exists()

        fig, axs = plt.subplots(1, 3, figsize=(21, 6), sharex=True)
        
        # --- (a) Global Temperature Anomaly ---
        ax_a = axs[0]
        if cmip6_plot_data['Global_Tas']['members']:
            for member_tas in cmip6_plot_data['Global_Tas']['members']:
                ax_a.plot(member_tas.year, member_tas, color='grey', alpha=0.3, linewidth=0.7)
        if cmip6_plot_data['Global_Tas']['mmm'] is not None:
            ax_a.plot(cmip6_plot_data['Global_Tas']['mmm'].year, cmip6_plot_data['Global_Tas']['mmm'], color='black', linewidth=2.5, label='CMIP6 MMM')
        
        ax_a.set_title('(a) Global Temperature Change', fontsize=12)
        ax_a.set_ylabel(f'Temperature Anomaly (°C relative to {config.CMIP6_PRE_INDUSTRIAL_REF_START}-{config.CMIP6_PRE_INDUSTRIAL_REF_END})', fontsize=10)
        ax_a.legend(fontsize=9)

        # --- (b) Summer Jet Latitude Anomaly ---
        ax_b = axs[1]
        if cmip6_plot_data['JJA_JetLat']['members']:
            for member_lat in cmip6_plot_data['JJA_JetLat']['members']:
                ax_b.plot(member_lat.season_year, member_lat, color='grey', alpha=0.3, linewidth=0.7)
        if cmip6_plot_data['JJA_JetLat']['mmm'] is not None:
            ax_b.plot(cmip6_plot_data['JJA_JetLat']['mmm'].season_year, cmip6_plot_data['JJA_JetLat']['mmm'], color='black', linewidth=2.5, label='CMIP6 MMM')
        
        if reanalysis_plot_data['JJA_JetLat'].get('20CRv3') is not None:
            reanalysis_20crv3_lat = reanalysis_plot_data['JJA_JetLat']['20CRv3']
            ax_b.plot(reanalysis_20crv3_lat.season_year, reanalysis_20crv3_lat, color='darkorange', linewidth=2, label='20CRv3')
        if reanalysis_plot_data['JJA_JetLat'].get('ERA5') is not None:
            reanalysis_era5_lat = reanalysis_plot_data['JJA_JetLat']['ERA5']
            ax_b.plot(reanalysis_era5_lat.season_year, reanalysis_era5_lat, color='purple', linewidth=2, label='ERA5')

        ax_b.set_title('(b) Summer (JJA) Jet Latitude Change', fontsize=12)
        ax_b.set_ylabel('Latitude Anomaly (°)', fontsize=10)
        ax_b.legend(fontsize=9)
        
        # --- (c) Winter Jet Speed Anomaly ---
        ax_c = axs[2]
        if cmip6_plot_data['DJF_JetSpeed']['members']:
            for member_spd in cmip6_plot_data['DJF_JetSpeed']['members']:
                ax_c.plot(member_spd.season_year, member_spd, color='grey', alpha=0.3, linewidth=0.7)
        if cmip6_plot_data['DJF_JetSpeed']['mmm'] is not None:
            ax_c.plot(cmip6_plot_data['DJF_JetSpeed']['mmm'].season_year, cmip6_plot_data['DJF_JetSpeed']['mmm'], color='black', linewidth=2.5, label='CMIP6 MMM')
        
        if reanalysis_plot_data['DJF_JetSpeed'].get('20CRv3') is not None:
            reanalysis_20crv3_spd = reanalysis_plot_data['DJF_JetSpeed']['20CRv3']
            ax_c.plot(reanalysis_20crv3_spd.season_year, reanalysis_20crv3_spd, color='darkorange', linewidth=2, label='20CRv3')
        if reanalysis_plot_data['DJF_JetSpeed'].get('ERA5') is not None:
            reanalysis_era5_spd = reanalysis_plot_data['DJF_JetSpeed']['ERA5']
            ax_c.plot(reanalysis_era5_spd.season_year, reanalysis_era5_spd, color='purple', linewidth=2, label='ERA5')

        ax_c.set_title('(c) Winter (DJF) Jet Speed Change', fontsize=12)
        ax_c.set_ylabel('Speed Anomaly (m/s)', fontsize=10)
        ax_c.legend(fontsize=9)

        # --- Final formatting for all axes ---
        for ax in axs:
            ax.grid(True, linestyle=':', alpha=0.7)
            ax.set_xlabel('Year', fontsize=10)
            ax.set_xlim(1850, 2100)
            ax.axhline(0, color='black', linewidth=0.5)

        fig.suptitle('Evolution of Key Climate Indices (20-Year Rolling Mean Anomaly)', fontsize=16, weight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
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
            ax.plot(x_fit, y_fit_iav, color='red', linestyle='--', linewidth=2, label=f'Obs. IAV Slope ($\\beta_{{obs}}$={beta_obs:.2f})')

        # Formatting
        ref_period = f"{Config.CMIP6_ANOMALY_REF_START}-{Config.CMIP6_ANOMALY_REF_END}"
        x_label = f'Change in {jet_key.replace("_", " ")}'
        y_label = f'Change in {impact_key.replace("_", " ")}'
        
        if "Lat" in jet_key: x_label += ' (°Lat)'
        elif "Speed" in jet_key: x_label += ' (m/s)'
        
        if "_pr" in impact_key: y_label += ' (%)'
        elif "_tas" in impact_key: y_label += ' (°C)'
        
        ax.set_xlabel(x_label, fontsize=9)
        ax.set_ylabel(y_label, fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.axhline(0, color='grey', lw=0.7); ax.axvline(0, color='grey', lw=0.7)
        ax.legend(fontsize=8)

    @staticmethod
    def plot_cmip6_scatter_comparison(cmip6_results, beta_obs_slopes, gwl_to_plot):
        """
        Creates a 2x2 subplot figure comparing CMIP6 projected changes for different
        jet indices and impact variables at a specific Global Warming Level.
        """
        if not cmip6_results or not beta_obs_slopes:
            logging.warning("Cannot plot CMIP6 scatter comparison: Missing results or beta slopes.")
            return
            
        logging.info(f"Plotting CMIP6 scatter comparison for {gwl_to_plot}°C GWL...")
        Visualizer.ensure_plot_dir_exists()

        fig, axs = plt.subplots(2, 2, figsize=(14, 12))
        
        plot_configs = [
            {'ax': axs[0, 0], 'jet_key': 'DJF_JetSpeed', 'impact_key': 'DJF_tas', 'beta_key': 'DJF_JetSpeed_vs_tas', 'title': 'Winter Temp vs. Jet Speed'},
            {'ax': axs[0, 1], 'jet_key': 'JJA_JetLat',   'impact_key': 'JJA_tas', 'beta_key': 'JJA_JetLat_vs_tas',   'title': 'Summer Temp vs. Jet Latitude'},
            {'ax': axs[1, 0], 'jet_key': 'DJF_JetSpeed', 'impact_key': 'DJF_pr',  'beta_key': 'DJF_JetSpeed_vs_pr',  'title': 'Winter Precip vs. Jet Speed'},
            {'ax': axs[1, 1], 'jet_key': 'JJA_JetLat',   'impact_key': 'JJA_pr',  'beta_key': 'JJA_JetLat_vs_pr',    'title': 'Summer Precip vs. Jet Latitude'},
        ]

        for config in plot_configs:
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
        
        ref_period = f"{Config.CMIP6_ANOMALY_REF_START}-{Config.CMIP6_ANOMALY_REF_END}"
        fig.suptitle(f"CMIP6 Projected Changes at {gwl_to_plot}°C Global Warming Level\n(Changes relative to {ref_period} PI)",
                     fontsize=16, weight='bold')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        filename = os.path.join(Config.PLOT_DIR, f"cmip6_scatter_comparison_gwl_{gwl_to_plot:.1f}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logging.info(f"Saved CMIP6 scatter comparison plot to {filename}")