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
            unit = '%' if variable == 'pr' else '°C'
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

    # All other plotting functions from your original script go here...
    # For example:
    # @staticmethod
    # def plot_seasonal_correlation_matrix(...):
    #
    # @staticmethod
    # def plot_jet_indices_timeseries(...):
    #
    # @staticmethod
    # def plot_cmip6_delta_impact_vs_delta_jet_at_gwl(...):
    
    @staticmethod
    def plot_jet_changes_vs_gwl(cmip6_results, filename="cmip6_jet_changes_vs_gwl.png"):
        """
        Plots CMIP6 jet index changes vs GWL, with percentile spread.
        This function was moved from StorylineAnalyzer to separate calculation from visualization.
        """
        logging.info("Plotting Jet Changes vs GWL (with calculated spread)...")
        Visualizer.ensure_plot_dir_exists()

        if not cmip6_results or 'all_individual_model_deltas_for_plot' not in cmip6_results:
            logging.warning("Cannot plot jet_changes_vs_gwl: Missing CMIP6 analysis results.")
            return

        all_deltas = cmip6_results['all_individual_model_deltas_for_plot']
        mmm_changes = cmip6_results.get('mmm_changes', {})
        jet_indices_to_plot = list(Config.STORYLINE_JET_CHANGES.keys())
        
        n_indices = len(jet_indices_to_plot)
        if n_indices == 0: return

        fig, axs = plt.subplots(1, n_indices, figsize=(7 * n_indices, 6), sharey=False, squeeze=False)
        axs = axs.flatten()

        for i, jet_idx in enumerate(jet_indices_to_plot):
            ax = axs[i]
            
            deltas_by_gwl = all_deltas[jet_idx]
            gwls_fine = sorted(deltas_by_gwl.keys())

            # --- Plot individual model lines ---
            # 1. Re-Strukturierung der Daten pro Modell
            model_runs = {}
            for gwl, model_deltas_dict in deltas_by_gwl.items():
                for model_key, delta_val in model_deltas_dict.items():
                    if model_key not in model_runs:
                        model_runs[model_key] = []
                    model_runs[model_key].append((gwl, delta_val))

            # 2. Jede Modell-Linie einzeln plotten
            for model_key, run_data in model_runs.items():
                # Sortiere nach GWL, um eine korrekte Linie zu zeichnen
                run_data.sort()
                gwls_sorted = [d[0] for d in run_data]
                values_sorted = [d[1] for d in run_data]
                if len(gwls_sorted) > 1: # Nur plotten, wenn mehr als ein Punkt vorhanden ist
                    ax.plot(gwls_sorted, values_sorted, marker='.', linestyle='-', color='darkgray', alpha=0.5, lw=0.8)

            # --- Plot percentile spread (robustly) ---
            # Extrahiere die Listen der Delta-Werte für jedes GWL
            delta_values_per_gwl = [list(deltas_by_gwl[gwl].values()) for gwl in gwls_fine]
            
            # Berechne Perzentile nur dort, wo Daten vorhanden sind
            p10 = [np.percentile(d, 10) if d else np.nan for d in delta_values_per_gwl]
            p90 = [np.percentile(d, 90) if d else np.nan for d in delta_values_per_gwl]
            
            # Filtere NaNs vor dem Plotten heraus, um Lücken zu vermeiden
            gwls_plot, p10_plot, p90_plot = [], [], []
            for i, gwl in enumerate(gwls_fine):
                if not np.isnan(p10[i]) and not np.isnan(p90[i]):
                    gwls_plot.append(gwl)
                    p10_plot.append(p10[i])
                    p90_plot.append(p90[i])

            if gwls_plot:
                ax.fill_between(gwls_plot, p10_plot, p90_plot, color='lightcoral', alpha=0.4, label='10-90th Percentile Spread')
                    
            # Plot MMM
            gwls_main = sorted(mmm_changes.keys())
            mmm_values = [mmm_changes[gwl].get(jet_idx, np.nan) for gwl in gwls_main]
            ax.plot(gwls_main, mmm_values, marker='o', linestyle='-', color='black', lw=2.5, markersize=7, label='Multi-Model Mean')

            # Plot Storyline markers
            for gwl, storylines in Config.STORYLINE_JET_CHANGES.get(jet_idx, {}).items():
                for name, value in storylines.items():
                    ax.plot(gwl, value, marker='X', markersize=10, linestyle='none', label=f'Storyline: {name}')

            # Formatting
            ylabel = f'Change in {jet_idx.replace("_", " ")}'
            if '_pr' in jet_idx: ylabel += ' (%)'
            elif 'Lat' in jet_idx: ylabel += ' (°Lat)'
            elif 'Speed' in jet_idx: ylabel += ' (m/s)'
            ax.set_ylabel(ylabel)
            ax.set_xlabel('Global Warming Level (°C)')
            ax.set_title(f'Projected Change in {jet_idx.replace("_", " ")}')
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.axhline(0, color='grey', lw=0.8)
            ax.legend(fontsize=8)
            
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        fig.suptitle("CMIP6 Projected Jet Changes vs. Global Warming Level", fontsize=16, weight='bold')
        filepath = os.path.join(Config.PLOT_DIR, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)