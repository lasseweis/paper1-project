"""
Jet stream analysis module.

This file contains the JetStreamAnalyzer class, which provides
specialized methods for calculating jet stream indices (speed and latitude)
from U850 wind data arrays.
"""
import numpy as np
import xarray as xr
import logging
import traceback

# Import the project's configuration
from config import Config

class JetStreamAnalyzer:
    """Class for jet stream-specific analyses."""

    @staticmethod
    def calculate_jet_speed_index(da_season):
        """
        Calculate the jet speed index from a seasonal U850 DataArray.
        Ensures a 1D time series output.
        """
        if da_season is None:
            return None

        try:
            lat_min, lat_max = Config.JET_SPEED_BOX_LAT_MIN, Config.JET_SPEED_BOX_LAT_MAX
            lon_min, lon_max = Config.JET_SPEED_BOX_LON_MIN, Config.JET_SPEED_BOX_LON_MAX
            
            if not all(hasattr(da_season, c) and da_season[c].size > 0 for c in ['lat', 'lon']):
                logging.warning(f"JetSpeed: da_season lacks lat/lon coordinates or they are empty.")
                return None

            lat_slice = slice(lat_min, lat_max)
            if da_season.lat.size > 1 and da_season.lat.values[0] > da_season.lat.values[-1]:
                lat_slice = slice(lat_max, lat_min)

            domain = da_season.sel(lat=lat_slice, lon=slice(lon_min, lon_max))
            
            if domain.lat.size > 0 and domain.lon.size > 0 and not domain.isnull().all():
                
                # --- KORREKTUR ---
                # Der räumliche Mittelwert wird direkt auf dem gesamten Gebietsausschnitt
                # berechnet, ohne vorher nicht-westliche Winde zu entfernen. Das ist die
                # Standardmethode und verhindert die Erzeugung von NaNs, wenn in Teilen
                # der Box östliche Winde auftreten.
                weights = np.cos(np.deg2rad(domain.lat))
                weights.name = "weights"
                
                weighted_mean = domain.weighted(weights).mean(dim=["lat", "lon"], skipna=True)
                
                if weighted_mean.ndim > 1:
                    time_dim = next((d for d in ['season_year', 'year', 'time'] if d in weighted_mean.dims), None)
                    if time_dim:
                        dims_to_squeeze = [d for d in weighted_mean.dims if d != time_dim]
                        weighted_mean = weighted_mean.squeeze(dim=dims_to_squeeze, drop=True)

                if weighted_mean.ndim > 1:
                        logging.error(f"JetSpeed Index is still >1D after squeeze: Dims {weighted_mean.dims}.")
                        return None
                
                weighted_mean.name = "jet_speed_index"
                if 'dataset' in da_season.attrs:
                    weighted_mean.attrs['dataset'] = da_season.attrs['dataset']
                return weighted_mean
            else:
                logging.warning(f"JetSpeed: Domain for index calculation is empty or contains only NaNs.")
                return None

        except Exception as e:
            logging.error(f"Error in calculate_jet_speed_index: {e}")
            traceback.print_exc()
            return None

    @staticmethod
    def calculate_jet_lat_index(da_season):
        """
        Calculate the jet latitude index from a seasonal U850 DataArray.
        Uses u^2 weighting as in Harvey et al. (2023). Ensures a 1D output.
        """
        if da_season is None:
            return None

        try:
            lat_min, lat_max = Config.JET_LAT_BOX_LAT_MIN, Config.JET_LAT_BOX_LAT_MAX
            lon_min, lon_max = Config.JET_LAT_BOX_LON_MIN, Config.JET_LAT_BOX_LON_MAX

            if not all(hasattr(da_season, c) and da_season[c].size > 0 for c in ['lat', 'lon']):
                logging.warning(f"JetLat: da_season lacks lat/lon coordinates or they are empty.")
                return None
            
            # === KORREKTUR START ===
            lat_slice = slice(lat_min, lat_max)
            if da_season.lat.size > 1 and da_season.lat.values[0] > da_season.lat.values[-1]:
                lat_slice = slice(lat_max, lat_min)
            # === KORREKTUR ENDE ===

            domain = da_season.sel(lat=lat_slice, lon=slice(lon_min, lon_max))
            
            if domain.lat.size > 0 and domain.lon.size > 0:
                zonal_avg_u = domain.mean(dim="lon", skipna=True)
                
                u_westerly = zonal_avg_u.where(zonal_avg_u > 0, 0)
                u_squared = u_westerly**2
                
                lat_coord_da = u_squared.lat
                
                numerator = (lat_coord_da * u_squared).sum(dim='lat', skipna=True)
                denominator = u_squared.sum(dim='lat', skipna=True)

                jet_latitude = xr.where(abs(denominator) > 1e-9, numerator / denominator, np.nan)
                
                if jet_latitude.ndim > 1:
                    time_dim = next((d for d in ['season_year', 'year', 'time'] if d in jet_latitude.dims), None)
                    if time_dim:
                        dims_to_squeeze = [d for d in jet_latitude.dims if d != time_dim]
                        jet_latitude = jet_latitude.squeeze(dim=dims_to_squeeze, drop=True)

                if jet_latitude.ndim > 1:
                     logging.error(f"JetLat Index is still >1D after squeeze: Dims {jet_latitude.dims}.")
                     return None
                
                jet_latitude.name = "jet_lat_index"
                if 'dataset' in da_season.attrs:
                    jet_latitude.attrs['dataset'] = da_season.attrs['dataset']
                return jet_latitude
            else:
                logging.warning(f"JetLat: Domain for index calculation is empty.")
                return None

        except Exception as e:
            logging.error(f"Error in calculate_jet_lat_index: {e}")
            traceback.print_exc()
            return None