"""
Statistical analysis utilities module.

This file contains the StatsAnalyzer class, which provides helper
functions for common statistical operations like linear regression,
normalization, and calculating rolling means.
"""
import numpy as np
import pandas as pd
import xarray as xr
import logging
import traceback
from scipy.stats import linregress

class StatsAnalyzer:
    """A collection of statistical analysis utilities."""

    @staticmethod
    def calculate_rolling_mean(series, window=5):
        """
        Calculates the centered rolling mean of a time series, handling NaNs.

        Parameters:
        -----------
        series : xr.DataArray or array-like
            The time series for which to calculate the rolling mean.
        window : int, default=5
            The window size for the rolling mean (in years).

        Returns:
        --------
        xr.DataArray or np.ndarray
            The smoothed time series.
        """
        # For xarray DataArray
        if isinstance(series, xr.DataArray):
            # Use xarray's built-in rolling method, which is more efficient and robust.
            if series.size >= window:
                # KORREKTUR: Die Dimension und Fenstergröße müssen als Dictionary übergeben werden.
                dim_name = next(iter(series.dims))
                smoothed_series = series.rolling(
                    {dim_name: window},
                    center=True
                ).mean()
                # Preserve attributes
                smoothed_series.attrs = dict(series.attrs, smoothed=True, window=window)
                return smoothed_series
            else:
                logging.warning(f"Series too short ({series.size}) for rolling window ({window}). Returning original.")
                return series
        
        # For NumPy arrays or other array-like objects
        else:
            values = np.array(series)
            if values.size < window:
                logging.warning(f"Series too short ({values.size}) for rolling window ({window}). Returning original.")
                return values
                
            # A simple implementation for numpy arrays
            return pd.Series(values).rolling(window=window, center=True, min_periods=1).mean().to_numpy()

    @staticmethod
    def calculate_regression(x, y):
        """
        Calculate linear regression between two 1D arrays.

        This method robustly handles NaN values, ensures 1D inputs, checks for
        matching lengths, and verifies sufficient valid data points and variance.

        Returns:
        --------
        tuple
            (slope, intercept, r_value, p_value, stderr) or (nan, nan, ...) on failure.
        """
        # --- Step 1: Validate inputs and convert to 1D NumPy arrays ---
        try:
            x_np = np.asarray(x).squeeze()
            y_np = np.asarray(y).squeeze()
            if x_np.ndim == 0: x_np = np.array([x_np])
            if y_np.ndim == 0: y_np = np.array([y_np])
        except Exception as e:
            logging.error(f"calculate_regression: Error during conversion to 1D numpy array: {e}")
            return (np.nan,) * 5

        if x_np.ndim != 1 or y_np.ndim != 1:
             logging.error(f"calculate_regression received non-1D input. x_shape:{x_np.shape}, y_shape:{y_np.shape}")
             return (np.nan,) * 5

        if len(x_np) != len(y_np):
             logging.error(f"calculate_regression received inputs of different lengths. x_len:{len(x_np)}, y_len:{len(y_np)}")
             return (np.nan,) * 5

        # --- Step 2: Find valid (finite) data points ---
        mask = np.isfinite(x_np) & np.isfinite(y_np)
        n_valid = np.sum(mask)

        if n_valid < 3: # Need at least 3 points for a meaningful regression
            return (np.nan,) * 5

        x_masked = x_np[mask]
        y_masked = y_np[mask]

        # --- Step 3: Check for variance after masking ---
        if np.var(x_masked) < 1e-10 or np.var(y_masked) < 1e-10:
             return (np.nan,) * 5 # Or return slope=0, etc., if that's desired for no variance.

        # --- Step 4: Perform regression ---
        try:
            return linregress(x_masked, y_masked)
        except ValueError as ve:
            logging.error(f"Error within linregress: {ve}. x_masked len:{len(x_masked)}, y_masked len:{len(y_masked)}")
            return (np.nan,) * 5
        except Exception as e_reg:
             logging.error(f"Unexpected error in linregress call: {e_reg}")
             traceback.print_exc()
             return (np.nan,) * 5

    @staticmethod
    def normalize(series):
        """
        Normalize a series to zero mean and unit variance.
        Preserves xarray.DataArray or pandas.Series structure if applicable.
        """
        if isinstance(series, xr.DataArray):
            original_attrs = series.attrs
            original_coords = series.coords
            original_dims = series.dims
            original_name = series.name
            values = series.data
        elif isinstance(series, pd.Series):
            original_name = series.name
            values = series.values
        else:
            values = np.array(series)

        mask = np.isfinite(values)
        if not np.any(mask): # If all values are NaN
            return series # Return the original object

        mean = np.mean(values[mask])
        std = np.std(values[mask])

        if std < 1e-9: # Avoid division by zero
            normalized_values = values - mean
        else:
            normalized_values = (values - mean) / std

        if isinstance(series, xr.DataArray):
            return xr.DataArray(
                data=normalized_values,
                coords=original_coords,
                dims=original_dims,
                attrs=original_attrs,
                name=original_name
            )
        elif isinstance(series, pd.Series):
             return pd.Series(
                 data=normalized_values,
                 index=series.index,
                 name=original_name
             )
        else:
            return normalized_values
        
    @staticmethod
    def analyze_drought_characteristics(spei_timeseries, threshold=-1.0):
        """
        Analyzes a SPEI timeseries to extract drought event characteristics.

        Parameters:
        -----------
        spei_timeseries : xr.DataArray
            A 1D DataArray of SPEI values with a 'time' dimension.
        threshold : float, default=-1.0
            The SPEI value below which a month is considered in drought.

        Returns:
        --------
        dict
            A dictionary containing drought metrics.
        """
        if spei_timeseries is None or spei_timeseries.size == 0:
            return {}

        # Ensure boolean series for drought periods
        in_drought = (spei_timeseries < threshold)

        if not in_drought.any():
            return {
                'total_drought_months': 0, 'number_of_events': 0,
                'mean_duration': 0, 'longest_duration': 0,
                'mean_intensity': np.nan, 'peak_intensity': np.nan
            }

        # Identify changes to find start/end of events by looking at the difference
        change = in_drought.astype(int).diff(dim='time', n=1)
        
        # An event starts where the change is 1, ends where it's -1
        starts = np.where(change == 1)[0]
        ends = np.where(change == -1)[0]
        
        # Handle cases where the series starts or ends within a drought event
        if in_drought.values[0]:
            starts = np.insert(starts, 0, 0)
        if in_drought.values[-1]:
            ends = np.append(ends, len(spei_timeseries) - 1)

        # Ensure starts and ends arrays are correctly paired
        if len(starts) > len(ends):
            starts = starts[:len(ends)]
        elif len(ends) > len(starts):
            ends = ends[1:]

        if len(starts) == 0:
            return {'total_drought_months': int(in_drought.sum()), 'number_of_events': 0}

        durations = (ends - starts) + 1
        
        event_slices = [slice(s, e + 1) for s, e in zip(starts, ends)]
        intensities = [spei_timeseries.isel(time=sl).mean().item() for sl in event_slices]
        peak_intensities = [spei_timeseries.isel(time=sl).min().item() for sl in event_slices]

        return {
            'total_drought_months': int(in_drought.sum()),
            'number_of_events': len(durations),
            'mean_duration': np.mean(durations) if len(durations) > 0 else 0,
            'longest_duration': np.max(durations) if len(durations) > 0 else 0,
            'mean_intensity': np.mean(intensities) if len(intensities) > 0 else np.nan,
            'peak_intensity': np.min(peak_intensities) if len(peak_intensities) > 0 else np.nan
        }