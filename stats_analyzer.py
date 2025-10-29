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
import statsmodels.api as sm

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

    @staticmethod
    def calculate_multiple_regression(y, x_vars):
        """
        Calculates multiple linear regression.

        Parameters:
        -----------
        y : array-like
            The dependent variable (e.g., precipitation).
        x_vars : dict
            A dictionary where keys are variable names and values are their time series (e.g., {'speed': jet_speed_ts, 'lat': jet_lat_ts}).

        Returns:
        --------
        dict
            A dictionary of coefficients (betas) for each x variable.
        """
        # Create a pandas DataFrame from the input
        df_dict = {'y': y}
        df_dict.update(x_vars)
        df = pd.DataFrame(df_dict).dropna()

        if len(df) < 10: # Need enough data points
            logging.warning("Not enough valid data points for multiple regression.")
            return None

        # Prepare the data for statsmodels
        Y = df['y']
        X = df[list(x_vars.keys())]
        X = sm.add_constant(X) # Adds the intercept (beta_0) to the model

        try:
            model = sm.OLS(Y, X).fit()
            # Return a dictionary of the coefficients, excluding the intercept
            return model.params.drop('const').to_dict()
        except Exception as e:
            logging.error(f"Error during multiple regression: {e}")
            return None
        
    @staticmethod
    def calculate_discharge_percentiles(daily_discharge_da, percentiles=None):
        """
        Calculates specific percentiles for a daily discharge time series.

        Parameters:
        -----------
        daily_discharge_da : xr.DataArray
            A 1D DataArray of daily discharge values.
        percentiles : list of float, default=None
            A list of percentiles to calculate (e.g., [1, 99]).
            If None, defaults to [1, 99].

        Returns:
        --------
        dict
            A dictionary where keys are the percentile (e.g., 'p01', 'p99')
            and values are the calculated discharge values.
        """
        if percentiles is None:
            percentiles = [1, 99]

        if daily_discharge_da is None or daily_discharge_da.size == 0:
            logging.warning("Cannot calculate percentiles from empty discharge series.")
            return {f"p{p:02d}": np.nan for p in percentiles}
        
        try:
            # Drop NaNs before calculating percentiles
            valid_data = daily_discharge_da.dropna(dim='time')
            if valid_data.size == 0:
                logging.warning("No valid (non-NaN) discharge data for percentiles.")
                return {f"p{p:02d}": np.nan for p in percentiles}

            # Calculate percentiles using xarray's quantile method
            q_values = valid_data.quantile([p/100.0 for p in percentiles], dim='time')
            
            results = {}
            for i, p in enumerate(percentiles):
                key = f"p{int(p):02d}"
                results[key] = q_values.isel(quantile=i).item()
            
            logging.info(f"Calculated discharge percentiles: {results}")
            return results
            
        except Exception as e:
            logging.error(f"Error in calculate_discharge_percentiles: {e}")
            return {f"p{p:02d}": np.nan for p in percentiles}

    @staticmethod
    def calculate_rolling_discharge_mean(daily_discharge_da, window_days=7):
        """
        Calculates the rolling mean for a daily discharge time series.
        This is typically used for analyzing 7-day low/high flow events.

        Parameters:
        -----------
        daily_discharge_da : xr.DataArray
            A 1D DataArray of daily discharge values with 'time' dimension.
        window_days : int, default=7
            The window size in days for the rolling mean.

        Returns:
        --------
        xr.DataArray
            A DataArray of the same length, containing the rolling mean.
            Edges will have NaNs where the window is incomplete.
        """
        if daily_discharge_da is None or daily_discharge_da.size < window_days:
            logging.warning(f"Series too short ({daily_discharge_da.size if daily_discharge_da is not None else 0}) for rolling window ({window_days}). Returning NaNs.")
            if daily_discharge_da is None:
                return None
            return xr.full_like(daily_discharge_da, np.nan)
        
        try:
            # Use xarray's rolling method.
            # center=False is typical for discharge, so the value at day 't'
            # represents the mean of [t-window+1, t].
            # min_periods=window_days ensures only full windows are calculated.
            rolling_mean_da = daily_discharge_da.rolling(
                time=window_days,
                center=False,
                min_periods=window_days
            ).mean()
            
            rolling_mean_da.attrs = dict(
                daily_discharge_da.attrs,
                long_name=f'{window_days}-day rolling mean of {daily_discharge_da.attrs.get("long_name", "discharge")}',
                window_days=window_days
            )
            return rolling_mean_da
            
        except Exception as e:
            logging.error(f"Error in calculate_rolling_discharge_mean: {e}")
            if daily_discharge_da is None:
                return None
            return xr.full_like(daily_discharge_da, np.nan)