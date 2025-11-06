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
import lmoments3 as lm  
from lmoments3 import distr

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
    def calculate_eva_thresholds(daily_timeseries, eva_type='low', q_days=[1, 7], return_periods=[10, 50], half_year_filter=None):
        """
        Calculates hydrological extreme value thresholds (e.g., 7Q10) from a daily time series.
        
        MODIFIED VERSION (v3.6 - Nur Quantile):
        - Verwendet NUR die empirische Quantilmethode (np.quantile), da der GEV-Fit unzuverlässig war.

        --- MODIFIED (Nov 5, 2025) ---
        - The 'if half_year_filter:' check is now more specific ('in ['winter', 'summer']')
          to allow 'full_year' or None to pass through and use the full dataset.
        --- END MODIFIED ---

        Parameters:
        -----------
        daily_timeseries : xr.DataArray
            A 1D DataArray of daily discharge values with a 'time' dimension.
        eva_type : str, default='low'
            'low' for minima, 'high' for maxima.
        q_days : list, default=[1, 7]
            List of n-day moving average windows.
        return_periods : list, default=[10, 50]
            List of return periods (in years) to calculate.
        half_year_filter : str, optional, default=None
            If 'winter' (Dec-May) or 'summer' (Jun-Nov), filters the daily
            data for these months before extracting annual extremes.

        Returns:
        --------
        dict
            A dictionary containing the calculated discharge thresholds (in m³/s).
            Example: {'1Q10': 950.5, '7Q10': 1050.0, ...}
        """
        if daily_timeseries is None or daily_timeseries.time.size < 365*5:
            logging.warning("Cannot calculate EVA thresholds: Daily time series is missing or too short.")
            return {}

        # --- Half-Year Filtering ---
        # *** MODIFIED LINE HERE ***
        if half_year_filter and half_year_filter in ['winter', 'summer']:
        # *** END MODIFIED LINE ***
            months = []
            if half_year_filter == 'winter':
                months = [12, 1, 2, 3, 4, 5]
                logging.info(f"Calculating EVA thresholds for WINTER half-year (Dec-May)...")
            elif half_year_filter == 'summer':
                months = [6, 7, 8, 9, 10, 11]
                logging.info(f"Calculating EVA thresholds for SUMMER half-year (Jun-Nov)...")
            
            if months:
                daily_timeseries_filtered = daily_timeseries.where(daily_timeseries.time.dt.month.isin(months), drop=True)
                if daily_timeseries_filtered.time.size == 0:
                    logging.error(f"No data for half-year filter '{half_year_filter}'.")
                    return {}
            else:
                # This path should not be reached if check is specific
                daily_timeseries_filtered = daily_timeseries
        else:
            if half_year_filter == 'full_year':
                logging.info("Calculating EVA thresholds for FULL YEAR...")
            daily_timeseries_filtered = daily_timeseries
        # --- Ende Half-Year Filtering ---

        thresholds_m3s = {}
        
        for q in q_days:
            # 1. Calculate n-day moving average on the FULL, UNFILTERED timeseries
            if q == 1:
                moving_avg_full = daily_timeseries
            else:
                moving_avg_full = daily_timeseries.rolling(time=q, center=True, min_periods=q).mean()
            
            # 2. Filter the resulting moving average series for the half-year
            # *** MODIFIED BLOCK HERE ***
            if half_year_filter and half_year_filter in ['winter', 'summer']:
                months = []
                if half_year_filter == 'winter':
                    months = [12, 1, 2, 3, 4, 5]
                elif half_year_filter == 'summer':
                    months = [6, 7, 8, 9, 10, 11]
                
                if months:
                    if 'month' not in moving_avg_full.coords:
                        moving_avg_full = moving_avg_full.assign_coords(month=("time", moving_avg_full.time.dt.month.data))
                    moving_avg_filtered = moving_avg_full.where(moving_avg_full.month.isin(months), drop=True)
                else:
                    moving_avg_filtered = moving_avg_full
            else:
                moving_avg_filtered = moving_avg_full
            # *** END MODIFIED BLOCK ***

            # 3. Extract annual extremes
            if eva_type == 'low':
                annual_extremes = moving_avg_filtered.groupby('time.year').min('time')
            else: # 'high'
                annual_extremes = moving_avg_filtered.groupby('time.year').max('time')
                
            clean_extremes = annual_extremes.dropna(dim='year').values
            
            if len(clean_extremes) < 20: 
                logging.warning(f"Skipping {q}Q analysis for {eva_type} ({half_year_filter}): Only {len(clean_extremes)} valid years. (Need > 20 for robust fit)")
                continue

            # --- START: NUR-QUANTIL-METHODE ---
            logging.info(f"Calculating EVA thresholds for {q}Q ({eva_type}, {half_year_filter or 'full_year'}) using empirical quantile method.")

            for T in return_periods:
                if eva_type == 'low':
                    prob = 1.0 / T
                    quantile_to_find = prob
                else: # 'high'
                    prob = 1.0 / T
                    quantile_to_find = 1.0 - prob
                
                discharge_val = np.quantile(clean_extremes, quantile_to_find, interpolation='linear')
                
                key = f'{q}Q{T}'
                thresholds_m3s[key] = discharge_val
            # --- ENDE: NUR-QUANTIL-METHODE ---
                
        return thresholds_m3s