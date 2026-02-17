
import sys
import os
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import xarray as xr

# Mocking modules that might not be available or problematic in this environment
sys.modules['cartopy'] = MagicMock()
sys.modules['cartopy.crs'] = MagicMock()
sys.modules['cartopy.feature'] = MagicMock()
sys.modules['matplotlib.pyplot'] = MagicMock()
sys.modules['matplotlib.gridspec'] = MagicMock()
sys.modules['seaborn'] = MagicMock()
sys.modules['scipy.stats'] = MagicMock()
sys.modules['statsmodels'] = MagicMock()
sys.modules['statsmodels.api'] = MagicMock()

# Import the Visualizer class
# We need to add the path to the system path
sys.path.append('/nas/home/vlw/Desktop/STREAM/Code/paper1-project')
from visualization import Visualizer

class TestVisualizerColorbars(unittest.TestCase):
    
    def test_colorbar_limits(self):
        # Create a mock DataArray for the difference map
        data = np.random.rand(10, 10)
        da = xr.DataArray(data, coords={'lat': np.arange(10), 'lon': np.arange(10)}, dims=('lat', 'lon'))
        
        # Mock composite results
        composite_results = {
            'future_extreme_mean': da,
            'future_non_extreme_mean': da,
            'hist_extreme_mean': da,
            'hist_non_extreme_mean': da,
            'diff_ext_non_future': da,
            'diff_ext_non_hist': da,
            'diff_fut_hist_ext': da,
            'diff_fut_hist_non': da,
            'sig_mask_ext_non_future': None,
            'sig_mask_ext_non_hist': None,
            'sig_mask_fut_hist_ext': None,
            'sig_mask_fut_hist_non': None,
            'hist_climatology_mean': da
        }
        
        # We need to intercept the local variables inside _plot_composite_3x3_panel
        # Since we cannot easily inspect local variables without a debugger or modification,
        # we will verify by running the function and checking if it completes without error
        # for all variable types. 
        # Ideally, we would inspect the arguments passed to ax.pcolormesh, but the function
        # creates its own axes.
        
        # Strategy: Mock plt.figure and the axes, then check calls to pcolormesh on the axes.
        
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_fig.add_subplot.return_value = mock_ax
        
        with patch('matplotlib.pyplot.figure', return_value=mock_fig):
            
            # Test UA850
            Visualizer._plot_composite_3x3_panel(composite_results, 3.0, '30Q10_low', 'ssp585', 'Winter',
                                                var_label='UA850', diff_unit='m/s')
            
            # Check if pcolormesh was called with correct vmin/vmax
            # The function makes multiple calls to pcolormesh. We check the last one or any.
            # Arguments are: x, y, data, cmap=..., vmin=..., vmax=...
            # We look for calls with vmin=-5.0 and vmax=5.0
            found_ua = False
            for call in mock_ax.pcolormesh.call_args_list:
                _, kwargs = call
                if kwargs.get('vmin') == -5.0 and kwargs.get('vmax') == 5.0:
                    found_ua = True
                    break
            self.assertTrue(found_ua, "UA850 should have vmin=-5.0 and vmax=5.0")
            
            # Reset mocks
            mock_ax.pcolormesh.reset_mock()
            
            # Test TAS
            Visualizer._plot_composite_3x3_panel(composite_results, 3.0, '30Q10_low', 'ssp585', 'Winter',
                                                var_label='TAS', diff_unit='C')
            found_tas = False
            for call in mock_ax.pcolormesh.call_args_list:
                _, kwargs = call
                if kwargs.get('vmin') == -10.0 and kwargs.get('vmax') == 10.0:
                    found_tas = True
                    break
            self.assertTrue(found_tas, "TAS should have vmin=-10.0 and vmax=10.0")

             # Reset mocks
            mock_ax.pcolormesh.reset_mock()

            # Test PR
            Visualizer._plot_composite_3x3_panel(composite_results, 3.0, '30Q10_low', 'ssp585', 'Winter',
                                                var_label='PR', diff_unit='mm/day')
            found_pr = False
            for call in mock_ax.pcolormesh.call_args_list:
                _, kwargs = call
                if kwargs.get('vmin') == -5.0 and kwargs.get('vmax') == 5.0:
                    found_pr = True
                    break
            self.assertTrue(found_pr, "PR should have vmin=-5.0 and vmax=5.0")

if __name__ == '__main__':
    unittest.main()
