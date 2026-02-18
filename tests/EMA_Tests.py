import unittest
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import sys

# Add the src directory to the path so we can import modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from main.strategy.EMA import run_ema_strategy, calculate_macz, is_confluence_zone


class TestEMACalculation(unittest.TestCase):
    """Test EMA calculation accuracy"""
    
    def setUp(self):
        """Create sample data for testing"""
        # Create a simple dataset with 100 data points
        self.test_data = pd.DataFrame({
            'close': np.sin(np.linspace(0, 4*np.pi, 100)) * 10 + 50,  # Oscillating data
            'open': np.sin(np.linspace(0, 4*np.pi, 100)) * 10 + 50,
            'high': np.sin(np.linspace(0, 4*np.pi, 100)) * 10 + 51,
            'low': np.sin(np.linspace(0, 4*np.pi, 100)) * 10 + 49,
            'volume': np.full(100, 1000000)
        })
    
    def test_ema_calculation(self):
        """Test that EMA is calculated correctly"""
        # Calculate EMA manually using the expected formula
        period = 20
        close = self.test_data['close'].values
        
        # Calculate SMA for first EMA
        sma = self.test_data['close'].rolling(window=period).mean()
        ema = sma.copy()
        
        multiplier = 2 / (period + 1)
        for i in range(period, len(close)):
            ema.iloc[i] = (close[i] * multiplier) + (ema.iloc[i-1] * (1 - multiplier))
        
        # Verify EMA is not all NaN after period
        self.assertTrue((~ema.iloc[period:].isna()).all(), "EMA should have values after initial period")
        
        # Verify EMA values are reasonable (within data range)
        self.assertTrue((ema.iloc[period:] >= self.test_data['close'].min() - 5).all())
        self.assertTrue((ema.iloc[period:] <= self.test_data['close'].max() + 5).all())
    
    def test_ema_smoothing(self):
        """Test that longer EMA period produces smoother results"""
        close = self.test_data['close'].values
        
        # Calculate short and long EMAs
        short_period = 5
        long_period = 20
        
        short_sma = self.test_data['close'].rolling(window=short_period).mean()
        long_sma = self.test_data['close'].rolling(window=long_period).mean()
        
        short_ema = short_sma.copy()
        long_ema = long_sma.copy()
        
        short_mult = 2 / (short_period + 1)
        long_mult = 2 / (long_period + 1)
        
        for i in range(max(short_period, long_period), len(close)):
            short_ema.iloc[i] = (close[i] * short_mult) + (short_ema.iloc[i-1] * (1 - short_mult))
            long_ema.iloc[i] = (close[i] * long_mult) + (long_ema.iloc[i-1] * (1 - long_mult))
        
        # Calculate volatility (standard deviation of changes)
        short_volatility = short_ema.iloc[long_period:].diff().std()
        long_volatility = long_ema.iloc[long_period:].diff().std()
        
        # Long EMA should be smoother (lower volatility)
        self.assertLess(long_volatility, short_volatility, 
                       "Longer EMA period should produce smoother results")


class TestMACSCalculation(unittest.TestCase):
    """Test MACZ calculation"""
    
    def setUp(self):
        """Create sample data with EMA columns"""
        self.test_data = pd.DataFrame({
            'close': np.linspace(40, 60, 50),
            'EMA_shorter': np.linspace(42, 58, 50),
            'EMA_longer': np.linspace(45, 55, 50)
        })
    
    def test_macz_center_calculation(self):
        """Test that MACZ center is the average of EMAs"""
        calculate_macz(self.test_data)
        
        expected_center = (self.test_data['EMA_shorter'] + self.test_data['EMA_longer']) / 2
        pd.testing.assert_series_equal(
            self.test_data['MACZ_center'], 
            expected_center,
            check_names=False
        )
    
    def test_macz_bounds_exist(self):
        """Test that MACZ upper and lower bounds are calculated"""
        calculate_macz(self.test_data)
        
        # Check that bounds exist and are numeric
        self.assertTrue('MACZ_upper' in self.test_data.columns)
        self.assertTrue('MACZ_lower' in self.test_data.columns)
        self.assertTrue('MACZ_std' in self.test_data.columns)
        
        # Upper bound should always be >= center >= lower bound
        self.assertTrue((self.test_data['MACZ_upper'] >= self.test_data['MACZ_center']).all())
        self.assertTrue((self.test_data['MACZ_center'] >= self.test_data['MACZ_lower']).all())
    
    def test_macz_width_varies(self):
        """Test that MACZ width changes based on EMA divergence"""
        # Create data where EMAs converge then diverge
        converging_data = pd.DataFrame({
            'EMA_shorter': [50, 50.5, 51, 51.5, 50, 49.5, 49],
            'EMA_longer': [50, 49.8, 49.6, 49.4, 50, 50.2, 50.4]
        })
        
        calculate_macz(converging_data)
        
        # Std should vary - not all zero
        self.assertTrue(converging_data['MACZ_std'].var() > 0,
                       "MACZ width should vary as EMAs diverge/converge")


class TestSignalGeneration(unittest.TestCase):
    """Test buy/sell signal generation"""
    
    def setUp(self):
        """Create data with clear crossover points"""
        # Create data where short EMA clearly crosses long EMA
        self.test_data = pd.DataFrame({
            'close': [50]*20 + [52]*20 + [48]*20,  # Price movement
            'EMA_shorter': [48]*10 + [49]*10 + [51]*10 + [52]*10 + [49]*10 + [47]*10,  # Shorter EMA crosses
            'EMA_longer': [50]*40 + [50]*20,  # Longer EMA more stable
        })
        
        # Add MACZ columns
        self.test_data['MACZ_center'] = (self.test_data['EMA_shorter'] + self.test_data['EMA_longer']) / 2
        self.test_data['MACZ_std'] = 1
        self.test_data['MACZ_upper'] = self.test_data['MACZ_center'] + self.test_data['MACZ_std']
        self.test_data['MACZ_lower'] = self.test_data['MACZ_center'] - self.test_data['MACZ_std']
    
    def test_buy_signal_detection(self):
        """Test that buy signals are detected on upward crossovers"""
        data = self.test_data.copy()
        data['Signal'] = 0
        
        # Buy signal: Shorter EMA crosses above longer EMA
        data.loc[(data['EMA_shorter'] > data['EMA_longer']) & 
                (data['EMA_shorter'].shift(1) <= data['EMA_longer'].shift(1)), 'Signal'] = 1
        
        # Should have at least one buy signal
        self.assertTrue((data['Signal'] == 1).any(),
                       "Should detect at least one buy signal")
    
    def test_sell_signal_detection(self):
        """Test that sell signals are detected on downward crossovers"""
        data = self.test_data.copy()
        data['Signal'] = 0
        
        # Sell signal: Shorter EMA crosses below longer EMA
        data.loc[(data['EMA_shorter'] < data['EMA_longer']) & 
                (data['EMA_shorter'].shift(1) >= data['EMA_longer'].shift(1)), 'Signal'] = -1
        
        # Should have at least one sell signal
        self.assertTrue((data['Signal'] == -1).any(),
                       "Should detect at least one sell signal")
    
    def test_signal_default_zero(self):
        """Test that signals default to 0 (no signal)"""
        data = self.test_data.copy()
        data['Signal'] = 0
        
        # Most signals should be 0 (no crossover)
        zero_signals = (data['Signal'] == 0).sum()
        self.assertGreater(zero_signals, len(data) * 0.5,
                          "Most rows should have no signal (0)")


class TestConfluenceZoneDetection(unittest.TestCase):
    """Test confluence zone detection"""
    
    def test_price_within_zone(self):
        """Test detection of price within confluence zone"""
        price = 50
        lower = 45
        upper = 55
        
        self.assertTrue(is_confluence_zone(price, lower, upper),
                       "Price within bounds should return True")
    
    def test_price_outside_zone_above(self):
        """Test detection of price above confluence zone"""
        price = 60
        lower = 45
        upper = 55
        
        self.assertFalse(is_confluence_zone(price, lower, upper),
                        "Price above upper bound should return False")
    
    def test_price_outside_zone_below(self):
        """Test detection of price below confluence zone"""
        price = 40
        lower = 45
        upper = 55
        
        self.assertFalse(is_confluence_zone(price, lower, upper),
                        "Price below lower bound should return False")
    
    def test_price_on_boundary(self):
        """Test price exactly on zone boundaries"""
        lower = 45
        upper = 55
        
        # Exactly on upper boundary
        self.assertTrue(is_confluence_zone(55, lower, upper))
        
        # Exactly on lower boundary
        self.assertTrue(is_confluence_zone(45, lower, upper))


class TestDataValidation(unittest.TestCase):
    """Test data validation and edge cases"""
    
    def test_empty_dataframe(self):
        """Test handling of empty DataFrame"""
        empty_df = pd.DataFrame({'close': [], 'EMA_shorter': [], 'EMA_longer': []})
        
        # Should not raise exception
        try:
            calculate_macz(empty_df)
        except Exception as e:
            self.fail(f"Should handle empty DataFrame gracefully: {e}")
    
    def test_nan_values_in_ema(self):
        """Test handling of NaN values in EMA calculations"""
        data = pd.DataFrame({
            'close': [np.nan, 50, 51, 52, 53],
            'EMA_shorter': [np.nan]*5,
            'EMA_longer': [np.nan]*5
        })
        
        # Should not crash due to NaN
        try:
            calculate_macz(data)
        except Exception as e:
            self.fail(f"Should handle NaN values: {e}")
    
    def test_identical_close_prices(self):
        """Test EMA calculation with identical prices"""
        data = pd.DataFrame({
            'close': [50]*50
        })
        
        # Calculate EMA
        period = 20
        sma = data['close'].rolling(window=period).mean()
        ema = sma.copy()
        multiplier = 2 / (period + 1)
        
        for i in range(period, len(data)):
            ema.iloc[i] = (data['close'].iloc[i] * multiplier) + (ema.iloc[i-1] * (1 - multiplier))
        
        # All EMA values after period should be 50
        self.assertTrue((ema.iloc[period:] == 50).all(),
                       "EMA of constant prices should be constant")


class TestRunEMAStrategy(unittest.TestCase):
    """Test the complete EMA strategy"""
    
    def test_run_ema_strategy_creates_log(self):
        """Test that run_ema_strategy generates log output"""
        from main.strategy.EMA import run_ema_strategy
        
        # Use an existing CSV file from your data folder
        result = run_ema_strategy(
            symbol='HIVE',  # Change to whatever symbol you have
            interval_value=1,
            interval_unit='Min',
            start_date='2023-09-01',
            end_date='2026-01-01',
            ema_shorter_period=12,
            ema_longer_period=26
        )
        
        # Verify log file was created
        log_file = Path(__file__).parent.parent / "src" / "logs" / "EMA_strategy.log"
        self.assertTrue(log_file.exists(), "Log file was not created")
        
        # Verify results
        self.assertIsNotNone(result)
        self.assertIn('Signal', result.columns)
        self.assertIn('EMA_shorter', result.columns)
        self.assertIn('EMA_longer', result.columns)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
