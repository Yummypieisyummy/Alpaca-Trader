'''
EMA strategy implementation. Uses arguments passed as the X - day EMA to determine buy/sell signals. Buy when price crosses above the EMA, sell when it crosses below.

Arguments:
    1. Symbol (e.g., AAPL)
    2. Interval value (e.g., 1)
    3. Interval unit (e.g., min, hour, day)
    4. Start date (YYYY-MM-DD)
    5. End date (YYYY-MM-DD)
    6. EMA shorter period (e.g., 20)
    7. EMA longer period (e.g., 50)
    8. Market hours only (true/false, optional, default true)

EMA Strategy Logic:
    1. Calculate the shorter and longer EMAs based on the specified periods.
    2. Generate buy signals when the shorter EMA crosses above the longer EMA during a moment of confluence. Make longer trades
    2a. Check price action and volume to confirm the buy signal, ensuring that the crossover is supported by strong market activity.
     - Look for increased volume and bullish price patterns (e.g., higher highs, higher lows) to validate the signal.
     - Avoid entering trades during low volume periods or when price action is weak, as these may indicate a false signal.
    3. Generate sell signals when the shorter EMA crosses below the longer EMA during a moment of confluence.

How EMA is calculated:
    Calculating the EMA needs one more observation than the SMA. 
    If you choose 20 days for your EMA, wait until the 20th day to get the SMA. Then, use that SMA as the first 
    EMA on the 21st day.

    Calculating the SMA is straightforward. It involves adding up the stock's closing prices over a period and 
    then dividing by the number of observations. For example, a 20-day SMA is just the sum of the closing prices
    for the past 20 trading days, divided by 20.

    Next, you must calculate the multiplier for smoothing (weighting) the EMA, which typically follows the formula:
    [2 ÷ (number of observations + 1)]. For a 20-day moving average, the multiplier would be [2/(20+1)]= 0.0952.

    Use this formula to calculate the current EMA:
    EMA = Closing price x multiplier + EMA (previous day) x (1-multiplier)
'''
import sys
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Set up logging
log_directory = Path(__file__).parent.parent.parent / "logs"
log_directory.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_directory / "EMA_strategy.log"),
        logging.StreamHandler()  # Also print to console
    ]
)
logger = logging.getLogger(__name__)

base_path = Path(__file__).parent.parent.parent.parent

def run_ema_strategy(symbol, interval_value, interval_unit, start_date, end_date, ema_shorter_period, ema_longer_period, market_hours_only=True):
    '''
    Execute the EMA strategy with the given parameters.
    
    Args:
        symbol: Stock symbol (e.g., 'AAPL')
        interval_value: Interval value (e.g., 1)
        interval_unit: Interval unit (e.g., 'min', 'hour', 'day')
        start_date: Start date (datetime object or YYYY-MM-DD string)
        end_date: End date (datetime object or YYYY-MM-DD string)
        ema_shorter_period: Shorter EMA period (e.g., 20)
        ema_longer_period: Longer EMA period (e.g., 50)
        market_hours_only: Whether to filter market hours only (default: True)
    
    Returns:
        DataFrame with strategy signals and analysis
    '''
    logger.info("Starting EMA strategy execution...")

    try:
        # Convert strings to datetime if necessary
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d")
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d")

        logger.info(f"Parameters - Symbol: {symbol}, Interval: {interval_value} {interval_unit}, Start Date: {start_date}, End Date: {end_date}, EMA Shorter Period: {ema_shorter_period}, EMA Longer Period: {ema_longer_period}, Market Hours Only: {market_hours_only}")

        # Load data
        market_data = pd.read_csv(base_path / "data" / f"{symbol.lower()}_{interval_value}{interval_unit}_{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}.csv")
        logger.info(f"Data loaded for {symbol}")
        logger.info(f"Loaded {len(market_data)} rows of data from {start_date} to {end_date}")

        #Calculate SMA for each period to get the first EMA value
        sma_shorter = market_data['close'].rolling(window=ema_shorter_period).mean()
        sma_longer = market_data['close'].rolling(window=ema_longer_period).mean()

        # Calculate the multiplier for smoothing the EMA
        multiplier_shorter = 2 / (ema_shorter_period + 1)
        multiplier_longer = 2 / (ema_longer_period + 1)

        # Start from the first EMA value (after the SMA period)
        market_data['EMA_shorter'] = sma_shorter
        market_data['EMA_longer'] = sma_longer

        logger.info("Calculating EMA values...")
        logger.debug(f"EMA calculations complete. Sample values:\n{market_data[['close', 'EMA_shorter', 'EMA_longer']].head()}")

        # Calculate EMA for each period starting from the first EMA value (EMA starts after the SMA period)
        for i in range(max(ema_shorter_period, ema_longer_period), len(market_data)):
            market_data.at[i, 'EMA_shorter'] = (market_data.at[i, 'close'] * multiplier_shorter) + (market_data.at[i-1, 'EMA_shorter'] * (1 - multiplier_shorter))
            market_data.at[i, 'EMA_longer'] = (market_data.at[i, 'close'] * multiplier_longer) + (market_data.at[i-1, 'EMA_longer'] * (1 - multiplier_longer))


        market_data = calculate_macz(market_data)

        # Generate buy/sell signals based on EMA crossovers
        market_data['Signal'] = 0  # Default to no signal

        # Buy signal: Shorter EMA crosses above longer EMA
        # This checks if the shorter-period EMA is currently above the longer-period EMA. This indicates the current trend is bullish.
        # The second part checks if the shorter-period EMA was previously below or equal to the longer-period EMA. This confirms that a crossover has occurred, signaling a potential buy opportunity.
        buy_crossover = (market_data['EMA_shorter'] > market_data['EMA_longer']) & (market_data['EMA_shorter'].shift(1) <= market_data['EMA_longer'].shift(1))

        in_confluence_zone = is_confluence_zone(
            market_data['close'],
            market_data['MACZ_lower'],
            market_data['MACZ_upper']
        )

        market_data.loc[buy_crossover & in_confluence_zone, 'Signal'] = 1  # Buy signal

        # Sell signal: Shorter EMA crosses below longer EMA
        # This checks if the shorter-period EMA is currently below the longer-period EMA. This indicates the current trend is bearish.
        # The second part checks if the shorter-period EMA was previously above or equal to the longer-period EMA. This confirms that a crossover has occurred, signaling a potential sell opportunity.
        market_data.loc[(market_data['EMA_shorter'] < market_data['EMA_longer']) & (market_data['EMA_shorter'].shift(1) >= market_data['EMA_longer'].shift(1)), 'Signal'] = -1  # Sell signal

        logger.info("EMA strategy execution complete.")
        logger.debug(f"Final strategy signals:\n{market_data[['close', 'EMA_shorter', 'EMA_longer', 'MACZ_lower', 'MACZ_upper', 'Signal']].head()}")
        logger.debug(f".\n.\n.\n.\n")
        logger.debug(f"Final strategy signals:\n{market_data[['close', 'EMA_shorter', 'EMA_longer', 'MACZ_lower', 'MACZ_upper', 'Signal']].tail()}")

        return market_data
    
    except Exception as e:
        logger.error(f"Error during EMA strategy execution: {e}")
        raise

def calculate_macz(market_data):
    '''
    Calculate Moving Average Confluence Zone (MACZ).
    
    Args:
        market_data: DataFrame with 'EMA_shorter' and 'EMA_longer' columns
        ema_shorter_period: Shorter EMA period (for reference)
        ema_longer_period: Longer EMA period (for reference)
    
    Returns:
        DataFrame with MACZ columns added
    '''
    market_data['MACZ_center'] = (market_data['EMA_shorter'] + market_data['EMA_longer']) / 2
    market_data['MACZ_std'] = market_data[['EMA_shorter', 'EMA_longer']].std(axis=1)
    market_data['MACZ_upper'] = market_data['MACZ_center'] + market_data['MACZ_std']
    market_data['MACZ_lower'] = market_data['MACZ_center'] - market_data['MACZ_std']
    
    return market_data

def is_confluence_zone(price, macz_lower, macz_upper):
    '''
    Check if price is within confluence zone for a row
    
    Args:
        price: Current price (e.g., close price)
        macz_lower: Lower bound of MACZ
        macz_upper: Upper bound of MACZ

    Returns:
        bool: True if price is within the confluence zone, False otherwise
    '''
    return (price >= macz_lower) & (price <= macz_upper)

# CLI version of EMA strategy ----------------------------------------------------------
if __name__ == "__main__":
    # Check for required arguments, exit if not provided
    if len(sys.argv) < 8:
        print("Usage: python EMA.py SYMBOL INTERVAL_VALUE INTERVAL_UNIT START_DATE END_DATE EMA_SHORTER_PERIOD EMA_LONGER_PERIOD [MARKET_HOURS_ONLY]")
        sys.exit(1)

    symbol = sys.argv[1].upper()
    interval_value = int(sys.argv[2])
    interval_unit = sys.argv[3]
    start_date = datetime.strptime(sys.argv[4], "%Y-%m-%d")
    end_date = datetime.strptime(sys.argv[5], "%Y-%m-%d")
    ema_shorter_period = int(sys.argv[6])
    ema_longer_period = int(sys.argv[7])
    market_hours_only = sys.argv[8].lower() == "true" if len(sys.argv) > 8 else True

    result = run_ema_strategy(symbol, interval_value, interval_unit, start_date, end_date, ema_shorter_period, ema_longer_period, market_hours_only)
# End of CLI version of EMA strategy ---------------------------------------------------
