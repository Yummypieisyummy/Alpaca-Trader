"""
UNIVERSAL DATA SCRAPER
======================
Adaptable scraper for any ticker and any time interval from Alpaca

Usage:
    python universal_data_scraper.py AAPL 1 Min 2024-01-01 2025-01-01
    python universal_data_scraper.py SPY 5 Min 2023-01-01 2024-01-01
    python universal_data_scraper.py TSLA 1 Hour 2020-01-01 2023-01-01
    python universal_data_scraper.py NVDA 1 Day 2019-01-01 2025-01-01
"""

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from datetime import datetime
import pandas as pd
import os
import sys
from dotenv import load_dotenv
from pathlib import Path

# Load API credentials
load_dotenv("API.env")
API_KEY = os.getenv("API_KEY")
SECRET_KEY = os.getenv("SECRET_KEY")


def parse_timeframe(interval_value, interval_unit):
    """
    Parse timeframe string to TimeFrame object
    
    Args:
        interval_value: int (e.g., 1, 5, 15)
        interval_unit: str (e.g., 'Min', 'Minute', 'Hour', 'Day')
    
    Returns:
        TimeFrame object
    """
    unit_map = {
        'min': TimeFrameUnit.Minute,
        'minute': TimeFrameUnit.Minute,
        'hour': TimeFrameUnit.Hour,
        'day': TimeFrameUnit.Day,
        'week': TimeFrameUnit.Week,
        'month': TimeFrameUnit.Month,
    }
    
    unit_key = interval_unit.lower().strip()
    if unit_key not in unit_map:
        raise ValueError(f"Invalid interval unit: {interval_unit}. Use Min/Minute/Hour/Day/Week/Month")
    
    return TimeFrame(int(interval_value), unit_map[unit_key])


def download_data(symbol, timeframe, start_date, end_date):
    """
    Download historical data from Alpaca
    
    Args:
        symbol: str (ticker symbol)
        timeframe: TimeFrame object
        start_date: datetime
        end_date: datetime
    
    Returns:
        DataFrame with OHLCV data
    """
    print(f"Downloading {symbol} data...")
    print(f"  Timeframe: {timeframe}")
    print(f"  Period: {start_date} to {end_date}")
    
    client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
    
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=timeframe,
        start=start_date,
        end=end_date,
        adjustment="all"
    )
    
    bars = client.get_stock_bars(request).df
    
    # Clean dataframe
    bars = bars.reset_index()
    bars = bars[bars["symbol"] == symbol]
    
    # Convert timezone to New York
    bars["timestamp"] = pd.to_datetime(bars["timestamp"])
    bars["timestamp"] = bars["timestamp"].dt.tz_convert("America/New_York")
    
    print(f"Downloaded {len(bars)} bars")
    return bars


def clean_intraday_data(df, market_hours_only=True):
    """
    Clean intraday data (minute/hour bars)
    
    Args:
        df: DataFrame with timestamp column
        market_hours_only: bool (keep only 9:30-16:00 ET)
    
    Returns:
        Cleaned DataFrame
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["timestamp"] = df["timestamp"].dt.tz_convert("America/New_York")
    
    if market_hours_only:
        # Keep only regular trading hours (9:30–16:00)
        df["hour"] = df["timestamp"].dt.hour
        df["minute"] = df["timestamp"].dt.minute
        df = df[((df["hour"] == 9) & (df["minute"] >= 30)) | 
                ((df["hour"] > 9) & (df["hour"] < 16)) |
                ((df["hour"] == 16) & (df["minute"] == 0))]
        df = df.drop(columns=["hour", "minute"])
    
    # Sort by timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    return df


def check_missing_bars(df, expected_minutes):
    """
    Check for missing bars in the data
    
    Args:
        df: DataFrame with timestamp column
        expected_minutes: int (expected gap in minutes, e.g., 1, 5, 60)
    """
    df["time_diff"] = df["timestamp"].diff().dt.total_seconds() / 60
    missing = df[df["time_diff"] != expected_minutes]
    
    print(f"\nMissing candles found: {len(missing)}")
    if len(missing) > 0:
        print(missing[["timestamp", "time_diff"]].head(10))
    
    return df.drop(columns=["time_diff"])


def save_data(df, symbol, interval_value, interval_unit, start_date, end_date, output_dir="data"):
    """
    Save data to CSV with standardized filename
    
    Args:
        df: DataFrame to save
        symbol: str (ticker)
        interval_value: int
        interval_unit: str
        start_date: datetime
        end_date: datetime
        output_dir: str (directory to save file)
    
    Returns:
        Path to saved file
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate filename
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")
    unit_short = interval_unit.lower()[:3]  # 'min', 'hou', 'day'

    filename = f"{symbol.lower()}_{interval_value}{unit_short}_{start_date_str}_{end_date_str}.csv"
    filepath = Path(output_dir) / filename
    
    df.to_csv(filepath, index=False)
    print(f"\nData saved to: {filepath}")
    print(f"Total bars: {len(df)}")
    
    return filepath


def main():
    """
    Main function - run from command line
    
    Usage:
        python universal_data_scraper.py SYMBOL INTERVAL_VALUE INTERVAL_UNIT START_DATE END_DATE [MARKET_HOURS_ONLY]
    
    Example:
        python universal_data_scraper.py NVDA 1 Min 2024-01-01 2025-01-01 True
        python universal_data_scraper.py SPY 5 Min 2023-01-01 2024-01-01
        python universal_data_scraper.py TSLA 1 Hour 2020-01-01 2023-01-01 False
    """
    if len(sys.argv) < 6:
        print("Usage: python universal_data_scraper.py SYMBOL INTERVAL_VALUE INTERVAL_UNIT START_DATE END_DATE [MARKET_HOURS_ONLY]")
        print("\nExamples:")
        print("  python universal_data_scraper.py NVDA 1 Min 2024-01-01 2025-01-01")
        print("  python universal_data_scraper.py SPY 5 Min 2023-01-01 2024-01-01 True")
        print("  python universal_data_scraper.py TSLA 1 Hour 2020-01-01 2023-01-01 False")
        print("\nInterval Units: Min/Minute, Hour, Day, Week, Month")
        sys.exit(1)
    
    # Parse arguments
    symbol = sys.argv[1].upper()
    interval_value = int(sys.argv[2])
    interval_unit = sys.argv[3]
    start_date = datetime.strptime(sys.argv[4], "%Y-%m-%d")
    end_date = datetime.strptime(sys.argv[5], "%Y-%m-%d")
    market_hours_only = sys.argv[6].lower() == 'true' if len(sys.argv) > 6 else True
    
    print("=" * 70)
    print(f"UNIVERSAL DATA SCRAPER")
    print("=" * 70)
    print(f"Symbol: {symbol}")
    print(f"Interval: {interval_value} {interval_unit}")
    print(f"Period: {start_date.date()} to {end_date.date()}")
    print(f"Market Hours Only: {market_hours_only}")
    print("=" * 70)
    
    # Parse timeframe
    try:
        timeframe = parse_timeframe(interval_value, interval_unit)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Download data
    df = download_data(symbol, timeframe, start_date, end_date)
    
    if df.empty:
        print("No data downloaded. Check symbol and date range.")
        sys.exit(1)
    
    print("\nDownload complete.")
    print(df.head())
    
    # Clean data (only for intraday)
    if interval_unit.lower() in ['min', 'minute', 'hour']:
        df = clean_intraday_data(df, market_hours_only=market_hours_only)
        print("\nCleaning complete.")
        print(df.head())
        
        # Check for missing bars
        expected_minutes = interval_value if interval_unit.lower() in ['min', 'minute'] else interval_value * 60
        df = check_missing_bars(df, expected_minutes)
    
    # Save data
    filepath = save_data(df, symbol, interval_value, interval_unit, start_date, end_date)
    
    # Show summary
    df["date"] = df["timestamp"].dt.date
    df["time"] = df["timestamp"].dt.time
    print("\nFirst/Last timestamps per day:")
    print(df.groupby("date")[["time"]].agg(['min', 'max']))
    
    print("\n✅ Complete! Data ready for backtesting.")


if __name__ == "__main__":
    main()