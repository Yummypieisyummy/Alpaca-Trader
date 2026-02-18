'''
Docstring for General Trader
Trade arguments:

    1. Symbol (e.g., AAPL)
    2. Interval value (e.g., 1)
    3. Interval unit (e.g., min, hour, day)
    4. Start date (YYYY-MM-DD)
    5. End date (YYYY-MM-DD)
    6. Market hours only (true/false, optional, default true)

'''
import sys
from datetime import datetime
import pandas as pd

if len(sys.argv) < 6:
    print("Usage: python General Trader.py SYMBOL INTERVAL_VALUE INTERVAL_UNIT START_DATE END_DATE [MARKET_HOURS_ONLY]")
    sys.exit(1)

symbol = sys.argv[1].upper()
interval_value = int(sys.argv[2])
interval_unit = sys.argv[3]
start_date = datetime.strptime(sys.argv[4], "%Y-%m-%d")
end_date = datetime.strptime(sys.argv[5], "%Y-%m-%d")
market_hours_only = sys.argv[6].lower() == 'true' if len(sys.argv) > 6 else True

df = pd.read_csv(base_path / "data" / f"{symbol.lower()}_{interval_value}{unit_short}_{start_year}-{end_year}_cleaned.csv")