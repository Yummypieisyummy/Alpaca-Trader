"""
ALPACA INTEGRATION QUICK START GUIDE
====================================

Three files work together to connect trading to Alpaca:

1. alpaca_trader.py - Core API wrapper
2. alpaca_live_runner.py - Paper trading bot with strategy logic  
3. This guide

SETUP & TESTING
===============

✓ API.env credentials loaded
✓ Account verified (equity, buying power accessible)
✓ Paper trading mode enabled
✓ Data connections tested


WHAT'S WORKING
==============

✓ Account Information
  - Get equity, cash, buying power
  - Track day trading limits
  - Monitor portfolio percentage

✓ Order Placement
  - Place buy/sell market orders
  - Track order status
  - Log all trades

✓ Position Tracking
  - Get current holdings
  - Monitor unrealized P&L
  - Check average fill prices

✓ Market Data
  - Get latest prices
  - Retrieve OHLCV bars (subscription permitting)
  - Historical data from CSV fallback


CURRENT STATUS
==============

BOT ACCOUNT:
  Equity: $100,000 (paper trading)
  Buying Power: $200,000 (2:1 margin)
  Current Position: NONE

LATEST DATA:
  Symbol: TSLA
  Price: $403.02
  Date Range: 2019-01-02 to 2024-12-31 (119,199 bars)
  
4-LAYER RISK SYSTEM:
  Layer 1: 2.5% risk per trade
  Layer 2: Slope-based sizing (1.4x strong / 0.7x weak)
  Layer 3: Equity acceleration (1.2x on new highs)
  Layer 4: Drawdown throttle (0.6x below -20% DD)


RUNNING THE BOT
===============

Option 1: Test without trading (current)
  python alpaca_live_runner.py
  
  Shows: Account balance, latest price, market state
  No orders placed

Option 2: Enable live paper trading loop
  
  Edit: alpaca_live_runner.py, line ~350
  Uncomment: bot.run(check_interval=300)
  
  This will:
  - Check for trading signals every 5 minutes
  - Place buy orders on entry signals
  - Place sell orders on exit signals
  - Log all trades to trade_log.json
  - Print summary on shutdown

Option 3: Test single order
  
  In Python:
  >>> from alpaca_trader import AlpacaTrader
  >>> trader = AlpacaTrader(symbol="TSLA", mode="paper")
  >>> order = trader.place_buy_order(10)
  >>> print(order.id)


KEY FEATURES IMPLEMENTED
========================

MARKET PRICE SIGNALS:
  - 750-period EMA trend detection
  - Entry: price > EMA×1.005 AND slope > threshold
  - Exit: price < EMA×0.995 AND slope < threshold
  - Adaptive thresholds based on volatility

POSITION SIZING (4 LAYERS):
  - Base: 2.5% risk × (capital / ATR×1.8)
  - Regime strength: multiply by slope score
  - Equity boost: multiply by 1.2 on new highs
  - Drawdown protection: multiply by 0.6 if DD < -20%

ERROR HANDLING:
  - Automatic retry on API failures
  - Graceful degradation if data unavailable
  - Comprehensive logging to file + console
  - Position closure on shutdown

RECORD KEEPING:
  - All orders logged with timestamps
  - Trade log exported to JSON
  - Performance metrics calculated
  - Win rate, profit factor, max drawdown tracked


TRADE LOG OUTPUT
================

When running with bot.run(), creates:
  - paper_trading.log (detailed events)
  - paper_trade_log.json (exportable data)
  
Each trade logged with:
  - Entry/exit timestamp
  - Number of shares
  - Price executed
  - Realized P&L


NEXT STEPS
==========

1. RUN BOT WITH TRADING:
   Uncomment bot.run() in alpaca_live_runner.py
   Watch for entry/exit signals in logs

2. CUSTOMIZE TRADING HOURS:
   Alpaca paper trading runs 24/5
   You can trade pre/post market or halt during holidays

3. ADD STOP LOSSES:
   Current: regime-based only  
   Could add: hard ATR stops or % stops if desired

4. INTEGRATE ALERTS:
   Current: logging only
   Could add: email/SMS notification on trades

5. SWITCH TO LIVE:
   When confident, change mode="live" in AlpacaTrader
   Same API, real account (be careful!)


IMPORTANT NOTES
===============

⚠ PAPER TRADING MODE
   - Real Alpaca account but simulated funds
   - No real money at risk
   - Good for testing execution logic
   - Fills may differ from live (slippage varies)

⚠ SUBSCRIPTION LIMITATIONS  
   - Real-time bars require SIP subscription
   - Fallback to CSV data works fine (already loaded)
   - Price quotes always available (free tier)

⚠ MARGIN ACCOUNR
   - Alpaca gives 2:1 margin on paper accounts
   - Don't over-leverage (could hit limits)
   - Position sizing caps at 90% of capital

⚠ WEEKEND/HOLIDAYS
   - Market closed: no order fills
   - Orders placed will queue until open
   - Be mindful of gap risk


DEBUGGING TIPS
==============

Check connection:
  python alpaca_trader.py
  (Should show account equity, current price)

View logs:
  Get-Content paper_trading.log -Tail 50
  (Last 50 lines of trading log)

Monitor orders:
  In alpaca_live_runner.py, add:
  >>> trader.trader.client.list_orders()
  (Show all recent orders)

Check position:
  >>> bot.trader.get_position()
  (Current holdings and P&L)


COMPARISON TO BACKTESTER
========================

backtester_v4.py
  - Uses CSV data (historical)
  - Simulates execution instantly
  - Shows theoretical returns
  - Good for strategy testing
  
alpaca_live_runner.py
  - Uses CSV data loaded at startup
  - Executes via Alpaca API
  - Real order fills and slippage
  - Good for execution testing
  
When to use each:
  1. Backtest strategy on historical data (backtester_v4)
  2. Paper trade to verify execution (alpaca_live_runner)
  3. Live trade when confident (mode="live")

PAPER TRADING RESULTS SO FAR
=============================

Initial Capital: $100,000
Current Equity: $100,000 (starting)
Position: None (waiting for entry signal)
Mode: Paper Trading (safe testing)

Ready to source real entry signals once bot.run() uncommented.

---

Questions? Review the detailed code comments in:
  - alpaca_trader.py (API wrapper details)
  - alpaca_live_runner.py (strategy logic)
"""

if __name__ == "__main__":
    import textwrap
    doc = __doc__
    print(textwrap.dedent(doc))
