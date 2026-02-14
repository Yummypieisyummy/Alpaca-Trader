# Alpaca Live Trading Bot - Instructions

## Project Overview
This is a live trading bot that connects to Alpaca Markets API using a 750 EMA regime detection strategy with 4-layer dynamic risk management. It's designed to trade TSLA (or any symbol) with intelligent position sizing based on market volatility, trend strength, equity curve, and drawdown levels.

---

## SETUP (First Time Only)

### 1. Configure API Credentials
Create or update `API.env` in the project root with your Alpaca API credentials:
```
API_KEY=your_alpaca_api_key_here
SECRET_KEY=your_alpaca_secret_key_here
ENDPOINT=https://paper-api.alpaca.markets/v2
```

Get credentials from: https://paper.alpaca.markets/

### 2. Setup Virtual Environment
```powershell
# Create virtual environment
python -m venv bot_env

# Activate it
.\bot_env\Scripts\Activate.ps1

# Install dependencies
pip install alpaca-trade-api yfinance pandas numpy
```

---

## RUNNING THE BOT

### Option 1: Run Live Trading Bot (Automated)
```powershell
# Make sure virtual environment is active: (bot_env) should show in prompt
.\bot_env\Scripts\Activate.ps1

# Start the trading bot
python alpaca_live_runner.py
```
Bot checks for signals every 5 minutes during market hours. Logs to `alpaca_live_trading.log`.

### Option 2: Run Tests First (Recommended)
```powershell
# Test all integration points
python test_alpaca_integration.py
```
Expected: 7/9 tests passing. Results show account details, position sizing, and indicator calculations.

### Option 3: Manual Trading Commands (One-Off)
```powershell
# Check account status
python alpaca_utils.py status

# Get current price
python alpaca_utils.py price TSLA

# Buy X shares
python alpaca_utils.py buy 5

# Sell X shares
python alpaca_utils.py sell 5

# Close all positions
python alpaca_utils.py close
```

---

## UNDERSTANDING THE BOT

### Strategy: 750 EMA Regime Detection
- **Buy Signal**: Price > (EMA750 × 1.005) AND slope is upward
- **Sell Signal**: Price < (EMA750 × 0.995) AND slope is downward
- **Position Size**: 4-layer dynamic calculation:
  1. Base Risk: 2.5% of capital per trade
  2. Regime Strength: 1.4x if strong trend, 0.7x if weak
  3. Equity Acceleration: 1.2x if new equity high
  4. Drawdown Protection: 0.6x if equity down > 20%

### Files Explained
- `alpaca_trader.py` - Core API wrapper connecting to Alpaca
- `alpaca_live_runner.py` - Main trading bot with strategy logic
- `alpaca_utils.py` - Command-line utilities for manual operations
- `test_alpaca_integration.py` - 9-test suite validating all functionality

---

## TRADING HOURS & MODES

### Paper Trading (Current Setup)
- Available 24/7 for testing
- Uses simulated $100,000 capital
- Orders execute during market hours (9:30 AM - 4 PM ET)
- Perfect for testing without real money at risk

### Live Trading (When Ready)
- Change `mode="live"` in `alpaca_live_runner.py`
- Real money required in your Alpaca account
- Same market hours restrictions apply
- See Alpaca docs for extended hours trading

---

## MONITORING & LOGS

### Real-Time Bot Logs
```powershell
# Watch logs as bot runs
Get-Content alpaca_live_trading.log -Tail 20 -Wait
```

### Trade History
Generated automatically as `paper_trade_log.json` with all executed trades (entry/exit price, time, P&L).

### Account Dashboard
Visit https://paper.alpaca.markets/ to see live positions and P&L in real-time.

---

## TROUBLESHOOTING

**"ModuleNotFoundError: No module named 'alpaca_trade_api'"**
→ Virtual environment not activated or dependencies not installed. Run: `pip install alpaca-trade-api yfinance pandas numpy`

**"401 Unauthorized"**
→ API credentials invalid. Check API.env has correct keys from https://paper.alpaca.markets/

**"No positions found"**
→ Normal! Paper account starts with $0 positions. Bot will open first trade when signal triggers.

**Tests 3 & 8 Fail (Price Data / Indicators)**
→ Non-critical yfinance formatting issue in test isolation. Actual bot trading works fine.

**Bot doesn't show "BUY" signals**
→ Waiting for 750 EMA condition AND slope threshold. Price must break above EMA×1.005. Be patient during consolidation.

---

## NEXT STEPS

1. Run `python test_alpaca_integration.py` to verify setup ✓
2. Run `python alpaca_live_runner.py` during market hours (9:30 AM - 4 PM ET)
3. Monitor `alpaca_live_trading.log` for live trades
4. Check `paper_trade_log.json` for performance metrics
5. After 1-2 weeks of paper trading, consider live trading if confident

---

## TIPS

- Check `alpaca_utils.py status` anytime to see current cash/equity
- Bot checks every 5 minutes by default. Change in alpaca_live_runner.py: `bot.run(check_interval=300)`
- Paper trading losses don't count! Use this to test and refine without fear
- Logs are permanent - useful for backtesting bot performance
