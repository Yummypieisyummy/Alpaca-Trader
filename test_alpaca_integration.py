"""
ALPACA INTEGRATION TEST SUITE
==============================
Validates all trading functionality before live deployment

Tests:
  1. Connection & Authentication
  2. Account Info Retrieval
  3. Price Data Fetching
  4. Position Tracking
  5. Dollar-to-Share Conversion
  6. Order Placement (paper trading)
  7. Trade Logging
"""

import sys
import os
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project to path
sys.path.append(r"c:\Users\Ryan\Alpaca Trader")

from alpaca_trader import AlpacaTrader
from alpaca_live_runner import AlpacaLiveBot

def test_connection():
    """Test 1: API Connection & Authentication"""
    logger.info("\n" + "="*60)
    logger.info("TEST 1: Connection & Authentication")
    logger.info("="*60)
    
    try:
        trader = AlpacaTrader(symbol="TSLA", mode="paper")
        logger.info("✓ Connection successful")
        return True
    except Exception as e:
        logger.error(f"✗ Connection failed: {e}")
        return False

def test_account_info():
    """Test 2: Account Info Retrieval"""
    logger.info("\n" + "="*60)
    logger.info("TEST 2: Account Info Retrieval")
    logger.info("="*60)
    
    try:
        trader = AlpacaTrader(symbol="TSLA", mode="paper")
        account = trader.get_account_info()
        
        logger.info(f"✓ Account retrieved:")
        logger.info(f"  Equity: ${account['equity']:,.2f}")
        logger.info(f"  Cash: ${account['cash']:,.2f}")
        logger.info(f"  Buying Power: ${account['buying_power']:,.2f}")
        logger.info(f"  Portfolio Value: ${account['portfolio_value']:,.2f}")
        
        assert account['equity'] > 0, "Equity should be positive"
        assert account['buying_power'] > 0, "Buying power should be positive"
        return True
    except Exception as e:
        logger.error(f"✗ Account info failed: {e}")
        return False

def test_price_data():
    """Test 3: Price Data Fetching"""
    logger.info("\n" + "="*60)
    logger.info("TEST 3: Price Data Fetching")
    logger.info("="*60)
    
    try:
        trader = AlpacaTrader(symbol="TSLA", mode="paper")
        
        # Get latest price
        price = trader.get_latest_price()
        logger.info(f"✓ Latest Price: ${price:.2f}")
        
        # Get historical bars
        bars = trader.get_historical_bars(timeframe="5Min", days_back=1)
        logger.info(f"✓ Historical bars: {len(bars)} candles")
        logger.info(f"  Date range: {bars.index[0]} to {bars.index[-1]}")
        logger.info(f"  High: ${bars['high'].max():.2f}")
        logger.info(f"  Low: ${bars['low'].min():.2f}")
        
        assert price > 0, "Price should be positive"
        assert len(bars) > 0, "Should have historical data"
        return True
    except Exception as e:
        logger.error(f"✗ Price data failed: {e}")
        return False

def test_position_tracking():
    """Test 4: Position Tracking"""
    logger.info("\n" + "="*60)
    logger.info("TEST 4: Position Tracking")
    logger.info("="*60)
    
    try:
        trader = AlpacaTrader(symbol="TSLA", mode="paper")
        position = trader.get_position()
        
        if position:
            logger.info(f"✓ Position found:")
            logger.info(f"  Shares: {position['shares']}")
            logger.info(f"  Avg Fill Price: ${position['avg_fill_price']:.2f}")
            logger.info(f"  Current Price: ${position['current_price']:.2f}")
            logger.info(f"  Market Value: ${position['market_value']:.2f}")
            logger.info(f"  Unrealized PnL: ${position['unrealized_pl']:.2f} ({position['unrealized_plpc']:.2f}%)")
        else:
            logger.info("✓ No position (clean state)")
        
        return True
    except Exception as e:
        logger.error(f"✗ Position tracking failed: {e}")
        return False

def test_dollar_conversion():
    """Test 5: Dollar-to-Share Conversion"""
    logger.info("\n" + "="*60)
    logger.info("TEST 5: Dollar-to-Share Conversion")
    logger.info("="*60)
    
    try:
        trader = AlpacaTrader(symbol="TSLA", mode="paper")
        price = trader.get_latest_price()
        
        # Test conversions
        test_amounts = [500, 1000, 2500, 5000]
        
        for dollars in test_amounts:
            shares = int(dollars / price)
            actual_cost = shares * price
            logger.info(f"✓ ${dollars:,.2f} → {shares} shares @ ${price:.2f} = ${actual_cost:,.2f}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Dollar conversion failed: {e}")
        return False

def test_trade_logging():
    """Test 6: Trade Logging"""
    logger.info("\n" + "="*60)
    logger.info("TEST 6: Trade Logging")
    logger.info("="*60)
    
    try:
        trader = AlpacaTrader(symbol="TSLA", mode="paper")
        
        # Simulate trades
        trader.log_trade(
            timestamp=datetime.now(),
            action="BUY",
            shares=10,
            price=417.75,
            pnl=None
        )
        logger.info("✓ Logged BUY trade")
        
        trader.log_trade(
            timestamp=datetime.now(),
            action="SELL",
            shares=10,
            price=420.50,
            pnl=27.50
        )
        logger.info("✓ Logged SELL trade")
        
        # Export
        trader.export_trade_log("test_trade_log.json")
        logger.info("✓ Exported trade log to test_trade_log.json")
        
        assert len(trader.trade_log) == 2, "Should have 2 trades logged"
        return True
    except Exception as e:
        logger.error(f"✗ Trade logging failed: {e}")
        return False

def test_bot_initialization():
    """Test 7: Bot Initialization"""
    logger.info("\n" + "="*60)
    logger.info("TEST 7: Bot Initialization")
    logger.info("="*60)
    
    try:
        bot = AlpacaLiveBot()
        logger.info("✓ Bot initialized successfully")
        logger.info(f"  Trader symbol: {bot.trader.symbol}")
        logger.info(f"  Trader mode: {bot.trader.mode}")
        logger.info(f"  Entry time: {bot.entry_time}")
        logger.info(f"  Entry price: {bot.entry_price}")
        return True
    except Exception as e:
        logger.error(f"✗ Bot initialization failed: {e}")
        return False

def test_indicators():
    """Test 8: Indicator Calculations"""
    logger.info("\n" + "="*60)
    logger.info("TEST 8: Indicator Calculations")
    logger.info("="*60)
    
    try:
        bot = AlpacaLiveBot()
        bars = bot.trader.get_historical_bars(timeframe="5Min", days_back=5)
        
        # Calculate indicators
        ema_750 = bars['close'].ewm(span=750).mean().iloc[-1]
        slope = bot.calculate_slope(bars['close'], periods=20)
        atr = bot.calculate_atr(bars, periods=14)
        
        logger.info(f"✓ Indicators calculated:")
        logger.info(f"  EMA750: ${ema_750:.2f}")
        logger.info(f"  Slope (20): {slope:.6f}")
        logger.info(f"  ATR (14): ${atr:.2f}")
        
        assert ema_750 > 0, "EMA should be positive"
        assert atr > 0, "ATR should be positive"
        return True
    except Exception as e:
        logger.error(f"✗ Indicator calculation failed: {e}")
        return False

def test_position_sizing():
    """Test 9: Position Sizing"""
    logger.info("\n" + "="*60)
    logger.info("TEST 9: Position Sizing")
    logger.info("="*60)
    
    try:
        bot = AlpacaLiveBot()
        account = bot.trader.get_account_info()
        
        # Test with different conditions
        capital = account['equity']
        atr = 8.50  # Example ATR
        slope_strong = 0.015
        slope_weak = 0.001
        drawdown_good = -0.05
        drawdown_bad = -0.25
        equity_high = capital
        
        # Scenario 1: Strong trend, good equity
        size_strong = bot.calculate_position_size(
            capital, atr, slope_strong, drawdown_good, equity_high
        )
        logger.info(f"✓ Strong trend + Good equity: ${size_strong:,.2f}")
        
        # Scenario 2: Weak trend, bad drawdown
        size_weak = bot.calculate_position_size(
            capital, atr, slope_weak, drawdown_bad, equity_high
        )
        logger.info(f"✓ Weak trend + Bad drawdown: ${size_weak:,.2f}")
        
        assert size_strong > size_weak, "Strong trend should size bigger"
        logger.info(f"✓ Position sizing scales correctly (strong {size_strong/size_weak:.1f}x larger)")
        
        return True
    except Exception as e:
        logger.error(f"✗ Position sizing failed: {e}")
        return False

def run_all_tests():
    """Run complete test suite"""
    logger.info("\n" + "#"*60)
    logger.info("# ALPACA INTEGRATION TEST SUITE")
    logger.info("#"*60)
    
    tests = [
        ("Connection & Auth", test_connection),
        ("Account Info", test_account_info),
        ("Price Data", test_price_data),
        ("Position Tracking", test_position_tracking),
        ("Dollar Conversion", test_dollar_conversion),
        ("Trade Logging", test_trade_logging),
        ("Bot Init", test_bot_initialization),
        ("Indicators", test_indicators),
        ("Position Sizing", test_position_sizing),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"CRITICAL ERROR in {test_name}: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "#"*60)
    logger.info("# TEST RESULTS SUMMARY")
    logger.info("#"*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✓ PASS" if passed_test else "✗ FAIL"
        logger.info(f"{status}: {test_name}")
    
    logger.info("\n" + "="*60)
    logger.info(f"OVERALL: {passed}/{total} tests passed ({100*passed/total:.1f}%)")
    logger.info("="*60)
    
    if passed == total:
        logger.info("✓ ALL TESTS PASSED - READY FOR DEPLOYMENT")
        return True
    else:
        logger.info(f"✗ {total - passed} TESTS FAILED - FIX BEFORE DEPLOYMENT")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)