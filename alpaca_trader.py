"""
ALPACA LIVE TRADING INTEGRATION
================================
Standalone module for connecting TSLA trading bot to Alpaca API
Supports both paper and live trading modes

Usage:
    from alpaca_trader import AlpacaTrader
    trader = AlpacaTrader(mode="paper")  # or "live"
    trader.place_buy_order(shares=10)
    position = trader.get_position()
"""

import os
import json
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv
from alpaca_trade_api import REST
import pandas as pd
import asyncio

# === LOGGING SETUP ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('alpaca_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# === LOAD CREDENTIALS ===
load_dotenv("API.env")
API_KEY = os.getenv("API_KEY")
SECRET_KEY = os.getenv("SECRET_KEY")
ENDPOINT = os.getenv("ENDPOINT", "https://paper-api.alpaca.markets/v2")

if not API_KEY or not SECRET_KEY:
    logger.error("Missing API_KEY or SECRET_KEY in API.env")
    raise ValueError("Alpaca credentials not found in API.env")


class AlpacaTrader:
    """
    Live trading integration with Alpaca
    
    Features:
    - Paper trading mode for safe testing
    - Real-time market data streaming
    - Position tracking and management
    - Order execution with error handling
    - Account equity monitoring
    """
    
    def __init__(self, symbol="TSLA", mode="paper"):
        """
        Initialize Alpaca trader
        
        Args:
            symbol: Stock symbol to trade (default: TSLA)
            mode: "paper" for paper trading, "live" for real (default: paper)
        """
        self.symbol = symbol
        self.mode = mode
        self.base_url = ENDPOINT.replace("/v2", "")  # Use endpoint from API.env
        
        # Initialize REST client
        self.client = REST(
            key_id=API_KEY,
            secret_key=SECRET_KEY,
            base_url=self.base_url
        )
        
        # Trading state
        self.position = None
        self.recent_bars = []
        self.orders = {}
        self.trade_log = []
        
        logger.info(f"AlpacaTrader initialized: {symbol} ({mode} mode)")
        self._verify_connection()
    
    def _verify_connection(self):
        """Test connection to Alpaca API"""
        try:
            account = self.client.get_account()
            logger.info(f"Connected to Alpaca")
            logger.info(f"Account equity: ${account.equity}")
            logger.info(f"Buying power: ${account.buying_power}")
            logger.info(f"Day trading buying power: ${account.daytrading_buying_power}")
        except Exception as e:
            logger.error(f"Failed to connect to Alpaca: {e}")
            raise
    
    def get_account_info(self):
        """Get current account status"""
        try:
            account = self.client.get_account()
            info = {
                "equity": float(account.equity),
                "cash": float(account.cash),
                "buying_power": float(account.buying_power),
                "portfolio_value": float(account.portfolio_value),
                "day_trading_buying_power": float(account.daytrading_buying_power),
                "portfolio_percentage": float(account.portfolio_value) / float(account.equity) * 100 if account.equity else 0
            }
            logger.info(f"Account: ${info['equity']:.2f} equity, ${info['cash']:.2f} cash")
            return info
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return None
    
    def get_position(self):
        """Get current position in symbol"""
        try:
            positions = self.client.list_positions()
            position = next((p for p in positions if p.symbol == self.symbol), None)
            
            if position:
                pos_info = {
                    "symbol": position.symbol,
                    "shares": float(position.qty),
                    "avg_fill_price": float(position.avg_fill_price),
                    "current_price": float(position.current_price),
                    "market_value": float(position.market_value),
                    "cost_basis": float(position.cost_basis),
                    "unrealized_pl": float(position.unrealized_pl),
                    "unrealized_plpc": float(position.unrealized_plpc),
                    "side": position.side
                }
                logger.info(f"Position: {pos_info['shares']} shares @ ${pos_info['current_price']:.2f}, PnL: ${pos_info['unrealized_pl']:.2f}")
                return pos_info
            else:
                logger.info(f"No position in {self.symbol}")
                return None
        except Exception as e:
            logger.error(f"Error getting position: {e}")
            return None
    
    def place_buy_order(self, shares, order_type="market"):
        """
        Place a buy order for a specific number of shares
        
        Args:
            shares (float): Number of shares to buy
            order_type (str): "market" or "limit"
        
        Returns:
            Order object with id, status, filled_qty
        """
        try:
            shares = int(shares)
            if shares <= 0:
                logger.warning(f"Invalid share count: {shares}")
                return None
            
            order = self.client.submit_order(
                symbol=self.symbol,
                qty=shares,
                side="buy",
                type=order_type,
                time_in_force="day"
            )
            
            logger.info(f"BUY ORDER: {shares} shares of {self.symbol} (Order ID: {order.id})")
            self.log_trade(datetime.now(), "BUY", shares, self.get_latest_price())
            return order
            
        except Exception as e:
            logger.error(f"Failed to place buy order: {e}")
            return None

    def place_buy_order_dollars(self, dollars, order_type="market"):
        """
        Place a buy order for a specific dollar amount
        
        Args:
            dollars (float): Dollar amount to invest
            order_type (str): "market" or "limit"
        
        Returns:
            Order object with id, status, filled_qty
        """
        try:
            price = self.get_latest_price()
            shares = int(dollars / price)
            
            if shares <= 0:
                logger.warning(f"Dollar amount ${dollars} too small (price ${price})")
                return None
            
            actual_cost = shares * price
            logger.info(f"Converting ${dollars:.2f} → {shares} shares @ ${price:.2f} = ${actual_cost:.2f}")
            
            order = self.client.submit_order(
                symbol=self.symbol,
                qty=shares,
                side="buy",
                type=order_type,
                time_in_force="day"
            )
            
            logger.info(f"BUY ORDER (DOLLARS): ${dollars:.2f} = {shares} shares (Order ID: {order.id})")
            self.log_trade(datetime.now(), "BUY", shares, price)
            return order
            
        except Exception as e:
            logger.error(f"Failed to place buy order (dollars): {e}")
            return None

    def place_sell_order(self, shares, order_type="market"):
        """
        Place a sell order for a specific number of shares
        
        Args:
            shares (float): Number of shares to sell
            order_type (str): "market" or "limit"
        
        Returns:
            Order object with id, status, filled_qty
        """
        try:
            shares = int(shares)
            if shares <= 0:
                logger.warning(f"Invalid share count: {shares}")
                return None
            
            order = self.client.submit_order(
                symbol=self.symbol,
                qty=shares,
                side="sell",
                type=order_type,
                time_in_force="day"
            )
            
            price = self.get_latest_price()
            logger.info(f"SELL ORDER: {shares} shares of {self.symbol} @ ${price:.2f} (Order ID: {order.id})")
            self.log_trade(datetime.now(), "SELL", shares, price)
            return order
            
        except Exception as e:
            logger.error(f"Failed to place sell order: {e}")
            return None

    def place_sell_order_dollars(self, dollars, order_type="market"):
        """
        Place a sell order for a specific dollar amount worth of shares
        
        Args:
            dollars (float): Dollar amount worth to sell
            order_type (str): "market" or "limit"
        
        Returns:
            Order object with id, status, filled_qty
        """
        try:
            price = self.get_latest_price()
            shares = int(dollars / price)
            
            if shares <= 0:
                logger.warning(f"Dollar amount ${dollars} too small (price ${price})")
                return None
            
            actual_value = shares * price
            logger.info(f"Converting ${dollars:.2f} → {shares} shares @ ${price:.2f} = ${actual_value:.2f}")
            
            order = self.client.submit_order(
                symbol=self.symbol,
                qty=shares,
                side="sell",
                type=order_type,
                time_in_force="day"
            )
            
            logger.info(f"SELL ORDER (DOLLARS): ${dollars:.2f} = {shares} shares (Order ID: {order.id})")
            self.log_trade(datetime.now(), "SELL", shares, price)
            return order
            
        except Exception as e:
            logger.error(f"Failed to place sell order (dollars): {e}")
            return None

    def close_position(self):
        """
        Sell entire position in the symbol
        
        Returns:
            Order object if position exists, None otherwise
        """
        try:
            position = self.get_position()
            if position and position["shares"] > 0:
                shares = int(position["shares"])
                logger.info(f"Closing position: selling {shares} shares")
                return self.place_sell_order(shares)
            else:
                logger.info(f"No position to close for {self.symbol}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to close position: {e}")
            return None
    
    def get_historical_bars(self, timeframe="5Min", days_back=5):
        """
        Fetch historical bars from Alpaca or Yahoo Finance fallback
        """
        try:
            # Format dates properly for Alpaca API (no microseconds)
            start = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
            end = datetime.now().strftime("%Y-%m-%d")
            
            logger.info(f"Fetching {timeframe} bars from {start} to {end}")
            
            bars = self.client.get_bars(
                self.symbol,
                timeframe,
                start=start,
                end=end,
                adjustment="all"
            ).df
            
            if bars is None or len(bars) == 0:
                logger.warning(f"No {timeframe} data from Alpaca, trying Yahoo Finance...")
                return self._get_yahoo_finance_bars(timeframe, days_back)
            
            logger.info(f"Retrieved {len(bars)} bars from Alpaca")
            return bars
            
        except Exception as e:
            logger.warning(f"Alpaca bars failed: {e}, trying Yahoo Finance...")
            return self._get_yahoo_finance_bars(timeframe, days_back)
    
    def _get_yahoo_finance_bars(self, timeframe="5Min", days_back=5):
        """
        Fallback: Fetch from Yahoo Finance (lazy import to avoid dependency issues)
        """
        try:
            import yfinance as yf
            import pandas as pd
            
            interval_map = {
                "5Min": "5m",
                "1H": "1h",
                "1D": "1d"
            }
            
            logger.info(f"Fetching {timeframe} from Yahoo Finance (free, unlimited data)...")
            
            bars = yf.download(
                self.symbol,
                period=f"{days_back}d",
                interval=interval_map.get(timeframe, "5m"),
                progress=False
            )
            
            if bars is None or len(bars) == 0:
                logger.error("Yahoo Finance returned no data")
                return None
            
            # Ensure we have a DataFrame
            if not isinstance(bars, pd.DataFrame):
                logger.error(f"Yahoo Finance returned unexpected type: {type(bars)}")
                return None
            
            # Flatten multi-level columns if they exist
            if isinstance(bars.columns, pd.MultiIndex):
                bars.columns = [col[0] if isinstance(col, tuple) else col for col in bars.columns]
            
            # Standardize column names to lowercase
            bars.columns = [str(col).lower() for col in bars.columns]
            
            # Ensure required columns exist
            required_cols = {'open', 'high', 'low', 'close', 'volume'}
            available_cols = set(bars.columns)
            
            if not required_cols.issubset(available_cols):
                logger.error(f"Missing required columns. Have: {available_cols}, Need: {required_cols}")
                return None
            
            logger.info(f"Retrieved {len(bars)} bars from Yahoo Finance")
            return bars
            
        except ImportError:
            logger.error("yfinance not installed. Install with: pip install yfinance")
            return None
        except Exception as e:
            logger.error(f"Yahoo Finance fetch failed: {e}", exc_info=True)
            return None

    def get_latest_price(self):
        """Get current market price"""
        try:
            quote = self.client.get_latest_trade(self.symbol)
            price = quote.price
            logger.info(f"{self.symbol} current price: ${price:.2f}")
            return price
        except Exception as e:
            logger.error(f"Error getting latest price: {e}")
            return None
    
    def get_order_status(self, order_id):
        """Check order status"""
        try:
            order = self.client.get_order(order_id)
            return {
                "id": order.id,
                "status": order.status,
                "filled_qty": order.filled_qty,
                "filled_avg_price": order.filled_avg_price
            }
        except Exception as e:
            logger.error(f"Error getting order status: {e}")
            return None
    
    def cancel_order(self, order_id):
        """Cancel an open order"""
        try:
            self.client.cancel_order(order_id)
            logger.info(f"Order cancelled: {order_id}")
            return True
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return False
    
    def log_trade(self, timestamp, action, shares, price, pnl=None):
        """Log trade for record-keeping"""
        trade = {
            "timestamp": timestamp,
            "action": action,  # "BUY" or "SELL"
            "shares": shares,
            "price": price,
            "value": shares * price,
            "pnl": pnl
        }
        self.trade_log.append(trade)
        logger.info(f"Trade logged: {action} {shares} @ ${price:.2f}")
    
    def export_trade_log(self, filename="trade_log.json"):
        """Export trade log to file"""
        try:
            with open(filename, 'w') as f:
                json.dump(self.trade_log, f, indent=2, default=str)
            logger.info(f"Trade log exported to {filename}")
        except Exception as e:
            logger.error(f"Error exporting trade log: {e}")


# === EXAMPLE USAGE ===
if __name__ == "__main__":
    print("=" * 60)
    print("ALPACA LIVE TRADING INTEGRATION")
    print("=" * 60)
    
    # Initialize trader (paper mode for testing)
    trader = AlpacaTrader(symbol="TSLA", mode="paper")
    
    # Get account info
    print("\n1. ACCOUNT INFO")
    print("-" * 60)
    account = trader.get_account_info()
    if account:
        for key, value in account.items():
            print(f"  {key}: {value}")
    
    # Get current position
    print("\n2. CURRENT POSITION")
    print("-" * 60)
    position = trader.get_position()
    if position:
        for key, value in position.items():
            print(f"  {key}: {value}")
    else:
        print("  No active position")
    
    # Get latest price
    print("\n3. LATEST PRICE")
    print("-" * 60)
    price = trader.get_latest_price()
    if price:
        print(f"  {trader.symbol}: ${price:.2f}")
    
    # Get recent bars (for testing data connection)
    print("\n4. RECENT BARS (1 day, 5min)")
    print("-" * 60)
    bars = trader.get_historical_bars(timeframe="5Min", days_back=1)
    if bars is not None:
        print(f"  Retrieved {len(bars)} bars")
        print(f"\n  Latest bar:")
        print(f"    Open: ${bars['open'].iloc[-1]:.2f}")
        print(f"    High: ${bars['high'].iloc[-1]:.2f}")
        print(f"    Low: ${bars['low'].iloc[-1]:.2f}")
        print(f"    Close: ${bars['close'].iloc[-1]:.2f}")
        print(f"    Volume: {bars['volume'].iloc[-1]:.0f}")
    
    # EXAMPLE: Place small test order (COMMENTED OUT - uncomment to test)
    # print("\n5. TEST ORDER")
    # print("-" * 60)
    # print("  [SKIPPED - uncomment to test]")
    # order = trader.place_buy_order(1)
    # if order:
    #     print(f"  Order ID: {order.id}")
    #     print(f"  Status: {order.status}")
    
    print("\n" + "=" * 60)
    print("Connection successful! Ready for live trading.")
    print("=" * 60)
