"""
ALPACA MULTI-SYMBOL 750 EMA BOT (STANDALONE)
=============================================
Adaptive dual-layer trading bot for NVDA, TSLA, XLK simultaneously.
No path dependencies - runs anywhere.

Usage:
    python alpaca_750ema_multi_standalone.py --mode paper --symbols NVDA TSLA XLK
"""

import logging
import time
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dotenv import load_dotenv
from alpaca_trade_api import REST

# === LOAD CREDENTIALS ===
load_dotenv("API.env")
API_KEY = os.getenv("API_KEY")
SECRET_KEY = os.getenv("SECRET_KEY")
ENDPOINT = os.getenv("ENDPOINT", "https://paper-api.alpaca.markets/v2")

if not API_KEY or not SECRET_KEY:
    raise ValueError("Missing API_KEY or SECRET_KEY in API.env")

# === LOGGING SETUP ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('alpaca_750ema_standalone.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SimpleAlpacaTrader:
    """Minimal Alpaca wrapper for trading"""
    
    def __init__(self, symbol: str, mode: str = "paper"):
        self.symbol = symbol
        self.mode = mode
        self.base_url = ENDPOINT.replace("/v2", "")
        self.client = REST(key_id=API_KEY, secret_key=SECRET_KEY, base_url=self.base_url)
        self.trade_log = []
        logger.info(f"Trader initialized: {symbol} ({mode})")
    
    def get_account_info(self) -> Dict:
        try:
            account = self.client.get_account()
            return {
                "equity": float(account.equity),
                "cash": float(account.cash),
                "buying_power": float(account.buying_power),
                "portfolio_value": float(account.portfolio_value),
            }
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return None
    
    def get_position(self) -> Optional[Dict]:
        try:
            positions = self.client.list_positions()
            position = next((p for p in positions if p.symbol == self.symbol), None)
            if position:
                return {
                    "symbol": position.symbol,
                    "shares": float(position.qty),
                    "avg_fill_price": float(position.avg_fill_price),
                    "current_price": float(position.current_price),
                    "market_value": float(position.market_value),
                    "unrealized_pl": float(position.unrealized_pl),
                    "unrealized_plpc": float(position.unrealized_plpc),
                }
            return None
        except Exception as e:
            logger.error(f"Error getting position: {e}")
            return None
    
    def get_latest_price(self) -> Optional[float]:
        try:
            quote = self.client.get_latest_trade(self.symbol)
            return float(quote.price)
        except Exception as e:
            logger.error(f"Error getting price: {e}")
            return None
    
    def get_historical_bars(self, timeframe: str = "5Min", days_back: int = 5) -> Optional[pd.DataFrame]:
        try:
            start = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
            end = datetime.now().strftime("%Y-%m-%d")
            
            bars = self.client.get_bars(self.symbol, timeframe, start=start, end=end, adjustment="all").df
            if bars is None or len(bars) == 0:
                logger.warning(f"No bars from Alpaca, trying yfinance...")
                return self._get_yfinance_bars(timeframe, days_back)
            
            return bars
        except Exception as e:
            logger.warning(f"Alpaca bars failed: {e}, trying yfinance...")
            return self._get_yfinance_bars(timeframe, days_back)
    
    def _get_yfinance_bars(self, timeframe: str = "5Min", days_back: int = 5) -> Optional[pd.DataFrame]:
        try:
            import yfinance as yf
            interval_map = {"1Min": "1m", "5Min": "5m", "1H": "1h", "1D": "1d"}
            bars = yf.download(self.symbol, period=f"{days_back}d", interval=interval_map.get(timeframe, "5m"), progress=False)
            if bars is None or len(bars) == 0:
                return None
            bars.columns = [str(c).lower() for c in bars.columns]
            return bars
        except:
            return None
    
    def place_buy_order_dollars(self, dollars: float) -> Optional[object]:
        try:
            price = self.get_latest_price()
            if not price or price <= 0:
                return None
            shares = int(dollars / price)
            if shares <= 0:
                return None
            order = self.client.submit_order(symbol=self.symbol, qty=shares, side="buy", type="market", time_in_force="day")
            logger.info(f"BUY ORDER: {shares} shares @ ${price:.2f}")
            return order
        except Exception as e:
            logger.error(f"Buy order error: {e}")
            return None
    
    def close_position(self) -> Optional[object]:
        try:
            position = self.get_position()
            if position and position["shares"] > 0:
                order = self.client.submit_order(symbol=self.symbol, qty=int(position["shares"]), side="sell", type="market", time_in_force="day")
                logger.info(f"SELL ORDER: {position['shares']} shares")
                return order
            return None
        except Exception as e:
            logger.error(f"Close order error: {e}")
            return None
    
    def export_trade_log(self, filename: str = "trade_log.json"):
        try:
            import json
            with open(filename, 'w') as f:
                json.dump(self.trade_log, f, indent=2, default=str)
        except:
            pass


class AlpacaMultiSymbolBot:
    """Multi-symbol 750 EMA bot"""
    
    def __init__(self, symbols: List[str] = None, mode: str = "paper"):
        if symbols is None:
            symbols = ["NVDA", "TSLA", "XLK"]
        
        self.symbols = symbols
        self.mode = mode
        self.traders = {sym: SimpleAlpacaTrader(symbol=sym, mode=mode) for sym in symbols}
        self.entry_prices = {sym: None for sym in symbols}
        self.entry_times = {sym: None for sym in symbols}
        self.peak_capitals = {sym: None for sym in symbols}
        
        logger.info(f"MultiSymbolBot initialized: {symbols}")
    
    def calculate_slope(self, data: pd.Series, periods: int = 20) -> float:
        if len(data) < periods:
            return 0.0
        recent = data.tail(periods).values
        x = np.arange(len(recent))
        slope = np.polyfit(x, recent, 1)[0]
        return slope
    
    def calculate_atr(self, bars: pd.DataFrame, periods: int = 14) -> float:
        if bars is None or len(bars) < periods:
            return 1.0
        high_low = bars['high'] - bars['low']
        high_close = abs(bars['high'] - bars['close'].shift())
        low_close = abs(bars['low'] - bars['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(periods).mean().iloc[-1]
        return atr if not np.isnan(atr) else 1.0
    
    def calculate_position_size(self, capital: float, atr: float, slope: float, current_drawdown: float, equity_high: float) -> float:
        risk_per_trade = 0.025
        base_position = capital * risk_per_trade / (atr * 1.8) if atr > 0 else capital * risk_per_trade
        
        if slope > 0.010:
            base_position *= 1.4
        elif slope < 0.001:
            base_position *= 0.7
        
        if capital > equity_high * 1.05:
            base_position *= 1.2
        
        if current_drawdown < -0.20:
            base_position *= 0.6
        
        return max(base_position, 0.0)
    
    def run(self, check_interval: int = 300, days_back: int = 5):
        logger.info("=" * 70)
        logger.info("ALPACA MULTI-SYMBOL 750 EMA BOT STARTED")
        logger.info("=" * 70)
        
        iteration = 0
        
        try:
            while True:
                iteration += 1
                timestamp = datetime.now()
                
                account = self.traders[self.symbols[0]].get_account_info()
                if not account:
                    logger.error("Cannot get account info")
                    time.sleep(check_interval)
                    continue
                
                current_capital = float(account['equity'])
                logger.info(f"\n[Iteration {iteration}] {timestamp} | Equity: ${current_capital:,.2f}")
                
                for symbol in self.symbols:
                    try:
                        trader = self.traders[symbol]
                        price = trader.get_latest_price()
                        position = trader.get_position()
                        bars = trader.get_historical_bars(timeframe="5Min", days_back=days_back)
                        
                        if not price or bars is None or len(bars) < 750:
                            logger.warning(f"{symbol}: Insufficient data")
                            continue
                        
                        ema_750 = bars['close'].ewm(span=750).mean().iloc[-1]
                        slope = self.calculate_slope(bars['close'], periods=20)
                        atr = self.calculate_atr(bars, periods=14)
                        
                        if self.peak_capitals[symbol] is None:
                            self.peak_capitals[symbol] = current_capital
                        else:
                            self.peak_capitals[symbol] = max(self.peak_capitals[symbol], current_capital)
                        
                        current_drawdown = (current_capital - self.peak_capitals[symbol]) / self.peak_capitals[symbol] if self.peak_capitals[symbol] > 0 else 0
                        
                        entry_threshold = 0.005
                        exit_threshold = -0.005
                        buffer = 0.005
                        
                        regime_entry = (price > ema_750 * (1 + buffer)) and (slope > entry_threshold)
                        regime_exit = (price < ema_750 * (1 - buffer)) and (slope < exit_threshold)
                        
                        logger.info(f"{symbol}: Price ${price:.2f} | EMA750 ${ema_750:.2f} | Slope {slope:.6f}")
                        
                        if regime_entry and not position:
                            position_dollars = self.calculate_position_size(current_capital, atr, slope, current_drawdown, self.peak_capitals[symbol])
                            if position_dollars > 0:
                                logger.info(f"{symbol} ENTRY | Size: ${position_dollars:.2f}")
                                order = trader.place_buy_order_dollars(position_dollars)
                                if order:
                                    self.entry_prices[symbol] = price
                                    self.entry_times[symbol] = timestamp
                        
                        elif regime_exit and position:
                            logger.info(f"{symbol} EXIT | PnL: ${position['unrealized_pl']:.2f}")
                            order = trader.close_position()
                            if order:
                                self.entry_prices[symbol] = None
                                self.entry_times[symbol] = None
                        
                        elif position:
                            logger.info(f"{symbol} HOLD | Shares: {position['shares']} | Value: ${position['market_value']:.2f}")
                    
                    except Exception as e:
                        logger.error(f"{symbol} error: {e}")
                
                logger.info(f"Sleeping {check_interval}s...")
                time.sleep(check_interval)
        
        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
        except Exception as e:
            logger.error(f"Fatal error: {e}", exc_info=True)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["paper", "live"], default="paper")
    parser.add_argument("--symbols", nargs="+", default=["NVDA", "TSLA", "XLK"])
    parser.add_argument("--interval", type=int, default=300)
    args = parser.parse_args()
    
    bot = AlpacaMultiSymbolBot(symbols=args.symbols, mode=args.mode)
    bot.run(check_interval=args.interval)


if __name__ == "__main__":
    main()
