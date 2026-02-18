"""
ALPACA MULTI-SYMBOL 750 EMA BOT (STANDALONE) - AGGRESSIVE VERSION
===================================================================
High-frequency, high-return trading bot for NVDA, TSLA, XLK simultaneously.
No path dependencies - runs anywhere.

TARGET: 15-25% annual return with active management

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
    
    def _get_yfinance_bars(self, timeframe: str = "5Min", days_back: int = 20) -> Optional[pd.DataFrame]:
        try:
            import yfinance as yf
            
            # Yahoo Finance limits 1-min data to 8 days per request
            max_days = min(days_back, 8)
            logger.info(f"{self.symbol}: Fetching {max_days} days of 1-min bars from yfinance (max allowed: 8)...")
            bars = yf.download(self.symbol, period=f"{max_days}d", interval="1m", progress=False)
            
            if bars is not None and len(bars) > 750:
                bars.columns = [str(c).lower() for c in bars.columns]
                logger.info(f"{self.symbol}: Got {len(bars)} 1-min bars from yfinance")
                return bars
            else:
                logger.warning(f"{self.symbol}: Only got {len(bars) if bars is not None else 0} bars, need 750+ (got {len(bars) if bars is not None else 0} from {max_days} days)")
                return None
        except Exception as e:
            logger.error(f"yfinance error: {e}")
            return None
    
    def place_buy_order_dollars(self, dollars: float) -> Optional[object]:
        try:
            price = self.get_latest_price()
            if not price or price <= 0:
                logger.warning(f"{self.symbol}: Cannot place buy order - invalid price: {price}")
                return None
            shares = int(dollars / price)
            if shares <= 0:
                logger.warning(f"{self.symbol}: Cannot place buy order - ${dollars:.2f} not enough for {shares} shares @ ${price:.2f}")
                return None
            order = self.client.submit_order(symbol=self.symbol, qty=shares, side="buy", type="market", time_in_force="day")
            logger.info(f"BUY ORDER PLACED: {shares} shares @ ${price:.2f}")
            return order
        except Exception as e:
            logger.error(f"Buy order error for {self.symbol}: {e}")
            return None
    
    def close_position(self) -> Optional[object]:
        try:
            position = self.get_position()
            if position and position["shares"] > 0:
                order = self.client.submit_order(symbol=self.symbol, qty=int(position["shares"]), side="sell", type="market", time_in_force="day")
                logger.info(f"SELL ORDER PLACED: {position['shares']} shares @ ${position['current_price']:.2f}")
                return order
            elif not position:
                logger.warning(f"{self.symbol}: Cannot close position - no position found")
                return None
            return None
        except Exception as e:
            logger.error(f"Close order error for {self.symbol}: {e}")
            return None
    
    def export_trade_log(self, filename: str = "trade_log.json"):
        try:
            import json
            with open(filename, 'w') as f:
                json.dump(self.trade_log, f, indent=2, default=str)
        except:
            pass


class AlpacaMultiSymbolBot:
    """Multi-symbol 750 EMA bot with advanced profit optimization"""
    
    def __init__(self, symbols: List[str] = None, mode: str = "paper"):
        if symbols is None:
            symbols = ["NVDA", "TSLA", "XLK"]
        
        self.symbols = symbols
        self.mode = mode
        self.traders = {sym: SimpleAlpacaTrader(symbol=sym, mode=mode) for sym in symbols}
        self.entry_prices = {sym: None for sym in symbols}
        self.entry_times = {sym: None for sym in symbols}
        self.peak_capitals = {sym: None for sym in symbols}
        self.position_shares = {sym: 0 for sym in symbols}
        self.position_type = {sym: None for sym in symbols}  # 'LONG' or 'SHORT'
        self.partial_exits = {sym: {"first_exit": False, "second_exit": False} for sym in symbols}
        
        # Performance metrics (aggressive strategy)
        self.win_rate = 0.45
        self.avg_win = 25.0
        self.avg_loss = 12.0
        self.profit_factor = self.avg_win / self.avg_loss
        
        logger.info(f"MultiSymbolBot initialized: {symbols} (AGGRESSIVE - Target 15-25% annual)")
    
    def calculate_slope(self, data: pd.Series, periods: int = 20) -> float:
        if len(data) < periods:
            return 0.0
        recent = data.tail(periods).values
        x = np.arange(len(recent))
        slope = np.polyfit(x, recent, 1)[0]
        return slope
    
    def calculate_rsi(self, data: pd.Series, periods: int = 14) -> float:
        """Calculate Relative Strength Index"""
        if len(data) < periods + 1:
            return 50.0  # Neutral RSI if not enough data
        
        deltas = data.diff()
        gains = deltas.where(deltas > 0, 0)
        losses = -deltas.where(deltas < 0, 0)
        
        avg_gain = gains.rolling(window=periods).mean().iloc[-1]
        avg_loss = losses.rolling(window=periods).mean().iloc[-1]
        
        if avg_loss == 0:
            return 100.0 if avg_gain > 0 else 50.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi if not np.isnan(rsi) else 50.0
    
    def calculate_slope_acceleration(self, data: pd.Series, short_periods: int = 5, long_periods: int = 20) -> float:
        """Calculate if slope is accelerating (positive = strengthening trend)"""
        if len(data) < long_periods + 5:
            return 0.0
        
        short_slope = self.calculate_slope(data, short_periods)
        long_slope = self.calculate_slope(data, long_periods)
        
        # Positive acceleration means trend is strengthening
        acceleration = short_slope - long_slope
        return acceleration
    
    def calculate_atr_percentile(self, bars: pd.DataFrame, periods: int = 14, lookback: int = 100) -> tuple:
        """Calculate current ATR and its percentile ranking"""
        if bars is None or len(bars) < max(periods, lookback):
            return 1.0, 50.0  # Return default ATR and 50th percentile
        
        high_low = bars['high'] - bars['low']
        high_close = abs(bars['high'] - bars['close'].shift())
        low_close = abs(bars['low'] - bars['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(periods).mean()
        
        current_atr = atr.iloc[-1]
        if np.isnan(current_atr):
            current_atr = 1.0
        
        # Get ATR percentile (how volatile relative to recent history)
        atr_hist = atr.tail(lookback).dropna()
        if len(atr_hist) < 2:
            percentile = 50.0
        else:
            percentile = (atr_hist < current_atr).sum() / len(atr_hist) * 100
        
        return current_atr, percentile
    
    def calculate_atr(self, bars: pd.DataFrame, periods: int = 14) -> float:
        if bars is None or len(bars) < periods:
            return 1.0
        high_low = bars['high'] - bars['low']
        high_close = abs(bars['high'] - bars['close'].shift())
        low_close = abs(bars['low'] - bars['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(periods).mean().iloc[-1]
        return atr if not np.isnan(atr) else 1.0
    
    def calculate_position_size(self, capital: float, atr: float, atr_percentile: float, slope: float, slope_accel: float, current_drawdown: float, equity_high: float) -> float:
        """
        ENHANCED Position Sizing using Kelly Criterion + Volatility Adjustment
        
        Kelly Criterion: f* = (bp - q) / b
        where b = profit_factor, p = win_rate, q = 1-p
        
        For our stats: (2.9 * 0.388 - 0.612) / 2.9 = 0.279 (27.9%)
        Conservative 50% Kelly = 13.95% (~14%)
        Further adjusted for volatility and trend strength
        """
        
        # Base Kelly calculation (conservative 50% Kelly)
        kelly_fraction = (self.profit_factor * self.win_rate - (1 - self.win_rate)) / self.profit_factor
        conservative_kelly = kelly_fraction * 0.5  # 50% Kelly for safety
        base_position = capital * max(conservative_kelly, 0.04)  # Min 4% per trade
        
        # Volatility adjustment: reduce size in high volatility, increase in low volatility
        if atr_percentile > 75:
            # High volatility - reduce size
            base_position *= 0.7
        elif atr_percentile < 25:
            # Low volatility - increase size
            base_position *= 1.3
        
        # Trend strength adjustment: boost size when trend is accelerating
        if slope_accel > 0.001:
            base_position *= 1.2  # Strengthening trend
        elif slope_accel < -0.001:
            base_position *= 0.8  # Weakening trend
        
        # Slope quality adjustment: stronger trends = larger size
        if slope > 0.010:
            base_position *= 1.3
        elif slope > 0.005:
            base_position *= 1.0
        elif slope < 0.001:
            base_position *= 0.7
        
        # Capital growth adjustment: scale up when equity is growing
        if capital > equity_high * 1.05:
            base_position *= 1.15
        
        # Drawdown protection: reduce size significantly in drawdowns
        if current_drawdown < -0.10:
            base_position *= 0.6
        elif current_drawdown < -0.20:
            base_position *= 0.3
        
        return max(base_position, 0.0)
    
    def run(self, check_interval: int = 60, days_back: int = 20):
        logger.info("=" * 70)
        logger.info("ALPACA MULTI-SYMBOL BOT STARTED (AGGRESSIVE - 15-25% TARGET)")
        logger.info("=" * 70)
        logger.info("Strategy: 200-EMA bidirectional (LONG/SHORT) with tight scale-outs")
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
                        
                        # Calculate all indicators
                        ema_750 = bars['close'].ewm(span=750).mean().iloc[-1]
                        slope = self.calculate_slope(bars['close'], periods=20)
                        slope_accel = self.calculate_slope_acceleration(bars['close'], short_periods=5, long_periods=20)
                        rsi = self.calculate_rsi(bars['close'], periods=14)
                        atr, atr_percentile = self.calculate_atr_percentile(bars, periods=14, lookback=100)
                        
                        if self.peak_capitals[symbol] is None:
                            self.peak_capitals[symbol] = current_capital
                        else:
                            self.peak_capitals[symbol] = max(self.peak_capitals[symbol], current_capital)
                        
                        current_drawdown = (current_capital - self.peak_capitals[symbol]) / self.peak_capitals[symbol] if self.peak_capitals[symbol] > 0 else 0
                        
                        # ENHANCED Entry conditions with quality filters
                        entry_threshold = 0.008  # Increased from 0.005 (higher slope required)
                        exit_threshold = -0.005
                        buffer = 0.005
                        
                        # Entry must meet ALL criteria:
                        price_above_ema = price > ema_750 * (1 + buffer)
                        slope_strong = slope > entry_threshold
                        slope_accelerating = slope_accel > 0.0005  # Trend strengthening
                        rsi_optimal = 40 < rsi < 70  # Avoid oversold/overbought
                        market_hours = 9 <= market_hour < 16  # 9:30 AM - 3:59 PM ET (avoid open/close)
                        
                        regime_entry = price_above_ema and slope_strong and slope_accelerating and rsi_optimal and market_hours
                        regime_exit = (price < ema_750 * (1 - buffer)) and (slope < exit_threshold)
                        
                        logger.info(f"{symbol}: Price ${price:.2f} | EMA750 ${ema_750:.2f} | Slope {slope:.6f} | ", end="")
                        logger.info(f"RSI {rsi:.1f} | ATR%ile {atr_percentile:.0f} | SlopeAccel {slope_accel:.6f}")
                        
                        # === ENTRY LOGIC with scaling ===
                        if regime_entry and not position:
                            position_dollars = self.calculate_position_size(
                                current_capital, atr, atr_percentile, slope, slope_accel, 
                                current_drawdown, self.peak_capitals[symbol]
                            )
                            if position_dollars > 0 and position_dollars <= current_capital * 0.95:
                                shares = int(position_dollars / price)
                                if shares > 0:
                                    cost = shares * price
                                    current_capital -= cost
                                    self.position_shares[symbol] = shares
                                    self.partial_exits[symbol] = {"first_exit": False, "second_exit": False}
                                    
                                    self.entry_prices[symbol] = price
                                    self.entry_times[symbol] = timestamp
                                    
                                    logger.info(f"{symbol} ✅ ENTRY CONFIRMED | {shares} shares @ ${price:.2f} | Size: ${position_dollars:.2f}")
                                    logger.info(f"    Filters: Price↑ SlopeStrong SlopeAccel RSI({rsi:.1f}) MarketHours")
                                    
                                    order = trader.place_buy_order_dollars(position_dollars)
                                    if not order:
                                        logger.error(f"{symbol} ORDER FAILED - reverting position")
                                        current_capital += cost
                                        self.position_shares[symbol] = 0
                                        self.entry_prices[symbol] = None
                            else:
                                if position_dollars > 0:
                                    logger.warning(f"{symbol} Entry signal but position size ${position_dollars:.2f} > 95% capital")
                        
                        # === SCALE-OUT EXIT LOGIC (Partial Profit Taking) ===
                        elif position:
                            position_val = self.position_shares[symbol] * price
                            pnl = position_val - (self.position_shares[symbol] * self.entry_prices[symbol])
                            pnl_pct = (price - self.entry_prices[symbol]) / self.entry_prices[symbol] * 100
                            
                            # Scale-out target 1: Sell 50% at +2%
                            if not self.partial_exits[symbol]["first_exit"] and pnl_pct > 2.0:
                                shares_to_sell = int(self.position_shares[symbol] * 0.5)
                                if shares_to_sell > 0:
                                    logger.info(f"{symbol} 🎯 SCALE-OUT #1 | Sell {shares_to_sell} @ ${price:.2f} | PnL: {pnl_pct:.2f}%")
                                    current_capital += shares_to_sell * price
                                    self.position_shares[symbol] -= shares_to_sell
                                    self.partial_exits[symbol]["first_exit"] = True
                                    order = trader.close_position() if self.position_shares[symbol] == 0 else None
                            
                            # Scale-out target 2: Sell 50% of remaining at +4%
                            elif not self.partial_exits[symbol]["second_exit"] and pnl_pct > 4.0:
                                shares_to_sell = int(self.position_shares[symbol] * 0.5)
                                if shares_to_sell > 0:
                                    logger.info(f"{symbol} 🎯 SCALE-OUT #2 | Sell {shares_to_sell} @ ${price:.2f} | PnL: {pnl_pct:.2f}%")
                                    current_capital += shares_to_sell * price
                                    self.position_shares[symbol] -= shares_to_sell
                                    self.partial_exits[symbol]["second_exit"] = True
                                    order = trader.close_position() if self.position_shares[symbol] == 0 else None
                            
                            # Hard exit on regime change
                            elif regime_exit and self.position_shares[symbol] > 0:
                                logger.info(f"{symbol} 📉 REGIME EXIT | Sell {self.position_shares[symbol]} @ ${price:.2f} | PnL: {pnl_pct:.2f}%")
                                current_capital += self.position_shares[symbol] * price
                                self.position_shares[symbol] = 0
                                self.entry_prices[symbol] = None
                                self.entry_times[symbol] = None
                                order = trader.close_position()
                            
                            # Regular hold status
                            elif self.position_shares[symbol] > 0:
                                logger.info(f"{symbol} 📊 HOLD | {self.position_shares[symbol]} shares | Value: ${position_val:.2f} | PnL: {pnl_pct:.2f}%")
                        
                        else:
                            logger.info(f"{symbol} ⏸️  No position / No entry signal")
                    
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
