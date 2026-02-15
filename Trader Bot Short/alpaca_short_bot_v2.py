"""
ALPACA SHORT BOT V2
===================
Institutional-grade day trader using Bollinger Band squeeze + volume.

Features:
- Multi-symbol trading (NVDA, TSLA, XLK)
- Risk-managed position sizing
- Market regime filter (avoid choppy hours)
- Real-time performance tracking
- Circuit breaker protection
- Bracket order simulation

Usage:
    python alpaca_short_bot_v2.py --mode paper --symbols NVDA TSLA
"""

import argparse
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Note: alpaca_trader needs to be imported from parent Trading Bot v1 folder
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "Trading Bot v1"))

from alpaca_trader import AlpacaTrader
from risk_manager import RiskManager

# === LOGGING ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot_short_v2.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MarketRegimeFilter:
    """Filter out choppy/low-volume trading hours"""
    
    @staticmethod
    def is_tradeable_hour(now: datetime) -> bool:
        """Skip first 30min (gap) and last hour (chop)"""
        hour = now.hour
        minute = now.minute
        
        # Skip 9:30-10:00 (market open, high slippage)
        if hour == 9 and minute < 60:
            return False
        
        # Skip 15:30-16:00 (market close, low volume, high spread)
        if hour == 15 and minute >= 30:
            return False
        
        # Skip after hours
        if hour >= 16 or hour < 9:
            return False
        
        return True
    
    @staticmethod
    def is_volume_sufficient(volume_ma: float, current_volume: float, threshold: float = 1.5) -> bool:
        """Check if volume is above average"""
        return current_volume > volume_ma * threshold


class BBSqueezeBot:
    """
    Multi-symbol Bollinger Band squeeze bot with risk management
    """
    
    def __init__(
        self,
        symbols: List[str],
        mode: str = "paper",
        check_interval: int = 60,
    ):
        self.symbols = symbols
        self.mode = mode
        self.check_interval = check_interval
        self.traders = {sym: AlpacaTrader(symbol=sym, mode=mode) for sym in symbols}
        self.risk_manager = RiskManager(
            initial_capital=100000.0,
            max_daily_loss_pct=0.02,  # 2% daily loss limit
            max_drawdown_pct=0.05,  # 5% max drawdown
            max_consecutive_losses=3,
            base_risk_pct=0.01,  # 1% risk per trade
        )
        
        # BB Squeeze params (optimized from research)
        self.bb_period = 20
        self.bb_std = 2.0
        self.squeeze_threshold = 0.5
        self.volume_threshold = 1.5
        self.take_profit = 0.0040  # 0.40%
        self.stop_loss = 0.0045  # 0.45%
        self.max_hold_bars = 120
        
        # State per symbol
        self.positions = {sym: None for sym in symbols}
        self.entry_times = {sym: None for sym in symbols}
        
        logger.info(f"BBSqueezeBot initialized: {symbols} in {mode} mode")
    
    def calculate_bollinger_bands(self, series: pd.Series) -> tuple:
        """Returns (upper, mid, lower) bands"""
        sma = series.rolling(self.bb_period).mean()
        std = series.rolling(self.bb_period).std()
        upper = sma + (std * self.bb_std)
        lower = sma - (std * self.bb_std)
        return upper, sma, lower
    
    def calculate_bb_width(self, series: pd.Series) -> pd.Series:
        """Band width as % of price"""
        upper, mid, lower = self.calculate_bollinger_bands(series)
        return (upper - lower) / series
    
    def analyze_symbol(self, symbol: str, bars: pd.DataFrame) -> Dict:
        """Analyze single symbol for entry/exit signals"""
        if bars is None or len(bars) < self.bb_period + 5:
            return {"ready": False}
        
        close = bars["close"]
        volume = bars["volume"]
        
        price = float(close.iloc[-1])
        upper, mid, lower = self.calculate_bollinger_bands(close)
        band_width = self.calculate_bb_width(close)
        vol_ma = volume.rolling(20).mean()
        
        if np.isnan(band_width.iloc[-1]) or np.isnan(vol_ma.iloc[-1]):
            return {"ready": False}
        
        return {
            "ready": True,
            "price": price,
            "upper": float(upper.iloc[-1]),
            "mid": float(mid.iloc[-1]),
            "lower": float(lower.iloc[-1]),
            "band_width": float(band_width.iloc[-1]),
            "volume": float(volume.iloc[-1]),
            "volume_ma": float(vol_ma.iloc[-1]),
            "atr": float(bars["high"].iloc[-1] - bars["low"].iloc[-1]),  # Simple ATR proxy
        }
    
    def get_entry_signal(self, analysis: Dict) -> bool:
        """Check entry conditions: squeeze + volume breakout"""
        if not analysis["ready"]:
            return False
        
        in_squeeze = analysis["band_width"] < (analysis["volume_ma"] * self.squeeze_threshold / 100)
        high_volume = analysis["volume"] > analysis["volume_ma"] * self.volume_threshold
        breakout = analysis["price"] > analysis["upper"]
        
        return in_squeeze and high_volume and breakout
    
    def get_exit_signal(self, symbol: str, analysis: Dict) -> Optional[str]:
        """Check exit conditions: TP, SL, or reversion"""
        if symbol not in self.positions or not self.positions[symbol]:
            return None
        
        entry_price = self.positions[symbol]["entry_price"]
        entry_time = self.entry_times[symbol]
        price = analysis["price"]
        
        # P&L check
        pnl_pct = (price - entry_price) / entry_price
        
        # Hold time check
        if entry_time:
            hold_seconds = (datetime.now() - entry_time).total_seconds()
            hold_bars = int(hold_seconds / 60)  # Assume 1min bars = 1 sec per bar
        else:
            hold_bars = 0
        
        # Exit signals
        if pnl_pct >= self.take_profit:
            return "TAKE_PROFIT"
        elif pnl_pct <= -self.stop_loss:
            return "STOP_LOSS"
        elif hold_bars >= self.max_hold_bars:
            return "MAX_HOLD"
        elif price <= analysis["mid"]:
            return "MEAN_REVERSION"
        
        return None
    
    def run(self, days_back: int = 5):
        """Main trading loop"""
        logger.info("=" * 70)
        logger.info("ALPACA SHORT BOT V2 STARTED")
        logger.info(f"Symbols: {self.symbols}")
        logger.info(f"Mode: {self.mode}")
        logger.info("=" * 70)
        
        iteration = 0
        
        try:
            while True:
                iteration += 1
                now = datetime.now()
                
                # Check circuit breaker
                account = self.traders[self.symbols[0]].get_account_info()
                if account:
                    self.risk_manager.update_equity(float(account["equity"]))
                    
                    if not self.risk_manager.is_trading_allowed():
                        logger.warning("Circuit breaker activated. Stopping bot.")
                        break
                
                # Log iteration header
                logger.info(f"\n[Iter {iteration}] {now.strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Check market hours
                if not MarketRegimeFilter.is_tradeable_hour(now):
                    logger.info(f"Not in tradeable hours. Sleeping {self.check_interval}s")
                    time.sleep(self.check_interval)
                    continue
                
                # Process each symbol
                for symbol in self.symbols:
                    try:
                        bars = self.traders[symbol].get_historical_bars(timeframe="1Min", days_back=days_back)
                        analysis = self.analyze_symbol(symbol, bars)
                        
                        if not analysis["ready"]:
                            logger.debug(f"{symbol}: Not enough data")
                            continue
                        
                        logger.info(
                            f"{symbol} | Price: ${analysis['price']:.2f} | "
                            f"BB: ${analysis['lower']:.2f}-${analysis['upper']:.2f} | "
                            f"Vol: {analysis['volume']:.0f} (MA {analysis['volume_ma']:.0f})"
                        )
                        
                        # === ENTRY ===
                        if not self.positions[symbol] and self.get_entry_signal(analysis):
                            position_dollars = self.risk_manager.get_position_size(
                                atr=analysis["atr"],
                                price=analysis["price"]
                            )
                            
                            if position_dollars > 0:
                                logger.info(f"{symbol} ENTRY: Size ${position_dollars:.2f}")
                                order = self.traders[symbol].place_buy_order_dollars(position_dollars)
                                
                                if order:
                                    self.positions[symbol] = {
                                        "entry_price": analysis["price"],
                                        "position_dollars": position_dollars,
                                    }
                                    self.entry_times[symbol] = now
                        
                        # === EXIT ===
                        elif self.positions[symbol]:
                            exit_reason = self.get_exit_signal(symbol, analysis)
                            
                            if exit_reason:
                                logger.info(f"{symbol} EXIT ({exit_reason})")
                                order = self.traders[symbol].close_position()
                                
                                if order:
                                    entry_price = self.positions[symbol]["entry_price"]
                                    pnl_pct = (analysis["price"] - entry_price) / entry_price
                                    
                                    self.risk_manager.log_trade(
                                        symbol=symbol,
                                        side="LONG",
                                        entry_price=entry_price,
                                        exit_price=analysis["price"],
                                        shares=self.positions[symbol]["position_dollars"] / entry_price,
                                    )
                                    
                                    self.positions[symbol] = None
                                    self.entry_times[symbol] = None
                        
                    except Exception as e:
                        logger.error(f"Error processing {symbol}: {e}", exc_info=True)
                
                # Log performance stats
                stats = self.risk_manager.get_stats()
                logger.info(
                    f"[STATS] Trades: {stats['total_trades']} | "
                    f"Win rate: {stats['win_rate']:.1%} | PF: {stats['profit_factor']:.2f} | "
                    f"Daily: {stats['daily_pnl']:.2f} ({stats['daily_pnl_pct']:.2%}) | "
                    f"Consecutive losses: {stats['consecutive_losses']}"
                )
                
                time.sleep(self.check_interval)
        
        except KeyboardInterrupt:
            logger.info("\n[STOP] Bot stopped by user")
            self._shutdown()
        
        except Exception as e:
            logger.error(f"[FATAL] {e}", exc_info=True)
            self._shutdown()
    
    def _shutdown(self):
        """Graceful shutdown"""
        logger.info("Closing all positions...")
        for symbol in self.symbols:
            try:
                if self.positions[symbol]:
                    self.traders[symbol].close_position()
            except:
                pass
        
        # Export results
        self.risk_manager.export_trades("trades_short_v2.csv")
        
        stats = self.risk_manager.get_stats()
        summary = [
            f"\n{'='*70}",
            f"FINAL STATS",
            f"{'='*70}",
            f"Total trades: {stats['total_trades']}",
            f"Win rate: {stats['win_rate']:.1%}",
            f"Profit factor: {stats['profit_factor']:.2f}",
            f"Daily P&L: ${stats['daily_pnl']:.2f} ({stats['daily_pnl_pct']:.2%})",
            f"Sharpe ratio: {stats['sharpe_ratio']:.2f}",
            f"Final equity: ${self.risk_manager.current_equity:,.2f}",
        ]
        
        for line in summary:
            logger.info(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["paper", "live"], default="paper")
    parser.add_argument("--symbols", nargs="+", default=["NVDA"])
    parser.add_argument("--interval", type=int, default=60, help="Check interval in seconds")
    args = parser.parse_args()
    
    bot = BBSqueezeBot(symbols=args.symbols, mode=args.mode, check_interval=args.interval)
    bot.run()


if __name__ == "__main__":
    main()
