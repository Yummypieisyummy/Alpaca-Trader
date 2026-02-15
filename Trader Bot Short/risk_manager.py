"""
RISK MANAGER
============
Position sizing, drawdown protection, and performance tracking for day trading bot.

Usage:
    from risk_manager import RiskManager
    rm = RiskManager(initial_capital=100000, max_daily_loss=0.02)
    rm.update_equity(current_equity)
    position_size = rm.get_position_size()
"""

import logging
from datetime import datetime, date
from typing import Dict, List

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Institutional risk management for day trading
    
    Features:
    - Equity-based position sizing
    - Max drawdown circuit breaker
    - Daily loss limit
    - Max consecutive losses
    - Win rate and Sharpe ratio tracking
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        max_daily_loss_pct: float = 0.02,  # 2% max loss per day
        max_drawdown_pct: float = 0.05,  # 5% max drawdown
        max_consecutive_losses: int = 3,
        base_risk_pct: float = 0.01,  # 1% risk per trade
        volatility_scalar: float = 1.0,  # Adjust based on ATR
    ):
        self.initial_capital = initial_capital
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.max_consecutive_losses = max_consecutive_losses
        self.base_risk_pct = base_risk_pct
        self.volatility_scalar = volatility_scalar
        
        # State
        self.current_equity = initial_capital
        self.peak_equity = initial_capital
        self.daily_start_equity = initial_capital
        self.current_date = date.today()
        
        # Performance tracking
        self.trades: List[Dict] = []
        self.consecutive_losses = 0
        self.daily_trades = 0
        
        logger.info(f"RiskManager initialized: ${initial_capital:,.0f} | Max daily loss: {max_daily_loss_pct:.1%} | Max DD: {max_drawdown_pct:.1%}")
    
    def update_equity(self, current_equity: float):
        """Update current equity and check limits"""
        self.current_equity = current_equity
        self.peak_equity = max(self.peak_equity, current_equity)
        
        # Check if new day
        if date.today() != self.current_date:
            self._reset_daily_state()
        
        self.current_date = date.today()
    
    def _reset_daily_state(self):
        """Reset daily tracking at market close"""
        logger.info(f"Daily reset: Equity ${self.current_equity:,.0f}")
        self.daily_start_equity = self.current_equity
        self.daily_trades = 0
    
    def is_trading_allowed(self) -> bool:
        """Check if bot should continue trading (circuit breaker)"""
        # Check daily loss limit
        daily_loss = (self.daily_start_equity - self.current_equity) / self.daily_start_equity
        if daily_loss > self.max_daily_loss_pct:
            logger.warning(f"CIRCUIT BREAKER: Daily loss {daily_loss:.2%} exceeds limit {self.max_daily_loss_pct:.2%}")
            return False
        
        # Check max drawdown
        drawdown = (self.peak_equity - self.current_equity) / self.peak_equity
        if drawdown > self.max_drawdown_pct:
            logger.warning(f"CIRCUIT BREAKER: Drawdown {drawdown:.2%} exceeds limit {self.max_drawdown_pct:.2%}")
            return False
        
        # Check max consecutive losses
        if self.consecutive_losses >= self.max_consecutive_losses:
            logger.warning(f"CIRCUIT BREAKER: {self.consecutive_losses} consecutive losses exceed limit {self.max_consecutive_losses}")
            return False
        
        return True
    
    def get_position_size(self, atr: float = 1.0, price: float = 100.0) -> float:
        """
        Calculate position size in dollars
        
        Args:
            atr: Average True Range (volatility measure)
            price: Current stock price
        
        Returns:
            Position size in dollars (capped at equity)
        """
        # Base risk per trade
        risk_per_trade = self.current_equity * self.base_risk_pct
        
        # Adjust for volatility (higher ATR = smaller position)
        volatility_adjustment = self.volatility_scalar / max(atr, 0.1)
        
        position_size = risk_per_trade * volatility_adjustment
        
        # Cap at 50% of equity (avoid over-leverage)
        max_position = self.current_equity * 0.50
        position_size = min(position_size, max_position)
        
        logger.debug(f"Position size: ${position_size:.2f} (ATR {atr:.2f}, vol scalar {self.volatility_scalar:.2f})")
        return max(position_size, 0.0)
    
    def log_trade(self, symbol: str, side: str, entry_price: float, exit_price: float, shares: float):
        """Log a completed trade"""
        pnl = (exit_price - entry_price) * shares if side == "LONG" else (entry_price - exit_price) * shares
        pnl_pct = (exit_price - entry_price) / entry_price if side == "LONG" else (entry_price - exit_price) / entry_price
        
        trade = {
            "timestamp": datetime.now(),
            "symbol": symbol,
            "side": side,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "shares": shares,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
        }
        
        self.trades.append(trade)
        self.daily_trades += 1
        
        # Track consecutive losses
        if pnl_pct < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        
        logger.info(
            f"Trade logged: {symbol} {side} | Entry ${entry_price:.2f} Exit ${exit_price:.2f} | "
            f"PnL ${pnl:.2f} ({pnl_pct:.2%}) | Consecutive losses: {self.consecutive_losses}"
        )
    
    def get_stats(self) -> Dict:
        """Calculate current performance metrics"""
        if not self.trades:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "sharpe_ratio": 0.0,
                "daily_pnl": 0.0,
                "daily_pnl_pct": 0.0,
                "daily_trades": self.daily_trades,
                "consecutive_losses": self.consecutive_losses,
            }
        
        wins = [t for t in self.trades if t["pnl_pct"] > 0]
        losses = [t for t in self.trades if t["pnl_pct"] <= 0]
        
        total_wins = sum(t["pnl_pct"] for t in wins)
        total_losses = sum(abs(t["pnl_pct"]) for t in losses)
        
        win_rate = len(wins) / len(self.trades) if self.trades else 0.0
        profit_factor = total_wins / total_losses if total_losses > 0 else (5.0 if total_wins > 0 else 0.0)
        
        avg_win = (total_wins / len(wins)) if wins else 0.0
        avg_loss = (total_losses / len(losses)) if losses else 0.0
        
        # Simple Sharpe (daily returns)
        daily_returns = [t["pnl_pct"] for t in self.trades]
        import numpy as np
        sharpe = np.mean(daily_returns) / (np.std(daily_returns) + 1e-6) if daily_returns else 0.0
        
        daily_pnl = sum(t["pnl"] for t in self.trades if t["timestamp"].date() == self.current_date)
        daily_pnl_pct = (self.current_equity - self.daily_start_equity) / self.daily_start_equity
        
        return {
            "total_trades": len(self.trades),
            "win_rate": float(win_rate),
            "profit_factor": float(profit_factor),
            "avg_win": float(avg_win),
            "avg_loss": float(avg_loss),
            "sharpe_ratio": float(sharpe),
            "daily_pnl": float(daily_pnl),
            "daily_pnl_pct": float(daily_pnl_pct),
            "daily_trades": self.daily_trades,
            "consecutive_losses": self.consecutive_losses,
        }
    
    def export_trades(self, filename: str = "trades.csv"):
        """Export all trades to CSV"""
        import csv
        
        if not self.trades:
            logger.warning("No trades to export")
            return
        
        with open(filename, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.trades[0].keys())
            writer.writeheader()
            for trade in self.trades:
                writer.writerow(trade)
        
        logger.info(f"Exported {len(self.trades)} trades to {filename}")
