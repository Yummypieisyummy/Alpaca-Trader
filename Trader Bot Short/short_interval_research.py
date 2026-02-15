r"""
SHORT INTERVAL RESEARCH
=======================
Backtest + optimize two strategies:
1. Mean-reversion: VWAP band + RSI oversold
2. Bollinger Band squeeze + volume breakout

Example:
    python short_interval_research.py --csv "..\/data\/nvda_1min_2024-2025_cleaned.csv" --symbol NVDA

Notes:
- Compares strategies on same data.
- Mean-reversion: low-PF, less slippage-sensitive.
- BB squeeze: higher-conviction breakouts, more consistent.
"""

import argparse
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class ParamsMeanReversion:
    rsi_period: int = 14
    rsi_entry: int = 35
    rsi_exit: int = 55
    vwap_window: int = 60
    vwap_band: float = 0.0025
    take_profit: float = 0.0035
    stop_loss: float = 0.0040
    max_hold_bars: int = 90
    position_pct: float = 0.05
    fee_per_trade: float = 0.0


@dataclass
class ParamsBBSqueeze:
    bb_period: int = 20
    bb_std: float = 2.0
    squeeze_vol_threshold: float = 0.5  # BB width < vol_ma * threshold
    volume_ma_period: int = 20
    volume_threshold_mult: float = 1.5  # entry vol > vol_ma * mult
    take_profit: float = 0.0040
    stop_loss: float = 0.0045
    max_hold_bars: int = 120
    position_pct: float = 0.05
    fee_per_trade: float = 0.0


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def _find_datetime_column(df: pd.DataFrame) -> str:
    candidates = ["datetime", "timestamp", "date", "time"]
    for col in candidates:
        if col in df.columns:
            return col
    return ""


def load_bars(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = _standardize_columns(df)

    dt_col = _find_datetime_column(df)
    if dt_col:
        df[dt_col] = pd.to_datetime(df[dt_col], utc=True)
        df = df.sort_values(dt_col)
        df = df.set_index(dt_col)
    else:
        df.index = pd.to_datetime(df.index, errors="ignore")

    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


def calculate_rsi(series: pd.Series, periods: int) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(periods).mean()
    avg_loss = loss.rolling(periods).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_rolling_vwap(bars: pd.DataFrame, window: int) -> pd.Series:
    typical_price = (bars["high"] + bars["low"] + bars["close"]) / 3
    pv = typical_price * bars["volume"]
    vwap = pv.rolling(window).sum() / bars["volume"].rolling(window).sum()
    return vwap


def calculate_bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Returns upper, middle (SMA), lower bands"""
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, sma, lower


def calculate_bb_squeeze(bars: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.Series:
    """Band width as % of close price"""
    upper, mid, lower = calculate_bollinger_bands(bars["close"], period, std_dev)
    band_width = (upper - lower) / bars["close"]
    return band_width


def backtest_mean_reversion(bars: pd.DataFrame, params: ParamsMeanReversion, initial_capital: float = 100000.0) -> Dict:
    close = bars["close"]
    rsi = calculate_rsi(close, params.rsi_period)
    vwap = calculate_rolling_vwap(bars, params.vwap_window)

    position = 0
    entry_price = 0.0
    entry_idx = None
    equity = initial_capital
    equity_curve = []
    trades = []

    for i in range(len(bars)):
        price = float(close.iloc[i])
        if math.isnan(price) or math.isnan(rsi.iloc[i]) or math.isnan(vwap.iloc[i]):
            equity_curve.append(equity)
            continue

        if position == 0:
            entry_signal = price < vwap.iloc[i] * (1 - params.vwap_band) and rsi.iloc[i] < params.rsi_entry
            if entry_signal:
                position = 1
                entry_price = price
                entry_idx = i
                position_value = equity * params.position_pct
                shares = position_value / entry_price
                position_value = shares * entry_price
                equity -= params.fee_per_trade
        else:
            pnl_pct = (price - entry_price) / entry_price
            hold_bars = i - entry_idx if entry_idx is not None else 0
            exit_signal = (
                price >= vwap.iloc[i]
                or rsi.iloc[i] >= params.rsi_exit
                or pnl_pct >= params.take_profit
                or pnl_pct <= -params.stop_loss
                or (params.max_hold_bars and hold_bars >= params.max_hold_bars)
            )

            if exit_signal:
                trade_pnl = (price - entry_price) / entry_price
                equity *= (1 + trade_pnl * params.position_pct)
                equity -= params.fee_per_trade
                trades.append({
                    "entry_idx": entry_idx,
                    "exit_idx": i,
                    "entry_price": entry_price,
                    "exit_price": price,
                    "pnl_pct": trade_pnl,
                    "hold_bars": hold_bars,
                })
                position = 0
                entry_price = 0.0
                entry_idx = None

        equity_curve.append(equity)

    return compute_metrics(equity_curve, trades, initial_capital)


def compute_metrics(equity_curve: List[float], trades: List[Dict], initial_capital: float) -> Dict:
    if not equity_curve:
        return {"trades": 0, "return_pct": 0.0, "max_dd": 0.0, "win_rate": 0.0, "profit_factor": 0.0}

    equity_series = pd.Series(equity_curve)
    returns = equity_series.pct_change().fillna(0.0)
    cumulative = equity_series.iloc[-1]

    peaks = equity_series.cummax()
    drawdowns = (equity_series - peaks) / peaks.replace(0, np.nan)
    max_dd = float(drawdowns.min()) if not drawdowns.empty else 0.0

    wins = [t for t in trades if t["pnl_pct"] > 0]
    losses = [t for t in trades if t["pnl_pct"] <= 0]

    gross_profit = sum(t["pnl_pct"] for t in wins)
    gross_loss = -sum(t["pnl_pct"] for t in losses)
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    win_rate = (len(wins) / len(trades)) if trades else 0.0
    return_pct = (cumulative / initial_capital - 1) * 100

    return {
        "trades": len(trades),
        "return_pct": float(return_pct),
        "max_dd": float(max_dd),
        "win_rate": float(win_rate),
        "profit_factor": float(profit_factor),
    }


def score_result(metrics: Dict) -> float:
    if metrics["trades"] == 0:
        return -1e9
    dd_penalty = abs(metrics["max_dd"]) + 1e-6
    pf = metrics["profit_factor"] if math.isfinite(metrics["profit_factor"]) else 5.0
    return (metrics["return_pct"] * pf) / dd_penalty


def backtest_bb_squeeze(bars: pd.DataFrame, params: ParamsBBSqueeze, initial_capital: float = 100000.0) -> Dict:
    """Bollinger Band squeeze + volume breakout strategy"""
    close = bars["close"]
    volume = bars["volume"]
    
    upper, mid, lower = calculate_bollinger_bands(close, params.bb_period, params.bb_std)
    band_width = calculate_bb_squeeze(bars, params.bb_period, params.bb_std)
    vol_ma = volume.rolling(params.volume_ma_period).mean()
    
    position = 0
    entry_price = 0.0
    entry_idx = None
    equity = initial_capital
    equity_curve = []
    trades = []
    
    for i in range(len(bars)):
        price = float(close.iloc[i])
        vol = float(volume.iloc[i])
        
        if i < max(params.bb_period, params.volume_ma_period):
            equity_curve.append(equity)
            continue
        
        if math.isnan(band_width.iloc[i]) or math.isnan(vol_ma.iloc[i]):
            equity_curve.append(equity)
            continue
        
        if position == 0:
            # Entry: squeeze detected + high volume breakout
            in_squeeze = band_width.iloc[i] < vol_ma.iloc[i] * params.squeeze_vol_threshold / 100
            high_vol = vol > vol_ma.iloc[i] * params.volume_threshold_mult
            breakout_up = price > upper.iloc[i]
            
            entry_signal = in_squeeze and high_vol and breakout_up
            if entry_signal:
                position = 1
                entry_price = price
                entry_idx = i
                equity -= params.fee_per_trade
        else:
            pnl_pct = (price - entry_price) / entry_price
            hold_bars = i - entry_idx if entry_idx is not None else 0
            
            # Exit: TP, SL, max hold, or mean reversion
            exit_signal = (
                pnl_pct >= params.take_profit
                or pnl_pct <= -params.stop_loss
                or (params.max_hold_bars and hold_bars >= params.max_hold_bars)
                or price <= mid.iloc[i]  # revert to mid
            )
            
            if exit_signal:
                equity *= (1 + pnl_pct * params.position_pct)
                equity -= params.fee_per_trade
                trades.append({
                    "entry_idx": entry_idx,
                    "exit_idx": i,
                    "entry_price": entry_price,
                    "exit_price": price,
                    "pnl_pct": pnl_pct,
                    "hold_bars": hold_bars,
                })
                position = 0
                entry_price = 0.0
                entry_idx = None
        
        equity_curve.append(equity)
    
    return compute_metrics(equity_curve, trades, initial_capital)


def optimize_mean_reversion(bars: pd.DataFrame, params_grid: List[ParamsMeanReversion]) -> pd.DataFrame:
    results = []
    for params in params_grid:
        metrics = backtest_mean_reversion(bars, params)
        row = {
            "score": score_result(metrics),
            **metrics,
            "rsi_entry": params.rsi_entry,
            "rsi_exit": params.rsi_exit,
            "vwap_band": params.vwap_band,
            "take_profit": params.take_profit,
            "stop_loss": params.stop_loss,
            "max_hold_bars": params.max_hold_bars,
        }
        results.append(row)

    df = pd.DataFrame(results)
    df = df.sort_values("score", ascending=False)
    return df


def split_bars(bars: pd.DataFrame, split_ratio: float = 0.7) -> Tuple[pd.DataFrame, pd.DataFrame]:
    split_idx = int(len(bars) * split_ratio)
    return bars.iloc[:split_idx], bars.iloc[split_idx:]


def build_grid_mean_reversion(quick: bool = True) -> List[ParamsMeanReversion]:
    if quick:
        rsi_entries = [30, 35]
        rsi_exits = [50, 55]
        vwap_bands = [0.0020, 0.0025]
        take_profits = [0.0030, 0.0035]
        stop_losses = [0.0035, 0.0040]
        max_holds = [60, 90]
    else:
        rsi_entries = [25, 30, 35, 40]
        rsi_exits = [50, 55, 60]
        vwap_bands = [0.0015, 0.0020, 0.0025, 0.0035]
        take_profits = [0.0025, 0.0030, 0.0035]
        stop_losses = [0.0035, 0.0040, 0.0045]
        max_holds = [30, 60, 90, 120]

    grid = []
    for rsi_entry in rsi_entries:
        for rsi_exit in rsi_exits:
            for vwap_band in vwap_bands:
                for tp in take_profits:
                    for sl in stop_losses:
                        for mh in max_holds:
                            grid.append(
                                ParamsMeanReversion(
                                    rsi_entry=rsi_entry,
                                    rsi_exit=rsi_exit,
                                    vwap_band=vwap_band,
                                    take_profit=tp,
                                    stop_loss=sl,
                                    max_hold_bars=mh,
                                )
                            )
    return grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to CSV with OHLCV data")
    parser.add_argument("--symbol", default="NVDA", help="Symbol for labeling outputs")
    parser.add_argument("--split", type=float, default=0.7, help="Train/test split ratio")
    parser.add_argument("--quick", action="store_true", help="Run a smaller parameter grid")
    parser.add_argument("--strategy", choices=["mr", "bb", "both"], default="both", help="Which strategy to test")
    args = parser.parse_args()

    bars = load_bars(args.csv)
    train_bars, test_bars = split_bars(bars, args.split)

    results_summary = [f"\n{'='*70}", f"STRATEGY COMPARISON: {args.symbol}", f"{'='*70}"]

    # MEAN REVERSION
    if args.strategy in ["mr", "both"]:
        print("\nOptimizing MEAN REVERSION strategy...")
        grid_mr = build_grid_mean_reversion(quick=args.quick)
        train_results_mr = optimize_mean_reversion(train_bars, grid_mr)
        best_mr = train_results_mr.iloc[0]

        best_params_mr = ParamsMeanReversion(
            rsi_entry=int(best_mr["rsi_entry"]),
            rsi_exit=int(best_mr["rsi_exit"]),
            vwap_band=float(best_mr["vwap_band"]),
            take_profit=float(best_mr["take_profit"]),
            stop_loss=float(best_mr["stop_loss"]),
            max_hold_bars=int(best_mr["max_hold_bars"]),
        )

        test_metrics_mr = backtest_mean_reversion(test_bars, best_params_mr)

        mr_lines = [
            f"\n[MEAN REVERSION]",
            f"Best params: RSI entry {best_params_mr.rsi_entry}, RSI exit {best_params_mr.rsi_exit}",
            f"VWAP band {best_params_mr.vwap_band:.4f}, TP {best_params_mr.take_profit:.4f}, SL {best_params_mr.stop_loss:.4f}",
            f"Max hold bars {best_params_mr.max_hold_bars}",
            f"Train: {best_mr['return_pct']:.2f}% return | {best_mr['profit_factor']:.2f} PF | {best_mr['max_dd']:.2%} max DD",
            f"Test:  {test_metrics_mr['return_pct']:.2f}% return | {test_metrics_mr['profit_factor']:.2f} PF | {test_metrics_mr['max_dd']:.2%} max DD",
        ]
        results_summary.extend(mr_lines)
        train_results_mr.to_csv(f"short_interval_{args.symbol.lower()}_mr_results.csv", index=False)

    # BOLLINGER BAND SQUEEZE
    if args.strategy in ["bb", "both"]:
        print("Testing BOLLINGER BAND SQUEEZE strategy...")
        bb_params = ParamsBBSqueeze()
        train_metrics_bb = backtest_bb_squeeze(train_bars, bb_params)
        test_metrics_bb = backtest_bb_squeeze(test_bars, bb_params)

        bb_lines = [
            f"\n[BOLLINGER BAND SQUEEZE + VOLUME]",
            f"Params: BB period {bb_params.bb_period}, std {bb_params.bb_std}, squeeze threshold {bb_params.squeeze_vol_threshold}",
            f"Volume threshold {bb_params.volume_threshold_mult}x, TP {bb_params.take_profit:.4f}, SL {bb_params.stop_loss:.4f}",
            f"Train: {train_metrics_bb['return_pct']:.2f}% return | {train_metrics_bb['profit_factor']:.2f} PF | {train_metrics_bb['max_dd']:.2%} max DD",
            f"Test:  {test_metrics_bb['return_pct']:.2f}% return | {test_metrics_bb['profit_factor']:.2f} PF | {test_metrics_bb['max_dd']:.2%} max DD",
        ]
        results_summary.extend(bb_lines)

    # COMPARISON
    if args.strategy == "both":
        winner = "MEAN REVERSION" if test_metrics_mr["profit_factor"] >= test_metrics_bb["profit_factor"] else "BOLLINGER BAND SQUEEZE"
        comp_lines = [
            f"\n[WINNER: {winner} on test set]",
            f"Mean Reversion PF: {test_metrics_mr['profit_factor']:.2f} vs BB Squeeze PF: {test_metrics_bb['profit_factor']:.2f}",
        ]
        results_summary.extend(comp_lines)

    results_summary.append(f"\n{'='*70}")

    out_summary = f"short_interval_{args.symbol.lower()}_summary.txt"
    with open(out_summary, "w", encoding="ascii") as f:
        f.write("\n".join(results_summary))

    print("\n".join(results_summary))
    print(f"\nSaved summary to: {out_summary}")


if __name__ == "__main__":
    main()
