@echo off
REM Launch the institutional-grade short bot
REM Run multiple symbols in paper trading mode

cd c:\Users\Ryan\Alpaca\ Trader
call .\bot_env\Scripts\activate.bat

echo.
echo =========================================
echo ALPACA SHORT BOT V2 - LAUNCH
echo =========================================
echo Mode: Paper Trading (safe)
echo Symbols: NVDA TSLA XLK
echo Check Interval: 60 seconds
echo.
echo Features:
echo - Bollinger Band squeeze + volume breakout
echo - Multi-symbol trading
echo - Risk-managed position sizing
echo - Circuit breaker protection
echo - Real-time performance stats
echo.

python "Trader Bot Short\alpaca_short_bot_v2.py" --mode paper --symbols NVDA TSLA XLK --interval 60

pause
