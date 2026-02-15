@echo off
REM Multi-symbol 750 EMA bot (NVDA, TSLA, XLK)
REM Same dual-layer strategy as original bot but trades all three

cd c:\Users\Ryan\Alpaca\ Trader
call .\bot_env\Scripts\activate.bat

echo.
echo =========================================
echo MULTI-SYMBOL 750 EMA BOT - PAPER TRADING
echo =========================================
echo Symbols: NVDA, TSLA, XLK
echo Mode: Paper Trading (safe to test)
echo Strategy: 750 EMA + 4-layer position sizing
echo Check Interval: 5 minutes
echo.

python "Trader Bot Short\alpaca_750ema_multi.py" --mode paper --symbols NVDA TSLA XLK --interval 300

pause
