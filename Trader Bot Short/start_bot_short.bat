@echo off
cd c:\Users\Ryan\Alpaca Trader
call .\bot_env\Scripts\activate.bat
python "Trading Bot v1\alpaca_live_runner_short.py"
pause
