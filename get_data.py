# get_data.py
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime

symbol = "XAUUSD"          # <-- check Market Watch if your broker uses variant like XAUUSDm or GOLD
timeframe = mt5.TIMEFRAME_M5
utc_from = datetime(2025,7,1)   # last month
utc_to   = datetime(2025,8,1)   # until Aug 1 2025


if not mt5.initialize():
    print("MT5 initialize() failed, check MT5 is running")
    raise SystemExit

rates = mt5.copy_rates_range(symbol, timeframe, utc_from, utc_to)
if rates is None or len(rates) == 0:
    print("No data. Check symbol name and MT5 terminal.")
    mt5.shutdown()
    raise SystemExit

df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')
df = df[['time','open','high','low','close','tick_volume']]
df.to_csv("XAUUSD_M5.csv", index=False)
print("Saved XAUUSD_M5.csv rows:", len(df))

mt5.shutdown()
