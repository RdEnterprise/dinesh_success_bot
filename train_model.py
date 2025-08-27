# train_model.py  (robust: normalizes CSV headers & maps common alternatives)
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import sys

# ---------- indicator helpers ----------
def ATR(df, n=14):
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def RSI(series, n=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.rolling(n).mean()
    ma_down = down.rolling(n).mean()
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))

def StochRSI(series, n=14, k=3):
    rsi = RSI(series, n)
    min_rsi = rsi.rolling(n).min()
    max_rsi = rsi.rolling(n).max()
    stochrsi = 100 * (rsi - min_rsi) / (max_rsi - min_rsi)
    return stochrsi.rolling(k).mean()

def Supertrend(df, period=10, multiplier=3):
    atr = ATR(df, period)
    hl2 = (df['high'] + df['low']) / 2
    basic_ub = hl2 + multiplier * atr
    basic_lb = hl2 - multiplier * atr
    final_ub = basic_ub.copy()
    final_lb = basic_lb.copy()
    for i in range(1, len(df)):
        final_ub.iat[i] = basic_ub.iat[i] if (basic_ub.iat[i] < final_ub.iat[i-1] or df['close'].iat[i-1] > final_ub.iat[i-1]) else final_ub.iat[i-1]
        final_lb.iat[i] = basic_lb.iat[i] if (basic_lb.iat[i] > final_lb.iat[i-1] or df['close'].iat[i-1] < final_lb.iat[i-1]) else final_lb.iat[i-1]
    supertrend = np.ones(len(df))
    for i in range(1, len(df)):
        supertrend[i] = -1 if df['close'].iat[i] <= final_ub.iat[i] else 1
    return pd.Series(supertrend, index=df.index), atr

# ---------- labeling ----------
def label_trades(df, atr_series, tp_mult=1.5, sl_mult=1.5, lookahead=12):
    n = len(df)
    labels = np.full(n, -1, dtype=int)
    for i in range(n - lookahead - 1):
        entry_price = df['open'].iat[i+1]
        atr = atr_series.iat[i+1]
        if pd.isna(atr) or atr == 0:
            labels[i] = -1
            continue
        tp = entry_price + tp_mult * atr
        sl = entry_price - sl_mult * atr
        win = False
        for j in range(i+1, i+1+lookahead):
            h = df['high'].iat[j]
            l = df['low'].iat[j]
            if h >= tp and l > sl:
                win = True; break
            if l <= sl and h < tp:
                win = False; break
            if h >= tp and l <= sl:
                c = df['close'].iat[j]
                win = c >= tp
                break
        labels[i] = 1 if win else 0
    return labels

# ---------- main ----------
FN = "XAUUSD_M5.csv"
if not os.path.exists(FN):
    print("ERROR: data file not found:", FN)
    print("Run get_data.py first (make sure MT5 is running and symbol name is correct).")
    sys.exit(1)

df = pd.read_csv(FN)

# Normalize column names: strip spaces and lowercase
df.columns = [c.strip().lower() for c in df.columns]

# Map common alternative names to expected names
col_map = {}
if 'tick_volume' not in df.columns:
    if 'volume' in df.columns:
        col_map['volume'] = 'tick_volume'
    elif 'real_volume' in df.columns:
        col_map['real_volume'] = 'tick_volume'
    elif 'tickvol' in df.columns:
        col_map['tickvol'] = 'tick_volume'
# rename any mapped columns
if col_map:
    df = df.rename(columns=col_map)

# required columns
required = ['time','open','high','low','close','tick_volume']
missing = [c for c in required if c not in df.columns]
if missing:
    print("ERROR: required columns missing from CSV:", missing)
    print("Found columns:", list(df.columns))
    print("If your broker uses a different symbol name, run get_data.py with the exact symbol shown in MT5 Market Watch.")
    sys.exit(1)

# if time column exists but not datetime, convert
try:
    df['time'] = pd.to_datetime(df['time'])
except Exception:
    pass

df = df[['time','open','high','low','close','tick_volume']].copy()
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# compute features
df['atr'] = ATR(df, 14)
df['stochrsi'] = StochRSI(df['close'], 14, 3)
df['stochrsi_d'] = df['stochrsi'].rolling(3).mean()
df['supertrend'], atr_tmp = Supertrend(df, period=10, multiplier=3)
df['atr'] = atr_tmp
df['ret1'] = df['close'].pct_change()
df['vol'] = df['tick_volume'].rolling(5).mean()

df['label'] = label_trades(df, df['atr'], tp_mult=1.5, sl_mult=1.5, lookahead=12)
df = df[df['label'] != -1].dropna().reset_index(drop=True)

features = ['stochrsi','stochrsi_d','supertrend','atr','ret1','vol']
X = df[features]
y = df['label']

split = int(0.8 * len(X))
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

print("Training rows:", len(X_train), "Test rows:", len(X_test))

model = RandomForestClassifier(n_estimators=150, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

pred = model.predict(X_test)
print(classification_report(y_test, pred))

joblib.dump(model, "xau_m5_rf.pkl")
print("Saved model: xau_m5_rf.pkl")

