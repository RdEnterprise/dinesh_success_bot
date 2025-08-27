# ai_server.py
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

# re-use same indicator helpers (copy from train_model.py)
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

# load model
model = joblib.load("xau_m5_rf.pkl")
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    payload = request.get_json(force=True)
    ohlc = payload.get("ohlc")
    if ohlc is None or len(ohlc) < 20:
        return jsonify({"error":"need at least 20 candles"}), 400

    df = pd.DataFrame(ohlc, columns=['open','high','low','close','tick_volume'])
    df['atr'] = ATR(df, 14)
    df['stochrsi'] = StochRSI(df['close'], 14, 3)
    df['stochrsi_d'] = df['stochrsi'].rolling(3).mean()
    df['supertrend'], atr_tmp = Supertrend(df, period=10, multiplier=3)
    df['atr'] = atr_tmp
    df['ret1'] = df['close'].pct_change()
    df['vol'] = df['tick_volume'].rolling(5).mean()

    last = df.iloc[-1]
    feats = [[
        float(last['stochrsi']),
        float(last['stochrsi_d']),
        int(last['supertrend']),
        float(last['atr']),
        float(last['ret1']),
        float(last['vol'])
    ]]

    prob = float(model.predict_proba(feats)[0][1])
    TH = payload.get("threshold", 0.62)
    decision = "SKIP"
    if prob > TH and int(last['supertrend']) == 1:
        decision = "BUY"
    elif prob > TH and int(last['supertrend']) == -1:
        decision = "SELL"

    entry = float(last['close'])
    atr = float(last['atr'])
    sl = None; tp = None
    if decision == "BUY":
        sl = entry - 1.5 * atr
        tp = entry + 1.5 * atr
    elif decision == "SELL":
        sl = entry + 1.5 * atr
        tp = entry - 1.5 * atr

    return jsonify({"decision":decision,"prob":prob,"entry":entry,"sl":sl,"tp":tp})

if __name__ == "__main__":
    print("AI server running at http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000)
