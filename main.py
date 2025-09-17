import os
import logging
import requests
import pandas as pd
import ta
from binance.client import Client
from flask import Flask, request

# --- Налаштування ---
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# --- Отримання топ USDT пар ---
def get_top_symbols(limit=10):
    tickers = client.get_ticker()
    usdt_pairs = [t for t in tickers if t["symbol"].endswith("USDT")]
    sorted_pairs = sorted(usdt_pairs, key=lambda x: float(x["quoteVolume"]), reverse=True)
    return [p["symbol"] for p in sorted_pairs[:limit]]

# --- Завантаження даних ---
def fetch_klines(symbol, interval="15m", limit=500):
    raw = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(raw, columns=[
        "time","open","high","low","close","volume","c1","c2","c3","c4","c5","c6"
    ])
    df = df[["time","open","high","low","close","volume"]].astype(float)
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    return df

# --- Фічі ---
def apply_features(df):
    df["ema_20"] = ta.trend.EMAIndicator(df["close"], 20).ema_indicator()
    df["ema_50"] = ta.trend.EMAIndicator(df["close"], 50).ema_indicator()
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], 14).rsi()
    df["atr"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], 14).average_true_range()
    df["macd"] = ta.trend.MACD(df["close"]).macd()
    df["macd_signal"] = ta.trend.MACD(df["close"]).macd_signal()
    df["stoch_rsi"] = ta.momentum.StochRSIIndicator(df["close"]).stochrsi()
    df["adx"] = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], 14).adx()
    # TODO: додати smart money features (fair value gap, imbalance, liquidity grab)
    return df

# --- Pre-top detect ---
def detect_signal(df):
    latest = df.iloc[-1]
    signals = []

    # EMA cross
    if latest["ema_20"] > latest["ema_50"]:
        signals.append("ema_bull")
    elif latest["ema_20"] < latest["ema_50"]:
        signals.append("ema_bear")

    # RSI zones
    if latest["rsi"] < 30:
        signals.append("rsi_oversold")
    elif latest["rsi"] > 70:
        signals.append("rsi_overbought")

    # MACD
    if latest["macd"] > latest["macd_signal"]:
        signals.append("macd_bull")
    elif latest["macd"] < latest["macd_signal"]:
        signals.append("macd_bear")

    # ADX strong trend
    if latest["adx"] > 25:
        signals.append("trend_strong")

    confidence = len(signals) / 5
    return signals, confidence

# --- Надсилання у Telegram ---
def send_signal(symbol, signals, confidence, price):
    if confidence < 0.5:  # фільтр слабих
        return
    text = f"""
⚡ *Signal*
Symbol: `{symbol}`
Price: `{price}`
Signals: {", ".join(signals)}
Confidence: *{confidence:.2f}*
"""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    requests.post(url, json={"chat_id": CHAT_ID, "text": text, "parse_mode": "Markdown"})

# --- Основний цикл ---
def run_analysis():
    symbols = get_top_symbols(limit=10)
    logging.info(f"Analyzing {symbols}")
    for s in symbols:
        df = fetch_klines(s, interval="15m", limit=200)
        df = apply_features(df)
        signals, conf = detect_signal(df)
        send_signal(s, signals, conf, df["close"].iloc[-1])

# --- Flask webhook ---
@app.route(f"/{TELEGRAM_TOKEN}", methods=["POST"])
def webhook():
    update = request.json
    logging.info(update)
    return "ok", 200

if __name__ == "__main__":
    run_analysis()
    app.run(host="0.0.0.0", port=5000)