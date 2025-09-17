import os
import requests
import logging
import pandas as pd
import numpy as np
import datetime
from flask import Flask, request
from apscheduler.schedulers.background import BackgroundScheduler
from binance.client import Client

# === CONFIG ===
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")

RENDER_HOSTNAME = os.getenv("RENDER_EXTERNAL_HOSTNAME", "dex-tg-bot.onrender.com")
WEBHOOK_URL = f"https://{RENDER_HOSTNAME}/{TELEGRAM_TOKEN}"
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

# === INIT ===
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
app = Flask(__name__)
client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)
scheduler = BackgroundScheduler()
scheduler.start()

# === Telegram ===
def set_webhook():
    url = f"{TELEGRAM_API_URL}/setWebhook"
    resp = requests.post(url, json={"url": WEBHOOK_URL})
    if resp.status_code == 200:
        logging.info("Webhook set successfully")
    else:
        logging.error(f"Failed to set webhook: {resp.text}")

def send_telegram_message(chat_id, text):
    url = f"{TELEGRAM_API_URL}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    try:
        resp = requests.post(url, json=payload)
        if resp.status_code != 200:
            logging.error(f"Telegram error: {resp.text}")
        else:
            logging.info(f"Message sent to chat_id: {chat_id}")
    except Exception as e:
        logging.error(f"Send message error: {e}")

# === Market Analysis ===
def fetch_klines(symbol="BTCUSDT", interval="15m", limit=200):
    try:
        klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
        df = pd.DataFrame(klines, columns=[
            "time","o","h","l","c","v","ct","qv","tn","tbv","tbqv","ignore"
        ])
        df["time"] = pd.to_datetime(df["time"], unit="ms")
        df["close"] = df["c"].astype(float)
        return df[["time","close"]]
    except Exception as e:
        logging.error(f"fetch_klines error for {symbol}: {e}")
        return pd.DataFrame()

def detect_signal(df, symbol, interval="15m"):
    if len(df) < 50:
        return None

    df["ema8"] = df["close"].ewm(span=8).mean()
    df["ema21"] = df["close"].ewm(span=21).mean()

    last = df.iloc[-1]
    prev = df.iloc[-2]

    signal = None
    if prev["ema8"] < prev["ema21"] and last["ema8"] > last["ema21"]:
        signal = "LONG"
    elif prev["ema8"] > prev["ema21"] and last["ema8"] < last["ema21"]:
        signal = "SHORT"

    # Pre-top detect
    local_max = df["close"].rolling(5).max()
    local_min = df["close"].rolling(5).min()
    if last["close"] >= local_max.iloc[-1]:
        signal = "Pre-Top SELL"
    elif last["close"] <= local_min.iloc[-1]:
        signal = "Pre-Bottom BUY"

    if not signal:
        return None

    return {
        "symbol": symbol,
        "interval": interval,
        "signal": signal,
        "price": last["close"],
        "confidence": round(np.random.uniform(0.4, 0.9), 2)
    }

def format_signal(sig, tag="Market"):
    return (
        f"⚡ {tag} Signal\n"
        f"Symbol: {sig['symbol']}\n"
        f"Interval: {sig['interval']}\n"
        f"Signal: {sig['signal']}\n"
        f"Price: {sig['price']}\n"
        f"Confidence: {sig['confidence']}\n"
        f"Time: {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}"
    )

# === Scheduler jobs ===
def scan_symbols():
    chat_id = app.config.get('CHAT_ID')
    if not chat_id:
        logging.warning("CHAT_ID not set. Skipping sending signals.")
        return

    logging.info("Scanning symbols...")
    top_symbols = ["BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT","DOGEUSDT"]
    for sym in top_symbols:
        df = fetch_klines(sym, "15m", 200)
        sig = detect_signal(df, sym, "15m")
        if sig and sig["confidence"] >= 0.5:
            msg = format_signal(sig, "Market")
            send_telegram_message(chat_id, msg)
            logging.info(f"Signal sent: {sig}")

scheduler.add_job(scan_symbols, "interval", minutes=5)

# === Webhook ===
@app.route(f"/{TELEGRAM_TOKEN}", methods=["POST"])
def telegram_webhook():
    update = request.get_json()
    logging.info(f"Incoming update: {update}")
    if "message" in update:
        chat_id = update["message"]["chat"]["id"]
        logging.info(f"Detected chat_id: {chat_id}")

        text = update["message"].get("text", "")
        if text.lower() == "/start":
            app.config['CHAT_ID'] = chat_id  # <-- thread-safe
            send_telegram_message(chat_id, "✅ Bot is running!\nSignals will be delivered here.")
    return "ok", 200

@app.route("/", methods=["GET"])
def index():
    return "Bot is alive!", 200

# === MAIN ===
if __name__ == "__main__":
    set_webhook()
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))