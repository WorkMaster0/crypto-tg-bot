# main.py — Pre-top бот з графіками, авто-видаленням фото та fake breakout
import os
import time
import json
import logging
import re
from datetime import datetime, timezone
from threading import Thread
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify
from apscheduler.schedulers.background import BackgroundScheduler
import ta

# ---------------- CONFIG ----------------
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
CHAT_ID = os.getenv("CHAT_ID", "")
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "")
PORT = int(os.getenv("PORT", "5000"))

TOP_LIMIT = 100
SCAN_INTERVAL_MINUTES = 1
EMA_SCAN_LIMIT = 300
PARALLEL_WORKERS = 6

STATE_FILE = "state.json"
LOG_FILE = "bot.log"

CONF_THRESHOLD_MEDIUM = 0.35
CONF_THRESHOLD_STRONG = 0.55

# ---------------- LOGGING ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)
logger = logging.getLogger("pretop-bot")

# ---------------- BINANCE ----------------
try:
    from binance.client import Client as BinanceClient
    client = BinanceClient(api_key=BINANCE_API_KEY, api_secret=BINANCE_API_SECRET)
except Exception:
    client = None
    logger.warning("Binance client unavailable")

# ---------------- FLASK ----------------
app = Flask(__name__)

# ---------------- STATE ----------------
def load_json_safe(path, default):
    try:
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
    except Exception as e:
        logger.exception("load_json_safe error %s: %s", path, e)
    return default

def save_json_safe(path, data):
    try:
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp, path)
    except Exception as e:
        logger.exception("save_json_safe error %s: %s", path, e)

state = load_json_safe(STATE_FILE, {"signals": {}, "last_scan": None, "signal_history": {}})

# ---------------- TELEGRAM ----------------
def send_telegram(text: str):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        return
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": CHAT_ID, "text": text, "parse_mode": "HTML"},
            timeout=10
        )
        if r.status_code != 200:
            logger.error("Telegram send failed: %s %s", r.status_code, r.text)
    except Exception as e:
        logger.exception("send_telegram error: %s", e)

def send_telegram_photo(photo_path, caption="", delete_after=600):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
        with open(photo_path, "rb") as f:
            files = {"photo": f}
            data = {"chat_id": CHAT_ID, "caption": caption, "parse_mode": "HTML"}
            r = requests.post(url, files=files, data=data, timeout=20)
            if r.status_code == 200:
                msg_id = r.json().get("result", {}).get("message_id")
                logger.info("Photo sent: msg_id=%s", msg_id)
                if msg_id:
                    scheduler.add_job(
                        delete_telegram_message,
                        "date",
                        run_date=datetime.now(timezone.utc) + pd.Timedelta(seconds=delete_after),
                        args=[msg_id],
                    )
            else:
                logger.error("Telegram photo send failed: %s %s", r.status_code, r.text)
    except Exception as e:
        logger.exception("send_telegram_photo error: %s", e)
    finally:
        try:
            os.remove(photo_path)
        except Exception:
            pass

def delete_telegram_message(message_id):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/deleteMessage"
        requests.post(url, json={"chat_id": CHAT_ID, "message_id": message_id}, timeout=10)
        logger.info("Deleted Telegram message_id=%s", message_id)
    except Exception as e:
        logger.exception("delete_telegram_message error: %s", e)

# ---------------- MARKET DATA ----------------
def get_all_usdt_symbols():
    if not client:
        return []
    try:
        ex = client.get_exchange_info()
        # Відсіюємо стейблкоїни та фондові інструменти
        blacklist = {"USDC", "FDUSD", "BUSD", "EUR", "TRY", "PAXG"}
        return [
            s["symbol"]
            for s in ex["symbols"]
            if s["quoteAsset"] == "USDT"
            and s["status"] == "TRADING"
            and not any(stable in s["baseAsset"] for stable in blacklist)
        ]
    except Exception as e:
        logger.exception("get_all_usdt_symbols error: %s", e)
        return []

def fetch_klines(symbol, interval="15m", limit=EMA_SCAN_LIMIT):
    try:
        kl = client.get_klines(symbol=symbol, interval=interval, limit=limit)
        df = pd.DataFrame(kl, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "qav", "num_trades", "tb_base", "tb_quote", "ignore"
        ])
        df = df[["open_time", "open", "high", "low", "close", "volume"]].astype(float)
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df.set_index("open_time", inplace=True)
        return df
    except Exception as e:
        logger.warning("fetch_klines %s error: %s", symbol, e)
        return None

# ---------------- FEATURES ----------------
def apply_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    try:
        df["ema_20"] = ta.trend.EMAIndicator(df["close"], 20).ema_indicator()
        df["ema_50"] = ta.trend.EMAIndicator(df["close"], 50).ema_indicator()
        df["RSI"] = ta.momentum.RSIIndicator(df["close"], 14).rsi()
        df["ATR"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"]).average_true_range()
        df["support"] = df["low"].rolling(20).min()
        df["resistance"] = df["high"].rolling(20).max()
    except Exception as e:
        logger.exception("apply_features error: %s", e)
    return df

# ---------------- SIGNALS ----------------
def detect_signal(df: pd.DataFrame):
    last = df.iloc[-1]
    votes = []

    if last["ema_20"] > last["ema_50"]:
        votes.append(("trend_up", 0.2))
    if last["RSI"] < 30:
        votes.append(("rsi_oversold", 0.15))
    if last["RSI"] > 70:
        votes.append(("rsi_overbought", -0.15))

    # Fake breakout (свічка закрилась нижче підтримки і швидко повернулась)
    if last["low"] < last["support"] * 0.995 and last["close"] > last["support"]:
        votes.append(("fake_breakout_long", 0.25))
    if last["high"] > last["resistance"] * 1.005 and last["close"] < last["resistance"]:
        votes.append(("fake_breakout_short", 0.25))

    action = "WATCH"
    if any("long" in v[0] for v in votes):
        action = "LONG"
    elif any("short" in v[0] for v in votes):
        action = "SHORT"

    confidence = max(0, min(1, 0.3 + sum(w for _, w in votes)))
    return action, votes, last, confidence

# ---------------- ANALYZE ----------------
def plot_signal_chart(symbol, df, action, votes, confidence, file_path):
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df["close"], label="Close", color="black")
    plt.plot(df.index, df["ema_20"], label="EMA 20", color="blue", alpha=0.7)
    plt.plot(df.index, df["ema_50"], label="EMA 50", color="red", alpha=0.7)
    plt.axhline(df["support"].iloc[-1], color="green", linestyle="--", label="Support")
    plt.axhline(df["resistance"].iloc[-1], color="orange", linestyle="--", label="Resistance")
    plt.title(f"{symbol} | {action} | Conf={confidence:.2f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()

def analyze_and_alert(symbol: str):
    df = fetch_klines(symbol)
    if df is None or len(df) < 50:
        return
    df = apply_features(df)
    action, votes, last, confidence = detect_signal(df)
    prev_signal = state["signals"].get(symbol, "")

    if action != "WATCH" and confidence >= CONF_THRESHOLD_MEDIUM and action != prev_signal:
        caption = (
            f"⚡ <b>TRADE SIGNAL</b>\n"
            f"Symbol: {symbol}\n"
            f"Action: {action}\n"
            f"Price: {last['close']:.6f}\n"
            f"Confidence: {confidence:.2f}\n"
            f"Patterns: {','.join([v[0] for v in votes])}\n"
            f"Time: {last.name}"
        )
        chart_path = f"{symbol}_{int(time.time())}.png"
        plot_signal_chart(symbol, df, action, votes, confidence, chart_path)
        send_telegram_photo(chart_path, caption=caption, delete_after=600)
        state["signals"][symbol] = action
        save_json_safe(STATE_FILE, state)

# ---------------- MASTER SCAN ----------------
def scan_top_symbols():
    symbols = get_all_usdt_symbols()
    if not symbols:
        return
    logger.info("Scanning %d symbols", len(symbols))
    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as exe:
        list(exe.map(analyze_and_alert, symbols))
    state["last_scan"] = str(datetime.now(timezone.utc))
    save_json_safe(STATE_FILE, state)
    logger.info("Scan finished at %s", state["last_scan"])

# ---------------- SCHEDULER ----------------
scheduler = BackgroundScheduler()
scheduler.add_job(scan_top_symbols, "interval", minutes=SCAN_INTERVAL_MINUTES,
                  id="scan_job", next_run_time=datetime.now())
scheduler.start()

# ---------------- FLASK ----------------
@app.route("/")
def home():
    return jsonify({"status": "ok", "time": str(datetime.now(timezone.utc))})

@app.route(f"/telegram_webhook/{TELEGRAM_TOKEN}", methods=["POST", "GET"])
def telegram_webhook():
    if request.method == "POST":
        update = request.get_json(force=True)
        if "message" in update:
            text = update["message"].get("text", "").lower()
            if text.startswith("/scan"):
                Thread(target=scan_top_symbols, daemon=True).start()
                send_telegram("Manual scan started.")
            elif text.startswith("/status"):
                send_telegram(f"Status:\nSignals={len(state.get('signals', {}))}\nLast scan={state.get('last_scan')}")
    return jsonify({"ok": True})

# ---------------- STARTUP ----------------
Thread(target=scan_top_symbols, daemon=True).start()

if __name__ == "__main__":
    logger.info("Starting bot")
    app.run(host="0.0.0.0", port=PORT)