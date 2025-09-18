# main.py — Pre-top bot з WebSocket для Render
import os
import json
import logging
import re
from datetime import datetime, timezone
from threading import Thread
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import numpy as np
import requests
from flask import Flask, request, jsonify
from apscheduler.schedulers.background import BackgroundScheduler
import ta

# ---------------- BINANCE CLIENT ----------------
from binance.client import Client as BinanceClient
from binance.streams import ThreadedWebsocketManager

# ---------------- CONFIG ----------------
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
CHAT_ID = os.getenv("CHAT_ID", "")
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "")
PORT = int(os.getenv("PORT", "5000"))

TOP_LIMIT = int(os.getenv("TOP_LIMIT", "100"))
PARALLEL_WORKERS = int(os.getenv("PARALLEL_WORKERS", "6"))

STATE_FILE = "state.json"
LOG_FILE = "bot.log"

CONF_THRESHOLD_MEDIUM = 0.1  # менший поріг для тесту
CONF_THRESHOLD_STRONG = 0.2

# ---------------- LOGGING ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)
logger = logging.getLogger("pretop-bot")

# ---------------- BINANCE ----------------
client = BinanceClient(api_key=BINANCE_API_KEY, api_secret=BINANCE_API_SECRET)
twm = ThreadedWebsocketManager(api_key=BINANCE_API_KEY, api_secret=BINANCE_API_SECRET)

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
MARKDOWNV2_ESCAPE = r"_*[]()~`>#+-=|{}.!"

def escape_md_v2(text: str) -> str:
    return re.sub(f"([{re.escape(MARKDOWNV2_ESCAPE)}])", r"\\\1", str(text))

def send_telegram(text: str):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        return
    try:
        payload = {
            "chat_id": CHAT_ID,
            "text": escape_md_v2(text),
            "parse_mode": "MarkdownV2",
            "disable_web_page_preview": True
        }
        r = requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", json=payload, timeout=10)
        if r.status_code != 200:
            logger.error("Telegram send failed: %s %s", r.status_code, r.text)
    except Exception as e:
        logger.exception("send_telegram error: %s", e)

def set_telegram_webhook(webhook_url: str):
    if not TELEGRAM_TOKEN or not webhook_url:
        return
    try:
        resp = requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/setWebhook", json={"url": webhook_url}, timeout=10)
        logger.info("setWebhook resp: %s", resp.text if resp else "None")
    except Exception as e:
        logger.exception("set_telegram_webhook error: %s", e)

# ---------------- MARKET DATA ----------------
def get_all_usdt_symbols():
    try:
        ex = client.get_exchange_info()
        return [s["symbol"] for s in ex["symbols"] if s["quoteAsset"] == "USDT" and s["status"] == "TRADING"]
    except Exception as e:
        logger.exception("get_all_usdt_symbols error: %s", e)
        return []

def fetch_klines(symbol, interval="1m", limit=500):
    try:
        kl = client.get_klines(symbol=symbol, interval=interval, limit=limit)
        df = pd.DataFrame(kl, columns=["open_time", "open", "high", "low", "close", "volume",
                                       "close_time", "qav", "num_trades", "tb_base", "tb_quote", "ignore"])
        df = df[["open_time", "open", "high", "low", "close", "volume"]].astype(float)
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df.set_index("open_time", inplace=True)
        return df
    except Exception as e:
        logger.warning("fetch_klines error %s: %s", symbol, e)
        return None

# ---------------- FEATURE ENGINEERING ----------------
def apply_all_features(df):
    df = df.copy()
    try:
        df["ema_8"] = ta.trend.EMAIndicator(df["close"], 8).ema_indicator()
        df["ema_20"] = ta.trend.EMAIndicator(df["close"], 20).ema_indicator()
        df["RSI_14"] = ta.momentum.RSIIndicator(df["close"], 14).rsi()
        stoch = ta.momentum.StochRSIIndicator(df["close"], 14, 3, 3)
        df["stoch_k"] = stoch.stochrsi_k()
        df["stoch_d"] = stoch.stochrsi_d()
        df["ha_close"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
        ha_open_vals = [(df["open"].iloc[0] + df["close"].iloc[0]) / 2]
        for i in range(1, len(df)):
            ha_open_vals.append((ha_open_vals[-1] + df["ha_close"].iloc[i - 1]) / 2)
        df["ha_open"] = ha_open_vals
        df["ha_dir"] = np.sign(df["ha_close"] - df["ha_open"])
        df["support"] = df["low"].rolling(20).min()
        df["resistance"] = df["high"].rolling(20).max()
    except Exception as e:
        logger.exception("apply_all_features error: %s", e)
    return df

# ---------------- SIGNAL DETECTION ----------------
def detect_signal(df):
    last = df.iloc[-1]
    votes = []

    if last["ema_8"] > last["ema_20"]:
        votes.append("ema_bull")
    else:
        votes.append("ema_bear")

    if last["RSI_14"] < 30:
        votes.append("rsi_oversold")
    elif last["RSI_14"] > 70:
        votes.append("rsi_overbought")

    if df["ha_dir"].iloc[-5:-1].gt(0).all() and last["ha_dir"] < 0:
        votes.append("ha_exhaustion")
    if df["ha_dir"].iloc[-5:-1].lt(0).all() and last["ha_dir"] > 0:
        votes.append("ha_exhaustion_buy")

    pretop = False
    if len(df) >= 10 and (last["close"] - df["close"].iloc[-10]) / df["close"].iloc[-10] > 0.1:
        pretop = True

    action = "WATCH"
    if last["close"] >= last["resistance"] * 0.995:
        action = "SHORT"
    elif last["close"] <= last["support"] * 1.005:
        action = "LONG"

    confidence = min(1.0, 0.05 * len(votes) + (0.1 if pretop else 0))
    return action, votes, pretop, last, confidence

# ---------------- ANALYZE SYMBOL ----------------
def analyze_and_alert(symbol):
    df = fetch_klines(symbol)
    if df is None or len(df) < 30:
        return
    df = apply_all_features(df)
    action, votes, pretop, last, confidence = detect_signal(df)

    prev_signal = state["signals"].get(symbol, "")

    if pretop:
        send_telegram(f"⚡ Pre-top detected for {symbol}, price={last['close']:.6f}")

    if action != "WATCH" and confidence > CONF_THRESHOLD_MEDIUM and action != prev_signal:
        msg = (
            f"⚡ <b>TRADE SIGNAL</b>\n"
            f"Symbol: {symbol}\n"
            f"Action: {action}\n"
            f"Price: {last['close']:.6f}\n"
            f"Support: {last['support']:.6f}\n"
            f"Resistance: {last['resistance']:.6f}\n"
            f"Confidence: {confidence:.2f}\n"
            f"Patterns: {','.join(votes)}\n"
            f"Pre-top: {pretop}\n"
            f"Time: {last.name}\n"
        )
        send_telegram(msg)
        state["signals"][symbol] = action
        save_json_safe(STATE_FILE, state)

# ---------------- MASTER SCAN ----------------
def scan_top_symbols():
    symbols = get_all_usdt_symbols()
    if not symbols:
        logger.warning("No symbols found for scanning.")
        return

    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as exe:
        list(exe.map(analyze_and_alert, symbols))

    state["last_scan"] = str(datetime.now(timezone.utc))
    save_json_safe(STATE_FILE, state)
    logger.info("Scan finished at %s", state["last_scan"])

# ---------------- SCHEDULER ----------------
scheduler = BackgroundScheduler()
scheduler.add_job(scan_top_symbols, "interval", minutes=1, next_run_time=datetime.now())
scheduler.start()
logger.info("Scheduler started")

# ---------------- FLASK ROUTES ----------------
@app.route("/")
def home():
    return jsonify({"status": "ok", "time": str(datetime.now(timezone.utc)), "state_signals_count": len(state.get("signals", {}))})

@app.route(f"/telegram_webhook/<token>", methods=["POST", "GET"])
def telegram_webhook(token):
    if token != TELEGRAM_TOKEN:
        return jsonify({"ok": False, "reason": "invalid token"}), 403
    if request.method == "POST":
        update = request.get_json(force=True)
        if "message" in update:
            msg = update["message"]
            text = msg.get("text", "").lower()
            if text.startswith("/scan"):
                Thread(target=scan_top_symbols, daemon=True).start()
                send_telegram("Manual scan started.")
            elif text.startswith("/status"):
                send_telegram(f"Status:\nSignals={len(state.get('signals', {}))}\nLast scan={state.get('last_scan')}")
    return jsonify({"ok": True})

# ---------------- AUTO REGISTER WEBHOOK ----------------
if WEBHOOK_URL and TELEGRAM_TOKEN:
    set_telegram_webhook(WEBHOOK_URL)

# ---------------- MAIN ----------------
if __name__ == "__main__":
    logger.info("Starting pre-top detector bot")
    app.run(host="0.0.0.0", port=PORT)