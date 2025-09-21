import os
import time
import json
import logging
import re
from datetime import datetime, timezone
from threading import Thread
import io
import asyncio
import pandas as pd
import matplotlib.pyplot as plt
import requests
from flask import Flask, request, jsonify
import ta
import mplfinance as mpf
from scipy.signal import find_peaks
import numpy as np
import websockets

# ---------------- CONFIG ----------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
CHAT_ID = os.getenv("CHAT_ID", "")
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "")
PORT = int(os.getenv("PORT", "5000"))

TOP_LIMIT = int(os.getenv("TOP_LIMIT", "50"))  # можна тримати менше монет
SCAN_INTERVAL_MINUTES = int(os.getenv("SCAN_INTERVAL_MINUTES", "1"))
PARALLEL_WORKERS = int(os.getenv("PARALLEL_WORKERS", "6"))

STATE_FILE = "state.json"
LOG_FILE = "bot.log"

CONF_THRESHOLD_MEDIUM = 0.60
CONF_THRESHOLD_STRONG = 0.80

# ---------------- LOGGING ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)
logger = logging.getLogger("pretop-bot")

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
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
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

def send_telegram(text: str, photo=None):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        return
    try:
        if photo:
            files = {'photo': ('signal.png', photo, 'image/png')}
            data = {'chat_id': CHAT_ID, 'caption': escape_md_v2(text), 'parse_mode': 'MarkdownV2'}
            requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto", data=data, files=files, timeout=10)
        else:
            payload = {
                "chat_id": CHAT_ID,
                "text": escape_md_v2(text),
                "parse_mode": "MarkdownV2",
                "disable_web_page_preview": True
            }
            requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", json=payload, timeout=10)
    except Exception as e:
        logger.exception("send_telegram error: %s", e)

# ---------------- WEBSOCKET KLINE MANAGER ----------------
class WebSocketKlineManager:
    def __init__(self, symbols, interval="15m"):
        self.symbols = [s.lower() for s in symbols]
        self.interval = interval
        self.data = {s.upper(): pd.DataFrame(columns=["open", "high", "low", "close", "volume"]) for s in symbols}
        self.tasks = []

    async def _subscribe(self):
        streams = "/".join([f"{s}@kline_{self.interval}" for s in self.symbols])
        url = f"wss://stream.binance.com:9443/stream?streams={streams}"
        while True:
            try:
                async with websockets.connect(url, ping_interval=20) as ws:
                    async for msg in ws:
                        data = json.loads(msg)
                        if "data" not in data:
                            continue
                        kline = data["data"]["k"]
                        if not kline["x"]:  # свічка ще не закрита
                            continue
                        symbol = kline["s"]
                        ts = pd.to_datetime(kline["t"], unit="ms", utc=True)
                        self.data[symbol].loc[ts] = {
                            "open": float(kline["o"]),
                            "high": float(kline["h"]),
                            "low": float(kline["l"]),
                            "close": float(kline["c"]),
                            "volume": float(kline["v"])
                        }
            except Exception as e:
                logger.error(f"[WebSocket] error: {e}, reconnecting in 5s...")
                await asyncio.sleep(5)

    def start(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._subscribe())

    def get_klines(self, symbol, limit=500):
        df = self.data.get(symbol.upper())
        if df is None or len(df) < 10:
            return None
        return df.tail(limit).copy()

# ---------------- INIT SYMBOLS ----------------
SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT"]  # можна розширити список
ws_manager = WebSocketKlineManager(SYMBOLS)

def run_ws():
    ws_manager.start()

Thread(target=run_ws, daemon=True).start()

# ---------------- FEATURES & SIGNALS ----------------
def apply_all_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    try:
        df["ema_8"] = ta.trend.EMAIndicator(df["close"], 8).ema_indicator()
        df["ema_20"] = ta.trend.EMAIndicator(df["close"], 20).ema_indicator()
        df["RSI_14"] = ta.momentum.RSIIndicator(df["close"], 14).rsi()
        macd = ta.trend.MACD(df["close"])
        df["MACD"] = macd.macd()
        df["MACD_signal"] = macd.macd_signal()
        df["MACD_hist"] = macd.macd_diff()
        adx = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], 14)
        df["ADX"] = adx.adx()
        df["ADX_pos"] = adx.adx_pos()
        df["ADX_neg"] = adx.adx_neg()
        df["support"] = df["low"].rolling(20).min()
        df["resistance"] = df["high"].rolling(20).max()
    except Exception as e:
        logger.exception("apply_all_features error: %s", e)
    return df

def detect_signal(df: pd.DataFrame):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    votes = []
    confidence = 0.2

    if last["ema_8"] > last["ema_20"]:
        votes.append("ema_bull")
        confidence += 0.1
    else:
        votes.append("ema_bear")
        confidence += 0.05

    if last["MACD_hist"] > 0:
        votes.append("macd_up")
        confidence += 0.1
    else:
        votes.append("macd_down")
        confidence += 0.05

    if last["RSI_14"] < 30:
        votes.append("rsi_oversold")
        confidence += 0.08
    elif last["RSI_14"] > 70:
        votes.append("rsi_overbought")
        confidence += 0.08

    if last["ADX"] > 25:
        votes.append("strong_trend")
        confidence *= 1.1

    # свічкові патерни
    body = last["close"] - last["open"]
    rng = last["high"] - last["low"]
    upper_shadow = last["high"] - max(last["close"], last["open"])
    lower_shadow = min(last["close"], last["open"]) - last["low"]
    candle_bonus = 1.0

    if lower_shadow > 2 * abs(body) and body > 0:
        votes.append("hammer_bull")
        candle_bonus = 1.2
    elif upper_shadow > 2 * abs(body) and body < 0:
        votes.append("shooting_star")
        candle_bonus = 1.2

    if body > 0 and prev["close"] < prev["open"] and last["close"] > prev["open"] and last["open"] < prev["close"]:
        votes.append("bullish_engulfing")
        candle_bonus = 1.25
    elif body < 0 and prev["close"] > prev["open"] and last["close"] < prev["open"] and last["open"] > prev["close"]:
        votes.append("bearish_engulfing")
        candle_bonus = 1.25

    if abs(body) < 0.1 * rng:
        votes.append("doji")
        candle_bonus = 1.1

    confidence *= candle_bonus

    # fake breakout
    if last["close"] > last["resistance"] * 0.995 and last["close"] < last["resistance"] * 1.01:
        votes.append("fake_breakout_short")
        confidence += 0.15
    elif last["close"] < last["support"] * 1.005 and last["close"] > last["support"] * 0.99:
        votes.append("fake_breakout_long")
        confidence += 0.15

    # pre-top
    pretop = False
    if len(df) >= 10:
        recent, last10 = df["close"].iloc[-1], df["close"].iloc[-10]
        if (recent - last10) / last10 > 0.1:
            pretop = True
            confidence += 0.1
            votes.append("pretop")

    action = "WATCH"
    if last["close"] >= last["resistance"] * 0.995:
        action = "SHORT"
    elif last["close"] <= last["support"] * 1.005:
        action = "LONG"

    confidence = max(0, min(1, confidence))
    return action, votes, pretop, last, confidence

# ---------------- ANALYZE SYMBOL ----------------
def analyze_and_alert(symbol: str):
    df = ws_manager.get_klines(symbol)
    if df is None or len(df) < 30:
        return
    df = apply_all_features(df)
    action, votes, pretop, last, confidence = detect_signal(df)
    prev_signal = state["signals"].get(symbol, "")

    logger.info("Symbol=%s action=%s confidence=%.2f votes=%s pretop=%s",
                symbol, action, confidence, [v for v in votes], pretop)

    if pretop:
        send_telegram(f"⚡ Pre-top detected for {symbol}, price={last['close']:.6f}")

    if action != "WATCH" and confidence >= CONF_THRESHOLD_MEDIUM and action != prev_signal:
        msg = (
            f"⚡ TRADE SIGNAL\n"
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
    for sym in SYMBOLS:
        analyze_and_alert(sym)
    state["last_scan"] = str(datetime.now(timezone.utc))
    save_json_safe(STATE_FILE, state)
    logger.info("Scan finished at %s", state["last_scan"])

# ---------------- FLASK ROUTES ----------------
@app.route("/")
def home():
    return jsonify({
        "status": "ok",
        "time": str(datetime.now(timezone.utc)),
        "signals": len(state.get("signals", {}))
    })

@app.route("/telegram_webhook/<token>", methods=["POST"])
def telegram_webhook(token):
    try:
        if token != TELEGRAM_TOKEN:
            logger.warning("Invalid token on webhook: %s", token)
            return jsonify({"ok": False, "error": "invalid token"}), 403

        update = request.get_json(force=True) or {}
        msg = update.get("message")
        if not msg:
            return jsonify({"ok": True})

        text = msg.get("text", "").lower().strip()

        if text.startswith("/scan"):
            send_telegram("⚡ Manual scan started.")
            Thread(target=scan_top_symbols, daemon=True).start()

        elif text.startswith("/status"):
            send_telegram(
                f"📝 Status:\nSignals={len(state.get('signals', {}))}\nLast scan={state.get('last_scan')}"
            )
    except Exception as e:
        logger.exception("telegram_webhook error: %s", e)
    return jsonify({"ok": True})

# ---------------- MAIN ----------------
if __name__ == "__main__":
    logger.info("Starting pre-top detector bot (WebSocket only)")
    Thread(target=scan_top_symbols, daemon=True).start()
    app.run(host="0.0.0.0", port=PORT)