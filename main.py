# main_ws.py — Pre-top бот з WebSocket, Telegram і backtest winrate
import os
import json
import logging
import re
from datetime import datetime, timezone
from threading import Thread
from concurrent.futures import ThreadPoolExecutor
import io

import pandas as pd
import matplotlib.pyplot as plt
import ta
import mplfinance as mpf
from scipy.signal import find_peaks
import numpy as np
import websocket
import threading
import time
import requests

# ---------------- CONFIG ----------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
CHAT_ID = os.getenv("CHAT_ID", "")
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "")
PORT = int(os.getenv("PORT", "5000"))

TOP_LIMIT = int(os.getenv("TOP_LIMIT", "100"))
EMA_SCAN_LIMIT = int(os.getenv("EMA_SCAN_LIMIT", "500"))
PARALLEL_WORKERS = int(os.getenv("PARALLEL_WORKERS", "6"))

STATE_FILE = "state.json"
LOG_FILE = "bot_ws.log"

CONF_THRESHOLD_MEDIUM = 0.60
CONF_THRESHOLD_STRONG = 0.80

# ---------------- LOGGING ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)
logger = logging.getLogger("pretop-bot-ws")

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
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp, path)
    except Exception as e:
        logger.exception("save_json_safe error %s: %s", path, e)

state = load_json_safe(STATE_FILE, {"signals": {}, "last_update": None, "signal_history": {}})

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

# ---------------- MARKET DATA ----------------
def get_all_usdt_symbols():
    try:
        url = "https://api.binance.com/api/v3/exchangeInfo"
        ex = requests.get(url, timeout=10).json()
        symbols = [
            s["symbol"] for s in ex["symbols"]
            if s["quoteAsset"] == "USDT" and s["status"] == "TRADING"
        ]
        blacklist = [
            "BUSD", "USDC", "FDUSD", "TUSD", "DAI", "EUR", "GBP", "AUD", "STRAX", "GNS", "ALCX",
            "BTCST", "COIN", "AAPL", "TSLA", "MSFT", "META", "GOOG", "USD1", "BTTC", "ARDR", "DF", "XNO"
        ]
        filtered = [s for s in symbols if not any(b in s for b in blacklist)]
        return filtered[:TOP_LIMIT]
    except Exception as e:
        logger.exception("get_all_usdt_symbols error: %s", e)
        return []

# ---------------- FEATURE ENGINEERING ----------------
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

# ---------------- SIGNAL DETECTION ----------------
def detect_signal(df: pd.DataFrame):
    if len(df) < 2:
        return "WATCH", [], False, None, 0.0

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

    # Fake breakout
    if last["close"] > last["resistance"] * 0.995 and last["close"] < last["resistance"] * 1.01:
        votes.append("fake_breakout_short")
        confidence += 0.15
    elif last["close"] < last["support"] * 1.005 and last["close"] > last["support"] * 0.99:
        votes.append("fake_breakout_long")
        confidence += 0.15

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

# ---------------- WEBSOCKET ----------------
symbol_dfs = {}
lock = threading.Lock()

def on_message(ws, message):
    import json
    msg = json.loads(message)
    if "k" not in msg:
        return
    k = msg["k"]
    symbol = msg["s"]
    candle_closed = k["x"]
    close = float(k["c"])
    high = float(k["h"])
    low = float(k["l"])
    open_ = float(k["o"])
    volume = float(k["v"])
    timestamp = pd.to_datetime(k["t"], unit="ms", utc=True)

    with lock:
        df = symbol_dfs.get(symbol, pd.DataFrame(columns=["open","high","low","close","volume"]))
        df.loc[timestamp] = [open_, high, low, close, volume]
        df = df.tail(EMA_SCAN_LIMIT)
        symbol_dfs[symbol] = df

    # Аналіз сигналу при закритті свічки
    if candle_closed and len(df) >= 20:
        df_features = apply_all_features(df)
        action, votes, pretop, last, confidence = detect_signal(df_features)
        prev_signal = state["signals"].get(symbol, "")
        if pretop or (action != "WATCH" and confidence >= CONF_THRESHOLD_MEDIUM and action != prev_signal):
            msg_text = f"⚡ Signal {symbol}\nAction: {action}\nPrice: {last['close']:.6f}\nConfidence: {confidence:.2f}\nPretop: {pretop}"
            send_telegram(msg_text)
            state["signals"][symbol] = action
            state["last_update"] = str(datetime.now(timezone.utc))
            save_json_safe(STATE_FILE, state)

def on_error(ws, error):
    logger.error("WebSocket error: %s", error)

def on_close(ws, close_status_code, close_msg):
    logger.warning("WebSocket closed, reconnecting in 5s...")
    time.sleep(5)
    start_ws(list(symbol_dfs.keys()))

def on_open(ws):
    logger.info("WebSocket connection opened")

def start_ws(symbols):
    if not symbols:
        return
    streams = "/".join([f"{s.lower()}@kline_1m" for s in symbols])
    url = f"wss://stream.binance.com:9443/stream?streams={streams}"
    ws = websocket.WebSocketApp(url, on_open=on_open, on_message=on_message, on_error=on_error, on_close=on_close)
    ws.run_forever(ping_interval=20, ping_timeout=10)

# ---------------- MASTER START ----------------
def start_bot():
    symbols = get_all_usdt_symbols()
    with lock:
        for s in symbols:
            if s not in symbol_dfs:
                symbol_dfs[s] = pd.DataFrame(columns=["open","high","low","close","volume"])
    Thread(target=start_ws, args=(symbols,), daemon=True).start()

if __name__ == "__main__":
    logger.info("Starting pre-top detector bot (WebSocket)")
    start_bot()
    while True:
        time.sleep(1)