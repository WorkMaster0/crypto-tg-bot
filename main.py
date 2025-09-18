# main_pro_ws.py — WebSocket Pre-Top Detector Bot з патернами та реальною confidence системою

import os
import json
import time
import logging
import re
from datetime import datetime, timezone, timedelta
from threading import Thread

import pandas as pd
import numpy as np
import requests
import ta
import websocket
import rel  # для стійкого reconnect

from binance.client import Client as BinanceClient

# ---------------- CONFIG ----------------
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
CHAT_ID = os.getenv("CHAT_ID", "")

STATE_FILE = "state_ws.json"
LOG_FILE = "bot_ws.log"

CANDLE_INTERVAL = "1m"  # робимо сигнал максимально чутливим
LOOKBACK = 200          # зберігаємо останні 200 свічок
COOLDOWN = 300          # мінімум 5 хв між сигналами на один символ

# ---------------- LOGGING ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)
logger = logging.getLogger("ws-bot")

# ---------------- BINANCE ----------------
client = BinanceClient(api_key=BINANCE_API_KEY, api_secret=BINANCE_API_SECRET)

# ---------------- STATE ----------------
def load_json_safe(path, default):
    try:
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
    except Exception as e:
        logger.exception("load_json_safe error %s: %s", e)
    return default

def save_json_safe(path, data):
    try:
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp, path)
    except Exception as e:
        logger.exception("save_json_safe error %s: %s", e)

state = load_json_safe(STATE_FILE, {"last_signal_time": {}})

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

# ---------------- FEATURE ENGINEERING ----------------
def apply_all_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    try:
        # Returns
        df["ret1"] = df["close"].pct_change(1)
        # EMA
        df["ema_8"] = ta.trend.EMAIndicator(df["close"], 8).ema_indicator()
        df["ema_20"] = ta.trend.EMAIndicator(df["close"], 20).ema_indicator()
        # ATR
        df["ATR_14"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], 14).average_true_range()
        # RSI
        df["RSI_14"] = ta.momentum.RSIIndicator(df["close"], 14).rsi()
        # MACD
        macd = ta.trend.MACD(df["close"])
        df["MACD"] = macd.macd()
        df["MACD_hist"] = macd.macd_diff()
        # ADX
        adx = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], 14)
        df["ADX"] = adx.adx()
        df["ADX_pos"] = adx.adx_pos()
        df["ADX_neg"] = adx.adx_neg()
        # Heikin Ashi
        ha_close = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
        ha_open_vals = [(df["open"].iloc[0] + df["close"].iloc[0]) / 2]
        for i in range(1, len(df)):
            ha_open_vals.append((ha_open_vals[-1] + ha_close.iloc[i - 1]) / 2)
        df["ha_close"] = ha_close
        df["ha_open"] = ha_open_vals
        df["ha_dir"] = np.sign(df["ha_close"] - df["ha_open"])
        # Support/Resistance
        df["support"] = df["low"].rolling(20).min()
        df["resistance"] = df["high"].rolling(20).max()
    except Exception as e:
        logger.exception("apply_all_features error: %s", e)
    return df

# ---------------- SIGNAL DETECTION ----------------
def detect_signal(df: pd.DataFrame):
    last = df.iloc[-1]
    votes = []
    score = 0.0

    # Технічні
    if last["ema_8"] > last["ema_20"]:
        votes.append("ema_bull"); score += 0.1
    if last["MACD_hist"] > 0:
        votes.append("macd_up"); score += 0.1
    if last["ADX"] > 25:
        votes.append("adx_strong"); score += 0.15

    # Ринкові
    if last["volume"] > df["volume"].rolling(20).mean().iloc[-1] * 2:
        votes.append("volume_spike"); score += 0.2
    if last["close"] >= last["resistance"] * 0.995:
        votes.append("near_resistance"); score += 0.1
    if last["close"] <= last["support"] * 1.005:
        votes.append("near_support"); score += 0.1

    # Патерни
    if df["ha_dir"].iloc[-5:-1].gt(0).all() and last["ha_dir"] < 0:
        votes.append("bearish_reversal"); score += 0.15
    if df["ha_dir"].iloc[-5:-1].lt(0).all() and last["ha_dir"] > 0:
        votes.append("bullish_reversal"); score += 0.15

    # Pre-top spike
    pretop = False
    if len(df) >= 10:
        recent, last10 = df["close"].iloc[-1], df["close"].iloc[-10]
        if (recent - last10) / last10 > 0.1:
            pretop = True
            votes.append("pretop_spike")
            score += 0.2

    confidence = min(1.0, score)

    action = "WATCH"
    if confidence > 0.65:
        action = "LONG" if "bullish_reversal" in votes or "ema_bull" in votes else "SHORT"

    return action, votes, pretop, last, confidence

# ---------------- PROCESS TICK ----------------
buffers = {}  # symbol -> DataFrame

def process_candle(symbol, kline):
    try:
        open_t = datetime.fromtimestamp(kline["t"]/1000, tz=timezone.utc)
        new_row = {
            "open": float(kline["o"]),
            "high": float(kline["h"]),
            "low": float(kline["l"]),
            "close": float(kline["c"]),
            "volume": float(kline["v"])
        }

        if symbol not in buffers:
            buffers[symbol] = pd.DataFrame([new_row], index=[open_t])
        else:
            df = buffers[symbol]
            df.loc[open_t] = new_row
            df = df.sort_index().iloc[-LOOKBACK:]
            buffers[symbol] = df

        if kline["x"]:  # closed candle
            df = apply_all_features(buffers[symbol])
            if len(df) > 30:
                action, votes, pretop, last, confidence = detect_signal(df)

                last_time = state["last_signal_time"].get(symbol)
                now = datetime.now(timezone.utc)

                if (pretop or action != "WATCH") and confidence > 0.4:
                    if not last_time or (now - datetime.fromisoformat(last_time)) > timedelta(seconds=COOLDOWN):
                        msg = (
                            f"⚡ Signal for {symbol}\n"
                            f"Action: {action}\n"
                            f"Price: {last['close']:.6f}\n"
                            f"Confidence: {confidence:.2f}\n"
                            f"Patterns: {','.join(votes)}\n"
                            f"Time: {now.isoformat()}"
                        )
                        send_telegram(msg)
                        state["last_signal_time"][symbol] = now.isoformat()
                        save_json_safe(STATE_FILE, state)

    except Exception as e:
        logger.exception("process_candle error %s: %s", symbol, e)

# ---------------- WEBSOCKET ----------------
def start_ws():
    symbols = [s["symbol"].lower() for s in client.get_exchange_info()["symbols"] if s["quoteAsset"] == "USDT" and s["status"] == "TRADING"]
    streams = [f"{s}@kline_{CANDLE_INTERVAL}" for s in symbols]
    url = f"wss://stream.binance.com:9443/stream?streams={'/'.join(streams)}"

    def on_message(ws, message):
        try:
            data = json.loads(message)
            kline = data["data"]["k"]
            symbol = kline["s"]
            process_candle(symbol, kline)
        except Exception as e:
            logger.exception("on_message error: %s", e)

    def on_error(ws, error):
        logger.error("WebSocket error: %s", error)

    def on_close(ws, *_):
        logger.warning("WebSocket closed, reconnecting...")

    def on_open(ws):
        logger.info("WebSocket opened and listening for %d symbols", len(symbols))

    ws = websocket.WebSocketApp(url, on_message=on_message, on_error=on_error, on_close=on_close, on_open=on_open)
    ws.run_forever(dispatcher=rel, reconnect=5)
    rel.signal(2, rel.abort)
    rel.dispatch()

# ---------------- MAIN ----------------
if __name__ == "__main__":
    logger.info("Starting WebSocket Pre-Top Bot")
    Thread(target=start_ws, daemon=True).start()
    while True:
        time.sleep(60)