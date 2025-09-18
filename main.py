# main.py — Повний бот з WebSocket, Flask вебхуком та покращеним confidence

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
from flask import Flask, request, jsonify

from binance.client import Client as BinanceClient
from binance.streams import ThreadedWebsocketManager
import ta

# ---------------- CONFIG ----------------
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
CHAT_ID = os.getenv("CHAT_ID", "")
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "")
PORT = int(os.getenv("PORT", "5000"))

STATE_FILE = "state.json"
LOG_FILE = "bot.log"

CONF_THRESHOLD_MEDIUM = 0.3
CONF_THRESHOLD_STRONG = 0.6

PARALLEL_WORKERS = 6

# ---------------- LOGGING ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)
logger = logging.getLogger("pretop-bot")

# ---------------- BINANCE ----------------
if BINANCE_API_KEY and BINANCE_API_SECRET:
    client = BinanceClient(api_key=BINANCE_API_KEY, api_secret=BINANCE_API_SECRET)
else:
    client = None
    logger.warning("Binance client unavailable or API keys missing")

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

# ---------------- FEATURE ENGINEERING ----------------
def apply_all_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    try:
        df["ret1"] = df["close"].pct_change(1)
        df["ret5"] = df["close"].pct_change(5)

        df["ema_8"] = ta.trend.EMAIndicator(df["close"], 8).ema_indicator()
        df["ema_20"] = ta.trend.EMAIndicator(df["close"], 20).ema_indicator()
        df["ema_50"] = ta.trend.EMAIndicator(df["close"], 50).ema_indicator()
        df["sma_50"] = df["close"].rolling(50).mean()

        df["ATR_14"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], 14).average_true_range()

        df["RSI_14"] = ta.momentum.RSIIndicator(df["close"], 14).rsi()
        stoch = ta.momentum.StochRSIIndicator(df["close"], 14, 3, 3)
        df["stoch_k"] = stoch.stochrsi_k()
        df["stoch_d"] = stoch.stochrsi_d()

        macd = ta.trend.MACD(df["close"])
        df["MACD"] = macd.macd()
        df["MACD_signal"] = macd.macd_signal()
        df["MACD_hist"] = macd.macd_diff()

        adx = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], 14)
        df["ADX"] = adx.adx()
        df["ADX_pos"] = adx.adx_pos()
        df["ADX_neg"] = adx.adx_neg()

        bb = ta.volatility.BollingerBands(df["close"], 20, 2)
        df["bb_h"] = bb.bollinger_hband()
        df["bb_l"] = bb.bollinger_lband()
        df["bb_width"] = (df["bb_h"] - df["bb_l"]) / df["bb_l"].replace(0, np.nan)

        pv = df["close"] * df["volume"]
        df["vwap_50"] = (pv.rolling(50).sum() / df["volume"].rolling(50).sum()).replace([np.inf, -np.inf], np.nan)
        df["OBV"] = ta.volume.OnBalanceVolumeIndicator(df["close"], df["volume"]).on_balance_volume()
        df["CCI"] = ta.trend.CCIIndicator(df["high"], df["low"], df["close"], 20).cci()

        ha_close = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
        ha_open_vals = [(df["open"].iloc[0] + df["close"].iloc[0]) / 2]
        for i in range(1, len(df)):
            ha_open_vals.append((ha_open_vals[-1] + ha_close.iloc[i - 1]) / 2)
        df["ha_close"] = ha_close
        df["ha_open"] = ha_open_vals
        df["ha_body"] = df["ha_close"] - df["ha_open"]
        df["ha_dir"] = np.sign(df["ha_body"])

        df["support"] = df["low"].rolling(20).min()
        df["resistance"] = df["high"].rolling(20).max()
    except Exception as e:
        logger.exception("apply_all_features error: %s", e)
    return df

# ---------------- SIGNAL DETECTION ----------------
def detect_signal(df: pd.DataFrame):
    last = df.iloc[-1]
    votes = []

    # Pattern votes
    if last["ema_8"] > last["ema_20"]:
        votes.append("ema_bull")
    else:
        votes.append("ema_bear")

    if last["MACD_hist"] > 0:
        votes.append("macd_up")
    else:
        votes.append("macd_down")

    if last["RSI_14"] > 70:
        votes.append("rsi_overbought")
    elif last["RSI_14"] < 30:
        votes.append("rsi_oversold")

    if last["stoch_k"] > 80:
        votes.append("stoch_overbought")
    elif last["stoch_k"] < 20:
        votes.append("stoch_oversold")

    if last["ADX"] > 25:
        votes.append("adx_up" if last["ADX_pos"] > last["ADX_neg"] else "adx_down")

    if df["ha_dir"].iloc[-5:-1].gt(0).all() and last["ha_dir"] < 0:
        votes.append("ha_exhaustion")
    if df["ha_dir"].iloc[-5:-1].lt(0).all() and last["ha_dir"] > 0:
        votes.append("ha_exhaustion_buy")

    pretop = (last["close"] - df["close"].iloc[-10]) / df["close"].iloc[-10] > 0.1 if len(df) >= 10 else False

    # Action
    action = "WATCH"
    if last["close"] >= last["resistance"] * 0.995:
        action = "SHORT"
    elif last["close"] <= last["support"] * 1.005:
        action = "LONG"

    # Confidence (покращено)
    conf = 0.05 + 0.05 * len(votes) + (0.15 if pretop else 0)
    conf = min(conf, 1.0)

    return action, votes, pretop, last, conf

# ---------------- ANALYZE SYMBOL ----------------
def analyze_symbol(symbol, df):
    df = apply_all_features(df)
    action, votes, pretop, last, confidence = detect_signal(df)

    prev_signal = state["signals"].get(symbol, "")

    # Pre-top повідомлення
    if pretop:
        send_telegram(f"⚡ Pre-top detected for {symbol}, price={last['close']:.6f}")

    # Основний сигнал
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

# ---------------- WEBSOCKET HANDLER ----------------
def handle_ws_message(msg):
    symbol = msg["s"].upper()
    k = msg["k"]
    if not k["x"]:
        return  # тільки закриті свічки
    df = pd.DataFrame([{
        "open_time": k["t"],
        "open": float(k["o"]),
        "high": float(k["h"]),
        "low": float(k["l"]),
        "close": float(k["c"]),
        "volume": float(k["v"])
    }])
    analyze_symbol(symbol, df)

def start_ws():
    if not client:
        logger.warning("Binance client unavailable for WS")
        return
    twm = ThreadedWebsocketManager(api_key=BINANCE_API_KEY, api_secret=BINANCE_API_SECRET)
    twm.start()
    symbols = [s["symbol"] for s in client.get_exchange_info()["symbols"] if s["quoteAsset"] == "USDT" and s["status"] == "TRADING"]
    for sym in symbols:
        twm.start_kline_socket(sym.lower(), handle_ws_message, interval='1m')

# ---------------- FLASK ROUTES ----------------
@app.route("/")
def home():
    return jsonify({"status": "ok", "time": str(datetime.now(timezone.utc)), "state_signals_count": len(state.get("signals", {}))})

@app.route(f"/telegram_webhook/<token>", methods=["POST"])
def telegram_webhook(token):
    if token != TELEGRAM_TOKEN:
        return jsonify({"ok": False, "reason": "invalid token"}), 403
    update = request.get_json(force=True)
    if "message" in update:
        msg = update["message"]
        text = msg.get("text", "").lower()
        if text.startswith("/scan"):
            Thread(target=start_manual_scan, daemon=True).start()
            send_telegram("Manual scan started.")
        elif text.startswith("/status"):
            send_telegram(f"Status:\nSignals={len(state.get('signals', {}))}\nLast scan={state.get('last_scan')}")
        elif text.startswith("/top"):
            pretop_tokens = [s for s in state.get("signals", {})]
            if pretop_tokens:
                send_telegram("Pre-top tokens:\n" + "\n".join(pretop_tokens))
            else:
                send_telegram("No pre-top tokens detected yet.")
    return jsonify({"ok": True})

# ---------------- MANUAL SCAN ----------------
def start_manual_scan():
    if not client:
        return
    symbols = [s["symbol"] for s in client.get_exchange_info()["symbols"] if s["quoteAsset"] == "USDT" and s["status"] == "TRADING"]
    for sym in symbols:
        kl = client.get_klines(symbol=sym, interval="1m", limit=50)
        df = pd.DataFrame(kl, columns=["open_time", "open","high","low","close","volume","close_time","qav","num_trades","tb_base","tb_quote","ignore"])
        df = df[["open_time","open","high","low","close","volume"]].astype(float)
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df.set_index("open_time", inplace=True)
        analyze_symbol(sym, df)
    state["last_scan"] = str(datetime.now(timezone.utc))
    save_json_safe(STATE_FILE, state)

# ---------------- WEBHOOK REGISTER ----------------
def auto_register_webhook():
    if WEBHOOK_URL and TELEGRAM_TOKEN:
        logger.info("Registering Telegram webhook: %s", WEBHOOK_URL)
        set_telegram_webhook(WEBHOOK_URL)

# ---------------- STARTUP ----------------
if __name__ == "__main__":
    Thread(target=auto_register_webhook, daemon=True).start()
    Thread(target=start_ws, daemon=True).start()
    logger.info("Pre-top detector bot started")
    app.run(host="0.0.0.0", port=PORT)