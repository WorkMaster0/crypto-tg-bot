# main.py
import os
import time
import json
import logging
import re
from datetime import datetime, timezone
from threading import Thread
from concurrent.futures import ThreadPoolExecutor
import io

import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
from flask import Flask, request, jsonify
from scipy.signal import find_peaks
import ta
import requests

try:
    from binance.client import Client as BinanceClient
    BINANCE_PY_AVAILABLE = True
except Exception:
    BINANCE_PY_AVAILABLE = False

# ---------------- CONFIG ----------------
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
PORT = int(os.getenv("PORT", "5000"))
WEBHOOK_URL = os.getenv("WEBHOOK_URL")
PARALLEL_WORKERS = int(os.getenv("PARALLEL_WORKERS", "6"))
EMA_SCAN_LIMIT = int(os.getenv("EMA_SCAN_LIMIT", "500"))
CONF_THRESHOLD_MEDIUM = 0.60
STATE_FILE = "state.json"

# ---------------- LOGGING ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("pretop-bot")

# ---------------- STATE ----------------
def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {"signals": {}, "last_scan": None}

def save_state():
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2, default=str)
    except Exception as e:
        logger.exception("save_state error: %s", e)

state = load_state()

# ---------------- BINANCE ----------------
client = None
if BINANCE_PY_AVAILABLE and BINANCE_API_KEY and BINANCE_API_SECRET:
    client = BinanceClient(api_key=BINANCE_API_KEY, api_secret=BINANCE_API_SECRET)
    logger.info("✅ Binance client initialized")
else:
    logger.warning("Binance client unavailable or keys missing")

# ---------------- TELEGRAM ----------------
MARKDOWNV2_ESCAPE = r"_*[]()~`>#+-=|{}.!"

def escape_md_v2(text: str) -> str:
    return re.sub(f"([{re.escape(MARKDOWNV2_ESCAPE)}])", r"\\\1", str(text))

def send_telegram(text: str, photo=None):
    if not TELEGRAM_TOKEN:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"
        if photo:
            files = {'photo': ('signal.png', photo, 'image/png')}
            data = {'chat_id': os.getenv("CHAT_ID"), 'caption': escape_md_v2(text), 'parse_mode': 'MarkdownV2'}
            requests.post(f"{url}/sendPhoto", data=data, files=files, timeout=10)
        else:
            payload = {"chat_id": os.getenv("CHAT_ID"), "text": escape_md_v2(text), "parse_mode":"MarkdownV2"}
            requests.post(f"{url}/sendMessage", json=payload, timeout=10)
    except Exception as e:
        logger.exception("send_telegram error: %s", e)

# ---------------- MARKET DATA ----------------
def get_all_usdt_symbols():
    if not client:
        return []
    try:
        info = client.get_exchange_info()
        symbols = [s["symbol"] for s in info["symbols"] if s["quoteAsset"]=="USDT" and s["status"]=="TRADING"]
        return symbols
    except Exception as e:
        logger.exception("get_all_usdt_symbols error: %s", e)
        return []

def fetch_klines(symbol, interval="15m", limit=EMA_SCAN_LIMIT):
    for _ in range(3):
        try:
            if not client:
                return None
            kl = client.get_klines(symbol=symbol, interval=interval, limit=limit)
            df = pd.DataFrame(kl, columns=["open_time","open","high","low","close","volume","close_time","qav","num_trades","tb_base","tb_quote","ignore"])
            df = df[["open_time","open","high","low","close","volume"]].astype(float)
            df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
            df.set_index("open_time", inplace=True)
            return df
        except Exception as e:
            logger.warning("fetch_klines %s error: %s", symbol, e)
            time.sleep(1)
    return None

# ---------------- FEATURES ----------------
def apply_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema_8"] = ta.trend.EMAIndicator(df["close"], 8).ema_indicator()
    df["ema_20"] = ta.trend.EMAIndicator(df["close"], 20).ema_indicator()
    df["RSI_14"] = ta.momentum.RSIIndicator(df["close"], 14).rsi()
    macd = ta.trend.MACD(df["close"])
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()
    df["MACD_hist"] = macd.macd_diff()
    adx = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], 14)
    df["ADX"] = adx.adx()
    df["support"] = df["low"].rolling(20).min()
    df["resistance"] = df["high"].rolling(20).max()
    return df

# ---------------- SIGNAL ----------------
def detect_signal(df: pd.DataFrame):
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last
    votes = []
    confidence = 0.2

    # EMA
    if last["ema_8"] > last["ema_20"]:
        votes.append("ema_bull")
        confidence += 0.1
    else:
        votes.append("ema_bear")
        confidence += 0.05

    # MACD
    if last["MACD_hist"] > 0:
        votes.append("macd_up")
        confidence += 0.1
    else:
        votes.append("macd_down")
        confidence += 0.05

    # RSI
    if last["RSI_14"] < 30:
        votes.append("rsi_oversold")
        confidence += 0.08
    elif last["RSI_14"] > 70:
        votes.append("rsi_overbought")
        confidence += 0.08

    # ADX
    if last.get("ADX",0) > 25:
        votes.append("strong_trend")
        confidence *= 1.1

    # Свічкові патерни
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

    # Pre-top
    pretop = False
    if len(df) >= 10 and (last["close"] - df["close"].iloc[-10])/df["close"].iloc[-10] > 0.1:
        pretop = True
        confidence += 0.1
        votes.append("pretop")

    # Action
    action = "WATCH"
    if last["close"] >= last["resistance"] * 0.995:
        action = "SHORT"
    elif last["close"] <= last["support"] * 1.005:
        action = "LONG"

    # ✅ НЕ обрізаємо confidence до 1, а можна відображати як %
    # confidence = max(0, min(1, confidence))  # <- видалити або закоментувати

    return action, votes, pretop, last, confidence

# ---------------- PLOT ----------------
def plot_signal_candles(df, symbol, action, votes, pretop, n_levels=5):
    df_plot = df[['open','high','low','close','volume']].copy()
    df_plot.index.name = "Date"
    closes = df['close'].values
    peaks, _ = find_peaks(closes, distance=5)
    troughs, _ = find_peaks(-closes, distance=5)
    hlines = list(sorted(closes[troughs])[:n_levels]) + list(sorted(closes[peaks], reverse=True)[:n_levels])
    addplots = []
    last = df.iloc[-1]
    if pretop:
        ydata = [np.nan]*(len(df)-3) + list(df['close'].iloc[-3:])
        addplots.append(mpf.make_addplot(ydata, type='scatter', markersize=120, marker='^', color='magenta'))
    patterns = {"bullish_engulfing":"green","bearish_engulfing":"red","hammer_bull":"lime","shooting_star":"orange","doji":"blue"}
    for pat, color in patterns.items():
        if pat in votes:
            ydata = [np.nan]*(len(df)-1) + [last['close']]
            addplots.append(mpf.make_addplot(ydata, type='scatter', markersize=80, marker='o', color=color))
    mc = mpf.make_marketcolors(up='green', down='red', wick='black', edge='black', volume='blue')
    s = mpf.make_mpf_style(marketcolors=mc, gridstyle='--', gridcolor='gray', facecolor='white')
    buf = io.BytesIO()
    mpf.plot(df_plot, type='candle', style=s, volume=True,
             addplot=addplots, hlines=dict(hlines=hlines, colors=['gray'], linestyle='dashed'),
             title=f"{symbol} — {action} — {','.join(votes)}",
             savefig=dict(fname=buf, dpi=100, bbox_inches='tight'))
    buf.seek(0)
    return buf

# ---------------- ANALYZE ----------------
def analyze_and_alert(symbol: str):
    df = fetch_klines(symbol)
    if df is None or len(df)<30:
        return
    df = apply_features(df)
    action, votes, pretop, last, confidence = detect_signal(df)
    prev_signal = state["signals"].get(symbol,"")
    logger.info("Symbol=%s action=%s confidence=%.2f votes=%s pretop=%s", symbol, action, confidence, votes, pretop)
    if action != "WATCH" and confidence>=CONF_THRESHOLD_MEDIUM and action != prev_signal:
        msg = f"⚡ {symbol} — {action}\nPrice={last['close']:.6f}\nSupport={last['support']:.6f}\nResistance={last['resistance']:.6f}\nConfidence={confidence:.2f}\nPatterns={','.join(votes)}\nPre-top={pretop}"
        photo_buf = plot_signal_candles(df,symbol,action,votes,pretop)
        send_telegram(msg, photo=photo_buf)
        state["signals"][symbol] = action
        save_state()

def scan_top_symbols():
    symbols = get_all_usdt_symbols()
    if not symbols: return
    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as exe:
        list(exe.map(analyze_and_alert,symbols))
    state["last_scan"] = str(datetime.now(timezone.utc))
    save_state()

# ---------------- FLASK ----------------
app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({"status":"ok","time":str(datetime.now(timezone.utc)),"signals":len(state.get("signals",{}))})

# Підтримуємо і /telegram_webhook, і /telegram_webhook/<anything...>
@app.route("/telegram_webhook", defaults={"path": ""}, methods=["POST"])
@app.route("/telegram_webhook/<path:path>", methods=["POST"])
def telegram_webhook(path):
    try:
        # path може бути '', 'TOKEN', або 'TOKEN/telegram_webhook/TOKEN' і т.д.
        token_in_path = ""
        if path:
            token_in_path = path.strip("/").split("/")[-1]

        # Якщо в URL токен не знайдено або не співпадає — відмовляємо
        if token_in_path != TELEGRAM_TOKEN:
            logger.warning("Invalid token in request path: %s (expected %s)", token_in_path, TELEGRAM_TOKEN)
            return jsonify({"ok": False, "error": "invalid token"}), 403

        update = request.get_json(force=True) or {}
        # лог корисних полів для дебагу
        logger.debug("Telegram update keys: %s", list(update.keys()))
        msg = update.get("message")
        if not msg:
            return jsonify({"ok": True})

        text = msg.get("text", "").strip().lower()
        # приклад обробки команд (залиши свою логіку)
        if text.startswith("/scan"):
            Thread(target=scan_top_symbols, daemon=True).start()
            send_telegram("⚡ Manual scan started.")
        elif text.startswith("/status"):
            send_telegram(f"📝 Status: signals={len(state.get('signals',{}))}, last_scan={state.get('last_scan')}")
        # ... інші команди як у тебе ...
    except Exception as e:
        logger.exception("telegram_webhook error: %s", e)

    return jsonify({"ok": True})

# ---------------- WEBHOOK ----------------
def setup_webhook():
    if not TELEGRAM_TOKEN or not WEBHOOK_URL:
        logger.error("❌ TELEGRAM_TOKEN or WEBHOOK_URL is missing! WEBHOOK_URL must be the base domain (no path).")
        return

    base = WEBHOOK_URL.rstrip("/")

    # Якщо користувач випадково вказав вже повний шлях або токен — використаємо як є
    if base.endswith(f"/telegram_webhook/{TELEGRAM_TOKEN}"):
        final_url = base
    elif base.endswith("/telegram_webhook"):
        final_url = f"{base}/{TELEGRAM_TOKEN}"
    elif "/telegram_webhook/" in base:
        # має /telegram_webhook/<something> — вважаймо, що там вже правильний шлях
        final_url = base
    else:
        final_url = f"{base}/telegram_webhook/{TELEGRAM_TOKEN}"

    api_base = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

    try:
        resp = requests.get(f"{api_base}/setWebhook", params={"url": final_url}, timeout=10)
        info = requests.get(f"{api_base}/getWebhookInfo", timeout=10).json()
        logger.info("Webhook set to: %s", final_url)
        logger.info("setWebhook resp: %s", resp.text if resp is not None else "None")
        logger.info("getWebhookInfo: %s", info)
    except Exception as e:
        logger.exception("Webhook setup error: %s", e)

# ---------------- MAIN ----------------
if __name__=="__main__":
    logger.info("Starting bot...")
    setup_webhook()
    Thread(target=scan_top_symbols, daemon=True).start()
    app.run(host="0.0.0.0", port=PORT)