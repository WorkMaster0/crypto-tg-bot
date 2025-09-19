# main.py — Повний покращений Pre-top бот з графіками та Telegram (свічкові графіки)
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
import requests
from flask import Flask, request, jsonify
from apscheduler.schedulers.background import BackgroundScheduler
import ta
import mplfinance as mpf
from scipy.signal import find_peaks
import numpy as np

# ---------------- BINANCE CLIENT ----------------
try:
    from binance.client import Client as BinanceClient
    BINANCE_PY_AVAILABLE = True
except Exception:
    BINANCE_PY_AVAILABLE = False

# ---------------- CONFIG ----------------
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
CHAT_ID = os.getenv("CHAT_ID", "")
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "")
PORT = int(os.getenv("PORT", "5000"))

TOP_LIMIT = int(os.getenv("TOP_LIMIT", "100"))
SCAN_INTERVAL_MINUTES = int(os.getenv("SCAN_INTERVAL_MINUTES", "1"))
EMA_SCAN_LIMIT = int(os.getenv("EMA_SCAN_LIMIT", "500"))
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

# ---------------- BINANCE ----------------
if BINANCE_PY_AVAILABLE and BINANCE_API_KEY and BINANCE_API_SECRET:
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
            files = {'photo': photo}
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
    if not client:
        return []
    try:
        ex = client.get_exchange_info()
        symbols = [
            s["symbol"] for s in ex["symbols"]
            if s["quoteAsset"] == "USDT" and s["status"] == "TRADING"
        ]
        blacklist = [
            "BUSD", "USDC", "FDUSD", "TUSD", "DAI", "EUR", "GBP", "AUD",
            "BTCST", "COIN", "AAPL", "TSLA", "MSFT", "META", "GOOG", "USD1"
        ]
        filtered = [s for s in symbols if not any(b in s for b in blacklist)]
        return filtered
    except Exception as e:
        logger.exception("get_all_usdt_symbols error: %s", e)
        return []

def fetch_klines(symbol, interval="15m", limit=EMA_SCAN_LIMIT):
    for attempt in range(3):
        try:
            if not client:
                raise RuntimeError("Binance client unavailable")
            kl = client.get_klines(symbol=symbol, interval=interval, limit=limit)
            df = pd.DataFrame(kl, columns=["open_time", "open", "high", "low", "close", "volume",
                                           "close_time", "qav", "num_trades", "tb_base", "tb_quote", "ignore"])
            df = df[["open_time", "open", "high", "low", "close", "volume"]].astype(float)
            df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
            df.set_index("open_time", inplace=True)
            return df
        except Exception as e:
            logger.warning("fetch_klines %s attempt %d error: %s", symbol, attempt + 1, e)
            time.sleep(0.5)
    return None

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

# ---------------- НОВИЙ plot_signal_candles з розумними рівнями ----------------
def plot_signal_candles(df, symbol, action, votes, pretop, n_levels=5):
    df_plot = df.copy()[['open','high','low','close','volume']]
    df_plot.index.name = "Date"

    closes = df['close'].values

    # Локальні максимуми → опори
    peaks, _ = find_peaks(closes, distance=5)
    peak_vals = closes[peaks]
    top_resistances = sorted(peak_vals, reverse=True)[:n_levels]

    # Локальні мінімуми → підтримки
    troughs, _ = find_peaks(-closes, distance=5)
    trough_vals = closes[troughs]
    top_supports = sorted(trough_vals)[:n_levels]

    hlines = list(top_supports) + list(top_resistances)

    addplots = []

    # Підсвітка Pre-top
    if pretop:
        addplots.append(
            mpf.make_addplot([df['close'].iloc[-i] for i in range(3,0,-1)]*1,
                             type='scatter', markersize=120, marker='^', color='magenta')
        )

    # Підсвітка патернів
    last = df.iloc[-1]
    patterns = {
        "bullish_engulfing": "green",
        "bearish_engulfing": "red",
        "hammer_bull": "lime",
        "shooting_star": "orange",
        "doji": "blue"
    }
    for pat, color in patterns.items():
        if pat in votes:
            addplots.append(
                mpf.make_addplot([last['close']]*len(df), type='scatter', markersize=80, marker='o', color=color)
            )

    mc = mpf.make_marketcolors(up='green', down='red', wick='black', edge='black', volume='blue')
    s = mpf.make_mpf_style(marketcolors=mc, gridstyle='--', gridcolor='gray', facecolor='white')

    buf = io.BytesIO()
    mpf.plot(
        df_plot,
        type='candle',
        style=s,
        volume=True,
        addplot=addplots,
        hlines=dict(hlines=hlines, colors=['gray'], linestyle='dashed'),
        title=f"{symbol} — {action} — {','.join([v for v in votes])}",
        ylabel='Price',
        ylabel_lower='Volume',
        savefig=dict(fname=buf, dpi=100, bbox_inches='tight')
    )
    buf.seek(0)
    return buf

# ---------------- ANALYZE SYMBOL ----------------
def analyze_and_alert(symbol: str):
    df = fetch_klines(symbol)
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
        photo_buf = plot_signal_candles(df, symbol, action, votes, pretop)
        send_telegram(msg, photo=photo_buf)
        state["signals"][symbol] = action
        save_json_safe(STATE_FILE, state)

# ---------------- MASTER SCAN ----------------
def scan_top_symbols():
    symbols = get_all_usdt_symbols()
    if not symbols:
        logger.warning("No symbols found for scanning.")
        return

    logger.info("Starting scan for %d symbols", len(symbols))
    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as exe:
        list(exe.map(analyze_and_alert, symbols))

    state["last_scan"] = str(datetime.now(timezone.utc))
    save_json_safe(STATE_FILE, state)
    logger.info("Scan finished at %s", state["last_scan"])

# ---------------- SCHEDULER ----------------
scheduler = BackgroundScheduler()
scheduler.add_job(scan_top_symbols, "interval", minutes=max(1, SCAN_INTERVAL_MINUTES),
                  id="scan_job", next_run_time=datetime.now())
scheduler.start()
logger.info("Scheduler started with interval %d minutes", SCAN_INTERVAL_MINUTES)

# ---------------- FLASK ROUTES ----------------
@app.route("/")
def home():
    return jsonify({"status": "ok", "time": str(datetime.now(timezone.utc)), "signals": len(state.get("signals", {}))})

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
            elif text.startswith("/top"):
                pretop_tokens = [s for s in state.get("signals", {})]
                if pretop_tokens:
                    send_telegram("Pre-top tokens:\n" + "\n".join(pretop_tokens))
                else:
                    send_telegram("No pre-top tokens detected yet.")
    return jsonify({"ok": True})

# ---------------- AUTO REGISTER WEBHOOK ----------------
def auto_register_webhook():
    if WEBHOOK_URL and TELEGRAM_TOKEN:
        logger.info("Registering Telegram webhook: %s", WEBHOOK_URL)
        set_telegram_webhook(WEBHOOK_URL)

Thread(target=auto_register_webhook, daemon=True).start()

# ---------------- WARMUP ----------------
def warmup_and_first_scan():
    try:
        scan_top_symbols()
    except Exception as e:
        logger.exception("warmup_and_first_scan error: %s", e)

Thread(target=warmup_and_first_scan, daemon=True).start()

# ---------------- MAIN ----------------
if __name__ == "__main__":
    logger.info("Starting pre-top detector bot")
    app.run(host="0.0.0.0", port=PORT)