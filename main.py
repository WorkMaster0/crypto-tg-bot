# main.py — Pre-top бот з графіками, Telegram і backtest winrate
import os
import time
import json
import logging
import re
from datetime import datetime, timezone
from threading import Thread
import io

import pandas as pd
import matplotlib.pyplot as plt
import requests
import ta
import mplfinance as mpf
from scipy.signal import find_peaks
import numpy as np
from flask import Flask, request, jsonify

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
from binance import ThreadedWebsocketManager
from binance.client import Client as BinanceClient

client = None
try:
    client = BinanceClient(api_key=BINANCE_API_KEY, api_secret=BINANCE_API_SECRET)
except Exception as e:
    logger.warning("Binance client unavailable: %s", e)

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

# ---------------- MARKET DATA ----------------
symbol_data = {}  # Зберігає DataFrame для всіх токенів

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
            "STRAX", "GNS", "ALCX", "BTCST", "COIN", "AAPL", "TSLA", 
            "MSFT", "META", "GOOG", "USD1", "BTTC", "ARDR", "DF", "XNO"
        ]
        filtered = [s for s in symbols if not any(b in s for b in blacklist)]
        return filtered[:TOP_LIMIT]
    except Exception as e:
        logger.exception("get_all_usdt_symbols error: %s", e)
        return []

# ---------------- DATA CACHE ----------------
def warmup_data():
    symbols = get_all_usdt_symbols()
    logger.info("Warming up data for %d symbols", len(symbols))
    for sym in symbols:
        kl = client.get_klines(symbol=sym, interval="15m", limit=EMA_SCAN_LIMIT)
        df = pd.DataFrame(kl, columns=["open_time","open","high","low","close","volume",
                                       "close_time","qav","num_trades","tb_base","tb_quote","ignore"])
        df = df[["open_time","open","high","low","close","volume"]].astype(float)
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df.set_index("open_time", inplace=True)
        symbol_data[sym] = df

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

# ---------------- ANALYZE SYMBOL ----------------
def analyze_and_alert(symbol: str):
    df = symbol_data.get(symbol)
    if df is None or len(df) < 30:
        return
    df = apply_all_features(df)
    action, votes, pretop, last, confidence = detect_signal(df)
    prev_signal = state["signals"].get(symbol, "")

    logger.info("Symbol=%s action=%s confidence=%.2f votes=%s pretop=%s",
                symbol, action, confidence, [v for v in votes], pretop)

    # Зберігаємо історію сигналів для /history
    hist = state["signal_history"].setdefault(symbol, [])
    hist.append({"time": str(last.name), "action": action, "price": last["close"]})
    hist = hist[-50:]
    state["signal_history"][symbol] = hist

    if pretop:
        Thread(target=send_telegram, args=(f"⚡ Pre-top detected for {symbol}, price={last['close']:.6f}",)).start()

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
        Thread(target=send_telegram, args=(msg, photo_buf)).start()
        state["signals"][symbol] = action

    save_json_safe(STATE_FILE, state)

# ---------------- PLOT SIGNAL CANDLES ----------------
def plot_signal_candles(df, symbol, action, votes, pretop, n_levels=5):
    df_plot = df.copy()[['open','high','low','close','volume']]
    df_plot.index.name = "Date"

    closes = df['close'].values
    peaks, _ = find_peaks(closes, distance=5)
    peak_vals = closes[peaks]
    top_resistances = sorted(peak_vals, reverse=True)[:n_levels]

    troughs, _ = find_peaks(-closes, distance=5)
    trough_vals = sorted(closes[troughs])[:n_levels]

    hlines = list(trough_vals) + list(top_resistances)
    addplots = []

    last = df.iloc[-1]

    # Pre-top highlight
    if pretop:
        ydata = [np.nan]*(len(df)-3) + list(df['close'].iloc[-3:])
        addplots.append(
            mpf.make_addplot(ydata, type='scatter', markersize=120, marker='^', color='magenta')
        )

    # Previous signals LONG/SHORT
    hist = state.get("signal_history", {}).get(symbol, [])
    for h in hist:
        if h["action"] in ["LONG", "SHORT"]:
            ts = pd.to_datetime(h["time"])
            if ts in df.index:
                idx = df.index.get_loc(ts)
                y = [np.nan] * len(df)
                y[idx] = h["price"]
                color = "green" if h["action"] == "LONG" else "red"
                addplots.append(
                    mpf.make_addplot(y, type="scatter", markersize=60, marker="o", color=color)
                )

    mc = mpf.make_marketcolors(up='green', down='red', wick='black', edge='black', volume='blue')
    s = mpf.make_mpf_style(marketcolors=mc, gridstyle='--', gridcolor='gray', facecolor='white')

    buf = io.BytesIO()
    mpf.plot(
        df_plot, type='candle', style=s, volume=True, addplot=addplots,
        hlines=dict(hlines=hlines, colors=['gray'], linestyle='dashed'),
        title=f"{symbol} — {action}", ylabel='Price', ylabel_lower='Volume',
        savefig=dict(fname=buf, dpi=100, bbox_inches='tight')
    )
    buf.seek(0)
    return buf

# ---------------- TELEGRAM WEBHOOK ----------------
@app.route("/telegram_webhook/<token>", methods=["POST"])
def telegram_webhook(token):
    try:
        if token != TELEGRAM_TOKEN:
            return jsonify({"ok": False, "error": "invalid token"}), 403

        update = request.get_json(force=True) or {}
        msg = update.get("message")
        if not msg:
            return jsonify({"ok": True})

        text = msg.get("text", "").lower().strip()

        if text.startswith("/scan"):
            for sym in symbol_data.keys():
                Thread(target=analyze_and_alert, args=(sym,)).start()
            send_telegram("⚡ Manual scan started.")

        elif text.startswith("/status"):
            send_telegram(f"📝 Status:\nSignals={len(state.get('signals', {}))}\nLast scan={state.get('last_scan')}")

        elif text.startswith("/history"):
            parts = text.split()
            if len(parts) >= 2:
                symbol = parts[1].upper()
                df = symbol_data.get(symbol)
                if df is not None and len(df) >= 30:
                    buf = plot_signal_candles(df, symbol, "", [], False)
                    send_telegram(f"📈 History for {symbol}", photo=buf)
                else:
                    send_telegram(f"❌ No data for {symbol}")

    except Exception as e:
        logger.exception("telegram_webhook error: %s", e)
    return jsonify({"ok": True})

# ---------------- TELEGRAM WEBHOOK SETUP ----------------
def setup_webhook():
    if not TELEGRAM_TOKEN or not WEBHOOK_URL:
        logger.error("❌ TELEGRAM_TOKEN or WEBHOOK_URL is missing!")
        return
    try:
        base_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"
        requests.get(f"{base_url}/deleteWebhook")
        webhook_url = f"{WEBHOOK_URL}/telegram_webhook/{TELEGRAM_TOKEN}"
        requests.get(f"{base_url}/setWebhook?url={webhook_url}")
        requests.get(f"{base_url}/getWebhookInfo")
    except Exception as e:
        logger.exception("Webhook setup error: %s", e)

# ---------------- WEBSOCKET SCANNER ----------------
def start_websocket():
    if not client:
        logger.warning("Binance client unavailable, websocket not started")
        return

    twm = ThreadedWebsocketManager(api_key=BINANCE_API_KEY, api_secret=BINANCE_API_SECRET)
    twm.start()

    def handle_kline(msg):
        symbol = msg['s']
        k = msg['k']
        if k['x']:  # закрита свічка
            df_new = pd.DataFrame([{
                'open_time': pd.to_datetime(k['t'], unit='ms', utc=True),
                'open': float(k['o']),
                'high': float(k['h']),
                'low': float(k['l']),
                'close': float(k['c']),
                'volume': float(k['v'])
            }])
            # оновлюємо кеш
            if symbol in symbol_data:
                symbol_data[symbol] = pd.concat([symbol_data[symbol], df_new]).tail(EMA_SCAN_LIMIT)
            else:
                symbol_data[symbol] = df_new
            analyze_and_alert(symbol)

    symbols = get_all_usdt_symbols()
    for sym in symbols:
        twm.start_kline_socket(callback=handle_kline, symbol=sym.lower(), interval='15m')

# ---------------- MAIN ----------------
if __name__ == "__main__":
    logger.info("Starting pre-top detector bot")
    setup_webhook()
    warmup_data()
    start_websocket()
    app.run(host="0.0.0.0", port=PORT)