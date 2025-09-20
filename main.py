# main.py — Pre-top бот з графіками, Telegram і backtest (WebSocket-версія)
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
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "").rstrip("/")
PORT = int(os.getenv("PORT", "5000"))

TOP_LIMIT = int(os.getenv("TOP_LIMIT", "30"))
SCAN_INTERVAL_MINUTES = int(os.getenv("SCAN_INTERVAL_MINUTES", "5"))
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

# ---------------- BINANCE CLIENT ----------------
try:
    from binance import ThreadedWebsocketManager
    from binance.client import Client as BinanceClient
    from binance.exceptions import BinanceAPIException
    client = None
    if BINANCE_API_KEY and BINANCE_API_SECRET:
        try:
            client = BinanceClient(api_key=BINANCE_API_KEY, api_secret=BINANCE_API_SECRET)
            logger.info("✅ Binance client initialized")
        except BinanceAPIException as e:
            logger.warning("Binance client init failed: %s", e)
except Exception as e:
    logger.warning("Binance library unavailable: %s", e)
    client = None

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
        logger.debug("Telegram token/chat missing — skipping send")
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

# ---------------- MARKET DATA CACHE ----------------
symbol_data = {}  # { "BTCUSDT": DataFrame(...) }
twm = None        # ThreadedWebsocketManager instance

# ---------------- HELPERS: SYMBOLS ----------------
def get_all_usdt_symbols():
    if not client:
        logger.warning("Binance client missing — cannot get symbols")
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
        logger.info("Symbols sample: %s", filtered[:20])
        return filtered[:TOP_LIMIT]
    except Exception as e:
        logger.exception("get_all_usdt_symbols error: %s", e)
        return []

# ---------------- SAFE FETCH (для історичних даних) ----------------
def fetch_klines_rest(symbol, interval="15m", limit=EMA_SCAN_LIMIT):
    if not client:
        return None
    for attempt in range(3):
        try:
            kl = client.get_klines(symbol=symbol, interval=interval, limit=limit)
            df = pd.DataFrame(kl, columns=["open_time","open","high","low","close","volume",
                                           "close_time","qav","num_trades","tb_base","tb_quote","ignore"])
            df = df[["open_time","open","high","low","close","volume"]].astype(float)
            df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
            df.set_index("open_time", inplace=True)
            time.sleep(0.15)
            return df
        except Exception as e:
            logger.warning("fetch_klines_rest %s attempt %d error: %s", symbol, attempt+1, e)
            time.sleep(0.5)
    return None

# ---------------- DATA WARMUP ----------------
def warmup_data():
    symbols = get_all_usdt_symbols()
    logger.info("Warming up data for %d symbols", len(symbols))
    for sym in symbols:
        try:
            df = None
            for attempt in range(3):
                df = fetch_klines_rest(sym)
                if df is not None:
                    break
                time.sleep(1)  # пауза між запитами
            if df is not None and len(df) > 0:
                symbol_data[sym] = df
            else:
                logger.warning("No data fetched for %s", sym)
            time.sleep(0.3)  # **обов'язкова пауза** між запитами, щоб не перевищити ліміт
        except Exception as e:
            logger.exception("warmup_data error for %s: %s", sym, e)

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

    body = last["close"] - last["open"]
    rng = last["high"] - last["low"]
    upper_shadow = last["high"] - max(last["close"], last["open"])
    lower_shadow = min(last["close"], last["open"]) - last["low"]

    if lower_shadow > 2 * abs(body) and body > 0:
        votes.append("hammer_bull")
    elif upper_shadow > 2 * abs(body) and body < 0:
        votes.append("shooting_star")

    if body > 0 and prev["close"] < prev["open"] and last["close"] > prev["open"] and last["open"] < prev["close"]:
        votes.append("bullish_engulfing")
    elif body < 0 and prev["close"] > prev["open"] and last["close"] < prev["open"] and last["open"] > prev["close"]:
        votes.append("bearish_engulfing")

    if abs(body) < 0.1 * rng:
        votes.append("doji")

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

# ---------------- ANALYZE SYMBOL ----------------
def analyze_and_alert(symbol: str):
    try:
        df = symbol_data.get(symbol)
        if df is None or len(df) < 30:
            return
        df_feat = apply_all_features(df)
        action, votes, pretop, last, confidence = detect_signal(df_feat)
        prev_signal = state["signals"].get(symbol, "")

        hist = state["signal_history"].setdefault(symbol, [])
        hist.append({"time": str(last.name), "action": action, "price": float(last["close"])})
        hist = hist[-200:]
        state["signal_history"][symbol] = hist

        if pretop:
            Thread(target=send_telegram, args=(f"⚡ Pre-top detected for {symbol}, price={last['close']:.6f}",), daemon=True).start()

        if action != "WATCH" and confidence >= CONF_THRESHOLD_MEDIUM and action != prev_signal:
            state["signals"][symbol] = action
            save_json_safe(STATE_FILE, state)

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

            def send_with_plot():
                try:
                    photo_buf = plot_signal_candles(df, symbol, action, votes, pretop)
                    send_telegram(msg, photo=photo_buf)
                except Exception as e:
                    logger.exception("send_with_plot error: %s", e)

            Thread(target=send_with_plot, daemon=True).start()
        else:
            save_json_safe(STATE_FILE, state)
    except Exception as e:
        logger.exception("analyze_and_alert error for %s: %s", symbol, e)

# ---------------- PLOT SIGNAL CANDLES ----------------
def plot_signal_candles(df, symbol, action, votes, pretop, n_levels=5):
    df_plot = df.copy()[['open','high','low','close','volume']]
    df_plot.index.name = "Date"

    closes = df['close'].values
    peaks, _ = find_peaks(closes, distance=5)
    peak_vals = closes[peaks] if len(peaks)>0 else np.array([])
    top_resistances = sorted(list(peak_vals), reverse=True)[:n_levels] if len(peak_vals)>0 else []

    troughs, _ = find_peaks(-closes, distance=5)
    trough_vals = closes[troughs] if len(troughs)>0 else np.array([])
    top_supports = sorted(list(trough_vals))[:n_levels] if len(trough_vals)>0 else []

    hlines = list(top_supports) + list(top_resistances)
    addplots = []

    last = df.iloc[-1]

    if pretop:
        ydata = [np.nan]*(len(df)-3) + list(df['close'].iloc[-3:])
        addplots.append(
            mpf.make_addplot(ydata, type='scatter', markersize=120, marker='^', color='magenta')
        )

    hist = state.get("signal_history", {}).get(symbol, [])
    for h in hist:
        try:
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
        except Exception:
            continue

    mc = mpf.make_marketcolors(up='green', down='red', wick='black', edge='black', volume='blue')
    s = mpf.make_mpf_style(marketcolors=mc, gridstyle='--', gridcolor='gray', facecolor='white')

    buf = io.BytesIO()
    mpf.plot(
        df_plot, type='candle', style=s, volume=True, addplot=addplots,
        hlines=dict(hlines=hlines, colors=['gray'], linestyle='dashed') if hlines else None,
        title=f"{symbol} — {action}", ylabel='Price', ylabel_lower='Volume',
        savefig=dict(fname=buf, dpi=100, bbox_inches='tight')
    )
    buf.seek(0)
    return buf

# ---------------- WEBSOCKET ----------------
def start_websocket():
    if not BINANCE_API_KEY or not BINANCE_API_SECRET:
        logger.warning("Binance API keys missing — websocket not started")
        return

    def handle_kline(msg):
        symbol = msg['s']
        k = msg['k']
        if k['x']:
            df_new = pd.DataFrame([{
                'open_time': pd.to_datetime(k['t'], unit='ms', utc=True),
                'open': float(k['o']),
                'high': float(k['h']),
                'low': float(k['l']),
                'close': float(k['c']),
                'volume': float(k['v'])
            }])
            if symbol in symbol_data:
                symbol_data[symbol] = pd.concat([symbol_data[symbol], df_new]).tail(EMA_SCAN_LIMIT)
            else:
                symbol_data[symbol] = df_new
            analyze_and_alert(symbol)

    while True:
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            global twm
            twm = ThreadedWebsocketManager(api_key=BINANCE_API_KEY, api_secret=BINANCE_API_SECRET)
            twm.start()

            symbols = get_all_usdt_symbols()
            if not symbols:
                logger.warning("No symbols fetched — retrying in 60 seconds")
                time.sleep(60)
                continue

            for sym in symbols:
                twm.start_kline_socket(callback=handle_kline, symbol=sym.lower(), interval='15m')
                logger.info(f"✅ Started websocket for {sym}")

            loop.run_forever()
        except Exception as e:
            logger.exception("Websocket error, restarting in 60s: %s", e)
            time.sleep(60)

# ---------------- SCANNER LOOP ----------------
def scanner_loop():
    while True:
        try:
            logger.info("🔄 Auto-scan started")
            for sym in list(symbol_data.keys()):
                try:
                    analyze_and_alert(sym)
                except Exception as e:
                    logger.exception("scanner_loop analyze error for %s: %s", sym, e)
            state["last_scan"] = str(datetime.now(timezone.utc))
            save_json_safe(STATE_FILE, state)
            logger.info("✅ Auto-scan finished")
        except Exception as e:
            logger.exception("Scanner loop error: %s", e)
        time.sleep(max(30, SCAN_INTERVAL_MINUTES * 60))

# ---------------- FLASK ROUTES ----------------
@app.route("/")
def home():
    return jsonify({
        "status": "ok",
        "time": str(datetime.now(timezone.utc)),
        "signals": len(state.get("signals", {})),
        "cached_symbols": len(symbol_data)
    })

@app.route("/telegram_webhook/<token>", methods=["POST"])
def telegram_webhook(token):
    try:
        if token != TELEGRAM_TOKEN:
            return jsonify({"ok": False, "error": "invalid token"}), 403

        update = request.get_json(force=True) or {}
        msg = update.get("message")
        if not msg:
            return jsonify({"ok": True})

        text = msg.get("text", "").strip()
        if not text:
            return jsonify({"ok": True})

        # Відділяємо команду від username та аргументів
        parts = text.split()
        cmd = parts[0].split("@")[0].lower()  # /scan, /status, /top, /history

        if cmd == "/scan":
            for sym in list(symbol_data.keys()):
                Thread(target=analyze_and_alert, args=(sym,), daemon=True).start()
            send_telegram("⚡ Manual scan started.")

        elif cmd == "/status":
            send_telegram(
                f"📝 Status:\n"
                f"Signals={len(state.get('signals', {}))}\n"
                f"Last scan={state.get('last_scan')}\n"
                f"Cached symbols={len(symbol_data)}"
            )

        elif cmd == "/top":
            scores = {}
            for sym, df in symbol_data.items():
                try:
                    if df is None or len(df) < 30:
                        continue
                    df_feat = apply_all_features(df)
                    _, _, _, _, conf = detect_signal(df_feat)
                    scores[sym] = conf
                except Exception:
                    continue
            top5 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
            if top5:
                msg_text = "🏆 Top5 tokens by confidence:\n" + "\n".join(
                    [f"{s[0]}: {s[1]*100:.1f}%" for s in top5]
                )
            else:
                msg_text = "❌ No data to compute top."
            send_telegram(msg_text)

        elif cmd == "/history":
            if len(parts) >= 2:
                symbol = parts[1].upper()
                df = symbol_data.get(symbol)
                if df is not None and len(df) >= 30:
                    buf = plot_signal_candles(df, symbol, "", [], False)
                    send_telegram(f"📈 History for {symbol}", photo=buf)
                else:
                    send_telegram(f"❌ No data for {symbol}")
            else:
                send_telegram("❌ Usage: /history SYMBOL")

        else:
            send_telegram("❌ Unknown command. Available: /scan, /status, /top, /history")

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
        requests.get(f"{base_url}/deleteWebhook", timeout=10)

        # Важливо: не додаємо "/telegram_webhook" у WEBHOOK_URL
        webhook_url = f"{WEBHOOK_URL}/telegram_webhook/{TELEGRAM_TOKEN}"
        requests.get(f"{base_url}/setWebhook?url={webhook_url}", timeout=10)

        info = requests.get(f"{base_url}/getWebhookInfo", timeout=10).json()
        logger.info("Webhook set: %s", webhook_url)
        logger.info("Webhook info: %s", info)
    except Exception as e:
        logger.exception("Webhook setup error: %s", e)

# ---------------- BACKGROUND TASKS ----------------
def start_background_tasks():
    Thread(target=warmup_data, daemon=True).start()
    Thread(target=start_websocket, daemon=True).start()
    Thread(target=scanner_loop, daemon=True).start()

# ---------------- INIT ----------------
logger.info("Starting pre-top detector bot")
setup_webhook()
start_background_tasks()

# ---------------- MAIN ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, threaded=True)