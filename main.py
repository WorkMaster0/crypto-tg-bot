# main.py — Pre-top бот з графіками, Telegram і backtest winrate (WebSocket-версія)
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

TOP_LIMIT = int(os.getenv("TOP_LIMIT", "100"))  # рекомендую 30-100 для початку
SCAN_INTERVAL_MINUTES = int(os.getenv("SCAN_INTERVAL_MINUTES", "5"))  # автоскан інтервал (мінімум 1)
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

# ---------------- BINANCE CLIENT (для warmup та метаданих) ----------------
try:
    from binance import ThreadedWebsocketManager
    from binance.client import Client as BinanceClient
    client = BinanceClient(api_key=BINANCE_API_KEY, api_secret=BINANCE_API_SECRET) if BINANCE_API_KEY and BINANCE_API_SECRET else None
except Exception as e:
    logger.exception("Binance library unavailable: %s", e)
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
twm = None        # ThreadedWebsocketManager instance (global, щоб його не зібрало GC)

# ---------------- HELPERS: SYMBOLS ----------------
def get_all_usdt_symbols():
    """
    Повертає обмежений список USDT-символів (TOP_LIMIT) з blacklist-фільтром.
    """
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
        if not filtered:
            logger.warning("No filtered symbols after blacklist (check exchange_info output)")
        # лог для діагностики (перші 20)
        logger.info("Symbols sample: %s", filtered[:20])
        return filtered[:TOP_LIMIT]
    except Exception as e:
        logger.exception("get_all_usdt_symbols error: %s", e)
        return []

# ---------------- SAFE FETCH (для warmup лише) ----------------
def fetch_klines_rest(symbol, interval="15m", limit=EMA_SCAN_LIMIT):
    """
    Лімітований REST fetch - використовується тільки для warmup.
    Має retry + delay щоб мінімізувати ризик бану.
    """
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
            time.sleep(0.15)  # невелика пауза між REST запитами
            return df
        except Exception as e:
            logger.warning("fetch_klines_rest %s attempt %d error: %s", symbol, attempt+1, e)
            time.sleep(0.5)
    return None

# ---------------- DATA WARMUP ----------------
def warmup_data():
    """
    Разове наповнення symbol_data через REST (обмежено TOP_LIMIT).
    Якщо Binance повертає помилку — просто лог і продовжує.
    """
    symbols = get_all_usdt_symbols()
    logger.info("Warming up data for %d symbols", len(symbols))
    for sym in symbols:
        try:
            df = fetch_klines_rest(sym)
            if df is not None and len(df) > 0:
                symbol_data[sym] = df
            else:
                logger.debug("No warmup data for %s", sym)
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
    try:
        df = symbol_data.get(symbol)
        if df is None or len(df) < 30:
            logger.debug("analyze_and_alert: no data for %s", symbol)
            return
        df_feat = apply_all_features(df)
        action, votes, pretop, last, confidence = detect_signal(df_feat)
        prev_signal = state["signals"].get(symbol, "")

        logger.info("Symbol=%s action=%s confidence=%.2f votes=%s pretop=%s",
                    symbol, action, confidence, [v for v in votes], pretop)

        # Save history
        hist = state["signal_history"].setdefault(symbol, [])
        hist.append({"time": str(last.name), "action": action, "price": float(last["close"])})
        hist = hist[-200:]  # keep limited history
        state["signal_history"][symbol] = hist

        if pretop:
            Thread(target=send_telegram, args=(f"⚡ Pre-top detected for {symbol}, price={last['close']:.6f}",), daemon=True).start()

        if action != "WATCH" and confidence >= CONF_THRESHOLD_MEDIUM and action != prev_signal:
            # update state before sending (to avoid duplicates)
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

            # build plot in background to avoid blocking WS thread
            def send_with_plot():
                try:
                    photo_buf = plot_signal_candles(df, symbol, action, votes, pretop)
                    send_telegram(msg, photo=photo_buf)
                except Exception as e:
                    logger.exception("send_with_plot error: %s", e)

            Thread(target=send_with_plot, daemon=True).start()

        else:
            # save state even if no new signal (history updated)
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

    # Pre-top highlight
    if pretop:
        ydata = [np.nan]*(len(df)-3) + list(df['close'].iloc[-3:])
        addplots.append(
            mpf.make_addplot(ydata, type='scatter', markersize=120, marker='^', color='magenta')
        )

    # Previous signals LONG/SHORT (by time)
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
            # пропускаємо точки, які вже не в індексі
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

# ---------------- WEBSOCKET (kline) ----------------
def start_websocket():
    global twm
    if not client:
        logger.warning("Binance client unavailable, websocket not started")
        return

    try:
        twm = ThreadedWebsocketManager(api_key=BINANCE_API_KEY, api_secret=BINANCE_API_SECRET)
        twm.start()
    except Exception as e:
        logger.exception("Failed to start ThreadedWebsocketManager: %s", e)
        return

    symbols = get_all_usdt_symbols()
    if not symbols:
        logger.warning("No symbols found for WebSocket subscription.")
        return

    logger.info("Subscribing to WebSocket for %d symbols (TOP_LIMIT=%d)", len(symbols), TOP_LIMIT)

    # build per-symbol callback to preserve symbol within closure
    def make_callback(sym):
        def _callback(msg):
            try:
                k = msg.get('k', {})
                if not k:
                    return
                # process only closed candles
                if k.get('x'):
                    df_new = pd.DataFrame([{
                        'open_time': pd.to_datetime(k['t'], unit='ms', utc=True),
                        'open': float(k['o']),
                        'high': float(k['h']),
                        'low': float(k['l']),
                        'close': float(k['c']),
                        'volume': float(k['v'])
                    }]).set_index('open_time')
                    # merge into cache
                    df_old = symbol_data.get(sym)
                    if df_old is not None:
                        combined = pd.concat([df_old, df_new])
                        combined = combined[~combined.index.duplicated(keep="last")].tail(EMA_SCAN_LIMIT)
                    else:
                        combined = df_new
                    symbol_data[sym] = combined
                    # analyze in background to avoid blocking ws thread
                    Thread(target=analyze_and_alert, args=(sym,), daemon=True).start()
            except Exception as e:
                logger.exception("WebSocket callback error for %s: %s", sym, e)
        return _callback

    for sym in symbols:
        try:
            cb = make_callback(sym)
            # start_kline_socket expects lowercase symbol name (e.g. btcusdt)
            twm.start_kline_socket(callback=cb, symbol=sym.lower(), interval='15m')
            time.sleep(0.05)  # slight pause between subscriptions
        except Exception as e:
            logger.exception("Failed to start kline socket for %s: %s", sym, e)

    logger.info("WebSocket subscriptions started")

# ---------------- SIMPLE SCANNER LOOP (fallback periodic re-check) ----------------
def scanner_loop():
    while True:
        try:
            logger.info("🔄 Auto-scan (re-analyze cache) started")
            # run analyze on cached symbols (no REST calls)
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
        time.sleep(max(30, SCAN_INTERVAL_MINUTES * 60))  # at least 30s

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
            logger.warning("Received webhook with invalid token: %s", token)
            return jsonify({"ok": False, "error": "invalid token"}), 403

        update = request.get_json(force=True) or {}
        msg = update.get("message")
        if not msg:
            return jsonify({"ok": True})

        text = msg.get("text", "").lower().strip()

        if text.startswith("/scan"):
            # analyze all cached symbols in background
            for sym in list(symbol_data.keys()):
                Thread(target=analyze_and_alert, args=(sym,), daemon=True).start()
            send_telegram("⚡ Manual scan started.")

        elif text.startswith("/status"):
            send_telegram(f"📝 Status:\nSignals={len(state.get('signals', {}))}\nLast scan={state.get('last_scan')}\nCached symbols={len(symbol_data)}")

        elif text.startswith("/top"):
            # quick top by current confidence (local, approximate)
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
                msg_text = "🏆 Top5 tokens by confidence:\n" + "\n".join([f"{s[0]}: {s[1]*100:.1f}%" for s in top5])
            else:
                msg_text = "❌ No data to compute top."
            send_telegram(msg_text)

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
        requests.get(f"{base_url}/deleteWebhook", timeout=10)
        webhook_url = f"{WEBHOOK_URL}/telegram_webhook/{TELEGRAM_TOKEN}"
        requests.get(f"{base_url}/setWebhook?url={webhook_url}", timeout=10)
        requests.get(f"{base_url}/getWebhookInfo", timeout=10)
        logger.info("Webhook setup attempted: %s", webhook_url)
    except Exception as e:
        logger.exception("Webhook setup error: %s", e)


# ---------------- START BACKGROUND TASKS ----------------
def start_background_tasks():
    Thread(target=warmup_data, daemon=True).start()
    Thread(target=start_websocket, daemon=True).start()
    Thread(target=scanner_loop, daemon=True).start()


# ---------------- INIT ON IMPORT ----------------
# Це виконається при старті gunicorn
logger.info("Starting pre-top detector bot (import phase)")
setup_webhook()
start_background_tasks()


# ---------------- MAIN (для локального запуску) ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, threaded=True)