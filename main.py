import os
import time
import json
import logging
import re
from datetime import datetime, timezone
from threading import Thread
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import matplotlib.pyplot as plt
import requests
import ta
import mplfinance as mpf
import numpy as np
import io

from binance.client import Client

# ---------------- LOGGING ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler("bot.log"), logging.StreamHandler()]
)
logger = logging.getLogger("pretop-bot")

# ---------------- CONFIG ----------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
CHAT_ID = os.getenv("CHAT_ID", "")
PORT = int(os.getenv("PORT", "5000"))
PARALLEL_WORKERS = int(os.getenv("PARALLEL_WORKERS", "6"))
EMA_SCAN_LIMIT = 500
STATE_FILE = "state.json"
CONF_THRESHOLD_MEDIUM = 0.3

# ---------------- BINANCE CLIENT ----------------
binance_client = Client(api_key="", api_secret="")

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

state = load_json_safe(STATE_FILE, {"signals": {}, "last_scan": None})

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
            payload = {"chat_id": CHAT_ID, "text": escape_md_v2(text), "parse_mode": "MarkdownV2"}
            requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", json=payload, timeout=10)
    except Exception as e:
        logger.exception("send_telegram error: %s", e)

# ---------------- WEBSOCKET / REST MANAGER ----------------
from websocket_manager import WebSocketKlineManager

ALL_USDT = [
    "BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT","ADAUSDT","DOGEUSDT",
    "DOTUSDT","TRXUSDT","LTCUSDT","AVAXUSDT","SHIBUSDT","LINKUSDT","ATOMUSDT","XMRUSDT",
    "ETCUSDT","XLMUSDT","APTUSDT","NEARUSDT","FILUSDT","ICPUSDT","GRTUSDT","AAVEUSDT",
    "SANDUSDT","AXSUSDT","FTMUSDT","THETAUSDT","EGLDUSDT","MANAUSDT","FLOWUSDT","HBARUSDT",
    "ALGOUSDT","ZECUSDT","EOSUSDT","KSMUSDT","CELOUSDT","SUSHIUSDT","CHZUSDT","KAVAUSDT",
    "ZILUSDT","ANKRUSDT","RAYUSDT","GMTUSDT","UNIUSDT","APEUSDT","PEPEUSDT","OPUSDT",
    "XTZUSDT","ALPHAUSDT","BALUSDT","COMPUSDT","CRVUSDT","SNXUSDT","RSRUSDT",
    "LOKUSDT","GALUSDT","WLDUSDT","JASMYUSDT","ONEUSDT","ARBUSDT","ALICEUSDT","XECUSDT",
    "FLMUSDT","CAKEUSDT","IMXUSDT","HOOKUSDT","MAGICUSDT","STGUSDT","FETUSDT",
    "PEOPLEUSDT","ASTRUSDT","ENSUSDT","CTSIUSDT","GALAUSDT","RADUSDT","IOSTUSDT","QTUMUSDT",
    "NPXSUSDT","DASHUSDT","ZRXUSDT","HNTUSDT","ENJUSDT","TFUELUSDT","TWTUSDT",
    "NKNUSDT","GLMRUSDT","ZENUSDT","STORJUSDT","ICXUSDT","XVGUSDT","FLOKIUSDT","BONEUSDT",
    "TRBUSDT","C98USDT","MASKUSDT","1000SHIBUSDT","1000PEPEUSDT","AMBUSDT","VEGUSDT","QNTUSDT",
    "RNDRUSDT","CHRUSDT","API3USDT","MTLUSDT","ALPUSDT","LDOUSDT","AXLUSDT","FUNUSDT",
    "OGUSDT","ORCUSDT","XAUTUSDT","ARUSDT","DYDXUSDT","RUNEUSDT","FLUXUSDT",
    "AGLDUSDT","PERPUSDT","MLNUSDT","NMRUSDT","LRCUSDT","COTIUSDT","ACHUSDT",
    "CKBUSDT","ACEUSDT","TRUUSDT","IPSUSDT","QIUSDT","GLMUSDT","ARNXUSDT",
    "MIRUSDT","ROSEUSDT","OXTUSDT","SPELLUSDT","SUNUSDT","SYSUSDT","TAOUSDT",
    "TLMUSDT","VLXUSDT","WAXPUSDT","XNOUSDT"
]

BINANCE_REST_URL = "https://fapi.binance.com/fapi/v1/klines"

def fetch_klines_rest(symbol, interval="15m", limit=500):
    try:
        resp = requests.get(BINANCE_REST_URL, params={"symbol": symbol, "interval": interval, "limit": limit}, timeout=5)
        data = resp.json()
        df = pd.DataFrame(data, columns=[
            "open_time","open","high","low","close","volume",
            "close_time","quote_asset_volume","trades",
            "taker_buy_base","taker_buy_quote","ignore"
        ])
        for col in ["open","high","low","close","volume"]:
            df[col] = df[col].astype(float)
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df.set_index("open_time", inplace=True)
        return df
    except Exception as e:
        logger.exception("REST fetch error for %s: %s", symbol, e)
        return None

ws_manager = WebSocketKlineManager(symbols=ALL_USDT, interval="15m")
Thread(target=ws_manager.start, daemon=True).start()

def fetch_klines(symbol, limit=500):
    df = ws_manager.get_klines(symbol, limit)
    if df is None or len(df) < 10:
        df = fetch_klines_rest(symbol, limit=limit)
    return df

# ---------------- PATTERN-BASED FEATURE ENGINEERING ----------------
def apply_all_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Support/Resistance (динамічні рівні за останні 20 свічок)
    df["support"] = df["low"].rolling(20).min()
    df["resistance"] = df["high"].rolling(20).max()

    # Volume analysis
    df["vol_ma20"] = df["volume"].rolling(20).mean()
    df["vol_spike"] = df["volume"] > 1.5 * df["vol_ma20"]

    # Candle structure
    df["body"] = df["close"] - df["open"]
    df["range"] = df["high"] - df["low"]
    df["upper_shadow"] = df["high"] - df[["close", "open"]].max(axis=1)
    df["lower_shadow"] = df[["close", "open"]].min(axis=1) - df["low"]

    return df

def get_symbol_market_cap(symbol: str) -> float:
    """
    Отримує капіталізацію токена з CoinGecko.
    Якщо немає даних -> повертає 0.
    """
    try:
        # Прибираємо "USDT" і зводимо до нижнього регістру
        base = symbol.replace("USDT", "").lower()

        url = f"https://api.coingecko.com/api/v3/coins/{base}"
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return 0.0

        data = resp.json()
        return data.get("market_data", {}).get("market_cap", {}).get("usd", 0.0)
    except Exception as e:
        logger.warning("Cannot fetch market cap for %s: %s", symbol, e)
        return 0.0

# ---------------- PATTERN-BASED SIGNAL DETECTION ----------------
def analyze_and_alert(symbol: str):
    try:
        # --- Перевірка капіталізації ---
        market_cap = get_symbol_market_cap(symbol)
        if market_cap < 5_000_000:
            logger.info("Skipping %s due to low market cap: %.2f", symbol, market_cap)
            return

        # --- Завантажуємо дані ---
        df = fetch_klines(symbol, interval="15m", limit=200)
        if df is None or len(df) < 50:
            logger.info("Symbol=%s: Not enough data", symbol)
            return

        # --- Базові метрики ---
        df["ATR"] = ta.atr(df["high"], df["low"], df["close"], length=14)
        atr = df["ATR"].iloc[-1]

        close = df["close"].iloc[-1]
        prev_close = df["close"].iloc[-2]

        # --- Визначаємо патерни ---
        patterns = []

        if ta.cdl_hammer(df["open"], df["high"], df["low"], df["close"]).iloc[-1] != 0:
            patterns.append("🔨 Hammer")

        if ta.cdl_shootingstar(df["open"], df["high"], df["low"], df["close"]).iloc[-1] != 0:
            patterns.append("🌠 Shooting Star")

        if ta.cdl_doji(df["open"], df["high"], df["low"], df["close"]).iloc[-1] != 0:
            patterns.append("✒ Doji")

        if ta.cdl_engulfing(df["open"], df["high"], df["low"], df["close"]).iloc[-1] > 0:
            patterns.append("🟢 Bullish Engulfing")

        if ta.cdl_engulfing(df["open"], df["high"], df["low"], df["close"]).iloc[-1] < 0:
            patterns.append("🔴 Bearish Engulfing")

        if ta.cdl_morningstar(df["open"], df["high"], df["low"], df["close"]).iloc[-1] != 0:
            patterns.append("🌅 Morning Star")

        if ta.cdl_eveningstar(df["open"], df["high"], df["low"], df["close"]).iloc[-1] != 0:
            patterns.append("🌆 Evening Star")

        if ta.cdl_hangingman(df["open"], df["high"], df["low"], df["close"]).iloc[-1] != 0:
            patterns.append("🪓 Hanging Man")

        if ta.cdl_harami(df["open"], df["high"], df["low"], df["close"]).iloc[-1] > 0:
            patterns.append("🟢 Bullish Harami")

        if ta.cdl_harami(df["open"], df["high"], df["low"], df["close"]).iloc[-1] < 0:
            patterns.append("🔴 Bearish Harami")

        # Якщо немає патернів — вихід
        if not patterns:
            return

        # --- Визначаємо дію ---
        if "Bullish" in " ".join(patterns) or "Hammer" in " ".join(patterns) or "Morning" in " ".join(patterns):
            action = "BUY"
        elif "Bearish" in " ".join(patterns) or "Shooting" in " ".join(patterns) or "Evening" in " ".join(patterns):
            action = "SELL"
        else:
            action = "WATCH"

        if action == "WATCH":
            return

        # --- Розрахунок рівнів ---
        stop_loss = close - atr if action == "BUY" else close + atr
        tp1 = close + 2 * atr if action == "BUY" else close - 2 * atr
        tp2 = close + 3 * atr if action == "BUY" else close - 3 * atr
        tp3 = close + 5 * atr if action == "BUY" else close - 5 * atr

        # --- Фільтр RR ---
        rr1 = abs((tp1 - close) / (close - stop_loss))
        if rr1 < 2:
            logger.info("Skipping %s due to low RR1=%.2f", symbol, rr1)
            return

        # --- Confidence ---
        confidence = 50 + len(patterns) * 10
        if action == "BUY" and close > prev_close:
            confidence += 10
        if action == "SELL" and close < prev_close:
            confidence += 10
        confidence = min(confidence, 95)

        # --- Повідомлення ---
        msg = (
            f"📊 {symbol}\n"
            f"💡 Action: {('🟢 BUY' if action == 'BUY' else '🔴 SELL')}\n"
            f"📈 Confidence: {confidence:.1f}%\n"
            f"🪙 Market Cap: {market_cap:,.0f} $\n"
            f"🎯 Targets:\n"
            f"   1️⃣ {tp1:.4f}\n"
            f"   2️⃣ {tp2:.4f}\n"
            f"   3️⃣ {tp3:.4f}\n"
            f"🛑 Stop: {stop_loss:.4f}\n"
            f"📌 Patterns: {', '.join(patterns)}"
        )

        send_telegram_message(msg)
        logger.info("Signal sent for %s (conf=%.1f, rr1=%.2f)", symbol, confidence, rr1)

    except Exception as e:
        logger.exception("Error in analyze_and_alert(%s): %s", symbol, e)

# ---------------- PLOT UTILITY ----------------
def plot_signal_candles(df, symbol, action, tp1=None, tp2=None, tp3=None, sl=None, entry=None):
    addplots = []
    if tp1: addplots.append(mpf.make_addplot([tp1]*len(df), color='green', linestyle="--"))
    if tp2: addplots.append(mpf.make_addplot([tp2]*len(df), color='lime', linestyle="--"))
    if tp3: addplots.append(mpf.make_addplot([tp3]*len(df), color='darkgreen', linestyle="--"))
    if sl: addplots.append(mpf.make_addplot([sl]*len(df), color='red', linestyle="--"))
    if entry: addplots.append(mpf.make_addplot([entry]*len(df), color='blue', linestyle="--"))

    fig, ax = mpf.plot(
        df.tail(200), type='candle', style='yahoo',
        title=f"{symbol} - {action}", addplot=addplots, returnfig=True
    )
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return buf

# ---------------- FETCH TOP SYMBOLS ----------------
def fetch_top_symbols(limit=300):
    try:
        # Беремо всі USDT-пари ф'ючерсів
        tickers = binance_client.futures_ticker()  # Для USDT-M ф'ючерсів
        usdt_pairs = [t for t in tickers if t['symbol'].endswith("USDT")]
        sorted_pairs = sorted(
            usdt_pairs,
            key=lambda x: abs(float(x.get("priceChangePercent", 0))),
            reverse=True
        )
        top_symbols = [d["symbol"] for d in sorted_pairs[:limit]]
        logger.info("Top %d symbols fetched: %s", limit, top_symbols[:10])
        return top_symbols
    except Exception as e:
        logger.exception("Error fetching top symbols: %s", e)
        return []

# ---------------- MASTER SCAN ----------------
def scan_all_symbols():
    symbols = fetch_top_symbols(limit=300)
    if not symbols:
        logger.warning("No symbols fetched, falling back to ALL_USDT list")
        symbols = ALL_USDT
    logger.info("Starting scan for %d symbols", len(symbols))
    def safe_analyze(sym):
        try:
            analyze_and_alert(sym)
        except Exception as e:
            logger.exception("Error analyzing symbol %s: %s", sym, e)
    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as exe:
        list(exe.map(safe_analyze, symbols))
    state["last_scan"] = str(datetime.now(timezone.utc))
    save_json_safe(STATE_FILE, state)
    logger.info("Scan finished at %s", state["last_scan"])

# ---------------- FLASK ----------------
from flask import Flask, request, jsonify
app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({
        "status": "ok",
        "time": str(datetime.now(timezone.utc)),
        "signals": len(state.get("signals", {}))
    })

@app.route("/telegram_webhook/<token>", methods=["POST"])
def telegram_webhook(token):
    if token != TELEGRAM_TOKEN:
        return jsonify({"ok": False, "error": "invalid token"}), 403
    update = request.get_json(force=True) or {}
    text = update.get("message", {}).get("text", "").lower().strip()
    if text.startswith("/scan"):
        send_telegram("⚡ Manual scan started.")
        Thread(target=scan_all_symbols, daemon=True).start()
    return jsonify({"ok": True})

# ---------------- MAIN ----------------
if __name__ == "__main__":
    logger.info("Starting pre-top detector bot")
    Thread(target=scan_all_symbols, daemon=True).start()
    app.run(host="0.0.0.0", port=PORT)