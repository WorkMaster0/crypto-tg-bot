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

        if os.path.exists(tmp):  # <-- перевірка
            os.replace(tmp, path)
        else:
            logger.error("Temp file %s not created, skipping save", tmp)

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
    "BTCUSDT","ETHUSDT","MYXUSDT","SOLUSDT","XRPUSDT","ADAUSDT","DAMUSDT",
    "DOTUSDT","TRXUSDT","LTCUSDT","AVAXUSDT","SHIBUSDT","BABYUSDT","QUSDT","XMRUSDT",
    "ETCUSDT","XLMUSDT","APTUSDT","TRADOORUSDT","FILUSDT","ICPUSDT","GRTUSDT","EIGENUSDT",
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
    "CKBUSDT","ACEUSDT","TRUUSDT","IPSUSDT","QIUSDT","GLMUSDT","ZORAUSDT",
    "MIRUSDT","ROSEUSDT","OXTUSDT","SPELLUSDT","SUNUSDT","SYSUSDT","TAOUSDT",
    "TLMUSDT","VLXUSDT","WAXPUSDT","XNOUSDT", "IPUSDT", "ALCHUSDT", "YALAUSDT",
    "PUMPBTCUSDT","0GUSDT","HIPPOUSDT","FRAGUSDT","BOOSTUSDT","KTAUSDT","TUTUSDT",
    "XPLUSDT","MNTUSDT","PTBUSDT","MUSDT","STBLUSDT","BBUSDT","ORDERUSDT",
    "NAORISUSDT","OPENUSDT","RHEAUSDT","FARTCOINUSDT","AGTUSDT","VINEUSDT","DOLOUSDT",
    "MERLUSDT","AVNTUSDT","SIGNUSDT","ASTERUSDT","B2USDT","JELLYJELLYUSDT","ALPINEUSDT",
    "MEUSDT","SOLVUSDT","PROMUSDT","PIPPINUSDT","BANKUSDT","SIRENUSDT","HUSDT", "SPX",
    "SKYUSDT","SOONUSDT","IDOLUSDT","PORT3USDT","CROSSUSDT","LINEAUSDT","ESPORTSUSDT",
    "YFIUSDT","SAPIENUSDT","ZEREBROUSDT","TAKEUSDT","HAEDALUSDT", "SAHARAUSDT","SANTOSUSDT",
    "HEMIUSDT", "THEUSDT", "NEIROETH", "TSTUSDT", "HEIUSDT", "DEXEUSDT", "USELESSUSDT"
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

# ---------------- ENHANCED FEATURE ENGINEERING ----------------
def apply_all_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # EMA
    df["ema_8"] = ta.trend.EMAIndicator(df["close"], 8).ema_indicator()
    df["ema_20"] = ta.trend.EMAIndicator(df["close"], 20).ema_indicator()
    df["ema_50"] = ta.trend.EMAIndicator(df["close"], 50).ema_indicator()

    # RSI
    df["RSI_14"] = ta.momentum.RSIIndicator(df["close"], 14).rsi()
    df["RSI_7"] = ta.momentum.RSIIndicator(df["close"], 7).rsi()

    # MACD
    macd = ta.trend.MACD(df["close"])
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()
    df["MACD_hist"] = macd.macd_diff()
    df["MACD_cross_up"] = ((df["MACD"] > df["MACD_signal"]) & (df["MACD"].shift(1) <= df["MACD_signal"].shift(1)))
    df["MACD_cross_down"] = ((df["MACD"] < df["MACD_signal"]) & (df["MACD"].shift(1) >= df["MACD_signal"].shift(1)))

    # ADX
    adx = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], 14)
    df["ADX"] = adx.adx()
    df["ADX_pos"] = adx.adx_pos()
    df["ADX_neg"] = adx.adx_neg()

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df["close"], 20, 2)
    df["BB_high"] = bb.bollinger_hband()
    df["BB_low"] = bb.bollinger_lband()
    
    # Support/Resistance
    df["support"] = df["low"].rolling(20).min()
    df["resistance"] = df["high"].rolling(20).max()

    # Volume spike
    df["vol_ma20"] = df["volume"].rolling(20).mean()
    df["vol_spike"] = df["volume"] > 1.5 * df["vol_ma20"]

    return df


# ---------------- ENHANCED SIGNAL DETECTION ----------------
def detect_signal(df: pd.DataFrame):
    """
    Оцінка сигналу на основі свічкових патернів, support/resistance,
    flip та volume confirmation.
    Повертає: action, votes, pretop, last_row, confidence
    """
    last = df.iloc[-1]
    prev = df.iloc[-2]
    votes = []
    confidence = 0.2

    # ---------------- Candlestick Patterns ----------------
    body = last["close"] - last["open"]
    rng = max(1e-9, last["high"] - last["low"])
    upper_shadow = last["high"] - max(last["close"], last["open"])
    lower_shadow = min(last["close"], last["open"]) - last["low"]

    if lower_shadow > 2 * abs(body) and body > 0:
        votes.append("hammer_bull"); confidence *= 1.2
    elif upper_shadow > 2 * abs(body) and body < 0:
        votes.append("shooting_star"); confidence *= 1.2

    if body > 0 and prev["close"] < prev["open"] and last["close"] > prev["open"] and last["open"] < prev["close"]:
        votes.append("bullish_engulfing"); confidence *= 1.25
    elif body < 0 and prev["close"] > prev["open"] and last["close"] < prev["open"] and last["open"] > prev["close"]:
        votes.append("bearish_engulfing"); confidence *= 1.25

    if abs(body) < 0.1 * rng:
        votes.append("doji"); confidence *= 1.1

    # ---------------- Support/Resistance Flip ----------------
    flip_long = last["close"] > last["support"] and prev["close"] < prev["support"]
    flip_short = last["close"] < last["resistance"] and prev["close"] > prev["resistance"]

    # ---------------- Volume Confirmation ----------------
    vol_confirm = last.get("vol_spike", False)

    # ---------------- Action Determination ----------------
    action = "WATCH"
    near_resistance = last["close"] >= last["resistance"] * 0.98
    near_support = last["close"] <= last["support"] * 1.02

    if (near_support or flip_long) and any(p in votes for p in ["hammer_bull","bullish_engulfing","doji"]) and vol_confirm:
        action = "LONG"; confidence += 0.15
    elif (near_resistance or flip_short) and any(p in votes for p in ["shooting_star","bearish_engulfing","doji"]) and vol_confirm:
        action = "SHORT"; confidence += 0.15

    # ---------------- Pre-Top Detection ----------------
    pretop = False
    if len(df) >= 10 and (last["close"] - df["close"].iloc[-10]) / df["close"].iloc[-10] > 0.10:
        pretop = True
        votes.append("pretop")
        confidence += 0.10

    confidence = max(0.0, min(1.0, confidence))

    return action, votes, pretop, last, confidence


def analyze_and_alert(symbol: str):
    """
    Аналіз сигналу на основі патернів свічок, support/resistance, volume.
    Вираховує три тейки, стоп, R/R та відправляє сигнал на Telegram.
    """
    df = fetch_klines(symbol, limit=200)
    if df is None or len(df) < 20:
        return

    df = df.copy()
    
    # ---------------- Feature Engineering ----------------
    # Support/Resistance (локальні min/max за 20 свічок)
    df["support"] = df["low"].rolling(20).min()
    df["resistance"] = df["high"].rolling(20).max()
    
    # Volume confirmation
    df["vol_ma20"] = df["volume"].rolling(20).mean()
    df["vol_spike"] = df["volume"] > 1.5 * df["vol_ma20"]

    last = df.iloc[-1]
    prev = df.iloc[-2]

    # ---------------- Candlestick Patterns ----------------
    votes = []
    confidence = 0.2
    body = last["close"] - last["open"]
    rng = max(1e-9, last["high"] - last["low"])
    upper_shadow = last["high"] - max(last["close"], last["open"])
    lower_shadow = min(last["close"], last["open"]) - last["low"]

    if lower_shadow > 2 * abs(body) and body > 0:
        votes.append("hammer_bull"); confidence *= 1.2
    elif upper_shadow > 2 * abs(body) and body < 0:
        votes.append("shooting_star"); confidence *= 1.2

    if body > 0 and prev["close"] < prev["open"] and last["close"] > prev["open"] and last["open"] < prev["close"]:
        votes.append("bullish_engulfing"); confidence *= 1.25
    elif body < 0 and prev["close"] > prev["open"] and last["close"] < prev["open"] and last["open"] > prev["close"]:
        votes.append("bearish_engulfing"); confidence *= 1.25

    if abs(body) < 0.1 * rng:
        votes.append("doji"); confidence *= 1.1

    # ---------------- Support/Resistance Flip ----------------
    flip_long = last["close"] > last["support"] and prev["close"] < prev["support"]
    flip_short = last["close"] < last["resistance"] and prev["close"] > prev["resistance"]

    # ---------------- Action Determination ----------------
    action = "WATCH"
    near_resistance = last["close"] >= last["resistance"] * 0.98
    near_support = last["close"] <= last["support"] * 1.02

    if (near_support or flip_long) and any(p in votes for p in ["hammer_bull","bullish_engulfing","doji"]) and last["vol_spike"]:
        action = "LONG"; confidence += 0.15
    elif (near_resistance or flip_short) and any(p in votes for p in ["shooting_star","bearish_engulfing","doji"]) and last["vol_spike"]:
        action = "SHORT"; confidence += 0.15

    confidence = max(0.0, min(1.0, confidence))
    if action == "WATCH":
        return  # не відправляємо сигнал

    # ---------------- Entry / Stop-Loss / Take-Profits ----------------
    if action == "LONG":
        entry = last["support"] * 1.001
        stop_loss = last["support"] * 0.99
        tp1 = entry + (last["resistance"] - entry) * 0.33
        tp2 = entry + (last["resistance"] - entry) * 0.66
        tp3 = last["resistance"]
    else:
        entry = last["resistance"] * 0.999
        stop_loss = last["resistance"] * 1.01
        tp1 = entry - (entry - last["support"]) * 0.33
        tp2 = entry - (entry - last["support"]) * 0.66
        tp3 = last["support"]

    # Risk/Reward
    rr1 = (tp1 - entry)/(entry - stop_loss) if action=="LONG" else (entry - tp1)/(stop_loss - entry)
    rr2 = (tp2 - entry)/(entry - stop_loss) if action=="LONG" else (entry - tp2)/(stop_loss - entry)
    rr3 = (tp3 - entry)/(entry - stop_loss) if action=="LONG" else (entry - tp3)/(stop_loss - entry)

    # ---------------- Message ----------------
    msg = (
        f"⚡ TRADE SIGNAL\n"
        f"Symbol: {symbol}\n"
        f"Action: {action}\n"
        f"Entry: {entry:.6f}\n"
        f"Stop-Loss: {stop_loss:.6f}\n"
        f"Take-Profit 1: {tp1:.6f} (RR {rr1:.2f})\n"
        f"Take-Profit 2: {tp2:.6f} (RR {rr2:.2f})\n"
        f"Take-Profit 3: {tp3:.6f} (RR {rr3:.2f})\n"
        f"Confidence: {confidence:.2f}\n"
        f"Patterns: {', '.join(votes)}\n"
    )

    # ---------------- Plot ----------------
    addplots = [
        mpf.make_addplot([tp1]*len(df), color='green'),
        mpf.make_addplot([tp2]*len(df), color='lime'),
        mpf.make_addplot([tp3]*len(df), color='darkgreen'),
        mpf.make_addplot([stop_loss]*len(df), color='red'),
        mpf.make_addplot([entry]*len(df), color='blue')
    ]

    fig, ax = mpf.plot(
        df.tail(200), type='candle', style='yahoo',
        title=f"{symbol} - {action}", addplot=addplots, returnfig=True
    )
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)

    # ---------------- Send Telegram ----------------
    send_telegram(msg, photo=buf)

    # ---------------- Save state ----------------
    state.setdefault("signals", {})[symbol] = {
        "action": action,
        "entry": entry,
        "sl": stop_loss,
        "tp1": tp1,
        "tp2": tp2,
        "tp3": tp3,
        "rr1": rr1,
        "rr2": rr2,
        "rr3": rr3,
        "confidence": confidence,
        "time": str(last.name),
        "last_price": float(last["close"]),
        "votes": votes
    }
    save_json_safe(STATE_FILE, state)

# ---------------- MASTER SCAN ----------------
def scan_all_symbols():
    symbols = list(ws_manager.data.keys()) or ALL_USDT
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