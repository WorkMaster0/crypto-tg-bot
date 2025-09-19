# main.py — Pre-top бот з TA-Lib патернами, fake-breakout фільтром, Telegram, Flask/webhook
import os
import time
import json
import logging
import re
from datetime import datetime, timezone
from threading import Lock
from concurrent.futures import ThreadPoolExecutor
import io

import pandas as pd
import matplotlib.pyplot as plt
import requests
from flask import Flask, request, jsonify
from apscheduler.schedulers.background import BackgroundScheduler
import ta
import mplfinance as mpf
from fpdf import FPDF

# optional libs
try:
    import talib
    TALIB_AVAILABLE = True
except Exception:
    TALIB_AVAILABLE = False

try:
    from binance.client import Client as BinanceClient
    BINANCE_PY_AVAILABLE = True
except Exception:
    BINANCE_PY_AVAILABLE = False

# ---------------- CONFIG (змінити у середовищі) ----------------
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
CHAT_ID = os.getenv("CHAT_ID", "")
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "")
PORT = int(os.getenv("PORT", "5000"))

SCAN_INTERVAL_MINUTES = int(os.getenv("SCAN_INTERVAL_MINUTES", "1"))
EMA_SCAN_LIMIT = int(os.getenv("EMA_SCAN_LIMIT", "500"))
PARALLEL_WORKERS = int(os.getenv("PARALLEL_WORKERS", "6"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "50"))

STATE_DIR = "data"
os.makedirs(STATE_DIR, exist_ok=True)
STATE_FILE = os.path.join(STATE_DIR, "state.json")
LOG_FILE = os.path.join(STATE_DIR, "bot.log")

CONF_THRESHOLD_MEDIUM = float(os.getenv("CONF_THRESHOLD_MEDIUM", "0.60"))
CONF_THRESHOLD_STRONG = float(os.getenv("CONF_THRESHOLD_STRONG", "0.80"))

# ---------------- LOGGING ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)
logger = logging.getLogger("pretop-bot")

# ---------------- BINANCE CLIENT ----------------
if BINANCE_PY_AVAILABLE and BINANCE_API_KEY and BINANCE_API_SECRET:
    client = BinanceClient(api_key=BINANCE_API_KEY, api_secret=BINANCE_API_SECRET)
else:
    client = None
    logger.warning("Binance client unavailable or API keys missing — market data will be limited or empty")

# ---------------- FLASK ----------------
app = Flask(__name__)

# ---------------- STATE + LOCK ----------------
state_lock = Lock()
scan_lock = Lock()
executor = ThreadPoolExecutor(max_workers=4)

def load_json_safe(path, default):
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logger.exception("load_json_safe error %s: %s", path, e)
    return default

def save_json_safe(path, data):
    try:
        with state_lock:
            snapshot = json.loads(json.dumps(data, default=str, ensure_ascii=False))
        dir_path = os.path.dirname(os.path.abspath(path))
        os.makedirs(dir_path, exist_ok=True)
        tmp = os.path.join(dir_path, os.path.basename(path) + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    except Exception as e:
        logger.exception("save_json_safe error %s: %s", path, e)

state = load_json_safe(STATE_FILE, {
    "signals": {},
    "last_scan": None,
    "signal_history": {},
    "win_stats": {},
    "user_profiles": {},
})

# ---------------- TELEGRAM ----------------
MARKDOWNV2_ESCAPE = r"_*[]()~`>#+=|{}.!"

def escape_md_v2(text: str) -> str:
    return re.sub(f"([{re.escape(MARKDOWNV2_ESCAPE)}])", r"\\\1", str(text))

def send_telegram(text: str, photo=None, chat_id=None, is_document=False, filename="file.pdf"):
    chat_id = chat_id or CHAT_ID
    if not TELEGRAM_TOKEN or not chat_id:
        logger.debug("Telegram token or chat_id missing, skipping send")
        return
    try:
        if photo:
            files = {}
            data = {'chat_id': str(chat_id), 'caption': escape_md_v2(text), 'parse_mode': 'MarkdownV2'}
            if is_document:
                files['document'] = (filename, photo, 'application/pdf')
                resp = requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendDocument", data=data, files=files, timeout=20)
            else:
                files['photo'] = ('chart.png', photo, 'image/png')
                resp = requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto", data=data, files=files, timeout=20)
            if resp is not None and resp.status_code != 200:
                logger.warning("Telegram send media failed: %s", resp.text)
        else:
            payload = {
                "chat_id": str(chat_id),
                "text": escape_md_v2(text),
                "parse_mode": "MarkdownV2",
                "disable_web_page_preview": True
            }
            resp = requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", json=payload, timeout=10)
            if resp is not None and resp.status_code != 200:
                logger.warning("Telegram send message failed: %s", resp.text)
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
USDT_SYMBOLS_CACHE = []

def init_usdt_symbols():
    global USDT_SYMBOLS_CACHE
    if client:
        try:
            ex = client.get_exchange_info()
            symbols = [
                s["symbol"] for s in ex["symbols"]
                if s["quoteAsset"] == "USDT" and s["status"] == "TRADING"
            ]
            blacklist = [
                "BUSD", "USDC", "FDUSD", "TUSD", "DAI", "EUR", "GBP", "AUD",
                "BTCST", "COIN", "AAPL", "TSLA", "MSFT", "META", "GOOG"
            ]
            USDT_SYMBOLS_CACHE = [s for s in symbols if not any(b in s for b in blacklist)]
            logger.info(f"Cached {len(USDT_SYMBOLS_CACHE)} USDT symbols")
        except Exception as e:
            logger.exception("init_usdt_symbols error: %s", e)

def get_all_usdt_symbols():
    global USDT_SYMBOLS_CACHE
    return USDT_SYMBOLS_CACHE

def fetch_klines(symbol, interval="15m", limit=EMA_SCAN_LIMIT):
    for attempt in range(3):
        try:
            if not client:
                raise RuntimeError("Binance client unavailable")
            kl = client.get_klines(symbol=symbol, interval=interval, limit=limit)
            df = pd.DataFrame(kl, columns=[
                "open_time","open","high","low","close","volume",
                "close_time","qav","num_trades","tb_base","tb_quote","ignore"
            ])
            df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
            df = df[["open_time","open","high","low","close","volume"]].astype({
                "open": float, "high": float, "low": float, "close": float, "volume": float
            })
            df.set_index("open_time", inplace=True)
            df["symbol"] = symbol
            return df
        except Exception as e:
            logger.warning("fetch_klines %s attempt %d error: %s", symbol, attempt+1, e)
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
        df["ATR"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], 14).average_true_range()
    except Exception as e:
        logger.exception("apply_all_features error: %s", e)
    return df

# ---------------- CANDLE PATTERNS ----------------
CANDLE_PATTERNS = {
    "CDLHAMMER": {"signal": "LONG", "weight": 0.10},
    "CDLINVERTEDHAMMER": {"signal": "LONG", "weight": 0.10},
    "CDLENGULFING": {"signal": "BOTH", "weight": 0.15},
    "CDLDOJI": {"signal": "NEUTRAL", "weight": 0.08},
    "CDLMORNINGSTAR": {"signal": "LONG", "weight": 0.12},
    "CDLEVENINGSTAR": {"signal": "SHORT", "weight": 0.12},
    "CDLHARAMI": {"signal": "BOTH", "weight": 0.10},
    "CDLDARKCLOUDCOVER": {"signal": "SHORT", "weight": 0.12},
    "CDL3WHITESOLDIERS": {"signal": "LONG", "weight": 0.15},
    "CDL3BLACKCROWS": {"signal": "SHORT", "weight": 0.15},
    "CDLSHOOTINGSTAR": {"signal": "SHORT", "weight": 0.10}
}

def apply_candle_patterns(df: pd.DataFrame) -> pd.DataFrame:
    if not TALIB_AVAILABLE:
        df = df.copy()
        last_body = df["close"] - df["open"]
        upper_shadow = df["high"] - df[["close","open"]].max(axis=1)
        lower_shadow = df[["close","open"]].min(axis=1) - df["low"]
        df["FALLBACK_HAMMER"] = ((lower_shadow > 2 * last_body.abs()) & (last_body > 0)).astype(int)
        df["FALLBACK_SHOOTINGSTAR"] = ((upper_shadow > 2 * last_body.abs()) & (last_body < 0)).astype(int)
        return df
    open_, high_, low_, close_ = df["open"], df["high"], df["low"], df["close"]
    df = df.copy()
    for name in CANDLE_PATTERNS.keys():
        func = getattr(talib, name, None)
        if func:
            try:
                df[name] = func(open_, high_, low_, close_)
            except Exception:
                df[name] = 0
        else:
            df[name] = 0
    return df

# ---------------- FAKE BREAKOUT DETECTOR ----------------
def detect_fake_breakout(df: pd.DataFrame) -> bool:
    try:
        if df is None or len(df) < 3:
            return False
        last = df.iloc[-1]
        prev = df.iloc[-2]
        supports = df['support'].dropna()
        resistances = df['resistance'].dropna()
        if len(supports) < 2 or len(resistances) < 2:
            return False
        s_level = supports.iloc[-2]
        r_level = resistances.iloc[-2]
        UP_BREAK_TOL = 1.002
        BACK_IN_TOL = 0.999
        DOWN_BREAK_TOL = 0.998
        BACK_UP_TOL = 1.001
        prev_broke_up = (prev['close'] > r_level * UP_BREAK_TOL) or (prev['high'] > r_level * UP_BREAK_TOL)
        last_back_below = last['close'] < r_level * BACK_IN_TOL
        prev_broke_down = (prev['close'] < s_level * DOWN_BREAK_TOL) or (prev['low'] < s_level * DOWN_BREAK_TOL)
        last_back_above = last['close'] > s_level * BACK_UP_TOL
        if prev_broke_up and last_back_below:
            return True
        if prev_broke_down and last_back_above:
            return True
    except Exception as e:
        logger.exception("detect_fake_breakout error: %s", e)
    return False

# ---------------- SIGNAL DETECTION ----------------
def detect_signal(df: pd.DataFrame, symbol: str):
    if df is None or len(df) < 5:
        return "WATCH", [], False, None, 0.0
    df = df.copy()
    last = df.iloc[-1]
    votes = []
    confidence = 0.2
    try:
        if last["ema_8"] > last["ema_20"]:
            votes.append("ema_bull")
            confidence += 0.08
        else:
            votes.append("ema_bear")
            confidence += 0.03
    except Exception:
        pass
    try:
        if last.get("MACD_hist", 0) > 0:
            votes.append("macd_up")
            confidence += 0.08
        else:
            votes.append("macd_down")
            confidence += 0.03
    except Exception:
        pass
    try:
        if last.get("RSI_14", 50) < 30:
            votes.append("rsi_oversold")
            confidence += 0.06
        elif last.get("RSI_14", 50) > 70:
            votes.append("rsi_overbought")
            confidence += 0.06
    except Exception:
        pass
    try:
        if last.get("ADX", 0) > 25:
            votes.append("strong_trend")
            confidence *= 1.05
    except Exception:
        pass
    try:
        for name, cfg in CANDLE_PATTERNS.items():
            val = int(last.get(name, 0)) if name in last.index else 0
            if val == 0:
                continue
            if cfg["signal"] == "LONG" and val > 0:
                votes.append(name.lower())
                confidence += cfg["weight"]
            elif cfg["signal"] == "SHORT" and val < 0:
                votes.append(name.lower())
                confidence += cfg["weight"]
            elif cfg["signal"] == "BOTH":
                if val > 0:
                    votes.append(name.lower() + "_bull")
                    confidence += cfg["weight"]
                elif val < 0:
                    votes.append(name.lower() + "_bear")
                    confidence += cfg["weight"]
            else:
                votes.append(name.lower())
                confidence += cfg["weight"]
    except Exception:
        pass
    if not TALIB_AVAILABLE:
        try:
            fb_hammer = int(last.get("FALLBACK_HAMMER", 0))
            fb_shoot = int(last.get("FALLBACK_SHOOTINGSTAR", 0))
            if fb_hammer:
                votes.append("fallback_hammer")
                confidence += 0.10
            if fb_shoot:
                votes.append("fallback_shootingstar")
                confidence += 0.10
        except Exception:
            pass
    pretop = False
    try:
        if len(df) >= 10 and (last["close"] - df["close"].iloc[-10]) / df["close"].iloc[-10] > 0.10:
            pretop = True
            votes.append("pretop")
            confidence += 0.08
    except Exception:
        pass
    action = "WATCH"
    try:
        if pd.notna(last.get("resistance")) and last["close"] >= last["resistance"] * 0.995:
            action = "SHORT"
        elif pd.notna(last.get("support")) and last["close"] <= last["support"] * 1.005:
            action = "LONG"
    except Exception:
        pass
    if action != "WATCH":
        fb = detect_fake_breakout(df)
        if not fb:
            logger.info(f"{symbol}: potential {action} ignored — no fake breakout detected")
            action = "WATCH"
        else:
            votes.append("fake_breakout")
            confidence += 0.08
    try:
        confidence = max(0.0, min(1.0, confidence))
    except Exception:
        confidence = 0.0
    return action, votes, pretop, last, confidence

# ---------------- INITIALIZE ----------------
init_usdt_symbols()

# ---------------- Scheduler ----------------
scheduler = BackgroundScheduler()
scheduler.add_job(
    lambda: executor.submit(lambda: scan_top_symbols_safe()),
    "interval",
    minutes=max(1, SCAN_INTERVAL_MINUTES),
    max_instances=2,
    coalesce=True,
    misfire_grace_time=120
)
scheduler.start()

# ---------------- RUN ----------------
if __name__ == "__main__":
    logger.info(f"Starting bot on port {PORT}")
    app.run(host="0.0.0.0", port=PORT)