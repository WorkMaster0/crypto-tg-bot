# main.py — Pre-top бот з графіками, Telegram і реальним backtest winrate
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
import ta
import mplfinance as mpf
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
PORT = int(os.getenv("PORT", "5000"))

TOP_LIMIT = int(os.getenv("TOP_LIMIT", "100"))
EMA_SCAN_LIMIT = int(os.getenv("EMA_SCAN_LIMIT", "500"))
CONF_THRESHOLD_MEDIUM = 0.60
CONF_THRESHOLD_STRONG = 0.80
CACHE_TTL = 120  # кеш на 2 хв

# ---------------- LOGGING ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler("bot.log"), logging.StreamHandler()]
)
logger = logging.getLogger("pretop-bot")

# ---------------- BINANCE ----------------
if BINANCE_PY_AVAILABLE and BINANCE_API_KEY and BINANCE_API_SECRET:
    from requests import Session
    session = Session()
    client = BinanceClient(
        api_key=BINANCE_API_KEY,
        api_secret=BINANCE_API_SECRET,
        requests_params={"timeout": 30}
    )
else:
    client = None
    logger.warning("Binance client unavailable or API keys missing")

# ---------------- FLASK ----------------
app = Flask(__name__)

# ---------------- STATE ----------------
STATE_FILE = "state.json"

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

state = load_json_safe(STATE_FILE, {"top_cache":{"timestamp":None,"data":[]}})

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
            payload = {"chat_id": CHAT_ID,"text":escape_md_v2(text),"parse_mode":"MarkdownV2","disable_web_page_preview":True}
            requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", json=payload, timeout=10)
    except Exception as e:
        logger.exception("send_telegram error: %s", e)

# ---------------- MARKET DATA ----------------
def get_all_usdt_symbols():
    if not client:
        return []
    try:
        ex = client.get_exchange_info()
        symbols = [s["symbol"] for s in ex["symbols"] if s["quoteAsset"]=="USDT" and s["status"]=="TRADING"]
        blacklist = ["BUSD","USDC","FDUSD","TUSD","DAI","EUR","GBP","AUD",
                     "STRAX","GNS","ALCX","BTCST","COIN","AAPL","TSLA",
                     "MSFT","META","GOOG","USD1","BTTC","ARDR","DF","XNO"]
        return [s for s in symbols if not any(b in s for b in blacklist)]
    except Exception as e:
        logger.exception("get_all_usdt_symbols error: %s", e)
        return []

def fetch_klines(symbol, interval="15m", limit=EMA_SCAN_LIMIT):
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
        return None

# ---------------- FEATURE ENGINEERING ----------------
def apply_all_features(df):
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
        df["support"] = df["low"].rolling(20).min()
        df["resistance"] = df["high"].rolling(20).max()
    except Exception as e:
        logger.exception("apply_all_features error: %s", e)
    return df

# ---------------- SIGNAL DETECTION ----------------
def detect_signal(df):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    confidence = 0.2
    if last["ema_8"] > last["ema_20"]: confidence+=0.1
    if last["MACD_hist"]>0: confidence+=0.1
    if last["RSI_14"]<30 or last["RSI_14"]>70: confidence+=0.08
    if last["ADX"]>25: confidence*=1.1

    body = last["close"]-last["open"]
    rng = last["high"]-last["low"]
    candle_bonus=1.0
    upper_shadow = last["high"] - max(last["close"],last["open"])
    lower_shadow = min(last["close"],last["open"])-last["low"]
    if lower_shadow > 2*abs(body) and body>0: candle_bonus=1.2
    if upper_shadow > 2*abs(body) and body<0: candle_bonus=1.2
    confidence*=candle_bonus

    action="WATCH"
    if last["close"]>=last["resistance"]*0.995: action="SHORT"
    elif last["close"]<=last["support"]*1.005: action="LONG"

    confidence = min(max(confidence,0),1)
    return action, confidence

# ---------------- BACKTEST WINRATE ----------------
def backtest_winrate(df, lookahead=5, profit_thresh=0.005):
    df = apply_all_features(df)
    results=[]
    for i in range(1,len(df)-lookahead):
        sub_df = df.iloc[:i+1]
        action, conf = detect_signal(sub_df)
        if action=="WATCH": continue
        entry_price = sub_df["close"].iloc[-1]
        future = df["close"].iloc[i+1:i+1+lookahead]
        win=False
        if action=="LONG" and (future>=entry_price*(1+profit_thresh)).any(): win=True
        if action=="SHORT" and (future<=entry_price*(1-profit_thresh)).any(): win=True
        results.append({"action":action,"confidence":conf,"win":win})
    return results

# ---------------- TOP SYMBOLS ----------------
def get_top5_symbols(symbols):
    results=[]
    for s in symbols:
        df=fetch_klines(s)
        if df is None or len(df)<20: continue
        signals = backtest_winrate(df)
        total = len(signals)
        if total==0: continue
        wins = sum(1 for sig in signals if sig["win"])
        results.append((s,wins/total,total))
    results.sort(key=lambda x:x[1],reverse=True)
    return results[:5]

def send_top_symbols_telegram():
    try:
        now = time.time()
        if state["top_cache"]["timestamp"] and now - state["top_cache"]["timestamp"] < CACHE_TTL:
            top5 = state["top_cache"]["data"]
        else:
            symbols = get_all_usdt_symbols()[:TOP_LIMIT]
            top5 = get_top5_symbols(symbols)
            state["top_cache"] = {"timestamp":now,"data":top5}
            save_json_safe(STATE_FILE,state)
        if top5:
            msg="🏆 Top5 tokens by real winrate:\n" + "\n".join([f"{s[0]}: {s[1]*100:.1f}% ({s[2]} signals)" for s in top5])
        else: msg="❌ No top symbols found"
        send_telegram(msg)
    except Exception as e:
        logger.exception("send_top_symbols_telegram error: %s", e)

# ---------------- TELEGRAM WEBHOOK ----------------
@app.route("/telegram_webhook/<token>", methods=["POST"])
def telegram_webhook(token):
    try:
        if token != TELEGRAM_TOKEN: return jsonify({"ok":False,"error":"invalid token"}),403
        update=request.get_json(force=True) or {}
        msg=update.get("message")
        if not msg: return jsonify({"ok":True})
        text=msg.get("text","").lower().strip()
        if text.startswith("/top"): 
            Thread(target=send_top_symbols_telegram, daemon=True).start()
            send_telegram("⏳ Processing top symbols, please wait...")
    except Exception as e:
        logger.exception("telegram_webhook error: %s", e)
    return jsonify({"ok":True})

# ---------------- MAIN ----------------
if __name__=="__main__":
    logger.info("Starting pre-top detector bot")
    app.run(host="0.0.0.0", port=PORT)