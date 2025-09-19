# main.py — Pre-top бот з графіками, Telegram і backtest winrate
import os
import time
import json
import logging
import re
from datetime import datetime, timezone
from threading import Thread
from concurrent.futures import ThreadPoolExecutor, as_completed
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

state = load_json_safe(STATE_FILE, {
    "signals": {}, 
    "last_scan": None, 
    "signal_history": {},
    "top_cache": {"timestamp": None, "data": []}
})

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
            requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto", data=data, files=files, timeout=15)
        else:
            payload = {
                "chat_id": CHAT_ID,
                "text": escape_md_v2(text),
                "parse_mode": "MarkdownV2",
                "disable_web_page_preview": True
            }
            requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", json=payload, timeout=15)
    except Exception as e:
        logger.exception("send_telegram error: %s", e)

# ---------------- MARKET DATA ----------------
def get_all_usdt_symbols():
    if not client:
        return []
    for attempt in range(5):
        try:
            ex = client.get_exchange_info()
            symbols = [s["symbol"] for s in ex["symbols"] 
                       if s["quoteAsset"]=="USDT" and s["status"]=="TRADING"]
            blacklist = ["BUSD","USDC","FDUSD","TUSD","DAI","EUR","GBP","AUD",
                         "STRAX","GNS","ALCX","BTCST","COIN","AAPL","TSLA",
                         "MSFT","META","GOOG","USD1","BTTC","ARDR","DF","XNO"]
            filtered = [s for s in symbols if not any(b in s for b in blacklist)]
            return filtered
        except Exception as e:
            logger.warning("get_all_usdt_symbols attempt %d failed: %s", attempt+1, e)
            time.sleep(0.5*(2**attempt))
    logger.error("Failed to fetch USDT symbols from Binance after retries")
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

# ---------------- FEATURES ----------------
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

    if last["ema_8"] > last["ema_20"]: votes.append("ema_bull"); confidence+=0.1
    else: votes.append("ema_bear"); confidence+=0.05

    if last["MACD_hist"] > 0: votes.append("macd_up"); confidence+=0.1
    else: votes.append("macd_down"); confidence+=0.05

    if last["RSI_14"] < 30: votes.append("rsi_oversold"); confidence+=0.08
    elif last["RSI_14"] > 70: votes.append("rsi_overbought"); confidence+=0.08

    if last["ADX"] > 25: votes.append("strong_trend"); confidence*=1.1

    body = last["close"] - last["open"]
    rng = last["high"] - last["low"]
    upper_shadow = last["high"] - max(last["close"], last["open"])
    lower_shadow = min(last["close"], last["open"]) - last["low"]
    candle_bonus = 1.0

    if lower_shadow > 2*abs(body) and body>0: votes.append("hammer_bull"); candle_bonus=1.2
    elif upper_shadow > 2*abs(body) and body<0: votes.append("shooting_star"); candle_bonus=1.2
    if body>0 and prev["close"]<prev["open"] and last["close"]>prev["open"] and last["open"]<prev["close"]: votes.append("bullish_engulfing"); candle_bonus=1.25
    elif body<0 and prev["close"]>prev["open"] and last["close"]<prev["open"] and last["open"]>prev["close"]: votes.append("bearish_engulfing"); candle_bonus=1.25
    if abs(body)<0.1*rng: votes.append("doji"); candle_bonus=1.1
    confidence*=candle_bonus

    if last["close"]>last["resistance"]*0.995 and last["close"]<last["resistance"]*1.01: votes.append("fake_breakout_short"); confidence+=0.15
    elif last["close"]<last["support"]*1.005 and last["close"]>last["support"]*0.99: votes.append("fake_breakout_long"); confidence+=0.15

    pretop=False
    if len(df)>=10 and (last["close"]-df["close"].iloc[-10])/df["close"].iloc[-10]>0.1: pretop=True; confidence+=0.1; votes.append("pretop")

    action="WATCH"
    if last["close"]>=last["resistance"]*0.995: action="SHORT"
    elif last["close"]<=last["support"]*1.005: action="LONG"

    confidence=max(0,min(1,confidence))
    return action, votes, pretop, last, confidence

# ---------------- BACKTEST ----------------
def backtest_winrate(df: pd.DataFrame):
    df = apply_all_features(df)
    results=[]
    for i in range(1,len(df)):
        sub_df=df.iloc[:i+1]
        action,votes,pretop,last,conf=detect_signal(sub_df)
        if conf>=CONF_THRESHOLD_MEDIUM:
            results.append((action, conf))
    return results

# ---------------- PLOT BACKTEST ----------------
def plot_backtest_signals(df, symbol, conf_threshold=CONF_THRESHOLD_MEDIUM):
    df_plot=df[['open','high','low','close','volume']]
    df_plot.index.name="Date"
    long_dates,long_prices=[],[]
    short_dates,short_prices=[],[]
    for i in range(1,len(df)):
        sub_df=df.iloc[:i+1]
        action,votes,pretop,last,conf=detect_signal(sub_df)
        if conf<conf_threshold: continue
        if action=="LONG": long_dates.append(last.name); long_prices.append(last["close"])
        elif action=="SHORT": short_dates.append(last.name); short_prices.append(last["close"])
    addplots=[]
    if long_dates: ydata=pd.Series([np.nan]*len(df),index=df.index); ydata.loc[long_dates]=long_prices; addplots.append(mpf.make_addplot(ydata,type='scatter',markersize=80,marker='^',color='green'))
    if short_dates: ydata=pd.Series([np.nan]*len(df),index=df.index); ydata.loc[short_dates]=short_prices; addplots.append(mpf.make_addplot(ydata,type='scatter',markersize=80,marker='v',color='red'))
    mc=mpf.make_marketcolors(up='green',down='red',wick='black',edge='black',volume='blue')
    s=mpf.make_mpf_style(marketcolors=mc,gridstyle='--',gridcolor='gray',facecolor='white')
    buf=io.BytesIO()
    mpf.plot(df_plot,type='candle',style=s,volume=True,addplot=addplots,title=f"{symbol} — Strong Signals",ylabel='Price',ylabel_lower='Volume',savefig=dict(fname=buf,dpi=100,bbox_inches='tight'))
    buf.seek(0)
    return buf

# ---------------- TOP SYMBOLS ----------------
def get_top5_symbols(symbols):
    results=[]
    for s in symbols:
        df=fetch_klines(s)
        if df is None or len(df)<20: continue
        signals=backtest_winrate(df)
        win_count=sum(1 for a,c in signals if c>=CONF_THRESHOLD_MEDIUM and a!="WATCH")
        total=len(signals)
        if total>0: results.append((s,win_count/total))
    results.sort(key=lambda x:x[1],reverse=True)
    return results[:5]

# ---------------- FULL SCAN ----------------
def scan_top_symbols():
    try:
        symbols = get_all_usdt_symbols()[:TOP_LIMIT]
        results=[]
        with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
            futures={executor.submit(get_top5_symbols,[s]):s for s in symbols}
            for fut in as_completed(futures):
                try:
                    res=fut.result()
                    if res: results.extend(res)
                except Exception as e:
                    logger.warning("scan error %s: %s", futures[fut], e)
        results.sort(key=lambda x:x[1],reverse=True)
        state["signals"]=dict(results)
        state["last_scan"]=datetime.now(timezone.utc).isoformat()
        save_json_safe(STATE_FILE,state)
        if results:
            top_text="🏆 Full Scan Results:\n"+"\n".join([f"{s}: {w*100:.1f}%" for s,w in results[:10]])
            send_telegram(top_text)
        else:
            send_telegram("❌ No results from scan")
    except Exception as e:
        logger.exception("scan_top_symbols error: %s", e)

# ---------------- TELEGRAM WEBHOOK ----------------
@app.route("/telegram_webhook/<token>", methods=["POST"])
def telegram_webhook(token):
    try:
        if token != TELEGRAM_TOKEN: return jsonify({"ok":False,"error":"invalid token"}),403
        update=request.get_json(force=True) or {}
        msg=update.get("message")
        if not msg: return jsonify({"ok":True})
        text=msg.get("text","").lower().strip()
        if text.startswith("/scan"): 
            Thread(target=scan_top_symbols, daemon=True).start()
            send_telegram("⚡ Manual scan started.")
        elif text.startswith("/status"): 
            send_telegram(f"📝 Status:\nSignals={len(state.get('signals',{}))}\nLast scan={state.get('last_scan')}")
        elif text.startswith("/top"):
            now = time.time()
            if state["top_cache"]["timestamp"] and now - state["top_cache"]["timestamp"] < 120:
                top5 = state["top_cache"]["data"]
            else:
                symbols = get_all_usdt_symbols()[:TOP_LIMIT]
                top5 = get_top5_symbols(symbols)
                state["top_cache"]={"timestamp":now,"data":top5}
                save_json_safe(STATE_FILE,state)
            if top5:
                msg_text="🏆 Top5 tokens by strong signal winrate:\n"+"\n".join([f"{s[0]}: {s[1]*100:.1f}%" for s in top5])
            else:
                msg_text="❌ No top symbols found"
            send_telegram(msg_text)
        elif text.startswith("/history"):
            parts=text.split()
            if len(parts)>=2:
                symbol=parts[1].upper()
                df=fetch_klines(symbol)
                if df is not None and len(df)>=30:
                    df=apply_all_features(df)
                    buf=plot_backtest_signals(df,symbol,conf_threshold=CONF_THRESHOLD_MEDIUM)
                    send_telegram(f"📈 Strong Signals for {symbol}",photo=buf)
                else: send_telegram(f"❌ No data for {symbol}")
    except Exception as e:
        logger.exception("telegram_webhook error: %s",e)
    return jsonify({"ok":True})

# ---------------- MAIN ----------------
if __name__=="__main__":
    logger.info("Starting pre-top detector bot")
    app.run(host="0.0.0.0",port=PORT)