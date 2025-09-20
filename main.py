# main.py — Оптимізований Pre-top бот
import os, time, json, logging, re, io
from datetime import datetime, timezone
from threading import Thread
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np
import requests
from flask import Flask, request, jsonify
import ta
import mplfinance as mpf

try:
    from binance.client import Client as BinanceClient
    BINANCE_AVAILABLE = True
except:
    BINANCE_AVAILABLE = False

# ---------------- CONFIG ----------------
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY","")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET","")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN","")
CHAT_ID = os.getenv("CHAT_ID","")
PORT = int(os.getenv("PORT","5000"))

TOP_LIMIT = 50
EMA_SCAN_LIMIT = 100   # використовуємо лише останні 100 свічок
PARALLEL_WORKERS = 4

STATE_FILE = "state.json"
LOG_FILE = "bot.log"

CONF_THRESHOLD_MEDIUM = 0.60
CONF_THRESHOLD_STRONG = 0.80

# ---------------- LOGGING ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", 
                    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()])
logger = logging.getLogger("pretop-bot")

app = Flask(__name__)

# ---------------- STATE ----------------
def load_json_safe(path, default): 
    try:
        if os.path.exists(path): 
            with open(path,"r") as f: return json.load(f)
    except: pass
    return default

def save_json_safe(path, data):
    try:
        tmp = path+".tmp"
        with open(tmp,"w") as f: json.dump(data,f,indent=2,default=str)
        os.replace(tmp,path)
    except: pass

state = load_json_safe(STATE_FILE, {"signals":{}, "last_scan":None, "top_cache":{"timestamp":None,"data":[]}})

# ---------------- TELEGRAM ----------------
MARKDOWNV2_ESCAPE = r"_*[]()~`>#+-=|{}.!"
def escape_md_v2(text): return re.sub(f"([{re.escape(MARKDOWNV2_ESCAPE)}])", r"\\\1", str(text))

def send_telegram(text, photo=None):
    if not TELEGRAM_TOKEN or not CHAT_ID: return
    try:
        if photo:
            files = {'photo': ('signal.png', photo, 'image/png')}
            data = {'chat_id': CHAT_ID, 'caption': escape_md_v2(text), 'parse_mode':'MarkdownV2'}
            requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto", data=data, files=files, timeout=15)
        else:
            payload = {"chat_id":CHAT_ID,"text":escape_md_v2(text),"parse_mode":"MarkdownV2","disable_web_page_preview":True}
            requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", json=payload, timeout=15)
    except Exception as e: logger.warning("Telegram error: %s", e)

# ---------------- BINANCE CLIENT ----------------
client=None
def get_client():
    global client
    if client is None and BINANCE_AVAILABLE and BINANCE_API_KEY and BINANCE_API_SECRET:
        from requests import Session
        session = Session()
        client = BinanceClient(api_key=BINANCE_API_KEY, api_secret=BINANCE_API_SECRET, requests_params={"timeout":30})
    return client

# ---------------- MARKET DATA ----------------
_symbols_cache={"timestamp":0,"symbols":[]}
def get_all_usdt_symbols(force_refresh=False):
    now=time.time()
    if not force_refresh and now-_symbols_cache["timestamp"]<300: return _symbols_cache["symbols"]
    client=get_client()
    if not client: return []
    try:
        ex=client.get_exchange_info()
        symbols=[s["symbol"] for s in ex["symbols"] if s["quoteAsset"]=="USDT" and s["status"]=="TRADING"]
        blacklist=["BUSD","USDC","FDUSD","TUSD","DAI","EUR","GBP","AUD"]
        filtered=[s for s in symbols if not any(b in s for b in blacklist)]
        _symbols_cache.update({"timestamp":now,"symbols":filtered})
        return filtered
    except: return []

def fetch_klines(symbol, interval="15m", limit=EMA_SCAN_LIMIT):
    client=get_client()
    if not client: return None
    try:
        kl=client.get_klines(symbol=symbol, interval=interval, limit=limit)
        df=pd.DataFrame(kl, columns=["open_time","open","high","low","close","volume","close_time","qav","num_trades","tb_base","tb_quote","ignore"])
        df=df[["open_time","open","high","low","close","volume"]].astype(float)
        df["open_time"]=pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df.set_index("open_time", inplace=True)
        return df
    except: return None

# ---------------- FEATURES ----------------
def apply_all_features(df):
    try:
        df["ema_8"]=ta.trend.EMAIndicator(df["close"],8).ema_indicator()
        df["ema_20"]=ta.trend.EMAIndicator(df["close"],20).ema_indicator()
        df["RSI_14"]=ta.momentum.RSIIndicator(df["close"],14).rsi()
        macd=ta.trend.MACD(df["close"])
        df["MACD"]=macd.macd(); df["MACD_signal"]=macd.macd_signal(); df["MACD_hist"]=macd.macd_diff()
        adx=ta.trend.ADXIndicator(df["high"],df["low"],df["close"],14)
        df["ADX"]=adx.adx(); df["ADX_pos"]=adx.adx_pos(); df["ADX_neg"]=adx.adx_neg()
        df["support"]=df["low"].rolling(20).min(); df["resistance"]=df["high"].rolling(20).max()
    except: pass
    return df

# ---------------- SIGNAL DETECTION ----------------
def detect_signal(df):
    last, prev = df.iloc[-1], df.iloc[-2]
    confidence=0.2; votes=[]
    if last["ema_8"]>last["ema_20"]: votes.append("ema_bull"); confidence+=0.1
    else: votes.append("ema_bear"); confidence+=0.05
    if last["MACD_hist"]>0: votes.append("macd_up"); confidence+=0.1
    else: votes.append("macd_down"); confidence+=0.05
    if last["RSI_14"]<30: votes.append("rsi_oversold"); confidence+=0.08
    elif last["RSI_14"]>70: votes.append("rsi_overbought"); confidence+=0.08
    if last["ADX"]>25: votes.append("strong_trend"); confidence*=1.1
    body=last["close"]-last["open"]; rng=last["high"]-last["low"]
    upper_shadow=last["high"]-max(last["close"],last["open"])
    lower_shadow=min(last["close"],last["open"])-last["low"]; candle_bonus=1.0
    if lower_shadow>2*abs(body) and body>0: votes.append("hammer_bull"); candle_bonus=1.2
    elif upper_shadow>2*abs(body) and body<0: votes.append("shooting_star"); candle_bonus=1.2
    confidence*=candle_bonus
    pretop=False
    if len(df)>=10 and (last["close"]-df["close"].iloc[-10])/df["close"].iloc[-10]>0.1: pretop=True; confidence+=0.1
    action="WATCH"
    if last["close"]>=last["resistance"]*0.995: action="SHORT"
    elif last["close"]<=last["support"]*1.005: action="LONG"
    confidence=max(0,min(1,confidence))
    return action, votes, pretop, last, confidence

# ---------------- BACKTEST ----------------
def backtest_winrate(df, stop_loss=0.02, take_profit=0.04, candles=5):
    df=apply_all_features(df)
    results=[]
    for i in range(candles,len(df)):
        sub=df.iloc[:i+1]; action,votes,pretop,last,conf=detect_signal(sub)
        if conf>=CONF_THRESHOLD_MEDIUM and action!="WATCH":
            entry=last["close"]
            future=df.iloc[i+1:i+1+candles]["close"].values
            win=False
            for p in future:
                if action=="LONG" and p>=entry*(1+take_profit): win=True; break
                elif action=="LONG" and p<=entry*(1-stop_loss): win=False; break
                elif action=="SHORT" and p<=entry*(1-take_profit): win=True; break
                elif action=="SHORT" and p>=entry*(1+stop_loss): win=False; break
            results.append((action, conf, win))
    return results

# ---------------- TOP SYMBOLS ----------------
def get_top5_symbols(symbols):
    results=[]
    for s in symbols:
        df=fetch_klines(s)
        if df is None or len(df)<20: continue
        signals=backtest_winrate(df)
        win_count=sum(1 for a,c,win in signals if win)
        total=len(signals)
        if total>0: results.append((s, win_count/total))
    results.sort(key=lambda x:x[1], reverse=True)
    return results[:5]

# ---------------- FULL SCAN ----------------
def scan_top_symbols():
    symbols=get_all_usdt_symbols()[:TOP_LIMIT]; results=[]
    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
        futures={executor.submit(get_top5_symbols,[s]):s for s in symbols}
        for fut in as_completed(futures):
            try:
                res=fut.result(); results.extend(res)
            except: pass
    results.sort(key=lambda x:x[1], reverse=True)
    state["signals"]=dict(results); state["last_scan"]=datetime.now(timezone.utc).isoformat()
    save_json_safe(STATE_FILE,state)
    if results: send_telegram("🏆 Full Scan Results:\n"+ "\n".join([f"{s}: {w*100:.1f}%" for s,w in results[:10]]))
    else: send_telegram("❌ No results from scan")

# ---------------- TELEGRAM WEBHOOK ----------------
@app.route("/telegram_webhook/<token>", methods=["POST"])
def telegram_webhook(token):
    if token!=TELEGRAM_TOKEN: return jsonify({"ok":False,"error":"invalid token"}),403
    update=request.get_json(force=True) or {}; msg=update.get("message")
    if not msg: return jsonify({"ok":True})
    text=msg.get("text","").lower().strip()
    if text.startswith("/scan"): Thread(target=scan_top_symbols, daemon=True).start(); send_telegram("⚡ Manual scan started.")
    elif text.startswith("/status"): send_telegram(f"📝 Status:\nSignals={len(state.get('signals',{}))}\nLast scan={state.get('last_scan')}")
    return jsonify({"ok":True})

if __name__=="__main__":
    logger.info("Starting bot")
    app.run(host="0.0.0.0", port=PORT)