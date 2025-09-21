# main_render_ws_safe.py — Pre-top бот + Flask для Render
import os
import json
import logging
import re
from datetime import datetime, timezone
import threading
import time
import io

import pandas as pd
import matplotlib.pyplot as plt
import ta
import mplfinance as mpf
from scipy.signal import find_peaks
import numpy as np
import websocket
import requests
from flask import Flask, jsonify

# ---------------- CONFIG ----------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
CHAT_ID = os.getenv("CHAT_ID", "").strip()
TOP_LIMIT = int(os.getenv("TOP_LIMIT", "100"))
EMA_SCAN_LIMIT = int(os.getenv("EMA_SCAN_LIMIT", "500"))
CONF_THRESHOLD_MEDIUM = 0.60
PORT = int(os.getenv("PORT", "10000"))
STATE_FILE = "state_render_ws.json"
LOG_FILE = "bot_render_ws_safe.log"

if not TELEGRAM_TOKEN or not CHAT_ID:
    raise ValueError("Telegram token and Chat ID must be set in environment variables!")

# ---------------- LOGGING ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)
logger = logging.getLogger("pretop-bot-render-ws-safe")

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

state = load_json_safe(STATE_FILE, {"signals": {}, "last_update": None})

# ---------------- TELEGRAM ----------------
MARKDOWNV2_ESCAPE = r"_*[]()~`>#+-=|{}.!"

def escape_md_v2(text: str) -> str:
    return re.sub(f"([{re.escape(MARKDOWNV2_ESCAPE)}])", r"\\\1", str(text))

def send_telegram(text: str, photo=None):
    """Відправка сигналу тільки через твій токен і Chat ID"""
    try:
        logger.info("Sending Telegram: %s", text)
        if photo:
            files = {'photo': ('signal.png', photo, 'image/png')}
            data = {'chat_id': CHAT_ID, 'caption': escape_md_v2(text), 'parse_mode': 'MarkdownV2'}
            resp = requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto",
                                 data=data, files=files, timeout=10)
        else:
            payload = {"chat_id": CHAT_ID, "text": escape_md_v2(text), "parse_mode": "MarkdownV2"}
            resp = requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                                 json=payload, timeout=10)
        if resp.status_code != 200:
            logger.warning("Telegram API response: %s - %s", resp.status_code, resp.text)
    except Exception as e:
        logger.exception("send_telegram error: %s", e)

# ---------------- MARKET DATA ----------------
def get_all_usdt_symbols():
    try:
        url = "https://api.binance.com/api/v3/exchangeInfo"
        ex = requests.get(url, timeout=10).json()
        symbols = [s["symbol"] for s in ex["symbols"] if s["quoteAsset"]=="USDT" and s["status"]=="TRADING"]
        blacklist=["BUSD","USDC","FDUSD","TUSD","DAI","EUR","GBP","AUD"]
        filtered=[s for s in symbols if not any(b in s for b in blacklist)]
        return filtered[:TOP_LIMIT]
    except Exception as e:
        logger.exception("get_all_usdt_symbols error: %s", e)
        return []

# ---------------- FEATURES ----------------
def apply_all_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    try:
        df["ema_8"] = ta.trend.EMAIndicator(df["close"], 8).ema_indicator()
        df["ema_20"] = ta.trend.EMAIndicator(df["close"], 20).ema_indicator()
        df["RSI_14"] = ta.momentum.RSIIndicator(df["close"], 14).rsi()
        macd = ta.trend.MACD(df["close"])
        df["MACD_hist"] = macd.macd_diff()
        adx = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], 14)
        df["ADX"] = adx.adx()
        df["support"] = df["low"].rolling(20).min()
        df["resistance"] = df["high"].rolling(20).max()
    except Exception as e:
        logger.exception("apply_all_features error: %s", e)
    return df

# ---------------- SIGNAL ----------------
def detect_signal(df: pd.DataFrame):
    if len(df)<2: return "WATCH",[],False,None,0.0
    last=df.iloc[-1]; votes=[]; confidence=0.2

    # EMA
    if last["ema_8"]>last["ema_20"]: votes.append("ema_bull"); confidence+=0.1
    else: votes.append("ema_bear"); confidence+=0.05
    # MACD
    if last["MACD_hist"]>0: votes.append("macd_up"); confidence+=0.1
    else: votes.append("macd_down"); confidence+=0.05
    # RSI
    if last["RSI_14"]<30: votes.append("rsi_oversold"); confidence+=0.08
    elif last["RSI_14"]>70: votes.append("rsi_overbought"); confidence+=0.08
    # ADX
    if last["ADX"]>25: votes.append("strong_trend"); confidence*=1.1

    # Свічкові патерни
    body=last["close"]-last["open"]; rng=last["high"]-last["low"]
    upper_shadow=last["high"]-max(last["close"],last["open"])
    lower_shadow=min(last["close"],last["open"])-last["low"]; candle_bonus=1.0

    if lower_shadow>2*abs(body) and body>0: votes.append("hammer_bull"); candle_bonus=1.2
    elif upper_shadow>2*abs(body) and body<0: votes.append("shooting_star"); candle_bonus=1.2
    if abs(body)<0.1*rng: votes.append("doji"); candle_bonus=1.1

    # Pre-top
    pretop=False
    if len(df)>=10 and (last["close"]-df["close"].iloc[-10])/df["close"].iloc[-10]>0.1:
        pretop=True; votes.append("pretop"); confidence+=0.1

    action="WATCH"
    if last["close"]>=last["resistance"]*0.995: action="SHORT"
    elif last["close"]<=last["support"]*1.005: action="LONG"

    confidence*=candle_bonus; confidence=max(0,min(1,confidence))
    return action,votes,pretop,last,confidence

# ---------------- BACKTEST WINRATE ----------------
def backtest_winrate(df: pd.DataFrame):
    df=apply_all_features(df)
    results=[]
    for i in range(1,len(df)):
        sub=df.iloc[:i+1]
        action,votes,pretop,last,conf=detect_signal(sub)
        if action in ["LONG","SHORT"]:
            entry=last["close"]
            support=last["support"]; resistance=last["resistance"]
            win=False; lose=False
            if action=="LONG":
                win=(df["high"].iloc[i:]>=resistance).any()
                lose=(df["low"].iloc[i:]<=entry-0.5*(resistance-entry)).any()
            else:
                win=(df["low"].iloc[i:]<=support).any()
                lose=(df["high"].iloc[i:]>=entry+0.5*(entry-support)).any()
            results.append((action,win,lose))
    total=len(results)
    wins=sum(1 for _,w,l in results if w)
    return wins/total if total>0 else 0, results

# ---------------- PLOT ----------------
def plot_signal(df,symbol,action,votes,pretop):
    df_plot=df.copy()[['open','high','low','close','volume']]
    df_plot.index.name="Date"
    closes=df['close'].values
    peaks,_=find_peaks(closes,distance=5)
    peak_vals=closes[peaks]
    top_resistances=sorted(peak_vals, reverse=True)[:5]
    troughs,_=find_peaks(-closes,distance=5)
    trough_vals=closes[troughs]
    top_supports=sorted(trough_vals)[:5]
    hlines=list(top_supports)+list(top_resistances)
    addplots=[]

    last=df.iloc[-1]
    if pretop:
        ydata=[np.nan]*(len(df)-3)+list(df['close'].iloc[-3:])
        addplots.append(mpf.make_addplot(ydata,type='scatter',markersize=120,marker='^',color='magenta'))

    patterns={"hammer_bull":"lime","shooting_star":"orange","doji":"blue"}
    for pat,color in patterns.items():
        if pat in votes:
            ydata=[np.nan]*(len(df)-1)+[last['close']]
            addplots.append(mpf.make_addplot(ydata,type='scatter',markersize=80,marker='o',color=color))

    mc=mpf.make_marketcolors(up='green',down='red',wick='black',edge='black',volume='blue')
    s=mpf.make_mpf_style(marketcolors=mc,gridstyle='--',gridcolor='gray',facecolor='white')
    buf=io.BytesIO()
    mpf.plot(df_plot,type='candle',style=s,volume=True,addplot=addplots,
             hlines=dict(hlines=hlines,colors=['gray'],linestyle='dashed'),
             title=f"{symbol} — {action} — {','.join(votes)}",
             ylabel='Price',ylabel_lower='Volume',
             savefig=dict(fname=buf,dpi=100,bbox_inches='tight'))
    buf.seek(0)
    return buf

# ---------------- WEBSOCKET ----------------
symbol_dfs={}; lock=threading.Lock()

def on_message(ws,msg):
    data=json.loads(msg); k=data.get("k"); s=data.get("s")
    if not k: return
    candle_closed=k["x"]; open_,high,low,close,vol=float(k["o"]),float(k["h"]),float(k["l"]),float(k["c"]),float(k["v"])
    ts=pd.to_datetime(k["t"],unit="ms",utc=True)
    with lock:
        df=symbol_dfs.get(s,pd.DataFrame(columns=["open","high","low","close","volume"]))
        df.loc[ts]=[open_,high,low,close,vol]; df=df.tail(EMA_SCAN_LIMIT); symbol_dfs[s]=df
    if candle_closed and len(df)>=20:
        df_features=apply_all_features(df)
        action,votes,pretop,last,conf=detect_signal(df_features)
        prev=state["signals"].get(s,"")
        if pretop or (action!="WATCH" and conf>=CONF_THRESHOLD_MEDIUM and action!=prev):
            winrate,_=backtest_winrate(df)
            buf=plot_signal(df,s,action,votes,pretop)
            send_telegram(f"⚡ {s} {action} price={last['close']:.6f} conf={conf:.2f} pretop={pretop} winrate={winrate*100:.1f}%", photo=buf)
            state["signals"][s]=action; state["last_update"]=str(datetime.now(timezone.utc)); save_json_safe(STATE_FILE,state)

def on_error(ws,err): logger.error("WebSocket error: %s",err)
def on_close(ws,cs,cm): logger.warning("WS closed, reconnect in 5s"); time.sleep(5); start_ws(list(symbol_dfs.keys()))
def on_open(ws): logger.info("WebSocket connected")

def start_ws(symbols):
    if not symbols: return
    streams="/".join([f"{s.lower()}@kline_1m" for s in symbols])
    url=f"wss://stream.binance.com:9443/stream?streams={streams}"
    ws=websocket.WebSocketApp(url,on_open=on_open,on_message=on_message,on_error=on_error,on_close=on_close)
    ws.run_forever(ping_interval=20,ping_timeout=10)

# ---------------- FLASK ----------------
flask_app=Flask(__name__)
@flask_app.route("/")
def home(): return jsonify({"status":"ok","time":str(datetime.now(timezone.utc))})
def run_flask(): flask_app.run(host="0.0.0.0", port=PORT)

# ---------------- START BOT ----------------
def start_bot():
    symbols=get_all_usdt_symbols()
    with lock:
        for s in symbols: symbol_dfs[s]=pd.DataFrame(columns=["open","high","low","close","volume"])
    threading.Thread(target=start_ws,args=(symbols,),daemon=True).start()

if __name__=="__main__":
    logger.info("Starting FULL pre-top WebSocket bot + Flask for Render port (SAFE)")
    threading.Thread(target=run_flask, daemon=True).start()
    start_bot()
    while True: time.sleep(1)