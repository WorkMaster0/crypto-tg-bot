# mega_bot_single.py — All-in-One Crypto AI Bot
# Python 3.x
# pip install pandas numpy ta scikit-learn joblib requests flask apscheduler matplotlib python-binance

import os, time, json, logging, threading, random, re
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, jsonify, request
from apscheduler.schedulers.background import BackgroundScheduler
from binance.client import Client
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import requests

# -------------------------
# CONFIG
# -------------------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN","")
CHAT_ID = os.getenv("CHAT_ID","")
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY","")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET","")
STATE_FILE = os.getenv("STATE_FILE","state.json")
EMA_STATS_FILE = os.getenv("EMA_STATS_FILE","ema_stats.json")
MODEL_DIR = os.getenv("MODEL_DIR","models")
PORT = int(os.getenv("PORT","5000"))
SCAN_INTERVAL_MINUTES = int(os.getenv("SCAN_INTERVAL_MINUTES","1"))
MONITOR_INTERVAL_MINUTES = int(os.getenv("MONITOR_INTERVAL_MINUTES","5"))
EMA_SCAN_LIMIT = int(os.getenv("EMA_SCAN_LIMIT","500"))

os.makedirs(MODEL_DIR,exist_ok=True)

# -------------------------
# LOGGING
# -------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger=logging.getLogger(__name__)

# -------------------------
# BINANCE CLIENT
# -------------------------
client=Client(api_key=BINANCE_API_KEY, api_secret=BINANCE_API_SECRET)

# -------------------------
# FLASK
# -------------------------
app=Flask(__name__)

# -------------------------
# STATE
# -------------------------
def load_json_safe(path,default):
    try:
        if os.path.exists(path):
            with open(path,"r") as f:
                return json.load(f)
    except: pass
    return default

def save_json_safe(path,data):
    try:
        tmp=path+".tmp"
        with open(tmp,"w") as f:
            json.dump(data,f,indent=2,default=str)
        os.replace(tmp,path)
    except: pass

state=load_json_safe(STATE_FILE,{"signals":{},"models":{}})
ema_stats=load_json_safe(EMA_STATS_FILE,{})

# -------------------------
# UTILITIES
# -------------------------
def utcnow_str():
    return datetime.now(timezone.utc).isoformat()

def escape_markdown_v2(text:str)->str:
    escape_chars=r'_*[]()~`>#+-=|{}.!'
    return re.sub(r'([%s])'%re.escape(escape_chars),r'\\\1',text)

def send_telegram_message(text):
    if not TELEGRAM_TOKEN or not CHAT_ID: return
    url=f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload={"chat_id":CHAT_ID,"text":escape_markdown_v2(text),
             "parse_mode":"MarkdownV2","disable_web_page_preview":True}
    try:
        requests.post(url,json=payload,timeout=10)
    except: pass

# -------------------------
# MARKET DATA
# -------------------------
def get_top_symbols_by_volume(limit=10):
    try:
        tickers=client.get_ticker()
        sorted_t=sorted(tickers,key=lambda x:float(x.get("quoteVolume",0)),reverse=True)
        return [t["symbol"] for t in sorted_t if t["symbol"].endswith("USDT")][:limit]
    except: return []

def get_klines_df(symbol,interval='5m',limit=EMA_SCAN_LIMIT):
    try:
        kl=client.get_klines(symbol=symbol,interval=interval,limit=limit)
        df=pd.DataFrame(kl,columns=['open_time','open','high','low','close','volume','close_time','qav','num_trades','taker_base_vol','taker_quote_vol','ignore'])
        df=df[['open_time','open','high','low','close','volume']].astype(float)
        df['open_time']=pd.to_datetime(df['open_time'],unit='ms')
        df.set_index('open_time',inplace=True)
        return df
    except: return None

# -------------------------
# INDICATORS
# -------------------------
def safe_apply_basic(df):
    df=df.copy()
    df['ema_5']=df['close'].ewm(span=5,adjust=False).mean()
    df['ema_13']=df['close'].ewm(span=13,adjust=False).mean()
    df['ema_34']=df['close'].ewm(span=34,adjust=False).mean()
    df['ret1']=df['close'].pct_change(1)
    df['vol_ma_20']=df['volume'].rolling(window=20,min_periods=1).mean()
    ha_close=(df['open']+df['high']+df['low']+df['close'])/4
    ha_open=(df['open']+df['close'])/2
    ha_open_vals=[ha_open.iloc[0]]
    for i in range(1,len(df)):
        ha_open_vals.append((ha_open_vals[-1]+ha_close.iloc[i-1])/2)
    df['ha_open']=ha_open_vals
    df['ha_close']=ha_close
    df['ha_green']=df['ha_close']>df['ha_open']
    return df

def detect_ema_cross(df,short='ema_5',long='ema_34'):
    if len(df)<3: return None
    a,b=df[short].iloc[-2],df[long].iloc[-2]
    a1,b1=df[short].iloc[-1],df[long].iloc[-1]
    if a<=b and a1>b1: return 'bull_cross'
    if a>=b and a1<b1: return 'bear_cross'
    return None

def detect_volume_spike(df,mult=1.7):
    if len(df)<5: return False
    vol=df['volume'].iloc[-1]
    ma=df['vol_ma_20'].iloc[-1]
    return vol>ma*mult

# -------------------------
# ML MODEL
# -------------------------
def features_for_ml(df):
    df=safe_apply_basic(df)
    df['vol_change']=df['volume']/(df['vol_ma_20']+1e-9)
    df['ema5_34_diff']=df['ema_5']-df['ema_34']
    return df.dropna()

def train_or_load_model(symbol,df,lookahead=5):
    model_path=os.path.join(MODEL_DIR,f"{symbol}_rf.joblib")
    df2=features_for_ml(df)
    if len(df2)<50: return None,None
    df2['future_ret']=df2['close'].shift(-lookahead)/df2['close']-1
    df2=df2.dropna()
    df2['target']=(df2['future_ret']>0).astype(int)
    features=['ret1','vol_change','ema5_34_diff']
    X,y=df2[features],df2['target']
    if os.path.exists(model_path):
        try: model=joblib.load(model_path); acc=accuracy_score(y,model.predict(X)); return model,acc
        except: os.remove(model_path)
    model=RandomForestClassifier(n_estimators=50,random_state=42)
    model.fit(X,y)
    joblib.dump(model,model_path)
    acc=accuracy_score(y,model.predict(X))
    return model,acc

# -------------------------
# EMA STATS
# -------------------------
def compute_and_save_ema_stats(symbol,interval='5m'):
    df=get_klines_df(symbol,interval)
    if df is None: return None
    df=safe_apply_basic(df)
    key=f"{symbol}_{interval}"
    ema_stats[key]={'bull_cross':detect_ema_cross(df)=='bull_cross','vol_spike':detect_volume_spike(df),'computed_at':utcnow_str()}
    save_json_safe(EMA_STATS_FILE,ema_stats)
    return ema_stats[key]

# -------------------------
# GRAPH
# -------------------------
def plot_symbol(symbol,interval='5m'):
    df=get_klines_df(symbol,interval)
    if df is None: return
    df=safe_apply_basic(df)
    plt.figure(figsize=(10,5))
    plt.plot(df.index,df['close'],label='Close')
    plt.plot(df.index,df['ema_5'],label='EMA5')
    plt.plot(df.index,df['ema_34'],label='EMA34')
    crosses=df[df.apply(lambda r: detect_ema_cross(df.loc[:r.name]),axis=1).notna()]
    for idx,row in crosses.iterrows():
        plt.scatter(idx,row['close'],color='red' if detect_ema_cross(df.loc[:idx])=='bear_cross' else 'green',s=80,marker='^')
    plt.title(symbol)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{symbol}_plot.png")
    plt.close()

# -------------------------
# ANALYZE SYMBOL
# -------------------------
def analyze_symbol(symbol):
    df=get_klines_df(symbol)
    if df is None: return None
    df=safe_apply_basic(df)
    cross=detect_ema_cross(df)
    vol_spike=detect_volume_spike(df)
    model,acc=train_or_load_model(symbol,df)
    plot_symbol(symbol)
    return {'symbol':symbol,'ema_cross':cross,'vol_spike':vol_spike,'ml_acc':acc}

# -------------------------
# SCAN
# -------------------------
def run_manual_scan():
    symbols=get_top_symbols_by_volume(limit=10)
    results=[]
    for s in symbols:
        res=analyze_symbol(s)
        if res: results.append(res)
    send_telegram_message(f"Scan done. Results: {json.dumps(results)}")

# -------------------------
# FLASK ENDPOINTS
# -------------------------
@app.route('/')
def home(): return jsonify({"status":"ok","time":utcnow_str()})

@app.route('/scan_now')
def scan_now():
    threading.Thread(target=run_manual_scan,daemon=True).start()
    return jsonify({"status":"scanning"})

@app.route(f"/telegram_webhook/<token>",methods=["POST"])
def telegram_webhook(token):
    if token!=TELEGRAM_TOKEN: return jsonify({"ok":False,"error":"invalid token"}),403
    data=request.get_json() or {}
    if "message" in data:
        txt=data["message"].get("text","")
        chat=data["message"]["chat"]["id"]
        if str(chat)!=str(CHAT_ID): return jsonify({"ok":True})
        if txt.startswith("/scan_now"):
            threading.Thread(target=run_manual_scan,daemon=True).start()
            send_telegram_message("Manual scan started.")
    return jsonify({"ok":True})

# -------------------------
# WARMUP
# -------------------------
def warmup_models_and_stats():
    symbols=get_top_symbols_by_volume(limit=5)
    for s in symbols:
        df=get_klines_df(s)
        if df is None: continue
        train_or_load_model(s,df)
        compute_and_save_ema_stats(s)

threading.Thread(target=warmup_models_and_stats,daemon=True).start()

# -------------------------
# SCHEDULER
# -------------------------
scheduler=BackgroundScheduler()
scheduler.add_job(run_manual_scan,'interval',minutes=max(1,SCAN_INTERVAL_MINUTES))
scheduler.start()

# -------------------------
# ENTRY
# -------------------------
if __name__=="__main__":
    logger.info("Starting Mega AI Bot")
    app.run(host="0.0.0.0",port=PORT)