# mega_bot.py — All-in-One Crypto AI Bot with EMA, ML, Telegram, Flask, Webhook, and Graphs
# Python 3.x, pip install: pandas numpy ta scikit-learn joblib requests flask apscheduler matplotlib

import os, time, json, logging, threading, traceback, re, random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timezone
from flask import Flask, jsonify, request
from apscheduler.schedulers.background import BackgroundScheduler
from concurrent.futures import ThreadPoolExecutor, as_completed
from binance.client import Client
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import ta
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
PARALLEL_WORKERS = int(os.getenv("PARALLEL_WORKERS","6"))
SCAN_INTERVAL_MINUTES = int(os.getenv("SCAN_INTERVAL_MINUTES","1"))
MONITOR_INTERVAL_MINUTES = int(os.getenv("MONITOR_INTERVAL_MINUTES","5"))
EMA_SCAN_LIMIT = int(os.getenv("EMA_SCAN_LIMIT","500"))
BACKTEST_LOOKAHEAD = int(os.getenv("BACKTEST_LOOKAHEAD","5"))
VOLUME_MULTIPLIER_THRESHOLD = 1.7

os.makedirs(MODEL_DIR,exist_ok=True)

# -------------------------
# LOGGING
# -------------------------
logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.FileHandler("bot.log"),logging.StreamHandler()])
logger=logging.getLogger(__name__)

# -------------------------
# BINANCE CLIENT
# -------------------------
client=Client(api_key=BINANCE_API_KEY,api_secret=BINANCE_API_SECRET)

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
    except:
        logger.exception("Failed to load %s",path)
    return default

def save_json_safe(path,data):
    try:
        tmp=path+".tmp"
        with open(tmp,"w") as f:
            json.dump(data,f,indent=2,default=str)
        os.replace(tmp,path)
    except:
        logger.exception("Failed to save %s",path)

state=load_json_safe(STATE_FILE,{"signals":{},"models":{},"signal_history":{}})
ema_stats=load_json_safe(EMA_STATS_FILE,{})

# -------------------------
# UTILITIES
# -------------------------
def utcnow_str():
    return datetime.now(timezone.utc).isoformat()

def escape_markdown_v2(text:str)->str:
    if not isinstance(text,str): text=str(text)
    escape_chars=r'_*[]()~`>#+-=|{}.!'
    return re.sub(r'([%s])'%re.escape(escape_chars),r'\\\1',text)

def send_telegram_message(text):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        logger.warning("Telegram not configured; skipping message")
        return None
    url=f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload={"chat_id":CHAT_ID,"text":escape_markdown_v2(text),
             "parse_mode":"MarkdownV2","disable_web_page_preview":True}
    try:
        r=requests.post(url,json=payload,timeout=10)
        if r.status_code!=200: logger.error("Telegram send failed: %s %s",r.status_code,r.text)
        return r
    except:
        logger.exception("Telegram send exception")
        return None

# -------------------------
# MARKET DATA HELPERS
# -------------------------
def get_all_usdt_symbols():
    try:
        info=client.get_exchange_info()
        return [s["symbol"] for s in info["symbols"] if s["quoteAsset"]=="USDT" and s["status"]=="TRADING"]
    except:
        logger.exception("get_all_usdt_symbols error")
        return []

def get_top_symbols_by_volume(limit=30):
    try:
        tickers=client.get_ticker()
        sorted_t=sorted(tickers,key=lambda x:float(x.get("quoteVolume",0)),reverse=True)
        return [t["symbol"] for t in sorted_t if t["symbol"].endswith("USDT")][:limit]
    except:
        logger.exception("get_top_symbols_by_volume error")
        return []

def get_klines_df(symbol,interval,limit=500,retry=3):
    for attempt in range(retry):
        try:
            kl=client.get_klines(symbol=symbol,interval=interval,limit=limit)
            df=pd.DataFrame(kl,columns=['open_time','open','high','low','close','volume','close_time','qav','num_trades','taker_base_vol','taker_quote_vol','ignore'])
            df=df[['open_time','open','high','low','close','volume']].copy()
            df[['open','high','low','close','volume']]=df[['open','high','low','close','volume']].astype(float)
            df['open_time']=pd.to_datetime(df['open_time'],unit='ms')
            df.set_index('open_time',inplace=True)
            return df
        except:
            time.sleep(0.3+attempt*0.2)
    return None

# -------------------------
# INDICATORS
# -------------------------
def safe_apply_basic(df):
    df=df.copy()
    if df is None or len(df)==0: return df
    try:
        df['ema_5']=df['close'].ewm(span=5,adjust=False).mean()
        df['ema_13']=df['close'].ewm(span=13,adjust=False).mean()
        df['ema_34']=df['close'].ewm(span=34,adjust=False).mean()
        df['ret1']=df['close'].pct_change(1)
        df['ret3']=df['close'].pct_change(3)
        df['ret10']=df['close'].pct_change(10)
        df['vol_ma_20']=df['volume'].rolling(window=20,min_periods=1).mean()
        df['vol_ma_50']=df['volume'].rolling(window=50,min_periods=1).mean()
        df['ret10_std']=df['ret1'].rolling(window=10,min_periods=1).std()
        df['OBV']=(np.sign(df['ret1']).fillna(0)*df['volume']).cumsum()
        ha_close=(df['open']+df['high']+df['low']+df['close'])/4
        ha_open=(df['open']+df['close'])/2
        ha_open_vals=[ha_open.iloc[0]]
        for i in range(1,len(df)):
            ha_open_vals.append((ha_open_vals[-1]+ha_close.iloc[i-1])/2)
        df['ha_open']=ha_open_vals
        df['ha_close']=ha_close
        df['ha_green']=df['ha_close']>df['ha_open']
    except:
        logger.exception("safe_apply_basic error")
    return df

def detect_ema_cross(df,short='ema_5',long='ema_34'):
    if df is None or len(df)<3: return None
    a=df[short].iloc[-2];b=df[long].iloc[-2]
    a1=df[short].iloc[-1];b1=df[long].iloc[-1]
    if pd.isna(a) or pd.isna(b) or pd.isna(a1) or pd.isna(b1): return None
    if a<=b and a1>b1: return 'bull_cross'
    if a>=b and a1<b1: return 'bear_cross'
    return None

def detect_volume_spike(df,mult=VOLUME_MULTIPLIER_THRESHOLD):
    if df is None or len(df)<5: return False
    vol=df['volume'].iloc[-1]
    ma=df['vol_ma_20'].iloc[-1] if 'vol_ma_20' in df.columns else np.nan
    if pd.isna(ma) or ma==0: return False
    return vol>ma*mult

# -------------------------
# MODEL HELPERS
# -------------------------
def features_for_ml(df):
    df=df.copy()
    df=safe_apply_basic(df)
    df['vol_change']=df['volume']/(df['vol_ma_20']+1e-9)
    df['ema5_34_diff']=df['ema_5']-df['ema_34']
    return df.dropna()

def safe_joblib_dump(model,path):
    tmp=path+".tmp"
    try:
        joblib.dump(model,tmp)
        os.replace(tmp,path)
        return True
    except:
        logger.exception("safe_joblib_dump failed for %s",path)
        return False

def train_or_load_model(symbol,df,lookahead=BACKTEST_LOOKAHEAD):
    model_path=os.path.join(MODEL_DIR,f"{symbol}_rf.joblib")
    df2=features_for_ml(df)
    if len(df2)<300: return None,None
    df2['future_ret']=df2['close'].shift(-lookahead)/df2['close']-1
    df2=df2.dropna()
    df2['target']=(df2['future_ret']>0).astype(int)
    features=['ret1','ret3','ret10','vol_change','ema5_34_diff','ret10_std']
    X=df2[features];y=df2['target']
    if os.path.exists(model_path) and os.path.getsize(model_path)>0:
        try:
            model=joblib.load(model_path)
            X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,shuffle=False)
            acc=accuracy_score(y_test,model.predict(X_test))
            return model,float(acc)
        except:
            os.remove(model_path)
    try:
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,shuffle=False)
        model=RandomForestClassifier(n_estimators=150,n_jobs=1,random_state=42)
        model.fit(X_train,y_train)
        acc=accuracy_score(y_test,model.predict(X_test))
        safe_joblib_dump(model,model_path)
        return model,float(acc)
    except:
        return None,None

# -------------------------
# EMA STATS
# -------------------------
def compute_ema_historical_stats(df,short_col='ema_5',long_col='ema_34',lookahead=BACKTEST_LOOKAHEAD):
    df=safe_apply_basic(df)
    res={'golden':{'returns':[]},'death':{'returns':[]}}
    for i in range(1,len(df)-lookahead):
        prev_s=df[short_col].iloc[i-1];prev_l=df[long_col].iloc[i-1]
        cur_s=df[short_col].iloc[i];cur_l=df[long_col].iloc[i]
        if pd.isna(prev_s) or pd.isna(prev_l) or pd.isna(cur_s) or pd.isna(cur_l): continue
        if prev_s<=prev_l and cur_s>cur_l: res['golden']['returns'].append(df['close'].iloc[i+lookahead]/df['close'].iloc[i]-1)
        if prev_s>=prev_l and cur_s<cur_l: res['death']['returns'].append(df['close'].iloc[i+lookahead]/df['close'].iloc[i]-1)
    out={}
    for k in ['golden','death']:
        r=res[k]['returns']
        if r:
            arr=np.array(r)
            out[k]={'count':int(len(r)),'win_rate':float((arr>0).mean()),'avg_return':float(arr.mean())}
        else: out[k]={'count':0,'win_rate':None,'avg_return':None}
    return out

def compute_and_save_ema_stats(symbol,interval):
    try:
        df=get_klines_df(symbol,interval,limit=EMA_SCAN_LIMIT)
        if df is None or len(df)<50: return None
        stats=compute_ema_historical_stats(df)
        key=f"{symbol}_{interval}"
        ema_stats[key]={'computed_at':utcnow_str(),'lookahead':BACKTEST_LOOKAHEAD,'stats':stats}
        save_json_safe(EMA_STATS_FILE,ema_stats)
        return ema_stats[key]
    except:
        logger.exception("compute_and_save_ema_stats error %s %s",symbol,interval)
        return None

# -------------------------
# ANALYZE SYMBOL
# -------------------------
def analyze_symbol(symbol,interval='5m'):
    try:
        df=get_klines_df(symbol,interval,limit=EMA_SCAN_LIMIT)
        if df is None or len(df)<30: return None
        df=safe_apply_basic(df)
        cross=detect_ema_cross(df)
        vol_spike=detect_volume_spike(df)
        model,acc=train_or_load_model(symbol,df)
        return {'symbol':symbol,'interval':interval,'ema_cross':cross,'vol_spike':vol_spike,'ml_acc':acc}
    except:
        logger.exception("analyze_symbol error %s",symbol)
        return None

# -------------------------
# SCAN FUNCTIONS
# -------------------------
def run_manual_scan():
    symbols=get_top_symbols_by_volume(limit=20)
    for s in symbols: compute_and_save_ema_stats(s,'5m')

def ema_scan_all(interval='5m'):
    symbols=get_top_symbols_by_volume(limit=30)
    for s in symbols: compute_and_save_ema_stats(s,interval)

def monitor_top_symbols():
    symbols=get_top_symbols_by_volume(limit=20)
    for s in symbols: compute_and_save_ema_stats(s,'1h')

# -------------------------
# FLASK ENDPOINTS
# -------------------------
@app.route('/')
def home():
    return jsonify({"status":"ok","time":utcnow_str()})

@app.route('/scan_now')
def scan_now():
    threading.Thread(target=run_manual_scan,daemon=True).start()
    return jsonify({"status":"scanning"})

@app.route('/status')
def status():
    return jsonify({"state_len":len(state.get('signals',{})),"ema_stats_count":len(ema_stats),"time":utcnow_str()})

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
        elif txt.startswith("/status"):
            st={"signals":len(state.get("signals",{})),"ema_stats":len(ema_stats)}
            send_telegram_message(f"Status: {json.dumps(st)}")
    return jsonify({"ok":True})

# -------------------------
# WARMUP
# -------------------------
def warmup_models_and_stats():
    logger.info("Warmup models & ema stats")
    symbols=get_top_symbols_by_volume(limit=30)
    for s in symbols:
        try:
            df=get_klines_df(s,'1h',limit=800)
            if df is None: continue
            train_or_load_model(s,df)
            compute_and_save_ema_stats(s,'5m')
            compute_and_save_ema_stats(s,'1h')
            time.sleep(0.6+random.random()*0.4)
        except:
            logger.exception("warmup error for %s",s)

threading.Thread(target=warmup_models_and_stats,daemon=True).start()

# -------------------------
# SCHEDULER
# -------------------------
scheduler=BackgroundScheduler()
scheduler.add_job(lambda: ema_scan_all(interval='5m'),'interval',minutes=max(1,SCAN_INTERVAL_MINUTES))
scheduler.add_job(monitor_top_symbols,'interval',minutes=max(1,MONITOR_INTERVAL_MINUTES))
scheduler.start()

# -------------------------
# ENTRY
# -------------------------
if __name__=="__main__":
    logger.info("Starting AI Web3 Trading Hub")
    app.run(host="0.0.0.0",port=PORT)