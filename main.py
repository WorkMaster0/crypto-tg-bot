# smc_bot_all_in_one.py
# Full SMC + ML + Binance + Gate + Telegram bot in one file
# Requirements: python-binance, ccxt, pandas, numpy, scikit-learn, joblib, flask, requests, apscheduler

import os, time, json, math, random, logging, re
from datetime import datetime, timezone
from threading import Thread
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np
import requests
import ccxt
from flask import Flask, jsonify
from apscheduler.schedulers.background import BackgroundScheduler
from binance.client import Client
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# -------------------------
# CONFIG / ENV
# -------------------------
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
GATE_API_KEY = os.getenv("GATE_API_KEY")
GATE_API_SECRET = os.getenv("GATE_API_SECRET")
MORALIS_API_KEY = os.getenv("MORALIS_API_KEY")

PORT = int(os.getenv("PORT", "5000"))
PARALLEL_WORKERS = int(os.getenv("PARALLEL_WORKERS", "6"))
SCAN_INTERVAL_MINUTES = int(os.getenv("SCAN_INTERVAL_MINUTES", "1"))
MONITOR_INTERVAL_MINUTES = int(os.getenv("MONITOR_INTERVAL_MINUTES", "5"))
MIN_ROWS_FOR_ML = 300
CONF_THRESHOLD_ALERT = 0.45
CONF_THRESHOLD_WATCH = 0.2
PUMP_PCT_THRESHOLD = 0.08
LIQUIDITY_SWEEP_TAIL_PCT = 0.008

MODEL_DIR = "models"
STATE_FILE = "state.json"
os.makedirs(MODEL_DIR, exist_ok=True)

WEIGHTS = {"order_block":0.25,"fvg":0.15,"liquidity_sweep":0.15,"pre_top_pump":0.2,"ml":0.2}

# -------------------------
# LOGGING
# -------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("smc-bot")

# -------------------------
# CLIENTS
# -------------------------
binance = Client(BINANCE_API_KEY, BINANCE_API_SECRET)
gate = ccxt.gateio({"apiKey": GATE_API_KEY, "secret": GATE_API_SECRET, "options":{"defaultType":"swap"}})

app = Flask(__name__)
state = {"signals": {}, "models": {}, "history": {}}

# -------------------------
# TELEGRAM HELPERS
# -------------------------
MDV2_ESCAPE = r"_*[]()~`>#+-=|{}.!"
def escape_md2(s): return re.sub("([" + re.escape(MDV2_ESCAPE) + "])", r"\\\1", str(s))
def send_telegram(msg):
    if not TELEGRAM_TOKEN or not CHAT_ID: return False
    try:
        r = requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                          json={"chat_id":CHAT_ID,"text":escape_md2(msg),"parse_mode":"MarkdownV2"})
        return r.status_code==200
    except Exception as e: logger.exception("Telegram send error: %s", e); return False

# -------------------------
# BINANCE / MARKET DATA
# -------------------------
def get_all_usdt_symbols():
    try: info = binance.get_exchange_info(); return [s['symbol'] for s in info['symbols'] if s['quoteAsset']=='USDT' and s['status']=='TRADING']
    except Exception as e: logger.exception(e); return []

def get_klines_df(symbol, interval, limit=500, retry=3):
    for i in range(retry):
        try:
            kl = binance.get_klines(symbol=symbol, interval=interval, limit=limit)
            df = pd.DataFrame(kl, columns=['open_time','open','high','low','close','volume','close_time','qav','num_trades','taker_base_vol','taker_quote_vol','ignore'])
            df = df[['open_time','open','high','low','close','volume']].astype(float)
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms'); df.set_index('open_time', inplace=True)
            return df
        except: time.sleep(0.5+random.random())
    return None

# -------------------------
# SMC DETECTORS
# -------------------------
def detect_order_blocks(df, lookback=60):
    if df is None or len(df)<lookback+5: return None
    rng = (df['high']-df['low']).rolling(lookback).mean(); recent_rng = rng.iloc[-1]; last_seg=df.iloc[-lookback:]
    hist_mean = rng.mean() if not math.isnan(rng.mean()) else recent_rng
    seg_range = last_seg['high'].max()-last_seg['low'].min()
    if seg_range<hist_mean*0.6: return {'top':float(last_seg['high'].max()),'bot':float(last_seg['low'].min())}
    return None

def detect_pre_top_pump(df):
    if df is None or len(df)<6: return None
    recent=df.iloc[-6:]; pct_move=(recent['close'].iloc[-1]/recent['close'].iloc[0])-1
    vol_ratio=recent['volume'].iloc[-1]/(recent['volume'].rolling(6).mean().iloc[-1]+1e-9)
    if pct_move>=PUMP_PCT_THRESHOLD and vol_ratio>=2.0: return {'pct':float(pct_move),'vol_ratio':float(vol_ratio)}
    return None

# -------------------------
# ML HELPERS
# -------------------------
def build_features(df):
    df = df.copy(); df['range']=df['high']-df['low']; df['body']=abs(df['close']-df['open']); df['close_o']=df['close']-df['open']
    df['p_change_1']=df['close'].pct_change(1); df['vol_ma_20']=df['volume'].rolling(20).mean(); df['vol_spike']=(df['volume']/(df['vol_ma_20']+1e-9)).fillna(1.0); return df.dropna()
def build_dataset(df, lookahead=5):
    df2=build_features(df); df2['future_ret']=df2['close'].shift(-lookahead)/df2['close']-1; df2=df2.dropna(); df2['target']=(df2['future_ret']>0).astype(int)
    features=['range','body','close_o','p_change_1','vol_spike']; return df2[features], df2['target']

def train_or_load_model(symbol, df):
    path=os.path.join(MODEL_DIR,f"{symbol}_rf.joblib")
    try: 
        if os.path.exists(path) and os.path.getsize(path)>0: return joblib.load(path), 0.8
    except: pass
    X,y=build_dataset(df); 
    if len(X)<MIN_ROWS_FOR_ML: return None,None
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,shuffle=False)
    model=RandomForestClassifier(n_estimators=200,n_jobs=1,random_state=42); model.fit(X_train,y_train)
    joblib.dump(model,path); return model,float(accuracy_score(y_test,model.predict(X_test)))

# -------------------------
# ANALYZE SYMBOL
# -------------------------
def analyze_symbol(symbol, interval='5m'):
    df=get_klines_df(symbol, interval, limit=600)
    if df is None or len(df)<12: return None
    price=float(df['close'].iloc[-1])
    ob=detect_order_blocks(df); pump=detect_pre_top_pump(df)
    model, ml_acc=train_or_load_model(symbol, df)
    ml_prob = None
    if model is not None:
        lastf = build_features(df).iloc[-1:][['range','body','close_o','p_change_1','vol_spike']]
        ml_prob = float(model.predict_proba(lastf)[0][1])
    conf = 0.0; direction=None
    if ob: conf+=WEIGHTS['order_block']; direction='LONG'
    if pump: conf+=WEIGHTS['pre_top_pump']; direction='SHORT'
    if ml_prob is not None: conf+=WEIGHTS['ml']; direction='LONG' if ml_prob>0.5 else 'SHORT'
    return {'symbol':symbol,'price':price,'confidence':conf,'direction':direction,'ml_prob':ml_prob}

# -------------------------
# SCAN WORKERS
# -------------------------
def scan_all(interval='5m'):
    syms=get_all_usdt_symbols()
    results=[]
    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as exe:
        futures={exe.submit(analyze_symbol,s,interval):s for s in syms}
        for fut in as_completed(futures):
            try:
                res=fut.result(); 
                if res and res['confidence']>=CONF_THRESHOLD_WATCH: results.append(res)
            except: pass
    for r in sorted(results,key=lambda x:x['confidence'],reverse=True)[:50]:
        send_telegram(f"SMC Alert {r['symbol']} Price:{r['price']:.4f} Conf:{r['confidence']:.2f} Dir:{r['direction']} ML:{r['ml_prob']}")
    return results

# -------------------------
# SCHEDULER
# -------------------------
scheduler=BackgroundScheduler()
scheduler.add_job(lambda: scan_all(interval='5m'),'interval',minutes=max(1,SCAN_INTERVAL_MINUTES))
scheduler.start()

# -------------------------
# FLASK ENDPOINTS
# -------------------------
@app.route('/')
def home(): return jsonify({"status":"ok","time":str(datetime.now(timezone.utc))})
@app.route('/scan_now')
def scan_now(): return jsonify({"results":scan_all()})

# -------------------------
# WARMUP / BACKGROUND
# -------------------------
def warmup(): 
    logger.info("Warmup started"); syms=get_all_usdt_symbols()[:20]
    for s in syms: df=get_klines_df(s,'1h',limit=600); train_or_load_model(s,df); time.sleep(0.2)
Thread(target=warmup,daemon=True).start()

# -------------------------
# MAIN
# -------------------------
if __name__=="__main__":
    logger.info("Starting SMC Bot All-In-One")
    scan_all()
    app.run(host='0.0.0.0',port=PORT)