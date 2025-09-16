# smc_web3_bot.py — multi-frame SMC bot with ML + pre-top detect + entry points + Telegram + charts
# Requires: python-binance, pandas, numpy, matplotlib, scikit-learn, joblib, flask, apscheduler, requests
# Env: BINANCE_API_KEY, BINANCE_API_SECRET, TELEGRAM_TOKEN, CHAT_ID

import os, time, json, random, logging, traceback, math, re
from datetime import datetime, timezone
from threading import Thread
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, jsonify
from apscheduler.schedulers.background import BackgroundScheduler
from binance.client import Client
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import requests

# ---------------- CONFIG ----------------
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
CHAT_ID = os.getenv("CHAT_ID", "")
PORT = int(os.getenv("PORT", "5000"))

PARALLEL_WORKERS = int(os.getenv("PARALLEL_WORKERS", "6"))
SCAN_INTERVAL_MINUTES = int(os.getenv("SCAN_INTERVAL_MINUTES", "1"))
MONITOR_INTERVAL_MINUTES = int(os.getenv("MONITOR_INTERVAL_MINUTES", "5"))

STATE_FILE = "state.json"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

PUMP_PCT_THRESHOLD = 0.08
LIQUIDITY_SWEEP_TAIL_PCT = 0.008
WEIGHTS = {"order_block":0.25,"fvg":0.15,"liquidity_sweep":0.15,"pre_top_pump":0.20,"ml":0.20}
CONF_THRESHOLD_ALERT = 0.45
CONF_THRESHOLD_WATCH = 0.20

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.FileHandler("bot.log"), logging.StreamHandler()])
logger = logging.getLogger("smc-bot")

# ---------------- Binance ----------------
client = Client(api_key=BINANCE_API_KEY, api_secret=BINANCE_API_SECRET)

# ---------------- Flask ----------------
app = Flask(__name__)

# ---------------- State ----------------
def load_json_safe(path, default):
    try:
        if os.path.exists(path):
            with open(path,"r") as f:
                return json.load(f)
    except: pass
    return default

def save_json_safe(path, data):
    try:
        tmp = path + ".tmp"
        with open(tmp,"w") as f:
            json.dump(data,f,indent=2,default=str)
        os.replace(tmp,path)
    except: pass

state = load_json_safe(STATE_FILE, {"signals":{}, "models":{}, "history":{}})

# ---------------- Telegram ----------------
MDV2_ESCAPE = r"_*[]()~`>#+-=|{}.!"
def escape_markdown_v2(s): return re.sub("([" + re.escape(MDV2_ESCAPE) + "])", r"\\\1", str(s)) if s else ""
def send_telegram_message(text, img_path=None):
    if not TELEGRAM_TOKEN or not CHAT_ID: return
    try:
        payload = {"chat_id": CHAT_ID,"text":escape_markdown_v2(text),"parse_mode":"MarkdownV2","disable_web_page_preview":True}
        if img_path:
            files = {"photo": open(img_path,"rb")}
            r = requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto", data={"chat_id":CHAT_ID,"caption":text}, files=files)
            files["photo"].close()
        else:
            r = requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", json=payload)
        if r.status_code != 200:
            logger.error("Telegram send failed: %s %s", r.status_code, r.text)
    except Exception as e:
        logger.exception("Telegram send exception: %s", e)

# ---------------- Market Data ----------------
def get_all_usdt_symbols():
    try:
        info = client.get_exchange_info()
        return [s['symbol'] for s in info['symbols'] if s['quoteAsset']=="USDT" and s['status']=="TRADING"]
    except:
        return []

def get_klines_df(symbol, interval, limit=500, retry=3):
    for attempt in range(retry):
        try:
            kl = client.get_klines(symbol=symbol, interval=interval, limit=limit)
            df = pd.DataFrame(kl, columns=['open_time','open','high','low','close','volume','close_time','qav','num_trades','taker_base_vol','taker_quote_vol','ignore'])
            df[['open','high','low','close','volume']] = df[['open','high','low','close','volume']].astype(float)
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            df.set_index('open_time', inplace=True)
            return df[['open','high','low','close','volume']]
        except Exception as e:
            time.sleep(0.5 + attempt)
    return None

# ---------------- SMC Detectors ----------------
def detect_order_blocks(df, lookback=60):
    if df is None or len(df)<lookback+5: return None
    rng = (df['high']-df['low']).rolling(lookback).mean()
    recent_rng = rng.iloc[-1]
    last_segment = df.iloc[-lookback:]
    hist_mean = rng.mean() if not math.isnan(rng.mean()) else recent_rng
    seg_range = last_segment['high'].max()-last_segment['low'].min()
    if seg_range<hist_mean*0.6:
        return {"type":"order_block","top":float(last_segment['high'].max()),"bot":float(last_segment['low'].min()),"range":float(seg_range)}
    return None

def detect_fvg(df):
    if df is None or len(df)<6: return None
    for i in range(len(df)-3,0,-1):
        a,b,c = df.iloc[i-1], df.iloc[i], df.iloc[i+1]
        b_low,b_high = min(b['open'],b['close']), max(b['open'],b['close'])
        a_high,c_low = max(a['open'],a['close']), min(c['open'],c['close'])
        if b_low>a_high and c_low>b_high: return {'type':'fvg_up','low':float(a_high),'high':float(b_low)}
        b_high,a_low,c_high = max(b['open'],b['close']), min(a['open'],a['close']), max(c['open'],c['close'])
        if b_high<a_low and c_high<b_low: return {'type':'fvg_down','low':float(b_high),'high':float(a_low)}
    return None

def detect_liquidity_sweep(df):
    if df is None or len(df)<3: return None
    last,prev = df.iloc[-1],df.iloc[-2]
    price = last['close']
    wick_up = last['high']-max(last['open'],last['close'])
    wick_down = min(last['open'],last['close'])-last['low']
    vol_spike = last['volume']>(df['volume'].rolling(20).mean().iloc[-1]*1.8 if len(df)>=20 else last['volume']*2)
    if vol_spike and wick_up>price*LIQUIDITY_SWEEP_TAIL_PCT: return {'type':'sweep_up','size':float(wick_up)}
    if vol_spike and wick_down>price*LIQUIDITY_SWEEP_TAIL_PCT: return {'type':'sweep_down','size':float(wick_down)}
    return None

def detect_pre_top_pump(df):
    if df is None or len(df)<6: return None
    recent = df.iloc[-6:]
    start,end = recent['close'].iloc[0],recent['close'].iloc[-1]
    pct_move = (end/start)-1
    vol_ratio = recent['volume'].iloc[-1]/(recent['volume'].rolling(6).mean().iloc[-1]+1e-9)
    if pct_move>=PUMP_PCT_THRESHOLD and vol_ratio>=2.0: return {'type':'pump','pct':float(pct_move),'vol_ratio':float(vol_ratio)}
    return None

def detect_market_structure_shift(df):
    if df is None or len(df)<10: return None
    last_close = df['close'].iloc[-1]
    prior_high,prior_low = df['high'].iloc[-10:-2].max(), df['low'].iloc[-10:-2].min()
    if last_close>prior_high: return 'BOS_UP'
    if last_close<prior_low: return 'BOS_DOWN'
    return None

# ---------------- ML ----------------
def build_features_from_df(df):
    df = df.copy()
    df['range']=df['high']-df['low']
    df['body']=abs(df['close']-df['open'])
    df['close_o']=df['close']-df['open']
    df['p_change_1']=df['close'].pct_change(1)
    df['vol_ma_20']=df['volume'].rolling(20).mean()
    df['vol_spike']=(df['volume']/(df['vol_ma_20']+1e-9)).fillna(1.0)
    return df.dropna()

def build_training_dataset(df, lookahead=5):
    df2=build_features_from_df(df)
    if len(df2)<50+lookahead: return None,None
    df2['future_ret']=df2['close'].shift(-lookahead)/df2['close']-1
    df2['target']=(df2['future_ret']>0).astype(int)
    features=['range','body','close_o','p_change_1','vol_spike']
    X,y=df2[features],df2['target']
    return X,y

def train_or_load_model(symbol, df, lookahead=5):
    model_path=os.path.join(MODEL_DIR,f"{symbol}_rf.joblib")
    try:
        if os.path.exists(model_path) and os.path.getsize(model_path)>0:
            model=joblib.load(model_path)
            acc=state.get('models',{}).get(f"{symbol}_meta",{}).get('acc',0.0)
            return model,acc
    except: pass
    X,y=build_training_dataset(df,lookahead)
    if X is None or len(X)<50: return None,None
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,shuffle=False)
    model=RandomForestClassifier(n_estimators=200,n_jobs=1,random_state=42)
    model.fit(X_train,y_train)
    acc=float(accuracy_score(y_test,model.predict(X_test)))
    joblib.dump(model,model_path)
    state.setdefault('models',{})[f"{symbol}_meta"]={'acc':acc,'trained_at':str(datetime.now(timezone.utc))}
    save_json_safe(STATE_FILE,state)
    return model,acc

# ---------------- Combine Evidence ----------------
def combine_evidence(evidence, ml_prob, price):
    score_long,score_short=0.0,0.0
    ob=evidence.get('order_block')
    if ob:
        dist_top=abs(price-ob['top'])/(price+1e-9)
        dist_bot=abs(price-ob['bot'])/(price+1e-9)
        score_short+=WEIGHTS['order_block']*max(0,1-dist_top*10)
        score_long+=WEIGHTS['order_block']*max(0,1-dist_bot*10)
    fvg=evidence.get('fvg')
    if fvg:
        if fvg['type']=='fvg_up': score_long+=WEIGHTS['fvg']
        elif fvg['type']=='fvg_down': score_short+=WEIGHTS['fvg']
    sweep=evidence.get('sweep')
    if sweep:
        if sweep['type']=='sweep_up': score_short+=WEIGHTS['liquidity_sweep']
        elif sweep['type']=='sweep_down': score_long+=WEIGHTS['liquidity_sweep']
    pump=evidence.get('pump')
    if pump: score_short+=WEIGHTS['pre_top_pump']
    mss=evidence.get('mss')
    if mss=='BOS_UP': score_long+=0.08
    if mss=='BOS_DOWN': score_short+=0.08
    if ml_prob is not None:
        score_long+=WEIGHTS['ml']*ml_prob
        score_short+=WEIGHTS['ml']*(1-ml_prob)
    total=score_long+score_short+1e-9
    conf=max(score_long,score_short)/total
    direction='LONG' if score_long>score_short and conf>0.01 else ('SHORT' if score_short>score_long and conf>0.01 else None)
    strength=min(1.0,max(score_long,score_short))
    return float(strength),direction

# ---------------- Analyze ----------------
def analyze_symbol(symbol,interval='5m'):
    df=get_klines_df(symbol,interval,limit=600)
    if df is None or len(df)<12: return None
    price=float(df['close'].iloc[-1])
    ob, fvg, sweep, pump, mss = detect_order_blocks(df), detect_fvg(df), detect_liquidity_sweep(df), detect_pre_top_pump(df), detect_market_structure_shift(df)
    evidence={'order_block':ob,'fvg':fvg,'sweep':sweep,'pump':pump,'mss':mss,'price':price}
    model,ml_acc=train_or_load_model(symbol,df)
    ml_prob=None
    if model:
        feat_df=build_features_from_df(df)
        if len(feat_df)>0:
            lastf=feat_df.iloc[-1:][['range','body','close_o','p_change_1','vol_spike']]
            try: ml_prob=float(model.predict_proba(lastf)[0][1])
            except: ml_prob=None
    conf,direction=combine_evidence(evidence,ml_prob,price)
    label="WATCH"
    if conf>=0.65: label="STRONG"
    elif conf>=0.45: label="MEDIUM"
    elif conf>=0.20: label="WEAK"
    out={"symbol":symbol,"interval":interval,"price":price,"evidence":evidence,"ml_prob":ml_prob,"ml_acc":ml_acc,"confidence":conf,"direction":direction,"label":label,"timestamp":str(df.index[-1])}
    return out

# ---------------- Scan ----------------
def ema_scan_all(interval='5m'):
    symbols=get_all_usdt_symbols()
    if not symbols: return
    results=[]
    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as exe:
        futures={exe.submit(analyze_symbol,s,interval):s for s in symbols}
        for fut in as_completed(futures):
            sym=futures[fut]
            try:
                res=fut.result()
                if res and res['confidence']>=CONF_THRESHOLD_WATCH:
                    results.append(res)
                    # Generate chart
                    df=get_klines_df(sym,interval,limit=100)
                    if df is not None:
                        plt.figure(figsize=(10,5))
                        plt.plot(df['close'],label='Close')
                        plt.scatter(df.index[-1],df['close'].iloc[-1],color='red',label='Current')
                        plt.title(f"{sym} {interval}")
                        chart_path=f"{sym}_{interval}.png"
                        plt.savefig(chart_path)
                        plt.close()
                        send_telegram_message(f"SMC Alert {sym} {interval} Direction: {res['direction']} Conf: {res['confidence']:.2f}",chart_path)
            except Exception as e:
                logger.exception("Scan future error %s: %s",sym,e)

# ---------------- Monitor ----------------
def monitor_top_symbols():
    symbols=[s for s in get_all_usdt_symbols()][:30]
    interval_list=['5m','15m','1h','4h']
    results=[]
    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as exe:
        futures=[]
        for s in symbols:
            for intr in interval_list: futures.append(exe.submit(analyze_symbol,s,intr))
        for fut in as_completed(futures):
            try:
                res=fut.result()
                if not res: continue
                direction=res['direction']
                conf=res['confidence']
                if direction and conf>=CONF_THRESHOLD_ALERT:
                    key=f"{res['symbol']}_{res['interval']}"
                    prev=state.get('signals',{}).get(key)
                    if prev!=direction:
                        send_telegram_message(f"⚡ Signal {res['symbol']} {res['interval']} Direction: {direction} Conf: {conf:.2f}")
                        state.setdefault('signals',{})[key]=direction
                        hist=state.setdefault('history',{}).setdefault(res['symbol'],[])
                        hist.append({'time':res['timestamp'],'interval':res['interval'],'signal':direction,'conf':conf})
                        if len(hist)>400: state['history'][res['symbol']]=hist[-400:]
                        save_json_safe(STATE_FILE,state)
                results.append(res)
            except Exception as e: logger.exception("Monitor future error: %s", e)

# ---------------- Scheduler ----------------
scheduler=BackgroundScheduler()
scheduler.add_job(lambda: ema_scan_all('5m'),'interval',minutes=max(1,SCAN_INTERVAL_MINUTES),id='smc_scan')
scheduler.add_job(monitor_top_symbols,'interval',minutes=max(1,MONITOR_INTERVAL_MINUTES),id='monitor_top')
scheduler.start()

# ---------------- Flask Routes ----------------
@app.route('/')
def home(): return jsonify({"status":"ok","time":str(datetime.now(timezone.utc))})
@app.route('/scan_now')
def scan_now():
    try: ema_scan_all('5m'); monitor_top_symbols(); return jsonify({"status":"scanned"})
    except Exception as e: return jsonify({"error":str(e)}),500
@app.route('/status')
def status(): return jsonify({"state":state,"time":str(datetime.now(timezone.utc))})

# ---------------- Warmup ----------------
def warmup():
    syms=[s for s in get_all_usdt_symbols()][:10]
    for s in syms:
        try:
            df=get_klines_df(s,'1h',limit=200)
            if df: train_or_load_model(s,df)
            time.sleep(0.5 + random.random()*0.5)
        except: pass
Thread(target=warmup,daemon=True).start()

# ---------------- Main ----------------
if __name__=="__main__":
    logger.info("Starting SMC Web3 bot")
    try: ema_scan_all('5m')
    except: pass
    app.run(host='0.0.0.0', port=PORT)