# smc_market_pulse_bot.py
# Python 1.13.4
# Binance + Telegram + SMC + ML + Multi-timeframe + Market Pulse

import os, time, json, math, random, logging, traceback, re
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Thread

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, jsonify
from apscheduler.schedulers.background import BackgroundScheduler
from binance.client import Client
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import requests

# -------------------------
# CONFIG
# -------------------------
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
LOG_FILE = "bot.log"

os.makedirs(MODEL_DIR, exist_ok=True)

MIN_ROWS_FOR_ML = 300
CONF_THRESHOLD_ALERT = 0.45
CONF_THRESHOLD_WATCH = 0.20
PUMP_PCT_THRESHOLD = 0.08
LIQUIDITY_SWEEP_TAIL_PCT = 0.008

WEIGHTS = {
    "order_block": 0.25,
    "fvg": 0.15,
    "liquidity_sweep": 0.15,
    "pre_top_pump": 0.20,
    "ml": 0.20
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)
logger = logging.getLogger("smc-bot")

client = Client(api_key=BINANCE_API_KEY, api_secret=BINANCE_API_SECRET)
app = Flask(__name__)

# -------------------------
# STATE / IO
# -------------------------
def load_json_safe(path, default):
    try:
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
    except Exception as e:
        logger.exception("load_json_safe failed: %s", e)
    return default

def save_json_safe(path, data):
    try:
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp, path)
    except Exception as e:
        logger.exception("save_json_safe failed: %s", e)

state = load_json_safe(STATE_FILE, {"signals": {}, "models": {}, "history": {}})

# -------------------------
# TELEGRAM
# -------------------------
MDV2_ESCAPE = r"_*[]()~`>#+-=|{}.!"

def escape_markdown_v2(s: str) -> str:
    if s is None: return ""
    return re.sub("([" + re.escape(MDV2_ESCAPE) + "])", r"\\\1", str(s))

def send_telegram_message(text: str):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        logger.warning("Telegram not configured; skipping send.")
        return False
    try:
        payload = {"chat_id": CHAT_ID,"text": escape_markdown_v2(text),"parse_mode":"MarkdownV2","disable_web_page_preview":True}
        r = requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", json=payload, timeout=10)
        if r.status_code != 200:
            logger.error("Telegram send failed: %s %s", r.status_code, r.text)
            return False
        return True
    except Exception as e:
        logger.exception("Telegram send exception: %s", e)
        return False

# -------------------------
# BINANCE DATA
# -------------------------
def get_all_usdt_symbols():
    try:
        info = client.get_exchange_info()
        return [s['symbol'] for s in info['symbols'] if s['quoteAsset']=='USDT' and s['status']=='TRADING']
    except Exception as e:
        logger.exception("get_all_usdt_symbols error: %s", e)
        return []

def get_klines_df(symbol, interval, limit=500, retry=3):
    for attempt in range(retry):
        try:
            kl = client.get_klines(symbol=symbol, interval=interval, limit=limit)
            if not kl: return None
            df = pd.DataFrame(kl, columns=['open_time','open','high','low','close','volume','close_time','qav','num_trades','taker_base_vol','taker_quote_vol','ignore'])
            df = df[['open_time','open','high','low','close','volume']].astype(float)
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            df.set_index('open_time', inplace=True)
            return df
        except Exception as e:
            logger.warning("get_klines_df %s %s attempt %d/%d error: %s", symbol, interval, attempt+1, retry, e)
            time.sleep(0.5 + attempt)
    return None

# -------------------------
# SMC DETECTORS
# -------------------------
def detect_order_blocks(df, lookback=60):
    if df is None or len(df)<lookback+5: return None
    rng = (df['high']-df['low']).rolling(lookback).mean()
    recent_rng = rng.iloc[-1]
    last_seg = df.iloc[-lookback:]
    hist_mean = rng.mean() if not math.isnan(rng.mean()) else recent_rng
    if hist_mean==0 or math.isnan(hist_mean): return None
    seg_range = last_seg['high'].max() - last_seg['low'].min()
    if seg_range < hist_mean*0.6:
        return {'type':'order_block','top':float(last_seg['high'].max()), 'bot':float(last_seg['low'].min()), 'range':float(seg_range)}
    return None

def detect_fair_value_gap(df):
    if df is None or len(df)<6: return None
    for i in range(len(df)-3,0,-1):
        a,b,c = df.iloc[i-1],df.iloc[i],df.iloc[i+1]
        b_low,b_high = min(b['open'],b['close']), max(b['open'],b['close'])
        a_high,c_low = max(a['open'],a['close']), min(c['open'],c['close'])
        # bullish
        if b_low>a_high and c_low>b_high: return {'type':'fvg_up','low':float(a_high),'high':float(b_low)}
        # bearish
        b_high,a_low,c_high = max(b['open'],b['close']), min(a['open'],a['close']), max(c['open'],c['close'])
        if b_high<a_low and c_high<b_low: return {'type':'fvg_down','low':float(b_high),'high':float(a_low)}
    return None

def detect_liquidity_sweep(df):
    if df is None or len(df)<3: return None
    last,prev = df.iloc[-1], df.iloc[-2]
    price = last['close']
    wick_up = last['high'] - max(last['open'], last['close'])
    wick_down = min(last['open'], last['close']) - last['low']
    vol_spike = last['volume'] > (df['volume'].rolling(20).mean().iloc[-1]*1.8 if len(df)>=20 else last['volume']*2)
    if vol_spike and wick_up > price*LIQUIDITY_SWEEP_TAIL_PCT: return {'type':'sweep_up','size':float(wick_up)}
    if vol_spike and wick_down > price*LIQUIDITY_SWEEP_TAIL_PCT: return {'type':'sweep_down','size':float(wick_down)}
    return None

def detect_pre_top_pump(df):
    if df is None or len(df)<6: return None
    recent = df.iloc[-6:]
    pct_move = (recent['close'].iloc[-1]/recent['close'].iloc[0])-1
    vol_ratio = recent['volume'].iloc[-1]/(recent['volume'].rolling(6).mean().iloc[-1]+1e-9)
    if pct_move>=PUMP_PCT_THRESHOLD and vol_ratio>=2.0: return {'type':'pump','pct':float(pct_move),'vol_ratio':float(vol_ratio)}
    return None

def detect_market_structure_shift(df):
    if df is None or len(df)<10: return None
    last_close = df['close'].iloc[-1]
    prior_high = df['high'].iloc[-10:-2].max()
    prior_low = df['low'].iloc[-10:-2].min()
    if last_close>prior_high: return 'BOS_UP'
    if last_close<prior_low: return 'BOS_DOWN'
    return None

# -------------------------
# ML
# -------------------------
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
    df2 = build_features_from_df(df)
    if len(df2)<lookahead+50: return None,None
    df2['future_ret']=df2['close'].shift(-lookahead)/df2['close']-1
    df2['target']=(df2['future_ret']>0).astype(int)
    features=['range','body','close_o','p_change_1','vol_spike']
    return df2[features], df2['target']

def train_or_load_model(symbol, df, lookahead=5):
    model_path=os.path.join(MODEL_DIR,f"{symbol}_rf.joblib")
    try:
        if os.path.exists(model_path) and os.path.getsize(model_path)>0:
            model=joblib.load(model_path)
            return model,state.get('models',{}).get(f"{symbol}_meta",{}).get('acc',0.0)
    except: pass
    X,y=build_training_dataset(df,lookahead)
    if X is None or len(X)<MIN_ROWS_FOR_ML: return None,None
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,shuffle=False)
    model=RandomForestClassifier(n_estimators=200,n_jobs=1,random_state=42)
    model.fit(X_train,y_train)
    acc=float(accuracy_score(y_test,model.predict(X_test)))
    tmp=model_path+".tmp"; joblib.dump(model,tmp); os.replace(tmp,model_path)
    state.setdefault('models',{})[f"{symbol}_meta"]={'acc':acc,'trained_at':str(datetime.now(timezone.utc))}
    save_json_safe(STATE_FILE,state)
    return model,acc

# -------------------------
# Combine evidence
# -------------------------
def combine_evidence(evidence, ml_prob):
    score_long=0.0; score_short=0.0
    ob = evidence.get('order_block'); price=evidence.get('price')
    if ob and price:
        dist_top=abs(price-ob['top'])/(price+1e-9)
        dist_bot=abs(price-ob['bot'])/(price+1e-9)
        score_short+=WEIGHTS['order_block']*max(0,1-dist_top*10)
        score_long+=WEIGHTS['order_block']*max(0,1-dist_bot*10)
    fvg=evidence.get('fvg')
    if fvg:
        score_long+=WEIGHTS['fvg'] if fvg['type']=='fvg_up' else 0
        score_short+=WEIGHTS['fvg'] if fvg['type']=='fvg_down' else 0
    sweep=evidence.get('sweep')
    if sweep:
        score_short+=WEIGHTS['liquidity_sweep'] if sweep['type']=='sweep_up' else 0
        score_long+=WEIGHTS['liquidity_sweep'] if sweep['type']=='sweep_down' else 0
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

# -------------------------
# Analyze symbol
# -------------------------
def analyze_symbol(symbol, interval='5m'):
    try:
        df=get_klines_df(symbol,interval,limit=600)
        if df is None or len(df)<12: return None
        price=float(df['close'].iloc[-1])
        ob=detect_order_blocks(df)
        fvg=detect_fair_value_gap(df)
        sweep=detect_liquidity_sweep(df)
        pump=detect_pre_top_pump(df)
        mss=detect_market_structure_shift(df)
        evidence={'order_block':ob,'fvg':fvg,'sweep':sweep,'pump':pump,'mss':mss,'price':price}
        model,ml_acc=train_or_load_model(symbol,df,lookahead=5)
        ml_prob=None
        if model is not None:
            feat_df=build_features_from_df(df)
            if len(feat_df)>0:
                lastf=feat_df.iloc[-1:][['range','body','close_o','p_change_1','vol_spike']]
                try: ml_prob=float(model.predict_proba(lastf)[0][1])
                except: ml_prob=None
        conf,direction=combine_evidence(evidence,ml_prob)
        label="WATCH"
        if conf>=0.65: label="STRONG"
        elif conf>=0.45: label="MEDIUM"
        elif conf>=0.20: label="WEAK"
        return {"symbol":symbol,"interval":interval,"price":price,"evidence":evidence,"ml_prob":ml_prob,"ml_acc":ml_acc,"confidence":conf,"direction":direction,"label":label,"timestamp":str(df.index[-1])}
    except Exception as e:
        logger.exception("analyze_symbol error %s: %s", symbol,e)
        return None

# -------------------------
# Graph generation
# -------------------------
def plot_signal(df,symbol,signal):
    fig,ax=plt.subplots(figsize=(8,4))
    ax.plot(df.index,df['close'],color='blue',label='Close')
    price=signal['price']
    direction=signal['direction']
    if direction=='LONG':
        ax.axhline(price,color='green',linestyle='--',label='Entry')
        ax.axhline(price*1.015,color='green',linestyle=':',label='TP')
        ax.axhline(price*0.995,color='red',linestyle=':',label='SL')
    elif direction=='SHORT':
        ax.axhline(price,color='red',linestyle='--',label='Entry')
        ax.axhline(price*0.985,color='red',linestyle=':',label='TP')
        ax.axhline(price*1.005,color='green',linestyle=':',label='SL')
    ax.set_title(f"{symbol} {signal['interval']} {direction} Conf:{signal['confidence']:.2f}")
    ax.legend()
    path=f"{symbol}_{signal['interval']}.png"
    plt.savefig(path)
    plt.close()
    return path

# -------------------------
# Scan all symbols
# -------------------------
def scan_all(interval='5m'):
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
            except: pass
    results_sorted=sorted(results,key=lambda x:x['confidence'],reverse=True)[:50]
    for r in results_sorted:
        df=get_klines_df(r['symbol'],r['interval'],limit=200)
        path=plot_signal(df,r['symbol'],r)
        text=f"*SMC Alert*\nSymbol:`{r['symbol']}` Interval:`{r['interval']}`\nDirection:`{r['direction'] or 'WATCH'}` ({r['label']})\nPrice:`{r['price']:.4f}` Conf:`{r['confidence']:.2f}`\nTime:`{r['timestamp']}`"
        if r['ml_prob'] is not None: text+=f"\nML ProbUp:`{r['ml_prob']:.2f}` ML Acc:`{r['ml_acc'] or 0:.2f}`"
        send_telegram_message(text)
        # optionally send image
        if os.path.exists(path):
            try:
                with open(path,"rb") as f:
                    requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto", files={"photo":f}, data={"chat_id":CHAT_ID})
            except: pass
        time.sleep(0.2)

# -------------------------
# Scheduler
# -------------------------
scheduler=BackgroundScheduler()
scheduler.add_job(lambda: scan_all('5m'),'interval',minutes=SCAN_INTERVAL_MINUTES,id='scan5m')
scheduler.start()

# -------------------------
# Flask
# -------------------------
@app.route('/')
def home(): return jsonify({"status":"ok","time":str(datetime.now(timezone.utc))})

@app.route('/scan_now')
def scan_now(): 
    try: scan_all('5m'); return jsonify({"status":"scanned"})
    except Exception as e: return jsonify({"error":str(e)}),500

@app.route('/status')
def status(): return jsonify({"state":state,"time":str(datetime.now(timezone.utc))})

# -------------------------
# Warmup
# -------------------------
def warmup():
    logger.info("Warmup started")
    symbols=get_all_usdt_symbols()[:20]
    for s in symbols:
        df=get_klines_df(s,'1h',limit=800)
        if df is not None: train_or_load_model(s,df)
        time.sleep(0.3+random.random()*0.2)
    logger.info("Warmup finished")

Thread(target=warmup,daemon=True).start()

# -------------------------
# MAIN
# -------------------------
if __name__=="__main__":
    logger.info("Starting SMC Market Pulse Bot")
    scan_all('5m')
    app.run(host='0.0.0.0',port=PORT)