# main.py — SMC-first multi-symbol scanner with pre-top / pump-detect + ML support
# Requirements: python-binance, pandas, numpy, scikit-learn, joblib, ta (optional), apscheduler, flask, requests
# Env: BINANCE_API_KEY, BINANCE_API_SECRET, TELEGRAM_TOKEN, CHAT_ID

import os
import time
import json
import math
import joblib
import random
import logging
import traceback
import re
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Thread

import numpy as np
import pandas as pd
import requests
from flask import Flask, jsonify
from apscheduler.schedulers.background import BackgroundScheduler
from binance.client import Client
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -------------------------
# CONFIG
# -------------------------
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
CHAT_ID = os.getenv("CHAT_ID", "")
PORT = int(os.getenv("PORT", "5000"))

PARALLEL_WORKERS = int(os.getenv("PARALLEL_WORKERS", "6"))
SCAN_INTERVAL_MINUTES = int(os.getenv("SCAN_INTERVAL_MINUTES", "1"))  # EMA-like scan (fast)
MONITOR_INTERVAL_MINUTES = int(os.getenv("MONITOR_INTERVAL_MINUTES", "5"))  # top-symbols deeper monitor

STATE_FILE = "state.json"
MODEL_DIR = "models"
EMA_STATS_FILE = "ema_stats.json"
LOG_FILE = "bot.log"

os.makedirs(MODEL_DIR, exist_ok=True)

# thresholds / hyperparams
MIN_ROWS_FOR_ML = 300
CONF_THRESHOLD_ALERT = 0.45  # threshold for sending "signal" messages
CONF_THRESHOLD_WATCH = 0.20  # threshold for watch messages
PUMP_PCT_THRESHOLD = 0.08  # 8% move in short window -> pump candidate
LIQUIDITY_SWEEP_TAIL_PCT = 0.008  # tail detection threshold

# weights for combining evidence (tuneable)
WEIGHTS = {
    "order_block": 0.25,
    "fvg": 0.15,
    "liquidity_sweep": 0.15,
    "pre_top_pump": 0.20,
    "ml": 0.20
}

# -------------------------
# LOGGING
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)
logger = logging.getLogger("smc-bot")

# -------------------------
# BINANCE CLIENT
# -------------------------
client = Client(api_key=BINANCE_API_KEY, api_secret=BINANCE_API_SECRET)

# -------------------------
# FLASK (status endpoints)
# -------------------------
app = Flask(__name__)

# -------------------------
# STATE / IO helpers
# -------------------------
def load_json_safe(path, default):
    try:
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
    except Exception as e:
        logger.exception("load_json_safe %s failed: %s", path, e)
    return default

def save_json_safe(path, data):
    try:
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp, path)
    except Exception as e:
        logger.exception("save_json_safe %s failed: %s", path, e)

state = load_json_safe(STATE_FILE, {"signals": {}, "models": {}, "history": {}})
ema_stats = load_json_safe(EMA_STATS_FILE, {})

# -------------------------
# Telegram helpers (escape for MarkdownV2)
# -------------------------
MDV2_ESCAPE = r"_*[]()~`>#+-=|{}.!"

def escape_markdown_v2(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    return re.sub("([" + re.escape(MDV2_ESCAPE) + "])", r"\\\1", s)

def send_telegram_message(text: str):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        logger.warning("Telegram not configured; skipping send.")
        return False
    try:
        payload = {
            "chat_id": CHAT_ID,
            "text": escape_markdown_v2(text),
            "parse_mode": "MarkdownV2",
            "disable_web_page_preview": True
        }
        r = requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", json=payload, timeout=10)
        if r.status_code != 200:
            logger.error("Telegram send failed: %s %s", r.status_code, r.text)
            return False
        return True
    except Exception as e:
        logger.exception("Telegram send exception: %s", e)
        return False

# -------------------------
# Market data helpers
# -------------------------
def get_all_usdt_symbols():
    try:
        info = client.get_exchange_info()
        syms = [s['symbol'] for s in info['symbols'] if s['quoteAsset'] == 'USDT' and s['status'] == 'TRADING']
        return syms
    except Exception as e:
        logger.exception("get_all_usdt_symbols error: %s", e)
        return []

def get_top_symbols_by_24h_change(limit=30):
    try:
        tickers = client.get_ticker()
        df = pd.DataFrame(tickers)
        df = df[df['symbol'].str.endswith('USDT')]
        df['priceChangePercent'] = df['priceChangePercent'].astype(float)
        df = df.sort_values('priceChangePercent', ascending=False)
        return df['symbol'].tolist()[:limit]
    except Exception as e:
        logger.exception("get_top_symbols_by_24h_change: %s", e)
        return []

def get_klines_df(symbol, interval, limit=500, retry=3):
    for attempt in range(retry):
        try:
            kl = client.get_klines(symbol=symbol, interval=interval, limit=limit)
            if not kl:
                return None
            df = pd.DataFrame(kl, columns=[
                'open_time','open','high','low','close','volume','close_time',
                'qav','num_trades','taker_base_vol','taker_quote_vol','ignore'
            ])
            df = df[['open_time','open','high','low','close','volume']].copy()
            df[['open','high','low','close','volume']] = df[['open','high','low','close','volume']].astype(float)
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            df.set_index('open_time', inplace=True)
            return df
        except Exception as e:
            logger.warning("get_klines_df %s %s attempt %d/%d error: %s", symbol, interval, attempt+1, retry, e)
            time.sleep(0.5 + attempt)
    return None

# -------------------------
# SMC-style detectors (no classic indicators)
# -------------------------
def detect_order_blocks(df, lookback=60, strength_top=3):
    """
    Detect simple order blocks: large consolidation candles before a breakout.
    Approach (simplified):
      - Find local consolidation zone: last N bars where range small relative to recent ATR-like measure.
      - Mark top/bottom of consolidation as order block.
    Returns dict with last order block (type, top, bottom, bars)
    """
    out = None
    if df is None or len(df) < lookback + 5:
        return None
    # compute rolling range metric
    rng = (df['high'] - df['low']).rolling(lookback).mean()
    recent_rng = rng.iloc[-1]
    last_segment = df.iloc[-lookback:]
    # threshold: if last segment range is small relative to historical mean -> consolidation
    hist_mean = rng.mean() if not math.isnan(rng.mean()) else recent_rng
    if hist_mean == 0 or math.isnan(hist_mean):
        return None
    # consolidation if segment max-min < hist_mean * 0.6
    seg_range = last_segment['high'].max() - last_segment['low'].min()
    if seg_range < hist_mean * 0.6:
        ob_top = last_segment['high'].max()
        ob_bot = last_segment['low'].min()
        out = {'type': 'order_block', 'top': float(ob_top), 'bot': float(ob_bot), 'range': float(seg_range)}
    return out

def detect_fair_value_gap(df):
    """
    Fair Value Gap (FVG) simplified:
      - Look for 3-candle pattern where middle candle has a large body and leaves a gap between body edges.
    Return last FVG side: up-gap or down-gap and edges.
    """
    if df is None or len(df) < 6:
        return None
    # examine last 6 candles
    for i in range(len(df)-3, 0, -1):
        a = df.iloc[i-1]; b = df.iloc[i]; c = df.iloc[i+1]
        # bullish FVG: b low > a high and c low > b high (rare strict). Simpler: gap between b body and neighbors
        b_body_low = min(b['open'], b['close'])
        b_body_high = max(b['open'], b['close'])
        a_body_high = max(a['open'], a['close'])
        c_body_low = min(c['open'], c['close'])
        # bullish gap
        if b_body_low > a_body_high and c_body_low > b_body_high:
            return {'type': 'fvg_up', 'low': float(a_body_high), 'high': float(b_body_low)}
        # bearish gap
        b_body_high = max(b['open'], b['close'])
        a_body_low = min(a['open'], a['close'])
        c_body_high = max(c['open'], c['close'])
        if b_body_high < a_body_low and c_body_high < b_body_low:
            return {'type': 'fvg_down', 'low': float(b_body_high), 'high': float(a_body_low)}
    return None

def detect_liquidity_sweep(df):
    """
    Detect wick/tail sweeps that remove liquidity: large tail beyond prior structure.
    Simplified: if last candle's wick (high-close or close-low) > LIQUIDITY_SWEEP_TAIL_PCT * price and volume spike -> sweep
    """
    if df is None or len(df) < 3:
        return None
    last = df.iloc[-1]
    prev = df.iloc[-2]
    price = last['close']
    wick_up = last['high'] - max(last['open'], last['close'])
    wick_down = min(last['open'], last['close']) - last['low']
    vol_spike = last['volume'] > (df['volume'].rolling(20).mean().iloc[-1] * 1.8 if len(df) >= 20 else last['volume'] * 2)
    if vol_spike and wick_up > price * LIQUIDITY_SWEEP_TAIL_PCT:
        return {'type': 'sweep_up', 'size': float(wick_up)}
    if vol_spike and wick_down > price * LIQUIDITY_SWEEP_TAIL_PCT:
        return {'type': 'sweep_down', 'size': float(wick_down)}
    return None

def detect_pre_top_pump(df, lookback_minutes=6, pct=PUMP_PCT_THRESHOLD):
    """
    Pump detector: sudden % rise during the last few bars + rising volume.
    Returns pump signal if recent percent change over short window > pct.
    """
    if df is None or len(df) < 6:
        return None
    # use last 6 bars
    recent = df.iloc[-6:]
    start = recent['close'].iloc[0]
    end = recent['close'].iloc[-1]
    pct_move = (end / start) - 1
    vol_ratio = recent['volume'].iloc[-1] / (recent['volume'].rolling(6).mean().iloc[-1] + 1e-9)
    if pct_move >= pct and vol_ratio >= 2.0:
        return {'type': 'pump', 'pct': float(pct_move), 'vol_ratio': float(vol_ratio)}
    return None

def detect_market_structure_shift(df):
    """
    Very simplified MSS: compare recent highs/lows to prior.
    If last close breaks prior swing high -> BOS up, if breaks prior swing low -> BOS down.
    """
    if df is None or len(df) < 10:
        return None
    last_close = df['close'].iloc[-1]
    prior_high = df['high'].iloc[-10:-2].max()
    prior_low = df['low'].iloc[-10:-2].min()
    if last_close > prior_high:
        return 'BOS_UP'
    if last_close < prior_low:
        return 'BOS_DOWN'
    return None

# -------------------------
# ML helpers (train on SMC-like features)
# -------------------------
def build_features_from_df(df):
    df = df.copy()
    df['range'] = df['high'] - df['low']
    df['body'] = abs(df['close'] - df['open'])
    df['close_o'] = df['close'] - df['open']
    df['p_change_1'] = df['close'].pct_change(1)
    df['vol_ma_20'] = df['volume'].rolling(20).mean()
    df['vol_spike'] = (df['volume'] / (df['vol_ma_20'] + 1e-9)).fillna(1.0)
    # last n values then dropna
    df = df.dropna()
    return df

def build_training_dataset(df, lookahead=5):
    df2 = build_features_from_df(df)
    if len(df2) < lookahead + 50:
        return None, None
    df2['future_ret'] = df2['close'].shift(-lookahead) / df2['close'] - 1
    df2 = df2.dropna()
    df2['target'] = (df2['future_ret'] > 0).astype(int)
    features = ['range', 'body', 'close_o', 'p_change_1', 'vol_spike']
    X = df2[features]
    y = df2['target']
    return X, y

def train_or_load_model(symbol, df, lookahead=5):
    model_path = os.path.join(MODEL_DIR, f"{symbol}_rf.joblib")
    try:
        # try load
        if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
            model = joblib.load(model_path)
            logger.info("Loaded model for %s", symbol)
            # quick check
            return model, state.get('models', {}).get(f"{symbol}_meta", {}).get('acc', 0.0)
    except Exception as e:
        logger.exception("Model load failed for %s: %s", symbol, e)
        try:
            os.remove(model_path)
        except Exception:
            pass

    # build dataset
    X, y = build_training_dataset(df, lookahead=lookahead)
    if X is None or len(X) < MIN_ROWS_FOR_ML:
        logger.info("%s not enough rows for ML (have %s)", symbol, len(X) if X is not None else 0)
        return None, None
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        model = RandomForestClassifier(n_estimators=200, n_jobs=1, random_state=42)
        model.fit(X_train, y_train)
        acc = float(accuracy_score(y_test, model.predict(X_test)))
        tmp = model_path + ".tmp"
        joblib.dump(model, tmp)
        os.replace(tmp, model_path)
        # store model metadata
        state.setdefault('models', {})[f"{symbol}_meta"] = {'acc': acc, 'trained_at': str(datetime.now(timezone.utc))}
        save_json_safe(STATE_FILE, state)
        logger.info("Trained and saved model for %s (acc=%.3f)", symbol, acc)
        return model, acc
    except Exception as e:
        logger.exception("Training failed for %s: %s", symbol, e)
        return None, None

# -------------------------
# Combine evidence -> confidence
# -------------------------
def combine_evidence(evidence, ml_prob):
    """
    evidence: dict with boolean/weights for each detector.
    ml_prob: probability up (0..1) or None
    returns confidence 0..1 and suggested direction ('LONG'/'SHORT'/None)
    """
    score_long = 0.0
    score_short = 0.0

    # order_block: if price near top -> short bias; near bot -> long bias
    ob = evidence.get('order_block')
    if ob:
        price = evidence.get('price')
        if price is None:
            pass
        else:
            # proximity
            dist_top = abs(price - ob['top']) / (price + 1e-9)
            dist_bot = abs(price - ob['bot']) / (price + 1e-9)
            # close to top -> short weight, close to bot -> long weight
            score_short += WEIGHTS['order_block'] * max(0, 1 - dist_top*10)
            score_long += WEIGHTS['order_block'] * max(0, 1 - dist_bot*10)

    # fvg: up-gap gives long on retrace, down-gap gives short on retrace
    fvg = evidence.get('fvg')
    if fvg:
        if fvg['type'] == 'fvg_up':
            score_long += WEIGHTS['fvg']
        elif fvg['type'] == 'fvg_down':
            score_short += WEIGHTS['fvg']

    # sweep: sweep_up can precede down move (liquidity grabbed) — context dependent; treat as contrarian
    sweep = evidence.get('sweep')
    if sweep:
        if sweep['type'] == 'sweep_up':
            # often sweep up -> short (stop run)
            score_short += WEIGHTS['liquidity_sweep']
        elif sweep['type'] == 'sweep_down':
            score_long += WEIGHTS['liquidity_sweep']

    # pump: pump gives short (pre-top) if overbought
    pump = evidence.get('pump')
    if pump:
        # pump gives short opportunity
        score_short += WEIGHTS['pre_top_pump']

    # MSS / BOS if present: BOS_UP favors long, BOS_DOWN favors short
    mss = evidence.get('mss')
    if mss == 'BOS_UP':
        score_long += 0.08
    if mss == 'BOS_DOWN':
        score_short += 0.08

    # incorporate ML: ml_prob is probability price will go up
    if ml_prob is not None:
        score_long += WEIGHTS['ml'] * ml_prob
        score_short += WEIGHTS['ml'] * (1 - ml_prob)

    # normalize
    total = score_long + score_short + 1e-9
    conf = max(score_long, score_short) / total
    direction = 'LONG' if score_long > score_short and conf > 0.01 else ('SHORT' if score_short > score_long and conf > 0.01 else None)
    # map conf to 0..1 not strictly normalized; scale by total evidence strength
    strength = min(1.0, max(score_long, score_short))
    # final confidence combine direction and strength (weighted)
    # we prefer conf based on evidence strength
    final_confidence = float(strength)
    return final_confidence, direction

# -------------------------
# Analyze one symbol (core worker)
# -------------------------
def analyze_symbol(symbol, interval='5m'):
    try:
        df = get_klines_df(symbol, interval, limit=600)
        if df is None or len(df) < 12:
            return None
        price = float(df['close'].iloc[-1])
        # SMC detectors
        ob = detect_order_blocks(df)
        fvg = detect_fair_value_gap(df)
        sweep = detect_liquidity_sweep(df)
        pump = detect_pre_top_pump(df)
        mss = detect_market_structure_shift(df)

        evidence = {
            'order_block': ob,
            'fvg': fvg,
            'sweep': sweep,
            'pump': pump,
            'mss': mss,
            'price': price
        }

        # ML
        model, ml_acc = train_or_load_model(symbol, df, lookahead=5)
        ml_prob = None
        if model is not None:
            feat_df = build_features_from_df(df)
            if len(feat_df) > 0:
                lastf = feat_df.iloc[-1:][['range','body','close_o','p_change_1','vol_spike']]
                try:
                    prob = model.predict_proba(lastf)[0][1]
                    ml_prob = float(prob)
                except Exception:
                    ml_prob = None

        conf, direction = combine_evidence(evidence, ml_prob)

        # classify label strength
        label = "WATCH"
        if conf >= 0.65:
            label = "STRONG"
        elif conf >= 0.45:
            label = "MEDIUM"
        elif conf >= 0.20:
            label = "WEAK"

        out = {
            "symbol": symbol,
            "interval": interval,
            "price": price,
            "evidence": evidence,
            "ml_prob": ml_prob,
            "ml_acc": ml_acc,
            "confidence": conf,
            "direction": direction,
            "label": label,
            "timestamp": str(df.index[-1])
        }
        return out
    except Exception as e:
        logger.exception("analyze_symbol error %s: %s", symbol, e)
        return None

# -------------------------
# Scans
# -------------------------
def ema_scan_all(interval='5m'):
    logger.info("Starting full SMC scan (interval=%s)...", interval)
    symbols = get_all_usdt_symbols()
    if not symbols:
        logger.warning("No symbols found.")
        return
    results = []
    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as exe:
        futures = {exe.submit(analyze_symbol, s, interval): s for s in symbols}
        for fut in as_completed(futures):
            sym = futures[fut]
            try:
                res = fut.result()
                if res:
                    # keep as 'watch' if >= watch threshold
                    if res['confidence'] >= CONF_THRESHOLD_WATCH:
                        results.append(res)
            except Exception as e:
                logger.exception("Future error %s: %s", sym, e)
    logger.info("Scan done, hits=%d", len(results))
    # sort and send limited messages (avoid spam)
    results_sorted = sorted(results, key=lambda x: x['confidence'], reverse=True)[:80]
    for r in results_sorted:
        # prepare message content
        patt = []
        ev = r['evidence']
        if ev.get('order_block'):
            patt.append('order_block')
        if ev.get('fvg'):
            patt.append(ev['fvg']['type'])
        if ev.get('sweep'):
            patt.append(ev['sweep']['type'])
        if ev.get('pump'):
            patt.append('pump')
        if ev.get('mss'):
            patt.append(ev.get('mss'))

        text = f"*SMC Alert* 🔔\nSymbol: `{r['symbol']}`\nInterval: `{r['interval']}`\nEvent: *{r['direction'] or 'WATCH'}* ({r['label']})\nPrice: `{r['price']}`\nConfidence: `{r['confidence']:.2f}`\nTime: `{r['timestamp']}`"
        if r['ml_prob'] is not None:
            text += f"\n🤖 ML ProbUp: `{r['ml_prob']:.2f}` | ML Acc: `{r['ml_acc'] or 0:.2f}`"
        if patt:
            text += f"\n🧩 Patterns: {', '.join(patt)}"
        # historical EMA stats: optional
        send_telegram_message(text)
        time.sleep(0.12)

def monitor_top_symbols():
    logger.info("Start detailed monitor for top symbols")
    symbols = get_top_symbols_by_24h_change(limit=60)
    interval_list = ['15m', '1h', '4h']
    results = []
    with ThreadPoolExecutor(max_workers=min(PARALLEL_WORKERS, 8)) as exe:
        futures = []
        for s in symbols:
            for intr in interval_list:
                futures.append(exe.submit(analyze_symbol, s, intr))
        for fut in as_completed(futures):
            try:
                res = fut.result()
                if not res:
                    continue
                # decide alerting
                conf = res['confidence']
                ml_prob = res['ml_prob']
                direction = res['direction']
                should_alert = False
                if direction and conf >= CONF_THRESHOLD_ALERT:
                    should_alert = True
                # ML override: if ml_prob very strong and acc decent
                if ml_prob and ml_prob >= 0.85 and (res.get('ml_acc') or 0) >= 0.6:
                    should_alert = True
                if should_alert:
                    key = f"{res['symbol']}_{res['interval']}"
                    prev = state.get('signals', {}).get(key)
                    if prev != direction:
                        # send alert
                        patt = []
                        ev = res['evidence']
                        if ev.get('order_block'): patt.append('order_block')
                        if ev.get('fvg'): patt.append(ev['fvg']['type'])
                        if ev.get('sweep'): patt.append(ev['sweep']['type'])
                        if ev.get('pump'): patt.append('pump')
                        text = f"⚡ *Signal*\nSymbol: `{res['symbol']}`\nInterval: `{res['interval']}`\nSignal: *{direction}* ({res['label']})\nPrice: `{res['price']}`\nConfidence: `{res['confidence']:.2f}`\nSRI: `-`\nTime: `{res['timestamp']}`"
                        if res['ml_prob'] is not None:
                            text += f"\n🤖 ML ProbUp: `{res['ml_prob']:.2f}` | ML Acc: `{res['ml_acc'] or 0:.2f}`"
                        if patt:
                            text += f"\n🧩 Patterns: {', '.join(patt)}"
                        send_telegram_message(text)
                        state.setdefault('signals', {})[key] = direction
                        # save history
                        hist = state.setdefault('history', {}).setdefault(res['symbol'], [])
                        hist.append({'time': res['timestamp'], 'interval': res['interval'], 'signal': direction, 'conf': res['confidence']})
                        if len(hist) > 400:
                            state['history'][res['symbol']] = hist[-400:]
                        save_json_safe(STATE_FILE, state)
                results.append(res)
            except Exception as e:
                logger.exception("monitor future error: %s", e)
    logger.info("Detailed monitor done, processed=%d", len(results))

# -------------------------
# Scheduler
# -------------------------
scheduler = BackgroundScheduler()
scheduler.add_job(lambda: ema_scan_all(interval='5m'), 'interval', minutes=max(1, SCAN_INTERVAL_MINUTES), id='smc_scan')
scheduler.add_job(monitor_top_symbols, 'interval', minutes=max(1, MONITOR_INTERVAL_MINUTES), id='monitor_top')
scheduler.start()

# -------------------------
# HTTP ROUTES
# -------------------------
@app.route('/')
def home():
    return jsonify({"status": "ok", "time": str(datetime.now(timezone.utc))})

@app.route('/scan_now')
def scan_now():
    try:
        ema_scan_all(interval='5m')
        monitor_top_symbols()
        return jsonify({"status": "scanned"})
    except Exception as e:
        logger.exception("scan_now error: %s", e)
        return jsonify({"error": str(e)}), 500

@app.route('/status')
def status():
    return jsonify({
        "state": state,
        "ema_stats_count": len(ema_stats),
        "time": str(datetime.now(timezone.utc))
    })

# -------------------------
# Warmup (train small models in background)
# -------------------------
def warmup():
    logger.info("Warmup background started")
    syms = get_top_symbols_by_24h_change(limit=30)
    for s in syms:
        try:
            df = get_klines_df(s, '1h', limit=800)
            if df is None:
                continue
            train_or_load_model(s, df, lookahead=5)
            time.sleep(0.5 + random.random()*0.5)
        except Exception as e:
            logger.exception("warmup error for %s: %s", s, e)
    logger.info("Warmup finished")

# start warmup in background
Thread(target=warmup, daemon=True).start()

# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    logger.info("Starting SMC bot (multi-symbol) — press ctrl+c to stop")
    # initial quick run
    try:
        ema_scan_all(interval='5m')
    except Exception as e:
        logger.exception("Initial scan failed: %s", e)
    app.run(host='0.0.0.0', port=PORT)