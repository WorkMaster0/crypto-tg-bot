# =========================
# mega_bot.py — Part 1: Config & Binance API
# =========================

import os
import time
import json
import math
import random
import threading
import logging
import traceback
import re
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from flask import Flask, jsonify, request
from apscheduler.schedulers.background import BackgroundScheduler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from binance.client import Client

# -------------------------
# CONFIG
# -------------------------

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
CHAT_ID = os.getenv("CHAT_ID", "")
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")

PARALLEL_WORKERS = int(os.getenv("PARALLEL_WORKERS", "6"))
STATE_FILE = os.getenv("STATE_FILE", "state.json")
MODEL_DIR = os.getenv("MODEL_DIR", "models")
EMA_STATS_FILE = os.getenv("EMA_STATS_FILE", "ema_stats.json")
LOG_FILE = os.getenv("LOG_FILE", "bot.log")
SCAN_INTERVAL_MINUTES = int(os.getenv("SCAN_INTERVAL_MINUTES", "1"))
MONITOR_INTERVAL_MINUTES = int(os.getenv("MONITOR_INTERVAL_MINUTES", "5"))
EMA_SCAN_LIMIT = int(os.getenv("EMA_SCAN_LIMIT", "500"))
BACKTEST_LOOKAHEAD = int(os.getenv("BACKTEST_LOOKAHEAD", "5"))
PORT = int(os.getenv("PORT", "5000"))

os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------------
# LOGGING
# -------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)
logger = logging.getLogger()

# -------------------------
# BINANCE CLIENT
# -------------------------

client = Client(api_key=BINANCE_API_KEY, api_secret=BINANCE_API_SECRET)

# -------------------------
# FLASK
# -------------------------

app = Flask(__name__)

# -------------------------
# STATE HELPERS
# -------------------------

def load_json_safe(path, default):
    try:
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
    except Exception:
        logger.exception("Failed to load %s", path)
    return default

def save_json_safe(path, data):
    try:
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp, path)
    except Exception:
        logger.exception("Failed to save %s", path)

state = load_json_safe(STATE_FILE, {"signals": {}, "models": {}, "signal_history": {}})
ema_stats = load_json_safe(EMA_STATS_FILE, {})

# -------------------------
# UTILS
# -------------------------

def utcnow_str():
    return datetime.now(timezone.utc).isoformat()

def escape_markdown_v2(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    escape_chars = r'_*[]()~`>#+-=|{}.!'
    return re.sub(r'([%s])' % re.escape(escape_chars), r'\\\1', text)
    
    # =========================
# mega_bot.py — Part 2: Market Data & Indicators
# =========================

# -------------------------
# MARKET DATA HELPERS
# -------------------------

def get_all_usdt_symbols():
    try:
        info = client.get_exchange_info()
        return [s["symbol"] for s in info["symbols"] if s["quoteAsset"]=="USDT" and s["status"]=="TRADING"]
    except Exception:
        logger.exception("get_all_usdt_symbols error")
        return []

def get_top_symbols_by_volume(limit=30):
    try:
        tickers = client.get_ticker()
        sorted_t = sorted(tickers, key=lambda x: float(x.get("quoteVolume", 0)), reverse=True)
        return [t["symbol"] for t in sorted_t if t["symbol"].endswith("USDT")][:limit]
    except Exception:
        logger.exception("get_top_symbols_by_volume error")
        return []

def get_klines_df(symbol, interval, limit=500, retry=3):
    for attempt in range(retry):
        try:
            kl = client.get_klines(symbol=symbol, interval=interval, limit=limit)
            df = pd.DataFrame(kl, columns=[
                'open_time','open','high','low','close','volume','close_time','qav','num_trades',
                'taker_base_vol','taker_quote_vol','ignore'])
            df = df[['open_time','open','high','low','close','volume']].copy()
            df[['open','high','low','close','volume']] = df[['open','high','low','close','volume']].astype(float)
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            df.set_index('open_time', inplace=True)
            return df
        except Exception:
            logger.warning(f"get_klines_df fail {symbol} attempt {attempt+1}")
            time.sleep(0.3 + attempt*0.2)
    return None

# -------------------------
# INDICATORS
# -------------------------

def safe_apply_basic(df):
    df = df.copy()
    if df is None or len(df) == 0:
        return df
    try:
        # EMAs
        df['ema_5'] = df['close'].ewm(span=5, adjust=False).mean()
        df['ema_13'] = df['close'].ewm(span=13, adjust=False).mean()
        df['ema_34'] = df['close'].ewm(span=34, adjust=False).mean()
        # Returns
        df['ret1'] = df['close'].pct_change(1)
        df['ret3'] = df['close'].pct_change(3)
        df['ret10'] = df['close'].pct_change(10)
        df['vol_ma_20'] = df['volume'].rolling(20, min_periods=1).mean()
        df['vol_ma_50'] = df['volume'].rolling(50, min_periods=1).mean()
        df['ret10_std'] = df['ret1'].rolling(10, min_periods=1).std()
        # OBV
        obv_series = (np.sign(df['ret1']).fillna(0) * df['volume']).cumsum()
        df['OBV'] = obv_series
        # Heikin-Ashi
        ha_close = (df['open'] + df['high'] + df['low'] + df['close']) / 4
        ha_open = (df['open'] + df['close']) / 2
        ha_open_vals = [ha_open.iloc[0]]
        for i in range(1, len(df)):
            ha_open_vals.append((ha_open_vals[-1] + ha_close.iloc[i-1]) / 2)
        df['ha_open'] = ha_open_vals
        df['ha_close'] = ha_close
        df['ha_green'] = df['ha_close'] > df['ha_open']
    except Exception:
        logger.exception("safe_apply_basic error")
    return df

# -------------------------
# PATTERN DETECTORS
# -------------------------

def detect_ema_cross(df, short='ema_5', long='ema_34'):
    if df is None or len(df) < 3:
        return None
    a = df[short].iloc[-2]; b = df[long].iloc[-2]
    a1 = df[short].iloc[-1]; b1 = df[long].iloc[-1]
    if pd.isna(a) or pd.isna(b) or pd.isna(a1) or pd.isna(b1): return None
    if a <= b and a1 > b1: return 'bull_cross'
    if a >= b and a1 < b1: return 'bear_cross'
    return None

def detect_volume_spike(df, mult=1.7):
    if df is None or len(df) < 5:
        return False
    vol = df['volume'].iloc[-1]
    ma = df['vol_ma_20'].iloc[-1] if 'vol_ma_20' in df.columns else np.nan
    if pd.isna(ma) or ma == 0: return False
    return vol > ma * mult

def detect_obv_divergence(df):
    if df is None or len(df) < 3 or 'OBV' not in df.columns:
        return None
    p_prev = df['close'].iloc[-2]; p_cur = df['close'].iloc[-1]
    obv_prev = df['OBV'].iloc[-2]; obv_cur = df['OBV'].iloc[-1]
    if pd.isna(obv_prev) or pd.isna(obv_cur): return None
    if p_cur > p_prev and obv_cur < obv_prev: return 'obv_div_down'
    if p_cur < p_prev and obv_cur > obv_prev: return 'obv_div_up'
    return None

def detect_pre_top(df, symbol):
    if df is None or len(df) < 30:
        return None
    signs = []
    r3 = df['ret3'].iloc[-1]; r10 = df['ret10'].iloc[-1]
    if not pd.isna(r3) and r3 > 0.05: signs.append('fast_run_short')
    if not pd.isna(r10) and r10 > 0.08: signs.append('fast_run_long')
    if detect_volume_spike(df): signs.append('vol_spike')
    obvd = detect_obv_divergence(df)
    if obvd: signs.append(obvd)
    if df['ha_green'].iloc[-3] and not df['ha_green'].iloc[-1]: signs.append('ha_flip')
    look = 50
    if len(df) > look:
        recent_max = df['close'].iloc[-look:-1].max()
        cur = df['close'].iloc[-1]
        if recent_max > 0 and (cur / recent_max) > 0.98: signs.append('near_local_max')
    if len(signs) >= 3:
        return {'pre_top': True, 'reasons': signs}
    return None
    
    # =========================
# mega_bot.py — Part 3: ML Models & SRI
# =========================

# -------------------------
# FEATURES FOR ML
# -------------------------

def features_for_ml(df):
    df = safe_apply_basic(df)
    df['vol_change'] = df['volume'] / (df['vol_ma_20'] + 1e-9)
    df['ema5_34_diff'] = df['ema_5'] - df['ema_34']
    df = df.dropna()
    return df

# -------------------------
# JOBLIB HELPERS
# -------------------------

def safe_joblib_dump(model, path):
    tmp = path + ".tmp"
    try:
        joblib.dump(model, tmp)
        os.replace(tmp, path)
        return True
    except Exception:
        logger.exception("safe_joblib_dump failed for %s", path)
        try:
            if os.path.exists(tmp): os.remove(tmp)
        except Exception:
            pass
        return False

# -------------------------
# TRAIN OR LOAD MODEL
# -------------------------

def train_or_load_model(symbol, df, lookahead=5):
    model_path = os.path.join(MODEL_DIR, f"{symbol}_rf.joblib")
    df2 = features_for_ml(df)
    if len(df2) < 300:
        logger.info(f"{symbol} not enough rows for ML")
        return None, None
    df2['future_ret'] = df2['close'].shift(-lookahead) / df2['close'] - 1
    df2 = df2.dropna()
    df2['target'] = (df2['future_ret'] > 0).astype(int)
    features = ['ret1','ret3','ret10','vol_change','ema5_34_diff','ret10_std']
    X = df2[features]; y = df2['target']
    
    # Try load existing
    if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
        try:
            model = joblib.load(model_path)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
            acc = accuracy_score(y_test, model.predict(X_test))
            return model, float(acc)
        except Exception:
            logger.exception("Failed loading model, will retrain")
            try: os.remove(model_path)
            except Exception: pass
    
    # Train new model
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        model = RandomForestClassifier(n_estimators=150, n_jobs=1, random_state=42)
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        if safe_joblib_dump(model, model_path):
            logger.info(f"Trained and saved model for {symbol} (acc={acc:.3f})")
        return model, float(acc)
    except Exception:
        logger.exception(f"Training failed for {symbol}")
        return None, None

# -------------------------
# SRI & CONFIDENCE
# -------------------------

def compute_sri(symbol, interval):
    key = f"{symbol}_{interval}"
    stats_entry = ema_stats.get(key)
    parts = []
    if stats_entry and 'stats' in stats_entry:
        g = stats_entry['stats'].get('golden')
        d = stats_entry['stats'].get('death')
        if g and g.get('win_rate') is not None: parts.append(g['win_rate'])
        if d and d.get('win_rate') is not None: parts.append(1.0 - d['win_rate'])
    m = state.get('models', {}).get(key)
    if m and m.get('acc') is not None: parts.append(float(m['acc']))
    if not parts: return 0.0
    return float(np.mean(parts))

def direction_from_signal(sig):
    if not sig: return None
    s = sig.lower()
    if 'long' in s or 'bull' in s or 'up' in s: return 'LONG'
    if 'short' in s or 'bear' in s or 'down' in s: return 'SHORT'
    return None

CONF_WEIGHTS = {
    "hist": 0.30,
    "ml_prob": 0.25,
    "ml_acc": 0.10,
    "pattern_agreement": 0.20,
    "volume": 0.05,
    "sri": 0.10
}

def compute_confidence(hist_stats, ml_prob, ml_acc, vol_spike, patterns, sri):
    score = 0.0
    if hist_stats and 'win_rate' in hist_stats and hist_stats['win_rate'] is not None:
        score += min(1.0, hist_stats['win_rate']) * CONF_WEIGHTS['hist']
    if ml_prob is not None:
        score += ml_prob * CONF_WEIGHTS['ml_prob']
    if ml_acc is not None:
        score += min(1.0, ml_acc) * CONF_WEIGHTS['ml_acc']
    if sri is not None:
        score += sri * CONF_WEIGHTS['sri']
    if vol_spike:
        score += CONF_WEIGHTS['volume']
    dirs = []
    for v in (patterns or {}).values():
        d = direction_from_signal(v)
        if d: dirs.append(d)
    if dirs:
        counts = {'LONG': dirs.count('LONG'), 'SHORT': dirs.count('SHORT')}
        maj_count = max(counts['LONG'], counts['SHORT'])
        pattern_ratio = maj_count / len(dirs)
        score += pattern_ratio * CONF_WEIGHTS['pattern_agreement']
        if ml_prob is not None:
            ml_dir = 'LONG' if ml_prob >= 0.5 else 'SHORT'
            maj = 'LONG' if counts['LONG'] >= counts['SHORT'] else 'SHORT'
            if ml_dir != maj:
                score -= 0.05
    score = max(0.0, min(1.0, score))
    return score

def strength_label(conf):
    if conf >= 0.65: return "STRONG"
    if conf >= 0.45: return "MEDIUM"
    if conf >= 0.20: return "WEAK"
    return "WATCH"
    
    # =========================
# mega_bot.py — Part 4: Symbol Analysis & Alerts
# =========================

import matplotlib.pyplot as plt

# -------------------------
# BACKTEST HISTORICAL PATTERN
# -------------------------
def backtest_pattern(df, lookahead=5):
    if df is None or len(df) < lookahead + 10:
        return None
    df = safe_apply_basic(df)
    results = []
    for i in range(0, len(df)-lookahead):
        if i < 2: continue
        a = df['ema_5'].iloc[i-1]; b = df['ema_34'].iloc[i-1]
        a1 = df['ema_5'].iloc[i]; b1 = df['ema_34'].iloc[i]
        sig = None
        if a <= b and a1 > b1: sig = 'LONG'
        if a >= b and a1 < b1: sig = 'SHORT'
        if sig:
            fut = (df['close'].iloc[i+lookahead] / df['close'].iloc[i]) - 1
            results.append({'index': df.index[i], 'signal': sig, 'future_return': fut})
    if not results: return None
    res_df = pd.DataFrame(results)
    stats = res_df.groupby('signal')['future_return'].agg(['mean','count', lambda x: (x>0).mean()]).rename(columns={'<lambda_0>':'win_rate'})
    out = {}
    for s in stats.index:
        out[s] = {'mean_return': float(stats.loc[s,'mean']), 'count': int(stats.loc[s,'count']), 'win_rate': float(stats.loc[s,'win_rate'])}
    return out

# -------------------------
# SYMBOL ANALYZER
# -------------------------
def analyze_symbol(symbol, interval='5m'):
    try:
        df = get_klines_df(symbol, interval, limit=EMA_SCAN_LIMIT)
        if df is None or len(df) < 30: return None
        df = safe_apply_basic(df)
        latest = df.iloc[-1]

        # Base EMA signal
        base_sig = None
        cross = detect_ema_cross(df, short='ema_5', long='ema_34')
        if cross == 'bull_cross': base_sig = 'LONG'
        if cross == 'bear_cross': base_sig = 'SHORT'

        # Historical backtest stats
        hist = backtest_pattern(df, lookahead=BACKTEST_LOOKAHEAD)
        hist_stats = hist.get(base_sig) if hist and base_sig in hist else None

        # ML model prediction
        model, acc = train_or_load_model(symbol, df, lookahead=BACKTEST_LOOKAHEAD)
        ml_prob = None
        if model:
            feat = features_for_ml(df).iloc[-1]
            features = ['ret1','ret3','ret10','vol_change','ema5_34_diff','ret10_std']
            X_row = pd.DataFrame([feat[features].values], columns=features)
            try: ml_prob = float(model.predict_proba(X_row)[0][1])
            except Exception: ml_prob = None
            key = f"{symbol}_{interval}"
            state.setdefault('models', {})[key] = {"acc": float(acc or 0), "trained_at": utcnow_str()}
            save_json_safe(STATE_FILE, state)

        # Pattern detectors
        vol_spike = detect_volume_spike(df)
        orderflow = detect_orderflow_imbalance(symbol)
        obv_div = detect_obv_divergence(df)
        pre_top = detect_pre_top(df, symbol)
        ema_cross = cross
        patterns = {
            'ema_cross': ema_cross,
            'vol_spike': 'vol_spike' if vol_spike else None,
            'orderflow': orderflow,
            'obv_div': obv_div,
            'pre_top': ('pre_top' if pre_top else None)
        }

        # Compute SRI and confidence
        sri = compute_sri(symbol, interval)
        conf = compute_confidence(hist_stats, ml_prob, float(acc or 0), vol_spike, patterns, sri)

        # Final signal decision
        final = None
        if base_sig:
            if hist_stats and hist_stats.get('count',0) >= MIN_TRADES_FOR_STATS and hist_stats.get('win_rate',0) < HIST_WINRATE_THRESHOLD:
                final = "ANTI_" + ("SHORT" if base_sig=="LONG" else "LONG")
            else: final = base_sig
        else:
            dirs = [direction_from_signal(v) for v in patterns.values() if direction_from_signal(v)]
            if dirs:
                if dirs.count('LONG') > dirs.count('SHORT'): final = 'LONG' if conf >= CONF_MIN_FOR_WATCH else None
                elif dirs.count('SHORT') > dirs.count('LONG'): final = 'SHORT' if conf >= CONF_MIN_FOR_WATCH else None

        # Plot chart with signals
        plot_symbol_chart(df, symbol, ema_cross, pre_top)

        return {
            'symbol': symbol,
            'interval': interval,
            'base_signal': base_sig,
            'final_signal': final,
            'hist_stats': hist_stats,
            'ml_prob': ml_prob,
            'ml_acc': float(acc or 0),
            'vol_spike': vol_spike,
            'orderflow': orderflow,
            'obv_div': obv_div,
            'pre_top': pre_top,
            'patterns': patterns,
            'confidence': conf,
            'sri': sri,
            'last_price': float(latest['close']),
            'timestamp': str(df.index[-1])
        }
    except Exception:
        logger.exception(f"analyze_symbol error {symbol}")
        return None

# -------------------------
# PLOT SYMBOL CHART
# -------------------------
def plot_symbol_chart(df, symbol, ema_signal=None, pre_top=None):
    plt.figure(figsize=(12,6))
    plt.plot(df.index, df['close'], label='Close', color='blue')
    plt.plot(df.index, df['ema_5'], label='EMA5', color='green')
    plt.plot(df.index, df['ema_34'], label='EMA34', color='red')
    if ema_signal:
        idx = df.index[-1]
        plt.scatter(idx, df['close'].iloc[-1], color='gold', s=100, label=f'EMA Signal: {ema_signal}')
    if pre_top:
        plt.scatter(df.index[-1], df['close'].iloc[-1], color='magenta', s=100, label='PreTop')
    plt.title(f"{symbol} Price Chart with EMA & Signals")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    chart_path = f"charts/{symbol}.png"
    os.makedirs("charts", exist_ok=True)
    plt.savefig(chart_path)
    plt.close()
    
    # =========================
# mega_bot.py — Part 5: Full Scan & Telegram Alerts
# =========================

# -------------------------
# SCAN ALL SYMBOLS
# -------------------------
def ema_scan_all(interval='5m', symbol_list=None):
    logger.info(f"Starting full scan (interval={interval})")
    symbols = symbol_list if symbol_list else get_all_usdt_symbols()
    results = []
    max_workers = max(1, min(PARALLEL_WORKERS, 12))
    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        futures = {exe.submit(analyze_symbol, s, interval): s for s in symbols}
        for fut in as_completed(futures):
            s = futures[fut]
            try:
                res = fut.result()
                if res and res.get('confidence',0) >= CONF_MIN_FOR_WATCH:
                    results.append(res)
            except Exception:
                logger.exception(f"scan future error {s}")
    logger.info(f"Scan finished hits={len(results)}")
    # sort & send messages
    if results:
        results_sorted = sorted(results, key=lambda x: x['confidence'], reverse=True)[:100]
        for r in results_sorted:
            key = f"{r['symbol']}_{interval}"
            stats_entry = ema_stats.get(key)
            label = strength_label(r['confidence'])
            event = r['final_signal'] or "WATCH"
            text = f"*EMA Alert* 🔔\nSymbol: `{r['symbol']}`\nInterval: `{r['interval']}`\nEvent: *{event}* ({label})\nPrice: `{r['last_price']}`\nConfidence: `{r['confidence']:.2f}`\nSRI: `{r['sri']:.2f}`\nTime: `{r['timestamp']}`"
            if r.get('ml_prob') is not None:
                text += f"\n🤖 ML ProbUp: `{r['ml_prob']:.2f}` | ML Acc: `{r['ml_acc']:.2f}`"
            if r.get('vol_spike'):
                text += "\n📈 Volume spike detected"
            if r.get('orderflow'):
                text += f"\n🔄 Orderflow: `{r['orderflow']}`"
            if r.get('obv_div'):
                text += f"\n📉 OBV divergence: `{r['obv_div']}`"
            if r.get('pre_top'):
                text += f"\n⚠️ PreTop detected: `{json.dumps(r['pre_top'])}`"
            if stats_entry and 'stats' in stats_entry:
                g = stats_entry['stats'].get('golden')
                d = stats_entry['stats'].get('death')
                if g and g['count']>0:
                    text += f"\n📊 Golden Cross history: count={g['count']} winrate={g['win_rate']*100:.1f}% avg_ret={g['avg_return']*100:.2f}%"
                if d and d['count']>0:
                    text += f"\n📊 Death Cross history: count={d['count']} winrate={d['win_rate']*100:.1f}% avg_ret={d['avg_return']*100:.2f}%"
            send_telegram_message(text)
            time.sleep(0.12)

# -------------------------
# MONITOR TOP SYMBOLS
# -------------------------
def monitor_top_symbols():
    logger.info("Start detailed monitor for top symbols")
    symbols = get_top_symbols_by_volume(limit=60)
    interval_list = ['15m','1h','4h','1d']
    results = []
    with ThreadPoolExecutor(max_workers=min(PARALLEL_WORKERS, 8)) as exe:
        futures = [exe.submit(analyze_symbol, s, intr) for s in symbols for intr in interval_list]
        for fut in as_completed(futures):
            try:
                res = fut.result()
                if not res: continue
                conf = res.get('confidence',0)
                ml_prob = res.get('ml_prob')
                should_alert = False
                if res.get('final_signal') and (conf >= CONF_MIN_FOR_ALERT or (ml_prob and ml_prob >= ML_PROB_THRESHOLD and res.get('ml_acc',0)>=0.55)):
                    should_alert = True
                key = f"{res['symbol']}_{res['interval']}"
                prev = state.get('signals', {}).get(key)
                if should_alert and prev != res['final_signal']:
                    label = strength_label(res.get('confidence',0))
                    patt_list = [v for v in (res.get('patterns') or {}).values() if v]
                    msg = f"⚡ *Signal*\nSymbol: `{res['symbol']}`\nInterval: `{res['interval']}`\nSignal: *{res['final_signal']}* ({label})\nPrice: `{res['last_price']}`\nConfidence: `{res['confidence']:.2f}`"
                    if res.get('hist_stats'):
                        hs = res['hist_stats']
                        msg += f"\n📊 History: WinRate {hs['win_rate']*100:.1f}% | Avg {hs['mean_return']*100:.2f}% | Trades {hs['count']}"
                    if res.get('ml_prob') is not None:
                        msg += f"\n🤖 ML ProbUp: {res['ml_prob']:.2f} | ML Acc: {res['ml_acc']:.2f}"
                    if res.get('vol_spike'): msg += "\n📈 Volume spike"
                    if patt_list: msg += f"\n🧩 Patterns: {', '.join(patt_list)}"
                    if res.get('pre_top'): msg += f"\n⚠️ PRE-TOP: {json.dumps(res['pre_top'])}"
                    send_telegram_message(msg)
                    # update state
                    state.setdefault('signals', {})[key] = res['final_signal']
                    state.setdefault('last_seen', {})[key] = {'time': res['timestamp'], 'price': res['last_price']}
                    history = state.setdefault('signal_history', {}).setdefault(res['symbol'], [])
                    history.append({'time': res['timestamp'], 'interval': res['interval'], 'signal': res['final_signal'], 'confidence': res['confidence']})
                    if len(history) > 500: state['signal_history'][res['symbol']] = history[-500:]
                    save_json_safe(STATE_FILE, state)
                results.append(res)
            except Exception:
                logger.exception("monitor future error")
    logger.info(f"Detailed monitor done, processed={len(results)}")

# -------------------------
# WARMUP TOP MODELS & EMA STATS
# -------------------------
def warmup_models_and_stats():
    logger.info("Warmup models & EMA stats for top symbols")
    symbols = get_top_symbols_by_volume(limit=30)
    for s in symbols:
        try:
            df = get_klines_df(s, '1h', limit=800)
            if df is None: continue
            model, acc = train_or_load_model(s, df, lookahead=BACKTEST_LOOKAHEAD)
            if model:
                key = f"{s}_1h"
                state.setdefault('models', {})[key] = {'acc': acc, 'trained_at': utcnow_str()}
                save_json_safe(STATE_FILE, state)
            compute_and_save_ema_stats(s, '5m')
            compute_and_save_ema_stats(s, '1h')
            time.sleep(0.6 + random.random()*0.4)
        except Exception:
            logger.exception(f"warmup error for {s}")

threading.Thread(target=warmup_models_and_stats, daemon=True).start()

# =========================
# mega_bot.py — Part 6: Scheduler, Flask & Bot Start
# ==========================

from flask import Flask, jsonify, request
from apscheduler.schedulers.background import BackgroundScheduler

# -------------------------
# FLASK APP
# -------------------------
app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({"status":"ok","time":utcnow_str()})

@app.route('/scan_now')
def scan_now():
    threading.Thread(target=run_manual_scan, daemon=True).start()
    return jsonify({"status":"scanning"})

@app.route('/status')
def status():
    return jsonify({
        "state_len": len(state.get('signals',{})),
        "ema_stats_count": len(ema_stats),
        "time": utcnow_str()
    })

@app.route(f"/telegram_webhook/<token>", methods=["POST"])
def telegram_webhook(token):
    if token != TELEGRAM_TOKEN:
        return jsonify({"ok": False, "error": "invalid token"}), 403
    data = request.get_json() or {}
    try:
        if "message" in data:
            txt = data["message"].get("text","")
            chat = data["message"]["chat"]["id"]
            if str(chat) != str(CHAT_ID):
                return jsonify({"ok": True})
            if txt.startswith("/scan_now"):
                threading.Thread(target=run_manual_scan, daemon=True).start()
                send_telegram_message("Manual scan started.")
            elif txt.startswith("/status"):
                st = {"signals": len(state.get("signals",{})), "ema_stats": len(ema_stats)}
                send_telegram_message(f"Status: {json.dumps(st)}")
    except Exception:
        logger.exception("webhook processing error")
    return jsonify({"ok": True})

# -------------------------
# MANUAL SCAN HELPER
# -------------------------
def run_manual_scan():
    try:
        ema_scan_all(interval='5m')
        monitor_top_symbols()
    except Exception:
        logger.exception("manual scan failure")

# -------------------------
# SCHEDULER
# -------------------------
scheduler = BackgroundScheduler()
scheduler.add_job(lambda: ema_scan_all(interval='5m'), 'interval', minutes=max(1, SCAN_INTERVAL_MINUTES), id='ema_scan')
scheduler.add_job(monitor_top_symbols, 'interval', minutes=max(1, MONITOR_INTERVAL_MINUTES), id='monitor_top')
scheduler.start()

# -------------------------
# MAIN START
# -------------------------
if __name__ == "__main__":
    logger.info("Starting AI Web3 Trading Hub (single-file)")
    # initial quick scan (warmup runs in background)
    try:
        threading.Thread(target=run_manual_scan, daemon=True).start()
    except Exception:
        logger.exception("initial scan failed")
    # run flask
    app.run(host="0.0.0.0", port=PORT)