# main.py (updated)
import os
import time
import json
import math
import joblib
import random
import requests
import logging
import traceback
import pandas as pd
import numpy as np
from datetime import datetime
from flask import Flask, jsonify
from apscheduler.schedulers.background import BackgroundScheduler
from concurrent.futures import ThreadPoolExecutor, as_completed
from binance.client import Client
import ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -------------------------
# CONFIG
# -------------------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
CHAT_ID = os.getenv("CHAT_ID", "")
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")

PARALLEL_WORKERS = int(os.getenv("PARALLEL_WORKERS", "6"))  # <= 8 recommended
STATE_FILE = "state.json"
MODEL_DIR = "models"
EMA_STATS_FILE = "ema_stats.json"
LOG_FILE = "bot.log"
SCAN_INTERVAL_MINUTES = int(os.getenv("SCAN_INTERVAL_MINUTES", "1"))  # EMA scan frequency
MONITOR_INTERVAL_MINUTES = int(os.getenv("MONITOR_INTERVAL_MINUTES", "5"))  # signals scan freq
EMA_SCAN_LIMIT = int(os.getenv("EMA_SCAN_LIMIT", "500"))  # klines to fetch for EMA scan
BACKTEST_LOOKAHEAD = int(os.getenv("BACKTEST_LOOKAHEAD", "5"))

# thresholds
HIST_WINRATE_THRESHOLD = 0.55  # if below -> consider anti-signal or warn
ML_PROB_THRESHOLD = 0.6
MIN_TRADES_FOR_STATS = 10
VOLUME_MULTIPLIER_THRESHOLD = 1.5  # volume spike factor vs rolling mean

os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------------
# LOGGING
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

# -------------------------
# BINANCE CLIENT
# -------------------------
client = Client(api_key=BINANCE_API_KEY, api_secret=BINANCE_API_SECRET)

# -------------------------
# FLASK
# -------------------------
app = Flask(__name__)

# -------------------------
# STATE LOAD/SAVE
# -------------------------
def load_json_safe(path, default):
    try:
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
    except Exception as e:
        logging.exception("Failed to load %s: %s", path, e)
    return default

def save_json_safe(path, data):
    try:
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp, path)
    except Exception as e:
        logging.exception("Failed to save %s: %s", path, e)

state = load_json_safe(STATE_FILE, {"signals": {}, "models": {}})
ema_stats = load_json_safe(EMA_STATS_FILE, {})

# -------------------------
# UTILITIES
# -------------------------
def send_telegram_message(text):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        logging.warning("Telegram not configured. Skipping send.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": text,
        "parse_mode": "Markdown",
        "disable_web_page_preview": True
    }
    try:
        resp = requests.post(url, json=payload, timeout=10)
        if resp.status_code != 200:
            logging.error("Telegram send failed: %s %s", resp.status_code, resp.text)
    except Exception as e:
        logging.exception("Telegram exception: %s", e)

def safe_sleep(sec):
    try:
        time.sleep(sec)
    except KeyboardInterrupt:
        raise

# -------------------------
# MARKET DATA
# -------------------------
def get_all_usdt_symbols():
    try:
        exchange_info = client.get_exchange_info()
        symbols = [s['symbol'] for s in exchange_info['symbols'] if s['quoteAsset'] == 'USDT' and s['status'] == 'TRADING']
        return symbols
    except Exception as e:
        logging.exception("Error fetching exchange info: %s", e)
        return []

def get_top_symbols_by_volume(limit=30):
    try:
        tickers = client.get_ticker()
        sorted_tickers = sorted(tickers, key=lambda x: float(x['quoteVolume']), reverse=True)
        symbols = [t['symbol'] for t in sorted_tickers if t['symbol'].endswith('USDT')]
        return symbols[:limit]
    except Exception as e:
        logging.exception("Error fetching tickers: %s", e)
        return []

def get_klines_df(symbol, interval, limit=500, retry=3):
    for attempt in range(retry):
        try:
            klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
            df = pd.DataFrame(klines, columns=[
                'open_time','open','high','low','close','volume','close_time','qav','num_trades',
                'taker_base_vol','taker_quote_vol','ignore'
            ])
            df = df[['open_time','open','high','low','close','volume']].copy()
            df[['open','high','low','close','volume']] = df[['open','high','low','close','volume']].astype(float)
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            df.set_index('open_time', inplace=True)
            return df
        except Exception as e:
            logging.warning("get_klines error %s (attempt %d/%d): %s", symbol, attempt+1, retry, e)
            safe_sleep(0.5 + attempt)
    return None

# -------------------------
# INDICATORS & STRATEGIES
# -------------------------
def apply_indicators(df):
    df = df.copy()
    try:
        df['ATR_10'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=10)
        df['ATR_50'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=50)
        df['RSI'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        macd = ta.trend.MACD(df['close'])
        df['MACD_hist'] = macd.macd_diff()
        df['Donchian_High'] = df['high'].rolling(window=20).max()
        df['Donchian_Low'] = df['low'].rolling(window=20).min()
        df['ema_8'] = ta.trend.EMAIndicator(df['close'], window=8).ema_indicator()
        df['ema_21'] = ta.trend.EMAIndicator(df['close'], window=21).ema_indicator()
        df['ema_50'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
        df['ema_200'] = ta.trend.EMAIndicator(df['close'], window=200).ema_indicator()
        df['vol_ma_20'] = df['volume'].rolling(window=20).mean()
    except Exception as e:
        logging.exception("apply_indicators error: %s", e)
    return df

def generate_signal_from_row(row):
    try:
        long_signal = (
            row['ATR_10'] < row['ATR_50'] and
            row['close'] > row['Donchian_High'] and
            (row['MACD_hist'] > 0 or row['RSI'] > 55)
        )
        short_signal = (
            row['ATR_10'] < row['ATR_50'] and
            row['close'] < row['Donchian_Low'] and
            (row['MACD_hist'] < 0 or row['RSI'] < 45)
        )
        if long_signal:
            return "LONG"
        if short_signal:
            return "SHORT"
    except Exception as e:
        logging.exception("generate_signal_from_row error: %s", e)
    return None

def detect_ema_crossover(df, short='ema_8', long='ema_21'):
    if len(df) < 3:
        return None
    a = df[short].iloc[-2]
    b = df[long].iloc[-2]
    a1 = df[short].iloc[-1]
    b1 = df[long].iloc[-1]
    if a <= b and a1 > b1:
        return "bull_cross"
    if a >= b and a1 < b1:
        return "bear_cross"
    return None

# -------------------------
# BACKTEST / HISTORICAL PATTERN ANALYSIS
# -------------------------
def backtest_pattern(df, lookahead=BACKTEST_LOOKAHEAD):
    df = df.copy()
    df = apply_indicators(df)
    results = []
    for i in range(0, len(df) - lookahead):
        row = df.iloc[i]
        sig = generate_signal_from_row(row)
        if sig:
            future_return = (df['close'].iloc[i+lookahead] / df['close'].iloc[i]) - 1
            results.append({'index': df.index[i], 'signal': sig, 'future_return': future_return})
    if not results:
        return None
    res_df = pd.DataFrame(results)
    stats = res_df.groupby('signal')['future_return'].agg(['mean','count', lambda x: (x>0).mean()]).rename(columns={'<lambda_0>':'win_rate'})
    out = {}
    for sig in stats.index:
        out[sig] = {
            'mean_return': float(stats.loc[sig,'mean']),
            'count': int(stats.loc[sig,'count']),
            'win_rate': float(stats.loc[sig,'win_rate'])
        }
    return out

# -------------------------
# MACHINE LEARNING PER SYMBOL
# -------------------------
def features_for_ml(df):
    df = df.copy()
    df['ret1'] = df['close'].pct_change(1)
    df['ret5'] = df['close'].pct_change(5)
    df['vol_change'] = df['volume'] / (df['vol_ma_20'] + 1e-9)
    df['ema8_21_diff'] = df['ema_8'] - df['ema_21']
    df['ema21_50_diff'] = df['ema_21'] - df['ema_50']
    df = df.dropna()
    return df

def train_or_load_model(symbol, df, lookahead=5):
    model_path = os.path.join(MODEL_DIR, f"{symbol}_rf.joblib")
    df = apply_indicators(df)
    df = features_for_ml(df)
    if len(df) < 300:
        logging.info("%s not enough rows for ML (need >= 300).", symbol)
        return None, None
    df['future_ret'] = df['close'].shift(-lookahead) / df['close'] - 1
    df = df.dropna()
    df['target'] = (df['future_ret'] > 0).astype(int)
    features = ['ATR_10','ATR_50','RSI','MACD_hist','ret1','ret5','vol_change','ema8_21_diff','ema21_50_diff']
    X = df[features]
    y = df['target']
    # load safely: check size and try/except
    if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
        try:
            model = joblib.load(model_path)
            logging.info("Loaded model for %s", symbol)
            # quick evaluation
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
            acc = accuracy_score(y_test, model.predict(X_test))
            return model, float(acc)
        except Exception as e:
            logging.exception("Loading model failed, will retrain: %s", e)
            # remove corrupted file to avoid repeated loading attempts
            try:
                os.remove(model_path)
                logging.info("Removed corrupted model file %s", model_path)
            except Exception:
                pass
    # train
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        model = RandomForestClassifier(n_estimators=150, n_jobs=1, random_state=42)
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        # safe save via tmp file then replace
        tmp_path = model_path + ".tmp"
        joblib.dump(model, tmp_path)
        os.replace(tmp_path, model_path)
        logging.info("Trained and saved model for %s (acc=%.3f)", symbol, acc)
        return model, float(acc)
    except Exception as e:
        logging.exception("Training model failed for %s: %s", symbol, e)
        return None, None

# -------------------------
# EMA HISTORICAL STATS (NEW)
# -------------------------
def compute_ema_historical_stats(df, short_col='ema_8', long_col='ema_21', lookahead=10):
    """
    Scans historical df for cross events (short crossing long)
    and measures returns lookahead bars after the cross.
    Returns dict with golden/death stats.
    """
    df = df.copy()
    df = apply_indicators(df)
    res = {'golden': {'count': 0, 'win_rate': None, 'avg_return': None, 'returns': []},
           'death': {'count': 0, 'win_rate': None, 'avg_return': None, 'returns': []}}
    for i in range(1, len(df)-lookahead):
        prev_short = df[short_col].iloc[i-1]
        prev_long = df[long_col].iloc[i-1]
        cur_short = df[short_col].iloc[i]
        cur_long = df[long_col].iloc[i]
        # golden cross
        if prev_short <= prev_long and cur_short > cur_long:
            fut = (df['close'].iloc[i+lookahead] / df['close'].iloc[i]) - 1
            res['golden']['returns'].append(fut)
        # death cross
        if prev_short >= prev_long and cur_short < cur_long:
            fut = (df['close'].iloc[i+lookahead] / df['close'].iloc[i]) - 1
            res['death']['returns'].append(fut)
    for k in ['golden', 'death']:
        r = res[k]['returns']
        if r:
            arr = np.array(r)
            res[k]['count'] = len(r)
            res[k]['win_rate'] = float((arr > 0).mean())
            res[k]['avg_return'] = float(arr.mean())
        else:
            res[k]['count'] = 0
            res[k]['win_rate'] = None
            res[k]['avg_return'] = None
    return res

def ensure_ema_stats_for_symbol(symbol, interval='5m', force_recompute=False):
    key = f"{symbol}_{interval}"
    if not force_recompute and key in ema_stats:
        return ema_stats[key]
    df = get_klines_df(symbol, interval, limit=EMA_SCAN_LIMIT)
    if df is None or len(df) < 50:
        return None
    stats = compute_ema_historical_stats(df, short_col='ema_8', long_col='ema_21', lookahead=BACKTEST_LOOKAHEAD)
    ema_stats[key] = {
        'computed_at': str(datetime.utcnow()),
        'lookahead': BACKTEST_LOOKAHEAD,
        'stats': stats
    }
    save_json_safe(EMA_STATS_FILE, ema_stats)
    return ema_stats[key]

# -------------------------
# SIGNAL SCORING & DECISION
# -------------------------
def compute_confidence(hist_stats, ml_prob, ml_acc, volume_spike):
    score = 0.0
    if hist_stats and 'win_rate' in hist_stats and hist_stats['win_rate'] is not None:
        score += min(1.0, hist_stats['win_rate']) * 0.5
    if ml_prob is not None:
        score += ml_prob * 0.3
    if ml_acc:
        score += min(0.2, ml_acc * 0.2)
    if volume_spike:
        score += 0.05
    return min(1.0, score)

# -------------------------
# SCAN WORKER FOR A SYMBOL
# -------------------------
def analyze_symbol(symbol, interval='1h'):
    try:
        df = get_klines_df(symbol, interval, limit=EMA_SCAN_LIMIT)
        if df is None or len(df) < 60:
            return None
        df = apply_indicators(df)
        latest = df.iloc[-1]
        base_signal = generate_signal_from_row(latest)
        hist = backtest_pattern(df, lookahead=BACKTEST_LOOKAHEAD)
        hist_stats = None
        if hist and base_signal in hist:
            hist_stats = hist[base_signal]
        model, acc = train_or_load_model(symbol, df, lookahead=BACKTEST_LOOKAHEAD)
        ml_prob = None
        if model is not None:
            # create DataFrame with same feature names to avoid sklearn warning
            feat_row = features_for_ml(df).iloc[-1]
            features = ['ATR_10','ATR_50','RSI','MACD_hist','ret1','ret5','vol_change','ema8_21_diff','ema21_50_diff']
            X_row = pd.DataFrame([feat_row[features].values], columns=features)
            ml_prob = float(model.predict_proba(X_row)[0][1])
        vol_spike = False
        if latest['volume'] > (latest['vol_ma_20'] * VOLUME_MULTIPLIER_THRESHOLD):
            vol_spike = True
        ema_cross = detect_ema_crossover(df, short='ema_8', long='ema_21')
        final_signal = None
        if base_signal:
            if hist_stats and hist_stats['count'] >= MIN_TRADES_FOR_STATS and hist_stats['win_rate'] < HIST_WINRATE_THRESHOLD:
                final_signal = "ANTI_"+("SHORT" if base_signal=="LONG" else "LONG")
            else:
                final_signal = base_signal
            conf = compute_confidence(hist_stats, ml_prob, acc, vol_spike)
            return {
                'symbol': symbol,
                'interval': interval,
                'base_signal': base_signal,
                'final_signal': final_signal,
                'hist_stats': hist_stats,
                'ml_prob': ml_prob,
                'ml_acc': acc,
                'vol_spike': vol_spike,
                'ema_cross': ema_cross,
                'confidence': conf,
                'last_price': float(latest['close']),
                'timestamp': str(df.index[-1])
            }
        else:
            if ema_cross:
                conf = 0.25 + (0.1 if vol_spike else 0.0)
                return {
                    'symbol': symbol,
                    'interval': interval,
                    'base_signal': None,
                    'final_signal': 'EMA_'+ema_cross,
                    'hist_stats': None,
                    'ml_prob': ml_prob,
                    'ml_acc': acc,
                    'vol_spike': vol_spike,
                    'ema_cross': ema_cross,
                    'confidence': conf,
                    'last_price': float(latest['close']),
                    'timestamp': str(df.index[-1])
                }
    except Exception as e:
        logging.exception("analyze_symbol error %s: %s", symbol, e)
    return None

# -------------------------
# EMA SCAN (ALL SYMBOLS) - parallel (uses EMA stats)
# -------------------------
def ema_scan_all(interval='5m'):
    logging.info("Starting EMA scan for all symbols (interval=%s)...", interval)
    symbols = get_all_usdt_symbols()
    results = []
    max_workers = PARALLEL_WORKERS
    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        futures = {exe.submit(analyze_symbol, s, interval): s for s in symbols}
        for fut in as_completed(futures):
            s = futures[fut]
            try:
                res = fut.result()
                if res and res.get('confidence', 0) >= 0.3:
                    results.append(res)
            except Exception as e:
                logging.exception("Future error for %s: %s", s, e)
    logging.info("EMA scan finished, hits=%d", len(results))
    if results:
        results_sorted = sorted(results, key=lambda x: x['confidence'], reverse=True)[:50]
        for r in results_sorted:
            # ensure ema stats available (compute if missing)
            key = f"{r['symbol']}_{interval}"
            stats_entry = ema_stats.get(key)
            if not stats_entry:
                stats_entry = ensure_ema_stats_for_symbol(r['symbol'], interval=interval, force_recompute=False)
            # build message
            text = "*EMA Alert* 🔔\n"
            text += f"Symbol: `{r['symbol']}`\nInterval: `{r['interval']}`\nEvent: *{r['final_signal']}* (ema8/21)\nPrice: `{r['last_price']}`\nConfidence: `{r['confidence']:.2f}`\nTime: `{r['timestamp']}`"
            if r['ml_prob'] is not None:
                text += f"\n🤖 ML Prob Up: `{r['ml_prob']:.2f}` | ML Acc: `{r['ml_acc']:.2f}`"
            if r['vol_spike']:
                text += "\n📈 Volume spike detected"
            # append historical EMA stats if available
            if stats_entry and 'stats' in stats_entry:
                g = stats_entry['stats'].get('golden')
                d = stats_entry['stats'].get('death')
                if g and g['count'] > 0:
                    text += f"\n📊 Golden Cross history: count={g['count']} winrate={g['win_rate']*100:.1f}% avg_ret={g['avg_return']*100:.2f}%"
                if d and d['count'] > 0:
                    text += f"\n📊 Death Cross history: count={d['count']} winrate={d['win_rate']*100:.1f}% avg_ret={d['avg_return']*100:.2f}%"
            send_telegram_message(text)
            safe_sleep(0.12)

# -------------------------
# MONITOR TOP SYMBOLS (DETAILED SIGNALS)
# -------------------------
def monitor_top_symbols():
    logging.info("Start detailed monitor for top symbols")
    symbols = get_top_symbols_by_volume(limit=60)
    interval_list = ['15m','1h','4h','1d']
    results = []
    with ThreadPoolExecutor(max_workers=min(PARALLEL_WORKERS, 8)) as exe:
        futures = []
        for s in symbols:
            for intr in interval_list:
                futures.append(exe.submit(analyze_symbol, s, intr))
        for fut in as_completed(futures):
            try:
                res = fut.result()
                if res:
                    conf = res.get('confidence', 0)
                    ml_prob = res.get('ml_prob')
                    should_alert = False
                    if res.get('final_signal') and (conf >= 0.45 or (ml_prob and ml_prob >= ML_PROB_THRESHOLD)):
                        should_alert = True
                    key = f"{res['symbol']}_{res['interval']}"
                    prev = state.get('signals', {}).get(key)
                    if should_alert and prev != res['final_signal']:
                        hist = res.get('hist_stats')
                        hist_text = ""
                        if hist:
                            hist_text = f"\n📊 History: WinRate {hist['win_rate']*100:.1f}% | Avg {hist['mean_return']*100:.2f}% | Trades {hist['count']}"
                        ml_text = ""
                        if res.get('ml_prob') is not None:
                            ml_text = f"\n🤖 ML ProbUp: {res['ml_prob']:.2f} | ML Acc: {res['ml_acc']:.2f}"
                        vol_text = "\n📈 Volume spike" if res.get('vol_spike') else ""
                        msg = f"⚡ *Signal*\nSymbol: `{res['symbol']}`\nInterval: `{res['interval']}`\nSignal: *{res['final_signal']}* (base: {res.get('base_signal')})\nPrice: `{res['last_price']}`\nConfidence: `{res['confidence']:.2f}`{hist_text}{ml_text}{vol_text}\nTime: `{res['timestamp']}`"
                        send_telegram_message(msg)
                        state.setdefault('signals', {})[key] = res['final_signal']
                        state.setdefault('last_seen', {})[key] = {'time': res['timestamp'], 'price': res['last_price']}
                        save_json_safe(STATE_FILE, state)
                    results.append(res)
            except Exception as e:
                logging.exception("monitor future error: %s", e)
    logging.info("Detailed monitor done, processed=%d", len(results))

# -------------------------
# SCHEDULER
# -------------------------
scheduler = BackgroundScheduler()
scheduler.add_job(lambda: ema_scan_all(interval='5m'), 'interval', minutes=max(1, SCAN_INTERVAL_MINUTES), id='ema_scan')
scheduler.add_job(monitor_top_symbols, 'interval', minutes=max(1, MONITOR_INTERVAL_MINUTES), id='monitor_top')
scheduler.start()

# -------------------------
# HTTP ROUTES
# -------------------------
@app.route('/')
def home():
    return jsonify({"status": "ok", "time": str(datetime.utcnow())})

@app.route('/scan_now')
def scan_now():
    try:
        ema_scan_all(interval='5m')
        monitor_top_symbols()
        return jsonify({"status": "scanned"})
    except Exception as e:
        logging.exception("scan_now error: %s", e)
        return jsonify({"error": str(e)}), 500

@app.route('/status')
def status():
    out = {
        "state": state,
        "ema_stats_summary_count": len(ema_stats),
        "time": str(datetime.utcnow())
    }
    return jsonify(out)

# -------------------------
# BOOTSTRAP: Warm-up models and EMA-stats for top symbols (background)
# -------------------------
def warmup_models_and_stats():
    logging.info("Warmup models and EMA stats for top symbols")
    symbols = get_top_symbols_by_volume(limit=30)
    for s in symbols:
        try:
            df = get_klines_df(s, '1h', limit=800)
            if df is None:
                continue
            # train/load model
            train_or_load_model(s, df, lookahead=BACKTEST_LOOKAHEAD)
            # compute ema stats for 5m and 1h (slow)
            ensure_ema_stats_for_symbol(s, interval='5m', force_recompute=False)
            ensure_ema_stats_for_symbol(s, interval='1h', force_recompute=False)
            safe_sleep(0.5 + random.random()*0.5)
        except Exception as e:
            logging.exception("warmup error for %s: %s", s, e)

import threading
t = threading.Thread(target=warmup_models_and_stats, daemon=True)
t.start()

# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    logging.info("Starting AI Futures/Signals Bot (updated)")
    try:
        ema_scan_all(interval='5m')
    except Exception as e:
        logging.exception("Initial ema_scan failed: %s", e)
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)