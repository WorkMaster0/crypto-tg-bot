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
def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {"signals": {}, "models": {}}

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)

state = load_state()

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

def tick_to_datetime(tick):
    return pd.to_datetime(tick, unit="ms")

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
        df['vol_ma_20'] = df['volume'].rolling(window=20).mean()
    except Exception as e:
        logging.exception("apply_indicators error: %s", e)
    return df

def generate_signal_from_row(row):
    # conservative: require ATR_10 < ATR_50 and donchian break + momentum
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

# EMA crossover detector
def detect_ema_crossover(df, short='ema_8', long='ema_21'):
    # returns 'bull_cross' or 'bear_cross' or None (checks last two bars)
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
    # convert to friendly format
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
    # produce feature matrix
    df = df.copy()
    df['ret1'] = df['close'].pct_change(1)
    df['ret5'] = df['close'].pct_change(5)
    df['vol_change'] = df['volume'] / (df['vol_ma_20'] + 1e-9)
    df['ema8_21_diff'] = df['ema_8'] - df['ema_21']
    df['ema21_50_diff'] = df['ema_21'] - df['ema_50']
    # drop NA
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
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            logging.info("Loaded model for %s", symbol)
            # quick evaluation
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
            acc = accuracy_score(y_test, model.predict(X_test))
            return model, float(acc)
        except Exception as e:
            logging.exception("Loading model failed, will retrain: %s", e)
    # train
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        model = RandomForestClassifier(n_estimators=150, n_jobs=1, random_state=42)
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        joblib.dump(model, model_path)
        logging.info("Trained and saved model for %s (acc=%.3f)", symbol, acc)
        return model, float(acc)
    except Exception as e:
        logging.exception("Training model failed for %s: %s", symbol, e)
        return None, None

# -------------------------
# SIGNAL SCORING & DECISION
# -------------------------
def compute_confidence(hist_stats, ml_prob, ml_acc, volume_spike):
    # combine pieces into a 0..1 score
    score = 0.0
    # history
    if hist_stats and 'win_rate' in hist_stats:
        score += min(1.0, hist_stats['win_rate']) * 0.5  # up to 0.5
    # ml
    if ml_prob is not None:
        score += ml_prob * 0.3  # up to 0.3
    # ml acc bonus
    if ml_acc:
        score += min(0.2, ml_acc * 0.2)  # up to 0.2
    # volume spike penalization or bonus
    if volume_spike:
        # if volume spike in direction of signal -> small bonus
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
        # generate base signal
        base_signal = generate_signal_from_row(latest)
        # backtest
        hist = backtest_pattern(df, lookahead=BACKTEST_LOOKAHEAD)
        hist_stats = None
        if hist and base_signal in hist:
            hist_stats = hist[base_signal]
        # ml
        model, acc = train_or_load_model(symbol, df, lookahead=BACKTEST_LOOKAHEAD)
        ml_prob = None
        if model is not None:
            feat_row = features_for_ml(df).iloc[-1]
            features = ['ATR_10','ATR_50','RSI','MACD_hist','ret1','ret5','vol_change','ema8_21_diff','ema21_50_diff']
            X_row = feat_row[features].values.reshape(1, -1)
            ml_prob = float(model.predict_proba(X_row)[0][1])
        # volume spike
        vol_spike = False
        if latest['volume'] > (latest['vol_ma_20'] * VOLUME_MULTIPLIER_THRESHOLD):
            vol_spike = True
        # EMA crossover
        ema_cross = detect_ema_crossover(df, short='ema_8', long='ema_21')
        # decide final signal with anti-pattern logic
        final_signal = None
        reason = []
        if base_signal:
            # if history shows poor performance -> anti-signal or warning
            if hist_stats and hist_stats['count'] >= MIN_TRADES_FOR_STATS and hist_stats['win_rate'] < HIST_WINRATE_THRESHOLD:
                final_signal = "ANTI_"+("SHORT" if base_signal=="LONG" else "LONG")
                reason.append("anti-pattern (low hist winrate)")
            else:
                final_signal = base_signal
            # compute confidence
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
            # maybe EMA crossover-only alert
            if ema_cross:
                # small confidence derived from crossover + volume
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
# EMA SCAN (ALL SYMBOLS) - parallel
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
                if res and res.get('confidence', 0) >= 0.3:  # threshold for notification
                    results.append(res)
            except Exception as e:
                logging.exception("Future error for %s: %s", s, e)
    logging.info("EMA scan finished, hits=%d", len(results))
    # send aggregated messages (limit messages to avoid spam)
    if results:
        # sort by confidence
        results_sorted = sorted(results, key=lambda x: x['confidence'], reverse=True)[:50]  # top 50
        for r in results_sorted:
            text = "*EMA Alert* 🔔\n"
            text += f"Symbol: `{r['symbol']}`\nInterval: `{r['interval']}`\nEvent: *{r['final_signal']}* (ema8/21)\nPrice: `{r['last_price']}`\nConfidence: `{r['confidence']:.2f}`\nTime: `{r['timestamp']}`"
            if r['ml_prob'] is not None:
                text += f"\n🤖 ML Prob Up: `{r['ml_prob']:.2f}` | ML Acc: `{r['ml_acc']:.2f}`"
            if r['vol_spike']:
                text += "\n📈 Volume spike detected"
            send_telegram_message(text)
            # small sleep to avoid Telegram flood
            safe_sleep(0.1)

# -------------------------
# MONITOR TOP SYMBOLS (DETAILED SIGNALS)
# -------------------------
def monitor_top_symbols():
    logging.info("Start detailed monitor for top symbols")
    symbols = get_top_symbols_by_volume(limit=60)  # expand as needed
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
                    # alert threshold: final_signal exists and confidence >= 0.45 or ML high prob
                    conf = res.get('confidence', 0)
                    ml_prob = res.get('ml_prob')
                    should_alert = False
                    if res.get('final_signal') and (conf >= 0.45 or (ml_prob and ml_prob >= ML_PROB_THRESHOLD)):
                        should_alert = True
                    # Avoid spam: check state
                    key = f"{res['symbol']}_{res['interval']}"
                    prev = state.get('signals', {}).get(key)
                    if should_alert and prev != res['final_signal']:
                        # construct message
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
                        # update state
                        state.setdefault('signals', {})[key] = res['final_signal']
                        # save last seen details
                        state.setdefault('last_seen', {})[key] = {'time': res['timestamp'], 'price': res['last_price']}
                        save_state(state)
                    # store for logging
                    results.append(res)
            except Exception as e:
                logging.exception("monitor future error: %s", e)
    logging.info("Detailed monitor done, processed=%d", len(results))

# -------------------------
# SCHEDULER
# -------------------------
scheduler = BackgroundScheduler()

# EMA scan job: frequent (1m/5m)
scheduler.add_job(lambda: ema_scan_all(interval='5m'), 'interval', minutes=max(1, SCAN_INTERVAL_MINUTES), id='ema_scan')

# Detailed monitor job: less frequent
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
    return jsonify(state)

# -------------------------
# BOOTSTRAP: Warm-up models for top symbols (background)
# -------------------------
def warmup_models():
    logging.info("Warmup models for top symbols")
    symbols = get_top_symbols_by_volume(limit=30)
    for s in symbols:
        try:
            df = get_klines_df(s, '1h', limit=800)
            if df is None:
                continue
            train_or_load_model(s, df, lookahead=BACKTEST_LOOKAHEAD)
            # small pause to respect rate-limits
            safe_sleep(0.5 + random.random()*0.5)
        except Exception as e:
            logging.exception("warmup error for %s: %s", s, e)

# run warmup in background thread to avoid blocking start
import threading
t = threading.Thread(target=warmup_models, daemon=True)
t.start()

# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    logging.info("Starting AI Futures/Signals Bot")
    # fire an initial quick scan
    try:
        ema_scan_all(interval='5m')
    except Exception as e:
        logging.exception("Initial ema_scan failed: %s", e)
    # run flask
    app.run(host='0.0.0.0', port=int(os.getenv("PORT", "5000")))