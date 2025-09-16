# main.py — enhanced with 10 extra features and robust Telegram messaging
import os
import time
import json
import math
import joblib
import random
import requests
import logging
import traceback
import re
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

PARALLEL_WORKERS = int(os.getenv("PARALLEL_WORKERS", "6"))  # <=8 recommended
STATE_FILE = "state.json"
MODEL_DIR = "models"
EMA_STATS_FILE = "ema_stats.json"
LOG_FILE = "bot.log"
SCAN_INTERVAL_MINUTES = int(os.getenv("SCAN_INTERVAL_MINUTES", "1"))
MONITOR_INTERVAL_MINUTES = int(os.getenv("MONITOR_INTERVAL_MINUTES", "5"))
EMA_SCAN_LIMIT = int(os.getenv("EMA_SCAN_LIMIT", "500"))
BACKTEST_LOOKAHEAD = int(os.getenv("BACKTEST_LOOKAHEAD", "5"))

# thresholds
HIST_WINRATE_THRESHOLD = 0.55
ML_PROB_THRESHOLD = 0.6
MIN_TRADES_FOR_STATS = 10
VOLUME_MULTIPLIER_THRESHOLD = 1.5

# confidence weights (tunable)
CONF_WEIGHTS = {
    "hist": 0.30,
    "ml_prob": 0.25,
    "ml_acc": 0.10,
    "pattern_agreement": 0.20,
    "volume": 0.05,
    "sri": 0.10  # Signal Reliability Index contribution
}

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

state = load_json_safe(STATE_FILE, {"signals": {}, "models": {}, "signal_history": {}})
ema_stats = load_json_safe(EMA_STATS_FILE, {})

# -------------------------
# UTILITIES
# -------------------------
def escape_markdown_v2(text: str) -> str:
    # Escape characters for MarkdownV2 as per Telegram requirements
    if not isinstance(text, str):
        text = str(text)
    escape_chars = r"_*[]()~`>#+-=|{}.!"
    return re.sub(f"([{re.escape(escape_chars)}])", r"\\\1", text)

def send_telegram_message(text):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        logging.warning("Telegram not configured. Skipping send.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": escape_markdown_v2(text),
        "parse_mode": "MarkdownV2",
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
    if df is None or len(df) == 0:
        return df
    try:
        # ATRs
        if len(df) >= 10:
            df['ATR_10'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=10)
        else:
            df['ATR_10'] = np.nan
        if len(df) >= 50:
            df['ATR_50'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=50)
        else:
            df['ATR_50'] = np.nan

        # RSI
        if len(df) >= 14:
            df['RSI'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        else:
            df['RSI'] = np.nan

        # MACD hist
        macd = ta.trend.MACD(df['close'])
        df['MACD_hist'] = macd.macd_diff()

        # Donchian
        df['Donchian_High'] = df['high'].rolling(window=20).max()
        df['Donchian_Low'] = df['low'].rolling(window=20).min()

        # EMAs
        df['ema_8'] = ta.trend.EMAIndicator(df['close'], window=8).ema_indicator()
        df['ema_21'] = ta.trend.EMAIndicator(df['close'], window=21).ema_indicator()
        df['ema_50'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
        df['ema_200'] = ta.trend.EMAIndicator(df['close'], window=200).ema_indicator()

        # Bollinger Bands
        if len(df) >= 20:
            bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
            df['bb_h'] = bb.bollinger_hband()
            df['bb_l'] = bb.bollinger_lband()
        else:
            df['bb_h'] = np.nan
            df['bb_l'] = np.nan

        # Volume MA
        df['vol_ma_20'] = df['volume'].rolling(window=20).mean()

        # ADX
        if len(df) >= 14:
            adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14)
            df['ADX'] = adx.adx()
            df['ADX_pos'] = adx.adx_pos()
            df['ADX_neg'] = adx.adx_neg()
        else:
            df['ADX'] = np.nan
            df['ADX_pos'] = np.nan
            df['ADX_neg'] = np.nan

        # StochRSI (if available in ta version)
        try:
            if len(df) >= 14:
                stochrsi = ta.momentum.StochRSIIndicator(df['close'], window=14, smooth1=3, smooth2=3)
                df['stochrsi_k'] = stochrsi.stochrsi_k()
                df['stochrsi_d'] = stochrsi.stochrsi_d()
            else:
                df['stochrsi_k'] = np.nan
                df['stochrsi_d'] = np.nan
        except Exception:
            # fallback to Stochastic oscillator on price (less ideal)
            try:
                so = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], window=14, smooth_window=3)
                df['stochrsi_k'] = so.stoch()
                df['stochrsi_d'] = so.stoch_signal()
            except Exception:
                df['stochrsi_k'] = np.nan
                df['stochrsi_d'] = np.nan

        # OBV
        try:
            obv = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume'])
            df['OBV'] = obv.on_balance_volume()
        except Exception:
            df['OBV'] = np.nan

        # CCI
        try:
            if len(df) >= 20:
                cci = ta.trend.CCIIndicator(df['high'], df['low'], df['close'], window=20)
                df['CCI'] = cci.cci()
            else:
                df['CCI'] = np.nan
        except Exception:
            df['CCI'] = np.nan

        # volatility regime: rolling std of returns
        df['ret1'] = df['close'].pct_change(1)
        df['ret20_std'] = df['ret1'].rolling(window=20).std()

        # Heikin-Ashi candles
        ha_close = (df['open'] + df['high'] + df['low'] + df['close']) / 4
        ha_open = df['open'].copy()
        # produce HA open iteratively for accuracy
        ha_open_vals = []
        prev_ha_open = (df['open'].iloc[0] + df['close'].iloc[0]) / 2
        prev_ha_close = ha_close.iloc[0]
        ha_open_vals.append(prev_ha_open)
        for i in range(1, len(df)):
            prev_ha_open = (ha_open_vals[-1] + prev_ha_close) / 2
            ha_open_vals.append(prev_ha_open)
            prev_ha_close = ha_close.iloc[i]
        df['ha_open'] = ha_open_vals
        df['ha_close'] = ha_close
        df['ha_high'] = df[['high', 'ha_open', 'ha_close']].max(axis=1)
        df['ha_low'] = df[['low', 'ha_open', 'ha_close']].min(axis=1)

        # Pivot points previous bar (classic)
        if len(df) >= 2:
            ph = df['high'].iloc[-2]
            pl = df['low'].iloc[-2]
            pc = df['close'].iloc[-2]
            P = (ph + pl + pc) / 3.0
            R1 = 2 * P - pl
            S1 = 2 * P - ph
            df['pivot_R1'] = R1
            df['pivot_S1'] = S1
        else:
            df['pivot_R1'] = np.nan
            df['pivot_S1'] = np.nan

    except Exception as e:
        logging.exception("apply_indicators error: %s", e)
    return df

def generate_signal_from_row(row):
    try:
        long_signal = (
            row.get('ATR_10', np.nan) < row.get('ATR_50', np.nan) and
            row['close'] > row.get('Donchian_High', np.nan) and
            (row.get('MACD_hist', 0) > 0 or row.get('RSI', 0) > 55)
        )
        short_signal = (
            row.get('ATR_10', np.nan) < row.get('ATR_50', np.nan) and
            row['close'] < row.get('Donchian_Low', np.nan) and
            (row.get('MACD_hist', 0) < 0 or row.get('RSI', 0) < 45)
        )
        if long_signal:
            return "LONG"
        if short_signal:
            return "SHORT"
    except Exception as e:
        logging.exception("generate_signal_from_row error: %s", e)
    return None

def detect_ema_crossover(df, short='ema_8', long='ema_21'):
    if df is None or len(df) < 3:
        return None
    a = df[short].iloc[-2]
    b = df[long].iloc[-2]
    a1 = df[short].iloc[-1]
    b1 = df[long].iloc[-1]
    if pd.isna(a) or pd.isna(b) or pd.isna(a1) or pd.isna(b1):
        return None
    if a <= b and a1 > b1:
        return "bull_cross"
    if a >= b and a1 < b1:
        return "bear_cross"
    return None

# --- New detectors (10 features) ---
def detect_bollinger_breakout(df):
    if df is None or len(df) < 20:
        return None
    h = df['bb_h'].iloc[-1]
    l = df['bb_l'].iloc[-1]
    c = df['close'].iloc[-1]
    if pd.isna(h) or pd.isna(l):
        return None
    if c > h:
        return 'bb_up'
    if c < l:
        return 'bb_down'
    return None

def detect_rsi_flip(df):
    if df is None or len(df) < 15:
        return None
    prev = df['RSI'].iloc[-2]
    cur = df['RSI'].iloc[-1]
    if pd.isna(prev) or pd.isna(cur):
        return None
    if prev < 30 and cur >= 30:
        return 'rsi_up'
    if prev > 70 and cur <= 70:
        return 'rsi_down'
    return None

def detect_macd_signal(df):
    if df is None or len(df) < 3:
        return None
    prev = df['MACD_hist'].iloc[-2]
    cur = df['MACD_hist'].iloc[-1]
    if pd.isna(prev) or pd.isna(cur):
        return None
    if prev <= 0 and cur > 0:
        return 'macd_up'
    if prev >= 0 and cur < 0:
        return 'macd_down'
    return None

def detect_ma50_200_cross(df):
    if df is None or len(df) < 201:
        return None
    a = df['ema_50'].iloc[-2]
    b = df['ema_200'].iloc[-2]
    a1 = df['ema_50'].iloc[-1]
    b1 = df['ema_200'].iloc[-1]
    if pd.isna(a) or pd.isna(b) or pd.isna(a1) or pd.isna(b1):
        return None
    if a <= b and a1 > b1:
        return 'ma50_up'
    if a >= b and a1 < b1:
        return 'ma50_down'
    return None

def detect_adx(df):
    if df is None or len(df) < 14:
        return None
    adx = df['ADX'].iloc[-1]
    adx_pos = df['ADX_pos'].iloc[-1] if 'ADX_pos' in df.columns else np.nan
    adx_neg = df['ADX_neg'].iloc[-1] if 'ADX_neg' in df.columns else np.nan
    if pd.isna(adx):
        return None
    if adx >= 25:
        if not pd.isna(adx_pos) and adx_pos > adx_neg:
            return 'adx_strong_up'
        if not pd.isna(adx_neg) and adx_neg > adx_pos:
            return 'adx_strong_down'
        return 'adx_strong'
    return None

def detect_stochrsi(df):
    if df is None or len(df) < 14:
        return None
    k = df.get('stochrsi_k', pd.Series([np.nan])).iloc[-1]
    d = df.get('stochrsi_d', pd.Series([np.nan])).iloc[-1]
    if pd.isna(k) or pd.isna(d):
        return None
    if (df['stochrsi_k'].iloc[-2] <= df['stochrsi_d'].iloc[-2]) and (k > d):
        return 'stoch_up'
    if (df['stochrsi_k'].iloc[-2] >= df['stochrsi_d'].iloc[-2]) and (k < d):
        return 'stoch_down'
    return None

def detect_obv_divergence(df):
    # simple divergence: price up + OBV down => divergence_down; price down + OBV up => divergence_up
    if df is None or len(df) < 10:
        return None
    prev_price = df['close'].iloc[-2]
    cur_price = df['close'].iloc[-1]
    prev_obv = df['OBV'].iloc[-2] if 'OBV' in df.columns else np.nan
    cur_obv = df['OBV'].iloc[-1] if 'OBV' in df.columns else np.nan
    if pd.isna(prev_obv) or pd.isna(cur_obv):
        return None
    if cur_price > prev_price and cur_obv < prev_obv:
        return 'obv_div_down'
    if cur_price < prev_price and cur_obv > prev_obv:
        return 'obv_div_up'
    return None

def detect_cci(df):
    if df is None or len(df) < 20:
        return None
    cci = df['CCI'].iloc[-1] if 'CCI' in df.columns else np.nan
    if pd.isna(cci):
        return None
    if cci > 100:
        return 'cci_overbought'
    if cci < -100:
        return 'cci_oversold'
    return None

def detect_atr_breakout(df, mult=1.5):
    if df is None or len(df) < 10:
        return None
    atr = df['ATR_10'].iloc[-1] if 'ATR_10' in df.columns else np.nan
    if pd.isna(atr):
        return None
    last_close = df['close'].iloc[-1]
    prev_close = df['close'].iloc[-2]
    if last_close > prev_close + mult * atr:
        return 'atr_break_up'
    if last_close < prev_close - mult * atr:
        return 'atr_break_down'
    return None

def detect_heikin_ashi_trend(df, length=3):
    if df is None or len(df) < length + 1:
        return None
    ha_close = df['ha_close']
    # check last `length` HA candles all green/red
    last = ha_close.iloc[-length:]
    prev = ha_close.iloc[-length-1:-1]
    if last.isnull().any():
        return None
    # green if ha_close > ha_open
    greens = (df['ha_close'].iloc[-length:] > df['ha_open'].iloc[-length:]).all()
    reds = (df['ha_close'].iloc[-length:] < df['ha_open'].iloc[-length:]).all()
    if greens:
        return 'ha_up'
    if reds:
        return 'ha_down'
    return None

def detect_pivot_probe(df):
    # price crosses previous pivot R1 or S1
    if df is None or len(df) < 2:
        return None
    r1 = df.get('pivot_R1', pd.Series([np.nan])).iloc[-1]
    s1 = df.get('pivot_S1', pd.Series([np.nan])).iloc[-1]
    c = df['close'].iloc[-1]
    if pd.isna(r1) or pd.isna(s1):
        return None
    if c > r1:
        return 'pivot_break_r1'
    if c < s1:
        return 'pivot_break_s1'
    return None

# -------------------------
# BACKTEST / HISTORICAL PATTERN ANALYSIS (unchanged)
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
# MACHINE LEARNING PER SYMBOL (unchanged)
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
    if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
        try:
            model = joblib.load(model_path)
            logging.info("Loaded model for %s", symbol)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
            acc = accuracy_score(y_test, model.predict(X_test))
            return model, float(acc)
        except Exception as e:
            logging.exception("Loading model failed, will retrain: %s", e)
            try:
                os.remove(model_path)
                logging.info("Removed corrupted model file %s", model_path)
            except Exception:
                pass
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        model = RandomForestClassifier(n_estimators=150, n_jobs=1, random_state=42)
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        tmp_path = model_path + ".tmp"
        joblib.dump(model, tmp_path)
        os.replace(tmp_path, model_path)
        logging.info("Trained and saved model for %s (acc=%.3f)", symbol, acc)
        return model, float(acc)
    except Exception as e:
        logging.exception("Training model failed for %s: %s", symbol, e)
        return None, None

# -------------------------
# EMA HISTORICAL STATS (unchanged)
# -------------------------
def compute_ema_historical_stats(df, short_col='ema_8', long_col='ema_21', lookahead=10):
    df = df.copy()
    df = apply_indicators(df)
    res = {'golden': {'count': 0, 'win_rate': None, 'avg_return': None, 'returns': []},
           'death': {'count': 0, 'win_rate': None, 'avg_return': None, 'returns': []}}
    for i in range(1, len(df)-lookahead):
        prev_short = df[short_col].iloc[i-1]
        prev_long = df[long_col].iloc[i-1]
        cur_short = df[short_col].iloc[i]
        cur_long = df[long_col].iloc[i]
        if prev_short <= prev_long and cur_short > cur_long:
            fut = (df['close'].iloc[i+lookahead] / df['close'].iloc[i]) - 1
            res['golden']['returns'].append(fut)
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
# UTILS: direction & SRI
# -------------------------
def direction_from_signal(sig):
    if not sig:
        return None
    s = sig.lower()
    if 'long' in s or 'bull' in s or 'up' in s:
        return 'LONG'
    if 'short' in s or 'bear' in s or 'down' in s:
        return 'SHORT'
    return None

def compute_sri(symbol, interval):
    """
    Signal Reliability Index: combine ema_stats (golden/death winrates) and model accuracy stored in state
    returns 0..1
    """
    key = f"{symbol}_{interval}"
    stats_entry = ema_stats.get(key)
    parts = []
    if stats_entry and 'stats' in stats_entry:
        g = stats_entry['stats'].get('golden')
        d = stats_entry['stats'].get('death')
        if g and g.get('win_rate') is not None:
            parts.append(g['win_rate'])
        if d and d.get('win_rate') is not None:
            # death winrate is good for SHORT; convert to 'anti-long' score by using (1 - death_winrate) for long reliability
            parts.append(1.0 - d['win_rate'])
    # model acc from state.models if present
    m = state.get('models', {}).get(key)
    if m:
        parts.append(float(m.get('acc', 0)))
    if not parts:
        return 0.0
    return float(np.mean(parts))

# -------------------------
# NEW: confidence (updated with SRI & pattern votes)
# -------------------------
def compute_confidence(hist_stats, ml_prob, ml_acc, volume_spike, patterns, sri):
    score = 0.0
    # history
    if hist_stats and 'win_rate' in hist_stats and hist_stats['win_rate'] is not None:
        score += min(1.0, hist_stats['win_rate']) * CONF_WEIGHTS['hist']
    # ml prob
    if ml_prob is not None:
        score += ml_prob * CONF_WEIGHTS['ml_prob']
    # ml acc
    if ml_acc:
        score += min(1.0, ml_acc) * CONF_WEIGHTS['ml_acc']
    # sri
    if sri is not None:
        score += sri * CONF_WEIGHTS['sri']
    # volume
    if volume_spike:
        score += CONF_WEIGHTS['volume']
    # pattern agreement
    dirs = []
    for k, v in (patterns or {}).items():
        d = direction_from_signal(v)
        if d:
            dirs.append(d)
    pattern_score = 0.0
    if dirs:
        counts = {'LONG': dirs.count('LONG'), 'SHORT': dirs.count('SHORT')}
        maj_count = max(counts['LONG'], counts['SHORT'])
        pattern_ratio = maj_count / len(dirs)
        pattern_score = pattern_ratio * CONF_WEIGHTS['pattern_agreement']
        score += pattern_score
    # penalty if ML direction contradicts majority of patterns
    if ml_prob is not None and dirs:
        ml_dir = 'LONG' if ml_prob >= 0.5 else 'SHORT'
        counts = {'LONG': dirs.count('LONG'), 'SHORT': dirs.count('SHORT')}
        maj = 'LONG' if counts['LONG'] >= counts['SHORT'] else 'SHORT'
        if ml_dir != maj:
            score -= 0.05
    score = max(0.0, min(1.0, score))
    return score

def strength_label(conf):
    if conf >= 0.65:
        return "STRONG"
    if conf >= 0.45:
        return "MEDIUM"
    if conf >= 0.2:
        return "WEAK"
    return "WATCH"

# -------------------------
# SCAN WORKER FOR A SYMBOL (enhanced patterns)
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

        # model
        model, acc = train_or_load_model(symbol, df, lookahead=BACKTEST_LOOKAHEAD)
        ml_prob = None
        if model is not None:
            feat_row = features_for_ml(df).iloc[-1]
            features = ['ATR_10','ATR_50','RSI','MACD_hist','ret1','ret5','vol_change','ema8_21_diff','ema21_50_diff']
            X_row = pd.DataFrame([feat_row[features].values], columns=features)
            ml_prob = float(model.predict_proba(X_row)[0][1])
            # store model acc to state for SRI computation
            key = f"{symbol}_{interval}"
            state.setdefault('models', {})[key] = {"acc": float(acc), "trained_at": str(datetime.utcnow())}
            save_json_safe(STATE_FILE, state)

        vol_spike = False
        if latest['volume'] > (latest['vol_ma_20'] * VOLUME_MULTIPLIER_THRESHOLD):
            vol_spike = True

        # new detectors
        ema_cross = detect_ema_crossover(df, short='ema_8', long='ema_21')
        bb = detect_bollinger_breakout(df)
        rsi = detect_rsi_flip(df)
        macd = detect_macd_signal(df)
        ma50 = detect_ma50_200_cross(df)
        adx = detect_adx(df)
        stoch = detect_stochrsi(df)
        obv_div = detect_obv_divergence(df)
        cci = detect_cci(df)
        atr_b = detect_atr_breakout(df)
        ha_trend = detect_heikin_ashi_trend(df)
        pivot_probe = detect_pivot_probe(df)

        patterns = {
            'ema_cross': ema_cross,
            'bb': bb,
            'rsi': rsi,
            'macd': macd,
            'ma50': ma50,
            'adx': adx,
            'stoch': stoch,
            'obv_div': obv_div,
            'cci': cci,
            'atr': atr_b,
            'ha': ha_trend,
            'pivot': pivot_probe
        }

        # compute SRI
        sri = compute_sri(symbol, interval)

        # compute confidence
        conf = compute_confidence(hist_stats, ml_prob, acc, vol_spike, patterns, sri)

        # determine final signal
        final_signal = None
        if base_signal:
            if hist_stats and hist_stats['count'] >= MIN_TRADES_FOR_STATS and hist_stats['win_rate'] < HIST_WINRATE_THRESHOLD:
                final_signal = "ANTI_"+("SHORT" if base_signal=="LONG" else "LONG")
            else:
                final_signal = base_signal
        else:
            # majority among patterns
            dirs = []
            for v in patterns.values():
                d = direction_from_signal(v)
                if d:
                    dirs.append(d)
            if dirs:
                if dirs.count('LONG') > dirs.count('SHORT'):
                    final_signal = "LONG" if conf >= 0.2 else None
                elif dirs.count('SHORT') > dirs.count('LONG'):
                    final_signal = "SHORT" if conf >= 0.2 else None
                else:
                    final_signal = None

        # build pattern summary for output
        patt_list = [v for v in patterns.values() if v]

        return {
            'symbol': symbol,
            'interval': interval,
            'base_signal': base_signal,
            'final_signal': final_signal,
            'hist_stats': hist_stats,
            'ml_prob': ml_prob,
            'ml_acc': acc,
            'vol_spike': vol_spike,
            'patterns': patterns,
            'pattern_list': patt_list,
            'confidence': conf,
            'sri': sri,
            'last_price': float(latest['close']),
            'timestamp': str(df.index[-1])
        }
    except Exception as e:
        logging.exception("analyze_symbol error %s: %s", symbol, e)
    return None

# -------------------------
# EMA SCAN & MONITOR (uses enhanced analyze_symbol outputs)
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
                if res and res.get('confidence', 0) >= 0.2:
                    results.append(res)
            except Exception as e:
                logging.exception("Future error for %s: %s", s, e)
    logging.info("EMA scan finished, hits=%d", len(results))
    if results:
        results_sorted = sorted(results, key=lambda x: x['confidence'], reverse=True)[:80]
        for r in results_sorted:
            key = f"{r['symbol']}_{interval}"
            stats_entry = ema_stats.get(key)
            if not stats_entry:
                stats_entry = ensure_ema_stats_for_symbol(r['symbol'], interval=interval, force_recompute=False)
            label = strength_label(r['confidence'])
            text = f"*EMA Alert* 🔔\nSymbol: `{r['symbol']}`\nInterval: `{r['interval']}`\nEvent: *{r['final_signal'] or 'WATCH'}* ({label})\nPrice: `{r['last_price']}`\nConfidence: `{r['confidence']:.2f}`\nSRI: `{r['sri']:.2f}`\nTime: `{r['timestamp']}`"
            if r['ml_prob'] is not None:
                text += f"\n🤖 ML ProbUp: `{r['ml_prob']:.2f}` | ML Acc: `{r['ml_acc']:.2f}`"
            if r['vol_spike']:
                text += "\n📈 Volume spike detected"
            if r.get('pattern_list'):
                text += f"\n🧩 Patterns: {', '.join(r['pattern_list'])}"
            if stats_entry and 'stats' in stats_entry:
                g = stats_entry['stats'].get('golden')
                d = stats_entry['stats'].get('death')
                if g and g['count'] > 0:
                    text += f"\n📊 Golden Cross history: count={g['count']} winrate={g['win_rate']*100:.1f}% avg_ret={g['avg_return']*100:.2f}%"
                if d and d['count'] > 0:
                    text += f"\n📊 Death Cross history: count={d['count']} winrate={d['win_rate']*100:.1f}% avg_ret={d['avg_return']*100:.2f}%"
            send_telegram_message(text)
            safe_sleep(0.12)

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
                    if res.get('final_signal') and (conf >= 0.45 or (ml_prob and ml_prob >= ML_PROB_THRESHOLD and res.get('ml_acc',0) >= 0.55)):
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
                        patt = res.get('pattern_list', [])
                        patt_text = f"\n🧩 Patterns: {', '.join(patt)}" if patt else ""
                        label = strength_label(res.get('confidence',0))
                        msg = f"⚡ *Signal*\nSymbol: `{res['symbol']}`\nInterval: `{res['interval']}`\nSignal: *{res['final_signal']}* ({label})\nPrice: `{res['last_price']}`\nConfidence: `{res['confidence']:.2f}`{hist_text}{ml_text}{vol_text}{patt_text}\nSRI: `{res.get('sri',0):.2f}`\nTime: `{res['timestamp']}`"
                        send_telegram_message(msg)
                        state.setdefault('signals', {})[key] = res['final_signal']
                        state.setdefault('last_seen', {})[key] = {'time': res['timestamp'], 'price': res['last_price']}
                        # keep limited signal history for symbol (for future analytics)
                        hist_key = res['symbol']
                        history = state.setdefault('signal_history', {}).setdefault(hist_key, [])
                        history.append({'time': res['timestamp'], 'interval': res['interval'], 'signal': res['final_signal'], 'confidence': res['confidence']})
                        # cap history length
                        if len(history) > 200:
                            state['signal_history'][hist_key] = history[-200:]
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
# BOOTSTRAP
# -------------------------
def warmup_models_and_stats():
    logging.info("Warmup models and EMA stats for top symbols")
    symbols = get_top_symbols_by_volume(limit=30)
    for s in symbols:
        try:
            df = get_klines_df(s, '1h', limit=800)
            if df is None:
                continue
            model, acc = train_or_load_model(s, df, lookahead=BACKTEST_LOOKAHEAD)
            # save model acc in state for sri calc
            if model is not None:
                key = f"{s}_1h"
                state.setdefault('models', {})[key] = {"acc": acc, "trained_at": str(datetime.utcnow())}
                save_json_safe(STATE_FILE, state)
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
    logging.info("Starting AI Futures/Signals Bot (enhanced)")
    try:
        ema_scan_all(interval='5m')
    except Exception as e:
        logging.exception("Initial ema_scan failed: %s", e)
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)