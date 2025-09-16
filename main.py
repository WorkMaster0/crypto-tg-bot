# main.py — Smart Money level + pre-top + pump/dump + ML + patterns
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
from datetime import datetime, timedelta
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

PARALLEL_WORKERS = int(os.getenv("PARALLEL_WORKERS", "6"))  # parallelism for scans
STATE_FILE = "state.json"
MODEL_DIR = "models"
EMA_STATS_FILE = "ema_stats.json"
LOG_FILE = "bot.log"
SCAN_INTERVAL_MINUTES = int(os.getenv("SCAN_INTERVAL_MINUTES", "1"))  # ema scan freq
MONITOR_INTERVAL_MINUTES = int(os.getenv("MONITOR_INTERVAL_MINUTES", "5"))  # detailed monitor freq
EMA_SCAN_LIMIT = int(os.getenv("EMA_SCAN_LIMIT", "500"))
BACKTEST_LOOKAHEAD = int(os.getenv("BACKTEST_LOOKAHEAD", "5"))

# thresholds & tunables
MIN_CONFIDENCE_FOR_ALERT = float(os.getenv("MIN_CONFIDENCE_FOR_ALERT", "0.45"))
VOLUME_SPIKE_MULTIPLIER = float(os.getenv("VOLUME_SPIKE_MULTIPLIER", "2.0"))
PUMP_MOVE_THRESHOLD = float(os.getenv("PUMP_MOVE_THRESHOLD", "0.04"))
DUMP_MOVE_THRESHOLD = float(os.getenv("DUMP_MOVE_THRESHOLD", "-0.04"))
LEVEL_PROXIMITY_PCT = float(os.getenv("LEVEL_PROXIMITY_PCT", "0.015"))  # 1.5% proximity to level counts as 'approaching'
ORDER_BLOCK_LOOKBACK = int(os.getenv("ORDER_BLOCK_LOOKBACK", "30"))  # bars to search for order-block-ish candle

# confidence components weights (tune)
CONF_WEIGHTS = {
    "ml": 0.25,
    "patterns": 0.25,
    "level_proximity": 0.25,
    "volume": 0.15,
    "sri": 0.10
}

# create dirs
os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------------
# LOGGING
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
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
# STATE LOAD/SAVE (safe)
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
# TELEGRAM SAFE ESCAPE & SENDING
# -------------------------
def escape_markdown_v2(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    # escape Telegram MarkdownV2 special characters
    escape_chars = r"_*[]()~`>#+-=|{}.!"
    return re.sub(f"([{re.escape(escape_chars)}])", r"\\\1", text)

def send_telegram_message(text):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        logging.warning("Telegram not configured. Skip send.")
        return None
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
            return resp.text
        return resp.json()
    except Exception as e:
        logging.exception("Telegram exception: %s", e)
        return None

# -------------------------
# MARKET DATA HELPERS
# -------------------------
def get_all_usdt_symbols():
    try:
        exchange_info = client.get_exchange_info()
        symbols = [s['symbol'] for s in exchange_info['symbols']
                   if s['quoteAsset'] == 'USDT' and s['status'] == 'TRADING']
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
            time.sleep(0.5 + attempt)
    return None

# -------------------------
# INDICATORS & SMART LEVELS
# -------------------------
def apply_indicators(df):
    df = df.copy()
    if df is None or len(df) == 0:
        return df
    try:
        # ATR, RSI, MACD, EMAs
        if len(df) >= 10:
            df['ATR_10'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=10)
        else:
            df['ATR_10'] = np.nan
        if len(df) >= 50:
            df['ATR_50'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=50)
        else:
            df['ATR_50'] = np.nan

        df['RSI'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        macd = ta.trend.MACD(df['close'])
        df['MACD_hist'] = macd.macd_diff()
        df['ema_8'] = ta.trend.EMAIndicator(df['close'], window=8).ema_indicator()
        df['ema_21'] = ta.trend.EMAIndicator(df['close'], window=21).ema_indicator()
        df['ema_50'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
        df['ema_200'] = ta.trend.EMAIndicator(df['close'], window=200).ema_indicator()

        # Bollinger
        if len(df) >= 20:
            bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
            df['bb_h'] = bb.bollinger_hband()
            df['bb_l'] = bb.bollinger_lband()
        else:
            df['bb_h'] = np.nan; df['bb_l'] = np.nan

        df['vol_ma_20'] = df['volume'].rolling(window=20).mean()

        # VWAP-like (rolling typical price * vol / vol sum)
        tp = (df['high'] + df['low'] + df['close']) / 3.0
        rolling_vol = df['volume'].rolling(window=50).sum()
        rolling_vwap = (tp * df['volume']).rolling(window=50).sum() / (rolling_vol + 1e-9)
        df['vwap50'] = rolling_vwap

        # Heiken Ashi
        ha_close = (df['open'] + df['high'] + df['low'] + df['close']) / 4
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

        # Quick pivot (previous bar)
        if len(df) >= 3:
            ph = df['high'].iloc[-2]; pl = df['low'].iloc[-2]; pc = df['close'].iloc[-2]
            P = (ph + pl + pc) / 3.0
            R1 = 2 * P - pl; S1 = 2 * P - ph
            df['pivot_R1'] = R1; df['pivot_S1'] = S1
        else:
            df['pivot_R1'] = np.nan; df['pivot_S1'] = np.nan

        # returns & volatility
        df['ret1'] = df['close'].pct_change(1)
        df['ret20_std'] = df['ret1'].rolling(window=20).std()

    except Exception as e:
        logging.exception("apply_indicators error: %s", e)
    return df

# Smart levels: local swing highs/lows clustering => support/resistance zones
def compute_swing_levels(df, lookback=200, n_peaks=6):
    """
    Find recent local highs/lows and cluster them to produce levels.
    Returns dict: {'resistances':[...],'supports':[...]}
    """
    res = {'resistances': [], 'supports': []}
    if df is None or len(df) < 20:
        return res
    series_high = df['high'].iloc[-lookback:] if len(df) >= lookback else df['high']
    series_low = df['low'].iloc[-lookback:] if len(df) >= lookback else df['low']
    # find local peaks via simple rolling comparison
    highs = []
    lows = []
    window = 5
    for i in range(window, len(series_high)-window):
        h = series_high.iloc[i]
        if h == series_high.iloc[i-window:i+window+1].max():
            highs.append((series_high.index[i], float(h)))
        l = series_low.iloc[i]
        if l == series_low.iloc[i-window:i+window+1].min():
            lows.append((series_low.index[i], float(l)))
    # take top n_peaks by recency
    highs = sorted(highs, key=lambda x: x[0], reverse=True)[:n_peaks]
    lows = sorted(lows, key=lambda x: x[0], reverse=True)[:n_peaks]
    # cluster nearby levels (within 1% of value)
    def cluster(vals, tol=0.01):
        clusters = []
        for t, v in vals:
            placed = False
            for c in clusters:
                if abs(c['level'] - v) / (c['level'] + 1e-9) < tol:
                    # weighted average
                    c['level'] = (c['level']*c['count'] + v) / (c['count'] + 1)
                    c['count'] += 1
                    c['times'].append(t)
                    placed = True
                    break
            if not placed:
                clusters.append({'level': v, 'count': 1, 'times': [t]})
        # sort by level desc for resistances, asc for supports will be handled by caller
        return clusters
    res['resistances'] = sorted([c['level'] for c in cluster(highs)], reverse=True)
    res['supports'] = sorted([c['level'] for c in cluster(lows)])
    return res

# detect order-block-ish candle: big directional candle with gap/imbalance
def detect_order_block(df, lookback=ORDER_BLOCK_LOOKBACK, mult=2.0):
    """
    Find recent large directional candle (high ATR multiple) as candidate order-block / imbalance zone.
    Returns dict with zone (high, low, idx) or None.
    """
    if df is None or len(df) < lookback + 2:
        return None
    window = lookback
    sub = df.iloc[-window:]
    atr = sub['ATR_10'].dropna()
    if atr.empty:
        return None
    mean_atr = atr.mean()
    # find candle with body > mult * mean_atr
    for i in range(len(sub)-2, -1, -1):  # search backwards, prefer recent
        body = abs(sub['close'].iloc[i] - sub['open'].iloc[i])
        if body > mult * mean_atr:
            # candidate: zone = full candle (high-low)
            return {'idx': sub.index[i], 'high': float(sub['high'].iloc[i]), 'low': float(sub['low'].iloc[i]), 'body': float(body)}
    return None

# detect imbalance: quick gap convex candle (close far from mid + big volume)
def detect_imbalance(df, lookback=20, vol_mult=2.0):
    if df is None or len(df) < lookback + 1:
        return None
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    vol_avg = df['volume'].iloc[-lookback:].mean()
    body = latest['close'] - latest['open']
    body_pct = abs(body) / (latest['close'] + 1e-9)
    vol_spike = latest['volume'] > vol_avg * vol_mult
    # imbalance up: big green body, big vol, close near high
    if body > 0 and vol_spike and (latest['close'] > latest['high'] - (latest['high'] - latest['low']) * 0.2):
        return {'type': 'imb_up', 'strength': body_pct}
    if body < 0 and vol_spike and (latest['close'] < latest['low'] + (latest['high'] - latest['low']) * 0.2):
        return {'type': 'imb_down', 'strength': body_pct}
    return None

# -------------------------
# PATTERN DETECTORS (existing ones + new)
# -------------------------
def detect_ema_crossover(df, short='ema_8', long='ema_21'):
    if df is None or len(df) < 3:
        return None
    a = df[short].iloc[-2]; b = df[long].iloc[-2]; a1 = df[short].iloc[-1]; b1 = df[long].iloc[-1]
    if pd.isna(a) or pd.isna(b) or pd.isna(a1) or pd.isna(b1):
        return None
    if a <= b and a1 > b1: return "bull_cross"
    if a >= b and a1 < b1: return "bear_cross"
    return None

def detect_bollinger_breakout(df):
    if df is None or len(df) < 20: return None
    h = df['bb_h'].iloc[-1]; l = df['bb_l'].iloc[-1]; c = df['close'].iloc[-1]
    if pd.isna(h) or pd.isna(l): return None
    if c > h: return 'bb_up'
    if c < l: return 'bb_down'
    return None

def detect_macd_signal(df):
    if df is None or len(df) < 3: return None
    prev = df['MACD_hist'].iloc[-2]; cur = df['MACD_hist'].iloc[-1]
    if pd.isna(prev) or pd.isna(cur): return None
    if prev <= 0 and cur > 0: return 'macd_up'
    if prev >= 0 and cur < 0: return 'macd_down'
    return None

def detect_rsi_flip(df):
    if df is None or len(df) < 15: return None
    prev = df['RSI'].iloc[-2]; cur = df['RSI'].iloc[-1]
    if pd.isna(prev) or pd.isna(cur): return None
    if prev < 30 and cur >= 30: return 'rsi_up'
    if prev > 70 and cur <= 70: return 'rsi_down'
    return None

# -------------------------
# ML helpers (unchanged idea from earlier)
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
    df_feat = features_for_ml(df)
    if len(df_feat) < 300:
        logging.info("%s not enough rows for ML (need >=300).", symbol)
        return None, None
    df_feat['future_ret'] = df_feat['close'].shift(-lookahead) / df_feat['close'] - 1
    df_feat = df_feat.dropna()
    df_feat['target'] = (df_feat['future_ret'] > 0).astype(int)
    features = ['ATR_10','ATR_50','RSI','MACD_hist','ret1','ret5','vol_change','ema8_21_diff','ema21_50_diff']
    X = df_feat[features]; y = df_feat['target']
    # load
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
    # train
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
# SRI (Signal Reliability Index)
# -------------------------
def compute_sri(symbol, interval):
    key = f"{symbol}_{interval}"
    stats_entry = ema_stats.get(key)
    parts = []
    if stats_entry and 'stats' in stats_entry:
        g = stats_entry['stats'].get('golden')
        d = stats_entry['stats'].get('death')
        if g and g.get('win_rate') is not None:
            parts.append(g['win_rate'])
        if d and d.get('win_rate') is not None:
            parts.append(1.0 - d['win_rate'])
    m = state.get('models', {}).get(key)
    if m:
        parts.append(float(m.get('acc', 0)))
    if not parts:
        return 0.0
    return float(np.mean(parts))

# -------------------------
# SMART-MONEY SCORE & FINAL DECISION
# -------------------------
def direction_from_signal(sig):
    if not sig: return None
    s = str(sig).lower()
    if any(x in s for x in ['long','bull','up','pump']): return 'LONG'
    if any(x in s for x in ['short','bear','down','dump','pre_top']): return 'SHORT'
    return None

def compute_smart_score(symbol, interval, ml_prob, patterns, level_proximity, vol_spike, sri):
    """
    Combine ML + pattern votes + proximity to smart levels + vol + sri into 0..1 score
    """
    score = 0.0
    # ML
    if ml_prob is not None:
        score += ml_prob * CONF_WEIGHTS['ml']
    # patterns agreement: count directions
    dirs = []
    for v in (patterns or {}).values():
        d = direction_from_signal(v)
        if d: dirs.append(d)
    if dirs:
        counts = {'LONG': dirs.count('LONG'), 'SHORT': dirs.count('SHORT')}
        maj = max(counts['LONG'], counts['SHORT'])
        ratio = maj / len(dirs)
        score += ratio * CONF_WEIGHTS['patterns']
    # level proximity
    if level_proximity is not None:
        score += (1.0 - level_proximity) * CONF_WEIGHTS['level_proximity']  # closer => bigger add
    # volume
    if vol_spike:
        score += CONF_WEIGHTS['volume']
    # sri
    if sri is not None:
        score += sri * CONF_WEIGHTS['sri']
    score = max(0.0, min(1.0, score))
    return score

def strength_label(conf):
    if conf >= 0.75: return "STRONG"
    if conf >= 0.55: return "MEDIUM"
    if conf >= 0.35: return "WEAK"
    return "WATCH"

# -------------------------
# PRE-TOP (anti-pump) detector
# -------------------------
def detect_pre_top(df):
    if df is None or len(df) < 20: return None
    latest = df.iloc[-1]
    prev5 = df['close'].iloc[-6]  # 5 bars ago
    price_change = (latest['close'] - prev5) / (prev5 + 1e-9)
    vol_spike = latest['volume'] > df['volume'].iloc[-20:].mean() * VOLUME_SPIKE_MULTIPLIER
    rsi_high = latest.get('RSI', 0) > 70
    bearish_signs = df['ema_8'].iloc[-1] < df['ema_21'].iloc[-1] or df['MACD_hist'].iloc[-1] < 0
    if price_change > 0.06 and vol_spike and rsi_high and bearish_signs:
        return {'type': 'PRE_TOP_SHORT', 'confidence': min(0.95, 0.55 + price_change)}
    return None

# -------------------------
# LEVEL PROXIMITY: check whether price is near support/resistance/orderblock/vwap
# -------------------------
def level_proximity_score(price, levels):
    # levels: list of numeric levels (higher score when closer)
    if not levels: return None
    dists = [abs(price - l) / (l + 1e-9) for l in levels]
    min_dist = min(dists)
    return float(min_dist)  # lower is better (0 = on level). We'll invert later.

def find_nearby_level_info(df, symbol_interval_levels):
    """
    For df (latest bar), compute proximity to support/resistance/vwap/orderblock/imbalance zones.
    Returns dict with proximity values (0..1 normalized approx)
    """
    if df is None or len(df) == 0:
        return {}
    price = df['close'].iloc[-1]
    levels = symbol_interval_levels  # dict with lists: supports/resistances, vwap, order_block zone
    prox = {}
    # supports/resistances
    sr_levels = []
    sr_levels.extend(levels.get('supports', []))
    sr_levels.extend(levels.get('resistances', []))
    p = level_proximity_score(price, sr_levels)
    prox['sr_prox'] = p
    # vwap
    vwap = levels.get('vwap50')
    if vwap is not None and not math.isnan(vwap):
        prox['vwap_prox'] = abs(price - vwap) / (vwap + 1e-9)
    else:
        prox['vwap_prox'] = None
    # order block zone: distance to zone edges, normalized by price
    ob = levels.get('order_block')
    if ob:
        high = ob.get('high'); low = ob.get('low')
        if low <= price <= high:
            prox['orderblock_prox'] = 0.0
        else:
            dist = min(abs(price - low), abs(price - high)) / (price + 1e-9)
            prox['orderblock_prox'] = dist
    else:
        prox['orderblock_prox'] = None
    # imbalance presence
    prox['has_imbalance'] = bool(levels.get('imbalance'))
    return prox

# -------------------------
# ANALYZE SYMBOL (main)
# -------------------------
def analyze_symbol(symbol, interval='5m'):
    try:
        df = get_klines_df(symbol, interval, limit=EMA_SCAN_LIMIT)
        if df is None or len(df) < 50:
            return None
        df = apply_indicators(df)
        # compute levels (swing levels, vwap50, orderblock, imbalance)
        levels = compute_swing_levels(df, lookback=200)
        levels['vwap50'] = df['vwap50'].iloc[-1] if 'vwap50' in df.columns else None
        ob = detect_order_block(df, lookback=ORDER_BLOCK_LOOKBACK)
        levels['order_block'] = ob
        imb = detect_imbalance(df)
        levels['imbalance'] = imb

        # pattern detectors
        ema_cross = detect_ema_crossover(df)
        bb = detect_bollinger_breakout(df)
        macd = detect_macd_signal(df)
        rsi_flip = detect_rsi_flip(df)

        patterns = {
            'ema_cross': ema_cross,
            'bb': bb,
            'macd': macd,
            'rsi': rsi_flip
        }

        # backtest hist
        hist = None
        try:
            hist = backtest_pattern(df, lookahead=BACKTEST_LOOKAHEAD)
        except Exception:
            hist = None

        # ML
        model, acc = train_or_load_model(symbol, df, lookahead=BACKTEST_LOOKAHEAD)
        ml_prob = None
        if model is not None:
            feat_row = features_for_ml(df).iloc[-1]
            features = ['ATR_10','ATR_50','RSI','MACD_hist','ret1','ret5','vol_change','ema8_21_diff','ema21_50_diff']
            X_row = pd.DataFrame([feat_row[features].values], columns=features)
            ml_prob = float(model.predict_proba(X_row)[0][1])
            # store model accuracy for SRI
            key = f"{symbol}_{interval}"
            state.setdefault('models', {})[key] = {"acc": float(acc), "trained_at": str(datetime.utcnow())}
            save_json_safe(STATE_FILE, state)
        else:
            acc = None

        # volume spike
        latest = df.iloc[-1]
        vol_spike = bool(latest['volume'] > (df['volume'].iloc[-20:].mean() * VOLUME_SPIKE_MULTIPLIER))

        # pre-top / pump/dump
        pre_top = detect_pre_top(df)
        pumpdump = None
        prev = df.iloc[-2]; move = (latest['close'] - prev['close']) / (prev['close'] + 1e-9)
        if move > PUMP_MOVE_THRESHOLD and vol_spike:
            pumpdump = {'type': 'PUMP_ALERT', 'move': move}
        elif move < DUMP_MOVE_THRESHOLD and vol_spike:
            pumpdump = {'type': 'DUMP_ALERT', 'move': move}

        # level proximity info
        prox = find_nearby_level_info(df, levels)
        # compute a normalized "level proximity" in 0..1 where 1 means very close (we invert dist)
        level_prox = None
        if prox.get('sr_prox') is not None:
            level_prox = max(0.0, 1.0 - prox['sr_prox'] / LEVEL_PROXIMITY_PCT)  # >0 when within threshold
            level_prox = float(min(1.0, max(0.0, level_prox)))  # clamp
        # if order block inside zone -> strong proximity
        if prox.get('orderblock_prox') == 0.0:
            level_prox = max(level_prox or 0.0, 0.9)

        # compute SRI
        sri = compute_sri(symbol, interval)

        # compute smart score
        smart_score = compute_smart_score(symbol, interval, ml_prob, patterns, (1 - level_prox) if level_prox is not None else None, vol_spike, sri)

        # determine final signal direction: prefer base_signal (generate_signal_from_row) but also allow pattern majority
        base_sig = generate_signal_from_row(latest)  # from earlier logic (donchian/atr/macd/rsi)
        pattern_dirs = []
        for v in patterns.values():
            d = direction_from_signal(v)
            if d: pattern_dirs.append(d)
        ml_dir = 'LONG' if (ml_prob or 0) >= 0.5 else 'SHORT'
        # trend filter: prefer not to send LONG in strong downtrend (ema50 < ema200)
        trend = 'UP' if latest['ema_50'] > latest['ema_200'] else 'DOWN'
        final_signal = None
        # priority: pre_top (short) and pumpdump (pump/dump alerts)
        meta_alert = None
        if pumpdump:
            meta_alert = pumpdump
        if pre_top:
            meta_alert = pre_top

        if meta_alert:
            # meta alerts are always reported (but with their own confidence)
            final_signal = 'SHORT' if meta_alert.get('type','').startswith('PRE_TOP') or meta_alert.get('type','')=='DUMP_ALERT' else 'LONG'
            conf = meta_alert.get('confidence', 0.6) if isinstance(meta_alert, dict) else 0.6
            label = strength_label(conf)
            text = f"🚨 *{meta_alert.get('type')}*\nSymbol: `{symbol}`\nInterval: `{interval}`\nSignal: *{final_signal}* ({label})\nPrice: `{latest['close']}`\nConfidence: `{conf:.2f}`\nTime: `{df.index[-1]}`"
            if prox.get('orderblock_prox') == 0.0:
                text += "\n📦 Orderblock zone"
            if levels.get('supports') or levels.get('resistances'):
                text += f"\n🔘 Levels nearby: supports={len(levels.get('supports',[]))} resistances={len(levels.get('resistances',[]))}"
            send_telegram_message(text)
            # store state quickly
            state.setdefault('signals', {})[f"{symbol}_{interval}"] = final_signal
            save_json_safe(STATE_FILE, state)
            return {'symbol': symbol, 'type': meta_alert.get('type'), 'final_signal': final_signal, 'confidence': conf}

        # Otherwise use smart_score and pattern/ML consensus
        # require at least minimal smart_score to alert
        if smart_score < MIN_CONFIDENCE_FOR_ALERT:
            return None

        # Decide direction: combine base_sig, pattern majority, ml_dir
        directions = []
        if base_sig: directions.append(base_sig)
        directions.extend(pattern_dirs)
        directions.append(ml_dir)
        # majority
        if directions:
            final = max(set(directions), key=directions.count)
            # block LONGs in downtrend unless very strong smart_score
            if final == 'LONG' and trend == 'DOWN' and smart_score < 0.7:
                return None
            final_signal = final
        else:
            final_signal = ml_dir

        label = strength_label(smart_score)
        patt_list = [v for v in patterns.values() if v]
        text = (f"⚡ *Smart-Level Alert*\nSymbol: `{symbol}`\nInterval: `{interval}`\nSignal: *{final_signal}* ({label})\n"
                f"Price: `{latest['close']}`\nSmartScore: `{smart_score:.2f}`\nSRI: `{sri:.2f}`\nTime: `{df.index[-1]}`")
        if ml_prob is not None:
            text += f"\n🤖 ML ProbUp: `{ml_prob:.2f}` | ML Acc: `{acc or 0:.2f}`"
        if vol_spike:
            text += "\n📈 Volume spike"
        if patt_list:
            text += f"\n🧩 Patterns: {', '.join(patt_list)}"
        # show nearby levels summary
        if levels.get('resistances'):
            top_res = levels['resistances'][:3]
            text += f"\n🔺 Resistances (top3): {', '.join([str(round(x,6)) for x in top_res])}"
        if levels.get('supports'):
            top_sup = levels['supports'][:3]
            text += f"\n🔻 Supports (top3): {', '.join([str(round(x,6)) for x in top_sup])}"
        if prox.get('orderblock_prox') == 0.0:
            text += "\n📦 Inside OrderBlock zone"
        if prox.get('has_imbalance'):
            text += "\n⚖️ Imbalance detected (recent large candle)"
        send_telegram_message(text)

        # persist
        key = f"{symbol}_{interval}"
        state.setdefault('signals', {})[key] = final_signal
        history = state.setdefault('signal_history', {}).setdefault(symbol, [])
        history.append({'time': str(df.index[-1]), 'interval': interval, 'signal': final_signal, 'score': smart_score})
        if len(history) > 300: state['signal_history'][symbol] = history[-300:]
        save_json_safe(STATE_FILE, state)

        return {
            'symbol': symbol, 'final_signal': final_signal, 'smart_score': smart_score, 'patterns': patterns, 'levels': levels
        }

    except Exception as e:
        logging.exception("analyze_symbol error %s: %s", symbol, e)
    return None

# -------------------------
# EMA SCAN ALL (parallel)
# -------------------------
def ema_scan_all(interval='5m'):
    logging.info("Starting full smart-level scan (interval=%s)...", interval)
    symbols = get_all_usdt_symbols()
    results = []
    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as exe:
        futures = {exe.submit(analyze_symbol, s, interval): s for s in symbols}
        for fut in as_completed(futures):
            s = futures[fut]
            try:
                res = fut.result()
                if res:
                    results.append(res)
            except Exception as e:
                logging.exception("Future error for %s: %s", s, e)
    logging.info("Scan finished. hits=%d", len(results))
    return results

# -------------------------
# Monitor top symbols (detailed)
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
                    results.append(res)
            except Exception as e:
                logging.exception("monitor future error: %s", e)
    logging.info("Detailed monitor done, processed=%d", len(results))
    return results

# -------------------------
# BACKGROUND WARMUP OF MODELS & EMA STATS
# -------------------------
def compute_and_save_ema_stats(symbols, intervals=['5m','1h']):
    for s in symbols:
        for intr in intervals:
            try:
                df = get_klines_df(s, intr, limit=EMA_SCAN_LIMIT)
                if df is None: continue
                stats = compute_ema_historical_stats(df, short_col='ema_8', long_col='ema_21', lookahead=BACKTEST_LOOKAHEAD)
                key = f"{s}_{intr}"
                ema_stats[key] = {'computed_at': str(datetime.utcnow()), 'stats': stats}
                save_json_safe(EMA_STATS_FILE, ema_stats)
                time.sleep(0.2)
            except Exception as e:
                logging.exception("ema stats compute error %s %s", s, intr)
def warmup_models_and_stats():
    logging.info("Warmup models and EMA stats")
    symbols = get_top_symbols_by_volume(limit=30)
    for s in symbols:
        try:
            df = get_klines_df(s, '1h', limit=800)
            if df is None: continue
            train_or_load_model(s, df, lookahead=BACKTEST_LOOKAHEAD)
            time.sleep(0.2)
        except Exception as e:
            logging.exception("warmup error for %s: %s", s, e)
    compute_and_save_ema_stats(symbols, intervals=['5m','1h'])

import threading
t = threading.Thread(target=warmup_models_and_stats, daemon=True)
t.start()

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
        r1 = ema_scan_all(interval='5m')
        r2 = monitor_top_symbols()
        return jsonify({"scanned": len(r1) + len(r2)})
    except Exception as e:
        logging.exception("scan_now error: %s", e)
        return jsonify({"error": str(e)}), 500

@app.route('/status')
def status():
    out = {"state": state, "ema_stats_count": len(ema_stats), "time": str(datetime.utcnow())}
    return jsonify(out)

# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    logging.info("Starting Smart-Money Analyzer Bot")
    try:
        ema_scan_all(interval='5m')
    except Exception as e:
        logging.exception("Initial scan failed: %s", e)
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)