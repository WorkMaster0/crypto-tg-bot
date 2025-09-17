# main.py — Universal multi-symbol pre-top detector + webhook-enabled Telegram alerts
# Requirements:
# pip install python-binance pandas numpy ta requests apscheduler flask

import os
import time
import math
import json
import random
import logging
import traceback
import re
from datetime import datetime, timezone
from threading import Thread

import pandas as pd
import numpy as np
import requests
from flask import Flask, request, jsonify
from apscheduler.schedulers.background import BackgroundScheduler
from concurrent.futures import ThreadPoolExecutor, as_completed

# technical analysis library
import ta

# optional: python-binance client (if not available, fetch klines differently)
try:
    from binance.client import Client as BinanceClient
    BINANCE_PY_AVAILABLE = True
except Exception:
    BINANCE_PY_AVAILABLE = False

# -------------------------
# CONFIG / ENV
# -------------------------
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
CHAT_ID = os.getenv("CHAT_ID", "")
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "")  # e.g. https://<app>.onrender.com/telegram_webhook/<TELEGRAM_TOKEN>
PORT = int(os.getenv("PORT", "5000"))

TOP_LIMIT = int(os.getenv("TOP_LIMIT", "10"))  # how many top USDT pairs to analyze
SCAN_INTERVAL_MINUTES = int(os.getenv("SCAN_INTERVAL_MINUTES", "1"))
EMA_SCAN_LIMIT = int(os.getenv("EMA_SCAN_LIMIT", "500"))

STATE_FILE = "state.json"
LOG_FILE = "bot.log"
MODEL_DIR = "models"  # reserved (not used here but kept)
os.makedirs(MODEL_DIR, exist_ok=True)

# thresholds
CONF_THRESHOLD_MEDIUM = 0.45
CONF_THRESHOLD_STRONG = 0.65
MIN_FEATURES_TO_SIGNAL = 3  # minimum number of confirming features to consider a signal

# worker threads
PARALLEL_WORKERS = int(os.getenv("PARALLEL_WORKERS", "6"))

# -------------------------
# LOGGING
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)
logger = logging.getLogger("pretop-bot")

# -------------------------
# BINANCE CLIENT (optional)
# -------------------------
if BINANCE_PY_AVAILABLE and BINANCE_API_KEY and BINANCE_API_SECRET:
    client = BinanceClient(api_key=BINANCE_API_KEY, api_secret=BINANCE_API_SECRET)
else:
    client = None
    if not BINANCE_PY_AVAILABLE:
        logger.warning("python-binance not available. Ensure you installed 'python-binance' for real Binance access.")
    else:
        logger.warning("BINANCE API keys not provided. Some features will not work.")

# -------------------------
# FLASK APP
# -------------------------
app = Flask(__name__)

# -------------------------
# STATE management
# -------------------------
def load_json_safe(path, default):
    try:
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
    except Exception as e:
        logger.exception("load_json_safe error %s: %s", path, e)
    return default

def save_json_safe(path, data):
    try:
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp, path)
    except Exception as e:
        logger.exception("save_json_safe error %s: %s", path, e)

state = load_json_safe(STATE_FILE, {"signals": {}, "last_scan": None, "signal_history": {}})

# -------------------------
# Telegram helpers
# -------------------------
MARKDOWNV2_ESCAPE = r"_*[]()~`>#+-=|{}.!"

def escape_md_v2(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    return re.sub(f"([{re.escape(MARKDOWNV2_ESCAPE)}])", r"\\\1", text)

def send_telegram(text: str):
    """Send message to CHAT_ID with MarkdownV2 escaping"""
    if not TELEGRAM_TOKEN or not CHAT_ID:
        logger.debug("Telegram not configured; message skipped.")
        return None
    payload = {
        "chat_id": CHAT_ID,
        "text": escape_md_v2(text),
        "parse_mode": "MarkdownV2",
        "disable_web_page_preview": True
    }
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        resp = requests.post(url, json=payload, timeout=10)
        if resp.status_code != 200:
            logger.error("Telegram send failed: %s %s", resp.status_code, resp.text)
        return resp
    except Exception as e:
        logger.exception("send_telegram exception: %s", e)
        return None

def set_telegram_webhook(webhook_url: str):
    """Register webhook (overwrites existing)"""
    if not TELEGRAM_TOKEN or not webhook_url:
        return None
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/setWebhook"
        resp = requests.post(url, json={"url": webhook_url}, timeout=10)
        logger.info("setWebhook resp: %s", resp.text if resp is not None else "None")
        return resp
    except Exception as e:
        logger.exception("set_telegram_webhook error: %s", e)
        return None

# -------------------------
# Market data helpers
# -------------------------
def get_all_usdt_symbols_from_binance():
    """Return list of trading symbols with USDT quote, using Binance client if available"""
    if not client:
        logger.warning("Binance client unavailable.")
        return []
    try:
        ex = client.get_exchange_info()
        symbols = [s["symbol"] for s in ex["symbols"] if s["quoteAsset"] == "USDT" and s["status"] == "TRADING"]
        return symbols
    except Exception as e:
        logger.exception("get_all_usdt_symbols_from_binance error: %s", e)
        return []

def get_top_symbols_by_volume(limit=TOP_LIMIT):
    """Return top USDT pairs by quoteVolume using ticker 24h"""
    if not client:
        logger.warning("Binance client unavailable; returning empty.")
        return []
    try:
        tickers = client.get_ticker()  # 24h tickers
        usdt = [t for t in tickers if t["symbol"].endswith("USDT")]
        sorted_t = sorted(usdt, key=lambda x: float(x.get("quoteVolume", 0)), reverse=True)
        return [t["symbol"] for t in sorted_t[:limit]]
    except Exception as e:
        logger.exception("get_top_symbols_by_volume error: %s", e)
        return []

def fetch_klines(symbol, interval="15m", limit=EMA_SCAN_LIMIT, retry=3):
    """Return DataFrame with columns time, open, high, low, close, volume indexed by time"""
    for attempt in range(retry):
        try:
            if client:
                kl = client.get_klines(symbol=symbol, interval=interval, limit=limit)
            else:
                # if no client, raise
                raise RuntimeError("Binance client not configured")
            df = pd.DataFrame(kl, columns=[
                "open_time","open","high","low","close","volume","close_time","qav","num_trades",
                "taker_base","taker_quote","ignore"
            ])
            df = df[["open_time","open","high","low","close","volume"]].copy()
            df[["open","high","low","close","volume"]] = df[["open","high","low","close","volume"]].astype(float)
            df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
            df.set_index("open_time", inplace=True)
            return df
        except Exception as e:
            logger.warning("fetch_klines %s attempt %d/%d error: %s", symbol, attempt+1, retry, e)
            time.sleep(0.5 + attempt)
    return None

# -------------------------
# FEATURE ENGINEERING — 20+ features + simple smart-money heuristics
# -------------------------
def apply_all_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if df is None or df.empty:
        return df

    try:
        # basic returns
        df["ret1"] = df["close"].pct_change(1)
        df["ret5"] = df["close"].pct_change(5)

        # EMAs & SMAs
        df["ema_8"] = ta.trend.EMAIndicator(df["close"], window=8).ema_indicator()
        df["ema_20"] = ta.trend.EMAIndicator(df["close"], window=20).ema_indicator()
        df["ema_50"] = ta.trend.EMAIndicator(df["close"], window=50).ema_indicator()
        df["sma_50"] = df["close"].rolling(window=50).mean()

        # ATR
        if len(df) >= 14:
            df["ATR_14"] = ta.volatility.AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=14).average_true_range()
        else:
            df["ATR_14"] = np.nan

        # RSI / StochRSI
        df["RSI_14"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
        try:
            stoch = ta.momentum.StochRSIIndicator(df["close"], window=14, smooth1=3, smooth2=3)
            df["stoch_k"] = stoch.stochrsi_k()
            df["stoch_d"] = stoch.stochrsi_d()
        except Exception:
            df["stoch_k"] = np.nan
            df["stoch_d"] = np.nan

        # MACD
        macd = ta.trend.MACD(df["close"])
        df["MACD"] = macd.macd()
        df["MACD_signal"] = macd.macd_signal()
        df["MACD_hist"] = macd.macd_diff()

        # ADX
        try:
            adx = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], window=14)
            df["ADX"] = adx.adx()
            df["ADX_pos"] = adx.adx_pos()
            df["ADX_neg"] = adx.adx_neg()
        except Exception:
            df["ADX"] = np.nan
            df["ADX_pos"] = np.nan
            df["ADX_neg"] = np.nan

        # Bollinger
        try:
            bb = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
            df["bb_h"] = bb.bollinger_hband()
            df["bb_l"] = bb.bollinger_lband()
            df["bb_width"] = (df["bb_h"] - df["bb_l"]) / df["bb_l"].replace(0, np.nan)
        except Exception:
            df["bb_h"] = np.nan
            df["bb_l"] = np.nan
            df["bb_width"] = np.nan

        # VWAP (per bar cumulative approach)
        # note: for intraday across session VWAP has session logic; we compute rolling VWAP for last 50 bars
        pv = df["close"] * df["volume"]
        df["vwap_50"] = (pv.rolling(50).sum() / df["volume"].rolling(50).sum()).replace([np.inf, -np.inf], np.nan)

        # OBV
        try:
            obv = ta.volume.OnBalanceVolumeIndicator(df["close"], df["volume"])
            df["OBV"] = obv.on_balance_volume()
        except Exception:
            df["OBV"] = np.nan

        # CCI
        try:
            cci = ta.trend.CCIIndicator(df["high"], df["low"], df["close"], window=20)
            df["CCI"] = cci.cci()
        except Exception:
            df["CCI"] = np.nan

        # Heikin-Ashi derived
        ha_close = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
        ha_open = df["open"].copy()
        ha_open_vals = []
        prev_ho = (df["open"].iloc[0] + df["close"].iloc[0]) / 2.0
        prev_hc = ha_close.iloc[0]
        ha_open_vals.append(prev_ho)
        for i in range(1, len(df)):
            prev_ho = (ha_open_vals[-1] + prev_hc) / 2.0
            ha_open_vals.append(prev_ho)
            prev_hc = ha_close.iloc[i]
        df["ha_open"] = ha_open_vals
        df["ha_close"] = ha_close
        df["ha_body"] = (df["ha_close"] - df["ha_open"])
        df["ha_dir"] = np.sign(df["ha_body"])

        # pivot simple (previous bar pivot R1/S1)
        if len(df) >= 2:
            ph = df["high"].iloc[-2]
            pl = df["low"].iloc[-2]
            pc = df["close"].iloc[-2]
            P = (ph + pl + pc) / 3.0
            R1 = 2 * P - pl
            S1 = 2 * P - ph
            df["pivot_R1"] = R1
            df["pivot_S1"] = S1
        else:
            df["pivot_R1"] = np.nan
            df["pivot_S1"] = np.nan

        # volatility regime
        df["ret1_std_20"] = df["ret1"].rolling(window=20).std()

        # simple liquidity / imbalance heuristics:
        # - fair value gap: previous candle body gap: if large empty range between candles bodies
        df["fvg_up"] = False
        df["fvg_down"] = False
        if len(df) >= 3:
            for i in range(2, len(df)):
                prev = df.iloc[i-1]
                pprev = df.iloc[i-2]
                # bullish FVG: pprev.high < prev.low (gap up)
                if pprev["high"] < prev["low"]:
                    df.iat[i, df.columns.get_loc("fvg_up")] = True
                # bearish FVG: pprev.low > prev.high (gap down)
                if pprev["low"] > prev["high"]:
                    df.iat[i, df.columns.get_loc("fvg_down")] = True

        # liquidity grab / wick spike: last candle with long wick beyond previous extremes
        df["liquidity_grab_up"] = False
        df["liquidity_grab_down"] = False
        if len(df) >= 2:
            last = df.iloc[-1]
            prev_high = df["high"].iloc[-2]
            prev_low = df["low"].iloc[-2]
            # price poked above previous high, then closed below -> liquidity grab up
            if last["high"] > prev_high and last["close"] < prev_high:
                df["liquidity_grab_up"].iloc[-1] = True
            if last["low"] < prev_low and last["close"] > prev_low:
                df["liquidity_grab_down"].iloc[-1] = True

        # volume spike relative to rolling mean
        df["vol_ma_20"] = df["volume"].rolling(window=20).mean()
        df["vol_spike"] = (df["volume"] > (df["vol_ma_20"] * 1.8)).astype(bool)

        # ATR breakout detection (impulse)
        df["atr_break"] = False
        if len(df) >= 2 and not pd.isna(df["ATR_14"].iloc[-1]):
            if abs(df["close"].iloc[-1] - df["close"].iloc[-2]) > 1.5 * df["ATR_14"].iloc[-1]:
                df["atr_break"].iloc[-1] = True

    except Exception as e:
        logger.exception("apply_all_features error: %s", e)

    return df

# -------------------------
# SIGNAL DETECTION (pre-top detect + general signals)
# -------------------------
def direction_from_vote(vote: str):
    if not vote:
        return None
    s = vote.lower()
    if any(x in s for x in ("bull", "up", "long", "buy")):
        return "LONG"
    if any(x in s for x in ("bear", "down", "short", "sell")):
        return "SHORT"
    return None

def detect_pre_top_and_signals(df: pd.DataFrame):
    """Return (final_signal_or_None, details_dict)
    details contains list of triggered features, individual votes, confidence score, label (WEAK/MEDIUM/STRONG), sri etc."""
    if df is None or df.empty:
        return None, {}

    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else last

    votes = []   # list of short descriptions
    feature_count = 0

    # 1) EMA cross (fast vs mid)
    if not pd.isna(last["ema_8"]) and not pd.isna(last["ema_20"]):
        if last["ema_8"] > last["ema_20"]:
            votes.append("ema8>ema20 (bull)")
            feature_count += 1
        elif last["ema_8"] < last["ema_20"]:
            votes.append("ema8<ema20 (bear)")
            feature_count += 1

    # 2) Price vs EMA50/SMA50
    if not pd.isna(last["ema_50"]):
        if last["close"] > last["ema_50"]:
            votes.append("price>ema50 (bull)")
            feature_count += 1
        else:
            votes.append("price<ema50 (bear)")
            feature_count += 1

    # 3) MACD histogram direction
    if not pd.isna(last["MACD_hist"]):
        if last["MACD_hist"] > 0:
            votes.append("macd_hist_up")
            feature_count += 1
        else:
            votes.append("macd_hist_down")
            feature_count += 1

    # 4) RSI extremes
    if not pd.isna(last["RSI_14"]):
        if last["RSI_14"] < 30:
            votes.append("rsi_oversold")
            feature_count += 1
        elif last["RSI_14"] > 70:
            votes.append("rsi_overbought")
            feature_count += 1

    # 5) ADX strong trend
    if not pd.isna(last.get("ADX", np.nan)):
        if last["ADX"] >= 25:
            if last.get("ADX_pos", 0) > last.get("ADX_neg", 0):
                votes.append("adx_strong_up")
            else:
                votes.append("adx_strong_down")
            feature_count += 1

    # 6) Bollinger breakout
    if not pd.isna(last.get("bb_h", np.nan)):
        if last["close"] > last["bb_h"]:
            votes.append("bb_break_up")
            feature_count += 1
        if last["close"] < last["bb_l"]:
            votes.append("bb_break_down")
            feature_count += 1

    # 7) VWAP touch / extreme
    if not pd.isna(last.get("vwap_50", np.nan)):
        if abs(last["close"] - last["vwap_50"]) / (last["vwap_50"] + 1e-9) < 0.002:
            votes.append("vwap_touch")
            feature_count += 1

    # 8) Volume spike
    if bool(last.get("vol_spike", False)):
        votes.append("vol_spike")
        feature_count += 1

    # 9) ATR breakout (impulse)
    if bool(last.get("atr_break", False)):
        votes.append("atr_break")
        feature_count += 1

    # 10) OBV divergence (simple)
    if "OBV" in df.columns and len(df) >= 3 and not pd.isna(df["OBV"].iloc[-1]) and not pd.isna(df["close"].iloc[-1]):
        # price up + OBV down -> distribution -> bias short
        if df["close"].iloc[-1] > df["close"].iloc[-2] and df["OBV"].iloc[-1] < df["OBV"].iloc[-2]:
            votes.append("obv_div_down")
            feature_count += 1
        if df["close"].iloc[-1] < df["close"].iloc[-2] and df["OBV"].iloc[-1] > df["OBV"].iloc[-2]:
            votes.append("obv_div_up")
            feature_count += 1

    # 11) Heikin-Ashi exhaustion (pre-top): many green HA then bearish HA
    if "ha_dir" in df.columns and len(df) >= 5:
        last_ha = df["ha_dir"].iloc[-1]
        prev_ha_window = df["ha_dir"].iloc[-5:-1]
        if (prev_ha_window > 0).all() and last_ha < 0:
            votes.append("ha_exhaustion")
            feature_count += 1

    # 12) Liquidity grab near extremes
    if bool(last.get("liquidity_grab_up", False)):
        votes.append("liquidity_grab_up")
        feature_count += 1
    if bool(last.get("liquidity_grab_down", False)):
        votes.append("liquidity_grab_down")
        feature_count += 1

    # 13) Fair value gap presence on last bar
    if bool(last.get("fvg_up", False)):
        votes.append("fvg_up")
        feature_count += 1
    if bool(last.get("fvg_down", False)):
        votes.append("fvg_down")
        feature_count += 1

    # 14) Pivot probe: price crossing previous R1 / S1
    if not pd.isna(last.get("pivot_R1", np.nan)):
        if last["close"] > last["pivot_R1"]:
            votes.append("pivot_break_r1")
            feature_count += 1
        if last["close"] < last["pivot_S1"]:
            votes.append("pivot_break_s1")
            feature_count += 1

    # 15) CCI extremes
    if not pd.isna(last.get("CCI", np.nan)):
        if last["CCI"] > 100:
            votes.append("cci_overbought")
            feature_count += 1
        if last["CCI"] < -100:
            votes.append("cci_oversold")
            feature_count += 1

    # 16) StochRSI cross
    if not pd.isna(last.get("stoch_k", np.nan)) and not pd.isna(last.get("stoch_d", np.nan)):
        if df["stoch_k"].iloc[-2] <= df["stoch_d"].iloc[-2] and df["stoch_k"].iloc[-1] > df["stoch_d"].iloc[-1]:
            votes.append("stoch_up")
            feature_count += 1
        if df["stoch_k"].iloc[-2] >= df["stoch_d"].iloc[-2] and df["stoch_k"].iloc[-1] < df["stoch_d"].iloc[-1]:
            votes.append("stoch_down")
            feature_count += 1

    # 17) small fractal (swing high / low)
    if len(df) >= 5:
        i = -3
        curr = df["high"].iloc[i]
        if df["high"].iloc[i] > df["high"].iloc[i-1] and df["high"].iloc[i] > df["high"].iloc[i+1]:
            votes.append("swing_high")
            feature_count += 1
        currl = df["low"].iloc[i]
        if df["low"].iloc[i] < df["low"].iloc[i-1] and df["low"].iloc[i] < df["low"].iloc[i+1]:
            votes.append("swing_low")
            feature_count += 1

    # 18) price relative to 50-sma (momentum)
    if not pd.isna(last.get("sma_50", np.nan)):
        if last["close"] > last["sma_50"]:
            votes.append("above_sma50")
            feature_count += 1
        else:
            votes.append("below_sma50")
            feature_count += 1

    # 19) short-term momentum exhaustion: last two closes drop after spike
    if len(df) >= 4:
        if df["close"].iloc[-4] < df["close"].iloc[-3] and df["close"].iloc[-3] < df["close"].iloc[-2] and df["close"].iloc[-1] < df["close"].iloc[-2]:
            # extended up then drop -> potential top
            votes.append("momentum_flip_down")
            feature_count += 1

    # 20) small candle patterns: bearish engulfing at top
    if len(df) >= 2:
        prev_c = df["close"].iloc[-2]
        prev_o = df["open"].iloc[-2]
        cur_c = df["close"].iloc[-1]
        cur_o = df["open"].iloc[-1]
        if prev_c > prev_o and cur_c < cur_o and cur_o > prev_c and cur_c < prev_o:
            votes.append("bearish_engulfing")
            feature_count += 1

    # derive majority direction of votes
    long_votes = sum(1 for v in votes if direction_from_vote(v) == "LONG")
    short_votes = sum(1 for v in votes if direction_from_vote(v) == "SHORT")

    # fallback: some votes are directional by their string (like "macd_hist_up" contains 'up')
    # compute crude confidence: normalized feature_count / possible_features (~20)
    conf = feature_count / 20.0
    # additional boost for volume + pattern agreement
    if "vol_spike" in votes:
        conf += 0.05
    # if many pattern agreement
    if long_votes + short_votes > 0:
        maj = max(long_votes, short_votes)
        conf += (maj / (long_votes + short_votes)) * 0.1  # up to +0.1

    conf = max(0.0, min(1.0, conf))

    # construct final label
    label = "WATCH"
    if conf >= CONF_THRESHOLD_STRONG:
        label = "STRONG"
    elif conf >= CONF_THRESHOLD_MEDIUM:
        label = "MEDIUM"
    elif conf >= 0.2:
        label = "WEAK"
    else:
        label = "WATCH"

    final_signal = None
    # require at least MIN_FEATURES_TO_SIGNAL features and some directional majority for final signal
    if feature_count >= MIN_FEATURES_TO_SIGNAL and (long_votes != short_votes):
        final_signal = "LONG" if long_votes > short_votes else "SHORT"
        # pre-top detect: if many short directional votes + ha_exhaustion/obv_div_down/liquidity_grab_up => pre-top SHORT
        if final_signal == "SHORT" and any(x in votes for x in ("ha_exhaustion","obv_div_down","liquidity_grab_up","bearish_engulfing","momentum_flip_down")):
            # strengthen short and flag as pre-top
            votes.append("pre-top_candidate")
            conf = min(1.0, conf + 0.08)

    # attach details
    details = {
        "votes": votes,
        "feature_count": feature_count,
        "long_votes": long_votes,
        "short_votes": short_votes,
        "confidence": conf,
        "label": label,
        "last_close": float(last["close"]),
        "time": str(last.name)
    }
    return final_signal, details

# -------------------------
# High level: analyze symbol and possibly send Telegram
# -------------------------
def analyze_and_maybe_alert(symbol: str, interval: str = "15m", send_alert=True):
    try:
        df = fetch_klines(symbol, interval=interval, limit=EMA_SCAN_LIMIT)
        if df is None or len(df) < 60:
            logger.debug("Not enough data for %s", symbol)
            return None

        df = apply_all_features(df)
        final_signal, details = detect_pre_top_and_signals(df)
        details["symbol"] = symbol
        details["interval"] = interval

        # only alert for MEDIUM / STRONG signals (user wanted no 'watch' messages)
        if final_signal and details["label"] in ("MEDIUM", "STRONG"):
            # check dedupe: don't resend identical signal for same symbol+interval within short period
            key = f"{symbol}_{interval}"
            last = state.get("signals", {}).get(key)
            last_time = None
            if last:
                last_time = last.get("time")
            # if last is same signal and within 30 minutes, skip
            resend_allowed = True
            if last and last.get("signal") == final_signal:
                try:
                    last_dt = datetime.fromisoformat(last_time)
                    age_mins = (datetime.now(timezone.utc) - last_dt).total_seconds() / 60.0
                    if age_mins < 30:
                        resend_allowed = False
                except Exception:
                    pass

            if resend_allowed and send_alert:
                # build pretty message
                label = details["label"]
                conf = details["confidence"]
                txt = (
                    f"⚡ *Signal*\n"
                    f"Symbol: `{symbol}`\n"
                    f"Interval: `{interval}`\n"
                    f"Signal: *{final_signal}* ({label})\n"
                    f"Price: `{details['last_close']}`\n"
                    f"Confidence: `{conf:.2f}`\n"
                    f"Time: `{details['time']}`\n"
                )
                if details.get("votes"):
                    txt += "🧩 Patterns: " + ", ".join(details["votes"]) + "\n"
                # send
                send_telegram(txt)
                # update state
                state.setdefault("signals", {})[key] = {"signal": final_signal, "time": details["time"], "confidence": conf}
                # record history
                hist = state.setdefault("signal_history", {}).setdefault(symbol, [])
                hist.append({"time": details["time"], "interval": interval, "signal": final_signal, "confidence": conf})
                if len(hist) > 500:
                    state["signal_history"][symbol] = hist[-500:]
                state["last_scan"] = str(datetime.now(timezone.utc))
                save_json_safe(STATE_FILE, state)
                logger.info("Alert sent %s %s %s (conf=%.2f)", symbol, interval, final_signal, conf)
        else:
            # store last scan time (no alert)
            state["last_scan"] = str(datetime.now(timezone.utc))
            save_json_safe(STATE_FILE, state)
        return {"symbol": symbol, "signal": final_signal, "details": details}
    except Exception as e:
        logger.exception("analyze_and_maybe_alert error %s: %s", symbol, e)
        return None

# -------------------------
# Master scan: top N symbols parallel
# -------------------------
def scan_top_symbols(interval="15m", top_limit=TOP_LIMIT, send_alert=True):
    logger.info("Starting scan_top_symbols interval=%s top=%d", interval, top_limit)
    symbols = get_top_symbols_by_volume(limit=top_limit)
    if not symbols:
        symbols = get_all_usdt_symbols_from_binance()[:top_limit]
    results = []
    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as exe:
        futures = {exe.submit(analyze_and_maybe_alert, s, interval, send_alert): s for s in symbols}
        for fut in as_completed(futures):
            s = futures[fut]
            try:
                res = fut.result()
                if res:
                    results.append(res)
            except Exception as e:
                logger.exception("scan future error %s: %s", s, e)
    logger.info("Scan finished, hits=%d", len(results))
    return results

# -------------------------
# Scheduler & webhook registration
# -------------------------
scheduler = BackgroundScheduler()

def scheduled_scan_job():
    try:
        scan_top_symbols(interval="15m", top_limit=TOP_LIMIT, send_alert=True)
    except Exception as e:
        logger.exception("scheduled_scan_job error: %s", e)

# register periodic job
scheduler.add_job(scheduled_scan_job, "interval", minutes=max(1, SCAN_INTERVAL_MINUTES), id="scan_job")
scheduler.start()

# auto set webhook if WEBHOOK_URL provided
def auto_register_webhook():
    if WEBHOOK_URL and TELEGRAM_TOKEN:
        logger.info("Registering Telegram webhook: %s", WEBHOOK_URL)
        set_telegram_webhook(WEBHOOK_URL)
    else:
        logger.info("WEBHOOK_URL or TELEGRAM_TOKEN not provided, skipping webhook registration.")

# run in background so Flask start isn't blocked
Thread(target=auto_register_webhook, daemon=True).start()

# -------------------------
# Flask routes
# -------------------------
@app.route("/")
def home():
    return jsonify({"status": "ok", "time": str(datetime.now(timezone.utc)), "state_signals_count": len(state.get("signals", {}))})

@app.route("/scan_now", methods=["GET"])
def scan_now_route():
    try:
        Thread(target=scan_top_symbols, kwargs={"interval": "15m", "top_limit": TOP_LIMIT, "send_alert": True}, daemon=True).start()
        return jsonify({"status": "scanning", "top_limit": TOP_LIMIT})
    except Exception as e:
        logger.exception("scan_now error: %s", e)
        return jsonify({"error": str(e)}), 500

@app.route("/status", methods=["GET"])
def status_route():
    return jsonify({
        "state": state,
        "time": str(datetime.now(timezone.utc))
    })

# Telegram webhook endpoint (path contains token for safety)
@app.route(f"/telegram_webhook/<token>", methods=["POST"])
def telegram_webhook(token):
    try:
        if token != TELEGRAM_TOKEN:
            return jsonify({"ok": False, "reason": "invalid token"}), 403
        update = request.get_json(force=True)
        logger.debug("Telegram update: %s", update)
        # we only care about commands for now
        if "message" in update:
            msg = update["message"]
            chat_id = msg["chat"]["id"]
            text = msg.get("text", "")
            if text and text.startswith("/scan"):
                # allow manual scan
                Thread(target=scan_top_symbols, kwargs={"interval": "15m", "top_limit": TOP_LIMIT, "send_alert": True}, daemon=True).start()
                send_telegram("Manual scan started.")
            elif text and text.startswith("/status"):
                send_telegram(f"Status: signals={len(state.get('signals',{}))}, last_scan={state.get('last_scan')}")
        return jsonify({"ok": True})
    except Exception as e:
        logger.exception("telegram_webhook error: %s", e)
        return jsonify({"ok": False, "error": str(e)}), 500

# -------------------------
# BOOTSTRAP initial warm scan (non-blocking)
# -------------------------
def warmup_and_first_scan():
    try:
        logger.info("Warmup: initial scan (top %d)", TOP_LIMIT)
        scan_top_symbols(interval="15m", top_limit=TOP_LIMIT, send_alert=False)
        # small sleep then perform alerting scan so we don't flood on startup
        time.sleep(2)
        scan_top_symbols(interval="15m", top_limit=TOP_LIMIT, send_alert=True)
    except Exception as e:
        logger.exception("warmup_and_first_scan error: %s", e)

Thread(target=warmup_and_first_scan, daemon=True).start()

# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    logger.info("Starting universal pre-top detector bot")
    # if webhook URL provided, ensure webhook set
    if WEBHOOK_URL:
        logger.info("Attempting webhook registration...")
        set_telegram_webhook(WEBHOOK_URL)
    # run Flask
    app.run(host="0.0.0.0", port=PORT)