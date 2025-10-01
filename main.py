import os
import time
import json
import logging
import re
from datetime import datetime, timezone
from threading import Thread
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import ta
import mplfinance as mpf
import io

from binance.client import Client

# ---------------- LOGGING ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler("bot.log"), logging.StreamHandler()]
)
logger = logging.getLogger("pretop-bot")

# ---------------- CONFIG ----------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
CHAT_ID = os.getenv("CHAT_ID", "")
PORT = int(os.getenv("PORT", "5000"))
PARALLEL_WORKERS = int(os.getenv("PARALLEL_WORKERS", "6"))
SCAN_INTERVAL = 60  # секунд
STATE_FILE = "state.json"
CONF_THRESHOLD_MEDIUM = 0.3

# ---------------- BINANCE CLIENT ----------------
binance_client = Client(api_key="", api_secret="")

# ---------------- BINANCE WEBSOCKET (для топ-символів) ----------------
from binance import ThreadedWebsocketManager
import threading

_tickers_cache = {}
_last_symbols = []

def start_ws_listener():
    """Запускає Binance Futures miniTicker WebSocket і оновлює кеш"""
    def handle_message(msg):
        global _tickers_cache
        if isinstance(msg, dict) and "s" in msg:
            sym = msg["s"]
            _tickers_cache[sym] = {
                "lastPrice": float(msg["c"]),
                "changePercent": float(msg["P"]),
                "volume": float(msg["v"]),
                "quoteVolume": float(msg["q"])
            }
        elif isinstance(msg, list):
            for t in msg:
                sym = t["s"]
                _tickers_cache[sym] = {
                    "lastPrice": float(t["c"]),
                    "changePercent": float(t["P"]),
                    "volume": float(t["v"]),
                    "quoteVolume": float(t["q"])
                }

    twm = ThreadedWebsocketManager()
    twm.start()
    twm.start_futures_miniticker_socket(callback=handle_message)
    threading.Thread(target=twm.join, daemon=True).start()

def fetch_top_symbols(limit=None):
    try:
        tickers = binance_client.futures_ticker()
        usdt_pairs = [t for t in tickers if t['symbol'].endswith("USDT")]

        # Сортуємо по % зміни (щоб активніші були на початку списку)
        sorted_pairs = sorted(
            usdt_pairs, 
            key=lambda x: abs(float(x.get("priceChangePercent", 0))), 
            reverse=True
        )

        # Якщо limit заданий – беремо тільки top-N, інакше всі
        if limit:
            return [d["symbol"] for d in sorted_pairs[:limit]]
        else:
            return [d["symbol"] for d in sorted_pairs]
    except Exception as e:
        logger.exception("Error fetching top symbols: %s", e)
        return ALL_USDT

# ---------------- STATE ----------------
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
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp, path)
    except Exception as e:
        logger.exception("save_json_safe error %s: %s", path, e)

state = load_json_safe(STATE_FILE, {"signals": {}, "last_scan": None})

# ---------------- TELEGRAM ----------------
MARKDOWNV2_ESCAPE = r"_*[]()~`>#+-=|{}.!"

def escape_md_v2(text: str) -> str:
    return re.sub(f"([{re.escape(MARKDOWNV2_ESCAPE)}])", r"\\\1", str(text))

def send_telegram(text: str, photo=None):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        return
    try:
        if photo:
            files = {'photo': ('signal.png', photo, 'image/png')}
            data = {'chat_id': CHAT_ID, 'caption': escape_md_v2(text), 'parse_mode': 'MarkdownV2'}
            requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto", data=data, files=files, timeout=10)
        else:
            payload = {"chat_id": CHAT_ID, "text": escape_md_v2(text), "parse_mode": "MarkdownV2"}
            requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", json=payload, timeout=10)
    except Exception as e:
        logger.exception("send_telegram error: %s", e)

# ---------------- BINANCE DATA FETCH ----------------
BINANCE_REST_URL = "https://fapi.binance.com/fapi/v1/klines"
ALL_USDT = ["BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT","ADAUSDT","DOGEUSDT",
            "DOTUSDT","TRXUSDT","LTCUSDT","AVAXUSDT","SHIBUSDT","LINKUSDT","ATOMUSDT","XMRUSDT",
            "ETCUSDT","XLMUSDT","APTUSDT","NEARUSDT","FILUSDT","ICPUSDT","GRTUSDT","AAVEUSDT"]  # можна додати більше

def fetch_klines_rest(symbol, interval="15m", limit=500):
    try:
        resp = requests.get(BINANCE_REST_URL, params={"symbol": symbol, "interval": interval, "limit": limit}, timeout=5)
        data = resp.json()
        df = pd.DataFrame(data, columns=[
            "open_time","open","high","low","close","volume",
            "close_time","quote_asset_volume","trades",
            "taker_buy_base","taker_buy_quote","ignore"
        ])
        for col in ["open","high","low","close","volume"]:
            df[col] = df[col].astype(float)
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df.set_index("open_time", inplace=True)
        return df
    except Exception as e:
        logger.exception("REST fetch error for %s: %s", symbol, e)
        return None

def fetch_klines(symbol, limit=500):
    df = fetch_klines_rest(symbol, limit=limit)
    return df

# ---------------- FEATURE ENGINEERING ----------------
def apply_pro_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # --- Support/Resistance ---
    df["support"] = df["low"].rolling(20).min()
    df["resistance"] = df["high"].rolling(20).max()

    # --- Volume ---
    df["vol_ma20"] = df["volume"].rolling(20).mean()
    df["vol_spike"] = df["volume"] > 1.5*df["vol_ma20"]
    df["volume_cluster"] = df["volume"] > 2*df["vol_ma20"]

    # --- Candle ---
    df["body"] = df["close"] - df["open"]
    df["range"] = df["high"] - df["low"]
    df["upper_shadow"] = df["high"] - df[["close","open"]].max(axis=1)
    df["lower_shadow"] = df[["close","open"]].min(axis=1) - df["low"]

    # --- Liquidity grabs / traps ---
    df["liquidity_grab_long"] = (df["low"] < df["support"]) & (df["close"] > df["support"])
    df["liquidity_grab_short"] = (df["high"] > df["resistance"]) & (df["close"] < df["resistance"])
    df["false_break_high"] = (df["high"] > df["resistance"]) & (df["close"] < df["resistance"])
    df["false_break_low"] = (df["low"] < df["support"]) & (df["close"] > df["support"])
    df["bull_trap"] = (df["close"] < df["open"]) & (df["high"] > df["resistance"])
    df["bear_trap"] = (df["close"] > df["open"]) & (df["low"] < df["support"])

    # --- Retests ---
    df["retest_support"] = abs(df["close"] - df["support"])/df["support"] < 0.003
    df["retest_resistance"] = abs(df["close"] - df["resistance"])/df["resistance"] < 0.003

    # --- Trend ---
    df["trend_ma"] = df["close"].rolling(20).mean()
    df["trend_up"] = df["close"] > df["trend_ma"]
    df["trend_down"] = df["close"] < df["trend_ma"]

    # --- Wick exhaustion ---
    df["long_lower_wick"] = df["lower_shadow"] > 2*abs(df["body"])
    df["long_upper_wick"] = df["upper_shadow"] > 2*abs(df["body"])

    # --- Momentum / Imbalance ---
    df["imbalance_up"] = (df["body"]>0) & (df["body"]>df["range"]*0.6)
    df["imbalance_down"] = (df["body"]<0) & (abs(df["body"])>df["range"]*0.6)

    # --- ATR Squeeze ---
    df["atr"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()
    df["squeeze"] = df["atr"] < df["atr"].rolling(50).mean()*0.7

    # --- Delta divergence ---
    df["delta_div_long"] = (df["body"]>0) & (df["volume"]<df["vol_ma20"])
    df["delta_div_short"] = (df["body"]<0) & (df["volume"]<df["vol_ma20"])

    # --- Breakout continuation ---
    df["breakout_cont_long"] = (df["close"]>df["resistance"]) & (df["volume"]>df["vol_ma20"])
    df["breakout_cont_short"] = (df["close"]<df["support"]) & (df["volume"]>df["vol_ma20"])

    # --- Combo ---
    df["combo_bullish"] = df["imbalance_up"] & df["vol_spike"] & df["trend_up"]
    df["combo_bearish"] = df["imbalance_down"] & df["vol_spike"] & df["trend_down"]

    # --- Accumulation ---
    df["accumulation_zone"] = (df["range"] < df["range"].rolling(20).mean()*0.5) & (df["volume"]>df["vol_ma20"])

    return df

# ---------------- ADVANCED 15 FEATURES ----------------
def apply_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    df = apply_pro_features(df)

    # 1. Volume Spike Divergence
    df["volume_spike_div"] = (df["volume"] > 2*df["vol_ma20"]) & (abs(df["close"].pct_change(3))<0.005)

    # 2. Pre-Top Candle Cluster
    df["pre_top_candle_cluster"] = ((df["high"] - df[["close","open"]].max(axis=1))/df["close"]>0.03) & (df["close"].rolling(5).max()==df["close"])

    # 3. Pre-Bottom Wick Absorption
    df["pre_bottom_wick_abs"] = ((df[["close","open"]].min(axis=1)-df["low"])/df["close"]>0.03) & (df["close"].rolling(5).min()==df["close"])

    # 4. OI Trend Confirmation (OI беремо як колонки df['OI'] при інтеграції)
    if "OI" in df.columns:
        df["oi_trend_up"] = (df["close"].diff()>0) & (df["OI"].diff()>0)
        df["oi_trend_down"] = (df["close"].diff()<0) & (df["OI"].diff()>0)

    # 5. Funding Rate Spike
    if "funding_rate" in df.columns:
        df["funding_spike"] = df["funding_rate"].diff() > 0.0005

    # 6. Volume-Price Divergence
    df["vp_div_top"] = (df["close"].diff(5)>0) & (df["volume"].diff(5)<0)
    df["vp_div_bottom"] = (df["close"].diff(5)<0) & (df["volume"].diff(5)>0)

    # 7. Wick-Volume Combo
    df["wick_vol_combo"] = ((df["high"]-df["close"])/df["close"]>0.03) & (df["volume"]>df["vol_ma20"]*1.5)

    # 8. EMA Squeeze Breakout
    df["ema20"] = df["close"].ewm(span=20).mean()
    df["ema50"] = df["close"].ewm(span=50).mean()
    df["ema_squeeze"] = abs(df["ema20"]-df["ema50"])/df["close"]<0.002
    df["ema_breakout_up"] = (df["close"]>df[["ema20","ema50"]].max(axis=1)) & df["ema_squeeze"]
    df["ema_breakout_down"] = (df["close"]<df[["ema20","ema50"]].min(axis=1)) & df["ema_squeeze"]

    # 9. RSI Divergence Extreme
    rsi = ta.momentum.RSIIndicator(df["close"],14).rsi()
    df["rsi_div_top"] = (df["close"].diff()>0) & (rsi.diff()<0) & (rsi>70)
    df["rsi_div_bottom"] = (df["close"].diff()<0) & (rsi.diff()>0) & (rsi<30)

    # 10. High-Low Volume Oscillator
    df["hl_vol_ratio"] = (df["high"]-df[["close","open"]].max(axis=1))/df["volume"]

    # 11. Candle Body Imbalance
    df["body_imbalance"] = (df["body"].shift(1)<df["body"]) & (df["body"].shift(2)<df["body"])

    # 12. Pre-Volume Exhaustion
    df["volume_drop_trend"] = (df["volume"]<df["vol_ma20"]*0.5) & (df["close"].diff(10)>0)

    # 13. OI/Price Divergence
    if "OI" in df.columns:
        df["oi_price_div"] = (df["close"].diff(5)>0) & (df["OI"].diff(5)<0)

    # 14. Funding-Volume Confluence
    if "funding_rate" in df.columns:
        df["funding_vol_confl"] = df["funding_spike"] & (df["volume"]>df["vol_ma20"]*1.5)

    # 15. Multi-Timeframe EMA Divergence
    df["ema_mtf_div"] = (df["ema20"]>df["ema50"]) & (df["ema20"].shift(12)<df["ema50"].shift(12))

    return df

# ---------------- SIGNAL DETECTION ----------------
def detect_signal_pro(df: pd.DataFrame):
    last = df.iloc[-1]
    votes = []
    confidence = 0.5
    pretop = False
    prebottom = False

    # --- Existing features ---
    for col, add_conf in [
        ("liquidity_grab_long",0.08), ("liquidity_grab_short",0.08),
        ("bull_trap",0.05), ("bear_trap",0.05),
        ("false_break_high",0.05), ("false_break_low",0.05),
        ("volume_cluster",0.05), ("breakout_cont_long",0.07), ("breakout_cont_short",0.07),
        ("imbalance_up",0.05), ("imbalance_down",0.05),
        ("squeeze",0.03),
        ("trend_up",0.05), ("trend_down",0.05),
        ("long_lower_wick",0.04), ("long_upper_wick",0.04),
        ("retest_support",0.05), ("retest_resistance",0.05),
        ("delta_div_long",0.06), ("delta_div_short",0.06),
        ("combo_bullish",0.1), ("combo_bearish",0.1),
        ("accumulation_zone",0.03)
    ]:
        if col in last and last[col]:
            votes.append(col)
            confidence += add_conf

    # --- Advanced features ---
    adv_features = [
        ("volume_spike_div",0.06), ("pre_top_candle_cluster",0.08),
        ("pre_bottom_wick_abs",0.08),
        ("oi_trend_up",0.05), ("oi_trend_down",0.05),
        ("funding_spike",0.06),
        ("vp_div_top",0.06), ("vp_div_bottom",0.06),
        ("wick_vol_combo",0.05),
        ("ema_breakout_up",0.07), ("ema_breakout_down",0.07),
        ("rsi_div_top",0.06), ("rsi_div_bottom",0.06),
        ("volume_drop_trend",0.05),
        ("oi_price_div",0.05),
        ("funding_vol_confl",0.08),
        ("ema_mtf_div",0.07)
    ]
    for col, add_conf in adv_features:
        if col in last and last[col]:
            votes.append(col)
            confidence += add_conf

    # --- Pre-top / Pre-bottom detection ---
    if len(df)>=10:
        if (last["close"]-df["close"].iloc[-10])/df["close"].iloc[-10]>0.10:
            pretop=True
            votes.append("pretop")
            confidence+=0.1
        if (df["close"].iloc[-10]-last["close"])/df["close"].iloc[-10]>0.10:
            prebottom=True
            votes.append("prebottom")
            confidence+=0.1

    # --- Probabilities LONG/SHORT ---
    score_long = sum([0.1 for s in ["combo_bullish","breakout_cont_long","delta_div_long","retest_support","liquidity_grab_long"] if s in last and last[s]])
    score_short = sum([0.1 for s in ["combo_bearish","breakout_cont_short","delta_div_short","retest_resistance","liquidity_grab_short"] if s in last and last[s]])

    prob_long = min(1.0, score_long)
    prob_short = min(1.0, score_short)

    # --- Action only on local extremes ---
    is_local_top = last["close"] == df["close"].rolling(5).max().iloc[-1]
    is_local_bottom = last["close"] == df["close"].rolling(5).min().iloc[-1]

    action = "WATCH"
    if (is_local_bottom or last["retest_support"]) and prob_long>0:
        action = "LONG"
    elif (is_local_top or last["retest_resistance"]) and prob_short>0:
        action = "SHORT"

    confidence = min(1.0,max(0.0,confidence))
    return action, votes, pretop, prebottom, last, confidence, prob_long, prob_short

# ---------------- QUALITY SCORE ----------------
def calculate_quality_score_pro(df,votes,confidence):
    score=confidence
    strong_signals = ["combo_bullish","combo_bearish","liquidity_grab_long","liquidity_grab_short",
                      "delta_div_long","delta_div_short","breakout_cont_long","breakout_cont_short"]
    medium_signals = ["bull_trap","bear_trap","false_break_high","false_break_low",
                      "volume_cluster","retest_support","retest_resistance"]
    weak_signals = ["trend_up","trend_down","long_lower_wick","long_upper_wick","squeeze",
                    "accumulation_zone","pretop"]
    for p in votes:
        if p in strong_signals: score+=0.1
        elif p in medium_signals: score+=0.05
        elif p in weak_signals: score+=0.02
    return min(score,1.0)

# ---------------- PLOT ----------------
def plot_signal_candles(df,symbol,action,tp1=None,tp2=None,tp3=None,sl=None,entry=None):
    addplots=[]
    if tp1: addplots.append(mpf.make_addplot([tp1]*len(df),color='green',linestyle="--"))
    if tp2: addplots.append(mpf.make_addplot([tp2]*len(df),color='lime',linestyle="--"))
    if tp3: addplots.append(mpf.make_addplot([tp3]*len(df),color='darkgreen',linestyle="--"))
    if sl: addplots.append(mpf.make_addplot([sl]*len(df),color='red',linestyle="--"))
    if entry: addplots.append(mpf.make_addplot([entry]*len(df),color='blue',linestyle="--"))

    fig,ax = mpf.plot(df.tail(200),type='candle',style='yahoo',title=f"{symbol} - {action}",addplot=addplots,returnfig=True)
    buf = io.BytesIO()
    fig.savefig(buf,format='png',bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return buf

# ---------------- ANALYZE AND ALERT ----------------
def analyze_and_alert(symbol:str):
    df = fetch_klines(symbol,limit=200)
    if df is None or len(df)<40: return
    df = apply_advanced_features(df)
    action,votes,pretop,prebottom,last,confidence,prob_long,prob_short = detect_signal_pro(df)

    if action=="WATCH": return

    # --- Entry / SL / TP using ATR ---
    atr = last["atr"] if "atr" in last else (last["high"]-last["low"])
    entry = last["close"]
    if action=="LONG":
        stop_loss = entry - atr*1.5
        tp1 = entry + atr*1.0
        tp2 = entry + atr*2.0
        tp3 = entry + atr*3.0
    elif action=="SHORT":
        stop_loss = entry + atr*1.5
        tp1 = entry - atr*1.0
        tp2 = entry - atr*2.0
        tp3 = entry - atr*3.0

    # --- R/R ---
    rr1 = (tp1-entry)/(entry-stop_loss) if action=="LONG" else (entry-tp1)/(stop_loss-entry)
    rr2 = (tp2-entry)/(entry-stop_loss) if action=="LONG" else (entry-tp2)/(stop_loss-entry)
    rr3 = (tp3-entry)/(entry-stop_loss) if action=="LONG" else (entry-tp3)/(stop_loss-entry)

    # --- Quality Score ---
    score = calculate_quality_score_pro(df,votes,confidence)
    logger.info("Symbol=%s action=%s confidence=%.2f score=%.2f votes=%s pretop=%s prebottom=%s RR1=%.2f",symbol,action,confidence,score,votes,pretop,prebottom,rr1)

    if confidence>=CONF_THRESHOLD_MEDIUM and score>=0.65 and rr1>=1.5:
        reasons=[]
        if "pretop" in votes: reasons.append("Pre-Top")
        if "prebottom" in votes: reasons.append("Pre-Bottom")
        if "combo_bullish" in votes or "combo_bearish" in votes: reasons.append("Combo")
        if "liquidity_grab_long" in votes or "liquidity_grab_short" in votes: reasons.append("Liquidity Grab")
        if "delta_div_long" in votes or "delta_div_short" in votes: reasons.append("Delta Divergence")
        if not reasons: reasons=["Pattern Mix"]

        msg=(
            f"⚡ TRADE SIGNAL\n"
            f"Symbol: {symbol}\n"
            f"Action: {action}\n"
            f"LONG probability: {prob_long:.2f} | SHORT probability: {prob_short:.2f}\n"
            f"Confidence: {confidence:.2f}\n"
            f"Quality Score: {score:.2f}\n"
            f"Entry: {entry:.6f}\n"
            f"Stop-Loss: {stop_loss:.6f}\n"
            f"Take-Profit 1: {tp1:.6f} (RR {rr1:.2f})\n"
            f"Take-Profit 2: {tp2:.6f} (RR {rr2:.2f})\n"
            f"Take-Profit 3: {tp3:.6f} (RR {rr3:.2f})\n"
            f"Reasons: {', '.join(reasons)}\n"
            f"Patterns: {', '.join(votes)}\n"
        )

        photo_buf = plot_signal_candles(df,symbol,action,tp1,tp2,tp3,stop_loss,entry)
        send_telegram(msg,photo=photo_buf)

        state.setdefault("signals",{})[symbol] = {
            "action":action,"entry":entry,"sl":stop_loss,"tp1":tp1,"tp2":tp2,"tp3":tp3,
            "rr1":rr1,"rr2":rr2,"rr3":rr3,"confidence":confidence,
            "score":score,"time":str(last.name),"last_price":float(last["close"]),
            "votes":votes,"prob_long":prob_long,"prob_short":prob_short
        }
        save_json_safe(STATE_FILE,state)

# ---------------- MASTER SCAN ----------------
def scan_all_symbols():
    symbols = fetch_top_symbols(limit=200)
    if not symbols: symbols = ALL_USDT
    logger.info("Scanning %d symbols",len(symbols))
    def safe_analyze(sym):
        try: analyze_and_alert(sym)
        except Exception as e: logger.exception("Error analyzing %s: %s",sym,e)
    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as exe:
        list(exe.map(safe_analyze,symbols))
    state["last_scan"]=str(datetime.now(timezone.utc))
    save_json_safe(STATE_FILE,state)
    logger.info("Scan finished at %s",state["last_scan"])

# ---------------- BACKGROUND SCANNER ----------------
def background_scanner():
    while True:
        try: scan_all_symbols()
        except Exception as e: logger.exception("Background scan error: %s", e)
        time.sleep(SCAN_INTERVAL)

Thread(target=background_scanner,daemon=True).start()

# ---------------- FLASK ----------------
from flask import Flask,request,jsonify
app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({"status":"ok","time":str(datetime.now(timezone.utc)),"signals":len(state.get("signals",{}))})

@app.route("/telegram_webhook/<token>",methods=["POST"])
def telegram_webhook(token):
    if token!=TELEGRAM_TOKEN:
        return jsonify({"ok":False,"error":"invalid token"}),403
    update = request.get_json(force=True) or {}
    text = update.get("message",{}).get("text","").lower().strip()
    if text.startswith("/scan"):
        send_telegram("⚡ Manual scan started.")
        Thread(target=scan_all_symbols,daemon=True).start()
    return jsonify({"ok":True})

# ---------------- MAIN ----------------
if __name__=="__main__":
    logger.info("Starting pre-top detector bot")
    start_ws_listener()
    app.run(host="0.0.0.0",port=PORT)