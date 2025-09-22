import os
import json
import logging
import re
from datetime import datetime, timezone
from threading import Thread
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests
import ta
import mplfinance as mpf
import matplotlib.pyplot as plt
import io

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
PARALLEL_WORKERS = int(os.getenv("PARALLEL_WORKERS", "6"))
STATE_FILE = "state.json"
CONF_THRESHOLD_MEDIUM = 0.3

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
        if os.path.exists(tmp):
            os.replace(tmp, path)
        else:
            logger.error("Temp file %s not created, skipping save", tmp)
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

# ---------------- GATE.IO API ----------------
GATE_IO_API_URL = "https://api.gateio.ws/api2/1/candlestick2"

ALL_SYMBOLS = [
    "BTC_USDT","ETH_USDT","ADA_USDT","SOL_USDT","DOT_USDT","XRP_USDT","LTC_USDT",
    "TRX_USDT","DOGE_USDT","AVAX_USDT","MATIC_USDT"
]

def fetch_klines(symbol, interval="15m", limit=200):
    try:
        resp = requests.get(GATE_IO_API_URL, params={"currency_pair": symbol, "type": interval, "range": limit}, timeout=5)
        data = resp.json()
        if not data or len(data) < 20:
            return None
        df = pd.DataFrame(data, columns=["timestamp","low","high","open","close","volume"])
        df[["low","high","open","close","volume"]] = df[["low","high","open","close","volume"]].astype(float)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        df.set_index("timestamp", inplace=True)
        return df
    except Exception as e:
        logger.exception("fetch_klines error %s: %s", symbol, e)
        return None

# ---------------- FEATURES ----------------
def apply_all_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["support"] = df["low"].rolling(20).min()
    df["resistance"] = df["high"].rolling(20).max()
    df["vol_ma20"] = df["volume"].rolling(20).mean()
    df["vol_spike"] = df["volume"] > 1.5 * df["vol_ma20"]
    df["body"] = df["close"] - df["open"]
    df["range"] = df["high"] - df["low"]
    df["upper_shadow"] = df["high"] - df[["close","open"]].max(axis=1)
    df["lower_shadow"] = df[["close","open"]].min(axis=1) - df["low"]
    return df

# ---------------- SIGNAL DETECTION ----------------
def detect_signal(df: pd.DataFrame):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    votes = []
    confidence = 0.3

    # Hammer / Shooting star
    if last["lower_shadow"] > 2*abs(last["body"]) and last["body"] > 0:
        votes.append("hammer_bull")
        confidence *= 1.5 if last["close"] <= last["support"]*1.02 else 1.2
    elif last["upper_shadow"] > 2*abs(last["body"]) and last["body"] < 0:
        votes.append("shooting_star")
        confidence *= 1.5 if last["close"] >= last["resistance"]*0.98 else 1.2

    # Engulfing
    if last["body"] > 0 and prev["body"] < 0 and last["close"] > prev["open"] and last["open"] < prev["close"]:
        votes.append("bullish_engulfing")
        confidence *= 1.25
    elif last["body"] < 0 and prev["body"] > 0 and last["close"] < prev["open"] and last["open"] > prev["close"]:
        votes.append("bearish_engulfing")
        confidence *= 1.25

    if last["vol_spike"]:
        votes.append("volume_spike")
        confidence *= 1.15

    # Fake breakout / flip
    if prev["close"] > prev["resistance"] and last["close"] < last["resistance"]: votes.append("fake_breakout_short"); confidence*=1.2
    if prev["close"] < prev["support"] and last["close"] > last["support"]: votes.append("fake_breakout_long"); confidence*=1.2
    if prev["close"] < prev["resistance"] and last["close"] > last["resistance"]: votes.append("resistance_flip_support"); confidence*=1.15
    if prev["close"] > prev["support"] and last["close"] < last["support"]: votes.append("support_flip_resistance"); confidence*=1.15

    # Pre-top
    pretop = False
    if len(df)>=10 and (last["close"]-df["close"].iloc[-10])/df["close"].iloc[-10] > 0.1:
        pretop = True
        votes.append("pretop")
        confidence += 0.1

    # Action
    action="WATCH"
    if last["close"] >= last["resistance"]*0.98: action="SHORT"
    elif last["close"] <= last["support"]*1.02: action="LONG"

    confidence = max(0.0, min(1.0, confidence))
    return action, votes, pretop, last, confidence

# ---------------- PLOT ----------------
def plot_signal_candles(df, symbol, action, tp1=None, tp2=None, tp3=None, sl=None, entry=None):
    addplots=[]
    if tp1: addplots.append(mpf.make_addplot([tp1]*len(df), color='green', linestyle="--"))
    if tp2: addplots.append(mpf.make_addplot([tp2]*len(df), color='lime', linestyle="--"))
    if tp3: addplots.append(mpf.make_addplot([tp3]*len(df), color='darkgreen', linestyle="--"))
    if sl: addplots.append(mpf.make_addplot([sl]*len(df), color='red', linestyle="--"))
    if entry: addplots.append(mpf.make_addplot([entry]*len(df), color='blue', linestyle="--"))
    fig, ax = mpf.plot(df.tail(120), type='candle', style='yahoo', title=f"{symbol} - {action}", addplot=addplots, returnfig=True)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return buf

# ---------------- ANALYZE ----------------
def analyze_and_alert(symbol):
    df = fetch_klines(symbol, limit=200)
    if df is None:
        logger.info("Symbol=%s: Not enough data", symbol)
        return

    df = apply_all_features(df)
    action, votes, pretop, last, confidence = detect_signal(df)
    if action=="WATCH": return

    atr = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], 14).average_true_range().iloc[-1]
    if action=="LONG":
        entry = last["support"]*1.001
        stop_loss = last["support"]*0.99
        tp1 = entry+atr; tp2=entry+2*atr; tp3=last["resistance"]
    else:
        entry = last["resistance"]*0.999
        stop_loss = last["resistance"]*1.01
        tp1 = entry-atr; tp2=entry-2*atr; tp3=last["support"]

    rr1 = (tp1-entry)/(entry-stop_loss) if action=="LONG" else (entry-tp1)/(stop_loss-entry)
    if rr1 < 2: return
    rr2 = (tp2-entry)/(entry-stop_loss) if action=="LONG" else (entry-tp2)/(stop_loss-entry)
    rr3 = (tp3-entry)/(entry-stop_loss) if action=="LONG" else (entry-tp3)/(stop_loss-entry)

    msg = (
        f"⚡ TRADE SIGNAL\n"
        f"Symbol: {symbol}\n"
        f"Action: {action}\n"
        f"🔹 Entry: {entry:.6f}\n"
        f"🛑 Stop-Loss: {stop_loss:.6f}\n"
        f"✅ TP1: {tp1:.6f} (RR {rr1:.2f})\n"
        f"✅✅ TP2: {tp2:.6f} (RR {rr2:.2f})\n"
        f"🏁 TP3: {tp3:.6f} (RR {rr3:.2f})\n"
        f"Confidence: {confidence:.2f}\n"
        f"Patterns: {', '.join(votes)}\n"
    )

    photo_buf = plot_signal_candles(df, symbol, action, tp1=tp1, tp2=tp2, tp3=tp3, sl=stop_loss, entry=entry)
    send_telegram(msg, photo=photo_buf)

    state.setdefault("signals", {})[symbol] = {
        "action": action, "entry": entry, "sl": stop_loss,
        "tp1": tp1, "tp2": tp2, "tp3": tp3,
        "rr1": rr1, "rr2": rr2, "rr3": rr3,
        "confidence": confidence, "time": str(last.name), "votes": votes
    }
    save_json_safe(STATE_FILE, state)

# ---------------- MASTER SCAN ----------------
def scan_all_symbols():
    logger.info("Starting scan for %d symbols", len(ALL_SYMBOLS))
    def safe_analyze(sym):
        try:
            analyze_and_alert(sym)
        except Exception as e:
            logger.exception("Error analyzing symbol %s: %s", sym, e)
    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as exe:
        list(exe.map(safe_analyze, ALL_SYMBOLS))
    state["last_scan"] = str(datetime.now(timezone.utc))
    save_json_safe(STATE_FILE, state)
    logger.info("Scan finished at %s", state["last_scan"])

# ---------------- FLASK ----------------
from flask import Flask, request, jsonify
app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({
        "status": "ok",
        "time": str(datetime.now(timezone.utc)),
        "signals": len(state.get("signals", {}))
    })

@app.route("/telegram_webhook/<token>", methods=["POST"])
def telegram_webhook(token):
    if token != TELEGRAM_TOKEN:
        return jsonify({"ok": False, "error": "invalid token"}), 403
    update = request.get_json(force=True) or {}
    text = update.get("message", {}).get("text", "").lower().strip()
    if text.startswith("/scan"):
        send_telegram("⚡ Manual scan started.")
        Thread(target=scan_all_symbols, daemon=True).start()
    return jsonify({"ok": True})

# ---------------- MAIN ----------------
if __name__ == "__main__":
    logger.info("Starting pre-top detector bot for Gate.io")
    Thread(target=scan_all_symbols, daemon=True).start()
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")))