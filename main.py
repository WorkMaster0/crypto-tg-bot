# main.py — Pre-top бот з графіками, Telegram, профілями користувачів та історією сигналів
import os
import time
import json
import logging
import re
from datetime import datetime, timezone
from threading import Thread, Lock
from concurrent.futures import ThreadPoolExecutor
import io
import pandas as pd
import matplotlib.pyplot as plt
import requests
from flask import Flask, request, jsonify
from apscheduler.schedulers.background import BackgroundScheduler
import ta
import mplfinance as mpf
from fpdf import FPDF  # Для PDF історії сигналів

# ---------------- BINANCE CLIENT ----------------
try:
    from binance.client import Client as BinanceClient
    BINANCE_PY_AVAILABLE = True
except Exception:
    BINANCE_PY_AVAILABLE = False

# ---------------- CONFIG ----------------
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
CHAT_ID = os.getenv("CHAT_ID", "")
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "")
PORT = int(os.getenv("PORT", "5000"))

TOP_LIMIT = int(os.getenv("TOP_LIMIT", "100"))
SCAN_INTERVAL_MINUTES = int(os.getenv("SCAN_INTERVAL_MINUTES", "1"))
EMA_SCAN_LIMIT = int(os.getenv("EMA_SCAN_LIMIT", "500"))
PARALLEL_WORKERS = int(os.getenv("PARALLEL_WORKERS", "6"))

STATE_FILE = "state.json"
LOG_FILE = "bot.log"

CONF_THRESHOLD_MEDIUM = 0.60
CONF_THRESHOLD_STRONG = 0.80

# ---------------- LOGGING ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)
logger = logging.getLogger("pretop-bot")

# ---------------- BINANCE ----------------
if BINANCE_PY_AVAILABLE and BINANCE_API_KEY and BINANCE_API_SECRET:
    client = BinanceClient(api_key=BINANCE_API_KEY, api_secret=BINANCE_API_SECRET)
else:
    client = None
    logger.warning("Binance client unavailable or API keys missing")

# ---------------- FLASK ----------------
app = Flask(__name__)

# ---------------- STATE + LOCK ----------------
state_lock = Lock()

def load_json_safe(path, default):
    try:
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
    except json.JSONDecodeError:
        logger.warning("Corrupted JSON file %s, resetting...", path)
    except Exception as e:
        logger.exception("load_json_safe error %s: %s", path, e)
    return default

def save_json_safe(path, data):
    try:
        with state_lock:
            snapshot = json.loads(json.dumps(data, default=str))  # безпечна копія
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2, ensure_ascii=False, default=str)
        os.replace(tmp, path)
    except Exception as e:
        logger.exception("save_json_safe error %s: %s", path, e)

state = load_json_safe(STATE_FILE, {
    "signals": {},
    "last_scan": None,
    "signal_history": {},
    "win_stats": {},
    "user_profiles": {}
})

# ---------------- TELEGRAM ----------------
MARKDOWNV2_ESCAPE = r"_*[]()~`>#+-=|{}.!"

def escape_md_v2(text: str) -> str:
    return re.sub(f"([{re.escape(MARKDOWNV2_ESCAPE)}])", r"\\\1", str(text))

def send_telegram(text: str, photo=None, chat_id=None):
    chat_id = chat_id or CHAT_ID
    if not TELEGRAM_TOKEN or not chat_id:
        return
    try:
        if photo:
            files = {'photo': photo}
            data = {'chat_id': chat_id, 'caption': escape_md_v2(text), 'parse_mode': 'MarkdownV2'}
            requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto", data=data, files=files, timeout=10)
        else:
            payload = {
                "chat_id": chat_id,
                "text": escape_md_v2(text),
                "parse_mode": "MarkdownV2",
                "disable_web_page_preview": True
            }
            requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", json=payload, timeout=10)
    except Exception as e:
        logger.exception("send_telegram error: %s", e)

def set_telegram_webhook(webhook_url: str):
    if not TELEGRAM_TOKEN or not webhook_url:
        return
    try:
        resp = requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/setWebhook", json={"url": webhook_url}, timeout=10)
        logger.info("setWebhook resp: %s", resp.text if resp else "None")
    except Exception as e:
        logger.exception("set_telegram_webhook error: %s", e)

# ---------------- MARKET DATA ----------------
def get_all_usdt_symbols():
    if not client:
        return []
    try:
        ex = client.get_exchange_info()
        symbols = [
            s["symbol"] for s in ex["symbols"]
            if s["quoteAsset"] == "USDT" and s["status"] == "TRADING"
        ]
        blacklist = [
            "BUSD", "USDC", "FDUSD", "TUSD", "DAI", "EUR", "GBP", "AUD",
            "BTCST", "COIN", "AAPL", "TSLA", "MSFT", "META", "GOOG"
        ]
        filtered = [s for s in symbols if not any(b in s for b in blacklist)]
        return filtered
    except Exception as e:
        logger.exception("get_all_usdt_symbols error: %s", e)
        return []

def fetch_klines(symbol, interval="15m", limit=EMA_SCAN_LIMIT):
    for attempt in range(3):
        try:
            if not client:
                raise RuntimeError("Binance client unavailable")
            kl = client.get_klines(symbol=symbol, interval=interval, limit=limit)
            df = pd.DataFrame(kl, columns=["open_time","open","high","low","close","volume",
                                           "close_time","qav","num_trades","tb_base","tb_quote","ignore"])
            df = df[["open_time","open","high","low","close","volume"]].astype(float)
            df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
            df.set_index("open_time", inplace=True)
            return df
        except Exception as e:
            logger.warning("fetch_klines %s attempt %d error: %s", symbol, attempt+1, e)
            time.sleep(0.5)
    return None

# ---------------- FEATURE ENGINEERING ----------------
def apply_all_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    try:
        df["ema_8"] = ta.trend.EMAIndicator(df["close"], 8).ema_indicator()
        df["ema_20"] = ta.trend.EMAIndicator(df["close"], 20).ema_indicator()
        df["RSI_14"] = ta.momentum.RSIIndicator(df["close"], 14).rsi()
        macd = ta.trend.MACD(df["close"])
        df["MACD"] = macd.macd()
        df["MACD_signal"] = macd.macd_signal()
        df["MACD_hist"] = macd.macd_diff()
        adx = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], 14)
        df["ADX"] = adx.adx()
        df["ADX_pos"] = adx.adx_pos()
        df["ADX_neg"] = adx.adx_neg()
        df["support"] = df["low"].rolling(20).min()
        df["resistance"] = df["high"].rolling(20).max()
        df["ATR"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], 14).average_true_range()
    except Exception as e:
        logger.exception("apply_all_features error: %s", e)
    return df

# ---------------- SIGNAL DETECTION ----------------
def detect_signal(df: pd.DataFrame):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    votes = []
    confidence = 0.2

    if last["ema_8"] > last["ema_20"]:
        votes.append("ema_bull")
        confidence += 0.1
    else:
        votes.append("ema_bear")
        confidence += 0.05

    if last["MACD_hist"] > 0:
        votes.append("macd_up")
        confidence += 0.1
    else:
        votes.append("macd_down")
        confidence += 0.05

    if last["RSI_14"] < 30:
        votes.append("rsi_oversold")
        confidence += 0.08
    elif last["RSI_14"] > 70:
        votes.append("rsi_overbought")
        confidence += 0.08

    if last["ADX"] > 25:
        votes.append("strong_trend")
        confidence *= 1.1

    # Свічкові патерни
    body = last["close"] - last["open"]
    rng = last["high"] - last["low"]
    upper_shadow = last["high"] - max(last["close"], last["open"])
    lower_shadow = min(last["close"], last["open"]) - last["low"]
    candle_bonus = 1.0

    if lower_shadow > 2 * abs(body) and body > 0:
        votes.append("hammer_bull")
        candle_bonus = 1.2
    elif upper_shadow > 2 * abs(body) and body < 0:
        votes.append("shooting_star")
        candle_bonus = 1.2

    if body > 0 and prev["close"] < prev["open"] and last["close"] > prev["open"] and last["open"] < prev["close"]:
        votes.append("bullish_engulfing")
        candle_bonus = 1.25
    elif body < 0 and prev["close"] > prev["open"] and last["close"] < prev["open"] and last["open"] > prev["close"]:
        votes.append("bearish_engulfing")
        candle_bonus = 1.25

    if abs(body) < 0.1 * rng:
        votes.append("doji")
        candle_bonus = 1.1

    confidence *= candle_bonus

    pretop = False
    if len(df) >= 10 and (last["close"] - df["close"].iloc[-10])/df["close"].iloc[-10] > 0.1:
        pretop = True
        confidence += 0.1
        votes.append("pretop")

    action = "WATCH"
    if last["close"] >= last["resistance"] * 0.995:
        action = "SHORT"
    elif last["close"] <= last["support"] * 1.005:
        action = "LONG"

    confidence = max(0, min(1, confidence))
    return action, votes, pretop, last, confidence

# ---------------- PLOT SIGNAL CANDLES ----------------
def plot_signal_candles(df, symbol, action, votes, pretop):
    df_plot = df[['open','high','low','close','volume']].copy()
    df_plot.index.name = "Date"
    addplots = []
    last = df.iloc[-1]

    if pretop:
        addplots.append(
            mpf.make_addplot([df['close'].iloc[-i] for i in range(3,0,-1)],
                             type='scatter', markersize=100, marker='^', color='magenta')
        )

    patterns = {
        "bullish_engulfing": "green",
        "bearish_engulfing": "red",
        "hammer_bull": "lime",
        "shooting_star": "orange",
        "doji": "blue",
    }
    for pat, color in patterns.items():
        if pat in votes:
            addplots.append(
                mpf.make_addplot([last['close']]*len(df), type='scatter', markersize=60, marker='o', color=color)
            )

    entry = last['close'] if action in ["LONG","SHORT"] else None
    stop = last['support'] if action=="LONG" else last['resistance'] if action=="SHORT" else None
    take_levels = []
    if entry and stop:
        atr = last.get("ATR",0.0)
        if action=="LONG":
            take_levels = [entry + atr * 0.5 * i for i in range(1, 11)]
            stop = entry - atr * 1.5
        elif action=="SHORT":
            take_levels = [entry - atr * 0.5 * i for i in range(1, 11)]
            stop = entry + atr * 1.5

    if entry:
        addplots.append(mpf.make_addplot([entry]*len(df), type='scatter', markersize=40, marker='v', color='blue'))
        addplots.append(mpf.make_addplot([stop]*len(df), type='scatter', markersize=40, marker='x', color='red'))
        for tl in take_levels:
            addplots.append(mpf.make_addplot([tl]*len(df), type='scatter', markersize=30, marker='^', color='green'))

    # тільки основні рівні
    h_support = df['support'].dropna().iloc[-3:] if len(df['support'].dropna())>=3 else df['support'].dropna()
    h_resistance = df['resistance'].dropna().iloc[-3:] if len(df['resistance'].dropna())>=3 else df['resistance'].dropna()
    hlines = list(h_support.unique()) + list(h_resistance.unique())

    mc = mpf.make_marketcolors(up='green', down='red', wick='black', edge='black', volume='blue')
    s = mpf.make_mpf_style(marketcolors=mc, gridstyle='--', gridcolor='gray', facecolor='white')

    buf = io.BytesIO()
    mpf.plot(df_plot, type='candle', style=s, volume=True, addplot=addplots,
             hlines=dict(hlines=hlines, colors=['gray'], linestyle='dashed', linewidths=0.7),
             title=f"{symbol} — {action}", ylabel='Price', ylabel_lower='Volume',
             savefig=dict(fname=buf, dpi=100, bbox_inches='tight', facecolor='white'),
             tight_layout=True)
    buf.seek(0)
    return buf

# ---------------- ANALYZE SYMBOL ----------------
def analyze_and_alert(symbol: str, chat_id=None):
    df = fetch_klines(symbol)
    if df is None or len(df)<30:
        return
    df = apply_all_features(df)
    action, votes, pretop, last, confidence = detect_signal(df)
    with state_lock:
        prev_signal = state["signals"].get(symbol,"")

    logger.info("Symbol=%s action=%s confidence=%.2f votes=%s pretop=%s",
                symbol, action, confidence, votes, pretop)

    with state_lock:
        # Win stats
        win_stats = state.get("win_stats",{})
        if symbol not in win_stats:
            win_stats[symbol] = {"total":0,"wins":0}
        win_stats[symbol]["total"] += 1
        if action!="WATCH" and confidence>=CONF_THRESHOLD_MEDIUM:
            if pretop or "strong_trend" in votes:
                win_stats[symbol]["wins"] += 1
        state["win_stats"] = win_stats

        # Історія сигналів
        if symbol not in state["signal_history"]:
            state["signal_history"][symbol] = []
        state["signal_history"][symbol].append({
            "time": str(datetime.now(timezone.utc)),
            "action": action,
            "price": last['close'],
            "confidence": confidence
        })

    save_json_safe(STATE_FILE, state)

    if pretop:
        send_telegram(f"⚡ Pre-top detected for {symbol}, price={last['close']:.6f}", chat_id=chat_id)

    if action!="WATCH" and confidence>=CONF_THRESHOLD_MEDIUM and action!=prev_signal:
        msg = (
            f"⚡ TRADE SIGNAL\nSymbol: {symbol}\nAction: {action}\nPrice: {last['close']:.6f}\n"
            f"Support: {last['support']:.6f}\nResistance: {last['resistance']:.6f}\n"
            f"Confidence: {confidence:.2f}\nPre-top: {pretop}\n"
        )
        photo_buf = plot_signal_candles(df, symbol, action, votes, pretop)
        send_telegram(msg, photo=photo_buf, chat_id=chat_id)
        with state_lock:
            state["signals"][symbol] = action
        save_json_safe(STATE_FILE, state)

# ---------------- SMART SCAN JOB ----------------
def smart_scan_job():
    try:
        symbols = get_all_usdt_symbols()
        if not symbols:
            logger.warning("No symbols found for scanning.")
            return

        logger.info("Starting scan for %d symbols", len(symbols))
        try:
            with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as exe:
                list(exe.map(analyze_and_alert, symbols))
        except Exception as e:
            logger.exception("ThreadPoolExecutor error: %s", e)

        with state_lock:
            state["last_scan"] = str(datetime.now(timezone.utc))
        save_json_safe(STATE_FILE, state)

        # адаптація частоти
        strong_count = sum(
            1 for h in state["signal_history"].values()
            if h and h[-1]["confidence"]>=0.7
        )
        if strong_count > 3:
            new_interval = 0.5  # кожні 30 сек
        elif strong_count == 0:
            new_interval = 3    # раз на 3 хв
        else:
            new_interval = 1    # раз на хв
        try:
            scheduler.reschedule_job("scan", trigger="interval", minutes=new_interval)
            logger.info("Rescheduled scan interval to %s min", new_interval)
        except Exception:
            pass

    except Exception as e:
        logger.exception("smart_scan_job error: %s", e)

# ---------------- API ROUTES ----------------
@app.route(f"/telegram_webhook/{TELEGRAM_TOKEN}", methods=["POST"])
def telegram_webhook():
    data = request.json
    if not data:
        return jsonify({"status":"error","reason":"empty"})
    try:
        chat_id = data["message"]["chat"]["id"]
        text = data["message"]["text"].strip().lower()
    except Exception:
        return jsonify({"status":"error","reason":"invalid payload"})

    if text=="/signals":
        with state_lock:
            sigs = state["signals"]
        msg = "📊 Active signals:\n" + "\n".join(f"{s}: {a}" for s,a in sigs.items() if a!="WATCH")
        send_telegram(msg or "No signals", chat_id=chat_id)
    elif text=="/profile":
        with state_lock:
            profile = state["user_profiles"].get(str(chat_id),{})
        msg = f"👤 Profile: {profile}" if profile else "No profile found"
        send_telegram(msg, chat_id=chat_id)
    elif text=="/history":
        buf = io.BytesIO()
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Signal History", ln=True, align="C")
        with state_lock:
            hist = state["signal_history"]
        for sym, entries in hist.items():
            pdf.cell(200, 10, txt=sym, ln=True)
            for e in entries[-5:]:
                pdf.cell(200, 10, txt=f"{e['time']} {e['action']} {e['price']:.6f} conf={e['confidence']:.2f}", ln=True)
        pdf.output(buf)
        buf.seek(0)
        send_telegram("📑 Signal history attached", photo=buf, chat_id=chat_id)

    return jsonify({"status":"ok"})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status":"ok","last_scan":state.get("last_scan")})

# ---------------- MAIN ----------------
if __name__=="__main__":
    set_telegram_webhook(WEBHOOK_URL)
    scheduler = BackgroundScheduler()
    scheduler.add_job(smart_scan_job, "interval", minutes=SCAN_INTERVAL_MINUTES, id="scan")
    scheduler.start()
    logger.info("Bot started on port %d", PORT)
    app.run(host="0.0.0.0", port=PORT)