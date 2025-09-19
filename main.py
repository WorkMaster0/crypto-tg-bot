# main.py — Pre-top бот з графіками, Telegram, профілями користувачів, історією сигналів та ML графіками
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
from fpdf import FPDF

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

STATE_DIR = "data"
os.makedirs(STATE_DIR, exist_ok=True)
STATE_FILE = os.path.join(STATE_DIR, "state.json")
LOG_FILE = os.path.join(STATE_DIR, "bot.log")

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
scan_lock = Lock()  # Lock для APScheduler та ручних сканів

def load_json_safe(path, default):
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logger.exception("load_json_safe error %s: %s", path, e)
    return default

def save_json_safe(path, data):
    try:
        with state_lock:
            snapshot = json.loads(json.dumps(data, default=str, ensure_ascii=False))
        dir_path = os.path.dirname(os.path.abspath(path))
        os.makedirs(dir_path, exist_ok=True)
        tmp = os.path.join(dir_path, os.path.basename(path) + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    except Exception as e:
        logger.exception("save_json_safe error %s: %s", path, e)

state = load_json_safe(STATE_FILE, {
    "signals": {},
    "last_scan": None,
    "signal_history": {},
    "win_stats": {},
    "user_profiles": {},
})

# ---------------- TELEGRAM ----------------
MARKDOWNV2_ESCAPE = r"_*[]()~`>#+=|{}.!"

def escape_md_v2(text: str) -> str:
    return re.sub(f"([{re.escape(MARKDOWNV2_ESCAPE)}])", r"\\\1", str(text))

def send_telegram(text: str, photo=None, chat_id=None, is_document=False, filename="file.pdf"):
    chat_id = chat_id or CHAT_ID
    if not TELEGRAM_TOKEN or not chat_id:
        return
    try:
        if photo:
            files = {}
            data = {'chat_id': str(chat_id), 'caption': escape_md_v2(text), 'parse_mode': 'MarkdownV2'}
            if is_document:
                files['document'] = (filename, photo, 'application/pdf')
                requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendDocument", data=data, files=files, timeout=15)
            else:
                files['photo'] = ('chart.png', photo, 'image/png')
                requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto", data=data, files=files, timeout=15)
        else:
            payload = {
                "chat_id": str(chat_id),
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
            df = pd.DataFrame(kl, columns=[
                "open_time","open","high","low","close","volume",
                "close_time","qav","num_trades","tb_base","tb_quote","ignore"
            ])
            df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
            df = df[["open_time","open","high","low","close","volume"]].astype({
                "open": float, "high": float, "low": float, "close": float, "volume": float
            })
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
    if df is None or len(df) < 2:
        return "WATCH", [], False, df.iloc[-1] if len(df) else None, 0.0

    last = df.iloc[-1]
    prev = df.iloc[-2]
    votes = []
    confidence = 0.2

    # EMA
    if last["ema_8"] > last["ema_20"]:
        votes.append("ema_bull")
        confidence += 0.1
    else:
        votes.append("ema_bear")
        confidence += 0.05

    # MACD
    if last["MACD_hist"] > 0:
        votes.append("macd_up")
        confidence += 0.1
    else:
        votes.append("macd_down")
        confidence += 0.05

    # RSI
    if last["RSI_14"] < 30:
        votes.append("rsi_oversold")
        confidence += 0.08
    elif last["RSI_14"] > 70:
        votes.append("rsi_overbought")
        confidence += 0.08

    # ADX
    if last["ADX"] > 25:
        votes.append("strong_trend")
        confidence *= 1.1

    # Свічні патерни
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

    # Pre-top
    pretop = False
    if len(df) >= 10 and (last["close"] - df["close"].iloc[-10]) / df["close"].iloc[-10] > 0.1:
        pretop = True
        confidence += 0.1
        votes.append("pretop")

    # Action
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
        scatter = [None]*(len(df)-1) + [last['close']]
        addplots.append(mpf.make_addplot(scatter, type='scatter', markersize=100, marker='^', color='magenta'))

    patterns = {
        "bullish_engulfing":"green", "bearish_engulfing":"red",
        "hammer_bull":"lime", "shooting_star":"orange", "doji":"blue"
    }
    for pat,color in patterns.items():
        if pat in votes:
            scatter = [None]*(len(df)-1) + [last['close']]
            addplots.append(mpf.make_addplot(scatter,type='scatter',markersize=60,marker='o',color=color))

    entry = last['close'] if action in ["LONG","SHORT"] else None
    stop = last['support'] if action=="LONG" else last['resistance'] if action=="SHORT" else None
    take_levels = []

    if entry and stop:
        atr = last.get("ATR",0.0)
        if action=="LONG":
            take_levels = [entry+atr*0.5, entry+atr*1.0, entry+atr*1.5]
            stop = entry-atr*1.0
        elif action=="SHORT":
            take_levels = [entry-atr*0.5, entry-atr*1.0, entry-atr*1.5]
            stop = entry+atr*1.0

    if entry:
        addplots.append(mpf.make_addplot([entry]*len(df), type='line', color='blue', linewidth=0.5))
        addplots.append(mpf.make_addplot([stop]*len(df), type='line', color='red', linewidth=0.5))
        for tl in take_levels:
            addplots.append(mpf.make_addplot([tl]*len(df), type='line', color='green', linewidth=0.5))

    h_support = df['support'].dropna().iloc[-5:]
    h_resistance = df['resistance'].dropna().iloc[-5:]
    hlines = list(h_support.unique()) + list(h_resistance.unique())

    mc = mpf.make_marketcolors(up='green', down='red', wick='black', edge='black', volume='blue')
    s = mpf.make_mpf_style(marketcolors=mc, gridstyle='--', gridcolor='gray', facecolor='white')

    buf = io.BytesIO()
    mpf.plot(
        df_plot,
        type='candle',
        style=s,
        volume=True,
        addplot=addplots,
        hlines=dict(hlines=hlines, colors=['gray'], linestyle='dashed'),
        title=f"{symbol} — {action}",
        ylabel='Price',
        ylabel_lower='Volume',
        savefig=dict(fname=buf, dpi=100, bbox_inches='tight', facecolor='white'),
        tight_layout=True
    )
    plt.close("all")
    buf.seek(0)
    return buf

# ---------------- PDF HISTORY ----------------
def generate_signal_history_pdf(symbol):
    history = state.get("signal_history", {}).get(symbol, [])
    if not history:
        return None
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial","B",16)
    pdf.cell(0,10,f"Signal History: {symbol}",ln=True,align="C")
    pdf.set_font("Arial","",12)
    for item in history[-50:]:
        pdf.cell(0,8,f"{item['time']} | {item['action']} | Price: {item['price']:.6f} | Conf: {item['confidence']:.2f}",ln=True)
    pdf_bytes = pdf.output(dest="S").encode("latin1", "ignore")
    buf = io.BytesIO(pdf_bytes)
    buf.seek(0)
    return buf

# ---------------- ANALYZE SYMBOL ----------------
def analyze_and_alert(symbol:str, chat_id=None):
    df = fetch_klines(symbol)
    if df is None or len(df) < 30:
        return
    df = apply_all_features(df)
    action, votes, pretop, last, confidence = detect_signal(df)
    if last is None:
        return

    with state_lock:
        prev_signal = state["signals"].get(symbol, "")
        win_stats = state.get("win_stats", {})
        if symbol not in win_stats:
            win_stats[symbol] = {"total":0, "wins":0}
        win_stats[symbol]["total"] += 1
        if action != "WATCH" and confidence >= CONF_THRESHOLD_MEDIUM:
            if pretop or "strong_trend" in votes:
                win_stats[symbol]["wins"] += 1
        state["win_stats"] = win_stats

        if symbol not in state["signal_history"]:
            state["signal_history"][symbol] = []
        state["signal_history"][symbol].append({
            "time": str(datetime.now(timezone.utc)),
            "action": action,
            "price": last['close'],
            "confidence": confidence,
            "votes": votes
        })
        state["signals"][symbol] = action
        save_json_safe(STATE_FILE, state)

    if pretop:
        send_telegram(f"⚡ Pre-top detected for {symbol}, price={last['close']:.6f}", chat_id=chat_id)

    if action != "WATCH" and confidence >= CONF_THRESHOLD_MEDIUM and action != prev_signal:
        msg = (
            f"⚡ TRADE SIGNAL\nSymbol: {symbol}\nAction: {action}\nPrice: {last['close']:.6f}\n"
            f"Support: {last['support']:.6f}\nResistance: {last['resistance']:.6f}\n"
            f"Confidence: {confidence:.2f}\nVotes: {', '.join(votes)}\nPre-top: {pretop}"
        )
        photo_buf = plot_signal_candles(df, symbol, action, votes, pretop)
        send_telegram(msg, photo=photo_buf, chat_id=chat_id)

# ---------------- SCAN ----------------
def scan_top_symbols_safe():
    if not scan_lock.acquire(blocking=False):
        logger.info("Previous scan still running, skipping this run")
        return
    try:
        scan_top_symbols()
    finally:
        scan_lock.release()

def scan_top_symbols():
    symbols = get_all_usdt_symbols()
    if not symbols:
        return
    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as exe:
        list(exe.map(analyze_and_alert, symbols))
    with state_lock:
        state["last_scan"] = str(datetime.now(timezone.utc))
        save_json_safe(STATE_FILE, state)

# ---------------- SCHEDULER ----------------
scheduler = BackgroundScheduler()
scheduler.add_job(scan_top_symbols_safe, "interval", minutes=SCAN_INTERVAL_MINUTES, next_run_time=datetime.now())
scheduler.start()

# ---------------- FLASK ROUTES ----------------
@app.route("/")
def home():
    return jsonify({"status":"ok", "time": str(datetime.now(timezone.utc)), "signals": len(state.get("signals", {}))})

@app.route(f"/telegram_webhook/<token>", methods=["POST","GET"])
def telegram_webhook(token):
    if token != TELEGRAM_TOKEN:
        return jsonify({"ok": False, "reason": "invalid token"}), 403
    if request.method == "POST":
        update = request.get_json(force=True)
        if "message" in update:
            msg = update["message"]
            chat_id = msg["chat"]["id"]
            text = msg.get("text", "").lower()

            if text.startswith("/scan"):
                Thread(target=scan_top_symbols_safe, daemon=True).start()
                send_telegram("Manual scan started.", chat_id=chat_id)

            elif text.startswith("/status"):
                send_telegram(f"Status:\nSignals={len(state.get('signals',{}))}\nLast scan={state.get('last_scan')}", chat_id=chat_id)

            elif text.startswith("/top"):
                win_stats = state.get("win_stats",{})
                if win_stats:
                    top5 = sorted(win_stats.items(), key=lambda x: (x[1]['wins']/x[1]['total'] if x[1]['total']>0 else 0), reverse=True)[:5]
                    lines = [f"{s[0]} — {((s[1]['wins']/s[1]['total'])*100 if s[1]['total']>0 else 0):.1f}%" for s in top5]
                    send_telegram("Top-5 tokens by win rate:\n" + "\n".join(lines), chat_id=chat_id)
                else:
                    send_telegram("No win stats available yet.", chat_id=chat_id)

            elif text.startswith("/profile"):
                parts = text.split()
                if len(parts) >= 2:
                    param = parts[1]
                    value = float(parts[2]) if len(parts) >= 3 else None
                    profile = state["user_profiles"].get(str(chat_id), {"confidence":0.6,"tfs":"15m","take_count":3})
                    if param=="confidence" and value:
                        profile["confidence"] = value
                        state["user_profiles"][str(chat_id)] = profile
                        save_json_safe(STATE_FILE,state)
                        send_telegram(f"Profile updated: {profile}", chat_id=chat_id)

            elif text.startswith("/history"):
                parts = text.split()
                if len(parts) >= 2:
                    symbol = parts[1].upper()
                    pdf_buf = generate_signal_history_pdf(symbol)
                    if pdf_buf:
                        send_telegram(f"📄 Signal history for {symbol}", photo=pdf_buf, chat_id=chat_id, is_document=True, filename=f"{symbol}_history.pdf")
                    else:
                        send_telegram(f"No history found for {symbol}", chat_id=chat_id)

            elif text.startswith("/mlplot"):
                parts = text.split()
                if len(parts) >= 2:
                    symbol = parts[1].upper()
                    history = state.get("signal_history", {}).get(symbol, [])
                    df = fetch_klines(symbol)
                    if df is not None and history:
                        fig, ax = plt.subplots(figsize=(12,6))
                        ax.plot(df.index, df['close'], label='Close')
                        for item in history:
                            t = pd.to_datetime(item['time']).tz_convert("UTC")
                            y = item['price']
                            color = 'green' if item['action']=='LONG' else 'red'
                            ax.scatter(t, y, color=color, s=50)
                        ax.set_title(f"{symbol} — All Signals")
                        ax.set_xlabel("Time")
                        ax.set_ylabel("Price")
                        buf = io.BytesIO()
                        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                        plt.close(fig)
                        buf.seek(0)
                        send_telegram(f"ML plot for {symbol}", photo=buf, chat_id=chat_id)
                    else:
                        send_telegram(f"No data or history for {symbol}", chat_id=chat_id)

    return jsonify({"ok": True})

# ---------------- AUTO REGISTER WEBHOOK ----------------
def auto_register_webhook():
    if WEBHOOK_URL and TELEGRAM_TOKEN:
        set_telegram_webhook(WEBHOOK_URL)

Thread(target=auto_register_webhook, daemon=True).start()

# ---------------- WARMUP ----------------
def warmup_and_first_scan():
    try:
        scan_top_symbols_safe()
    except Exception as e:
        logger.exception("warmup_and_first_scan error: %s", e)

Thread(target=warmup_and_first_scan, daemon=True).start()

# ---------------- MAIN ----------------
if __name__=="__main__":
    app.run(host="0.0.0.0", port=PORT)