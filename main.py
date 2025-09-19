# main.py — Повний бот Pre-top з TA-Lib, fake-breakout, Telegram, Flask/webhook
import os
import time
import json
import logging
import re
import requests
from datetime import datetime, timezone
from threading import Lock
from concurrent.futures import ThreadPoolExecutor
import io

import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify
from apscheduler.schedulers.background import BackgroundScheduler
import ta
import mplfinance as mpf
from fpdf import FPDF

# optional libs
try:
    import talib
    TALIB_AVAILABLE = True
except Exception:
    TALIB_AVAILABLE = False

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

SCAN_INTERVAL_MINUTES = int(os.getenv("SCAN_INTERVAL_MINUTES", "1"))
EMA_SCAN_LIMIT = int(os.getenv("EMA_SCAN_LIMIT", "500"))
PARALLEL_WORKERS = int(os.getenv("PARALLEL_WORKERS", "6"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "50"))

STATE_DIR = "data"
os.makedirs(STATE_DIR, exist_ok=True)
STATE_FILE = os.path.join(STATE_DIR, "state.json")
LOG_FILE = os.path.join(STATE_DIR, "bot.log")

CONF_THRESHOLD_MEDIUM = float(os.getenv("CONF_THRESHOLD_MEDIUM", "0.60"))
CONF_THRESHOLD_STRONG = float(os.getenv("CONF_THRESHOLD_STRONG", "0.80"))

# ---------------- LOGGING ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)
logger = logging.getLogger("pretop-bot")

# ---------------- BINANCE CLIENT ----------------
if BINANCE_PY_AVAILABLE and BINANCE_API_KEY and BINANCE_API_SECRET:
    client = BinanceClient(api_key=BINANCE_API_KEY, api_secret=BINANCE_API_SECRET)
else:
    client = None
    logger.warning("Binance client unavailable or API keys missing — market data will be limited")

# ---------------- FLASK ----------------
app = Flask(__name__)

# ---------------- STATE + LOCK ----------------
state_lock = Lock()
scan_lock = Lock()
executor = ThreadPoolExecutor(max_workers=4)

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
})

# ---------------- CACHED USDT SYMBOLS ----------------
USDT_SYMBOLS_CACHE = []

def init_usdt_symbols():
    global USDT_SYMBOLS_CACHE
    if not client:
        logger.warning("Binance client not available — cannot fetch USDT symbols")
        USDT_SYMBOLS_CACHE = []
        return
    try:
        ex = client.get_exchange_info()
        symbols = [s["symbol"] for s in ex["symbols"] if s["quoteAsset"]=="USDT" and s["status"]=="TRADING"]
        blacklist = ["BUSD","USDC","FDUSD","TUSD","DAI","EUR","GBP","AUD","BTCST","COIN","AAPL","TSLA","MSFT","META","GOOG"]
        USDT_SYMBOLS_CACHE = [s for s in symbols if not any(b in s for b in blacklist)]
        logger.info(f"Cached {len(USDT_SYMBOLS_CACHE)} USDT symbols")
    except Exception as e:
        logger.exception("init_usdt_symbols error: %s", e)
        USDT_SYMBOLS_CACHE = []

def get_all_usdt_symbols():
    return USDT_SYMBOLS_CACHE

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
                requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendDocument", data=data, files=files, timeout=20)
            else:
                files['photo'] = ('chart.png', photo, 'image/png')
                requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto", data=data, files=files, timeout=20)
        else:
            payload = {"chat_id": str(chat_id), "text": escape_md_v2(text), "parse_mode": "MarkdownV2"}
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
            df["symbol"] = symbol
            return df
        except Exception as e:
            logger.warning("fetch_klines %s attempt %d error: %s", symbol, attempt+1, e)
            time.sleep(0.5)
    return None

# ---------------- FEATURE ENGINEERING ----------------
def apply_all_features(df):
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

# ---------------- CANDLE PATTERNS ----------------
CANDLE_PATTERNS = {
    "CDLHAMMER": {"signal": "LONG", "weight": 0.10},
    "CDLINVERTEDHAMMER": {"signal": "LONG", "weight": 0.10},
    "CDLENGULFING": {"signal": "BOTH", "weight": 0.15},
    "CDLDOJI": {"signal": "NEUTRAL", "weight": 0.08},
    "CDLMORNINGSTAR": {"signal": "LONG", "weight": 0.12},
    "CDLEVENINGSTAR": {"signal": "SHORT", "weight": 0.12},
    "CDLHARAMI": {"signal": "BOTH", "weight": 0.10},
    "CDLDARKCLOUDCOVER": {"signal": "SHORT", "weight": 0.12},
    "CDL3WHITESOLDIERS": {"signal": "LONG", "weight": 0.15},
    "CDL3BLACKCROWS": {"signal": "SHORT", "weight": 0.15},
    "CDLSHOOTINGSTAR": {"signal": "SHORT", "weight": 0.10}
}

def apply_candle_patterns(df):
    if not TALIB_AVAILABLE:
        df = df.copy()
        last_body = df["close"] - df["open"]
        upper_shadow = df["high"] - df[["close","open"]].max(axis=1)
        lower_shadow = df[["close","open"]].min(axis=1) - df["low"]
        df["FALLBACK_HAMMER"] = ((lower_shadow > 2*last_body.abs()) & (last_body>0)).astype(int)
        df["FALLBACK_SHOOTINGSTAR"] = ((upper_shadow > 2*last_body.abs()) & (last_body<0)).astype(int)
        return df
    open_, high_, low_, close_ = df["open"], df["high"], df["low"], df["close"]
    df = df.copy()
    for name in CANDLE_PATTERNS.keys():
        func = getattr(talib, name, None)
        if func:
            try:
                df[name] = func(open_, high_, low_, close_)
            except:
                df[name] = 0
        else:
            df[name] = 0
    return df

# ---------------- FAKE BREAKOUT ----------------
def detect_fake_breakout(df):
    try:
        if df is None or len(df)<3:
            return False
        last, prev = df.iloc[-1], df.iloc[-2]
        supports = df['support'].dropna()
        resistances = df['resistance'].dropna()
        if len(supports)<2 or len(resistances)<2:
            return False
        s_level, r_level = supports.iloc[-2], resistances.iloc[-2]
        if ((prev['close']>r_level*1.002 or prev['high']>r_level*1.002) and last['close']<r_level*0.999):
            return True
        if ((prev['close']<s_level*0.998 or prev['low']<s_level*0.998) and last['close']>s_level*1.001):
            return True
    except:
        pass
    return False

# ---------------- SIGNAL DETECTION ----------------
def detect_signal(df, symbol):
    if df is None or len(df)<5:
        return "WATCH", [], False, None, 0.0
    df = df.copy()
    last = df.iloc[-1]
    votes = []
    confidence = 0.2
    # EMA
    try:
        if last["ema_8"]>last["ema_20"]:
            votes.append("ema_bull")
            confidence+=0.08
        else:
            votes.append("ema_bear")
            confidence+=0.03
    except:
        pass
    # MACD
    try:
        if last.get("MACD_hist",0)>0:
            votes.append("macd_up")
            confidence+=0.08
        else:
            votes.append("macd_down")
            confidence+=0.03
    except:
        pass
    # RSI
    try:
        if last.get("RSI_14",50)<30:
            votes.append("rsi_oversold")
            confidence+=0.06
        elif last.get("RSI_14",50)>70:
            votes.append("rsi_overbought")
            confidence+=0.06
    except:
        pass
    # ADX
    try:
        if last.get("ADX",0)>25:
            votes.append("strong_trend")
            confidence*=1.05
    except:
        pass
    # candle patterns
    try:
        for name,cfg in CANDLE_PATTERNS.items():
            val = int(last.get(name,0))
            if val==0:
                continue
            if cfg["signal"]=="LONG" and val>0:
                votes.append(name.lower())
                confidence+=cfg["weight"]
            elif cfg["signal"]=="SHORT" and val<0:
                votes.append(name.lower())
                confidence+=cfg["weight"]
            elif cfg["signal"]=="BOTH":
                if val>0:
                    votes.append(name.lower()+"_bull")
                    confidence+=cfg["weight"]
                elif val<0:
                    votes.append(name.lower()+"_bear")
                    confidence+=cfg["weight"]
            else:
                votes.append(name.lower())
                confidence+=cfg["weight"]
    except:
        pass
    # fallback
    if not TALIB_AVAILABLE:
        try:
            fb_hammer = int(last.get("FALLBACK_HAMMER",0))
            fb_shoot = int(last.get("FALLBACK_SHOOTINGSTAR",0))
            if fb_hammer: votes.append("fallback_hammer"); confidence+=0.10
            if fb_shoot: votes.append("fallback_shootingstar"); confidence+=0.10
        except: pass
    # Pretop
    pretop=False
    try:
        if len(df)>=10 and (last["close"]-df["close"].iloc[-10])/df["close"].iloc[-10]>0.10:
            pretop=True
            votes.append("pretop")
            confidence+=0.08
    except: pass
    # action
    action="WATCH"
    try:
        if pd.notna(last.get("resistance")) and last["close"]>=last["resistance"]*0.995:
            action="SHORT"
        elif pd.notna(last.get("support")) and last["close"]<=last["support"]*1.005:
            action="LONG"
    except: pass
    if action!="WATCH":
        fb=detect_fake_breakout(df)
        if not fb:
            action="WATCH"
        else:
            votes.append("fake_breakout")
            confidence+=0.08
    confidence=max(0.0,min(1.0,confidence))
    return action,votes,pretop,last,confidence

# ---------------- PLOT & PDF ----------------
def plot_signal_candles(df,symbol,action,votes,pretop=False):
    df_plot = df[['open','high','low','close','volume']].copy()
    df_plot.index.name="Date"
    addplots=[]
    last=df.iloc[-1]
    markers=[]
    for v in votes:
        if "hammer" in v: markers.append(("^",last['close'],90))
        elif "shooting" in v or "star" in v: markers.append(("v",last['close'],60))
        elif "engulf" in v: markers.append(("o",last['close'],70))
        elif "pretop" in v: markers.append(("s",last['close'],50))
        elif "fake_breakout" in v: markers.append(("*",last['close'],80))
    for m,y,size in markers:
        addplots.append(mpf.make_addplot([y]*len(df_plot),scatter=True,marker=m,markersize=size))
    fig,ax=mpf.plot(df_plot,type='candle',style='charles',volume=True,addplot=addplots,returnfig=True)
    buf=io.BytesIO()
    fig.savefig(buf,format='png',bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf

def generate_signal_history_pdf(symbol):
    pdf=FPDF()
    pdf.set_auto_page_break(auto=True, margin=10)
    history=state.get("signal_history",{}).get(symbol.upper(),[])
    if not history: return None
    for rec in history[-20:]:
        pdf.add_page()
        pdf.set_font("Arial","B",12)
        pdf.cell(0,10,f"Signal {symbol.upper()} at {rec.get('time')}",ln=1)
        pdf.set_font("Arial","",10)
        pdf.multi_cell(0,8,f"Action: {rec.get('action')}\nVotes: {','.join(rec.get('votes',[]))}\nConfidence: {rec.get('confidence')}")
    buf=io.BytesIO()
    pdf.output(buf)
    buf.seek(0)
    return buf

# ---------------- ANALYZE & ALERT ----------------
def analyze_and_alert(symbol):
    df=fetch_klines(symbol)
    if df is None: return
    df=apply_all_features(df)
    df=apply_candle_patterns(df)
    action,votes,pretop,last,confidence=detect_signal(df,symbol)
    record={"time":str(datetime.now(timezone.utc)),"action":action,"votes":votes,"confidence":confidence}
    with state_lock:
        state["signal_history"].setdefault(symbol.upper(),[]).append(record)
        state["signals"][symbol.upper()]=action
        save_json_safe(STATE_FILE,state)
    if action!="WATCH":
        buf=plot_signal_candles(df,symbol,action,votes,pretop)
        send_telegram(f"{symbol} {action} detected. Votes: {','.join(votes)} Conf:{confidence:.2f}",photo=buf)

# ---------------- SCAN ----------------
def scan_top_symbols_safe():
    if not scan_lock.acquire(blocking=False):
        logger.info("Previous scan still running, skipping")
        return
    try:
        scan_top_symbols()
    finally:
        scan_lock.release()

def scan_top_symbols():
    symbols = get_all_usdt_symbols()
    if not symbols:
        logger.warning("No symbols to scan")
        return
    total = len(symbols)
    logger.info(f"Scanning {total} symbols in batches of {BATCH_SIZE}")
    for i in range(0, total, BATCH_SIZE):
        batch = symbols[i:i+BATCH_SIZE]
        with ThreadPoolExecutor(max_workers=min(PARALLEL_WORKERS,len(batch))) as exe:
            list(exe.map(analyze_and_alert, batch))
    with state_lock:
        state["last_scan"] = str(datetime.now(timezone.utc))
        save_json_safe(STATE_FILE, state)
    logger.info("Scan complete")

# ---------------- FLASK ROUTES ----------------
@app.route("/")
def home():
    return "Bot is running!"

@app.route("/telegram_webhook/<token>", methods=["POST"])
def telegram_webhook(token):
    data = request.json
    executor.submit(handle_telegram_update, token, data)
    return jsonify({"ok": True})

def handle_telegram_update(token, update):
    try:
        if token != TELEGRAM_TOKEN:
            return
        msg = update.get("message") or update.get("edited_message") or {}
        text = msg.get("text","")
        chat_id = msg.get("chat",{}).get("id")
        if not text or not chat_id:
            return
        parts = text.strip().split()
        cmd = parts[0].lower()
        if cmd == "/start":
            send_telegram("Привіт! Я бот сигналів. Команди: /status, /scan, /history <SYMBOL>", chat_id)
        elif cmd == "/status":
            with state_lock:
                last_scan = state.get("last_scan")
                active_signals = {k:v for k,v in state.get("signals", {}).items() if v!="WATCH"}
                send_telegram(f"Last scan: {last_scan}\nActive signals: {len(active_signals)}", chat_id)
        elif cmd == "/scan":
            executor.submit(scan_top_symbols_safe)
            send_telegram("Запуск скану...", chat_id)
        elif cmd == "/history":
            if len(parts)>=2:
                symbol = parts[1].upper()
                pdf_buf = generate_signal_history_pdf(symbol)
                if pdf_buf:
                    send_telegram(f"Signal history for {symbol}", photo=pdf_buf, chat_id=chat_id, is_document=True, filename=f"{symbol}_history.pdf")
                else:
                    send_telegram(f"No history for {symbol}", chat_id)
            else:
                send_telegram("Usage: /history <SYMBOL>", chat_id)
        else:
            send_telegram("Unknown command. Use /status, /scan, /history <SYMBOL>", chat_id)
    except Exception as e:
        logger.exception("handle_telegram_update error: %s", e)

# ---------------- SCHEDULER ----------------
scheduler = BackgroundScheduler()
scheduler.add_job(scan_top_symbols_safe, "interval", minutes=max(1,SCAN_INTERVAL_MINUTES), max_instances=2, coalesce=True, misfire_grace_time=120)
scheduler.start()

# ---------------- INIT ----------------
init_usdt_symbols()
if WEBHOOK_URL and TELEGRAM_TOKEN:
    set_telegram_webhook(f"{WEBHOOK_URL}/{TELEGRAM_TOKEN}")

# ---------------- RUN ----------------
if __name__ == "__main__":
    logger.info(f"Starting bot on port {PORT}")
    app.run(host="0.0.0.0", port=PORT)