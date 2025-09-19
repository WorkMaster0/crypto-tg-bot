# main.py — Pre-top бот з TA-Lib патернами, fake-breakout фільтром, Telegram, Flask/webhook
import os
import time
import json
import logging
import re
from datetime import datetime, timezone
from threading import Lock
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

# ---------------- CONFIG (змінити у середовищі) ----------------
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
CHAT_ID = os.getenv("CHAT_ID", "")  # дефолтний чат
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "")  # https://your-domain/telegram_webhook/<token>
PORT = int(os.getenv("PORT", "5000"))

SCAN_INTERVAL_MINUTES = int(os.getenv("SCAN_INTERVAL_MINUTES", "1"))
EMA_SCAN_LIMIT = int(os.getenv("EMA_SCAN_LIMIT", "500"))
PARALLEL_WORKERS = int(os.getenv("PARALLEL_WORKERS", "6"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "50"))

STATE_DIR = "data"
os.makedirs(STATE_DIR, exist_ok=True)
STATE_FILE = os.path.join(STATE_DIR, "state.json")
LOG_FILE = os.path.join(STATE_DIR, "bot.log")

# confidence thresholds
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
    logger.warning("Binance client unavailable or API keys missing — market data will be limited or empty")

# ---------------- FLASK ----------------
app = Flask(__name__)

# ---------------- STATE + LOCK ----------------
state_lock = Lock()
scan_lock = Lock()  # додатковий лок, щоб не виконувати повністю паралельні важкі scan-і одночасно
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
    "user_profiles": {},
})

# ---------------- TELEGRAM ----------------
MARKDOWNV2_ESCAPE = r"_*[]()~`>#+=|{}.!"

def escape_md_v2(text: str) -> str:
    return re.sub(f"([{re.escape(MARKDOWNV2_ESCAPE)}])", r"\\\1", str(text))

def send_telegram(text: str, photo=None, chat_id=None, is_document=False, filename="file.pdf"):
    chat_id = chat_id or CHAT_ID
    if not TELEGRAM_TOKEN or not chat_id:
        logger.debug("Telegram token or chat_id missing, skipping send")
        return
    try:
        if photo:
            files = {}
            data = {'chat_id': str(chat_id), 'caption': escape_md_v2(text), 'parse_mode': 'MarkdownV2'}
            if is_document:
                files['document'] = (filename, photo, 'application/pdf')
                resp = requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendDocument", data=data, files=files, timeout=20)
            else:
                files['photo'] = ('chart.png', photo, 'image/png')
                resp = requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto", data=data, files=files, timeout=20)
            if resp is not None and resp.status_code != 200:
                logger.warning("Telegram send media failed: %s", resp.text)
        else:
            payload = {
                "chat_id": str(chat_id),
                "text": escape_md_v2(text),
                "parse_mode": "MarkdownV2",
                "disable_web_page_preview": True
            }
            resp = requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", json=payload, timeout=10)
            if resp is not None and resp.status_code != 200:
                logger.warning("Telegram send message failed: %s", resp.text)
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
        logger.warning("Binance client not available — get_all_usdt_symbols returns empty list")
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
            # attach symbol for convenience
            df["symbol"] = symbol
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

# ---------------- CANDLE PATTERNS (TA-Lib) ----------------
# налаштуй ваги та сигнал для кожного патерну
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

def apply_candle_patterns(df: pd.DataFrame) -> pd.DataFrame:
    if not TALIB_AVAILABLE:
        logger.warning("TA-Lib not available — skipping TA-Lib patterns")
        # fallback: keep existing manual detection for a couple of simple patterns (hammer/shooting)
        df = df.copy()
        # manual hammer/shooting-star heuristics as fallback
        last_body = df["close"] - df["open"]
        rng = df["high"] - df["low"]
        upper_shadow = df["high"] - df[["close","open"]].max(axis=1)
        lower_shadow = df[["close","open"]].min(axis=1) - df["low"]
        df["FALLBACK_HAMMER"] = ((lower_shadow > 2 * last_body.abs()) & (last_body > 0)).astype(int)
        df["FALLBACK_SHOOTINGSTAR"] = ((upper_shadow > 2 * last_body.abs()) & (last_body < 0)).astype(int)
        return df

    open_, high_, low_, close_ = df["open"], df["high"], df["low"], df["close"]
    df = df.copy()
    for name in CANDLE_PATTERNS.keys():
        func = getattr(talib, name, None)
        if func:
            try:
                df[name] = func(open_, high_, low_, close_)
            except Exception as e:
                logger.exception("talib %s error: %s", name, e)
                df[name] = 0
        else:
            df[name] = 0
    return df

# ---------------- FAKE BREAKOUT DETECTOR ----------------
def detect_fake_breakout(df: pd.DataFrame) -> bool:
    """
    Fake breakout:
    - На попередній свічці було пробиття (close або high вище resistance або нижче support),
      а на останній свічці ціна повернулася назад (закрилась всередині діапазону).
    Повертає True якщо є ознаки fake breakout.
    """
    try:
        if df is None or len(df) < 3:
            return False
        last = df.iloc[-1]
        prev = df.iloc[-2]

        # беремо рівні support/resistance з попередніх даних (щоб не дивитись на "оновлений" рівень)
        # якщо немає — не вважаємо
        supports = df['support'].dropna()
        resistances = df['resistance'].dropna()
        if len(supports) < 2 or len(resistances) < 2:
            return False
        s_level = supports.iloc[-2]
        r_level = resistances.iloc[-2]

        # Параметри толерантності — можна налаштувати
        UP_BREAK_TOL = 1.002  # >0.2% above level considered breakout
        BACK_IN_TOL = 0.999   # back inside if < level * 0.999

        DOWN_BREAK_TOL = 0.998
        BACK_UP_TOL = 1.001

        # fake breakout вгору: prev пробив вверх, last закрився всередині (повернувся під resistance)
        prev_broke_up = (prev['close'] > r_level * UP_BREAK_TOL) or (prev['high'] > r_level * UP_BREAK_TOL)
        last_back_below = last['close'] < r_level * BACK_IN_TOL

        # fake breakout вниз: prev пробив вниз, last закрився всередині (повернувся вище support)
        prev_broke_down = (prev['close'] < s_level * DOWN_BREAK_TOL) or (prev['low'] < s_level * DOWN_BREAK_TOL)
        last_back_above = last['close'] > s_level * BACK_UP_TOL

        if prev_broke_up and last_back_below:
            return True
        if prev_broke_down and last_back_above:
            return True
    except Exception as e:
        logger.exception("detect_fake_breakout error: %s", e)
    return False

# ---------------- SIGNAL DETECTION ----------------
def detect_signal(df: pd.DataFrame, symbol: str):
    """
    Повертає: action (LONG/SHORT/WATCH), votes(list), pretop(bool), last_row(Series), confidence(float)
    """
    if df is None or len(df) < 5:
        return "WATCH", [], False, None, 0.0

    df = df.copy()
    last = df.iloc[-1]
    prev = df.iloc[-2]
    votes = []
    confidence = 0.2  # базова впевненість

    # EMA
    try:
        if last["ema_8"] > last["ema_20"]:
            votes.append("ema_bull")
            confidence += 0.08
        else:
            votes.append("ema_bear")
            confidence += 0.03
    except Exception:
        pass

    # MACD
    try:
        if last.get("MACD_hist", 0) > 0:
            votes.append("macd_up")
            confidence += 0.08
        else:
            votes.append("macd_down")
            confidence += 0.03
    except Exception:
        pass

    # RSI
    try:
        if last.get("RSI_14", 50) < 30:
            votes.append("rsi_oversold")
            confidence += 0.06
        elif last.get("RSI_14", 50) > 70:
            votes.append("rsi_overbought")
            confidence += 0.06
    except Exception:
        pass

    # ADX
    try:
        if last.get("ADX", 0) > 25:
            votes.append("strong_trend")
            confidence *= 1.05
    except Exception:
        pass

    # TA-Lib candle patterns (використовує CANDLE_PATTERNS словник)
    try:
        for name, cfg in CANDLE_PATTERNS.items():
            val = int(last.get(name, 0)) if name in last.index else 0
            if val == 0:
                continue
            # val > 0 => bullish pattern, val < 0 => bearish (залежно від TA-Lib)
            if cfg["signal"] == "LONG":
                if val > 0:
                    votes.append(name.lower())
                    confidence += cfg["weight"]
            elif cfg["signal"] == "SHORT":
                if val < 0:
                    votes.append(name.lower())
                    confidence += cfg["weight"]
            elif cfg["signal"] == "BOTH":
                if val > 0:
                    votes.append(name.lower() + "_bull")
                    confidence += cfg["weight"]
                elif val < 0:
                    votes.append(name.lower() + "_bear")
                    confidence += cfg["weight"]
            else:  # NEUTRAL
                votes.append(name.lower())
                confidence += cfg["weight"]
    except Exception as e:
        logger.exception("candle pattern processing error: %s", e)

    # Свічні хендлери fallback (якщо TA-Lib не доступен)
    if not TALIB_AVAILABLE:
        try:
            fb_hammer = int(last.get("FALLBACK_HAMMER", 0))
            fb_shoot = int(last.get("FALLBACK_SHOOTINGSTAR", 0))
            if fb_hammer:
                votes.append("fallback_hammer")
                confidence += 0.10
            if fb_shoot:
                votes.append("fallback_shootingstar")
                confidence += 0.10
        except Exception:
            pass

    # Pre-top (проста логіка)
    pretop = False
    try:
        if len(df) >= 10 and (last["close"] - df["close"].iloc[-10]) / df["close"].iloc[-10] > 0.10:
            pretop = True
            votes.append("pretop")
            confidence += 0.08
    except Exception:
        pass

    # Action by price vs support/resistance
    action = "WATCH"
    try:
        if pd.notna(last.get("resistance")) and last["close"] >= last["resistance"] * 0.995:
            action = "SHORT"
        elif pd.notna(last.get("support")) and last["close"] <= last["support"] * 1.005:
            action = "LONG"
    except Exception:
        pass

    # Якщо action визначено — застосувати fake breakout фільтр: сигнал лише при fake breakout
    if action != "WATCH":
        fb = detect_fake_breakout(df)
        if not fb:
            # немає фейку — ігноруємо
            logger.info(f"{symbol}: potential {action} ignored — no fake breakout detected")
            action = "WATCH"
        else:
            votes.append("fake_breakout")
            confidence += 0.08

    # Нормалізація confidence
    try:
        confidence = max(0.0, min(1.0, confidence))
    except Exception:
        confidence = 0.0

    return action, votes, pretop, last, confidence

# ---------------- PLOT SIGNAL CANDLES ----------------
def plot_signal_candles(df, symbol, action, votes, pretop=False):
    df_plot = df[['open','high','low','close','volume']].copy()
    df_plot.index.name = "Date"
    addplots = []
    last = df.iloc[-1]

    # mark patterns
    markers = []
    for v in votes:
        if "hammer" in v:
            markers.append(("^", last['close'], 90))
        elif "shooting" in v or "star" in v:
            markers.append(("v", last['close'], 60))
        elif "engulf" in v:
            markers.append(("o", last['close'], 70))
        elif "fake_breakout" in v:
            markers.append(("D", last['close'], 80))

    for marker, price, size in markers:
        scatter = [None]*(len(df)-1) + [price]
        addplots.append(mpf.make_addplot(scatter, type='scatter', markersize=size//10, marker=marker, color='magenta'))

    entry = last['close'] if action in ["LONG","SHORT"] else None
    stop = last['support'] if action=="LONG" else last['resistance'] if action=="SHORT" else None
    take_levels = []

    if entry and stop:
        atr = last.get("ATR", 0.0) or 0.0
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

    # horizontal lines from last few support/resistance
    h_support = df['support'].dropna().iloc[-5:] if 'support' in df else []
    h_resistance = df['resistance'].dropna().iloc[-5:] if 'resistance' in df else []
    hlines = []
    if hasattr(h_support, "unique"):
        hlines += list(h_support.unique())
    if hasattr(h_resistance, "unique"):
        hlines += list(h_resistance.unique())

    mc = mpf.make_marketcolors(up='green', down='red', wick='black', edge='black', volume='blue')
    s = mpf.make_mpf_style(marketcolors=mc, gridstyle='--', gridcolor='gray', facecolor='white')

    buf = io.BytesIO()
    try:
        mpf.plot(
            df_plot,
            type='candle',
            style=s,
            volume=True,
            addplot=addplots,
            hlines=dict(hlines=hlines, colors=['gray'], linestyle='dashed') if hlines else None,
            title=f"{symbol} — {action}",
            ylabel='Price',
            ylabel_lower='Volume',
            savefig=dict(fname=buf, dpi=100, bbox_inches='tight', facecolor='white'),
            tight_layout=True
        )
    except Exception as e:
        logger.exception("mpf.plot error: %s", e)
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
    for item in history[-100:]:
        time_str = item.get('time', '')
        action = item.get('action', '')
        price = item.get('price', 0.0)
        conf = item.get('confidence', 0.0)
        votes = ",".join(item.get('votes', []))
        pdf.multi_cell(0,8,f"{time_str} | {action} | Price: {price:.8f} | Conf: {conf:.2f} | {votes}")
    pdf_bytes = pdf.output(dest="S").encode("latin1", "ignore")
    buf = io.BytesIO(pdf_bytes)
    buf.seek(0)
    return buf

# ---------------- ANALYZE SYMBOL ----------------
def analyze_and_alert(symbol: str, chat_id=None):
    df = fetch_klines(symbol)
    if df is None or len(df) < 30:
        logger.debug(f"{symbol}: not enough data or fetch failed")
        return
    df = apply_all_features(df)
    df = apply_candle_patterns(df)
    action, votes, pretop, last, confidence = detect_signal(df, symbol)
    if last is None:
        return

    with state_lock:
        prev_signal = state["signals"].get(symbol, "")
        win_stats = state.get("win_stats", {})
        if symbol not in win_stats:
            win_stats[symbol] = {"total":0, "wins":0}
        win_stats[symbol]["total"] += 1
        if action != "WATCH" and confidence >= CONF_THRESHOLD_MEDIUM:
            if pretop or "strong_trend" in votes or "fake_breakout" in votes:
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

    logger.info(f"{symbol}: action={action}, confidence={confidence:.2f}, votes={votes}")

    # Pretop notification (можна завжди надсилати)
    if pretop:
        send_telegram(f"⚡ Pre-top detected for {symbol}, price={last['close']:.6f}", chat_id=chat_id)

    # Основний сигнал — лише при зміні або новому сигналі
    if action != "WATCH" and confidence >= CONF_THRESHOLD_MEDIUM and action != prev_signal:
        msg = (
            f"⚡ TRADE SIGNAL\nSymbol: {symbol}\nAction: {action}\nPrice: {last['close']:.6f}\n"
            f"Confidence: {confidence:.2f}\nVotes: {', '.join(votes)}"
        )
        buf = plot_signal_candles(df, symbol, action, votes, pretop)
        send_telegram(msg, photo=buf, chat_id=chat_id)

# ---------------- SCAN ALL SYMBOLS BATCH ----------------
def scan_top_symbols_safe():
    # захищаємося локом, щоб не запускати scan повністю двічі одночасно
    if not scan_lock.acquire(blocking=False):
        logger.info("Previous scan still running, skipping this run (scan_lock)")
        return
    try:
        scan_top_symbols()
    except Exception as e:
        logger.exception("scan_top_symbols_safe error: %s", e)
    finally:
        scan_lock.release()

def scan_top_symbols():
    symbols = get_all_usdt_symbols()
    if not symbols:
        logger.warning("No symbols to scan")
        return
    total = len(symbols)
    logger.info(f"Starting scan for {total} symbols in batches of {BATCH_SIZE}")

    for i in range(0, total, BATCH_SIZE):
        batch = symbols[i:i+BATCH_SIZE]
        workers = min(PARALLEL_WORKERS, len(batch))
        logger.info(f"Processing batch {i//BATCH_SIZE + 1} with {len(batch)} symbols (workers={workers})")
        with ThreadPoolExecutor(max_workers=workers) as exe:
            list(exe.map(analyze_and_alert, batch))
        logger.info(f"Completed batch {i//BATCH_SIZE + 1} / {((total-1)//BATCH_SIZE)+1}")

    with state_lock:
        state["last_scan"] = str(datetime.now(timezone.utc))
        save_json_safe(STATE_FILE, state)
    logger.info("Scan completed for all symbols")

# ---------------- FLASK ROUTES ----------------
@app.route("/")
def home():
    return "Bot is running!"

@app.route(f"/telegram_webhook/<token>", methods=["POST"])
def telegram_webhook(token):
    # швидко відповідаємо 200, обробку робимо в окремому потоці
    data = request.json
    executor.submit(handle_telegram_update, token, data)
    return jsonify({"ok": True})

def handle_telegram_update(token, update):
    try:
        if token != TELEGRAM_TOKEN:
            logger.warning("Received webhook with unexpected token")
            return
        if not update:
            return
        # parse message
        msg = update.get("message") or update.get("edited_message") or {}
        text = msg.get("text", "")
        chat = msg.get("chat", {})
        chat_id = chat.get("id")
        user = msg.get("from", {}).get("username", "")
        logger.info("Telegram update from %s: %s", user, text)

        if not text:
            return

        parts = text.strip().split()
        cmd = parts[0].lower()
        if cmd == "/start":
            send_telegram("Привіт! Я бот сигналів. Доступні команди: /status, /scan, /history <SYMBOL>", chat_id=chat_id)
        elif cmd == "/status":
            with state_lock:
                last_scan = state.get("last_scan")
                active_signals = {k:v for k,v in state.get("signals", {}).items() if v != "WATCH"}
                msg_text = f"Last scan: {last_scan}\nActive signals: {len(active_signals)}"
            send_telegram(msg_text, chat_id=chat_id)
        elif cmd == "/scan":
            # запуск скану в бекграунді
            executor.submit(scan_top_symbols_safe)
            send_telegram("Запуск скану (в бекграунді)...", chat_id=chat_id)
        elif cmd == "/history":
            if len(parts) >= 2:
                symbol = parts[1].upper()
                pdf_buf = generate_signal_history_pdf(symbol)
                if pdf_buf:
                    send_telegram(f"Signal history for {symbol}", photo=pdf_buf, chat_id=chat_id, is_document=True, filename=f"{symbol}_history.pdf")
                else:
                    send_telegram(f"No history for {symbol}", chat_id=chat_id)
            else:
                send_telegram("Usage: /history <SYMBOL>", chat_id=chat_id)
        else:
            send_telegram("Unknown command. Use /status, /scan, /history <SYMBOL>", chat_id=chat_id)
    except Exception as e:
        logger.exception("handle_telegram_update error: %s", e)

@app.route(f"/telegram_webhook/{TELEGRAM_TOKEN}", methods=["POST"])
def telegram_webhook_token():
    # backward-compatible route — просто повертає OK
    return jsonify({"ok": True})

# ---------------- SCHEDULER ----------------
scheduler = BackgroundScheduler()
scheduler.add_job(
    scan_top_symbols_safe,
    "interval",
    minutes=max(1, SCAN_INTERVAL_MINUTES),
    max_instances=2,       # дозволяє до 2 одночасних (щоб уникнути skip warning)
    coalesce=True,         # об'єднати пропущені запуски
    misfire_grace_time=120 # секунд, при невеликій затримці все одно виконає
)
scheduler.start()

# ---------------- WEBHOOK INIT ----------------
if WEBHOOK_URL and TELEGRAM_TOKEN:
    try:
        set_telegram_webhook(WEBHOOK_URL)
    except Exception as e:
        logger.exception("Failed to set webhook: %s", e)

# ---------------- RUN ----------------
if __name__ == "__main__":
    logger.info(f"Starting bot on port {PORT}")
    app.run(host="0.0.0.0", port=PORT)