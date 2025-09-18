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
import copy

import pandas as pd
import matplotlib.pyplot as plt
import requests
from flask import Flask, request, jsonify
from apscheduler.schedulers.background import BackgroundScheduler
import ta
import mplfinance as mpf
from fpdf import FPDF  # pip install fpdf

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
CHAT_ID = os.getenv("CHAT_ID", "")  # основний чат (опціонально)
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
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except json.JSONDecodeError:
        logger.warning("Corrupted JSON file %s, resetting...", path)
    except Exception as e:
        logger.exception("load_json_safe error %s: %s", path, e)
    return default

def save_json_safe(path, data):
    """Безпечне записування state: робимо snapshot під lock, зберігаємо в tmp, потім os.replace."""
    try:
        # snapshot в JSON-підтримуваний формат (щоб уникнути помилок при паралельних змінах)
        with state_lock:
            snapshot = json.loads(json.dumps(data, default=str, ensure_ascii=False))
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    except Exception as e:
        logger.exception("save_json_safe error %s: %s", path, e)

# init state
state = load_json_safe(STATE_FILE, {
    "signals": {},
    "last_scan": None,
    "signal_history": {},
    "win_stats": {},
    "user_profiles": {}  # chat_id -> {confidence: 0.6, take_count: 3, ...}
})

# ---------------- TELEGRAM ----------------
MARKDOWNV2_ESCAPE = r"_*[]()~`>#+-=|{}.!"

def escape_md_v2(text: str) -> str:
    return re.sub(f"([{re.escape(MARKDOWNV2_ESCAPE)}])", r"\\\1", str(text))

def send_telegram(text: str, photo=None, chat_id=None):
    """Відправка в Telegram. photo = BytesIO (seek(0) перед відправкою)."""
    chat_id = chat_id or CHAT_ID
    if not TELEGRAM_TOKEN or not chat_id:
        return
    try:
        if photo:
            # ensure pointer at start
            try:
                photo.seek(0)
            except Exception:
                pass
            files = {'photo': ('chart.png', photo, 'image/png')}
            data = {'chat_id': str(chat_id), 'caption': escape_md_v2(text), 'parse_mode': 'MarkdownV2'}
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
            df = pd.DataFrame(kl, columns=["open_time","open","high","low","close","volume",
                                           "close_time","qav","num_trades","tb_base","tb_quote","ignore"])
            # convert types
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
        # ATR для динамічного тейку/стопу
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

    # Fake breakout
    if last["close"] > last["resistance"] * 0.995 and last["close"] < last["resistance"] * 1.01:
        votes.append("fake_breakout_short")
        confidence += 0.15
    elif last["close"] < last["support"] * 1.005 and last["close"] > last["support"] * 0.99:
        votes.append("fake_breakout_long")
        confidence += 0.15

    # Pre-top
    pretop = False
    if len(df) >= 10:
        recent, last10 = df["close"].iloc[-1], df["close"].iloc[-10]
        if (recent - last10) / last10 > 0.1:
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

# ---------------- PLOT SIGNAL (старий) ----------------
def plot_signal(df, symbol, action, votes, pretop):
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(df.index, df["close"], label="Close", color="blue")
    ax.plot(df.index, df["ema_8"], label="EMA8", color="green")
    ax.plot(df.index, df["ema_20"], label="EMA20", color="orange")
    ax.fill_between(df.index, df["support"], df["resistance"], color='grey', alpha=0.2)
    if pretop:
        ax.scatter(df.index[-1], df["close"].iloc[-1], color="red", s=80, marker="^", label="Pre-top")
    ax.set_title(f"{symbol} — {action} — {','.join([v for v in votes])}")
    ax.set_ylabel("Price")
    ax.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf

# ---------------- НОВИЙ plot_signal_candles (оновлено) ----------------
def _last_unique_levels(series, n=5):
    """Повертає до n останніх унікальних рівнів (збереження порядку з кінця)."""
    vals = list(series.dropna().tolist())
    seen = set()
    res = []
    for v in reversed(vals):
        if v not in seen:
            seen.add(v)
            res.append(v)
            if len(res) >= n:
                break
    return list(reversed(res))

def plot_signal_candles(df, symbol, action, votes, pretop):
    df_plot = df[['open','high','low','close','volume']].copy()
    df_plot.index.name = "Date"
    addplots = []
    last = df.iloc[-1]

    # Pre-top: останні 3 свічки
    if pretop:
        pts = [df['close'].iloc[-3], df['close'].iloc[-2], df['close'].iloc[-1]]
        addplots.append(mpf.make_addplot(pts, type='scatter', markersize=80, marker='^', color='magenta'))

    # Патерни (позначка у районі останньої ціни)
    patterns = {
        "bullish_engulfing": "green",
        "bearish_engulfing": "red",
        "hammer_bull": "lime",
        "shooting_star": "orange",
        "doji": "blue",
        "fake_breakout_short": "darkred",
        "fake_breakout_long": "darkgreen"
    }
    for pat, color in patterns.items():
        if pat in votes:
            addplots.append(mpf.make_addplot([last['close']]*len(df), type='scatter', markersize=40, marker='o', color=color))

    # Динамічні тейки/стопи (10 рівнів) — використовуємо ATR
    entry = last['close'] if action in ["LONG","SHORT"] else None
    atr = float(last.get("ATR") or 0.0)
    take_levels = []
    stop = None
    if entry and atr > 0:
        # multipliers: робимо більш реалістичні — дальніші і нерівномірні
        mults = [0.5, 1.0, 1.5, 2.0, 2.5, 3.5, 5.0, 7.0, 9.0, 12.0]  # 10 рівнів
        if action == "LONG":
            take_levels = [entry + atr * m for m in mults]
            stop = entry - atr * 1.5
        else:  # SHORT
            take_levels = [entry - atr * m for m in mults]
            stop = entry + atr * 1.5

    # Додаємо точки entry/stop/теїків як addplot (тонкі маркери)
    if entry:
        addplots.append(mpf.make_addplot([entry]*len(df), type='scatter', markersize=28, marker='v', color='blue'))
    if stop:
        addplots.append(mpf.make_addplot([stop]*len(df), type='scatter', markersize=24, marker='x', color='red'))
    for tl in take_levels:
        # дрібніші маркери і трохи прозоріше (якщо потрібна прозорість — можна задати rgba)
        addplots.append(mpf.make_addplot([tl]*len(df), type='scatter', markersize=12, marker='^', color='green'))

    # Основні рівні support/resistance — беремо лише останні 3 унікальні кожного типу
    h_support = _last_unique_levels(df['support'], n=3)
    h_resistance = _last_unique_levels(df['resistance'], n=3)
    hlines = list(h_support) + list(h_resistance)

    # Ринок кольори — тонше відображення (менші linewidths через rc)
    mc = mpf.make_marketcolors(up='green', down='red', wick='black', edge='black', volume='blue')
    s = mpf.make_mpf_style(marketcolors=mc, gridstyle='--', gridcolor='lightgray',
                           facecolor='white', rc={'axes.linewidth': 0.5, 'lines.linewidth': 0.6})

    buf = io.BytesIO()
    mpf.plot(
        df_plot,
        type='candle',
        style=s,
        volume=True,
        addplot=addplots,
        hlines=dict(hlines=hlines, colors=['gray'], linestyle='dashed', linewidths=0.7),
        title=f"{symbol} — {action}",
        ylabel='Price',
        ylabel_lower='Volume',
        savefig=dict(fname=buf, dpi=120, bbox_inches='tight')
    )
    buf.seek(0)
    return buf

# ---------------- ANALYZE SYMBOL ----------------
def analyze_and_alert(symbol: str, chat_id=None):
    df = fetch_klines(symbol)
    if df is None or len(df) < 30:
        return
    df = apply_all_features(df)
    action, votes, pretop, last, confidence = detect_signal(df)

    with state_lock:
        prev_signal = state["signals"].get(symbol, "")

    logger.info("Symbol=%s action=%s confidence=%.2f votes=%s pretop=%s",
                symbol, action, confidence, [v for v in votes], pretop)

    # Оновлення статистики під lock
    with state_lock:
        win_stats = state.get("win_stats", {})
        if symbol not in win_stats:
            win_stats[symbol] = {"total": 0, "wins": 0}
        win_stats[symbol]["total"] += 1
        if action != "WATCH" and confidence >= CONF_THRESHOLD_MEDIUM:
            if pretop or "strong_trend" in votes:
                win_stats[symbol]["wins"] += 1
        state["win_stats"] = win_stats

        # Історія сигналів
        if symbol not in state["signal_history"]:
            state["signal_history"][symbol] = []
        state["signal_history"][symbol].append({
            "time": str(datetime.now(timezone.utc)),
            "action": action,
            "price": float(last['close']),
            "confidence": float(confidence)
        })

    # Зберігаємо snapshot
    save_json_safe(STATE_FILE, state)

    # Якщо pretop — повідомляємо (без фото)
    if pretop:
        # відправляємо тим, хто підписаний або в загальний чат
        recipients = []
        with state_lock:
            for cid, prof in state.get("user_profiles", {}).items():
                try:
                    if float(prof.get("confidence", 0.6)) <= confidence:
                        recipients.append(int(cid))
                except Exception:
                    pass
        # головний чат
        if CHAT_ID:
            recipients.append(CHAT_ID)
        # якщо виклик ручний через chat_id — надсилаємо туди
        if chat_id:
            recipients = [chat_id]

        for rc in set(recipients):
            send_telegram(f"⚡ Pre-top detected for {symbol}, price={last['close']:.6f}", chat_id=rc)

    # Повідомлення про торгову ідею (з графіком) — надсилаємо тільки коли є зміна сигналу і довіра >= поріг
    if action != "WATCH" and confidence >= CONF_THRESHOLD_MEDIUM and action != prev_signal:
        msg = (
            f"⚡ TRADE SIGNAL\n"
            f"Symbol: {symbol}\n"
            f"Action: {action}\n"
            f"Price: {last['close']:.6f}\n"
            f"Support: {last['support']:.6f}\n"
            f"Resistance: {last['resistance']:.6f}\n"
            f"Confidence: {confidence:.2f}\n"
        )
        photo_buf = plot_signal_candles(df, symbol, action, votes, pretop)

        # Відправляємо в основний чат + кожному користувачу, у кого профіль дозволяє
        recipients = set()
        if CHAT_ID:
            recipients.add(int(CHAT_ID))
        with state_lock:
            for cid, prof in state.get("user_profiles", {}).items():
                try:
                    if float(prof.get("confidence", 0.6)) <= confidence:
                        recipients.add(int(cid))
                except Exception:
                    pass
        # Якщо виклик був ручний — відправляємо тільки туди
        if chat_id:
            recipients = {chat_id}

        for rc in recipients:
            # повинен використовувати окремий буфер (BytesIO) для кожного відправлення або rewind
            try:
                photo_buf.seek(0)
            except Exception:
                pass
            send_telegram(msg, photo=photo_buf, chat_id=rc)

        # запис нового активного сигналу
        with state_lock:
            state["signals"][symbol] = action
        save_json_safe(STATE_FILE, state)

# ---------------- SCAN HELPERS ----------------
def scan_top_symbols(manual_chat_id=None):
    symbols = get_all_usdt_symbols()
    if not symbols:
        logger.warning("No symbols found for scanning.")
        return

    logger.info("Starting scan for %d symbols", len(symbols))
    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as exe:
        # передаємо manual_chat_id лише першому виклику (не потрібно для кожного символу)
        list(exe.map(lambda s: analyze_and_alert(s, chat_id=manual_chat_id), symbols))

    with state_lock:
        state["last_scan"] = str(datetime.now(timezone.utc))
    save_json_safe(STATE_FILE, state)
    logger.info("Scan finished at %s", state["last_scan"])

# ---------------- SMART SCAN JOB ----------------
# Scheduler створюємо раніше, щоб джоб мав доступ до глобальної змінної
scheduler = BackgroundScheduler()

def smart_scan_job():
    try:
        symbols = get_all_usdt_symbols()
        if not symbols:
            logger.warning("No symbols found for scanning.")
            return

        logger.info("Starting smart scan for %d symbols", len(symbols))
        try:
            with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as exe:
                list(exe.map(analyze_and_alert, symbols))
        except Exception as e:
            logger.exception("ThreadPoolExecutor error: %s", e)

        with state_lock:
            state["last_scan"] = str(datetime.now(timezone.utc))
        save_json_safe(STATE_FILE, state)
        logger.info("Smart scan finished at %s", state["last_scan"])

        # адаптація частоти сканів: просте правило на основі останніх confidence у історії
        strong_count = 0
        with state_lock:
            for v in state.get("signal_history", {}).values():
                if v and v[-1].get("confidence", 0) >= 0.7:
                    strong_count += 1
        if strong_count > 3:
            new_interval = max(0.5, SCAN_INTERVAL_MINUTES / 2.0)  # наприклад, 30 сек / або половина
        elif strong_count == 0:
            new_interval = min(max(1, SCAN_INTERVAL_MINUTES * 2), 10)  # рідше
        else:
            new_interval = SCAN_INTERVAL_MINUTES

        # reschedule job if змінився інтервал (в секундах — APScheduler вимагає хвилини коли trigger='interval' з minutes)
        try:
            job = scheduler.get_job("scan")
            if job:
                # якщо new_interval відрізняється від поточного
                cur_minutes = job.trigger.interval.total_seconds() / 60.0
                if abs(cur_minutes - new_interval) > 0.1:
                    scheduler.reschedule_job("scan", trigger="interval", minutes=new_interval)
                    logger.info("Rescheduled scan interval to %s min", new_interval)
        except Exception as e:
            logger.debug("Could not reschedule scan job: %s", e)

    except Exception as e:
        logger.exception("smart_scan_job error: %s", e)

# ---------------- FLASK / TELEGRAM WEBHOOK ----------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "running", "last_scan": state.get("last_scan"), "webhook": f"/telegram_webhook/<TOKEN>"})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "last_scan": state.get("last_scan")})

@app.route("/telegram_webhook/<token>", methods=["POST", "GET"])
def telegram_webhook(token):
    # підтримуємо перевірку токену в URL
    if token != TELEGRAM_TOKEN:
        return jsonify({"ok": False, "reason": "invalid token"}), 403

    if request.method == "GET":
        return jsonify({"ok": True})

    try:
        update = request.get_json(force=True, silent=True)
        if not update:
            return jsonify({"ok": False, "reason": "empty"}), 400

        # логування (коротко)
        logger.debug("Incoming update: %s", update)

        # обробка повідомлення
        if "message" in update:
            msg = update["message"]
            chat_id = msg["chat"]["id"]
            text = msg.get("text", "").strip().lower()

            # прості команди
            if text.startswith("/scan"):
                # ручний запуск — запускаємо окремим потоком, щоб швидко відповісти webhook
                Thread(target=scan_top_symbols, args=(chat_id,), daemon=True).start()
                send_telegram("Manual scan started.", chat_id=chat_id)
            elif text.startswith("/status"):
                with state_lock:
                    send_telegram(f"Status:\nSignals={len(state.get('signals', {}))}\nLast scan={state.get('last_scan')}", chat_id=chat_id)
            elif text.startswith("/top"):
                with state_lock:
                    win_stats = state.get("win_stats", {})
                    if win_stats:
                        top5 = sorted(win_stats.items(), key=lambda x: (x[1]['wins'] / x[1]['total'] if x[1]['total'] > 0 else 0), reverse=True)[:5]
                        lines = [f"{s[0]} — {((s[1]['wins']/s[1]['total'])*100 if s[1]['total']>0 else 0):.1f}%" for s in top5]
                        send_telegram("Top-5 tokens by win rate:\n" + "\n".join(lines), chat_id=chat_id)
                    else:
                        send_telegram("No win stats available yet.", chat_id=chat_id)
            elif text.startswith("/signals"):
                with state_lock:
                    sigs = state.get("signals", {})
                msgout = "📊 Active signals:\n" + "\n".join(f"{s}: {a}" for s,a in sigs.items() if a != "WATCH")
                send_telegram(msgout or "No signals", chat_id=chat_id)
            elif text.startswith("/profile"):
                # формат: /profile confidence 0.7
                parts = text.split()
                profile = state["user_profiles"].get(str(chat_id), {"confidence": 0.6, "take_count": 3})
                if len(parts) >= 3:
                    param = parts[1]
                    value = parts[2]
                    try:
                        if param == "confidence":
                            profile["confidence"] = float(value)
                        elif param == "take_count":
                            profile["take_count"] = int(value)
                        state["user_profiles"][str(chat_id)] = profile
                        save_json_safe(STATE_FILE, state)
                        send_telegram(f"Profile updated: {profile}", chat_id=chat_id)
                    except Exception:
                        send_telegram("Invalid profile command or value", chat_id=chat_id)
                else:
                    send_telegram(f"Profile: {profile}", chat_id=chat_id)
            elif text.startswith("/history"):
                # Генеруємо PDF з історією (невелика вибірка)
                buf = io.BytesIO()
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                pdf.cell(0, 10, "Signal History", ln=True, align="C")
                with state_lock:
                    hist = state.get("signal_history", {})
                for sym, entries in hist.items():
                    pdf.cell(0, 8, sym, ln=True)
                    for sig in entries[-10:]:
                        line = f"{sig.get('time')} {sig.get('action')} price={sig.get('price'):.6f} conf={sig.get('confidence'):.2f}"
                        pdf.multi_cell(0, 7, line)
                pdf_bytes = pdf.output(dest='S').encode('latin-1')
                buf.write(pdf_bytes)
                buf.seek(0)
                send_telegram("📑 Signal history PDF", photo=buf, chat_id=chat_id)
            else:
                send_telegram("Received: " + text, chat_id=chat_id)

        return jsonify({"ok": True})
    except Exception as e:
        logger.exception("telegram_webhook error: %s", e)
        return jsonify({"ok": False, "error": str(e)}), 500

# ---------------- SCHEDULER START ----------------
# реєструємо job зі стандартним інтервалом; в smart_scan_job ми можемо reschedule його
scheduler.add_job(smart_scan_job, "interval", minutes=SCAN_INTERVAL_MINUTES, id="scan", next_run_time=datetime.now())
scheduler.start()
logger.info("Scheduler started with interval %s minutes", SCAN_INTERVAL_MINUTES)

# запуск першого скану одразу у фоні, щоб не чекати інтервалу
Thread(target=smart_scan_job, daemon=True).start()

# ---------------- MAIN ----------------
if __name__ == "__main__":
    # встановити webhook (якщо задано)
    if WEBHOOK_URL:
        set_telegram_webhook(WEBHOOK_URL)
    logger.info("Starting pre-top detector bot (dev mode) on port %d", PORT)
    app.run(host="0.0.0.0", port=PORT)