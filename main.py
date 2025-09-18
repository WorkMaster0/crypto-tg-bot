# main.py — Pre-top бот з графіками, Telegram, профілями користувачів, історією сигналів та backtest
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
    """Безпечне записування state: snapshot під lock, зберігаємо в tmp, потім os.replace."""
    try:
        with state_lock:
            snapshot = json.loads(json.dumps(data, default=str, ensure_ascii=False))
        dir_path = os.path.dirname(os.path.abspath(path))
        os.makedirs(dir_path, exist_ok=True)
        tmp = os.path.join(dir_path, os.path.basename(path) + ".tmp")

        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2, ensure_ascii=False)

        if not os.path.exists(tmp):
            logger.error("Temp file %s was not created!", tmp)
            return

        os.replace(tmp, path)
    except Exception as e:
        logger.exception("save_json_safe error %s: %s", path, e)

# init state
state = load_json_safe(STATE_FILE, {
    "signals": {},
    "last_scan": None,
    "signal_history": {},
    "win_stats": {},
    "user_profiles": {},
    "backtest_stats": {},
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

    if last["close"] > last["resistance"] * 0.995 and last["close"] < last["resistance"] * 1.01:
        votes.append("fake_breakout_short")
        confidence += 0.15
    elif last["close"] < last["support"] * 1.005 and last["close"] > last["support"] * 0.99:
        votes.append("fake_breakout_long")
        confidence += 0.15

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

# ---------------- PLOT SIGNAL ----------------
def _last_unique_levels(series, n=5):
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

    if pretop:
        pts = [df['close'].iloc[-3], df['close'].iloc[-2], df['close'].iloc[-1]]
        addplots.append(mpf.make_addplot(pts, type='scatter', markersize=80, marker='^', color='magenta'))

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

    entry = last['close'] if action in ["LONG","SHORT"] else None
    atr = float(last.get("ATR") or 0.0)
    take_levels = []
    stop = None
    if entry and atr > 0:
        mults = [0.5, 1.0, 1.5, 2.0, 2.5, 3.5, 5.0, 7.0, 9.0, 12.0]
        if action == "LONG":
            take_levels = [entry + atr * m for m in mults]
            stop = entry - atr * 1.5
        else:
            take_levels = [entry - atr * m for m in mults]
            stop = entry + atr * 1.5

    if entry:
        addplots.append(mpf.make_addplot([entry]*len(df), type='scatter', markersize=28, marker='v', color='blue'))
    if stop:
        addplots.append(mpf.make_addplot([stop]*len(df), type='scatter', markersize=24, marker='x', color='red'))
    for tl in take_levels:
        addplots.append(mpf.make_addplot([tl]*len(df), type='scatter', markersize=12, marker='^', color='green'))

    h_support = _last_unique_levels(df['support'], n=3)
    h_resistance = _last_unique_levels(df['resistance'], n=3)
    hlines = list(h_support) + list(h_resistance)

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

# ---------------- BACKTEST SIGNALS ----------------
def backtest_signals(symbol, df=None):
    """Розрахунок реального winrate по історії сигналів"""
    try:
        if df is None:
            df = fetch_klines(symbol, limit=EMA_SCAN_LIMIT*2)
        if df is None or len(df) < 30:
            return 0.0
        df = apply_all_features(df)
        wins = 0
        total = 0
        for i in range(20, len(df)):
            sub_df = df.iloc[:i]
            action, votes, pretop, last, conf = detect_signal(sub_df)
            if action in ["LONG","SHORT"] and conf >= CONF_THRESHOLD_MEDIUM:
                total += 1
                entry = last['close']
                atr = float(last.get("ATR") or 0.0)
                if atr <= 0:
                    continue
                if action=="LONG" and (df["close"].iloc[i]-entry) >= atr:
                    wins += 1
                elif action=="SHORT" and (entry-df["close"].iloc[i]) >= atr:
                    wins +=1
        rate = (wins/total) if total>0 else 0
        with state_lock:
            state.setdefault("backtest_stats", {})[symbol] = {"winrate": rate, "total": total, "wins": wins}
        save_json_safe(STATE_FILE, state)
        return rate
    except Exception as e:
        logger.exception("backtest_signals error %s: %s", symbol, e)
        return 0.0

# ---------------- ML / Ranking топ-5 ----------------
def rank_top_symbols():
    """Розраховує топ-5 токенів за ймовірністю profit на основі confidence + історії"""
    ranked = []
    with state_lock:
        for sym, sig in state.get("signals", {}).items():
            conf = sig.get("confidence", 0.0)
            win_stat = state.get("backtest_stats", {}).get(sym, {}).get("winrate", 0.0)
            score = 0.6*conf + 0.4*win_stat
            ranked.append((sym, score))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return [s for s,_ in ranked[:5]]

# ---------------- SCAN TASK ----------------
def scan_market(send_signals=True):
    logger.info("Starting market scan...")
    symbols = get_all_usdt_symbols()[:TOP_LIMIT]
    results = []

    def worker(symbol):
        df = fetch_klines(symbol)
        if df is None or len(df)<20:
            return None
        df = apply_all_features(df)
        action, votes, pretop, last, conf = detect_signal(df)
        return (symbol, action, votes, pretop, conf, df)

    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as exe:
        futures = [exe.submit(worker, s) for s in symbols]
        for f in futures:
            res = f.result()
            if res:
                results.append(res)

    for symbol, action, votes, pretop, conf, df in results:
        with state_lock:
            state["signals"][symbol] = {"action": action, "votes": votes, "pretop": pretop, "confidence": conf, "last_update": datetime.now(timezone.utc).isoformat()}
        save_json_safe(STATE_FILE, state)

        # --- новий блок: автоматична відправка сигналів ---
        if send_signals and action in ["LONG","SHORT"] and conf>=CONF_THRESHOLD_MEDIUM:
            text = f"Signal {symbol}: {action} | Confidence: {conf:.2f}\nVotes: {votes}"
            chart = plot_signal_candles(df, symbol, action, votes, pretop)
            send_telegram(text, photo=chart)

    logger.info("Market scan complete. Signals: %d", len(results))

# ---------------- TELEGRAM WEBHOOK ----------------
@app.route("/telegram_webhook/<token>", methods=["POST"])
def telegram_webhook(token):
    if token != TELEGRAM_TOKEN:
        return "Unauthorized", 403
    try:
        data = request.get_json()
        if not data:
            return "No data", 400

        message = data.get("message")
        if message:
            chat_id = message["chat"]["id"]
            text = message.get("text", "")

            # --- команди ---
            if text.startswith("/start"):
                send_telegram("Бот запущено!", chat_id=chat_id)
            elif text.startswith("/top5"):
                top = rank_top_symbols()
                msg = "Топ 5 токенів за ML + winrate:\n" + "\n".join(top)
                send_telegram(msg, chat_id=chat_id)
            elif text.startswith("/mlplot"):
                parts = text.split()
                if len(parts)>=2:
                    symbol = parts[1].upper()
                    df = fetch_klines(symbol, limit=EMA_SCAN_LIMIT*2)
                    if df is not None:
                        df = apply_all_features(df)
                        # графік всіх точок входу/виходу
                        chart_buf = plot_signal_candles(df, symbol, "ML PLOT", [], pretop=False)
                        send_telegram(f"ML plot for {symbol}", photo=chart_buf, chat_id=chat_id)
                    else:
                        send_telegram(f"Не вдалося отримати дані по {symbol}", chat_id=chat_id)
                else:
                    send_telegram("Вкажіть токен: /mlplot SYMBOL", chat_id=chat_id)
            else:
                send_telegram(f"Отримано: {text}", chat_id=chat_id)

        return jsonify({"ok": True})
    except Exception as e:
        logger.exception("Webhook error: %s", e)
        return "Error", 500

# ---------------- SCHEDULER ----------------
scheduler = BackgroundScheduler()
scheduler.add_job(scan_market, 'interval', minutes=SCAN_INTERVAL_MINUTES)
scheduler.start()

# ---------------- MAIN ----------------
if __name__ == "__main__":
    set_telegram_webhook(WEBHOOK_URL)
    scan_market()
    app.run(host="0.0.0.0", port=PORT)