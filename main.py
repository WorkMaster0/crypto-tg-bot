# main.py — Pre-top бот з графіками, Telegram і backtest winrate
import os
import time
import json
import logging
import re
from datetime import datetime, timezone
from threading import Thread
from concurrent.futures import ThreadPoolExecutor
import io

import pandas as pd
import matplotlib.pyplot as plt
import requests
from flask import Flask, request, jsonify
import ta
import mplfinance as mpf
from scipy.signal import find_peaks
import numpy as np

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
PARALLEL_WORKERS = int(os.getenv("PARALLEL_WORKERS", "3"))

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
    from requests import Session
    session = Session()
    client = BinanceClient(
        api_key=BINANCE_API_KEY,
        api_secret=BINANCE_API_SECRET,
        requests_params={"timeout": 30}
    )
else:
    client = None
    logger.warning("Binance client unavailable or API keys missing")

# ---------------- FLASK ----------------
app = Flask(__name__)

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

state = load_json_safe(STATE_FILE, {"signals": {}, "last_scan": None, "signal_history": {}})

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
            payload = {
                "chat_id": CHAT_ID,
                "text": escape_md_v2(text),
                "parse_mode": "MarkdownV2",
                "disable_web_page_preview": True
            }
            requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", json=payload, timeout=10)
    except Exception as e:
        logger.exception("send_telegram error: %s", e)

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
            "BUSD", "USDC", "FDUSD", "TUSD", "DAI", "EUR", "GBP", "AUD", "STRAX", "GNS", "ALCX",
            "BTCST", "COIN", "AAPL", "TSLA", "MSFT", "META", "GOOG", "USD1", "BTTC", "ARDR", "DF", "XNO"
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
            df = pd.DataFrame(kl, columns=["open_time", "open", "high", "low", "close", "volume",
                                           "close_time", "qav", "num_trades", "tb_base", "tb_quote", "ignore"])
            df = df[["open_time", "open", "high", "low", "close", "volume"]].astype(float)
            df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
            df.set_index("open_time", inplace=True)
            return df
        except Exception as e:
            logger.warning("fetch_klines %s attempt %d error: %s", symbol, attempt + 1, e)
            time.sleep(0.5)
    return None

# ---------------- FEATURES ----------------
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
    except Exception as e:
        logger.exception("apply_all_features error: %s", e)
    return df

# ---------------- SIGNAL DETECTION ----------------
def detect_signal(df: pd.DataFrame):
    last = df.iloc[-1]
    prev = df.iloc[-2]
    votes = []
    confidence = 0.2

    if last["ema_8"] > last["ema_20"]: votes.append("ema_bull"); confidence += 0.1
    else: votes.append("ema_bear"); confidence += 0.05

    if last["MACD_hist"] > 0: votes.append("macd_up"); confidence += 0.1
    else: votes.append("macd_down"); confidence += 0.05

    if last["RSI_14"] < 30: votes.append("rsi_oversold"); confidence += 0.08
    elif last["RSI_14"] > 70: votes.append("rsi_overbought"); confidence += 0.08

    if last["ADX"] > 25: votes.append("strong_trend"); confidence *= 1.1

    body = last["close"] - last["open"]
    rng = last["high"] - last["low"]
    upper_shadow = last["high"] - max(last["close"], last["open"])
    lower_shadow = min(last["close"], last["open"]) - last["low"]
    candle_bonus = 1.0

    if lower_shadow > 2 * abs(body) and body > 0: votes.append("hammer_bull"); candle_bonus = 1.2
    elif upper_shadow > 2 * abs(body) and body < 0: votes.append("shooting_star"); candle_bonus = 1.2
    if body > 0 and prev["close"] < prev["open"] and last["close"] > prev["open"] and last["open"] < prev["close"]: votes.append("bullish_engulfing"); candle_bonus=1.25
    elif body < 0 and prev["close"] > prev["open"] and last["close"] < prev["open"] and last["open"] > prev["close"]: votes.append("bearish_engulfing"); candle_bonus=1.25
    if abs(body) < 0.1 * rng: votes.append("doji"); candle_bonus = 1.1

    confidence *= candle_bonus

    if last["close"] > last["resistance"]*0.995 and last["close"] < last["resistance"]*1.01: votes.append("fake_breakout_short"); confidence += 0.15
    elif last["close"] < last["support"]*1.005 and last["close"] > last["support"]*0.99: votes.append("fake_breakout_long"); confidence += 0.15

    pretop = False
    if len(df) >= 10:
        recent, last10 = df["close"].iloc[-1], df["close"].iloc[-10]
        if (recent - last10) / last10 > 0.1:
            pretop = True; confidence += 0.1; votes.append("pretop")

    action = "WATCH"
    if last["close"] >= last["resistance"]*0.995: action = "SHORT"
    elif last["close"] <= last["support"]*1.005: action = "LONG"

    confidence = max(0, min(1, confidence))
    return action, votes, pretop, last, confidence

# ---------------- BACKTEST ----------------
def backtest_winrate(df: pd.DataFrame):
    df = apply_all_features(df)
    results = []

    for i in range(1, len(df)):
        sub_df = df.iloc[:i+1]
        action, votes, pretop, last, conf = detect_signal(sub_df)
        if action in ["LONG", "SHORT"] and conf >= CONF_THRESHOLD_MEDIUM:
            entry = last["close"]
            future = df["close"].iloc[i+1:].values
            win = False
            for p in future:
                if action=="LONG" and p >= entry*(1+0.04): win=True; break
                elif action=="LONG" and p <= entry*(1-0.02): win=False; break
                elif action=="SHORT" and p <= entry*(1-0.04): win=True; break
                elif action=="SHORT" and p >= entry*(1+0.02): win=False; break
            results.append((action, win))
    total = len(results)
    wins = sum(1 for _, w in results if w)
    winrate = wins/total if total>0 else 0
    return winrate, results

# ---------------- TOP5 ----------------
def get_top5_symbols(symbols):
    winrates = {}
    def calc_wr(sym):
        df = fetch_klines(sym)
        if df is None or len(df) < 30: return None
        wr, _ = backtest_winrate(df)
        return (sym, wr)

    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as exe:
        futures = [exe.submit(calc_wr, s) for s in symbols]
        for fut in futures:
            res = fut.result()
            if res: winrates[res[0]] = res[1]

    sorted_wr = sorted(winrates.items(), key=lambda x: x[1], reverse=True)
    return sorted_wr[:5]

# ---------------- ANALYZE SYMBOL ----------------
def analyze_and_alert(symbol: str):
    df = fetch_klines(symbol)
    if df is None or len(df) < 30: return
    df = apply_all_features(df)
    action, votes, pretop, last, confidence = detect_signal(df)
    prev_signal = state["signals"].get(symbol, "")

    if pretop:
        send_telegram(f"⚡ Pre-top detected for {symbol}, price={last['close']:.6f}")

    if action != "WATCH" and confidence >= CONF_THRESHOLD_MEDIUM and action != prev_signal:
        msg = (
            f"⚡ TRADE SIGNAL\nSymbol: {symbol}\nAction: {action}\nPrice: {last['close']:.6f}\n"
            f"Support: {last['support']:.6f}\nResistance: {last['resistance']:.6f}\n"
            f"Confidence: {confidence:.2f}\nPatterns: {','.join(votes)}\nPre-top: {pretop}\nTime: {last.name}\n"
        )
        photo_buf = None
        state["signals"][symbol] = action
        save_json_safe(STATE_FILE, state)
        send_telegram(msg, photo=photo_buf)

# ---------------- MASTER SCAN ----------------
def scan_top_symbols():
    symbols = get_all_usdt_symbols()
    if not symbols: return
    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as exe:
        list(exe.map(analyze_and_alert, symbols))
    state["last_scan"] = str(datetime.now(timezone.utc))
    save_json_safe(STATE_FILE, state)

# ---------------- FLASK ----------------
@app.route("/")
def home():
    return jsonify({
        "status": "ok",
        "time": str(datetime.now(timezone.utc)),
        "signals": len(state.get("signals", {}))
    })

@app.route("/telegram_webhook/<token>", methods=["POST"])
def telegram_webhook(token):
    try:
        if token != TELEGRAM_TOKEN: return jsonify({"ok": False, "error": "invalid token"}), 403
        update = request.get_json(force=True) or {}
        msg = update.get("message")
        if not msg: return jsonify({"ok": True})
        text = msg.get("text", "").lower().strip()

        if text.startswith("/scan"):
            send_telegram("⚡ Manual scan started.")
            Thread(target=scan_top_symbols, daemon=True).start()

        elif text.startswith("/status"):
            send_telegram(f"📝 Status:\nSignals={len(state.get('signals', {}))}\nLast scan={state.get('last_scan')}")

        elif text.startswith("/top"):
            symbols = get_all_usdt_symbols()[:TOP_LIMIT]
            if not symbols:
                send_telegram("❌ No symbols available for top scan")
            else:
                top5 = get_top5_symbols(symbols)
                if not top5:
                    send_telegram("❌ No data to calculate top symbols")
                else:
                    msg_text = "🏆 Top5 tokens by winrate:\n" + "\n".join(
                        [f"{s[0]}: {s[1]*100:.1f}%" for s in top5]
                    )
                    send_telegram(msg_text)

    except Exception as e:
        logger.exception("telegram_webhook error: %s", e)
    return jsonify({"ok": True})

# ---------------- MAIN ----------------
if __name__ == "__main__":
    logger.info("Starting pre-top detector bot")
    app.run(host="0.0.0.0", port=PORT)