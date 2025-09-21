import os
import time
import logging
import re
from datetime import datetime, timezone
from threading import Thread
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import matplotlib.pyplot as plt
import requests
from flask import Flask, request, jsonify
import ta
import mplfinance as mpf
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
PORT = int(os.getenv("PORT", "5000"))
PARALLEL_WORKERS = int(os.getenv("PARALLEL_WORKERS", "6"))
EMA_SCAN_LIMIT = 500
STATE_FILE = "state.parquet"  # Паркет замість JSON
CONF_THRESHOLD_MEDIUM = 0.60

# ---------------- STATE ----------------
if os.path.exists(STATE_FILE):
    try:
        signals_df = pd.read_parquet(STATE_FILE)
    except:
        signals_df = pd.DataFrame(columns=[
            "symbol","time","action","confidence","price","votes","pretop",
            "support","resistance","tp","sl"
        ])
else:
    signals_df = pd.DataFrame(columns=[
        "symbol","time","action","confidence","price","votes","pretop",
        "support","resistance","tp","sl"
    ])

def save_signals():
    try:
        signals_df.to_parquet(STATE_FILE, index=False)
    except Exception as e:
        logger.exception("Failed to save signals: %s", e)

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

# ---------------- WEBSOCKET MANAGER ----------------
from websocket_manager import WebSocketKlineManager

ALL_USDT = [
    "BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT","ADAUSDT","DOGEUSDT","MATICUSDT",
    "DOTUSDT","TRXUSDT","LTCUSDT","AVAXUSDT","SHIBUSDT","LINKUSDT","ATOMUSDT","XMRUSDT",
    "ETCUSDT","XLMUSDT","APTUSDT","NEARUSDT","FILUSDT","ICPUSDT","GRTUSDT","AAVEUSDT",
    "SANDUSDT","AXSUSDT","FTMUSDT","THETAUSDT","EGLDUSDT","MANAUSDT","FLOWUSDT","HBARUSDT",
    "ALGOUSDT","ZECUSDT","EOSUSDT","KSMUSDT","CELOUSDT","SUSHIUSDT","CHZUSDT","KAVAUSDT",
    "ZILUSDT","ANKRUSDT","RAYUSDT","GMTUSDT","UNIUSDT","APEUSDT","PEPEUSDT","OPUSDT",
    "LINKUSDT","XTZUSDT","ALPHAUSDT","BALUSDT","COMPUSDT","CRVUSDT","SNXUSDT","RSRUSDT",
    "LOKUSDT","GALUSDT","WLDUSDT","JASMYUSDT","ONEUSDT","ARBUSDT","ALICEUSDT",
    "XECUSDT","FLMUSDT","CAKEUSDT","AXSUSDT","IMXUSDT","HOOKUSDT","MAGICUSDT","STGUSDT",
    "FETUSDT","PEOPLEUSDT","ASTRUSDT","ENSUSDT","CTSIUSDT","GALAUSDT","RADUSDT","IOSTUSDT",
    "QTUMUSDT","NPXSUSDT","DASHUSDT","ZRXUSDT","HNTUSDT","ENJUSDT","ICPUSDT",
    "TFUELUSDT","KLAYUSDT","TWTUSDT","NKNUSDT","GLMRUSDT","ZENUSDT","STORJUSDT",
    "ICXUSDT","XVGUSDT","FLOKIUSDT","BONEUSDT","TRBUSDT","C98USDT","MASKUSDT",
    "1000SHIBUSDT","1000PEPEUSDT","AMBUSDT","VEGUSDT","QNTUSDT","RNDRUSDT","CHRUSDT",
    "API3USDT","MTLUSDT","ALPUSDT","LDOUSDT","AXLUSDT","FUNUSDT","OGUSDT",
    "ORCUSDT","XAUTUSDT","ARUSDT","DYDXUSDT","RUNEUSDT","FLUXUSDT","AGIXUSDT","AGLDUSDT",
    "PERPUSDT","STMXUSDT","MLNUSDT","NMRUSDT","LRCUSDT","COTIUSDT","ACHUSDT","CKBUSDT","ACEUSDT","TRUUSDT","IPSUSDT","QIUSDT","GALUSDT","GLMUSDT",
    "BALUSDT","MDXUSDT","ARNXUSDT","PORTOUSDT","MIRUSDT","ROSEUSDT","OXTUSDT","SPELLUSDT","STRAXUSDT",
    "SUNUSDT","SYSUSDT","TAOUSDT","TLMUSDT","VLXUSDT","WAXPUSDT","WIFUSDT","XNOUSDT","XEMUSDT"
]

ws_manager = WebSocketKlineManager(symbols=ALL_USDT, interval="15m")
Thread(target=ws_manager.start, daemon=True).start()

def fetch_klines(symbol, limit=500):
    return ws_manager.get_klines(symbol, limit)

# ---------------- FEATURE ENGINEERING ----------------
def apply_all_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
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
    return df

# ---------------- SIGNAL DETECTION ----------------
def detect_signal(df: pd.DataFrame):
    df = apply_all_features(df)
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
    fake_breakout = False
    if last["close"] > last["resistance"] * 0.995 and last["close"] < last["resistance"] * 1.01:
        votes.append("fake_breakout_short")
        confidence += 0.15
        fake_breakout = True
    elif last["close"] < last["support"] * 1.005 and last["close"] > last["support"] * 0.99:
        votes.append("fake_breakout_long")
        confidence += 0.15
        fake_breakout = True

    # Pre-top
    pretop = False
    if len(df) >= 10:
        recent, last10 = df["close"].iloc[-1], df["close"].iloc[-10]
        if (recent - last10) / last10 > 0.1:
            pretop = True
            votes.append("pretop")
            confidence += 0.1

    # Action
    action = "WATCH"
    if last["close"] >= last["resistance"] * 0.995:
        action = "SHORT"
    elif last["close"] <= last["support"] * 1.005:
        action = "LONG"

    confidence = max(0, min(1, confidence))
    if not (fake_breakout or pretop):
        action = "WATCH"

    # Take profit / stop loss
    tp, sl = None, None
    if action == "LONG":
        tp = last["resistance"]
        sl = last["support"]
    elif action == "SHORT":
        tp = last["support"]
        sl = last["resistance"]

    return action, votes, pretop, last, confidence, tp, sl

# ---------------- PLOT SIGNAL ----------------
def plot_signal_candles(df, symbol, action, votes, pretop, tp=None, sl=None):
    df_plot = df.tail(50).copy()
    addplot = []
    if tp: addplot.append(mpf.make_addplot([tp]*len(df_plot), color='green'))
    if sl: addplot.append(mpf.make_addplot([sl]*len(df_plot), color='red'))
    fig, ax = mpf.plot(
        df_plot, type='candle', style='yahoo',
        title=f"{symbol} - {action}", returnfig=True, addplot=addplot
    )
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return buf

# ---------------- ANALYZE SYMBOL ----------------
def analyze_and_alert(symbol: str):
    global signals_df
    df = fetch_klines(symbol)
    if df is None or len(df) < 30:
        return
    action, votes, pretop, last, confidence, tp, sl = detect_signal(df)
    prev_signal = signals_df[signals_df["symbol"] == symbol]["action"].iloc[-1] if not signals_df[signals_df["symbol"] == symbol].empty else None

    logger.info(
        "Symbol=%s action=%s confidence=%.2f votes=%s pretop=%s",
        symbol, action, confidence, votes, pretop
    )

    if action != "WATCH" and confidence >= CONF_THRESHOLD_MEDIUM and action != prev_signal:
        signal_reasons = []
        if "fake_breakout_short" in votes or "fake_breakout_long" in votes:
            signal_reasons.append("Fake Breakout")
        if pretop:
            signal_reasons.append("Pre-Top")
        if not signal_reasons:
            signal_reasons = ["Other patterns"]

        msg = (
            f"⚡ TRADE SIGNAL\n"
            f"Symbol: {symbol}\n"
            f"Action: {action}\n"
            f"Price: {last['close']:.6f}\n"
            f"Confidence: {confidence:.2f}\n"
            f"TP: {tp:.6f}\n"
            f"SL: {sl:.6f}\n"
            f"Reasons: {','.join(signal_reasons)}\n"
            f"Patterns: {','.join(votes)}\n"
            f"Time: {last.name}\n"
        )

        photo_buf = plot_signal_candles(df, symbol, action, votes, pretop, tp, sl)
        send_telegram(msg, photo=photo_buf)

        # Додаємо сигнал в історію
        new_row = pd.DataFrame([{
            "symbol": symbol,
            "time": last.name,
            "action": action,
            "confidence": confidence,
            "price": last["close"],
            "votes": ",".join(votes),
            "pretop": pretop,
            "support": last["support"],
            "resistance": last["resistance"],
            "tp": tp,
            "sl": sl
        }])
        signals_df = pd.concat([signals_df, new_row], ignore_index=True)
        save_signals()

# ---------------- MASTER SCAN ----------------
def scan_all_symbols():
    symbols = list(ws_manager.data.keys())
    if not symbols:
        logger.warning("No symbols loaded from WebSocket")
        return

    logger.info("Starting scan for %d symbols", len(symbols))

    def safe_analyze(sym):
        try:
            analyze_and_alert(sym)
        except Exception as e:
            logger.exception("Error analyzing symbol %s: %s", sym, e)

    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as exe:
        list(exe.map(safe_analyze, symbols))

# ---------------- FLASK ----------------
app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({
        "status": "ok",
        "time": str(datetime.now(timezone.utc)),
        "signals": len(signals_df)
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
    elif text.startswith("/history"):
        last10 = signals_df.tail(10)
        msg = "\n".join([
            f"{row.symbol}: {row.action} @ {row.price:.4f} TP={row.tp:.4f} SL={row.sl:.4f}" for _, row in last10.iterrows()
        ])
        send_telegram("📜 Last 10 signals:\n" + msg)
    elif text.startswith("/top"):
        def calc_winrate(group):
            wins = 0
            total = 0
            for _, row in group.iterrows():
                if pd.isna(row.tp) or pd.isna(row.sl):
                    continue
                total += 1
                if row.action == "LONG" and row.price >= row.tp:
                    wins += 1
                elif row.action == "SHORT" and row.price <= row.tp:
                    wins += 1
            return wins / total if total > 0 else 0

        top_symbols = signals_df.groupby("symbol").apply(calc_winrate).sort_values(ascending=False).head(10)
        msg = "\n".join([f"{sym}: {winrate:.0%}" for sym, winrate in top_symbols.items()])
        send_telegram("🏆 Top symbols by winrate:\n" + msg)

    return jsonify({"ok": True})

# ---------------- MAIN ----------------
if __name__ == "__main__":
    logger.info("Starting pre-top detector bot")
    Thread(target=scan_all_symbols, daemon=True).start()
    app.run(host="0.0.0.0", port=PORT)