import os
import io
import requests
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
from flask import Flask, request, jsonify
from concurrent.futures import ThreadPoolExecutor

# ================== ENV ==================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
if not TELEGRAM_TOKEN or not CHAT_ID:
    raise ValueError("Відсутній TELEGRAM_TOKEN або CHAT_ID у .env!")

PORT = int(os.getenv("PORT", 5000))
WEBHOOK_URL_BASE = os.getenv("WEBHOOK_URL_BASE")  # Наприклад: https://<твій-домен>.onrender.com
WEBHOOK_URL_PATH = f"/telegram_webhook/{TELEGRAM_TOKEN}"

# ================== Flask ==================
app = Flask(__name__)

# ================== Допоміжні функції ==================
def get_klines(symbol, interval="1h", limit=200):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    data = requests.get(url).json()
    return {
        "c": np.array([float(d[4]) for d in data], dtype=float),
        "o": np.array([float(d[1]) for d in data], dtype=float),
        "h": np.array([float(d[2]) for d in data], dtype=float),
        "l": np.array([float(d[3]) for d in data], dtype=float),
        "v": np.array([float(d[5]) for d in data], dtype=float),
    }

def find_support_resistance(closes, window=20, delta=0.005):
    levels = []
    for i in range(window, len(closes)-window):
        local_max = max(closes[i-window:i+window+1])
        local_min = min(closes[i-window:i+window+1])
        if abs(closes[i] - local_max)/local_max < delta:
            levels.append(local_max)
        if abs(closes[i] - local_min)/local_min < delta:
            levels.append(local_min)
    return sorted(list(set(levels)))

# ================== TELEGRAM ==================
def send_telegram(text: str, photo=None):
    try:
        if photo:
            files = {'photo': ('signal.png', photo, 'image/png')}
            data = {'chat_id': CHAT_ID, 'caption': text, 'parse_mode': 'HTML'}
            requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto", data=data, files=files, timeout=10)
        else:
            payload = {"chat_id": CHAT_ID, "text": text, "parse_mode": "HTML"}
            requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", json=payload, timeout=10)
    except Exception as e:
        print(f"[ERROR] Telegram send failed: {e}")

# ================== ГРАФІК ==================
def plot_signal(df_data, symbol, level, last_price, signal_type):
    import pandas as pd
    df = pd.DataFrame({
        "Open": df_data["o"],
        "High": df_data["h"],
        "Low": df_data["l"],
        "Close": df_data["c"],
        "Volume": df_data["v"]
    })
    df.index = pd.date_range(end=pd.Timestamp.now(), periods=len(df))

    addplots = [mpf.make_addplot([level]*len(df), color='red' if signal_type=="SHORT" else 'green', linestyle="--")]
    fig, ax = mpf.plot(df.tail(60), type='candle', style='yahoo', addplot=addplots,
                       returnfig=True, title=f"{symbol} {signal_type} signal")
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return buf

# ================== SMART AUTO ==================
def analyze_symbol(symbol):
    try:
        df = get_klines(symbol)
        closes = df["c"]
        last_price = closes[-1]
        sr_levels = find_support_resistance(closes, window=20, delta=0.005)

        for lvl in sr_levels:
            if last_price > lvl * 1.01:
                pct = (last_price - lvl)/lvl*100
                text = f"🚀 <b>LONG breakout</b> {symbol}\nРівень: {lvl:.4f}\nЦіна: {last_price:.4f} ({pct:.2f}% вище)"
                photo = plot_signal(df, symbol, lvl, last_price, "LONG")
                send_telegram(text, photo)
                return
            elif last_price < lvl * 0.99:
                pct = (lvl - last_price)/lvl*100
                text = f"⚡ <b>SHORT breakout</b> {symbol}\nРівень: {lvl:.4f}\nЦіна: {last_price:.4f} ({pct:.2f}% нижче)"
                photo = plot_signal(df, symbol, lvl, last_price, "SHORT")
                send_telegram(text, photo)
                return
    except Exception as e:
        print(f"[ERROR] {symbol}: {e}")

def smart_auto():
    try:
        url = "https://api.binance.com/api/v3/ticker/24hr"
        data = requests.get(url).json()
        symbols = [d for d in data if d["symbol"].endswith("USDT") and float(d["quoteVolume"]) > 5_000_000]
        symbols = sorted(symbols, key=lambda x: abs(float(x["priceChangePercent"])), reverse=True)
        top_symbols = [s["symbol"] for s in symbols[:30]]

        if not top_symbols:
            send_telegram("ℹ️ Жодних монет не знайдено для аналізу.")
            return

        with ThreadPoolExecutor(max_workers=6) as executor:
            executor.map(analyze_symbol, top_symbols)

    except Exception as e:
        send_telegram(f"❌ Error: {e}")

# ================== WEBHOOK ==================
@app.route(WEBHOOK_URL_PATH, methods=['POST'])
def telegram_webhook():
    update = request.get_json(force=True) or {}
    text = update.get("message", {}).get("text", "").lower().strip()
    if text == "/smart_auto":
        send_telegram("⚡ Виконую /smart_auto ...")
        smart_auto()
    return jsonify({"ok": True})

@app.route("/", methods=['GET'])
def index():
    return "Bot is running ✅", 200

# ================== WEBHOOK SETUP ==================
def setup_webhook():
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/setWebhook"
        resp = requests.post(url, json={"url": WEBHOOK_URL_BASE + WEBHOOK_URL_PATH})
        print("[INFO] Webhook встановлено:", resp.json())
    except Exception as e:
        print("[ERROR] Не вдалося встановити webhook:", e)

setup_webhook()

# ================== RUN ==================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)