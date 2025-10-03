import os
import io
import requests
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify

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
        "c": [float(d[4]) for d in data],
        "v": [float(d[5]) for d in data],
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
def send_telegram(text: str):
    payload = {
        "chat_id": CHAT_ID,
        "text": text,
        "parse_mode": "HTML"
    }
    requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", json=payload, timeout=10)

# ================== SMART AUTO ==================
def smart_auto():
    try:
        url = "https://api.binance.com/api/v3/ticker/24hr"
        data = requests.get(url).json()

        symbols = [d for d in data if d["symbol"].endswith("USDT") and float(d["quoteVolume"]) > 5_000_000]
        symbols = sorted(symbols, key=lambda x: abs(float(x["priceChangePercent"])), reverse=True)
        top_symbols = [s["symbol"] for s in symbols[:30]]

        signals = []
        for symbol in top_symbols:
            try:
                df = get_klines(symbol, interval="1h", limit=200)
                closes = np.array(df["c"], dtype=float)
                last_price = closes[-1]
                sr_levels = find_support_resistance(closes, window=20, delta=0.005)

                signal = None
                for lvl in sr_levels:
                    if last_price > lvl * 1.01:
                        signal = f"🚀 LONG breakout {symbol} біля {lvl:.4f}"
                        break
                    elif last_price < lvl * 0.99:
                        signal = f"⚡ SHORT breakout {symbol} біля {lvl:.4f}"
                        break
                if signal:
                    signals.append(signal)
            except Exception as e:
                print(f"[ERROR] {symbol}: {e}")
                continue

        if not signals:
            send_telegram("ℹ️ Жодних сигналів не знайдено.")
        else:
            text = "<b>Smart Auto S/R Signals</b>\n\n" + "\n\n".join(signals)
            send_telegram(text)

    except Exception as e:
        send_telegram(f"❌ Error: {e}")

# ================== WEBHOOK ROUTE ==================
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

# ================== SETUP WEBHOOK ==================
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