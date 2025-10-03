import os
import io
import requests
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify
import threading

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
def get_klines(symbol, interval="1h", limit=60):  # Зменшили limit для економії пам'яті
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    data = requests.get(url, timeout=5).json()
    return {
        "c": [float(d[4]) for d in data],
        "v": [float(d[5]) for d in data],
    }

def find_support_resistance(closes, window=10, delta=0.005):  # Зменшили window
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
        payload = {"chat_id": CHAT_ID, "text": text, "parse_mode": "HTML"}
        if photo:
            files = {'photo': ('signal.png', photo, 'image/png')}
            data = {'chat_id': CHAT_ID, 'caption': text, 'parse_mode': 'HTML'}
            requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto", data=data, files=files, timeout=10)
        else:
            requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", json=payload, timeout=10)
    except Exception as e:
        print(f"[ERROR] Telegram send failed: {e}")

# ================== SMART AUTO ==================
def plot_signal(closes, symbol, level, last_price):
    plt.figure(figsize=(6,4))
    plt.plot(closes, label="Close")
    plt.axhline(level, color='red', linestyle="--", label=f"Level {level:.4f}")
    plt.title(f"{symbol} breakout")
    plt.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close()
    return buf

def smart_auto():
    try:
        url = "https://api.binance.com/api/v3/ticker/24hr"
        data = requests.get(url, timeout=5).json()

        # Беремо лише USDT пари з великим обсягом
        symbols = [d for d in data if d["symbol"].endswith("USDT") and float(d["quoteVolume"]) > 5_000_000]
        symbols = sorted(symbols, key=lambda x: abs(float(x["priceChangePercent"])), reverse=True)
        top_symbols = [s["symbol"] for s in symbols[:30]]

        signals = []
        for symbol in top_symbols:
            try:
                df = get_klines(symbol, interval="1h", limit=60)
                closes = np.array(df["c"], dtype=float)
                last_price = closes[-1]
                sr_levels = find_support_resistance(closes, window=10, delta=0.005)

                for lvl in sr_levels:
                    pct_diff = (last_price - lvl) / lvl * 100
                    if last_price > lvl * 1.01:
                        signal = f"🚀 LONG breakout {symbol} біля {lvl:.4f} ({pct_diff:+.2f}%)"
                        photo_buf = plot_signal(closes, symbol, lvl, last_price)
                        send_telegram(signal, photo=photo_buf)
                        break
                    elif last_price < lvl * 0.99:
                        signal = f"⚡ SHORT breakout {symbol} біля {lvl:.4f} ({pct_diff:+.2f}%)"
                        photo_buf = plot_signal(closes, symbol, lvl, last_price)
                        send_telegram(signal, photo=photo_buf)
                        break

            except Exception as e:
                print(f"[ERROR] {symbol}: {e}")
                continue

        if not signals:
            send_telegram("ℹ️ Жодних сигналів не знайдено.")

    except Exception as e:
        send_telegram(f"❌ Error: {e}")

# ================== WEBHOOK ROUTE ==================
@app.route(WEBHOOK_URL_PATH, methods=['POST'])
def telegram_webhook():
    update = request.get_json(force=True) or {}
    text = update.get("message", {}).get("text", "").lower().strip()
    if text == "/smart_auto":
        send_telegram("⚡ Виконую /smart_auto ...")
        threading.Thread(target=smart_auto, daemon=True).start()  # <-- Фоновий потік
    return jsonify({"ok": True})

@app.route("/", methods=['GET'])
def index():
    return "Bot is running ✅", 200

# ================== SETUP WEBHOOK ==================
def setup_webhook():
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/setWebhook"
        resp = requests.post(url, json={"url": WEBHOOK_URL_BASE + WEBHOOK_URL_PATH}, timeout=5)
        print("[INFO] Webhook встановлено:", resp.json())
    except Exception as e:
        print("[ERROR] Не вдалося встановити webhook:", e)

setup_webhook()

# ================== RUN ==================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)