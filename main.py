import os
import io
import re
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
    data = requests.get(url, timeout=10).json()
    return {
        "c": [float(d[4]) for d in data],
        "h": [float(d[2]) for d in data],
        "l": [float(d[3]) for d in data],
        "v": [float(d[5]) for d in data],
    }

def find_support_resistance(prices, window=20, delta=0.005):
    sr_levels = []
    for i in range(window, len(prices)-window):
        local_max = max(prices[i-window:i+window+1])
        local_min = min(prices[i-window:i+window+1])
        if prices[i] == local_max and all(abs(prices[i]-lvl)/lvl > delta for lvl in sr_levels):
            sr_levels.append(prices[i])
        elif prices[i] == local_min and all(abs(prices[i]-lvl)/lvl > delta for lvl in sr_levels):
            sr_levels.append(prices[i])
    return sorted(sr_levels)

def plot_candles(symbol, interval="1h", limit=100):
    df = get_klines(symbol, interval=interval, limit=limit)
    closes = df["c"][-limit:]
    highs = df["h"][-limit:]
    lows = df["l"][-limit:]
    opens = [closes[0]] + closes[:-1]  # простий approximation

    fig, ax = plt.subplots(figsize=(8,4))
    for i in range(len(closes)):
        color = "green" if closes[i] >= opens[i] else "red"
        ax.plot([i,i],[lows[i], highs[i]], color=color)
        ax.plot([i-0.1,i+0.1],[closes[i], closes[i]], color=color, linewidth=3)
    ax.set_title(symbol)
    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return buf

# ================== TELEGRAM ==================
def send_telegram(text: str, photo=None):
    if photo:
        files = {'photo': ('signal.png', photo, 'image/png')}
        data = {'chat_id': CHAT_ID, 'caption': text, 'parse_mode': 'HTML'}
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto", data=data, files=files)
    else:
        payload = {"chat_id": CHAT_ID, "text": text, "parse_mode": "HTML"}
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", json=payload, timeout=10)

# ================== SMART AUTO ==================
def smart_auto():
    try:
        print("[INFO] Запускаю smart_auto()...")

        url = "https://api.binance.com/api/v3/ticker/24hr"
        data = requests.get(url, timeout=10).json()

        print(f"[INFO] Отримано {len(data)} монет з Binance")

        # Фільтруємо тільки USDT-пари з нормальним об'ємом
        symbols = [d for d in data if d["symbol"].endswith("USDT") and float(d["quoteVolume"]) > 5_000_000]
        print(f"[INFO] Фільтровані монети (>5M USDT volume): {len(symbols)}")

        # Сортуємо за % зміни ціни за 24 години
        symbols = sorted(symbols, key=lambda x: abs(float(x["priceChangePercent"])), reverse=True)

        top_symbols = [s["symbol"] for s in symbols[:20]]
        print(f"[INFO] TOP 20 монет: {top_symbols}")

        all_signals = []

        for symbol in top_symbols:
            try:
                df = get_klines(symbol, interval="1h", limit=200)
                closes = np.array(df["c"], dtype=float)
                volumes = np.array(df["v"], dtype=float)
                last_price = closes[-1]

                if len(closes) < 20:
                    print(f"[WARN] {symbol}: недостатньо даних")
                    continue  

                sr_levels = find_support_resistance(closes, window=20, delta=0.005)
                signals = []

                # ---------- Breakout & Fake breakout ----------
                for lvl in sr_levels:
                    diff = last_price - lvl
                    diff_pct = (diff / lvl) * 100
                    if last_price > lvl * 1.01:
                        signals.append(f"🚀 LONG breakout біля {lvl:.4f}")
                    elif last_price < lvl * 0.99:
                        signals.append(f"⚡ SHORT breakout біля {lvl:.4f}")
                    elif abs(last_price - lvl)/lvl <= 0.01:
                        signals.append(f"⚠️ Fake breakout біля {lvl:.4f}")

                # ---------- Pre-top ----------
                if len(closes) >= 4:
                    impulse = (closes[-1] - closes[-4]) / closes[-4]
                    vol_spike = volumes[-1] > 1.5 * np.mean(volumes[-20:])
                    nearest_res = max([lvl for lvl in sr_levels if lvl < last_price], default=None)
                    if impulse > 0.08 and vol_spike and nearest_res is not None:
                        signals.append(f"⚠️ Pre-top біля {nearest_res:.4f}")

                if signals:
                    print(f"[SIGNAL] {symbol}: {signals}")
                    all_signals.append(f"<b>{symbol}</b>\n" + "\n".join(signals))
                else:
                    print(f"[INFO] {symbol}: сигналів нема")

            except Exception as e:
                print(f"[ERROR] {symbol}: {e}")
                continue

        # Надсилаємо результати
        if not all_signals:
            print("[INFO] Жодних сигналів не знайдено")
            send_telegram("ℹ️ Жодних сигналів не знайдено.")
        else:
            text = "<b>Smart Auto S/R Signals</b>\n\n" + "\n\n".join(all_signals)
            first_symbol = re.search(r"<b>(\w+)</b>", all_signals[0]).group(1)
            photo = plot_candles(first_symbol)
            send_telegram(text, photo=photo)

    except Exception as e:
        print(f"[FATAL] smart_auto(): {e}")
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