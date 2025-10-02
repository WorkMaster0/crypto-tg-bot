import os
import io
import requests
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, request
from telebot import TeleBot, types

# ================== ENV ==================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
if not TELEGRAM_TOKEN:
    raise ValueError("Відсутній TELEGRAM_TOKEN у .env!")

PORT = int(os.getenv("PORT", 5000))
WEBHOOK_URL_BASE = os.getenv("WEBHOOK_URL_BASE")  # Наприклад: https://<твій-домен>.onrender.com
WEBHOOK_URL_PATH = f"/telegram_webhook/{TELEGRAM_TOKEN}"

# ================== Flask & Bot ==================
app = Flask(__name__)
bot = TeleBot(TELEGRAM_TOKEN, parse_mode="HTML")

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

# ================== Команда /smart_auto ==================
@bot.message_handler(commands=['smart_auto'])
def smart_auto_handler(message):
    try:
        url = "https://api.binance.com/api/v3/ticker/24hr"
        data = requests.get(url).json()

        symbols = [
            d for d in data
            if d["symbol"].endswith("USDT") and float(d["quoteVolume"]) > 5_000_000
        ]

        symbols = sorted(symbols, key=lambda x: abs(float(x["priceChangePercent"])), reverse=True)
        top_symbols = [s["symbol"] for s in symbols[:30]]

        signals = []
        for symbol in top_symbols:
            try:
                df = get_klines(symbol, interval="1h", limit=200)
                closes = np.array(df["c"], dtype=float)
                volumes = np.array(df["v"], dtype=float)

                if len(closes) < 50:
                    continue

                sr_levels = find_support_resistance(closes, window=20, delta=0.005)
                last_price = closes[-1]

                signal = None
                for lvl in sr_levels:
                    diff = last_price - lvl
                    diff_pct = (diff / lvl) * 100
                    if last_price > lvl * 1.01:
                        signal = f"🚀 LONG breakout: ціна пробила опір {lvl:.4f}\n📊 Ринкова: {last_price:.4f} | Відрив: {diff:+.4f} ({diff_pct:+.2f}%)"
                        break
                    elif last_price < lvl * 0.99:
                        signal = f"⚡ SHORT breakout: ціна пробила підтримку {lvl:.4f}\n📊 Ринкова: {last_price:.4f} | Відрив: {diff:+.4f} ({diff_pct:+.2f}%)"
                        break

                impulse = (closes[-1] - closes[-4]) / closes[-4] if len(closes) >= 4 else 0
                vol_spike = volumes[-1] > 1.5 * np.mean(volumes[-20:]) if len(volumes) >= 20 else False
                nearest_res = max([lvl for lvl in sr_levels if lvl < last_price], default=None)
                if impulse > 0.08 and vol_spike and nearest_res is not None:
                    diff = last_price - nearest_res
                    diff_pct = (diff / nearest_res) * 100
                    signal = f"⚠️ Pre-top detected: можливий short біля {nearest_res:.4f}\n📊 Ринкова: {last_price:.4f} | Відрив: {diff:+.4f} ({diff_pct:+.2f}%)"

                if signal:
                    signals.append(f"<b>{symbol}</b>\n{signal}")

            except Exception as e:
                print(f"[ERROR] Обробка символу {symbol}: {e}")
                continue

        if not signals:
            bot.send_message(message.chat.id, "ℹ️ Жодних сигналів не знайдено.")
        else:
            text = "<b>Smart Auto S/R Signals</b>\n\n" + "\n\n".join(signals)
            bot.send_message(message.chat.id, text, parse_mode="HTML")

    except Exception as e:
        bot.send_message(message.chat.id, f"❌ Error: {e}")

# ================== Webhook route ==================
@app.route(WEBHOOK_URL_PATH, methods=['POST'])
def telegram_webhook():
    json_data = request.get_json()
    if json_data:
        update = types.Update.de_json(json_data)
        print("[UPDATE] Надійшов апдейт:", update)
        bot.process_new_updates([update])
    return "", 200

@app.route("/", methods=['GET'])
def index():
    return "Bot is running ✅", 200

# ================== Setup webhook (без before_first_request) ==================
def setup_webhook():
    try:
        bot.remove_webhook()
        bot.set_webhook(url=WEBHOOK_URL_BASE + WEBHOOK_URL_PATH)
        print(f"[INFO] Webhook встановлено: {WEBHOOK_URL_BASE + WEBHOOK_URL_PATH}")
    except Exception as e:
        print(f"[ERROR] Не вдалося встановити webhook: {e}")

setup_webhook()

# ================== Run app ==================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)