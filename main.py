# main.py

import os
import requests
import numpy as np
import matplotlib.pyplot as plt
import io
from flask import Flask, request, send_file
import telebot

# ------------------ Налаштування ------------------
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
WEBHOOK_URL = os.environ.get("WEBHOOK_URL")  # https://yourapp.onrender.com/<TELEGRAM_TOKEN>

bot = telebot.TeleBot(TELEGRAM_TOKEN)
app = Flask(__name__)

# ------------------ Допоміжні функції ------------------

def get_klines(symbol, interval="1h", limit=200):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    data = requests.get(url).json()
    # повертаємо словник списків
    return {
        "o": [float(k[1]) for k in data],
        "h": [float(k[2]) for k in data],
        "l": [float(k[3]) for k in data],
        "c": [float(k[4]) for k in data],
        "v": [float(k[5]) for k in data]
    }

def find_support_resistance(prices, window=20, delta=0.005):
    levels = []
    for i in range(window, len(prices) - window):
        high_range = max(prices[i-window:i+window])
        low_range = min(prices[i-window:i+window])
        if abs(prices[i] - high_range) / high_range < delta:
            levels.append(prices[i])
        elif abs(prices[i] - low_range) / low_range < delta:
            levels.append(prices[i])
    return sorted(list(set(levels)))

def plot_candles(symbol, df):
    fig, ax = plt.subplots(figsize=(10,5))
    o, h, l, c = df["o"], df["h"], df["l"], df["c"]
    for i in range(len(c)):
        color = 'green' if c[i] >= o[i] else 'red'
        ax.plot([i, i], [l[i], h[i]], color='black')
        ax.add_patch(plt.Rectangle((i-0.3, min(o[i], c[i])), 0.6, abs(c[i]-o[i]), color=color))
    ax.set_title(symbol)
    ax.set_xlabel("Candle")
    ax.set_ylabel("Price")
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return buf

# ------------------ Команди бота ------------------

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

                # Pre-top / pump
                impulse = (closes[-1] - closes[-4]) / closes[-4] if len(closes) >= 4 else 0
                vol_spike = volumes[-1] > 1.5 * np.mean(volumes[-20:]) if len(volumes) >= 20 else False
                nearest_res = max([lvl for lvl in sr_levels if lvl < last_price], default=None)
                if impulse > 0.08 and vol_spike and nearest_res is not None:
                    diff = last_price - nearest_res
                    diff_pct = (diff / nearest_res) * 100
                    signal = f"⚠️ Pre-top detected: можливий short біля {nearest_res:.4f}\n📊 Ринкова: {last_price:.4f} | Відрив: {diff:+.4f} ({diff_pct:+.2f}%)"

                if signal:
                    signals.append(f"<b>{symbol}</b>\n{signal}")

                    # Надсилаємо графік свічок
                    img_buf = plot_candles(symbol, df)
                    bot.send_photo(message.chat.id, img_buf)

            except Exception:
                continue

        if not signals:
            bot.send_message(message.chat.id, "ℹ️ Жодних сигналів не знайдено.")
        else:
            text = "<b>Smart Auto S/R Signals</b>\n\n" + "\n\n".join(signals)
            bot.send_message(message.chat.id, text, parse_mode="HTML")

    except Exception as e:
        bot.send_message(message.chat.id, f"❌ Error: {e}")

# ------------------ Webhook ------------------

bot.remove_webhook()
bot.set_webhook(url=WEBHOOK_URL)
print(f"Webhook встановлено: {WEBHOOK_URL}")

@app.route(f"/{TELEGRAM_TOKEN}", methods=['POST'])
def webhook():
    json_str = request.get_data().decode('utf-8')
    update = telebot.types.Update.de_json(json_str)
    bot.process_new_updates([update])
    return '', 200

@app.route("/")
def index():
    return "Bot is running!"

# ------------------ Запуск локально ------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))