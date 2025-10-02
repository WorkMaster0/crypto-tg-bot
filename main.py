# main.py
import os
import io
import requests
import numpy as np
from flask import Flask, request, jsonify
from telebot import TeleBot, types
import matplotlib.pyplot as plt

# ---------------- ENV ----------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")  
PORT = int(os.getenv("PORT", "5000"))
WEBHOOK_URL = os.getenv("WEBHOOK_URL")  

if not TELEGRAM_TOKEN or not CHAT_ID or not WEBHOOK_URL:
    raise ValueError("❌ TELEGRAM_TOKEN, CHAT_ID та WEBHOOK_URL повинні бути встановлені!")

bot = TeleBot(TELEGRAM_TOKEN)
app = Flask(__name__)

# ---------------- HELPERS ----------------
def get_klines(symbol, interval="1h", limit=200):
    url = f"https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()
    df = {
        "c": [float(x[4]) for x in data],
        "v": [float(x[5]) for x in data]
    }
    return df

def find_support_resistance(closes, window=20, delta=0.005):
    sr_levels = []
    for i in range(window, len(closes)):
        window_data = closes[i-window:i]
        min_val = np.min(window_data)
        max_val = np.max(window_data)
        if all(abs(min_val - x)/x < delta for x in window_data):
            sr_levels.append(min_val)
        if all(abs(max_val - x)/x < delta for x in window_data):
            sr_levels.append(max_val)
    return sr_levels

def plot_chart(closes, entry=None, tp=None, sl=None):
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(closes, label="Close", color="blue")
    if entry: ax.axhline(entry, color="orange", linestyle="--", label=f"Entry {entry:.4f}")
    if tp: ax.axhline(tp, color="green", linestyle="--", label=f"TP {tp:.4f}")
    if sl: ax.axhline(sl, color="red", linestyle="--", label=f"SL {sl:.4f}")
    ax.legend()
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="PNG")
    plt.close(fig)
    buf.seek(0)
    return buf

# ---------------- COMMAND ----------------
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

        for symbol in top_symbols:
            try:
                df = get_klines(symbol)
                if not df or len(df.get("c", [])) < 50:
                    continue

                closes = np.array(df["c"])
                volumes = np.array(df["v"])
                sr_levels = find_support_resistance(closes)
                last_price = closes[-1]

                signal = None
                for lvl in sr_levels:
                    if last_price > lvl * 1.01:
                        signal = f"🚀 LONG breakout: ціна пробила опір {lvl:.4f}\n📊 Ринкова: {last_price:.4f}"
                        break
                    elif last_price < lvl * 0.99:
                        signal = f"⚡ SHORT breakout: ціна пробила підтримку {lvl:.4f}\n📊 Ринкова: {last_price:.4f}"
                        break

                impulse = (closes[-1] - closes[-4])/closes[-4] if len(closes) >= 4 else 0
                vol_spike = volumes[-1] > 1.5 * np.mean(volumes[-20:]) if len(volumes) >= 20 else False
                nearest_res = max([lvl for lvl in sr_levels if lvl < last_price], default=None)
                if impulse > 0.08 and vol_spike and nearest_res is not None:
                    signal = f"⚠️ Pre-top detected: можливий short біля {nearest_res:.4f}\n📊 Ринкова: {last_price:.4f}"

                if signal:
                    chart_buf = plot_chart(closes, entry=last_price)
                    bot.send_photo(message.chat.id, chart_buf, caption=f"<b>{symbol}</b>\n{signal}", parse_mode="HTML")
            except Exception:
                continue

    except Exception as e:
        bot.send_message(message.chat.id, f"❌ Error: {e}")

# ---------------- FLASK WEBHOOK ----------------
@app.route('/webhook', methods=['POST'])
def webhook():
    json_str = request.get_data().decode('utf-8')
    update = types.Update.de_json(json_str)
    bot.process_new_updates([update])
    return jsonify({"status": "ok"})

@app.route('/')
def index():
    return "Bot is running"

# ---------------- SETUP WEBHOOK ----------------
@app.before_serving
def setup_webhook():
    bot.remove_webhook()
    bot.set_webhook(url=WEBHOOK_URL)
    print(f"Webhook set to {WEBHOOK_URL}")

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)