# main.py
import os
import numpy as np
import requests
from flask import Flask, request
import matplotlib.pyplot as plt
from io import BytesIO
from telegram import Update, Bot
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

BOT_TOKEN = os.getenv("BOT_TOKEN")
WEBHOOK_URL = os.getenv("WEBHOOK_URL")  # https://your-app.onrender.com/webhook

if not BOT_TOKEN or not WEBHOOK_URL:
    raise RuntimeError("BOT_TOKEN і WEBHOOK_URL повинні бути задані у .env")

app = Flask(__name__)
bot = Bot(BOT_TOKEN)

# ------------------ UTILS ------------------
def get_klines(symbol, interval="1h", limit=200):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    try:
        data = requests.get(url, params=params, timeout=10).json()
        return {
            "c": [float(d[4]) for d in data],
            "h": [float(d[2]) for d in data],
            "l": [float(d[3]) for d in data],
            "v": [float(d[5]) for d in data],
        }
    except:
        return {}

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
    df = get_klines(symbol, interval, limit)
    if not df or len(df["c"])==0:
        return None
    plt.figure(figsize=(10,4))
    plt.plot(df["c"], label="Close")
    plt.title(f"{symbol} Candles")
    plt.legend()
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()
    return buf

# ------------------ COMMAND HANDLERS ------------------
async def smart_sr(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    parts = message.text.split()
    if len(parts) < 2:
        await message.reply_text("⚠️ Використання: /smart_sr BTCUSDT")
        return
    symbol = parts[1].upper()
    df = get_klines(symbol)
    if not df or len(df["c"])==0:
        await message.reply_text(f"❌ Дані для {symbol} недоступні")
        return

    closes = np.array(df["c"], dtype=float)
    volumes = np.array(df["v"], dtype=float)
    sr_levels = find_support_resistance(closes)
    last_price = closes[-1]
    signal = "ℹ️ Патерн не знайдено"
    for lvl in sr_levels:
        if last_price > lvl*1.01:
            signal = f"🚀 LONG breakout: ціна пробила опір {lvl:.4f}"
        elif last_price < lvl*0.99:
            signal = f"⚡ SHORT breakout: ціна пробила підтримку {lvl:.4f}"

    impulse = (closes[-1]-closes[-4])/closes[-4] if len(closes)>=4 else 0
    vol_spike = volumes[-1] > 1.5*np.mean(volumes[-20:]) if len(volumes)>=20 else False
    nearest_res = max([lvl for lvl in sr_levels if lvl < last_price], default=None)
    if impulse>0.08 and vol_spike and nearest_res is not None:
        signal += f"\n⚠️ Pre-top detected: можливий short біля {nearest_res:.4f}"

    img_buf = plot_candles(symbol)
    if img_buf:
        await bot.send_photo(chat_id=message.chat.id, photo=img_buf,
                             caption=f"<b>{symbol} — Smart S/R Analysis</b>\n\n{signal}",
                             parse_mode="HTML")
    else:
        await message.reply_text(signal)

async def smart_auto(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message
    try:
        url = "https://api.binance.com/api/v3/ticker/24hr"
        data = requests.get(url, timeout=10).json()
        symbols = [d for d in data if d["symbol"].endswith("USDT") and float(d["quoteVolume"])>5_000_000]
        symbols = sorted(symbols, key=lambda x: abs(float(x["priceChangePercent"])), reverse=True)
        top_symbols = [s["symbol"] for s in symbols[:30]]

        signals = []
        for symbol in top_symbols:
            df = get_klines(symbol)
            if not df or len(df["c"])<50:
                continue
            closes = np.array(df["c"], dtype=float)
            volumes = np.array(df["v"], dtype=float)
            sr_levels = find_support_resistance(closes)
            last_price = closes[-1]
            signal = None
            for lvl in sr_levels:
                if last_price>lvl*1.01:
                    diff_pct = (last_price-lvl)/lvl*100
                    signal = f"🚀 LONG breakout: ціна пробила опір {lvl:.4f} | Відрив {diff_pct:+.2f}%"
                    break
                elif last_price<lvl*0.99:
                    diff_pct = (last_price-lvl)/lvl*100
                    signal = f"⚡ SHORT breakout: ціна пробила підтримку {lvl:.4f} | Відрив {diff_pct:+.2f}%"
                    break
            impulse = (closes[-1]-closes[-4])/closes[-4] if len(closes)>=4 else 0
            vol_spike = volumes[-1]>1.5*np.mean(volumes[-20:]) if len(volumes)>=20 else False
            nearest_res = max([lvl for lvl in sr_levels if lvl < last_price], default=None)
            if impulse>0.08 and vol_spike and nearest_res is not None:
                diff_pct = (last_price-nearest_res)/nearest_res*100
                signal = f"⚠️ Pre-top: можливий short біля {nearest_res:.4f} | Відрив {diff_pct:+.2f}%"
            if signal:
                signals.append(f"<b>{symbol}</b>\n{signal}")

        if not signals:
            await message.reply_text("ℹ️ Жодних сигналів не знайдено.")
        else:
            text = "<b>Smart Auto S/R Signals</b>\n\n" + "\n\n".join(signals)
            await message.reply_text(text, parse_mode="HTML")
    except Exception as e:
        await message.reply_text(f"❌ Error: {e}")

# ------------------ APPLICATION ------------------
application = ApplicationBuilder().token(BOT_TOKEN).build()
application.add_handler(CommandHandler("smart_sr", smart_sr))
application.add_handler(CommandHandler("smart_auto", smart_auto))

# ------------------ FLASK WEBHOOK ------------------
@app.route("/webhook", methods=["POST"])
def webhook():
    update = Update.de_json(request.get_json(force=True), bot)
    loop = asyncio.get_event_loop()
    if loop.is_running():
        # якщо вже є loop (Gunicorn), створюємо таск
        asyncio.create_task(application.process_update(update))
    else:
        loop.run_until_complete(application.process_update(update))
    return "OK"

@app.route("/", methods=["GET", "HEAD"])
def root():
    return {"message": "Bot is running"}

# ------------------ START ------------------
if __name__ == "__main__":
    bot.delete_webhook()
    bot.set_webhook(WEBHOOK_URL)
    print(f"🌍 Webhook встановлено: {WEBHOOK_URL}")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))