import os
import numpy as np
import requests
from flask import Flask, request
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

# =======================
# Telegram bot setup
# =======================
BOT_TOKEN = os.environ.get("BOT_TOKEN")
WEBHOOK_PATH = "/webhook"
WEBHOOK_URL = f"{os.environ.get('RENDER_EXTERNAL_URL')}{WEBHOOK_PATH}"

app = Flask(__name__)
telegram_app = ApplicationBuilder().token(BOT_TOKEN).build()

# =======================
# Binance API
# =======================
BINANCE_API_KEY = os.environ.get("BINANCE_API_KEY")
BINANCE_API_SECRET = os.environ.get("BINANCE_API_SECRET")
BINANCE_BASE = "https://api.binance.com/api/v3"

def get_klines(symbol, interval="1h", limit=200):
    url = f"{BINANCE_BASE}/klines?symbol={symbol}&interval={interval}&limit={limit}"
    response = requests.get(url)
    data = response.json()
    if not data or isinstance(data, dict) and data.get("code"):
        return None
    # повертаємо словник зі списками
    return {
        "c": [float(d[4]) for d in data],  # close
        "h": [float(d[2]) for d in data],  # high
        "l": [float(d[3]) for d in data],  # low
        "v": [float(d[5]) for d in data],  # volume
    }

# =======================
# Support/Resistance
# =======================
def find_support_resistance(prices, window=20, delta=0.005):
    sr_levels = []
    for i in range(window, len(prices)-window):
        local_max = max(prices[i-window:i+window+1])
        local_min = min(prices[i-window:i+window+1])
        if prices[i] == local_max:
            if all(abs(prices[i]-lvl)/lvl > delta for lvl in sr_levels):
                sr_levels.append(prices[i])
        elif prices[i] == local_min:
            if all(abs(prices[i]-lvl)/lvl > delta for lvl in sr_levels):
                sr_levels.append(prices[i])
    return sorted(sr_levels)

# =======================
# Commands
# =======================
async def smart_sr(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("⚠️ Використання: /smart_sr BTCUSDT")
        return

    symbol = context.args[0].upper()
    df = get_klines(symbol)
    if not df:
        await update.message.reply_text(f"❌ Дані для {symbol} недоступні")
        return

    closes = np.array(df['c'], dtype=float)
    volumes = np.array(df['v'], dtype=float)

    sr_levels = find_support_resistance(closes)
    last_price = closes[-1]
    signal = "ℹ️ Патерн не знайдено"

    for lvl in sr_levels:
        if last_price > lvl * 1.01:
            signal = f"🚀 LONG breakout: ціна пробила опір {lvl:.4f}"
        elif last_price < lvl * 0.99:
            signal = f"⚡ SHORT breakout: ціна пробила підтримку {lvl:.4f}"

    # Pre-top / pump
    impulse = (closes[-1] - closes[-4])/closes[-4] if len(closes)>=4 else 0
    vol_spike = volumes[-1] > 1.5 * np.mean(volumes[-20:]) if len(volumes)>=20 else False
    nearest_resistance = max([lvl for lvl in sr_levels if lvl < last_price], default=None)
    if impulse > 0.08 and vol_spike and nearest_resistance:
        signal += f"\n⚠️ Pre-top detected: можливий short біля {nearest_resistance:.4f}"

    await update.message.reply_text(f"{symbol} — Smart S/R Analysis\n\n{signal}")


async def smart_auto(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Отримуємо 24h тикери з Binance
    try:
        data = requests.get(f"{BINANCE_BASE}/ticker/24hr").json()
    except:
        await update.message.reply_text("❌ Не вдалося отримати дані Binance")
        return

    symbols = [
        d for d in data
        if d["symbol"].endswith("USDT") and float(d["quoteVolume"]) > 5_000_000
    ]

    # сортуємо за % зміни ціни
    symbols = sorted(symbols, key=lambda x: abs(float(x["priceChangePercent"])), reverse=True)[:30]
    signals = []

    for s in symbols:
        symbol = s["symbol"]
        df = get_klines(symbol)
        if not df:
            continue

        closes = np.array(df["c"], dtype=float)
        volumes = np.array(df["v"], dtype=float)
        sr_levels = find_support_resistance(closes)
        last_price = closes[-1]
        signal = None

        for lvl in sr_levels:
            if last_price > lvl * 1.01:
                signal = f"🚀 LONG breakout: {lvl:.4f}"
                break
            elif last_price < lvl * 0.99:
                signal = f"⚡ SHORT breakout: {lvl:.4f}"
                break

        # Pre-top
        impulse = (closes[-1] - closes[-4])/closes[-4] if len(closes)>=4 else 0
        vol_spike = volumes[-1] > 1.5*np.mean(volumes[-20:]) if len(volumes)>=20 else False
        nearest_res = max([lvl for lvl in sr_levels if lvl < last_price], default=None)
        if impulse > 0.08 and vol_spike and nearest_res:
            signal = f"⚠️ Pre-top: можливий short біля {nearest_res:.4f}"

        if signal:
            signals.append(f"{symbol}: {signal}")

    if not signals:
        await update.message.reply_text("ℹ️ Жодних сигналів не знайдено.")
    else:
        await update.message.reply_text("\n".join(signals))


# =======================
# Register handlers
# =======================
telegram_app.add_handler(CommandHandler("smart_sr", smart_sr))
telegram_app.add_handler(CommandHandler("smart_auto", smart_auto))

# =======================
# Flask webhook
# =======================
@app.route(WEBHOOK_PATH, methods=["POST"])
def webhook():
    update = Update.de_json(request.get_json(force=True), telegram_app.bot)
    telegram_app.create_task(telegram_app.process_update(update))
    return "OK"

# =======================
# Run Flask + set webhook
# =======================
if __name__ == "__main__":
    import asyncio
    asyncio.run(telegram_app.bot.set_webhook(WEBHOOK_URL))
    print(f"🌍 Webhook встановлено: {WEBHOOK_URL}")

    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))