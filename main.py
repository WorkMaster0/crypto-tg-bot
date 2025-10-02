import os
import asyncio
import logging
import numpy as np
import requests
from flask import Flask, request

from aiogram import Bot, Dispatcher, types
from aiogram.client.default import DefaultBotProperties
from aiogram.filters import Command

# ----------------- ЛОГІ -----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pump-bot")

# ----------------- ENV -----------------
BOT_TOKEN = os.getenv("BOT_TOKEN")
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "https://crypto-tg-bot-vl7x.onrender.com")
PORT = int(os.getenv("PORT", 5000))

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN not set in .env")

# ----------------- INIT -----------------
bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode="HTML"))
dp = Dispatcher()
app = Flask(__name__)

# ----------------- ФУНКЦІЇ -----------------
def get_klines(symbol, interval="1h", limit=200):
    """ Отримання свічок з Binance API """
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = requests.get(url, params=params)
    data = r.json()
    if not isinstance(data, list):
        return None
    closes = [float(c[4]) for c in data]
    highs = [float(c[2]) for c in data]
    lows = [float(c[3]) for c in data]
    volumes = [float(c[5]) for c in data]
    return {"c": closes, "h": highs, "l": lows, "v": volumes}

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

# ----------------- КОМАНДИ -----------------
@dp.message(Command("smart_sr"))
async def smart_sr_handler(message: types.Message):
    parts = message.text.split()
    if len(parts) < 2:
        return await message.answer("⚠️ Використання: /smart_sr BTCUSDT")

    symbol = parts[1].upper()
    try:
        df = get_klines(symbol, interval="1h", limit=200)
        if not df or len(df.get("c", [])) == 0:
            return await message.answer(f"❌ Дані для {symbol} недоступні")

        closes = np.array(df['c'], dtype=float)
        volumes = np.array(df['v'], dtype=float)
        sr_levels = find_support_resistance(closes, window=20, delta=0.005)
        last_price = closes[-1]

        # Перевірка breakout
        signal = "ℹ️ Патерн не знайдено"
        for lvl in sr_levels:
            if last_price > lvl * 1.01:
                signal = f"🚀 LONG breakout: ціна пробила опір {lvl:.4f}"
            elif last_price < lvl * 0.99:
                signal = f"⚡ SHORT breakout: ціна пробила підтримку {lvl:.4f}"

        # Перевірка pre-top
        impulse = (closes[-1] - closes[-4]) / closes[-4] if len(closes) >= 4 else 0
        vol_spike = volumes[-1] > 1.5 * np.mean(volumes[-20:]) if len(volumes) >= 20 else False
        nearest_res = max([lvl for lvl in sr_levels if lvl < last_price], default=None)
        if impulse > 0.08 and vol_spike and nearest_res is not None:
            signal += f"\n⚠️ Pre-top detected: можливий short біля {nearest_res:.4f}"

        await message.answer(f"<b>{symbol} — Smart S/R Analysis</b>\n\n{signal}", parse_mode="HTML")

    except Exception as e:
        await message.answer(f"❌ Error: {e}")

@dp.message(Command("smart_auto"))
async def smart_auto_handler(message: types.Message):
    try:
        url = "https://api.binance.com/api/v3/ticker/24hr"
        data = requests.get(url).json()

        # відбираємо USDT-пари з об'ємом
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
                if not df or len(df.get("c", [])) < 50:
                    continue

                closes = np.array(df["c"], dtype=float)
                volumes = np.array(df["v"], dtype=float)
                sr_levels = find_support_resistance(closes, window=20, delta=0.005)
                last_price = closes[-1]

                signal = None
                for lvl in sr_levels:
                    if last_price > lvl * 1.01:
                        signal = f"🚀 LONG breakout: пробив {lvl:.4f}"
                        break
                    elif last_price < lvl * 0.99:
                        signal = f"⚡ SHORT breakout: пробив {lvl:.4f}"
                        break

                impulse = (closes[-1] - closes[-4]) / closes[-4] if len(closes) >= 4 else 0
                vol_spike = volumes[-1] > 1.5 * np.mean(volumes[-20:]) if len(volumes) >= 20 else False
                nearest_res = max([lvl for lvl in sr_levels if lvl < last_price], default=None)
                if impulse > 0.08 and vol_spike and nearest_res is not None:
                    signal = f"⚠️ Pre-top detected: можливий short біля {nearest_res:.4f}"

                if signal:
                    signals.append(f"<b>{symbol}</b>\n{signal}")

            except Exception:
                continue

        if signals:
            await message.answer("<b>Smart Auto S/R Signals</b>\n\n" + "\n\n".join(signals))
        else:
            await message.answer("ℹ️ Жодних сигналів не знайдено.")

    except Exception as e:
        await message.answer(f"❌ Error: {e}")

# ----------------- FLASK ROUTES -----------------
@app.route("/", methods=["GET", "HEAD"])
def index():
    return "Bot is running!"

@app.route("/webhook", methods=["POST"])
async def webhook():
    update = types.Update.model_validate(request.json)
    await dp.feed_update(bot, update)
    return "ok"

# ----------------- STARTUP -----------------
async def on_startup():
    await bot.delete_webhook(drop_pending_updates=True)
    await bot.set_webhook(f"{WEBHOOK_URL}/webhook")
    print(f"🌍 Webhook встановлено: {WEBHOOK_URL}/webhook")

# ----------------- RUN -----------------
if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.create_task(on_startup())
    app.run(host="0.0.0.0", port=PORT)