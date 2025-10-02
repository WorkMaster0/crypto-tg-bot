import os
import logging
import numpy as np
import requests
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, types
from aiogram.client.default import DefaultBotProperties
from aiogram.filters import Command
from flask import Flask, request

load_dotenv()
logging.basicConfig(level=logging.INFO)

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
WEBHOOK_URL = os.getenv("WEBHOOK_URL")  # твій URL на Render
if not BOT_TOKEN or not WEBHOOK_URL:
    raise ValueError("❌ TELEGRAM_BOT_TOKEN або WEBHOOK_URL не знайдено у .env")

# ✅ правильна ініціалізація для aiogram v3.7+
bot = Bot(
    token=BOT_TOKEN,
    default=DefaultBotProperties(parse_mode="HTML")
)
dp = Dispatcher()

# ---------- BINANCE API ----------
def get_klines(symbol, interval="1h", limit=200):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    resp = requests.get(url, params=params)
    if resp.status_code != 200:
        return None
    data = resp.json()
    if not isinstance(data, list):
        return None

    return {
        "t": [x[0] for x in data],
        "o": [float(x[1]) for x in data],
        "h": [float(x[2]) for x in data],
        "l": [float(x[3]) for x in data],
        "c": [float(x[4]) for x in data],
        "v": [float(x[5]) for x in data],
    }

# ---------- Логіка S/R ----------
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

# ---------- /smart_sr ----------
@dp.message(Command("smart_sr"))
async def smart_sr_handler(message: types.Message):
    parts = message.text.split()
    if len(parts) < 2:
        return await message.answer("⚠️ Використання: /smart_sr BTCUSDT")

    symbol = parts[1].upper()
    try:
        df = get_klines(symbol, interval="1h", limit=200)
        if not df or len(df["c"]) == 0:
            return await message.answer(f"❌ Дані для {symbol} недоступні")

        closes = np.array(df["c"], dtype=float)
        volumes = np.array(df["v"], dtype=float)
        sr_levels = find_support_resistance(closes, window=20, delta=0.005)
        last_price = closes[-1]

        signal = "ℹ️ Патерн не знайдено"
        for lvl in sr_levels:
            if last_price > lvl * 1.01:
                signal = f"🚀 LONG breakout: ціна пробила опір {lvl:.4f}"
            elif last_price < lvl * 0.99:
                signal = f"⚡ SHORT breakout: ціна пробила підтримку {lvl:.4f}"

        impulse = (closes[-1] - closes[-4]) / closes[-4] if len(closes) >= 4 else 0
        vol_spike = volumes[-1] > 1.5 * np.mean(volumes[-20:]) if len(volumes) >= 20 else False
        nearest_res = max([lvl for lvl in sr_levels if lvl < last_price], default=None)

        if impulse > 0.08 and vol_spike and nearest_res is not None:
            signal += f"\n⚠️ Pre-top detected: можливий short біля {nearest_res:.4f}"

        await message.answer(f"<b>{symbol} — Smart S/R Analysis</b>\n\n{signal}")

    except Exception as e:
        await message.answer(f"❌ Error: {e}")

# ---------- /smart_auto ----------
@dp.message(Command("smart_auto"))
async def smart_auto_handler(message: types.Message):
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
                if not df or len(df["c"]) < 50:
                    continue

                closes = np.array(df["c"], dtype=float)
                volumes = np.array(df["v"], dtype=float)
                sr_levels = find_support_resistance(closes, window=20, delta=0.005)
                last_price = closes[-1]

                signal = None
                for lvl in sr_levels:
                    diff = last_price - lvl
                    diff_pct = (diff / lvl) * 100
                    if last_price > lvl * 1.01:
                        signal = f"🚀 {symbol} LONG breakout\n📊 Ціна: {last_price:.4f} | Опір: {lvl:.4f} ({diff_pct:+.2f}%)"
                        break
                    elif last_price < lvl * 0.99:
                        signal = f"⚡ {symbol} SHORT breakout\n📊 Ціна: {last_price:.4f} | Підтримка: {lvl:.4f} ({diff_pct:+.2f}%)"
                        break

                impulse = (closes[-1] - closes[-4]) / closes[-4] if len(closes) >= 4 else 0
                vol_spike = volumes[-1] > 1.5 * np.mean(volumes[-20:]) if len(volumes) >= 20 else False
                nearest_res = max([lvl for lvl in sr_levels if lvl < last_price], default=None)

                if impulse > 0.08 and vol_spike and nearest_res is not None:
                    diff = last_price - nearest_res
                    diff_pct = (diff / nearest_res) * 100
                    signal = f"⚠️ {symbol} Pre-top: можливий short біля {nearest_res:.4f}\n📊 Ціна: {last_price:.4f} ({diff_pct:+.2f}%)"

                if signal:
                    signals.append(signal)
            except Exception:
                continue

        if signals:
            await message.answer("<b>Smart Auto S/R Signals</b>\n\n" + "\n\n".join(signals))
        else:
            await message.answer("ℹ️ Жодних сигналів не знайдено.")

    except Exception as e:
        await message.answer(f"❌ Error: {e}")

# ---------- Flask webhook ----------
app = Flask(__name__)

@app.route("/webhook", methods=["POST"])
async def webhook():
    update = types.Update.model_validate(request.json)
    await dp.feed_update(bot, update)
    return "ok"

@app.route("/")
def index():
    return "✅ Bot is running with webhook"

async def on_startup():
    await bot.set_webhook(f"{WEBHOOK_URL}/webhook")

if __name__ == "__main__":
    import asyncio
    loop = asyncio.get_event_loop()
    loop.run_until_complete(on_startup())
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))