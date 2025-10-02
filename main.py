import os
import asyncio
import logging
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from flask import Flask, request, jsonify
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
import requests

# ------------------------------------------------------------------------------
# Логи
# ------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("telegram-bot")

# ------------------------------------------------------------------------------
# ENV
# ------------------------------------------------------------------------------
BOT_TOKEN = os.getenv("BOT_TOKEN")
APP_URL = os.getenv("APP_URL", "https://crypto-tg-bot-vl7x.onrender.com")  # твій рендер URL

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN не задано у змінних оточення")

# ------------------------------------------------------------------------------
# Flask та Bot
# ------------------------------------------------------------------------------
app = Flask(__name__)
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

# ------------------------------------------------------------------------------
# Функції для памп-дамп сигналів
# ------------------------------------------------------------------------------
def get_klines(symbol: str, interval="1h", limit=200):
    """
    Отримуємо історію з Binance (можна змінити під свій API)
    """
    try:
        url = "https://api.binance.com/api/v3/klines"
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        data = requests.get(url, params=params, timeout=10).json()
        if not data:
            return None
        return {
            "o": [float(x[1]) for x in data],
            "h": [float(x[2]) for x in data],
            "l": [float(x[3]) for x in data],
            "c": [float(x[4]) for x in data],
            "v": [float(x[5]) for x in data],
        }
    except Exception as e:
        logger.error(f"get_klines error: {e}")
        return None

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

def plot_candles(symbol, interval="1h", limit=100):
    df = get_klines(symbol, interval, limit)
    if not df:
        return None
    closes = df["c"]
    plt.figure(figsize=(8,4))
    plt.plot(closes, label=symbol)
    plt.title(f"{symbol} - Last {limit} candles")
    plt.xlabel("Candle")
    plt.ylabel("Price")
    plt.grid(True)
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()
    return buf

# ------------------------------------------------------------------------------
# Команди бота
# ------------------------------------------------------------------------------
@dp.message(Command("start"))
async def start_cmd(message: types.Message):
    await message.answer(
        "Привіт! 👋\n"
        "Команди:\n"
        "• /smart_sr SYMBOL — аналіз S/R рівнів\n"
        "• /smart_auto — топ сигналів памп-дамп"
    )

@dp.message(Command("smart_sr"))
async def smart_sr_cmd(message: types.Message):
    parts = message.text.split()
    if len(parts) < 2:
        return await message.reply("⚠️ Використання: /smart_sr BTCUSDT")
    symbol = parts[1].upper()
    df = get_klines(symbol, interval="1h", limit=200)
    if not df:
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

    # pre-top / pump
    impulse = (closes[-1] - closes[-4]) / closes[-4] if len(closes) >= 4 else 0
    vol_spike = volumes[-1] > 1.5 * np.mean(volumes[-20:]) if len(volumes) >= 20 else False
    nearest_resistance = max([lvl for lvl in sr_levels if lvl < last_price], default=None)
    if impulse > 0.08 and vol_spike and nearest_resistance is not None:
        signal += f"\n⚠️ Pre-top detected: можливий short біля {nearest_resistance:.4f}"

    img = plot_candles(symbol)
    if img:
        await bot.send_photo(message.chat.id, img, caption=f"<b>{symbol} — Smart S/R Analysis</b>\n\n{signal}", parse_mode="HTML")
    else:
        await message.answer(signal)

@dp.message(Command("smart_auto"))
async def smart_auto_cmd(message: types.Message):
    try:
        url = "https://api.binance.com/api/v3/ticker/24hr"
        data = requests.get(url).json()
        symbols = [d for d in data if d["symbol"].endswith("USDT") and float(d["quoteVolume"]) > 5_000_000]
        symbols = sorted(symbols, key=lambda x: abs(float(x["priceChangePercent"])), reverse=True)[:30]

        signals = []
        for s in symbols:
            symbol = s["symbol"]
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
                    signal = f"🚀 LONG breakout: пробито опір {lvl:.4f}"
                    break
                elif last_price < lvl * 0.99:
                    signal = f"⚡ SHORT breakout: пробито підтримку {lvl:.4f}"
                    break

            impulse = (closes[-1]-closes[-4])/closes[-4] if len(closes)>=4 else 0
            vol_spike = volumes[-1] > 1.5 * np.mean(volumes[-20:]) if len(volumes)>=20 else False
            nearest_res = max([lvl for lvl in sr_levels if lvl<last_price], default=None)
            if impulse>0.08 and vol_spike and nearest_res:
                signal = f"⚠️ Pre-top: можливий short біля {nearest_res:.4f}"

            if signal:
                signals.append(f"<b>{symbol}</b>\n{signal}")

        if not signals:
            await message.answer("ℹ️ Сигналів не знайдено")
        else:
            await message.answer("<b>Smart Auto Signals</b>\n\n" + "\n\n".join(signals), parse_mode="HTML")
    except Exception as e:
        await message.answer(f"❌ Error: {e}")

# ------------------------------------------------------------------------------
# Flask webhook
# ------------------------------------------------------------------------------
@app.route("/webhook", methods=["POST"])
def webhook():
    try:
        update = types.Update.model_validate(request.json)
        asyncio.get_event_loop().create_task(dp.feed_update(bot, update))
        return jsonify({"ok": True})
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/", methods=["GET", "HEAD"])
def root():
    return "Bot is running ✅"

# ------------------------------------------------------------------------------
# Запуск
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    async def set_webhook():
        try:
            await bot.delete_webhook(drop_pending_updates=True)
            await bot.set_webhook(f"{APP_URL}/webhook")
            logger.info(f"🌍 Webhook встановлено: {APP_URL}/webhook")
        except Exception as e:
            logger.error(f"Не вдалося встановити вебхук: {e}")

    asyncio.run(set_webhook())
    port = int(os.getenv("PORT", 5000))
    uvicorn.run(app, host="0.0.0.0", port=port)