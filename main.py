import os
import asyncio
import io
import requests
import numpy as np
import matplotlib.pyplot as plt

from flask import Flask, request
from aiogram import Bot, Dispatcher, types
from aiogram.client.default import DefaultBotProperties
from aiogram.filters import Command

# ================== CONFIG ==================
BOT_TOKEN = os.getenv("BOT_TOKEN")
WEBHOOK_URL = os.getenv("WEBHOOK_URL")  # наприклад: https://crypto-tg-bot-xxxx.onrender.com

if not BOT_TOKEN or not WEBHOOK_URL:
    raise ValueError("❌ BOT_TOKEN або WEBHOOK_URL не знайдено у змінних оточення")

bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode="HTML"))
dp = Dispatcher()
app = Flask(__name__)

# ================== HELPERS ==================
def get_klines(symbol: str, interval="1h", limit=200):
    """
    Отримує свічки з Binance API
    """
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    data = requests.get(url, timeout=10).json()
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

def plot_candles(symbol, interval="1h", limit=100):
    df = get_klines(symbol, interval=interval, limit=limit)
    if not df:
        return None
    closes = df["c"]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(closes, label="Close Price")
    ax.set_title(f"{symbol} — {interval} candles")
    ax.legend()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)
    return buf

def find_support_resistance(prices, window=20, delta=0.005):
    """
    Автоматично знаходить локальні S/R рівні
    """
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

# ================== HANDLERS ==================
@dp.message(Command("start"))
async def start_handler(message: types.Message):
    await message.answer("👋 Привіт! Я бот для аналізу памп-дамп ситуацій.\n\n"
                         "Доступні команди:\n"
                         "• /smart_sr BTCUSDT — аналіз S/R\n"
                         "• /smart_auto — авто-сканер топ монет")

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

        # Перевірка pump / pre-top
        impulse = (closes[-1] - closes[-4]) / closes[-4] if len(closes) >= 4 else 0
        vol_spike = volumes[-1] > 1.5 * np.mean(volumes[-20:]) if len(volumes) >= 20 else False
        nearest_res = max([lvl for lvl in sr_levels if lvl < last_price], default=None)
        if impulse > 0.08 and vol_spike and nearest_res is not None:
            signal += f"\n⚠️ Pre-top detected: можливий short біля {nearest_res:.4f}"

        # Відправляємо графік
        img = plot_candles(symbol, interval="1h", limit=100)
        if img:
            await bot.send_photo(message.chat.id, img,
                                 caption=f"<b>{symbol} — Smart S/R Analysis</b>\n\n{signal}",
                                 parse_mode="HTML")
        else:
            await message.answer(signal)

    except Exception as e:
        await message.answer(f"❌ Error: {e}")

@dp.message(Command("smart_auto"))
async def smart_auto_handler(message: types.Message):
    try:
        url = "https://api.binance.com/api/v3/ticker/24hr"
        data = requests.get(url, timeout=10).json()

        # тільки USDT-пари з великим об'ємом
        symbols = [
            d for d in data
            if d["symbol"].endswith("USDT") and float(d["quoteVolume"]) > 5_000_000
        ]

        # сортуємо за % зміни
        symbols = sorted(
            symbols,
            key=lambda x: abs(float(x["priceChangePercent"])),
            reverse=True
        )

        top_symbols = [s["symbol"] for s in symbols[:15]]

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
                    diff = last_price - lvl
                    diff_pct = (diff / lvl) * 100

                    if last_price > lvl * 1.01:
                        signal = (f"🚀 LONG breakout: {lvl:.4f}\n"
                                  f"📊 {last_price:.4f} | Δ {diff:+.4f} ({diff_pct:+.2f}%)")
                        break
                    elif last_price < lvl * 0.99:
                        signal = (f"⚡ SHORT breakout: {lvl:.4f}\n"
                                  f"📊 {last_price:.4f} | Δ {diff:+.4f} ({diff_pct:+.2f}%)")
                        break

                impulse = (closes[-1] - closes[-4]) / closes[-4] if len(closes) >= 4 else 0
                vol_spike = volumes[-1] > 1.5 * np.mean(volumes[-20:]) if len(volumes) >= 20 else False
                nearest_res = max([lvl for lvl in sr_levels if lvl < last_price], default=None)
                if impulse > 0.08 and vol_spike and nearest_res is not None:
                    diff = last_price - nearest_res
                    diff_pct = (diff / nearest_res) * 100
                    signal = (f"⚠️ Pre-top: short біля {nearest_res:.4f}\n"
                              f"📊 {last_price:.4f} | Δ {diff:+.4f} ({diff_pct:+.2f}%)")

                if signal:
                    signals.append(f"<b>{symbol}</b>\n{signal}")

            except Exception:
                continue

        if not signals:
            await message.answer("ℹ️ Жодних сигналів не знайдено.")
        else:
            text = "<b>Smart Auto S/R Signals</b>\n\n" + "\n\n".join(signals)
            await message.answer(text, parse_mode="HTML")

    except Exception as e:

# ================== FLASK WEBHOOK ==================
@app.route("/webhook", methods=["POST"])
async def webhook():
    update = types.Update.model_validate(request.json)
    await dp.feed_update(bot, update)
    return "ok"

# ================== STARTUP ==================
async def on_startup():
    # ❌ Прибираємо старий webhook
    await bot.delete_webhook(drop_pending_updates=True)
    # ✅ Ставимо новий з простим шляхом /webhook
    await bot.set_webhook(f"{WEBHOOK_URL}/webhook")
    print(f"🌍 Webhook встановлено: {WEBHOOK_URL}/webhook")

def main():
    loop = asyncio.get_event_loop()
    loop.run_until_complete(on_startup())
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))

if __name__ == "__main__":
    main()