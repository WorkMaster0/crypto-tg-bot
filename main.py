import os
import asyncio
import aiohttp
import numpy as np
from flask import Flask, request
from aiogram import Bot, Dispatcher, types
import feedparser

# --- Environment Variables ---
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
WEBHOOK_URL = "https://dex-tg-bot.onrender.com/"
WEBHOOK_PATH = f"/webhook/{TELEGRAM_TOKEN}"
WEBHOOK_FULL_URL = WEBHOOK_URL + WEBHOOK_PATH

# --- Flask App ---
app = Flask(__name__)

# --- Telegram Bot ---
bot = Bot(token=TELEGRAM_TOKEN, parse_mode="HTML")
dp = Dispatcher()

dp.startup.register(lambda _: bot)  # реєстрація бота в dispatcher

# --- Live Market Bot with Social & Agents ---
class LiveMarketBot:
    def __init__(self, symbols=["BTCUSDT","ETHUSDT","BNBUSDT"], depth=20,
                 rss_urls=None, twitter_rss_urls=None):
        self.symbols = symbols
        self.depth = depth
        self.order_books = {s:{} for s in symbols}
        self.prev_imbalances = {s:0 for s in symbols}
        self.social_signal = 0
        self.rss_urls = rss_urls or ["https://t.me/s/public_crypto_channel?format=rss"]
        self.twitter_rss_urls = twitter_rss_urls or [
            "https://nitter.net/search/rss?q=%23BTC",
            "https://nitter.net/search/rss?q=%23ETH"
        ]

    async def fetch_order_book(self, symbol):
        url = f"https://api.binance.com/api/v3/depth?symbol={symbol}&limit={self.depth}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                self.order_books[symbol] = await resp.json()

    async def fetch_public_trades(self, symbol):
        url = f"https://api.binance.com/api/v3/trades?symbol={symbol}&limit=50"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                trades = await resp.json()
        whale_volume = sum([float(t['qty']) for t in trades if float(t['qty']) > 5])
        return whale_volume

    async def fetch_social_signal(self):
        sentiment_score = 0
        # Telegram RSS
        for url in self.rss_urls:
            feed = feedparser.parse(url)
            for entry in feed.entries[:5]:
                text = entry.title + " " + getattr(entry, 'summary', '')
                sentiment_score += self.analyze_sentiment(text)
        # Twitter RSS
        for url in self.twitter_rss_urls:
            feed = feedparser.parse(url)
            for entry in feed.entries[:5]:
                text = entry.title + " " + getattr(entry, 'summary', '')
                sentiment_score += self.analyze_sentiment(text)
        total_sources = len(self.rss_urls) + len(self.twitter_rss_urls)
        self.social_signal = max(min(sentiment_score / total_sources, 1), -1)

    def analyze_sentiment(self, text):
        fomo_keywords = ["pump", "moon", "bullish", "rise"]
        fud_keywords = ["dump", "bearish", "crash", "drop"]
        text_lower = text.lower()
        score = 0
        for word in fomo_keywords:
            if word in text_lower:
                score += 1
        for word in fud_keywords:
            if word in text_lower:
                score -= 1
        return score

    def calculate_imbalance(self, symbol, whale_volume=0):
        ob = self.order_books[symbol]
        if not ob: return 0
        bids = np.array([float(b[0])*float(b[1]) for b in ob['bids']])
        asks = np.array([float(a[0])*float(a[1]) for a in ob['asks']])
        imbalance = bids.sum() - asks.sum()
        imbalance += whale_volume * 0.5
        imbalance += self.social_signal * 5000
        return imbalance

    def generate_signal(self, symbol, imbalance):
        change = imbalance - self.prev_imbalances[symbol]
        self.prev_imbalances[symbol] = imbalance
        threshold = max(1000, abs(imbalance) * 0.05)
        ob = self.order_books[symbol]
        if imbalance > threshold and change > 0:
            entry = float(ob['bids'][0][0])
            return {
                'symbol': symbol,
                'action': 'BUY',
                'entry': entry,
                'take_profit': entry * 1.006,
                'stop_loss': entry * 0.995,
                'imbalance': imbalance,
                'social_signal': self.social_signal
            }
        elif imbalance < -threshold and change < 0:
            entry = float(ob['asks'][0][0])
            return {
                'symbol': symbol,
                'action': 'SELL',
                'entry': entry,
                'take_profit': entry * 0.994,
                'stop_loss': entry * 1.005,
                'imbalance': imbalance,
                'social_signal': self.social_signal
            }
        return None

    async def fetch_monthly_highlow(self, symbol):
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1d&limit=30"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                klines = await resp.json()
        if not klines:
            return None, None
        high = max([float(k[2]) for k in klines])
        low = min([float(k[3]) for k in klines])
        return high, low

# --- Telegram Commands ---
@dp.message(commands=["start"])
async def start_handler(message: types.Message):
    await message.reply("Живий ринок запущений! Очікуйте сигнали та аналіз топ токенів.")

@dp.message(commands=["status"])
async def status_handler(message: types.Message):
    await message.reply("Бот працює. Сигнали генеруються на основі imbalance, whales і соцхвиль.")

@dp.message(commands=["help"])
async def help_handler(message: types.Message):
    await message.reply("/start - запуск\n/status - стан\n/highlow - топ хай/лоу токени за місяць\n/help - допомога")

@dp.message(commands=["highlow"])
async def highlow_handler(message: types.Message):
    highs = []
    lows = []
    for symbol in bot_core.symbols:
        high, low = await bot_core.fetch_monthly_highlow(symbol)
        if high and low:
            highs.append({'symbol': symbol, 'highPrice': high})
            lows.append({'symbol': symbol, 'lowPrice': low})
    highs_sorted = sorted(highs, key=lambda x: x['highPrice'], reverse=True)[:5]
    lows_sorted = sorted(lows, key=lambda x: x['lowPrice'])[:5]
    text = "🚀 Top Highs (last 30 days):\n" + "\n".join([f"{x['symbol']}: {x['highPrice']}" for x in highs_sorted])
    text += "\n\n📉 Top Lows (last 30 days):\n" + "\n".join([f"{x['symbol']}: {x['lowPrice']}" for x in lows_sorted])
    await message.reply(text)

# --- Flask Webhook ---
@app.route(f"/webhook/{TELEGRAM_TOKEN}", methods=["POST"])
async def webhook():
    update = types.Update.de_json(await request.get_json())
    await dp.process_update(update)
    return "OK"

# --- Background Market Task ---
async def live_market_task():
    global bot_core
    bot_core = LiveMarketBot()
    while True:
        try:
            await bot_core.fetch_social_signal()
            for symbol in bot_core.symbols:
                await bot_core.fetch_order_book(symbol)
                whale_volume = await bot_core.fetch_public_trades(symbol)
                imbalance = bot_core.calculate_imbalance(symbol, whale_volume)
                signal = bot_core.generate_signal(symbol, imbalance)
                if signal:
                    text = (
                        f"🔥 {signal['symbol']} Signal: {signal['action']}\n"
                        f"Entry: {signal['entry']}\nTP: {signal['take_profit']}\nSL: {signal['stop_loss']}\n"
                        f"Imbalance: {signal['imbalance']:.2f}\nSocial Pulse: {signal['social_signal']:.2f}"
                    )
                    await bot.send_message(TELEGRAM_CHAT_ID, text)
        except Exception as e:
            print("Market loop error:", e)
        await asyncio.sleep(3)

# --- Main ---
if __name__ == "__main__":
    asyncio.run(bot.set_webhook(WEBHOOK_FULL_URL))
    loop = asyncio.get_event_loop()
    loop.create_task(live_market_task())
    # Flask запускається через gunicorn:
    # gunicorn main:app --bind 0.0.0.0:$PORT --workers 2