#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import logging
import requests
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image
from dotenv import load_dotenv
from telebot import TeleBot
import http.server
import socketserver
import threading

# ---------------- LOGGING ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger("smart-auto-bot")

# ---------------- LOAD ENV ----------------
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
PORT = int(os.getenv("PORT", 5000))
bot = TeleBot(TELEGRAM_TOKEN)

# ---------------- HTTP SERVER (для Render) ----------------
def start_http():
    class Handler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            logger.debug("HTTP: " + format % args)
    port = PORT
    try:
        with socketserver.TCPServer(("", port), Handler) as httpd:
            logger.info("HTTP server listening on port %d", port)
            httpd.serve_forever()
    except Exception as e:
        logger.exception("HTTP server error: %s", e)

threading.Thread(target=start_http, daemon=True).start()

# ---------------- HELPER FUNCTIONS ----------------
def get_klines(symbol, interval="1h", limit=200):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    r = requests.get(url, timeout=10)
    data = r.json()
    return {
        "t": [x[0] for x in data],
        "o": [float(x[1]) for x in data],
        "h": [float(x[2]) for x in data],
        "l": [float(x[3]) for x in data],
        "c": [float(x[4]) for x in data],
        "v": [float(x[5]) for x in data],
    }

def find_support_resistance(closes, window=20):
    levels = []
    for i in range(window, len(closes)-window):
        high_range = closes[i-window:i+window]
        low_range = closes[i-window:i+window]
        if closes[i] == max(high_range):
            levels.append(closes[i])
        elif closes[i] == min(low_range):
            levels.append(closes[i])
    return sorted(list(set(levels)))

def plot_candles(closes, highs, lows, opens, sr_levels, symbol):
    fig, ax = plt.subplots(figsize=(10,5))
    x = np.arange(len(closes))
    ax.plot(x, closes, color='black', label='Close')
    ax.fill_between(x, lows, highs, color='lightgray', alpha=0.5)
    for lvl in sr_levels:
        ax.hlines(lvl, x[0], x[-1], colors='blue', linestyles='--', alpha=0.6)
    ax.set_title(symbol)
    ax.set_ylabel("Price")
    ax.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf

# ---------------- SMART AUTO COMMAND ----------------
@bot.message_handler(commands=['smart_auto'])
def smart_auto_handler(message):
    try:
        url = "https://api.binance.com/api/v3/ticker/24hr"
        data = requests.get(url, timeout=10).json()

        # Фільтр USDT пар з об'ємом >5M
        symbols = [
            d for d in data
            if d["symbol"].endswith("USDT") and float(d["quoteVolume"]) > 5_000_000
        ]

        # Топ 30 за абсолютною зміною ціни
        symbols = sorted(symbols, key=lambda x: abs(float(x["priceChangePercent"])), reverse=True)
        top_symbols = [s["symbol"] for s in symbols[:30]]

        signals = []

        for symbol in top_symbols:
            try:
                df = get_klines(symbol, interval="1h", limit=200)
                if len(df["c"]) < 50:
                    continue

                closes = np.array(df["c"], dtype=float)
                volumes = np.array(df["v"], dtype=float)
                highs = np.array(df["h"], dtype=float)
                lows = np.array(df["l"], dtype=float)
                opens = np.array(df["o"], dtype=float)

                sr_levels = find_support_resistance(closes, window=20)
                last_price = closes[-1]

                signal = None
                for lvl in sr_levels:
                    diff = last_price - lvl
                    diff_pct = (diff / lvl) * 100
                    if last_price > lvl * 1.01:
                        signal = (
                            f"🚀 LONG breakout: ціна пробила опір {lvl:.4f}\n"
                            f"📊 Ринкова: {last_price:.4f} | Відрив: {diff:+.4f} ({diff_pct:+.2f}%)"
                        )
                        break
                    elif last_price < lvl * 0.99:
                        signal = (
                            f"⚡ SHORT breakout: ціна пробила підтримку {lvl:.4f}\n"
                            f"📊 Ринкова: {last_price:.4f} | Відрив: {diff:+.4f} ({diff_pct:+.2f}%)"
                        )
                        break

                impulse = (closes[-1] - closes[-4]) / closes[-4] if len(closes) >= 4 else 0
                vol_spike = volumes[-1] > 1.5 * np.mean(volumes[-20:]) if len(volumes) >= 20 else False
                nearest_res = max([lvl for lvl in sr_levels if lvl < last_price], default=None)
                if impulse > 0.08 and vol_spike and nearest_res is not None:
                    diff = last_price - nearest_res
                    diff_pct = (diff / nearest_res) * 100
                    signal = (
                        f"⚠️ Pre-top detected: можливий short біля {nearest_res:.4f}\n"
                        f"📊 Ринкова: {last_price:.4f} | Відрив: {diff:+.4f} ({diff_pct:+.2f}%)"
                    )

                if signal:
                    img_buf = plot_candles(closes, highs, lows, opens, sr_levels, symbol)
                    signals.append({"text": f"<b>{symbol}</b>\n{signal}", "photo": img_buf})

            except Exception:
                continue

        if not signals:
            bot.send_message(message.chat.id, "ℹ️ Жодних сигналів не знайдено.")
        else:
            for s in signals:
                bot.send_photo(message.chat.id, photo=s["photo"], caption=s["text"], parse_mode="HTML")

    except Exception as e:
        bot.send_message(message.chat.id, f"❌ Error: {e}")

# ---------------- RUN BOT ----------------
logger.info("Smart Auto Bot started")
bot.infinity_polling()