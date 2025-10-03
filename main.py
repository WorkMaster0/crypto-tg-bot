import os
import requests
import numpy as np
from flask import Flask, request
from telebot import TeleBot, types

# ================== ENV ==================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
if not TELEGRAM_TOKEN:
    raise ValueError("❌ Відсутній TELEGRAM_TOKEN у .env!")

PORT = int(os.getenv("PORT", 5000))
WEBHOOK_URL_BASE = os.getenv("WEBHOOK_URL_BASE")  # Наприклад: https://crypto-tg-bot.onrender.com
if not WEBHOOK_URL_BASE:
    raise ValueError("❌ Відсутній WEBHOOK_URL_BASE у .env!")

WEBHOOK_URL_PATH = f"/telegram_webhook/{TELEGRAM_TOKEN}"

# ================== Flask & Bot ==================
app = Flask(__name__)
bot = TeleBot(TELEGRAM_TOKEN, parse_mode="HTML")

# ================== Допоміжні функції ==================
def get_klines(symbol, interval="1h", limit=200):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    data = requests.get(url).json()
    return {
        "c": [float(d[4]) for d in data],
        "v": [float(d[5]) for d in data],
    }

def find_support_resistance(closes, window=20, delta=0.005):
    levels = []
    for i in range(window, len(closes) - window):
        local_max = max(closes[i - window:i + window + 1])
        local_min = min(closes[i - window:i + window + 1])
        if abs(closes[i] - local_max) / local_max < delta:
            levels.append(local_max)
        if abs(closes[i] - local_min) / local_min < delta:
            levels.append(local_min)
    return sorted(list(set(levels)))

# ================== Команди ==================
@bot.message_handler(commands=['start'])
def start_handler(message):
    bot.reply_to(message, "👋 Привіт! Я бот для пошуку сигналів.\nСпробуй команду /smart_auto 🚀")

@bot.message_handler(commands=['help'])
def help_handler(message):
    text = (
        "📌 Доступні команди:\n"
        "/start - перевірити роботу бота\n"
        "/help - список команд\n"
        "/smart_auto - запустити сканер сигналів"
    )
    bot.send_message(message.chat.id, text)

@bot.message_handler(commands=['smart_auto'])
def smart_auto_handler(message):
    print(f"[COMMAND] Виконано /smart_auto від {message.from_user.id}")
    try:
        url = "https://api.binance.com/api/v3/ticker/24hr"
        data = requests.get(url).json()

        symbols = [
            d for d in data
            if d["symbol"].endswith("USDT") and float(d["quoteVolume"]) > 5_000_000
        ]

        print(f"[DEBUG] Відібрано {len(symbols)} монет з об’ємом >5M")

        symbols = sorted(symbols, key=lambda x: abs(float(x["priceChangePercent"])), reverse=True)
        top_symbols = [s["symbol"] for s in symbols[:30]]

        print(f"[DEBUG] Топ {len(top_symbols)} монет для аналізу: {top_symbols}")

        signals = []
        for symbol in top_symbols:
            try:
                df = get_klines(symbol, interval="1h", limit=200)
                closes = np.array(df["c"], dtype=float)

                if len(closes) < 50:
                    continue

                sr_levels = find_support_resistance(closes, window=20, delta=0.005)
                last_price = closes[-1]

                signal = None
                for lvl in sr_levels:
                    if last_price > lvl * 1.01:
                        signal = f"🚀 LONG breakout <b>{symbol}</b> біля {lvl:.4f}"
                        break
                    elif last_price < lvl * 0.99:
                        signal = f"⚡ SHORT breakout <b>{symbol}</b> біля {lvl:.4f}"
                        break

                if signal:
                    signals.append(signal)

            except Exception as e:
                print(f"[ERROR] Символ {symbol}: {e}")
                continue

        if not signals:
            print("[DEBUG] Сигнали не знайдено")
            bot.send_message(message.chat.id, "ℹ️ Жодних сигналів не знайдено.")
        else:
            print(f"[DEBUG] Надсилаю {len(signals)} сигналів")
            text = "<b>Smart Auto S/R Signals</b>\n\n" + "\n\n".join(signals)
            bot.send_message(message.chat.id, text, parse_mode="HTML")

    except Exception as e:
        print(f"[FATAL ERROR] {e}")
        bot.send_message(message.chat.id, f"❌ Error: {e}")

# ================== Webhook routes ==================
@app.route(WEBHOOK_URL_PATH, methods=['POST'])
def telegram_webhook():
    try:
        json_str = request.get_data().decode("utf-8")  # Отримуємо сировий JSON
        update = types.Update.de_json(json_str)       # Парсимо правильно
        print(f"[UPDATE] Надійшов апдейт: {json_str}")  # Для дебагу
        bot.process_new_updates([update])             # Обробка ботом
    except Exception as e:
        print(f"[ERROR] Webhook обробка: {e}")
    return "", 200

@app.route("/", methods=['GET'])
def index():
    return "Bot is running ✅", 200

# ================== Setup webhook ==================
def setup_webhook():
    try:
        bot.remove_webhook()
        webhook_url = WEBHOOK_URL_BASE + WEBHOOK_URL_PATH
        bot.set_webhook(url=webhook_url)
        print(f"[INFO] Webhook встановлено: {webhook_url}")
    except Exception as e:
        print(f"[ERROR] Не вдалося встановити webhook: {e}")

setup_webhook()

# ================== Run app ==================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)