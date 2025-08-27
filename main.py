import os
import json
from flask import Flask, request
from app.bot import bot
import app.handlers  # імпортуємо всі твої хендлери
import telebot.types

app = Flask(__name__)

# 🔑 Токен і URL для webhook
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
BASE_URL = os.getenv("RENDER_EXTERNAL_URL", "https://crypto-tg-bot-2.onrender.com")
WEBHOOK_URL = f"{BASE_URL}/{TOKEN}"


# 🔹 Endpoint для отримання апдейтів від Telegram
@app.route(f"/{TOKEN}", methods=["POST"])
def receive_update():
    try:
        json_str = request.get_data().decode("UTF-8")
        update = telebot.types.Update.de_json(json.loads(json_str))
        bot.process_new_updates([update])
    except Exception as e:
        print(f"❌ Error processing update: {e}")
    return "OK", 200


# 🔹 Health-check
@app.route("/")
def home():
    return "✅ Crypto Bot is running with webhook!"


# 🔹 Ставимо webhook при старті
with app.app_context():
    bot.remove_webhook()
    bot.set_webhook(url=WEBHOOK_URL)
    print(f"🌍 Webhook set to {WEBHOOK_URL}")