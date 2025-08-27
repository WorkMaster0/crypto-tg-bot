import os
from flask import Flask, request
from app.bot import bot
import app.handlers  # імпортуємо всі твої команди, щоб вони зареєструвались

app = Flask(__name__)

# 🔑 Токен і URL для webhook
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
WEBHOOK_URL = f"https://crypto-tg-bot-2.onrender.com/{TOKEN}"  # заміни на свій Render URL

# 🔹 Endpoint для отримання апдейтів від Telegram
@app.route(f"/{TOKEN}", methods=["POST"])
def receive_update():
    json_str = request.get_data().decode("UTF-8")
    update = bot._convert_update(json_str)
    bot.process_new_updates([update])
    return "OK", 200

# 🔹 Health-check (щоб Render бачив, що сервіс живий)
@app.route("/")
def home():
    return "✅ Crypto Bot is running with webhook!"

# 🔹 При першому запиті Flask ставить webhook
@app.before_first_request
def setup_webhook():
    bot.remove_webhook()
    bot.set_webhook(url=WEBHOOK_URL)
    print(f"🌍 Webhook set to {WEBHOOK_URL}")