import threading
from flask import Flask
from app.bot import bot
import app.handlers  # імпортуємо, щоб підвантажились усі команди

app = Flask(__name__)

# 🔹 Health-check endpoint для Render
@app.route("/")
def home():
    return "✅ Crypto Bot is running!"

# 🔹 Функція запуску Telegram-бота
def run_bot():
    print("🤖 Bot polling started...")
    bot.infinity_polling(skip_pending=True)

# 🔹 Запускаємо бота у фоновому потоці
threading.Thread(target=run_bot, daemon=True).start()

# ⚠️ Важливо:
# - НЕ запускаємо app.run() тут, бо Gunicorn це робитиме сам
# - Gunicorn викличе app з цього файлу, коли ти напишеш: gunicorn main:app