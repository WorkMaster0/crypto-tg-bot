import threading
import os
from flask import Flask
from app.bot import bot
import app.handlers  # підвантажує всі команди

app = Flask(__name__)

# 🔹 Health-check для Render
@app.route("/")
def home():
    return "✅ Crypto Bot is running!"

def run_bot():
    print("🤖 Bot polling started...")
    bot.infinity_polling()

if __name__ == "__main__":
    # Запускаємо бота в окремому потоці
    threading.Thread(target=run_bot).start()

    # Flask слухає порт, який Render дає через ENV
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)