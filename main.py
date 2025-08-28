# main.py
import os
import threading
from flask import Flask
from app.bot import bot
import app.handlers

app = Flask(__name__)

@app.route("/")
def home():
    return "✅ Crypto Bot is running!"

def run_bot():
    print("🤖 Bot polling started...")
    try:
        bot.infinity_polling(skip_pending=True)
    except Exception as e:
        print(f"Bot error: {e}")
        # Спробуємо перезапустити через 60 секунд
        threading.Timer(60.0, run_bot).start()

# Запускаємо бота тільки якщо це головний процес
if __name__ == "__main__":
    # Запускаємо бота в окремому потоці
    bot_thread = threading.Thread(target=run_bot, daemon=True)
    bot_thread.start()
    
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
