import threading
import os
from flask import Flask
from app.bot import bot
import app.handlers  # реєструє всі хендлери

app = Flask(__name__)

@app.route("/")
def home():
    return "✅ Crypto Bot is running!"

def run_bot():
    print("🤖 Bot polling started...")
    bot.infinity_polling(skip_pending=True)

if __name__ == "__main__":
    threading.Thread(target=run_bot, daemon=True).start()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)