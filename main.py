# main.py
import os
from flask import Flask
from app.bot import bot
import app.handlers  # реєструє всі хендлери

app = Flask(__name__)

@app.route("/")
def home():
    return "✅ Crypto Bot is running!"

# НЕ запускаємо бота автоматично
# Бот буде запускатися тільки якщо це не Render
if os.environ.get('RENDER', None):
    print("✅ Running on Render - Bot will be started separately")
else:
    print("🤖 Bot polling started locally...")
    bot.infinity_polling(skip_pending=True)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
