# main.py
import os
from flask import Flask, request
from app.bot import bot
import app.handlers  # Імпортуємо всі обробники

app = Flask(__name__)

@app.route("/")
def home():
    return "✅ Crypto Bot is running!"

@app.route('/webhook', methods=['POST'])
def webhook():
    """Обробка вебхуків від Telegram"""
    if request.method == 'POST':
        try:
            # Отримуємо оновлення від Telegram
            json_data = request.get_json()
            update = telebot.types.Update.de_json(json_data)
            
            # Передаємо оновлення боту
            bot.process_new_updates([update])
            return 'OK'
        except Exception as e:
            print(f"Webhook error: {e}")
            return 'ERROR', 500
    return 'Invalid method', 400

# Налаштовуємо вебхук при запуску
def setup_webhook():
    try:
        # Отримуємо URL з Render
        render_url = os.getenv('RENDER_EXTERNAL_URL')
        if render_url:
            webhook_url = f"{render_url}/webhook"
            bot.remove_webhook()
            bot.set_webhook(url=webhook_url)
            print(f"✅ Webhook set to: {webhook_url}")
        else:
            print("⚠️  RENDER_EXTERNAL_URL not set, using polling for development")
            bot.remove_webhook()
    except Exception as e:
        print(f"❌ Webhook setup error: {e}")

if __name__ == "__main__":
    # Налаштовуємо вебхук
    setup_webhook()
    
    # Запускаємо Flask
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)