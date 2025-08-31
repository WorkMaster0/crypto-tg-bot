# webhook.py
from flask import Flask, request
from app.bot import bot  # Імпортуємо вашого бота
import os

app = Flask(__name__)

# Імпортуємо всі обробники
from app.handlers import *

@app.route('/webhook', methods=['POST'])
def webhook():
    """Обробка запитів від Telegram"""
    if request.method == 'POST':
        try:
            # Отримуємо оновлення від Telegram
            update = telebot.types.Update.de_json(request.get_json())
            # Передаємо боту на обробку
            bot.process_new_updates([update])
            return 'OK'
        except Exception as e:
            print(f"Webhook error: {e}")
            return 'ERROR'
    return 'Invalid method'

@app.route('/')
def home():
    """Стартова сторінка для перевірки"""
    return '🤖 Telegram Bot is running! Use /start command'

if __name__ == '__main__':
    # Налаштовуємо вебхук
    webhook_url = f"https://{os.getenv('RENDER_EXTERNAL_HOSTNAME')}/webhook"
    
    print(f"Setting webhook to: {webhook_url}")
    bot.remove_webhook()
    bot.set_webhook(url=webhook_url)
    
    # Запускаємо Flask сервер
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)