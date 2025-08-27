from telebot import TeleBot
from flask import Flask, request
import threading
import time
import requests
from app.config import TELEGRAM_BOT_TOKEN, RENDER_APP_URL

bot = TeleBot(TELEGRAM_BOT_TOKEN)
app = Flask(__name__)

# Реєстрація обробників (як раніше)
from app.handlers.base import register_base_handlers
from app.handlers.price import register_price_handlers
from app.handlers.analysis import register_analysis_handlers
from app.handlers.chart import register_chart_handlers

register_base_handlers(bot)
register_price_handlers(bot)
register_analysis_handlers(bot)
register_chart_handlers(bot)

# Webhook обробник
@app.route('/webhook', methods=['POST'])
def webhook():
    if request.headers.get('content-type') == 'application/json':
        json_string = request.get_data().decode('utf-8')
        update = telebot.types.Update.de_json(json_string)
        bot.process_new_updates([update])
        return ''
    return 'Invalid content type', 403

# Функція для підтримки живості
def keep_alive():
    while True:
        try:
            if RENDER_APP_URL:
                requests.get(RENDER_APP_URL)
            time.sleep(120)  # Пінґ кожні 2 хвилини
        except:
            pass

# Ініціалізація webhook
def setup_webhook():
    if RENDER_APP_URL:
        bot.remove_webhook()
        time.sleep(1)
        bot.set_webhook(url=f"{RENDER_APP_URL}/webhook")

# Запуск Flask сервера
def run_flask():
    app.run(host='0.0.0.0', port=5000)

def run_bot():
    print("🤖 Бот запущено з Webhook...")
    setup_webhook()
    
    # Запускаємо пінґер в окремому потоці
    if RENDER_APP_URL:
        thread = threading.Thread(target=keep_alive, daemon=True)
        thread.start()
    
    run_flask()
