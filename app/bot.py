from telebot import TeleBot
from flask import Flask, request
import threading
import time
import requests
from app.config import TELEGRAM_BOT_TOKEN, RENDER_APP_URL

# Ініціалізація Flask додатку
app = Flask(__name__)

# Створення екземпляру бота
bot = TeleBot(TELEGRAM_BOT_TOKEN)

# Імпортуємо та реєструємо обробники
from app.handlers import register_all_handlers
register_all_handlers(bot)  # ← передаємо bot у функцію

# Решта коду (webhook, Flask routes)...

# Webhook обробник для Telegram
@app.route('/webhook', methods=['POST'])
def webhook():
    """Обробник вебхука від Telegram"""
    if request.headers.get('content-type') == 'application/json':
        json_string = request.get_data().decode('utf-8')
        update = bot.update_de_json(json_string)
        bot.process_new_updates([update])
        return '', 200
    return 'Invalid content type', 403

# Роут для перевірки працездатності
@app.route('/health', methods=['GET'])
def health_check():
    """Перевірка статусу бота"""
    return {'status': 'ok', 'bot': 'running'}, 200

def keep_alive():
    """Функція для підтримки живості інстансу на Render"""
    while True:
        try:
            if RENDER_APP_URL:
                # Пінгуємо наш власний додаток кожні 2 хвилини
                health_url = f"{RENDER_APP_URL}/health"
                response = requests.get(health_url, timeout=10)
                print(f"🔄 Keep-alive ping: {response.status_code} - {time.strftime('%H:%M:%S')}")
            time.sleep(120)  # Пінґ кожні 2 хвилини
        except Exception as e:
            print(f"❌ Keep-alive error: {e}")
            time.sleep(60)

def setup_webhook():
    """Налаштування вебхука для Telegram"""
    try:
        if RENDER_APP_URL:
            # Видаляємо старий вебхук
            bot.remove_webhook()
            time.sleep(1)
            
            # Встановлюємо новий вебхук
            webhook_url = f"{RENDER_APP_URL}/webhook"
            bot.set_webhook(url=webhook_url)
            print(f"✅ Webhook set to: {webhook_url}")
        else:
            print("ℹ️ Running in polling mode (local development)")
    except Exception as e:
        print(f"❌ Webhook setup error: {e}")

def run_flask():
    """Запуск Flask сервера"""
    print("🌐 Starting Flask server...")
    app.run(host='0.0.0.0', port=5000, debug=False)

def run_bot():
    """Головна функція запуску бота"""
    print("🤖 Initializing Crypto Analysis Bot...")
    
    # Налаштовуємо вебхук
    setup_webhook()
    
    # Запускаємо пінґер для підтримки живості (тільки на Render)
    if RENDER_APP_URL:
        print("🔗 Starting keep-alive thread for Render...")
        keep_alive_thread = threading.Thread(target=keep_alive, daemon=True)
        keep_alive_thread.start()
    
    # Запускаємо Flask сервер (ВЕБХУК режим)
    print("🌐 Starting Flask server in WEBHOOK mode...")
    app.run(host='0.0.0.0', port=5000, debug=False)

# Обробник помилок бота
@bot.error_handler
def error_handler(update, error):
    """Глобальний обробник помилок"""
    print(f"❌ Bot error: {error}")

if __name__ == '__main__':
    run_bot()
