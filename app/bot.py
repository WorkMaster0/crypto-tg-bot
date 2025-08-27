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

# Імпорт та реєстрація обробників
from app.handlers.base import register_base_handlers
from app.handlers.price import register_price_handlers
from app.handlers.analysis import register_analysis_handlers
from app.handlers.chart import register_chart_handlers

# Реєстрація всіх обробників
register_base_handlers(bot)
register_price_handlers(bot)
register_analysis_handlers(bot)
register_chart_handlers(bot)

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

# Роут для ручного тестування
@app.route('/test', methods=['GET'])
def test_route():
    """Тестовий роут для перевірки"""
    return {'message': 'Bot is working!', 'timestamp': time.time()}, 200

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
            time.sleep(60)  # Чекаємо менше при помилці

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
            
            # Перевіряємо статус вебхука
            webhook_info = bot.get_webhook_info()
            print(f"📊 Webhook info: {webhook_info.url}")
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
    
    # Запускаємо Flask сервер
    run_flask()

# Обробник помилок
@bot.message_handler(func=lambda message: True)
def handle_all_messages(message):
    """Обробка всіх невідомих повідомлень"""
    bot.reply_to(message, "❌ Невідома команда. Використовуйте /help для довідки")

# Обробник помилок бота
@bot.error_handler
def error_handler(update, error):
    """Глобальний обробник помилок"""
    print(f"❌ Bot error: {error}")
    try:
        if update and hasattr(update, 'message'):
            bot.reply_to(update.message, "⚠️ Сталася помилка. Спробуйте пізніше.")
    except:
        pass

# Додаткові функції для управління ботом
def get_bot_info():
    """Отримати інформацію про бота"""
    try:
        me = bot.get_me()
        return f"Bot: @{me.username}, ID: {me.id}"
    except Exception as e:
        return f"Error getting bot info: {e}"

def check_webhook_status():
    """Перевірити статус вебхука"""
    try:
        webhook_info = bot.get_webhook_info()
        return {
            'url': webhook_info.url,
            'has_custom_certificate': webhook_info.has_custom_certificate,
            'pending_update_count': webhook_info.pending_update_count
        }
    except Exception as e:
        return f"Error checking webhook: {e}"

# Команда для адмінів (додати в handlers/base.py пізніше)
@bot.message_handler(commands=['admin'])
def admin_command(message):
    """Адмінська команда для перевірки статусу"""
    if message.from_user.id == 123456789:  # Замінити на ваш Telegram ID
        bot_info = get_bot_info()
        webhook_status = check_webhook_status()
        bot.reply_to(message, f"👨‍💻 Admin Panel:\n{bot_info}\nWebhook: {webhook_status}")
    else:
        bot.reply_to(message, "⛔ Доступ заборонено")

if __name__ == '__main__':
    run_bot()
