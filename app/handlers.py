from telebot import TeleBot
from telebot.types import Message

def register_all_handlers(bot: TeleBot):
    
    @bot.message_handler(commands=['start'])
    def send_welcome(message: Message):
        bot.reply_to(message, "✅ Бот працює! Використовуй /help")
    
    @bot.message_handler(commands=['help'])
    def send_help(message: Message):
        bot.reply_to(message, "📋 Доступні команди: /start, /help")
    
    @bot.message_handler(commands=['test'])
    def test_cmd(message: Message):
        bot.reply_to(message, "🧪 Тестовий режим. Все OK!")
    
    @bot.message_handler(func=lambda message: True)
    def handle_unknown(message: Message):
        bot.reply_to(message, "❌ Невідома команда. /help")
