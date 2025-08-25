import telebot
from app.config import TELEGRAM_BOT_TOKEN

bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN, parse_mode="HTML")

def start_polling():
    print("Bot is running...")
    bot.infinity_polling(timeout=60, long_polling_timeout=60)