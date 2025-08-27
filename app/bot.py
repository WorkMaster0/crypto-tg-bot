from telebot import TeleBot
from app.config import TELEGRAM_BOT_TOKEN

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")

# HTML безпечніший за Markdown щодо символів
bot = TeleBot(TELEGRAM_BOT_TOKEN, parse_mode="HTML")