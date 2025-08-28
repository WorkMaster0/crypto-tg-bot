# worker.py
from app.bot import bot
import app.handlers  # реєструє всі хендлери

print("🤖 Bot polling started as Worker...")
bot.infinity_polling()
