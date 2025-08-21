import telebot
from app.config import TELEGRAM_BOT_TOKEN

bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)

@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "🚀 Crypto Analysis Bot is alive! Use /analyze BTC/USDT")

@bot.message_handler(commands=['analyze'])
def analyze_command(message):
    bot.reply_to(message, "📊 Analysis feature is coming soon!")

def main():
    print("Bot is running...")
    bot.infinity_polling()

if __name__ == '__main__':
    main()
