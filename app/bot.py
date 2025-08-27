import telebot
from app.config import TELEGRAM_BOT_TOKEN
from app.analytics import get_price, generate_signal, trend_strength
from app.chart import plot_candles

bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)

@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "🚀 Crypto Analysis Bot is alive! Use /analyze BTCUSDT")

@bot.message_handler(commands=['price'])
def price_handler(message):
    args = message.text.split()
    if len(args) > 1:
        symbol = args[1].upper()
        price = get_price(symbol)
        if price > 0:
            bot.reply_to(message, f"💰 *{symbol}* price: `${price:.2f}`")
        else:
            bot.reply_to(message, "❌ Не вдалося отримати ціну")
    else:
        bot.reply_to(message, "⚠️ Використання: /price BTCUSDT")

@bot.message_handler(commands=['analyze'])
def analyze_handler(message):
    args = message.text.split()
    if len(args) > 1:
        symbol = args[1].upper()
        signal = generate_signal(symbol)
        bot.reply_to(message, signal, parse_mode='Markdown')
    else:
        bot.reply_to(message, "⚠️ Використання: /analyze BTCUSDT")

@bot.message_handler(commands=['trend'])
def trend_handler(message):
    args = message.text.split()
    if len(args) > 1:
        symbol = args[1].upper()
        trend = trend_strength(symbol)
        bot.reply_to(message, trend, parse_mode='Markdown')
    else:
        bot.reply_to(message, "⚠️ Використання: /trend BTCUSDT")

@bot.message_handler(commands=['chart'])
def chart_handler(message):
    args = message.text.split()
    if len(args) > 1:
        symbol = args[1].upper()
        img = plot_candles(symbol)
        if img:
            bot.send_photo(message.chat.id, img, caption=f"📈 Графік {symbol} (1h)")
        else:
            bot.reply_to(message, "❌ Не вдалося побудувати графік")
    else:
        bot.reply_to(message, "⚠️ Використання: /chart BTCUSDT")

@bot.message_handler(commands=['help'])
def send_help(message):
    help_text = """
📌 *Доступні команди:*
/start - Перевірити статус бота
/analyze BTCUSDT - Аналіз + рівні підтримки/опору
/price BTCUSDT - Поточна ціна
/trend BTCUSDT - Аналіз тренду
/chart BTCUSDT - Графік ціни
/help - Довідка

🔸 *Приклади:*
/analyze BTCUSDT
/price ETHUSDT
/chart SOLUSDT
"""
    bot.reply_to(message, help_text, parse_mode='Markdown')

def main():
    print("Bot is running...")
    bot.infinity_polling()

if __name__ == '__main__':
    main()
