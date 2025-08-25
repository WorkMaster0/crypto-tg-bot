from app.bot import bot
from app.analytics import get_price, generate_signal, trend_strength
from app.chart import plot_candles

# 🔹 Старий /start
@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "🚀 Crypto Analysis Bot is alive! Use /analyze BTCUSDT")

# 🔹 Старий /analyze (заглушка)
@bot.message_handler(commands=['analyze'])
def old_analyze_command(message):
    args = message.text.split()
    if len(args) == 1:
        bot.reply_to(message, "📊 Analysis feature is coming soon!")

# 🔹 Новий /price
@bot.message_handler(commands=['price'])
def price_handler(message):
    args = message.text.split()
    if len(args) > 1:
        symbol = args[1].upper()
        price = get_price(symbol)
        bot.reply_to(message, f"💰 {symbol} price: *{price:.2f}* USDT")
    else:
        bot.reply_to(message, "⚠️ Usage: /price BTCUSDT")

# 🔹 Новий /analyze з сигналами
@bot.message_handler(commands=['analyze'])
def analyze_handler(message):
    args = message.text.split()
    if len(args) > 1:
        symbol = args[1].upper()
        signal = generate_signal(symbol)
        bot.reply_to(message, signal)

# 🔹 Новий /trend
@bot.message_handler(commands=['trend'])
def trend_handler(message):
    args = message.text.split()
    if len(args) > 1:
        symbol = args[1].upper()
        trend = trend_strength(symbol)
        bot.reply_to(message, trend)
    else:
        bot.reply_to(message, "⚠️ Usage: /trend BTCUSDT")

# 🔹 Новий /chart
@bot.message_handler(commands=['chart'])
def chart_handler(message):
    args = message.text.split()
    if len(args) > 1:
        symbol = args[1].upper()
        img = plot_candles(symbol)
        bot.send_photo(message.chat.id, img)
    else:
        bot.reply_to(message, "⚠️ Usage: /chart BTCUSDT")

# 🔹 Новий /help
@bot.message_handler(commands=['help'])
def send_help(message):
    bot.reply_to(message,
"""
📌 *Available Commands:*
/start - Check bot status
/analyze BTCUSDT - Get support/resistance + signal
/price BTCUSDT - Current price
/trend BTCUSDT - Market trend
/chart BTCUSDT - Send chart
/heatmap - Top movers (coming soon 🚀)
""")