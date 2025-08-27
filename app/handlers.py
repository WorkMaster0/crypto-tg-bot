from app.bot import bot
from app.analytics import get_price, generate_signal, trend_strength, calculate_indicators, get_levels
from app.chart import plot_candles

# 🔹 /start
@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "🚀 Crypto Analysis Bot is alive! Use /analyze BTCUSDT")

# 🔹 /help
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
/indicators BTCUSDT - Technical indicators (RSI, EMA, MACD)
/levels BTCUSDT - Support and resistance levels
/heatmap - Top movers (coming soon 🚀)
""")

# 🔹 /price
@bot.message_handler(commands=['price'])
def price_handler(message):
    args = message.text.split()
    if len(args) > 1:
        symbol = args[1].upper()
        price = get_price(symbol)
        bot.reply_to(message, f"💰 {symbol} price: *{price:.2f}* USDT")
    else:
        bot.reply_to(message, "⚠️ Usage: /price BTCUSDT")

# 🔹 /analyze
@bot.message_handler(commands=['analyze'])
def analyze_handler(message):
    args = message.text.split()
    if len(args) > 1:
        symbol = args[1].upper()
        signal = generate_signal(symbol)
        bot.reply_to(message, signal)
    else:
        bot.reply_to(message, "⚠️ Usage: /analyze BTCUSDT")

# 🔹 /trend
@bot.message_handler(commands=['trend'])
def trend_handler(message):
    args = message.text.split()
    if len(args) > 1:
        symbol = args[1].upper()
        trend = trend_strength(symbol)
        bot.reply_to(message, trend)
    else:
        bot.reply_to(message, "⚠️ Usage: /trend BTCUSDT")

# 🔹 /chart
@bot.message_handler(commands=['chart'])
def chart_handler(message):
    args = message.text.split()
    if len(args) > 1:
        symbol = args[1].upper()
        img = plot_candles(symbol)
        bot.send_photo(message.chat.id, img)
    else:
        bot.reply_to(message, "⚠️ Usage: /chart BTCUSDT")

# 🔹 /indicators
@bot.message_handler(commands=['indicators'])
def indicators_handler(message):
    args = message.text.split()
    if len(args) > 1:
        symbol = args[1].upper()
        indicators = calculate_indicators(symbol)
        response = f"📊 Technical Indicators for {symbol}:\n"
        for key, value in indicators.items():
            response += f"{key}: {value}\n"
        bot.reply_to(message, response)
    else:
        bot.reply_to(message, "⚠️ Usage: /indicators BTCUSDT")

# 🔹 /levels
@bot.message_handler(commands=['levels'])
def levels_handler(message):
    args = message.text.split()
    if len(args) > 1:
        symbol = args[1].upper()
        support, resistance = get_levels(symbol)
        bot.reply_to(message,
f"📌 Levels for {symbol}:\nSupport: {support}\nResistance: {resistance}")
    else:
        bot.reply_to(message, "⚠️ Usage: /levels BTCUSDT")

# 🔹 /heatmap (заглушка)
@bot.message_handler(commands=['heatmap'])
def heatmap_handler(message):
    bot.reply_to(message, "🚀 Heatmap coming soon!")