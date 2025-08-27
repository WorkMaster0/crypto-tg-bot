# handlers.py
from app.bot import bot
from app.analytics import (
    get_price, generate_signal, trend_strength,
    support_resistance_levels, rsi_indicator,
    ema_indicator, sma_indicator, macd_indicator,
)
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
/analyze BTCUSDT - Full analysis with support/resistance + indicators
/price BTCUSDT - Current price
/trend BTCUSDT - Market trend
/chart BTCUSDT - Send chart
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

# 🔹 /analyze - повний аналіз з індикаторами
@bot.message_handler(commands=['analyze'])
def analyze_handler(message):
    args = message.text.split()
    if len(args) > 1:
        symbol = args[1].upper()
        price = get_price(symbol)
        signal = generate_signal(symbol)
        trend = trend_strength(symbol)
        support, resistance = support_resistance_levels(symbol)
        rsi = rsi_indicator(symbol)
        ema = ema_indicator(symbol)
        sma = sma_indicator(symbol)
        macd = macd_indicator(symbol)

        response = f"""
📊 *Analysis for {symbol}*:

💰 Price: {price:.2f} USDT
📈 Signal: {signal}
🔹 Trend: {trend}

📌 Support: {support:.2f}
📌 Resistance: {resistance:.2f}

📊 Indicators:
- RSI: {rsi:.2f}
- EMA: {ema:.2f}
- SMA: {sma:.2f}
- MACD: {macd:.2f}
"""
        bot.reply_to(message, response)
    else:
        bot.reply_to(message, "⚠️ Usage: /analyze BTCUSDT")