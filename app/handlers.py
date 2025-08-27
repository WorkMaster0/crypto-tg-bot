from telebot import TeleBot
from telebot.types import Message
from app.analytics import get_price, generate_signal, trend_strength

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


"""
📌 *Available Commands:*
/start - Check bot status
/analyze BTCUSDT - Get support/resistance + signal
/price BTCUSDT - Current price
/trend BTCUSDT - Market trend
/chart BTCUSDT - Send chart
/heatmap - Top movers (coming soon 🚀)
""")

# 🔹 Нова команда /advanced для розширеного аналізу
@bot.message_handler(commands=['advanced'])
def advanced_handler(message):
    args = message.text.split()
    if len(args) > 1:
        symbol = args[1].upper()
        from app.analytics import advanced_analysis
        analysis = advanced_analysis(symbol)
        bot.reply_to(message, analysis, parse_mode='Markdown')
    else:
        bot.reply_to(message, "⚠️ Usage: /advanced BTCUSDT")

# 🔹 Нова команда /volume для аналізу об'ємів
@bot.message_handler(commands=['volume'])
def volume_handler(message):
    args = message.text.split()
    if len(args) > 1:
        symbol = args[1].upper()
        from app.analytics import volume_analysis
        analysis = volume_analysis(symbol)
        bot.reply_to(message, analysis, parse_mode='Markdown')
    else:
        bot.reply_to(message, "⚠️ Usage: /volume BTCUSDT")

# 🔹 Нова команда /levels для детальних рівнів
@bot.message_handler(commands=['levels'])
def levels_handler(message):
    args = message.text.split()
    if len(args) > 1:
        symbol = args[1].upper()
        from app.analytics import detailed_levels
        levels = detailed_levels(symbol)
        bot.reply_to(message, levels, parse_mode='Markdown')
    else:
        bot.reply_to(message, "⚠️ Usage: /levels BTCUSDT")

# 🔹 Нова команда /signal для швидкого сигналу
@bot.message_handler(commands=['signal'])
def signal_handler(message):
    args = message.text.split()
    if len(args) > 1:
        symbol = args[1].upper()
        from app.analytics import quick_signal
        signal = quick_signal(symbol)
        bot.reply_to(message, signal, parse_mode='Markdown')
    else:
        bot.reply_to(message, "⚠️ Usage: /signal BTCUSDT")

# 🔹 Нова команда /alerts для налаштування сповіщень
@bot.message_handler(commands=['alerts'])
def alerts_handler(message):
    bot.reply_to(message,
"""
🔔 *Alert System (Coming Soon):*
- Price alerts
- Volume spike alerts  
- Breakout alerts
- Trend change alerts

Stay tuned! 🚀
""", parse_mode='Markdown')

# 🔹 Обробка невідомих команд
@bot.message_handler(func=lambda message: True)
def handle_unknown(message):
    bot.reply_to(message, "❌ Unknown command. Use /help for available commands.")

def register_all_handlers(bot_instance: TeleBot):
    """Функція для реєстрації всіх обробників"""
    global bot
    bot = bot_instance
    
    # Реєструємо всі обробники
    bot.register_message_handler(send_welcome)
    bot.register_message_handler(old_analyze_command)
    bot.register_message_handler(price_handler)
    bot.register_message_handler(analyze_handler)
    bot.register_message_handler(trend_handler)
    bot.register_message_handler(chart_handler)
    bot.register_message_handler(send_help)
    bot.register_message_handler(advanced_handler)
    bot.register_message_handler(volume_handler)
    bot.register_message_handler(levels_handler)
    bot.register_message_handler(signal_handler)
    bot.register_message_handler(alerts_handler)
    bot.register_message_handler(handle_unknown)
