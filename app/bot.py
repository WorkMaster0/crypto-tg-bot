import asyncio
from telegram.ext import Application, CommandHandler
from app.config import TELEGRAM_BOT_TOKEN

# Проста команда для тесту
async def start(update, context):
    await update.message.reply_text('🚀 Crypto Analysis Bot is alive! Use /analyze BTC/USDT')

# Основна функція для запуску бота
def main():
    # Створюємо Application
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    # Додаємо обробники команд
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("analyze", analyze_command))  # Ми її створимо далі
    
    # Запускаємо бота
    print("Bot is running...")
    application.run_polling()

if __name__ == '__main__':
    main()
