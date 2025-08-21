import os
import asyncio
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from app.config import TELEGRAM_BOT_TOKEN

# Ініціалізація бота та диспетчера
bot = Bot(token=TELEGRAM_BOT_TOKEN)
dp = Dispatcher()

# Обробка команди /start
@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    await message.answer("🚀 Crypto Analysis Bot is alive! Use /analyze BTC/USDT")

# Обробка команди /analyze (заглушка)
@dp.message(Command("analyze"))
async def cmd_analyze(message: types.Message):
    await message.answer("📊 Analysis feature is coming soon!")

# Основна функція для запуску бота
async def main():
    print("Bot is running...")
    await dp.start_polling(bot)

if __name__ == '__main__':
    asyncio.run(main())
