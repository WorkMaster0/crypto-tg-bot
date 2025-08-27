import telebot
import os
import traceback

# ====== Налаштування ======
BOT_TOKEN = os.getenv("BOT_TOKEN")  # у Render треба додати в Environment variables
bot = telebot.TeleBot(BOT_TOKEN, parse_mode="HTML")


# ====== Команди ======
@bot.message_handler(commands=['start'])
def start(message):
    bot.send_message(message.chat.id, "👋 Привіт! Я бот для криптоаналітики.")


@bot.message_handler(commands=['help'])
def help_cmd(message):
    bot.send_message(message.chat.id, "🛠 Доступні команди:\n/start\n/help")


# ====== Універсальний catcher для всіх повідомлень ======
@bot.message_handler(func=lambda msg: True)
def catch_all(message):
    try:
        # Тут можеш ставити свою основну логіку
        bot.send_message(message.chat.id, f"📩 Ти написав: {message.text}")

    except Exception as e:
        # Локальна помилка для конкретного апдейту
        error_text = f"⚠️ Помилка: {e}\n\n{traceback.format_exc()}"
        print(error_text)  # пишемо в логи Render
        bot.send_message(message.chat.id, "❌ Сталася помилка під час обробки твого запиту.")


# ====== Глобальний catcher ======
def run_bot():
    while True:
        try:
            bot.infinity_polling(timeout=60, long_polling_timeout=30)
        except Exception as e:
            print("🔥 Глобальна помилка:", e)
            traceback.print_exc()


if __name__ == "__main__":
    run_bot()
