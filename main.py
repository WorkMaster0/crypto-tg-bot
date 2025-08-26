from app.bot import bot
import app.handlers  # завантажує всі команди

def main():
    print("✅ Bot is running...")
    bot.infinity_polling()

if __name__ == '__main__':
    main()
