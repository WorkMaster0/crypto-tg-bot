from app.bot import start_polling
import app.handlers  # noqa: F401  (реєструє хендлери)

if __name__ == "__main__":
    start_polling()