import os
from dotenv import load_dotenv

# Обов'язково: на хостингу увімкни змінні середовища або .env
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")

# Базові налаштування
BINANCE_REST = "https://api.binance.com"
DEFAULT_INTERVAL = os.getenv("DEFAULT_INTERVAL", "1h")  # '1m','5m','15m','1h','4h','1d'
TIMEZONE = os.getenv("TIMEZONE", "Europe/Rome")
RISK_PCT_DEFAULT = float(os.getenv("RISK_PCT_DEFAULT", "1.0"))  # % ризику на угоду
LOOKBACK_CANDLES = int(os.getenv("LOOKBACK_CANDLES", "500"))    # скільки свічок тягнути

# Графіки (рендер без X-сервера)
os.environ.setdefault("MPLBACKEND", "Agg")

# «Файли збереження» (прості JSON у корені; можна замінити на БД)
SUBSCRIPTIONS_PATH = os.getenv("SUBSCRIPTIONS_PATH", "subscriptions.json")
SETTINGS_PATH = os.getenv("SETTINGS_PATH", "settings.json")
