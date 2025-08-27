import os
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
RENDER_APP_URL = os.getenv('RENDER_APP_URL')  # https://your-app-name.onrender.com

# Binance
BINANCE_BASES = [
    "https://api.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
]
DEFAULT_INTERVAL = os.getenv("DEFAULT_INTERVAL", "1h")
KLINES_LIMIT = int(os.getenv("KLINES_LIMIT", "500"))

# Параметри пошуку рівнів
PIVOT_LEFT_RIGHT = int(os.getenv("PIVOT_LEFT_RIGHT", "5"))  # сусіди для фракталів
MAX_LEVELS = int(os.getenv("MAX_LEVELS", "6"))

# Безпека HTTP
HTTP_TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "10"))

# Парсинг повідомлень
ALLOWED_INTERVALS = {"1m","3m","5m","15m","30m","1h","2h","4h","6h","8h","12h","1d","3d","1w","1M"}
