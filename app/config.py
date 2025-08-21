import os
from dotenv import load_dotenv

load_dotenv()  # Завантажує змінні з .env файлу

TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHANNEL_ID = os.getenv('TELEGRAM_CHANNEL_ID')
