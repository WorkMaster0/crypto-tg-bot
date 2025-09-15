from flask import Flask
from apscheduler.schedulers.background import BackgroundScheduler
from strategy import apply_strategy
from binance_data import get_klines, get_top_symbols
from telegram_alert import send_telegram_message
import json
import os

app = Flask(__name__)

STATE_FILE = 'state.json'

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_state(state):
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f)

state = load_state()

def monitor():
    print("🔍 Monitoring started...")
    symbols = get_top_symbols(limit=30)
    intervals = ['15m', '1h', '4h', '1d']

    for symbol in symbols:
        for interval in intervals:
            try:
                df = get_klines(symbol, interval)
                signal = apply_strategy(df)

                if signal:
                    key = f"{symbol}_{interval}"
                    if state.get(key) != signal:
                        message = f"⚡ *New Signal!*\nSymbol: `{symbol}`\nInterval: `{interval}`\nSignal: *{signal}*"
                        send_telegram_message(message)
                        state[key] = signal

            except Exception as e:
                print(f"Error for {symbol} {interval}: {e}")
    
    save_state(state)

# Плануємо виконання кожні 5 хвилин
scheduler = BackgroundScheduler()
scheduler.add_job(monitor, 'interval', minutes=5)
scheduler.start()

@app.route('/')
def home():
    return "📡 DEX TG BOT Monitoring Running"

if __name__ == '__main__':
    app.run()