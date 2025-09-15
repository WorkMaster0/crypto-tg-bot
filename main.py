import os
import json
import requests
import pandas as pd
from flask import Flask
from apscheduler.schedulers.background import BackgroundScheduler
from binance.client import Client
import ta

# === Env config ===
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

# === Flask init ===
app = Flask(__name__)

# === Binance init ===
client = Client(api_key='', api_secret='')

# === State file ===
STATE_FILE = 'state.json'

# === Load/save signal state ===
def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_state(state):
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f)

state = load_state()

# === Get top symbols by volume ===
def get_top_symbols(limit=30):
    tickers = client.get_ticker()
    sorted_tickers = sorted(tickers, key=lambda x: float(x['quoteVolume']), reverse=True)
    symbols = [t['symbol'] for t in sorted_tickers if t['symbol'].endswith('USDT')]
    return symbols[:limit]

# === Get historical kline data ===
def get_klines(symbol, interval, limit=100):
    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(klines, columns=[
        'Open time', 'Open', 'High', 'Low', 'Close', 'Volume',
        'Close time', 'Quote asset volume', 'Number of trades',
        'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'
    ])
    df = df[['Open time', 'Open', 'High', 'Low', 'Close', 'Volume']]
    df = df.astype(float)
    df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')
    df.set_index('Open time', inplace=True)
    return df

# === Strategy logic ===
def apply_strategy(df):
    df['ATR_10'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=10)
    df['ATR_50'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=50)
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    macd = ta.trend.MACD(df['Close'])
    df['MACD_hist'] = macd.macd_diff()
    df['Donchian_High'] = df['High'].rolling(window=20).max()
    df['Donchian_Low'] = df['Low'].rolling(window=20).min()

    latest = df.iloc[-1]

    long_signal = (
        latest['ATR_10'] < latest['ATR_50'] and
        latest['Close'] > latest['Donchian_High'] and
        (latest['MACD_hist'] > 0 or latest['RSI'] > 55)
    )

    short_signal = (
        latest['ATR_10'] < latest['ATR_50'] and
        latest['Close'] < latest['Donchian_Low'] and
        (latest['MACD_hist'] < 0 or latest['RSI'] < 45)
    )

    if long_signal:
        return 'LONG'
    elif short_signal:
        return 'SHORT'
    else:
        return None

# === Telegram alert ===
def send_telegram_message(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": text,
        "parse_mode": "Markdown"
    }
    try:
        requests.post(url, json=payload)
    except Exception as e:
        print(f"Telegram error: {e}")

# === Monitoring logic ===
def monitor():
    print("🔎 Running signal scan...")
    symbols = get_top_symbols()
    intervals = ['15m', '1h', '4h', '1d']

    for symbol in symbols:
        for interval in intervals:
            try:
                df = get_klines(symbol, interval)
                signal = apply_strategy(df)

                key = f"{symbol}_{interval}"
                if signal and state.get(key) != signal:
                    message = f"⚡ *New Signal!*\nSymbol: `{symbol}`\nInterval: `{interval}`\nSignal: *{signal}*"
                    send_telegram_message(message)
                    state[key] = signal

            except Exception as e:
                print(f"Error with {symbol} {interval}: {e}")

    save_state(state)

# === Scheduler ===
scheduler = BackgroundScheduler()
scheduler.add_job(monitor, 'interval', minutes=5)
scheduler.start()

# === Flask route ===
@app.route('/')
def home():
    return "✅ DEX TG BOT is Live and Scanning..."

# === Start app ===
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)