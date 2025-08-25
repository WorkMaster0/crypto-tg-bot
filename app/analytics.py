import requests
import numpy as np
from app.config import BINANCE_API_URL

def get_price(symbol: str) -> float:
    url = f"{BINANCE_API_URL}/ticker/price"
    response = requests.get(url, params={"symbol": symbol})
    data = response.json()
    return float(data["price"])

def get_klines(symbol: str, interval="1h", limit=100):
    url = f"{BINANCE_API_URL}/klines"
    response = requests.get(url, params={"symbol": symbol, "interval": interval, "limit": limit})
    data = response.json()
    return [(float(x[1]), float(x[2]), float(x[3]), float(x[4])) for x in data]  # (open, high, low, close)

def support_resistance(symbol: str, interval="1h", limit=200):
    candles = get_klines(symbol, interval, limit)
    highs = [c[1] for c in candles]
    lows = [c[2] for c in candles]

    resistance = np.percentile(highs, 90)  # верхній рівень
    support = np.percentile(lows, 10)      # нижній рівень
    return support, resistance

def generate_signal(symbol: str):
    price = get_price(symbol)
    support, resistance = support_resistance(symbol)

    if price <= support * 1.01:
        return f"🟢 LONG signal on {symbol}\nPrice: {price:.2f}\nSupport: {support:.2f} | Resistance: {resistance:.2f}"
    elif price >= resistance * 0.99:
        return f"🔴 SHORT signal on {symbol}\nPrice: {price:.2f}\nSupport: {support:.2f} | Resistance: {resistance:.2f}"
    else:
        return f"⚪ Neutral zone on {symbol}\nPrice: {price:.2f}\nSupport: {support:.2f} | Resistance: {resistance:.2f}"

def trend_strength(symbol: str):
    candles = get_klines(symbol, "1h", 50)
    closes = [c[3] for c in candles]
    x = np.arange(len(closes))
    slope = np.polyfit(x, closes, 1)[0]

    if slope > 0:
        return f"📈 {symbol} is in uptrend (slope {slope:.2f})"
    elif slope < 0:
        return f"📉 {symbol} is in downtrend (slope {slope:.2f})"
    else:
        return f"➖ {symbol} is sideways"