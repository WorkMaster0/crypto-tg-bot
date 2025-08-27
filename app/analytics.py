# analytics.py
import ccxt
import pandas as pd
import numpy as np

# 🔹 Підключення до Binance
exchange = ccxt.binance({
    'enableRateLimit': True,
})

# 🔹 Отримання ціни
def get_price(symbol: str) -> float:
    ticker = exchange.fetch_ticker(symbol)
    return ticker['last']

# 🔹 Генерація сигналу на купівлю/продаж
def generate_signal(symbol: str) -> str:
    df = fetch_ohlcv(symbol)
    close = df['close'].iloc[-1]
    ema_short = df['close'].ewm(span=10, adjust=False).mean().iloc[-1]
    ema_long = df['close'].ewm(span=50, adjust=False).mean().iloc[-1]

    if ema_short > ema_long:
        return "🟢 Buy Signal"
    elif ema_short < ema_long:
        return "🔴 Sell Signal"
    else:
        return "⚪ Neutral"

# 🔹 Сила тренду
def trend_strength(symbol: str) -> str:
    df = fetch_ohlcv(symbol)
    rsi_val = rsi_indicator(symbol)
    if rsi_val > 70:
        return "📉 Overbought"
    elif rsi_val < 30:
        return "📈 Oversold"
    else:
        return "↔ Neutral trend"

# 🔹 Рівні підтримки та опору
def support_resistance_levels(symbol: str) -> tuple:
    df = fetch_ohlcv(symbol)
    high = df['high'].iloc[-20:]
    low = df['low'].iloc[-20:]
    support = low.min()
    resistance = high.max()
    return support, resistance

# 🔹 RSI
def rsi_indicator(symbol: str, period: int = 14) -> float:
    df = fetch_ohlcv(symbol)
    delta = df['close'].diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1]

# 🔹 EMA
def ema_indicator(symbol: str, period: int = 20) -> float:
    df = fetch_ohlcv(symbol)
    return df['close'].ewm(span=period, adjust=False).mean().iloc[-1]

# 🔹 SMA
def sma_indicator(symbol: str, period: int = 20) -> float:
    df = fetch_ohlcv(symbol)
    return df['close'].rolling(period).mean().iloc[-1]

# 🔹 MACD
def macd_indicator(symbol: str, fast: int = 12, slow: int = 26, signal_period: int = 9) -> float:
    df = fetch_ohlcv(symbol)
    ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    return macd.iloc[-1] - signal.iloc[-1]

# 🔹 Отримання OHLCV (дані свічок)
def fetch_ohlcv(symbol: str, timeframe: str = '1h', limit: int = 100) -> pd.DataFrame:
    data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df