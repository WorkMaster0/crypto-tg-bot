import ccxt
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import BollingerBands

# Ініціалізація біржі
exchange = ccxt.binance()

def get_price(symbol):
    """Отримати поточну ціну"""
    try:
        ticker = exchange.fetch_ticker(symbol)
        return float(ticker['last'])
    except:
        return 0.0

def get_ohlcv(symbol, timeframe='1h', limit=100):
    """Отримати OHLCV дані"""
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()

def generate_signal(symbol):
    """Згенерувати торговий сигнал"""
    df = get_ohlcv(symbol)
    if df.empty:
        return "❌ Помилка отримання даних"
    
    # Розрахунок індикаторів
    df['rsi'] = RSIIndicator(df['close']).rsi()
    macd = MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    
    # Знаходження рівнів підтримки/опору
    support, resistance = find_support_resistance(df)
    
    # Аналіз сигналу
    current_price = df['close'].iloc[-1]
    last_rsi = df['rsi'].iloc[-1]
    
    signal = ""
    if last_rsi < 30 and current_price <= support * 1.02:
        signal = "🟢 BUY (Oversold at support)"
    elif last_rsi > 70 and current_price >= resistance * 0.98:
        signal = "🔴 SELL (Overbought at resistance)"
    else:
        signal = "🟡 HOLD (No clear signal)"
    
    return f"""
📊 *Аналіз {symbol}*
💰 Ціна: ${current_price:.2f}
📈 RSI: {last_rsi:.1f}
🔵 Підтримка: ${support:.2f}
🔴 Опір: ${resistance:.2f}
🚦 Сигнал: {signal}
"""

def find_support_resistance(df, window=20):
    """Знайти рівні підтримки та опору"""
    df['support'] = df['low'].rolling(window=window).min()
    df['resistance'] = df['high'].rolling(window=window).max()
    return df['support'].iloc[-1], df['resistance'].iloc[-1]

def trend_strength(symbol):
    """Визначити силу тренду"""
    df = get_ohlcv(symbol)
    if df.empty:
        return "❌ Помилка отримання даних"
    
    # EMA для різних періодів
    ema_20 = EMAIndicator(df['close'], window=20).ema_indicator()
    ema_50 = EMAIndicator(df['close'], window=50).ema_indicator()
    
    trend = "🟢 BULLISH" if ema_20.iloc[-1] > ema_50.iloc[-1] else "🔴 BEARISH"
    strength = abs((ema_20.iloc[-1] - ema_50.iloc[-1]) / ema_50.iloc[-1] * 100)
    
    return f"""
📈 *Тренд {symbol}*
Напрямок: {trend}
Сила: {strength:.1f}%
EMA 20: ${ema_20.iloc[-1]:.2f}
EMA 50: ${ema_50.iloc[-1]:.2f}
"""
