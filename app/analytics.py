import yfinance as yf
import pandas as pd
import talib

# 🔹 Поточна ціна
def get_price(symbol: str) -> float:
    data = yf.Ticker(symbol + "-USD").history(period="1d")
    if data.empty:
        return 0.0
    return data['Close'][-1]

# 🔹 Сигнали для трейдингу
def generate_signal(symbol: str) -> str:
    data = yf.Ticker(symbol + "-USD").history(period="60d", interval="1d")
    if data.empty:
        return "⚠️ Data not available."
    
    close = data['Close']
    rsi = talib.RSI(close, timeperiod=14)
    ma_short = talib.SMA(close, timeperiod=10)
    ma_long = talib.SMA(close, timeperiod=50)

    signal = ""
    if rsi.iloc[-1] < 30:
        signal += "📉 Oversold – possible BUY signal.\n"
    elif rsi.iloc[-1] > 70:
        signal += "📈 Overbought – possible SELL signal.\n"
    
    if ma_short.iloc[-1] > ma_long.iloc[-1]:
        signal += "✅ Uptrend detected by SMA crossover.\n"
    else:
        signal += "⚠️ Downtrend detected by SMA crossover.\n"
    
    return signal

# 🔹 Сила тренду
def trend_strength(symbol: str) -> str:
    data = yf.Ticker(symbol + "-USD").history(period="30d", interval="1d")
    if data.empty:
        return "⚠️ Data not available."
    
    close = data['Close']
    ema_short = talib.EMA(close, timeperiod=5)
    ema_long = talib.EMA(close, timeperiod=20)

    strength = (ema_short.iloc[-1] - ema_long.iloc[-1]) / ema_long.iloc[-1] * 100
    return f"📊 Trend strength: {strength:.2f}%"

# 🔹 Технічні індикатори
def calculate_indicators(symbol: str) -> dict:
    data = yf.Ticker(symbol + "-USD").history(period="60d", interval="1d")
    if data.empty:
        return {"Error": "Data not available"}
    
    close = data['Close']
    high = data['High']
    low = data['Low']

    indicators = {
        "RSI": talib.RSI(close, timeperiod=14).iloc[-1],
        "EMA_10": talib.EMA(close, timeperiod=10).iloc[-1],
        "EMA_50": talib.EMA(close, timeperiod=50).iloc[-1],
        "MACD": talib.MACD(close)[0].iloc[-1],
        "ATR": talib.ATR(high, low, close, timeperiod=14).iloc[-1]
    }
    return indicators

# 🔹 Рівні підтримки і опору
def get_levels(symbol: str) -> tuple:
    data = yf.Ticker(symbol + "-USD").history(period="90d", interval="1d")
    if data.empty:
        return (0.0, 0.0)
    
    close = data['Close']
    pivot = (data['High'].max() + data['Low'].min() + close.iloc[-1]) / 3
    support = pivot - (data['High'].max() - data['Low'].min()) / 2
    resistance = pivot + (data['High'].max() - data['Low'].min()) / 2
    return (round(support, 2), round(resistance, 2))