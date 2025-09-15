import pandas as pd
import ta  # Technical Analysis library

def apply_strategy(df):
    df = df.copy()
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