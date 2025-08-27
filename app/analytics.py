import ccxt
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, EMAIndicator, IchimokuIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice

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

def find_support_resistance(df, window=20):
    """Знайти рівні підтримки та опору"""
    df['support'] = df['low'].rolling(window=window).min()
    df['resistance'] = df['high'].rolling(window=window).max()
    return df['support'].iloc[-1], df['resistance'].iloc[-1]

def calculate_super_trend(df, period=10, multiplier=3):
    """Розрахунок SuperTrend індикатора"""
    hl2 = (df['high'] + df['low']) / 2
    atr = AverageTrueRange(df['high'], df['low'], df['close'], window=period).average_true_range()
    
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)
    
    super_trend = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)
    
    for i in range(1, len(df)):
        if df['close'].iloc[i] > upper_band.iloc[i-1]:
            super_trend.iloc[i] = lower_band.iloc[i]
            direction.iloc[i] = 1
        elif df['close'].iloc[i] < lower_band.iloc[i-1]:
            super_trend.iloc[i] = upper_band.iloc[i]
            direction.iloc[i] = -1
        else:
            super_trend.iloc[i] = super_trend.iloc[i-1]
            direction.iloc[i] = direction.iloc[i-1]
    
    return super_trend * direction

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
    last_macd = df['macd'].iloc[-1]
    last_macd_signal = df['macd_signal'].iloc[-1]
    
    signal = ""
    if last_rsi < 30 and current_price <= support * 1.02 and last_macd > last_macd_signal:
        signal = "🟢 STRONG BUY (Oversold + Support + MACD Bullish)"
    elif last_rsi > 70 and current_price >= resistance * 0.98 and last_macd < last_macd_signal:
        signal = "🔴 STRONG SELL (Overbought + Resistance + MACD Bearish)"
    elif last_rsi < 35 and current_price <= support * 1.05:
        signal = "🟡 WEAK BUY (Near support, wait confirmation)"
    elif last_rsi > 65 and current_price >= resistance * 0.95:
        signal = "🟡 WEAK SELL (Near resistance, wait confirmation)"
    else:
        signal = "⚪️ HOLD (No clear signal)"
    
    return f"""
📊 *Аналіз {symbol}*
💰 Ціна: ${current_price:.2f}
📈 RSI: {last_rsi:.1f}
📊 MACD: {last_macd:.4f}
🔵 Підтримка: ${support:.2f}
🔴 Опір: ${resistance:.2f}
🚦 Сигнал: {signal}
"""

def trend_strength(symbol):
    """Визначити силу тренду"""
    df = get_ohlcv(symbol)
    if df.empty:
        return "❌ Помилка отримання даних"
    
    # EMA для різних періодів
    ema_20 = EMAIndicator(df['close'], window=20).ema_indicator()
    ema_50 = EMAIndicator(df['close'], window=50).ema_indicator()
    ema_100 = EMAIndicator(df['close'], window=100).ema_indicator()
    
    trend = "🟢 STRONG BULLISH" if ema_20.iloc[-1] > ema_50.iloc[-1] > ema_100.iloc[-1] else "🔴 STRONG BEARISH" if ema_20.iloc[-1] < ema_50.iloc[-1] < ema_100.iloc[-1] else "🟡 SIDEWAYS"
    strength = abs((ema_20.iloc[-1] - ema_50.iloc[-1]) / ema_50.iloc[-1] * 100)
    
    return f"""
📈 *Тренд {symbol}*
Напрямок: {trend}
Сила: {strength:.1f}%
EMA 20: ${ema_20.iloc[-1]:.2f}
EMA 50: ${ema_50.iloc[-1]:.2f}
EMA 100: ${ema_100.iloc[-1]:.2f}
"""

def advanced_analysis(symbol):
    """Розширений аналіз з потужними індикаторами"""
    df = get_ohlcv(symbol, '4h', 200)
    
    if df.empty:
        return "❌ Помилка отримання даних"
    
    # Ішимоку
    ichimoku = IchimokuIndicator(df['high'], df['low'])
    df['ichimoku_a'] = ichimoku.ichimoku_a()
    df['ichimoku_b'] = ichimoku.ichimoku_b()
    df['ichimoku_base'] = ichimoku.ichimoku_base_line()
    df['ichimoku_conversion'] = ichimoku.ichimoku_conversion_line()
    
    # ATR (волатильність)
    atr = AverageTrueRange(df['high'], df['low'], df['close'])
    df['atr'] = atr.average_true_range()
    
    # Стохастик
    stoch = StochasticOscillator(df['high'], df['low'], df['close'])
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()
    
    # VWAP
    vwap = VolumeWeightedAveragePrice(df['high'], df['low'], df['close'], df['volume'])
    df['vwap'] = vwap.volume_weighted_average_price()
    
    # Супертренд
    df['super_trend'] = calculate_super_trend(df)
    
    # Bollinger Bands
    bb = BollingerBands(df['close'])
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    
    current = df.iloc[-1]
    price = current['close']
    
    # Аналіз сигналів
    signals = []
    
    # Ішимоку сигнал
    if price > current['ichimoku_a'] and price > current['ichimoku_b']:
        signals.append("🟢 Ішимоку: STRONG BULLISH")
    elif price < current['ichimoku_a'] and price < current['ichimoku_b']:
        signals.append("🔴 Ішимоку: STRONG BEARISH")
    else:
        signals.append("🟡 Ішимоку: NEUTRAL")
    
    # Супертренд сигнал
    if current['super_trend'] > 0:
        signals.append(f"🟢 Супертренд: BUY (SL: ${current['super_trend']:.2f})")
    else:
        signals.append(f"🔴 Супертренд: SELL (SL: ${abs(current['super_trend']):.2f})")
    
    # Стохастик
    if current['stoch_k'] < 20 and current['stoch_d'] < 20:
        signals.append("📉 Стохастик: STRONG OVERSOLD")
    elif current['stoch_k'] > 80 and current['stoch_d'] > 80:
        signals.append("📈 Стохастик: STRONG OVERBOUGHT")
    
    # VWAP
    if price > current['vwap']:
        signals.append("🟢 VWAP: BULLISH (Above average)")
    else:
        signals.append("🔴 VWAP: BEARISH (Below average)")
    
    # Bollinger Bands
    if price < current['bb_lower']:
        signals.append("📉 BB: OVERSOLD (Below lower band)")
    elif price > current['bb_upper']:
        signals.append("📈 BB: OVERBOUGHT (Above upper band)")
    
    # Визначення загального сигналу
    buy_signals = sum(1 for s in signals if '🟢' in s or '📉' in s)
    sell_signals = sum(1 for s in signals if '🔴' in s or '📈' in s)
    
    if buy_signals >= 3 and sell_signals <= 1:
        final_signal = "🎯 STRONG BUY SIGNAL"
    elif sell_signals >= 3 and buy_signals <= 1:
        final_signal = "🎯 STRONG SELL SIGNAL"
    else:
        final_signal = "⚪️ MIXED SIGNALS (Wait confirmation)"
    
    return f"""
🎯 *РОЗШИРЕНИЙ АНАЛІЗ {symbol}*
💰 Ціна: ${price:.2f}
📊 ATR (волатильність): {current['atr']:.2f}
📈 VWAP: ${current['vwap']:.2f}

📊 *СИГНАЛИ ІНДИКАТОРІВ:*
{chr(10).join(signals)}

💡 *РЕКОМЕНДАЦІЯ:*
{final_signal}

🎯 *РІВНІ СТОП-ЛОСС:*
- Консервативний: {current['atr']*1.5:.2f}
- Агресивний: {current['atr']*2:.2f}

🎯 *РІВНІ ТЕЙК-ПРОФІТ:*
- Консервативний: {current['atr']*3:.2f}  
- Агресивний: {current['atr']*4:.2f}
"""
