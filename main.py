from binance_data import get_klines
from strategy import apply_strategy
from telegram_alert import send_telegram_message

symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT']  # ← доповнимо до топ-30
intervals = ['15m', '1h', '4h', '1d']

for symbol in symbols:
    for interval in intervals:
        try:
            df = get_klines(symbol, interval)
            signal = apply_strategy(df)

            if signal:
                msg = f"⚡ *Signal Detected*\nSymbol: `{symbol}`\nInterval: `{interval}`\nSignal: *{signal}*"
                send_telegram_message(msg)

        except Exception as e:
            print(f"Error with {symbol} {interval}: {e}")