from binance.client import Client
import pandas as pd
import os

client = Client(api_key='', api_secret='')

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

def get_top_symbols(limit=30):
    tickers = client.get_ticker()
    sorted_tickers = sorted(tickers, key=lambda x: float(x['quoteVolume']), reverse=True)
    symbols = [t['symbol'] for t in sorted_tickers if t['symbol'].endswith('USDT')]
    return symbols[:limit]