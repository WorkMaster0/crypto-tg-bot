import matplotlib.pyplot as plt
import io
from app.analytics import get_klines

def plot_candles(symbol: str, interval="1h", limit=50):
    candles = get_klines(symbol, interval, limit)
    closes = [c[3] for c in candles]

    plt.figure(figsize=(10,5))
    plt.plot(closes, label=f"{symbol} close prices")
    plt.legend()
    plt.title(f"{symbol} Chart ({interval})")
    plt.xlabel("Candles")
    plt.ylabel("Price")

    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    plt.close()
    return buffer