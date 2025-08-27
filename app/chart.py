import mplfinance as mpf
import matplotlib.pyplot as plt
from io import BytesIO
from app.analytics import get_ohlcv  # ← ЦЕЙ ІМПОРТ ПОВИНЕН БУТИ

def plot_candles(symbol, timeframe='1h'):
    """Побудувати свічковий графік"""
    df = get_ohlcv(symbol, timeframe)
    if df.empty:
        return None
    
    # Підготовка даних
    df.set_index('timestamp', inplace=True)
    
    # Створення графіка
    fig, axes = mpf.plot(
        df,
        type='candle',
        style='charles',
        volume=True,
        returnfig=True,
        figsize=(10, 8)
    )
    
    # Збереження в буфер
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    
    return buf
