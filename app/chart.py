import io
from typing import List, Optional
import matplotlib
matplotlib.use("Agg")  # без GUI
import matplotlib.pyplot as plt

def plot_levels(df, supports: List[float], resistances: List[float], title: str = "", last_price: Optional[float]=None) -> bytes:
    """
    Створює PNG графік (OHLC-клоуз) з горизонтальними лініями підтримок/опор.
    Повертає байти PNG.
    """
    fig = plt.figure(figsize=(10, 5))
    ax = plt.gca()
    ax.plot(df["close_dt"], df["close"], linewidth=1.4)
    for s in supports:
        ax.axhline(s, linestyle="--", linewidth=0.9)
    for r in resistances:
        ax.axhline(r, linestyle="--", linewidth=0.9)
    if last_price is not None:
        ax.axhline(last_price, linestyle=":", linewidth=1.2)
    ax.set_title(title)
    ax.set_xlabel("time")
    ax.set_ylabel("price")
    fig.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=140)
    plt.close(fig)
    buf.seek(0)
    return buf.read()