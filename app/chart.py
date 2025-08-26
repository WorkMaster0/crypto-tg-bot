import io
import numpy as np
import matplotlib
matplotlib.use("Agg")  # для серверів без дисплея
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from app.analytics import get_klines, ema, find_levels
from app.config import DEFAULT_INTERVAL, KLINES_LIMIT

def _to_dt(ts: np.ndarray):
    return [datetime.utcfromtimestamp(int(x)) for x in ts]

def plot_candles(symbol: str, interval: str = DEFAULT_INTERVAL, limit: int = 200, with_levels: bool = True) -> io.BytesIO:
    data = get_klines(symbol, interval=interval, limit=min(limit, KLINES_LIMIT))
    t, o, h, l, c = data["t"], data["o"], data["h"], data["l"], data["c"]
    dt = _to_dt(t)
    e50 = ema(c, 50)
    e200 = ema(c, 200)
    levels = find_levels(data) if with_levels else None

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)

    # Свічки
    for i in range(len(c)):
        color = "green" if c[i] >= o[i] else "red"
        ax.vlines(dt[i], l[i], h[i], linewidth=1, color=color, alpha=0.8)
        ax.vlines(dt[i], min(o[i], c[i]), max(o[i], c[i]), linewidth=6, color=color, alpha=0.8)

    # EMA
    ax.plot(dt, e50, linewidth=1.2, label="EMA50")
    ax.plot(dt, e200, linewidth=1.2, label="EMA200")

    # Рівні
    if levels:
        for s in levels["supports"]:
            ax.hlines(s, dt[0], dt[-1], linestyles="dashed", linewidth=0.8)
        for r in levels["resistances"]:
            ax.hlines(r, dt[0], dt[-1], linestyles="dashed", linewidth=0.8)

    ax.set_title(f"{symbol.upper()}  [{interval}]")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Price")

    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    fig.autofmt_xdate()

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf