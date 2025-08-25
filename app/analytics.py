import json
import math
import time
from datetime import datetime, timezone
from typing import List, Dict, Tuple, Optional

import requests
import numpy as np
import pandas as pd

from app.config import BINANCE_REST, LOOKBACK_CANDLES

# ---------- Допоміжні утиліти ----------

def _ts_to_dt(ts_ms: int) -> datetime:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)

def now_utc():
    return datetime.now(tz=timezone.utc)

# ---------- Завантаження даних з Binance ----------

def get_klines(symbol: str, interval: str, limit: int = LOOKBACK_CANDLES) -> pd.DataFrame:
    """
    Повертає DataFrame з колонками: open_time, open, high, low, close, volume, close_time.
    """
    url = f"{BINANCE_REST}/api/v3/klines"
    params = {"symbol": symbol.replace("/", ""), "interval": interval, "limit": min(limit, 1000)}
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    raw = r.json()

    cols = [
        "open_time","open","high","low","close","volume","close_time","qav","num_trades",
        "taker_base","taker_quote","ignore"
    ]
    df = pd.DataFrame(raw, columns=cols)
    df["open_time"] = df["open_time"].astype(np.int64)
    df["close_time"] = df["close_time"].astype(np.int64)
    for col in ["open","high","low","close","volume"]:
        df[col] = df[col].astype(float)
    df["open_dt"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_dt"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    return df[["open_dt","close_dt","open","high","low","close","volume"]]

def get_price(symbol: str) -> float:
    url = f"{BINANCE_REST}/api/v3/ticker/price"
    r = requests.get(url, params={"symbol": symbol.replace("/", "")}, timeout=10)
    r.raise_for_status()
    return float(r.json()["price"])

def get_24h_tickers() -> List[Dict]:
    url = f"{BINANCE_REST}/api/v3/ticker/24hr"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return r.json()

def get_usdt_symbols(limit: int = 150) -> List[str]:
    """
    Топ USDT-пар за обсягом (наближено), обрізаємо список.
    """
    tickers = get_24h_tickers()
    usdt = [t for t in tickers if t["symbol"].endswith("USDT")]
    # сортуємо за quoteVolume (як рядок float)
    usdt.sort(key=lambda x: float(x.get("quoteVolume", 0.0)), reverse=True)
    return [t["symbol"] for t in usdt[:limit]]

# ---------- Індикатори ----------

def sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n).mean()

def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=series.index).rolling(n).mean()
    roll_down = pd.Series(down, index=series.index).rolling(n).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100 / (1 + rs))

def true_range(h, l, c_prev):
    return np.maximum(h - l, np.maximum(np.abs(h - c_prev), np.abs(l - c_prev)))

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    c_prev = df["close"].shift(1)
    tr = true_range(df["high"].values, df["low"].values, c_prev.values)
    return pd.Series(tr, index=df.index).rolling(n).mean()

# ---------- Рівні підтримки/опору ----------

def swing_points(df: pd.DataFrame, left: int = 2, right: int = 2) -> Tuple[pd.Series, pd.Series]:
    """
    Визначає локальні максимуми (свінг-хай) і мінімуми (свінг-лоу).
    left/right — скільки свічок ліворуч/праворуч має бути нижче/вище відповідно.
    """
    highs = df["high"].values
    lows = df["low"].values
    sw_high = []
    sw_low = []
    for i in range(left, len(df) - right):
        window_h = highs[i - left:i + right + 1]
        window_l = lows[i - left:i + right + 1]
        if highs[i] == window_h.max() and np.argmax(window_h) == left:
            sw_high.append(True)
        else:
            sw_high.append(False)
        if lows[i] == window_l.min() and np.argmin(window_l) == left:
            sw_low.append(True)
        else:
            sw_low.append(False)
    # паддінг до довжини
    sw_high = [False]*left + sw_high + [False]*right
    sw_low  = [False]*left + sw_low  + [False]*right
    return pd.Series(sw_high, index=df.index), pd.Series(sw_low, index=df.index)

def cluster_levels(levels: List[float], tolerance: float = 0.002) -> List[float]:
    """
    Кластеризація рівнів: якщо відхилення < tolerance (0.2%) — об'єднуємо.
    Повертаємо відсортований список.
    """
    if not levels:
        return []
    levels = sorted(levels)
    clusters = [[levels[0]]]
    for x in levels[1:]:
        if abs(x - np.mean(clusters[-1])) / np.mean(clusters[-1]) <= tolerance:
            clusters[-1].append(x)
        else:
            clusters.append([x])
    return sorted([float(np.mean(c)) for c in clusters])

def find_support_resistance(df: pd.DataFrame,
                            left: int = 2,
                            right: int = 2,
                            tolerance: float = 0.003,
                            max_levels: int = 6) -> Tuple[List[float], List[float]]:
    """
    Знаходить свінг-рівні та кластеризує.
    """
    sw_high, sw_low = swing_points(df, left=left, right=right)
    resistances = cluster_levels(df.loc[sw_high, "high"].tolist(), tolerance=tolerance)
    supports    = cluster_levels(df.loc[sw_low,  "low"].tolist(),  tolerance=tolerance)
    # беремо найбільш релевантні біля останньої ціни
    last = df["close"].iloc[-1]
    supports = sorted(supports, key=lambda x: abs(x - last))[:max_levels]
    resistances = sorted(resistances, key=lambda x: abs(x - last))[:max_levels]
    return supports, resistances

# ---------- Сигнали ----------

def generate_signal(df: pd.DataFrame) -> Dict:
    """
    Проста логіка:
    - якщо ціна тестує підтримку (+/- 0.5 ATR) і RSI < 35 -> Long
    - якщо ціна тестує опір (+/- 0.5 ATR) і RSI > 65 -> Short
    - якщо пробиття рівня > 0.7 ATR + збільшений обсяг -> відповідний сигнал
    """
    df = df.copy()
    df["rsi"] = rsi(df["close"])
    df["atr"] = atr(df)
    sup, res = find_support_resistance(df)
    last_price = df["close"].iloc[-1]
    last_rsi = df["rsi"].iloc[-1]
    last_atr = df["atr"].iloc[-1]
    vol = df["volume"].iloc[-1]
    vol_avg_