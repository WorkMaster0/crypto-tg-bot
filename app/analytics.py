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
    vol_avg = df["volume"].rolling(20).mean().iloc[-1]

    near_sup = min(sup, key=lambda x: abs(x - last_price)) if sup else None
    near_res = min(res, key=lambda x: abs(x - last_price)) if res else None

    decision = "NEUTRAL"
    confidence = 0.5
    reason = []

    if near_sup and abs(last_price - near_sup) <= 0.5 * last_atr and last_rsi < 35:
        decision = "LONG"
        confidence = 0.6
        reason.append(f"Тест підтримки ~{near_sup:.4f} + RSI {last_rsi:.1f}")
        if vol > 1.2 * vol_avg:
            confidence += 0.1
            reason.append("Обсяг вище середнього")
    elif near_res and abs(last_price - near_res) <= 0.5 * last_atr and last_rsi > 65:
        decision = "SHORT"
        confidence = 0.6
        reason.append(f"Тест опору ~{near_res:.4f} + RSI {last_rsi:.1f}")
        if vol > 1.2 * vol_avg:
            confidence += 0.1
            reason.append("Обсяг вище середнього")
    else:
        # Пробиття
        if near_res and last_price - near_res > 0.7 * last_atr and vol > 1.1 * vol_avg:
            decision = "LONG"
            confidence = 0.55
            reason.append("Пробиття опору з обсягом")
        elif near_sup and near_sup - last_price > 0.7 * last_atr and vol > 1.1 * vol_avg:
            decision = "SHORT"
            confidence = 0.55
            reason.append("Пробиття підтримки з обсягом")

    confidence = float(max(0.0, min(0.95, confidence)))
    payload = {
        "decision": decision,
        "confidence": confidence,
        "price": last_price,
        "rsi": float(last_rsi),
        "atr": float(last_atr),
        "supports": sup,
        "resistances": res,
        "near_support": near_sup,
        "near_resistance": near_res,
        "volume": float(vol),
        "volume_avg": float(vol_avg),
    }
    payload["reason"] = "; ".join(reason) if reason else "Сигнал слабкий або відсутній"
    return payload

# ---------- Бек-тест (спрощено) ----------

def backtest_level_rsi(df: pd.DataFrame, rsi_buy=35, rsi_sell=65, atr_touch=0.5) -> Dict:
    """
    Правила в бек-тесті:
    - Long: коли ціна в межах 0.5*ATR від підтримки і RSI<35. Вихід: ціна доходить до найближчого опору або стоп під мінімум (1*ATR).
    - Short: дзеркально.
    Результат: кількість угод, winrate, PnL у % від ціни.
    """
    df = df.copy()
    df["rsi"] = rsi(df["close"])
    df["atr"] = atr(df)
    sup, res = find_support_resistance(df)
    trades = []
    for i in range(30, len(df) - 2):
        price = df["close"].iloc[i]
        r = df["rsi"].iloc[i]
        a = df["atr"].iloc[i]
        if not np.isfinite(a) or a <= 0:
            continue
        if sup:
            near_sup = min(sup, key=lambda x: abs(x - price))
        else:
            near_sup = None
        if res:
            near_res = min(res, key=lambda x: abs(x - price))
        else:
            near_res = None

        # Long
        if near_sup and abs(price - near_sup) <= atr_touch * a and r < rsi_buy:
            tp = min(res) if res else price + 2*a
            sl = price - a
            exit_price = None
            for j in range(i+1, min(i+60, len(df))):
                low = df["low"].iloc[j]
                high = df["high"].iloc[j]
                # перевірка стопа/тейку
                if low <= sl:
                    exit_price = sl
                    break
                if high >= tp:
                    exit_price = tp
                    break
            if exit_price is None:
                exit_price = df["close"].iloc[min(i+60, len(df)-1)]
            pnl = (exit_price - price) / price
            trades.append(pnl)

        # Short
        if near_res and abs(price - near_res) <= atr_touch * a and r > rsi_sell:
            tp = max(sup) if sup else price - 2*a
            sl = price + a
            exit_price = None
            for j in range(i+1, min(i+60, len(df))):
                low = df["low"].iloc[j]
                high = df["high"].iloc[j]
                if high >= sl:
                    exit_price = sl
                    break
                if low <= tp:
                    exit_price = tp
                    break
            if exit_price is None:
                exit_price = df["close"].iloc[min(i+60, len(df)-1)]
            pnl = (price - exit_price) / price
            trades.append(pnl)

    if not trades:
        return {"trades": 0, "winrate": 0.0, "pnl_pct": 0.0}
    wins = [x for x in trades if x > 0]
    pnl_pct = sum(trades) * 100
    return {"trades": len(trades), "winrate": round(len(wins)/len(trades)*100, 2), "pnl_pct": round(pnl_pct, 2)}

# ---------- Alpha/Heatmap/Сканери ----------

def heatmap_top_moves(limit: int = 15) -> List[Tuple[str, float]]:
    tickers = get_24h_tickers()
    # зміна у %
    pairs = [(t["symbol"], float(t.get("priceChangePercent", 0.0))) for t in tickers if t["symbol"].endswith("USDT")]
    pairs.sort(key=lambda x: abs(x[1]), reverse=True)
    return pairs[:limit]

def alpha_volatility_compression(top_n: int = 10, interval: str = "1h") -> List[Tuple[str, float]]:
    """
    Унікальна фіча: знаходимо пари з найменшим ATR% за останні 100 свічок -> можливий майбутній прорив.
    ATR% = ATR / Close * 100
    """
    symbols = get_usdt_symbols(limit=80)
    results = []
    for s in symbols:
        try:
            df = get_klines(s, interval, limit=150)
            a = atr(df, 14)
            atrp = (a / df["close"]) * 100
            val = float(atrp.iloc[-1])
            if np.isfinite(val):
                results.append((s, val))
            time.sleep(0.05)  # не спамимо API
        except Exception:
            continue
    results.sort(key=lambda x: x[1])  # найменший ATR%
    return results[:top_n]

# ---------- Розмір позиції / Risk ----------

def position_size(balance: float, risk_pct: float, entry: float, stop: float) -> Dict:
    risk_amount = balance * (risk_pct / 100.0)
    per_unit_risk = abs(entry - stop)
    if per_unit_risk <= 0:
        return {"size": 0.0, "risk_amount": 0.0, "rr": None}
    size = risk_amount / per_unit_risk
    rr2 = abs((entry - (entry + (entry - stop) * 2)) / (entry - stop))  # умовний RR=2 приклад
    return {"size": size, "risk_amount": risk_amount, "rr": rr2}