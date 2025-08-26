import time
import math
import requests
import numpy as np
from typing import Dict, List, Tuple
from app.config import (
    BINANCE_BASES, HTTP_TIMEOUT, KLINES_LIMIT, DEFAULT_INTERVAL,
    PIVOT_LEFT_RIGHT, MAX_LEVELS
)

# ---------- BINANCE HELPERS ----------

def _binance_get(path: str, params: Dict) -> dict:
    last_error = None
    for base in BINANCE_BASES:
        try:
            r = requests.get(f"{base}{path}", params=params, timeout=HTTP_TIMEOUT)
            if r.status_code == 200:
                return r.json()
            last_error = RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
        except Exception as e:
            last_error = e
            continue
    raise last_error or RuntimeError("Binance unreachable")

def normalize_symbol(s: str) -> str:
    s = s.strip().upper().replace("/", "")
    return s

def get_price(symbol: str) -> float:
    symbol = normalize_symbol(symbol)
    data = _binance_get("/api/v3/ticker/price", {"symbol": symbol})
    return float(data["price"])

def get_klines(symbol: str, interval: str = DEFAULT_INTERVAL, limit: int = KLINES_LIMIT) -> Dict[str, np.ndarray]:
    symbol = normalize_symbol(symbol)
    data = _binance_get("/api/v3/klines", {"symbol": symbol, "interval": interval, "limit": limit})
    # kline: [ openTime, open, high, low, close, volume, closeTime, ... ]
    arr = np.array(data, dtype=object)
    ts = (arr[:,0].astype(np.int64) // 1000).astype(np.int64)
    o = arr[:,1].astype(float)
    h = arr[:,2].astype(float)
    l = arr[:,3].astype(float)
    c = arr[:,4].astype(float)
    v = arr[:,5].astype(float)
    return {"t": ts, "o": o, "h": h, "l": l, "c": c, "v": v}

# ---------- TECH UTILS ----------

def ema(series: np.ndarray, period: int) -> np.ndarray:
    alpha = 2.0 / (period + 1.0)
    out = np.empty_like(series, dtype=float)
    out[0] = series[0]
    for i in range(1, len(series)):
        out[i] = alpha * series[i] + (1 - alpha) * out[i-1]
    return out

def atr(h: np.ndarray, l: np.ndarray, c: np.ndarray, period: int = 14) -> np.ndarray:
    tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
    tr[0] = h[0] - l[0]
    return ema(tr, period)

# ---------- LEVELS (Support/Resistance) ----------

def _pivot_high(h: np.ndarray, i: int, left_right: int) -> bool:
    left = max(0, i - left_right)
    right = min(len(h), i + left_right + 1)
    return h[i] == np.max(h[left:right])

def _pivot_low(l: np.ndarray, i: int, left_right: int) -> bool:
    left = max(0, i - left_right)
    right = min(len(l), i + left_right + 1)
    return l[i] == np.min(l[left:right])

def find_levels(candles: Dict[str, np.ndarray],
                left_right: int = PIVOT_LEFT_RIGHT,
                max_levels: int = MAX_LEVELS) -> Dict[str, List[float]]:
    h, l, c = candles["h"], candles["l"], candles["c"]
    last_price = c[-1]
    _atr = atr(h, l, c, 14)[-1]
    tol = max(_atr * 0.5, last_price * 0.002)  # 0.5 ATR або 0.2%

    highs, lows = [], []
    for i in range(left_right, len(h) - left_right):
        if _pivot_high(h, i, left_right):
            highs.append(h[i])
        if _pivot_low(l, i, left_right):
            lows.append(l[i])

    # Кластеризація рівнів за допуском tol
    def cluster(levels: List[float]) -> List[float]:
        if not levels: return []
        levels = sorted(levels)
        clusters: List[List[float]] = [[levels[0]]]
        for x in levels[1:]:
            if abs(x - np.mean(clusters[-1])) <= tol:
                clusters[-1].append(x)
            else:
                clusters.append([x])
        # усереднюємо кластери і сортуємо за кількістю торкань (більше — важливіші), потім за близькістю до ціни
        weighted = [(float(np.mean(g)), len(g)) for g in clusters]
        weighted.sort(key=lambda e: (-e[1], abs(e[0] - last_price)))
        return [w[0] for w in weighted[:max_levels]]

    resistances = cluster(highs)
    supports = cluster(lows)

    # гарантовано повертаємо відсортовані від меншого до більшого
    supports = sorted(set(supports))
    resistances = sorted(set(resistances))

    # найближчі до поточної ціни
    near_support = max([s for s in supports if s <= last_price], default=None)
    near_resist = min([r for r in resistances if r >= last_price], default=None)

    return {
        "supports": supports,
        "resistances": resistances,
        "near_support": near_support,
        "near_resistance": near_resist,
        "atr": float(_atr),
        "tolerance": float(tol),
        "last_price": float(last_price),
    }

# ---------- TREND & SIGNALS ----------

def trend_strength_text(candles: Dict[str, np.ndarray]) -> str:
    c = candles["c"]
    ema50 = ema(c, 50)
    ema200 = ema(c, 200)
    slope = (ema50[-1] - ema50[-10]) / (ema50[-10] + 1e-9) * 100.0  # % за 10 свічок
    state = "UP" if ema50[-1] > ema200[-1] else ("DOWN" if ema50[-1] < ema200[-1] else "FLAT")
    vol = np.std(c[-50:]) / (np.mean(c[-50:]) + 1e-9) * 100.0
    return f"Trend: <b>{state}</b> | EMA50-200 Δ: {((ema50[-1]-ema200[-1])/ema200[-1]*100):.2f}% | Slope(50): {slope:.2f}% | Volatility(50): {vol:.2f}%"

def generate_signal_text(symbol: str, interval: str = DEFAULT_INTERVAL) -> str:
    candles = get_klines(symbol, interval=interval, limit=KLINES_LIMIT)
    c, h, l = candles["c"], candles["h"], candles["l"]
    last = c[-1]
    e50 = ema(c, 50)
    e200 = ema(c, 200)
    _atr = atr(h, l, c, 14)[-1]
    levels = find_levels(candles)

    trend = "UP" if e50[-1] > e200[-1] else ("DOWN" if e50[-1] < e200[-1] else "FLAT")
    sup = levels["near_support"]
    res = levels["near_resistance"]

    # логіка сигналів
    txt = [f"📊 <b>{symbol.upper()}</b> [{interval}]  |  Price: <b>{last:.4f}</b>"]
    txt.append(trend_strength_text(candles))

    if sup and last > sup and (last - sup) <= max(_atr, last*0.004):
        # відбій від підтримки
        stop = sup - max(_atr*0.5, last*0.003)
        tp = res if res else last + 2.0 * _atr
        txt.append(f"✅ <b>LONG idea</b>: entry ~{last:.4f}, SL {stop:.4f}, TP {tp:.4f} (near support)")
    if res and last < res and (res - last) <= max(_atr, last*0.004):
        # відбій від опору
        stop = res + max(_atr*0.5, last*0.003)
        tp = sup if sup else last - 2.0 * _atr
        txt.append(f"❌ <b>SHORT idea</b>: entry ~{last:.4f}, SL {stop:.4f}, TP {tp:.4f} (near resistance)")

    # пробій
    if res and last > res * 1.001 and trend == "UP":
        stop = res - max(_atr*0.5, last*0.003)
        tp = last + 2.0 * _atr
        txt.append(f"🚀 <b>Breakout LONG</b> above {res:.4f}: SL {stop:.4f}, TP {tp:.4f}")
    if sup and last < sup * 0.999 and trend == "DOWN":
        stop = sup + max(_atr*0.5, last*0.003)
        tp = last - 2.0 * _atr
        txt.append(f"🔻 <b>Breakdown SHORT</b> below {sup:.4f}: SL {stop:.4f}, TP {tp:.4f}")

    if len(txt) == 2:
        txt.append("ℹ️ Чітких точок входу не знайдено. Зачекайте нової свічки або змініть інтервал.")

    # список основних рівнів
    lv_s = ", ".join(f"{x:.4f}" for x in levels["supports"][:MAX_LEVELS])
    lv_r = ", ".join(f"{x:.4f}" for x in levels["resistances"][:MAX_LEVELS])
    txt.append(f"Levels → S: [{lv_s}] | R: [{lv_r}]  | ATR(14): {levels['atr']:.4f}")

    return "\n".join(txt)

# ---------- HEATMAP (Top movers) ----------

def top_movers(limit: int = 10) -> List[Tuple[str, float, float]]:
    """Return list of (symbol, change%, quoteVolume USDT) for USDT pairs."""
    data = _binance_get("/api/v3/ticker/24hr", {})
    movers = []
    for item in data:
        s = item.get("symbol","")
        if not s.endswith("USDT"): 
            continue
        try:
            chg = float(item.get("priceChangePercent","0"))
            qv = float(item.get("quoteVolume","0"))
            movers.append((s, chg, qv))
        except:
            continue
    movers.sort(key=lambda x: (x[1], x[2]), reverse=True)
    return movers[:limit]

# ---------- RISK MANAGEMENT ----------

def position_size(balance: float, risk_pct: float, entry: float, stop: float) -> Dict[str, float]:
    """
    Розрахунок розміру позиції: ризик = balance * risk_pct.
    К-сть токенів = ризик / |entry - stop|.
    """
    risk_amount = balance * (risk_pct / 100.0)
    per_unit_risk = abs(entry - stop)
    if per_unit_risk <= 0:
        raise ValueError("Entry і Stop повинні різнитись")
    qty = risk_amount / per_unit_risk
    return {"risk_amount": risk_amount, "qty": qty, "rr_one_tp": 2*per_unit_risk}  # для R:R 1:2