import time
import math
import requests
import numpy as np
from typing import Dict, List, Tuple, Optional
from cachetools import cached, TTLCache  # Додано імпорт для кешу

from app.config import (
    BINANCE_BASES,
    HTTP_TIMEOUT,
    KLINES_LIMIT,
    DEFAULT_INTERVAL,
    PIVOT_LEFT_RIGHT,
    MAX_LEVELS
)

# ---------- BINANCE HELPERS (CACHED) ----------
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

@cached(cache=TTLCache(maxsize=100, ttl=60))
def get_price(symbol: str) -> float:
    symbol = normalize_symbol(symbol)
    data = _binance_get("/api/v3/ticker/price", {"symbol": symbol})
    return float(data["price"])

def get_klines(symbol: str, interval: str = DEFAULT_INTERVAL, limit: int = KLINES_LIMIT) -> Dict[str, np.ndarray]:
    symbol = normalize_symbol(symbol)
    # Створюємо унікальний ключ для кешування на основі параметрів
    cache_key = f"klines_{symbol}_{interval}_{limit}"
    
    data = _binance_get("/api/v3/klines", {"symbol": symbol, "interval": interval, "limit": limit})
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

# ---------- NEW INDICATORS ----------
def calculate_rsi(close_prices: np.ndarray, period: int = 14) -> np.ndarray:
    """Розрахунок RSI."""
    delta = np.diff(close_prices)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = ema(gain, period)
    avg_loss = ema(loss, period)
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = np.concatenate(([np.nan], rsi))
    return rsi

def calculate_macd(close_prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Розрахунок MACD: лінія MACD, сигнальна лінія, гістограмма."""
    ema_fast = ema(close_prices, fast)
    ema_slow = ema(close_prices, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    macd_histogram = macd_line - signal_line
    return macd_line, signal_line, macd_histogram

def get_multi_timeframe_trend(symbol: str, current_interval: str) -> str:
    """
    Визначає загальний тренд на основі старшого таймфрейму.
    Правило: 1h -> 4h, 4h -> 1d, тощо.
    """
    higher_tf_map = {
        '1m': '5m', '5m': '15m', '15m': '30m',
        '30m': '1h', '1h': '4h', '4h': '1d', '1d': '1w'
    }
    higher_interval = higher_tf_map.get(current_interval, '4h')
    try:
        candles = get_klines(symbol, interval=higher_interval, limit=100)
        c = candles["c"]
        e50 = ema(c, 50)
        e200 = ema(c, 200)
        if e50[-1] > e200[-1] * 1.02:
            return "STRONG_UP"
        elif e50[-1] < e200[-1] * 0.98:
            return "STRONG_DOWN"
        else:
            return "NEUTRAL"
    except Exception as e:
        print(f"Помилка MTF аналізу для {symbol} на {higher_interval}: {e}")
        return "NEUTRAL"

def get_crypto_sentiment():
    """
    Отримує простий індекс настроїв (Fear & Greed Index) з Alternative.me.
    """
    try:
        url = "https://api.alternative.me/fng/"
        response = requests.get(url, timeout=10)
        data = response.json()
        if 'data' in data and len(data['data']) > 0:
            value = int(data['data'][0]['value'])
            classification = data['data'][0]['value_classification']
            return value, classification
        return None, "No data"
    except Exception as e:
        print(f"Помилка отримання індексу настроїв: {e}")
        return None, "Error"

# ---------- LEVELS (Support/Resistance) ----------
def _pivot_high(h: np.ndarray, i: int, left_right: int) -> bool:
    left = max(0, i - left_right)
    right = min(len(h), i + left_right + 1)
    return h[i] == np.max(h[left:right])

def _pivot_low(l: np.ndarray, i: int, left_right: int) -> bool:
    left = max(0, i - left_right)
    right = min(len(l), i + left_right + 1)
    return l[i] == np.min(l[left:right])

def find_levels(candles: Dict[str, np.ndarray], left_right: int = PIVOT_LEFT_RIGHT, max_levels: int = MAX_LEVELS) -> Dict[str, List[float]]:
    h, l, c = candles["h"], candles["l"], candles["c"]
    last_price = c[-1]
    _atr = atr(h, l, c, 14)[-1]
    tol = max(_atr * 0.5, last_price * 0.002)
    highs, lows = [], []
    for i in range(left_right, len(h) - left_right):
        if _pivot_high(h, i, left_right):
            highs.append(h[i])
        if _pivot_low(l, i, left_right):
            lows.append(l[i])
    def cluster(levels: List[float]) -> List[float]:
        if not levels:
            return []
        levels = sorted(levels)
        clusters: List[List[float]] = [[levels[0]]]
        for x in levels[1:]:
            if abs(x - np.mean(clusters[-1])) <= tol:
                clusters[-1].append(x)
            else:
                clusters.append([x])
        weighted = [(float(np.mean(g)), len(g)) for g in clusters]
        weighted.sort(key=lambda e: (-e[1], abs(e[0] - last_price)))
        return [w[0] for w in weighted[:max_levels]]
    resistances = cluster(highs)
    supports = cluster(lows)
    supports = sorted(set(supports))
    resistances = sorted(set(resistances))
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
    slope = (ema50[-1] - ema50[-10]) / (ema50[-10] + 1e-9) * 100.0
    state = "UP" if ema50[-1] > ema200[-1] else ("DOWN" if ema50[-1] < ema200[-1] else "FLAT")
    vol = np.std(c[-50:]) / (np.mean(c[-50:]) + 1e-9) * 100.0
    return f"Trend: <b>{state}</b> | EMA50-200 Δ: {((ema50[-1]-ema200[-1])/ema200[-1]*100):.2f}% | Slope(50): {slope:.2f}% | Volatility(50): {vol:.2f}%"

def generate_signal_text(symbol: str, interval: str = DEFAULT_INTERVAL) -> str:
    candles = get_klines(symbol, interval=interval, limit=KLINES_LIMIT)
    c, h, l, v = candles["c"], candles["h"], candles["l"], candles["v"]
    last = c[-1]
    e50 = ema(c, 50)
    e200 = ema(c, 200)
    _atr = atr(h, l, c, 14)[-1]
    levels = find_levels(candles)
    trend = "UP" if e50[-1] > e200[-1] else ("DOWN" if e50[-1] < e200[-1] else "FLAT")
    sup = levels["near_support"]
    res = levels["near_resistance"]

    # --- NEW: INDICATOR CALCULATIONS ---
    rsi = calculate_rsi(c, period=14)
    macd_line, signal_line, macd_histogram = calculate_macd(c)
    avg_volume = np.mean(v[-20:])
    last_volume = v[-1]
    volume_ok = last_volume > avg_volume
    higher_tf_trend = get_multi_timeframe_trend(symbol, interval)
    sentiment_value, sentiment_text = get_crypto_sentiment()

    # --- SIGNAL LOGIC WITH CONFLUENCE ---
    txt = [f"📊 <b>{symbol.upper()}</b> [{interval}] | Price: <b>{last:.4f}</b>"]
    txt.append(trend_strength_text(candles))
    txt.append(f"RSI(14): {rsi[-1]:.2f} | MACD Hist: {macd_histogram[-1]:.4f} | Vol: {'↑' if volume_ok else '↓'}")
    txt.append(f"HTF Trend: {higher_tf_trend}")
    if sentiment_value:
        txt.append(f"🎭 Fear & Greed: {sentiment_value} ({sentiment_text})")

    confluence_score = 0
    signal_direction = None
    reason = []
    entry, stop, tp = None, None, None

    # LONG Signal Logic (Bounce from Support)
    if sup and last > sup and (last - sup) <= max(_atr, last*0.004):
        signal_direction = "LONG"
        entry = last
        stop = sup - max(_atr*0.5, last*0.003)
        tp = res if res else last + 2.0 * _atr

        if 30 < rsi[-1] < 70:
            confluence_score += 1
            reason.append("RSI ok")
        if macd_histogram[-1] > 0 or macd_line[-1] > signal_line[-1]:
            confluence_score += 1
            reason.append("MACD bull")
        if volume_ok:
            confluence_score += 1
            reason.append("High vol")
        if higher_tf_trend == "STRONG_UP":
            confluence_score += 2
            reason.append("HTF UP")
        elif higher_tf_trend == "STRONG_DOWN":
            confluence_score -= 2
            reason.append("HTF DOWN")
        if sentiment_value is not None and sentiment_value < 30:
            confluence_score += 1
            reason.append("Extreme Fear")

    # SHORT Signal Logic (Bounce from Resistance)
    elif res and last < res and (res - last) <= max(_atr, last*0.004):
        signal_direction = "SHORT"
        entry = last
        stop = res + max(_atr*0.5, last*0.003)
        tp = sup if sup else last - 2.0 * _atr

        if 30 < rsi[-1] < 70:
            confluence_score += 1
            reason.append("RSI ok")
        if macd_histogram[-1] < 0 or macd_line[-1] < signal_line[-1]:
            confluence_score += 1
            reason.append("MACD bear")
        if volume_ok:
            confluence_score += 1
            reason.append("High vol")
        if higher_tf_trend == "STRONG_DOWN":
            confluence_score += 2
            reason.append("HTF DOWN")
        elif higher_tf_trend == "STRONG_UP":
            confluence_score -= 2
            reason.append("HTF UP")
        if sentiment_value is not None and sentiment_value > 70:
            confluence_score += 1
            reason.append("Extreme Greed")

    # --- FORM FINAL SIGNAL MESSAGE ---
    if signal_direction and confluence_score >= 3:
        txt.append(f"✅ <b>{signal_direction} CONFLUENCE ({confluence_score}/7)</b>")
        txt.append(f"Reason: {', '.join(reason)}")
        txt.append(f"Entry ~{entry:.4f}, SL {stop:.4f}, TP {tp:.4f}")
    elif signal_direction:
        txt.append(f"🟡 Weak {signal_direction} signal ({confluence_score}/7). Reason: {', '.join(reason)}")
    else:
        txt.append("ℹ️ No clear entry points found. Wait for a new candle or change the interval.")

    lv_s = ", ".join(f"{x:.4f}" for x in levels["supports"][:MAX_LEVELS])
    lv_r = ", ".join(f"{x:.4f}" for x in levels["resistances"][:MAX_LEVELS])
    txt.append(f"Levels → S: [{lv_s}] | R: [{lv_r}] | ATR(14): {levels['atr']:.4f}")
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
    risk_amount = balance * (risk_pct / 100.0)
    per_unit_risk = abs(entry - stop)
    if per_unit_risk <= 0:
        raise ValueError("Entry і Stop повинні різнитись")
    qty = risk_amount / per_unit_risk
    return {"risk_amount": risk_amount, "qty": qty, "rr_one_tp": 2*per_unit_risk}

# ---------- ATR SQUEEZE SCANNER ----------
# ---------- ATR SQUEEZE SCANNER ----------
def find_atr_squeeze(symbol: str, interval: str = '1h', limit: int = 100) -> float:
    """
    Повертає коефіцієнт стискання волатильності (current_atr / atr_ma_20).
    Чим менше значення — тим сильніше стиснення (наприклад, 0.70).
    """
    try:
        candles = get_klines(symbol, interval=interval, limit=limit)
        h, l, c = candles["h"], candles["l"], candles["c"]

        atr_values = atr(h, l, c, 14)
        # Маємо мати хоча б 20 значень ATR для порівняння із середнім
        if len(atr_values) < 20:
            return 1.0

        current_atr = float(atr_values[-1])
        atr_ma = float(np.mean(atr_values[-20:]))
        if atr_ma == 0:
            return 1.0

        return current_atr / atr_ma

    except Exception as e:
        print(f"Помилка розрахунку стискання для {symbol}: {e}")
        return 1.0
        
        # ---------- LIQUIDITY TRAP DETECTOR ----------
def detect_liquidity_trap(symbol: str, interval: str = "1h", lookback: int = 50):
    """
    Шукає пастки ліквідності (фальшиві пробої).
    Умови:
    - Пробій локального high/low
    - Закриття свічки назад у діапазон
    - Аномально великий об'єм
    """

    candles = get_klines(symbol, interval=interval, limit=lookback)
    h, l, c, v = candles["h"], candles["l"], candles["c"], candles["v"]

    local_high = max(h[:-1])
    local_low = min(l[:-1])
    last_close = c[-1]
    last_open = c[-2]
    last_high = h[-1]
    last_low = l[-1]
    last_vol = v[-1]

    avg_vol = np.mean(v[:-1]) if len(v) > 1 else last_vol

    # Умови пастки
    trap_signal = None
    if last_high > local_high and last_close < local_high and last_vol > 1.5 * avg_vol:
        trap_signal = f"🐻 <b>Short Trap</b> на {symbol} – фальшивий пробій вверх!"
    elif last_low < local_low and last_close > local_low and last_vol > 1.5 * avg_vol:
        trap_signal = f"🐂 <b>Long Trap</b> на {symbol} – фальшивий пробій вниз!"

    return trap_signal
    
    # ---------- IMPULSE + COMPRESSION detector ----------
import numpy as np
import pandas as pd

def _simple_atr(series_high, series_low, series_close, period=14):
    high = pd.Series(series_high)
    low = pd.Series(series_low)
    close = pd.Series(series_close)
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    return atr.values

def detect_impulse_compression(symbol: str,
                               interval: str = "1h",
                               lookback: int = 120,
                               impulse_atr_mult: float = 1.8,
                               compression_length: int = 6,
                               compression_atr_mult: float = 0.7):
    """
    Шукає структуру: сильний імпульс (1-2 свічки) → коротке стиснення (n свічок з низьким ATR).
    Повертає dict з info: found (bool), score (0..100), direction, details.
    Параметри можна підлаштувати:
      - impulse_atr_mult: скільки ATR має бути у імпульсі (1.5..3.0)
      - compression_length: скільки свічок має тривати стиснення
      - compression_atr_mult: компресія як частка середнього ATR (0.5..0.9)
    """

    # Отримуємо OHLCV — припускаємо наявність fetch_ohlcv/ get_klines
    df = fetch_ohlcv(symbol, timeframe=interval, limit=lookback)
    if df is None or df.empty:
        return {"error": "no data"}

    # Забезпечимо потрібні колонки
    # df -> columns: timestamp, open, high, low, close, volume
    highs = np.array(df['high'], dtype=float)
    lows = np.array(df['low'], dtype=float)
    closes = np.array(df['close'], dtype=float)
    vols = np.array(df['volume'], dtype=float)

    # ATR series
    atr_series = _simple_atr(highs, lows, closes, period=14)
    if len(atr_series) < 30:
        return {"error": "insufficient data"}

    # 1) знайдемо останній імпульс: дивимось на останні 3 свічки
    last_idx = len(closes) - 1
    # розглядаємо імпульс як max руху серед останніх 1-2 свічок
    candidates = []
    for i in range(1, 3):  # 1 або 2-свічковий імпульс
        idx = last_idx - i + 1
        if idx <= 0:
            continue
        move = abs(closes[idx] - closes[idx-1]) if idx-1 >= 0 else 0
        rel = move / max(1e-9, atr_series[idx])
        candidates.append((i, idx, move, rel))

    if not candidates:
        return {"found": False, "reason": "no candidates"}

    # вибираємо найбільший відносний рух
    cand = max(candidates, key=lambda x: x[3])
    impulse_bars, impulse_idx, impulse_move, impulse_rel = cand

    impulse_detected = impulse_rel >= impulse_atr_mult

    # 2) перевірка стиснення після імпульсу: беремо стиснення до останньої свічки перед імпульсом (або після?)
    # Логіка: після імпульсу дивимось backward — чи була пауза перед імпульсом?
    # Альтернативно — дивимось forward: якщо імпульс був 2 свічки тому, перевіряємо свічки між імпульсом та зараз.
    # Тут беремо стиснення *перед* імпульсом (тобто імпульс виходить із стискання) — класичний pattern.
    comp_end = impulse_idx - 1
    comp_start = max(0, comp_end - compression_length + 1)
    if comp_end <= comp_start:
        compression_atr = np.mean(atr_series[comp_start:comp_end+1]) if comp_end >= comp_start else np.nan
    else:
        compression_atr = np.mean(atr_series[comp_start:comp_end+1])

    # порівняємо з довгим середнім ATR (наприклад середні 50)
    long_atr_avg = np.mean(atr_series[-50:]) if len(atr_series) >= 50 else np.mean(atr_series)

    compression_detected = False
    if not np.isnan(compression_atr) and long_atr_avg > 0:
        compression_detected = (compression_atr <= compression_atr_mult * long_atr_avg)

    # 3) volume confirmation: об'єм на імпульсовій свічці
    vol_impulse = vols[impulse_idx] if impulse_idx < len(vols) else vols[-1]
    vol_avg = np.mean(vols[max(0, last_idx-50):last_idx]) if last_idx >= 1 else vol_impulse
    vol_confirm = (vol_impulse >= 1.2 * vol_avg)

    # 4) trend bias (EMA20 vs EMA50)
    series_close = pd.Series(closes)
    ema20 = series_close.ewm(span=20, adjust=False).mean().iloc[-1]
    ema50 = series_close.ewm(span=50, adjust=False).mean().iloc[-1]
    trend_bias = 1 if ema20 > ema50 else -1

    # determine direction of impulse
    direction = "UNKNOWN"
    # check sign of last impulse move (close - prev close)
    prev_close = closes[impulse_idx - 1] if impulse_idx - 1 >= 0 else closes[impulse_idx]
    sign = np.sign(closes[impulse_idx] - prev_close)
    if sign > 0:
        direction = "UP"
    elif sign < 0:
        direction = "DOWN"

    # 5) score aggregation (weights)
    score = 0.0
    if impulse_detected:
        # relative strength adds to score (scaled)
        score += min(30.0, 30.0 * (impulse_rel / (impulse_atr_mult if impulse_atr_mult>0 else 1)))
    if compression_detected:
        score += 30.0
    if vol_confirm:
        score += 20.0
    # trend bias alignment
    if (direction == "UP" and trend_bias == 1) or (direction == "DOWN" and trend_bias == -1):
        score += 20.0

    score = max(0.0, min(100.0, score))

    # verdict logic
    if impulse_detected and compression_detected and score >= 65:
        verdict = "CONTINUE_UP" if direction == "UP" else "CONTINUE_DOWN"
    elif impulse_detected and compression_detected:
        verdict = "WATCH"
    else:
        verdict = "NO_PATTERN"

    details = {
        "impulse_detected": bool(impulse_detected),
        "impulse_rel": float(impulse_rel),
        "impulse_bars": int(impulse_bars),
        "impulse_idx": int(impulse_idx),
        "compression_detected": bool(compression_detected),
        "compression_atr": float(compression_atr) if not np.isnan(compression_atr) else None,
        "long_atr_avg": float(long_atr_avg),
        "vol_impulse": float(vol_impulse),
        "vol_avg": float(vol_avg),
        "vol_confirm": bool(vol_confirm),
        "direction": direction,
        "trend_bias": int(trend_bias),
        "score": float(score),
        "verdict": verdict,
        "last_price": float(closes[-1]),
        "ema20": float(ema20),
        "ema50": float(ema50),
    }

    return details