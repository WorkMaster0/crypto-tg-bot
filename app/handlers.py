import requests
from telebot import types
import numpy as np
from app.bot import bot
from app.analytics import (
    get_price, get_klines, generate_signal_text, trend_strength_text,
    find_levels, top_movers, position_size, normalize_symbol, find_atr_squeeze  # <-- Додано find_atr_squeeze
)
from app.chart import plot_candles
from app.config import DEFAULT_INTERVAL, ALLOWED_INTERVALS

# просте зберігання налаштувань чату в ОЗП
_user_defaults = {}  # chat_id -> {"interval": "1h"}

def _default_interval(chat_id):
    return _user_defaults.get(chat_id, {}).get("interval", DEFAULT_INTERVAL)

def _parse_args(msg_text: str):
    parts = msg_text.split()
    symbol = None
    interval = None
    if len(parts) >= 2:
        symbol = normalize_symbol(parts[1])
    if len(parts) >= 3 and parts[2] in ALLOWED_INTERVALS:
        interval = parts[2]
    return symbol, interval

# ---------- /start ----------
@bot.message_handler(commands=['start'])
def start(message):
    bot.reply_to(message, (
        "🚀 <b>Crypto Analysis Bot</b> запущено!\n"
        "Використання прикладів:\n"
        "• <code>/price BTCUSDT</code>\n"
        "• <code>/analyze BTCUSDT 1h</code>\n"
        "• <code>/levels BTCUSDT 4h</code>\n"
        "• <code>/chart BTCUSDT 1h</code>\n"
        "• <code>/trend BTCUSDT</code>\n"
        "• <code>/heatmap</code>\n"
        "• <code>/risk 1000 1 65000 64000</code>  (баланс 1000$, ризик 1%, вхід 65000, стоп 64000)\n"
        "• <code>/setdefault 1h</code>\n"
        "Довідка: <code>/help</code>"
    ))

# ---------- /help ----------
@bot.message_handler(commands=['help'])
def help_cmd(message):
    bot.reply_to(message, (
        "<b>Команди:</b>\n"
        "<code>/price SYMBOL</code> — поточна ціна\n"
        "<code>/analyze SYMBOL [interval]</code> — сигнал + рівні S/R\n"
        "<code>/levels SYMBOL [interval]</code> — список рівнів підтримки/опору\n"
        "<code>/chart SYMBOL [interval]</code> — графік з EMA та рівнями\n"
        "<code>/trend SYMBOL [interval]</code> — сила тренду\n"
        "<code>/heatmap [N]</code> — топ рухів USDT-пар (за 24h)\n"
        "<code>/risk balance risk% entry stop</code> — розмір позиції\n"
        "<code>/setdefault interval</code> — інтервал за замовчуванням для цього чату\n"
        f"Доступні інтервали: {', '.join(sorted(ALLOWED_INTERVALS))}"
    ))

# ---------- /price ----------
@bot.message_handler(commands=['price'])
def price_handler(message):
    symbol, interval = _parse_args(message.text)
    if not symbol:
        return bot.reply_to(message, "⚠️ Приклад: <code>/price BTCUSDT</code>")
    try:
        price = get_price(symbol)
        bot.reply_to(message, f"💰 <b>{symbol}</b> = <b>{price:.6f}</b> USDT")
    except Exception as e:
        bot.reply_to(message, f"❌ Помилка: {e}")

# ---------- /levels ----------
@bot.message_handler(commands=['levels'])
def levels_handler(message):
    symbol, interval = _parse_args(message.text)
    if not symbol:
        return bot.reply_to(message, "⚠️ Приклад: <code>/levels BTCUSDT 1h</code>")
    interval = interval or _default_interval(message.chat.id)
    try:
        candles = get_klines(symbol, interval=interval)
        lv = find_levels(candles)
        s = ", ".join(f"{x:.4f}" for x in lv["supports"])
        r = ", ".join(f"{x:.4f}" for x in lv["resistances"])
        bot.reply_to(message, (
            f"🔎 <b>{symbol}</b> [{interval}] Levels\n"
            f"Supports: {s or '—'}\n"
            f"Resistances: {r or '—'}\n"
            f"Nearest S: <b>{lv['near_support']:.4f}</b> | "
            f"Nearest R: <b>{lv['near_resistance']:.4f}</b>\n"
            f"ATR(14): {lv['atr']:.4f} | tol: {lv['tolerance']:.4f}"
        ))
    except Exception as e:
        bot.reply_to(message, f"❌ Помилка: {e}")

# ---------- /analyze ----------
@bot.message_handler(commands=['analyze'])
def analyze_handler(message):
    symbol, interval = _parse_args(message.text)
    if not symbol:
        return bot.reply_to(message, "⚠️ Приклад: <code>/analyze BTCUSDT 1h</code>")
    interval = interval or _default_interval(message.chat.id)
    try:
        text = generate_signal_text(symbol, interval=interval)
        bot.reply_to(message, text)
    except Exception as e:
        bot.reply_to(message, f"❌ Помилка: {e}")

# ---------- /trend ----------
@bot.message_handler(commands=['trend'])
def trend_handler(message):
    symbol, interval = _parse_args(message.text)
    if not symbol:
        return bot.reply_to(message, "⚠️ Приклад: <code>/trend BTCUSDT 4h</code>")
    interval = interval or _default_interval(message.chat.id)
    try:
        candles = get_klines(symbol, interval=interval)
        txt = trend_strength_text(candles)
        bot.reply_to(message, f"📈 <b>{symbol}</b> [{interval}]  {txt}")
    except Exception as e:
        bot.reply_to(message, f"❌ Помилка: {e}")

# ---------- /chart ----------
@bot.message_handler(commands=['chart'])
def chart_handler(message):
    symbol, interval = _parse_args(message.text)
    if not symbol:
        return bot.reply_to(message, "⚠️ Приклад: <code>/chart BTCUSDT 1h</code>")
    interval = interval or _default_interval(message.chat.id)
    try:
        img = plot_candles(symbol, interval=interval, limit=200, with_levels=True)
        bot.send_photo(message.chat.id, img)
    except Exception as e:
        bot.reply_to(message, f"❌ Помилка: {e}")

# ---------- /heatmap ----------
@bot.message_handler(commands=['heatmap'])
def heatmap_handler(message):
    parts = message.text.split()
    try:
        n = int(parts[1]) if len(parts) > 1 else 10
    except:
        n = 10
    try:
        movers = top_movers(limit=min(max(n, 1), 20))
        lines = ["🔥 <b>Top movers (24h, USDT pairs)</b>"]
        for i, (s, chg, qv) in enumerate(movers, 1):
            lines.append(f"{i}. <b>{s}</b>  {chg:+.2f}%  | vol≈{qv/1e6:.2f}M")
        bot.reply_to(message, "\n".join(lines))
    except Exception as e:
        bot.reply_to(message, f"❌ Помилка: {e}")

# ---------- /risk ----------
@bot.message_handler(commands=['risk'])
def risk_handler(message):
    parts = message.text.split()
    if len(parts) < 5:
        return bot.reply_to(message, "⚠️ Приклад: <code>/risk 1000 1 65000 64000</code> (balance risk% entry stop)")
    try:
        balance = float(parts[1])
        risk_pct = float(parts[2])
        entry = float(parts[3])
        stop = float(parts[4])
        res = position_size(balance, risk_pct, entry, stop)
        bot.reply_to(message, (
            f"🧮 Risk: {risk_pct:.2f}% від ${balance:.2f} → ${res['risk_amount']:.2f}\n"
            f"📦 Position size ≈ <b>{res['qty']:.6f}</b> токенів\n"
            f"🎯 1R ≈ {abs(entry - stop):.4f} | 2R TP ≈ {entry + (res['rr_one_tp'] if entry>stop else -res['rr_one_tp']):.4f}"
        ))
    except Exception as e:
        bot.reply_to(message, f"❌ Помилка: {e}")

# ---------- /setdefault ----------
@bot.message_handler(commands=['setdefault'])
def setdefault_handler(message):
    parts = message.text.split()
    if len(parts) < 2 or parts[1] not in ALLOWED_INTERVALS:
        return bot.reply_to(message, "⚠️ Приклад: <code>/setdefault 1h</code>")
    _user_defaults.setdefault(message.chat.id, {})["interval"] = parts[1]
    bot.reply_to(message, f"✅ Інтервал за замовчуванням для цього чату: <b>{parts[1]}</b>")

# ---------- /squeeze ----------
@bot.message_handler(commands=['squeeze'])
def squeeze_handler(message):
    try:
        import requests
        import numpy as np

        url = "https://api.binance.com/api/v3/ticker/24hr"
        data = requests.get(url).json()

        # ✅ фільтруємо лише USDT-пари з нормальним об’ємом
        symbols = [
            d for d in data
            if d["symbol"].endswith("USDT") and float(d["quoteVolume"]) > 5_000_000
        ]

        # ✅ сортуємо по % зміни (як у smart_auto)
        symbols = sorted(
            symbols,
            key=lambda x: abs(float(x["priceChangePercent"])),
            reverse=True
        )

        # беремо топ-30
        top_symbols = [s["symbol"] for s in symbols[:30]]

        signals = []
        for symbol in top_symbols:
            try:
                df = get_klines(symbol, interval="1h", limit=200)
                if not df or len(df.get("c", [])) < 50:
                    continue

                closes = np.array(df["c"], dtype=float)
                volumes = np.array(df["v"], dtype=float)

                # ---- Bollinger Bands ----
                period = 20
                if len(closes) < period:
                    continue

                ma = np.convolve(closes, np.ones(period)/period, mode='valid')
                std = np.array([closes[i-period+1:i+1].std() for i in range(period-1, len(closes))])

                upper = ma + 2 * std
                lower = ma - 2 * std
                width = (upper - lower) / ma  # ширина смуги

                last_price = closes[-1]
                last_ma = ma[-1]
                last_width = width[-1]
                prev_width = width[-5:].mean()

                # ---- Умови для squeeze ----
                squeeze_detected = last_width < 0.02 and last_width < prev_width * 0.7
                breakout_up = last_price > upper[-1]
                breakout_down = last_price < lower[-1]

                signal = None
                if squeeze_detected:
                    if breakout_up and volumes[-1] > np.mean(volumes[-20:]) * 1.5:
                        diff = ((last_price - upper[-1]) / upper[-1]) * 100
                        signal = f"🚀 LONG squeeze breakout вище {upper[-1]:.4f} ({diff:+.2f}%)"
                    elif breakout_down and volumes[-1] > np.mean(volumes[-20:]) * 1.5:
                        diff = ((last_price - lower[-1]) / lower[-1]) * 100
                        signal = f"⚡ SHORT squeeze breakout нижче {lower[-1]:.4f} ({diff:+.2f}%)"

                if signal:
                    signals.append(f"<b>{symbol}</b>\n{signal}")

            except Exception:
                continue

        if not signals:
            bot.send_message(message.chat.id, "ℹ️ Жодних squeeze-сигналів не знайдено.")
        else:
            text = "<b>Squeeze Scanner Signals</b>\n\n" + "\n\n".join(signals)
            bot.send_message(message.chat.id, text, parse_mode="HTML")

    except Exception as e:
        bot.send_message(message.chat.id, f"❌ Помилка сканера: {e}")

        # ---------- /trap ----------
@bot.message_handler(commands=['trap'])
def trap_scanner(message):
    """Сканує топ пари на пастки ліквідності"""
    top_pairs = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 
                 'XRPUSDT', 'ADAUSDT', 'AVAXUSDT', 'DOTUSDT', 
                 'DOGEUSDT', 'LINKUSDT']

    traps = []
    for pair in top_pairs:
        try:
            signal = detect_liquidity_trap(pair, interval="1h", lookback=50)
            if signal:
                traps.append(signal)
        except Exception as e:
            print(f"Помилка для {pair}: {e}")
            continue

    if traps:
        bot.send_message(message.chat.id, 
                         "🔍 <b>Виявлені пастки ліквідності:</b>\n\n" + "\n".join(traps),
                         parse_mode="HTML")
    else:
        bot.send_message(message.chat.id, 
                         "✅ Пасток ліквідності не знайдено на 1h таймфреймі.")
                        
# ---------- /smart_sr ----------
def find_support_resistance(prices, window=20, delta=0.005):
    """
    Автоматично знаходить локальні S/R рівні
    prices: масив цін (закриття)
    window: скільки свічок дивимося для локального максимуму/мінімуму
    delta: мінімальна дистанція між рівнями (5%)
    """
    sr_levels = []
    for i in range(window, len(prices)-window):
        local_max = max(prices[i-window:i+window+1])
        local_min = min(prices[i-window:i+window+1])
        if prices[i] == local_max:
            if all(abs(prices[i]-lvl)/lvl > delta for lvl in sr_levels):
                sr_levels.append(prices[i])
        elif prices[i] == local_min:
            if all(abs(prices[i]-lvl)/lvl > delta for lvl in sr_levels):
                sr_levels.append(prices[i])
    return sorted(sr_levels)

@bot.message_handler(commands=['smart_sr'])
def smart_sr_handler(message):
    parts = message.text.split()
    if len(parts) < 2:
        return bot.reply_to(message, "⚠️ Використання: /smart_sr BTCUSDT")
    symbol = parts[1].upper()
    
    try:
        # Отримуємо дані
        df = get_klines(symbol, interval="1h", limit=200)
        if not df or len(df.get('c', [])) == 0:
            return bot.send_message(message.chat.id, f"❌ Дані для {symbol} недоступні")

        closes = np.array(df['c'], dtype=float)
        highs = np.array(df['h'], dtype=float)
        lows = np.array(df['l'], dtype=float)
        volumes = np.array(df['v'], dtype=float)
        
        # Знаходимо S/R рівні
        sr_levels = find_support_resistance(closes, window=20, delta=0.005)
        last_price = closes[-1]

        # Перевірка breakout
        signal = "ℹ️ Патерн не знайдено"
        for lvl in sr_levels:
            if last_price > lvl * 1.01:
                signal = f"🚀 LONG breakout: ціна пробила опір {lvl:.4f}"
            elif last_price < lvl * 0.99:
                signal = f"⚡ SHORT breakout: ціна пробила підтримку {lvl:.4f}"

        # Перевірка pre-top / pump
        if len(closes) >= 4:
            impulse = (closes[-1] - closes[-4]) / closes[-4]
        else:
            impulse = 0
        vol_spike = volumes[-1] > 1.5 * np.mean(volumes[-20:]) if len(volumes) >= 20 else False
        nearest_resistance = max([lvl for lvl in sr_levels if lvl < last_price], default=None)
        if impulse > 0.08 and vol_spike and nearest_resistance is not None:
            signal += f"\n⚠️ Pre-top detected: можливий short біля {nearest_resistance:.4f}"

        # Генеруємо графік
        img = plot_candles(symbol, interval="1h", limit=100)
        bot.send_photo(message.chat.id, img, caption=f"<b>{symbol} — Smart S/R Analysis</b>\n\n{signal}", parse_mode="HTML")
        
    except Exception as e:
        bot.send_message(message.chat.id, f"❌ Error: {e}")

# ---------- /smart_auto ----------
@bot.message_handler(commands=['smart_auto'])
def smart_auto_handler(message):
    try:
        import requests
        import numpy as np

        url = "https://api.binance.com/api/v3/ticker/24hr"
        data = requests.get(url).json()

        # ✅ фільтруємо тільки USDT-пари з нормальним об'ємом (щоб уникнути сміттєвих монет)
        symbols = [
            d for d in data
            if d["symbol"].endswith("USDT") and float(d["quoteVolume"]) > 5_000_000
        ]

        # ✅ сортуємо за % зміни ціни за 24 години (топ рухомі монети)
        symbols = sorted(
            symbols,
            key=lambda x: abs(float(x["priceChangePercent"])),
            reverse=True
        )

        # беремо топ-30 найактивніших
        top_symbols = [s["symbol"] for s in symbols[:30]]

        signals = []
        for symbol in top_symbols:
            try:
                df = get_klines(symbol, interval="1h", limit=200)
                if not df or len(df.get("c", [])) < 50:
                    continue

                closes = np.array(df["c"], dtype=float)
                volumes = np.array(df["v"], dtype=float)

                sr_levels = find_support_resistance(closes, window=20, delta=0.005)
                last_price = closes[-1]

                signal = None
                for lvl in sr_levels:
                    diff = last_price - lvl
                    diff_pct = (diff / lvl) * 100

                    if last_price > lvl * 1.01:
                        signal = (
                            f"🚀 LONG breakout: ціна пробила опір {lvl:.4f}\n"
                            f"📊 Ринкова: {last_price:.4f} | Відрив: {diff:+.4f} ({diff_pct:+.2f}%)"
                        )
                        break
                    elif last_price < lvl * 0.99:
                        signal = (
                            f"⚡ SHORT breakout: ціна пробила підтримку {lvl:.4f}\n"
                            f"📊 Ринкова: {last_price:.4f} | Відрив: {diff:+.4f} ({diff_pct:+.2f}%)"
                        )
                        break

                # Перевірка pre-top / pump
                impulse = (closes[-1] - closes[-4]) / closes[-4] if len(closes) >= 4 else 0
                vol_spike = volumes[-1] > 1.5 * np.mean(volumes[-20:]) if len(volumes) >= 20 else False
                nearest_res = max([lvl for lvl in sr_levels if lvl < last_price], default=None)
                if impulse > 0.08 and vol_spike and nearest_res is not None:
                    diff = last_price - nearest_res
                    diff_pct = (diff / nearest_res) * 100
                    signal = (
                        f"⚠️ Pre-top detected: можливий short біля {nearest_res:.4f}\n"
                        f"📊 Ринкова: {last_price:.4f} | Відрив: {diff:+.4f} ({diff_pct:+.2f}%)"
                    )

                if signal:
                    signals.append(f"<b>{symbol}</b>\n{signal}")

            except Exception:
                continue

        if not signals:
            bot.send_message(message.chat.id, "ℹ️ Жодних сигналів не знайдено.")
        else:
            text = "<b>Smart Auto S/R Signals</b>\n\n" + "\n\n".join(signals)
            bot.send_message(message.chat.id, text, parse_mode="HTML")

    except Exception as e:
        bot.send_message(message.chat.id, f"❌ Error: {e}")
        
# ---------- /patern ----------
@bot.message_handler(commands=['patern', 'pattern'])
def pattern_handler(message):
    """
    Автоматичний пошук торгових патернів
    Використання: /patern [SYMBOL] [INTERVAL]
    """
    try:
        parts = message.text.split()
        symbol = None
        interval = None
        
        if len(parts) >= 2:
            symbol = parts[1].upper()
            if not symbol.endswith('USDT'):
                symbol += 'USDT'
        
        if len(parts) >= 3 and parts[2] in ALLOWED_INTERVALS:
            interval = parts[2]
        else:
            interval = _default_interval(message.chat.id)
        
        if not symbol:
            # Сканування топ монет якщо символ не вказано
            return scan_top_patterns(message)
        
        # Отримуємо дані
        candles = get_klines(symbol, interval=interval, limit=100)
        if not candles or len(candles['c']) < 20:
            bot.reply_to(message, f"❌ Недостатньо даних для {symbol} [{interval}]")
            return
        
        # Конвертуємо дані
        opens = np.array(candles['o'], dtype=float)
        highs = np.array(candles['h'], dtype=float)
        lows = np.array(candles['l'], dtype=float)
        closes = np.array(candles['c'], dtype=float)
        volumes = np.array(candles['v'], dtype=float)
        
        patterns = []
        
        # 1. Перевірка на пробій рівнів
        sr_levels = find_levels(candles)
        current_price = closes[-1]
        
        # Перевірка пробою опору
        for resistance in sr_levels['resistances']:
            if current_price > resistance * 1.01 and current_price < resistance * 1.03:
                patterns.append(("RESISTANCE_BREAKOUT", "LONG", f"Пробиття опору {resistance:.4f}"))
                break
        
        # Перевірка пробою підтримки
        for support in sr_levels['supports']:
            if current_price < support * 0.99 and current_price > support * 0.97:
                patterns.append(("SUPPORT_BREAKOUT", "SHORT", f"Пробиття підтримки {support:.4f}"))
                break
        
        # 2. Перевірка на класичні свічкові патерни
        # Бульish Engulfing
        if len(closes) >= 3:
            prev_open = opens[-2]
            prev_close = closes[-2]
            current_open = opens[-1]
            current_close = closes[-1]
            
            # Бульish Engulfing
            if prev_close < prev_open and current_close > current_open and current_close > prev_open and current_open < prev_close:
                patterns.append(("BULLISH_ENGULFING", "LONG", "Бульish Engulfing патерн"))
            
            # Беарish Engulfing
            if prev_close > prev_open and current_close < current_open and current_close < prev_open and current_open > prev_close:
                patterns.append(("BEARISH_ENGULFING", "SHORT", "Беарish Engulfing патерн"))
            
            # Hammer
            body_size = abs(current_close - current_open)
            lower_wick = min(current_open, current_close) - lows[-1]
            upper_wick = highs[-1] - max(current_open, current_close)
            
            if lower_wick > body_size * 2 and upper_wick < body_size * 0.5 and current_close > current_open:
                patterns.append(("HAMMER", "LONG", "Hammer патерн"))
            
            # Shooting Star
            if upper_wick > body_size * 2 and lower_wick < body_size * 0.5 and current_close < current_open:
                patterns.append(("SHOOTING_STAR", "SHORT", "Shooting Star патерн"))
        
        # 3. Перевірка на трійне дно/вершину
        if len(closes) >= 15:
            # Проста перевірка на формування трійної вершини
            last_15_highs = highs[-15:]
            last_15_lows = lows[-15:]
            
            # Пошук локальних максимумів
            peaks = []
            for i in range(5, len(last_15_highs)-5):
                if (last_15_highs[i] > last_15_highs[i-1] and 
                    last_15_highs[i] > last_15_highs[i+1] and
                    last_15_highs[i] > np.mean(last_15_highs)):
                    peaks.append((i, last_15_highs[i]))
            
            # Пошук локальних мінімумів
            troughs = []
            for i in range(5, len(last_15_lows)-5):
                if (last_15_lows[i] < last_15_lows[i-1] and 
                    last_15_lows[i] < last_15_lows[i+1] and
                    last_15_lows[i] < np.mean(last_15_lows)):
                    troughs.append((i, last_15_lows[i]))
            
            # Перевірка на трійну вершину
            if len(peaks) >= 3:
                peaks.sort(key=lambda x: x[1], reverse=True)
                if abs(peaks[0][1] - peaks[1][1]) / peaks[0][1] < 0.02 and abs(peaks[0][1] - peaks[2][1]) / peaks[0][1] < 0.02:
                    patterns.append(("TRIPLE_TOP", "SHORT", "Трійна вершина"))
            
            # Перевірка на трійне дно
            if len(troughs) >= 3:
                troughs.sort(key=lambda x: x[1])
                if abs(troughs[0][1] - troughs[1][1]) / troughs[0][1] < 0.02 and abs(troughs[0][1] - troughs[2][1]) / troughs[0][1] < 0.02:
                    patterns.append(("TRIPLE_BOTTOM", "LONG", "Трійне дно"))
        
        # 4. Перевірка на прапори
        if len(closes) > 20:
            # Аналіз тренду
            price_change = (closes[-1] - closes[-20]) / closes[-20]
            
            if abs(price_change) > 0.05:  # Мінімум 5% рух
                # Аналіз консолідації
                last_5_range = max(highs[-5:]) - min(lows[-5:])
                prev_5_range = max(highs[-10:-5]) - min(lows[-10:-5])
                
                if last_5_range < prev_5_range * 0.6:  # Консолідація
                    if price_change > 0:
                        patterns.append(("BULL_FLAG", "LONG", "Бичачий прапор"))
                    else:
                        patterns.append(("BEAR_FLAG", "SHORT", "Ведмежий прапор"))
        
        if not patterns:
            bot.reply_to(message, f"🔍 Для {symbol} [{interval}] торгових патернів не знайдено")
            return
        
        # Формуємо відповідь
        response = [f"🎯 <b>Знайдені патерни для {symbol} [{interval}]:</b>\n"]
        
        for pattern_name, signal_type, description in patterns:
            emoji = "🟢" if signal_type == "LONG" else "🔴"
            response.append(f"{emoji} <b>{pattern_name}</b> → {signal_type}")
            response.append(f"   📝 {description}")
        
        response.append(f"\n📊 <i>Загалом знайдено {len(patterns)} патерн(ів)</i>")
        
        # Відправляємо графік
        try:
            img = plot_candles(symbol, interval=interval, limit=100)
            bot.send_photo(message.chat.id, img, caption="\n".join(response), parse_mode="HTML")
        except:
            bot.reply_to(message, "\n".join(response), parse_mode="HTML")
            
    except Exception as e:
        bot.reply_to(message, f"❌ Помилка при пошуку патернів: {str(e)}")

def scan_top_patterns(message):
    """
    Сканує топ монети на наявність патернів
    """
    try:
        url = "https://api.binance.com/api/v3/ticker/24hr"
        response = requests.get(url)
        data = response.json()
        
        # Фільтруємо USDT пари з високим обсягом
        usdt_pairs = [d for d in data if d['symbol'].endswith('USDT') and float(d['quoteVolume']) > 10000000]
        top_pairs = sorted(usdt_pairs, key=lambda x: float(x['quoteVolume']), reverse=True)[:15]
        
        found_patterns = []
        
        for pair in top_pairs:
            symbol = pair['symbol']
            try:
                candles = get_klines(symbol, interval="1h", limit=50)
                if not candles or len(candles['c']) < 20:
                    continue
                
                closes = np.array(candles['c'], dtype=float)
                opens = np.array(candles['o'], dtype=float)
                highs = np.array(candles['h'], dtype=float)
                lows = np.array(candles['l'], dtype=float)
                
                current_price = closes[-1]
                prev_close = closes[-2] if len(closes) >= 2 else current_price
                
                # Проста перевірка на пробій
                price_change = (current_price - prev_close) / prev_close
                
                if abs(price_change) > 0.03:  # 3% зміна
                    direction = "LONG" if price_change > 0 else "SHORT"
                    found_patterns.append((symbol, "BREAKOUT", direction, f"{abs(price_change)*100:.1f}%"))
                
            except:
                continue
        
        if not found_patterns:
            bot.reply_to(message, "🔍 Торгових патернів не знайдено у топ монетах")
            return
        
        # Формуємо відповідь
        response = ["🔍 <b>Топ монети з торговими патернами (1h):</b>\n"]
        
        for symbol, pattern, direction, change in found_patterns[:10]:
            emoji = "🟢" if direction == "LONG" else "🔴"
            response.append(f"{emoji} {symbol}: {pattern} {direction} ({change})")
        
        bot.reply_to(message, "\n".join(response), parse_mode="HTML")
        
    except Exception as e:
        bot.reply_to(message, f"❌ Помилка при скануванні топ монет: {str(e)}")
        
        # ---------- /analyze_auto ----------
@bot.message_handler(commands=['analyze_auto'])
def analyze_auto_handler(message):
    """
    Автоматичне сканування топ токенів на всіх таймфреймах
    Знаходить токени з 6-7 сигналами
    """
    try:
        # Відправляємо повідомлення про початок сканування
        processing_msg = bot.send_message(message.chat.id, "🔍 Сканую топ токени... Це може зайняти кілька хвилин")
        
        # Отримуємо топ токени за обсягом
        url = "https://api.binance.com/api/v3/ticker/24hr"
        response = requests.get(url)
        data = response.json()
        
        # Фільтруємо USDT пари з високим обсягом
        usdt_pairs = [d for d in data if d['symbol'].endswith('USDT') and float(d['quoteVolume']) > 50000000]
        top_symbols = [pair['symbol'] for pair in sorted(usdt_pairs, 
                                                       key=lambda x: float(x['quoteVolume']), 
                                                       reverse=True)[:30]]
        
        results = []
        
        # Скануємо кожен токен на всіх таймфреймах
        for symbol in top_symbols:
            try:
                symbol_signals = []
                
                for interval in ALLOWED_INTERVALS:
                    try:
                        # Отримуємо сигнали для кожного таймфрейму
                        candles = get_klines(symbol, interval=interval, limit=100)
                        if not candles or len(candles['c']) < 20:
                            continue
                        
                        # Генеруємо текст сигналу
                        signal_text = generate_signal_text(symbol, interval=interval)
                        
                        # Перевіряємо чи є сильний сигнал
                        if any(keyword in signal_text for keyword in ['🟢 STRONG LONG', '🔴 STRONG SHORT', 'сильний', 'потенційний']):
                            signal_type = "LONG" if "🟢" in signal_text else "SHORT"
                            symbol_signals.append((interval, signal_type, signal_text))
                            
                    except Exception as e:
                        continue
                
                # Якщо знайшли 6+ сигналів для цього токена
                if len(symbol_signals) >= 6:
                    # Групуємо сигнали по типу
                    long_signals = sum(1 for _, signal_type, _ in symbol_signals if signal_type == "LONG")
                    short_signals = sum(1 for _, signal_type, _ in symbol_signals if signal_type == "SHORT")
                    
                    results.append({
                        'symbol': symbol,
                        'total_signals': len(symbol_signals),
                        'long_signals': long_signals,
                        'short_signals': short_signals,
                        'signals': symbol_signals
                    })
                    
            except Exception as e:
                continue
        
        # Видаляємо повідомлення про обробку
        try:
            bot.delete_message(message.chat.id, processing_msg.message_id)
        except:
            pass
        
        if not results:
            bot.reply_to(message, "🔍 Не знайдено токенів з 6+ сигналами")
            return
        
        # Сортуємо за кількістю сигналів
        results.sort(key=lambda x: x['total_signals'], reverse=True)
        
        # Формуємо відповідь
        response = ["🎯 <b>Токени з 6+ сигналами:</b>\n"]
        
        for result in results[:10]:  # Показуємо топ-10
            dominant_signal = "🟢 LONG" if result['long_signals'] > result['short_signals'] else "🔴 SHORT"
            response.append(
                f"\n📊 <b>{result['symbol']}</b> - {result['total_signals']} сигналів "
                f"({result['long_signals']}🟢 {result['short_signals']}🔴) - {dominant_signal}"
            )
            
            # Додаємо інформацію про таймфрейми
            for interval, signal_type, signal_text in result['signals'][:5]:  # Перші 5 сигналів
                emoji = "🟢" if signal_type == "LONG" else "🔴"
                response.append(f"   {emoji} {interval}: {signal_type}")
        
        response.append(f"\n📈 <i>Знайдено {len(results)} токен(ів) з 6+ сигналами</i>")
        
        # Додаємо кнопки для швидкого аналізу топ токенів
        markup = types.InlineKeyboardMarkup()
        for result in results[:3]:
            markup.add(types.InlineKeyboardButton(
                f"📊 {result['symbol']}", 
                callback_data=f"analyze_{result['symbol']}"
            ))
        
        bot.send_message(message.chat.id, "\n".join(response), 
                        parse_mode="HTML", reply_markup=markup)
        
    except Exception as e:
        bot.reply_to(message, f"❌ Помилка при скануванні: {str(e)}")

# ---------- /analyze_multi ----------
@bot.message_handler(commands=['analyze_multi'])
def analyze_multi_handler(message):
    """
    Швидке сканування топ-10 токенів на 1h таймфреймі
    """
    try:
        processing_msg = bot.send_message(message.chat.id, "🔍 Швидке сканування топ-10 токенів...")
        
        # Отримуємо топ токени
        url = "https://api.binance.com/api/v3/ticker/24hr"
        response = requests.get(url)
        data = response.json()
        
        usdt_pairs = [d for d in data if d['symbol'].endswith('USDT') and float(d['quoteVolume']) > 50000000]
        top_symbols = [pair['symbol'] for pair in sorted(usdt_pairs, 
                                                       key=lambda x: float(x['quoteVolume']), 
                                                       reverse=True)[:10]]
        
        strong_signals = []
        
        for symbol in top_symbols:
            try:
                signal_text = generate_signal_text(symbol, interval="1h")
                
                # Перевіряємо на сильні сигнали
                if any(keyword in signal_text for keyword in ['🟢 STRONG LONG', '🔴 STRONG SHORT']):
                    signal_type = "LONG" if "🟢" in signal_text else "SHORT"
                    strong_signals.append((symbol, signal_type, signal_text))
                    
            except Exception:
                continue
        
        try:
            bot.delete_message(message.chat.id, processing_msg.message_id)
        except:
            pass
        
        if not strong_signals:
            bot.reply_to(message, "🔍 Не знайдено сильних сигналів у топ-10 токенів (1h)")
            return
        
        response = ["⚡ <b>Сильні сигнали у топ-10 токенів (1h):</b>\n"]
        
        for symbol, signal_type, signal_text in strong_signals:
            emoji = "🟢" if signal_type == "LONG" else "🔴"
            # Беремо тільки перші 2 рядки з сигналу
            lines = signal_text.split('\n')
            short_signal = ' | '.join(lines[:2])
            response.append(f"\n{emoji} <b>{symbol}</b>: {short_signal}")
        
        bot.reply_to(message, "\n".join(response), parse_mode="HTML")
        
    except Exception as e:
        bot.reply_to(message, f"❌ Помилка: {str(e)}")

# ---------- Callback для аналізу конкретного токена ----------
@bot.callback_query_handler(func=lambda call: call.data.startswith('analyze_'))
def analyze_callback_handler(call):
    """
    Обробка callback для детального аналізу токена
    """
    try:
        symbol = call.data.replace('analyze_', '')
        bot.send_message(call.message.chat.id, f"🔍 Аналізую {symbol}...")
        
        # Аналізуємо на всіх таймфреймах
        response = [f"📊 <b>Детальний аналіз {symbol}:</b>\n"]
        
        for interval in ALLOWED_INTERVALS:
            try:
                signal_text = generate_signal_text(symbol, interval=interval)
                
                # Перевіряємо чи є сигнал
                if any(keyword in signal_text for keyword in ['🟢', '🔴', 'LONG', 'SHORT']):
                    # Спрощуємо вивід
                    lines = signal_text.split('\n')
                    short_info = f"{lines[0]} | {lines[1]}" if len(lines) > 1 else lines[0]
                    response.append(f"\n{interval}: {short_info}")
                    
            except Exception:
                continue
        
        # Додаємо графік
        try:
            img = plot_candles(symbol, interval="1h", limit=100)
            bot.send_photo(call.message.chat.id, img, caption="\n".join(response), parse_mode="HTML")
        except:
            bot.send_message(call.message.chat.id, "\n".join(response), parse_mode="HTML")
            
    except Exception as e:
        bot.send_message(call.message.chat.id, f"❌ Помилка: {str(e)}")

# ---------- /analyze_top ----------
@bot.message_handler(commands=['analyze_top'])
def analyze_top_handler(message):
    """
    Сканування топ-5 токенів з найбільшою кількістю сигналів
    """
    try:
        parts = message.text.split()
        min_signals = 5  # Мінімальна кількість сигналів
        
        if len(parts) >= 2:
            try:
                min_signals = int(parts[1])
            except:
                pass
        
        processing_msg = bot.send_message(message.chat.id, f"🔍 Шукаю токени з {min_signals}+ сигналами...")
        
        # Отримуємо топ токени
        url = "https://api.binance.com/api/v3/ticker/24hr"
        response = requests.get(url)
        data = response.json()
        
        usdt_pairs = [d for d in data if d['symbol'].endswith('USDT') and float(d['quoteVolume']) > 30000000]
        top_symbols = [pair['symbol'] for pair in sorted(usdt_pairs, 
                                                       key=lambda x: float(x['quoteVolume']), 
                                                       reverse=True)[:20]]
        
        token_stats = []
        
        for symbol in top_symbols:
            try:
                signal_count = 0
                signal_details = []
                
                for interval in ALLOWED_INTERVALS:
                    try:
                        signal_text = generate_signal_text(symbol, interval=interval)
                        
                        # Рахуємо сигнали
                        if "🟢" in signal_text or "🔴" in signal_text:
                            signal_count += 1
                            signal_type = "LONG" if "🟢" in signal_text else "SHORT"
                            signal_details.append((interval, signal_type))
                            
                    except Exception:
                        continue
                
                if signal_count >= min_signals:
                    token_stats.append({
                        'symbol': symbol,
                        'signal_count': signal_count,
                        'details': signal_details
                    })
                    
            except Exception:
                continue
        
        try:
            bot.delete_message(message.chat.id, processing_msg.message_id)
        except:
            pass
        
        if not token_stats:
            bot.reply_to(message, f"🔍 Не знайдено токенів з {min_signals}+ сигналами")
            return
        
        # Сортуємо за кількістю сигналів
        token_stats.sort(key=lambda x: x['signal_count'], reverse=True)
        
        response = [f"🏆 <b>Топ токени з {min_signals}+ сигналами:</b>\n"]
        
        for stat in token_stats[:10]:
            # Групуємо сигнали по типу
            long_count = sum(1 for _, signal_type in stat['details'] if signal_type == "LONG")
            short_count = sum(1 for _, signal_type in stat['details'] if signal_type == "SHORT")
            
            response.append(
                f"\n📈 <b>{stat['symbol']}</b> - {stat['signal_count']} сигналів "
                f"({long_count}🟢 {short_count}🔴)"
            )
            
            # Додаємо топ-3 таймфрейми
            for interval, signal_type in stat['details'][:3]:
                emoji = "🟢" if signal_type == "LONG" else "🔴"
                response.append(f"   {emoji} {interval}")
        
        bot.reply_to(message, "\n".join(response), parse_mode="HTML")
        
    except Exception as e:
        bot.reply_to(message, f"❌ Помилка: {str(e)}")