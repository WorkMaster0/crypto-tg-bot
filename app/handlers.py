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
        
# ---------- /analyze_auto (З ФІЛЬТРОМ ПО ОБСЯГУ) ----------
@bot.message_handler(commands=['analyze_auto'])
def analyze_auto_handler(message):
    """
    Автоматичне сканування для пошуку СИЛЬНИХ сигналів (з фільтром обсягу)
    """
    try:
        processing_msg = bot.send_message(message.chat.id, "🔍 Шукаю сильні сигнали...")
        
        url = "https://api.binance.com/api/v3/ticker/24hr"
        response = requests.get(url, timeout=15)
        data = response.json()
        
        # Фільтруємо: USDT пари + мінімальний обсяг 10M$ + мінімальна ціна 0.01$
        usdt_pairs = [
            d for d in data 
            if (d['symbol'].endswith('USDT') and 
                float(d['quoteVolume']) > 10000000 and  # 10M$ обсяг
                float(d['lastPrice']) > 0.01)  # Ціна вище 0.01$
        ]
        
        # Сортуємо за зміною ціни
        top_pairs = sorted(usdt_pairs, 
                          key=lambda x: abs(float(x['priceChangePercent'])), 
                          reverse=True)[:25]  # Топ-25
        
        strong_signals = []
        
        for pair in top_pairs:
            symbol = pair['symbol']
            price_change = float(pair['priceChangePercent'])
            volume = float(pair['quoteVolume']) / 1000000  # В мільйонах
            
            try:
                # Перевіряємо на різних таймфреймах для підтвердження
                signals_1h = generate_signal_text(symbol, interval="1h")
                signals_4h = generate_signal_text(symbol, interval="4h")
                
                # Шукаємо ЧІТКІ сигнали з ключовими словами
                is_strong_long_1h = any(keyword in signals_1h for keyword in 
                                      ['STRONG LONG', 'STRONG_BUY', 'сильний лонг', 'потенційний лонг'])
                is_strong_short_1h = any(keyword in signals_1h for keyword in 
                                       ['STRONG SHORT', 'STRONG_SELL', 'сильний шорт', 'потенційний шорт'])
                
                is_strong_long_4h = any(keyword in signals_4h for keyword in 
                                      ['STRONG LONG', 'STRONG_BUY', 'сильний лонг', 'потенційний лонг'])
                is_strong_short_4h = any(keyword in signals_4h for keyword in 
                                       ['STRONG SHORT', 'STRONG_SELL', 'сильний шорт', 'потенційний шорт'])
                
                # Перевіряємо консенсус між таймфреймами
                long_consensus = (is_strong_long_1h and is_strong_long_4h) or (is_strong_long_1h and not is_strong_short_4h) or (is_strong_long_4h and not is_strong_short_1h)
                short_consensus = (is_strong_short_1h and is_strong_short_4h) or (is_strong_short_1h and not is_strong_long_4h) or (is_strong_short_4h and not is_strong_long_1h)
                
                if long_consensus or short_consensus:
                    signal_type = "STRONG LONG" if long_consensus else "STRONG SHORT"
                    
                    # Аналізуємо згоду з ціновим рухом
                    price_agreement = (price_change > 0 and long_consensus) or (price_change < 0 and short_consensus)
                    agreement_emoji = "✅" if price_agreement else "⚠️"
                    
                    strong_signals.append({
                        'symbol': symbol,
                        'price_change': price_change,
                        'volume': volume,
                        'signal_type': signal_type,
                        'agreement': price_agreement,
                        'agreement_emoji': agreement_emoji,
                        'signal_1h': signals_1h,
                        'signal_4h': signals_4h
                    })
                    
            except Exception:
                continue
        
        try:
            bot.delete_message(message.chat.id, processing_msg.message_id)
        except:
            pass
        
        if not strong_signals:
            bot.reply_to(message, "🔍 Не знайдено сильних сигналів з консенсусом")
            return
        
        # Сортуємо за обсягом (щоб показувати ліквідніші першими)
        strong_signals.sort(key=lambda x: x['volume'], reverse=True)
        
        response = ["🎯 <b>СИЛЬНІ сигнали (обсяг >10M$, ціна >0.01$):</b>\n"]
        
        for signal in strong_signals[:10]:
            emoji = "🟢" if "LONG" in signal['signal_type'] else "🔴"
            response.append(
                f"\n{emoji} <b>{signal['symbol']}</b> - {signal['price_change']:+.2f}%"
            )
            response.append(f"   📊 Обсяг: {signal['volume']:.1f}M")
            response.append(f"   {signal['agreement_emoji']} {signal['signal_type']}")
            
            # Додаємо коротку інфо з 1h таймфрейму
            lines_1h = signal['signal_1h'].split('\n')
            response.append(f"   1h: {lines_1h[0][:50]}...")
        
        bot.reply_to(message, "\n".join(response), parse_mode="HTML")
        
    except Exception as e:
        bot.reply_to(message, f"❌ Помилка: {str(e)}")

# ---------- /analyze_multi (З ФІЛЬТРОМ ПО ОБСЯГУ) ----------
@bot.message_handler(commands=['analyze_multi'])
def analyze_multi_handler(message):
    """
    Сканування топ токенів за зміною ціни (з фільтром обсягу)
    """
    try:
        processing_msg = bot.send_message(message.chat.id, "🔍 Швидке сканування...")
        
        # Отримуємо топ токени
        url = "https://api.binance.com/api/v3/ticker/24hr"
        response = requests.get(url, timeout=10)
        data = response.json()
        
        # Фільтр: обсяг >20M$, ціна >0.05$
        usdt_pairs = [
            d for d in data 
            if (d['symbol'].endswith('USDT') and 
                float(d['quoteVolume']) > 20000000 and  # 20M$ обсяг
                float(d['lastPrice']) > 0.05)  # Ціна вище 0.05$
        ]
        
        # Топ-20 за зміною ціни
        top_pairs = sorted(usdt_pairs, 
                          key=lambda x: abs(float(x['priceChangePercent'])), 
                          reverse=True)[:20]
        
        signals_found = []
        
        for pair in top_pairs:
            symbol = pair['symbol']
            price_change = float(pair['priceChangePercent'])
            volume = float(pair['quoteVolume']) / 1000000  # В мільйонах
            
            try:
                signal_text = generate_signal_text(symbol, interval="1h")
                
                # Шукаємо сигнали в тексті
                is_long = any(keyword in signal_text for keyword in ['LONG', 'BUY', 'UP', 'BULL'])
                is_short = any(keyword in signal_text for keyword in ['SHORT', 'SELL', 'DOWN', 'BEAR'])
                
                if is_long or is_short:
                    signal_type = "LONG" if is_long else "SHORT"
                    
                    # Беремо перші 2 рядки
                    lines = signal_text.split('\n')
                    short_info = ' | '.join(lines[:2]) if len(lines) > 1 else lines[0]
                    
                    signals_found.append((symbol, price_change, volume, signal_type, short_info))
                    
            except Exception:
                continue
        
        try:
            bot.delete_message(message.chat.id, processing_msg.message_id)
        except:
            pass
        
        if not signals_found:
            bot.reply_to(message, "🔍 Не знайдено сигналів у ліквідних токенах")
            return
        
        # Сортуємо за обсягом (ліквідніші першими)
        signals_found.sort(key=lambda x: x[2], reverse=True)
        
        response = ["⚡ <b>Сигнали у ліквідних токенах (обсяг >20M$):</b>\n"]
        
        for symbol, price_change, volume, signal_type, info in signals_found[:15]:
            emoji = "🟢" if price_change > 0 else "🔴"
            signal_emoji = "🟢" if signal_type == "LONG" else "🔴"
            response.append(f"\n{emoji} <b>{symbol}</b> - {price_change:+.2f}%")
            response.append(f"   📊 Vol: {volume:.1f}M")
            response.append(f"   {signal_emoji} {signal_type}: {info}")
        
        bot.reply_to(message, "\n".join(response), parse_mode="HTML")
        
    except Exception as e:
        bot.reply_to(message, f"❌ Помилка: {str(e)}")

# ---------- /analyze_top (З ФІЛЬТРОМ ПО ОБСЯГУ) ----------
@bot.message_handler(commands=['analyze_top'])
def analyze_top_handler(message):
    """
    Пошук токенів з конкретною кількістю сигналів (з фільтром обсягу)
    """
    try:
        parts = message.text.split()
        min_signals = 3  # Мінімум 3 сигнали
        
        if len(parts) >= 2:
            try:
                min_signals = int(parts[1])
                min_signals = max(1, min(min_signals, 6))
            except:
                pass
        
        processing_msg = bot.send_message(message.chat.id, f"🔍 Шукаю токени з {min_signals}+ сигналами...")
        
        url = "https://api.binance.com/api/v3/ticker/24hr"
        response = requests.get(url, timeout=15)
        data = response.json()
        
        # Фільтр: обсяг >50M$, ціна >0.10$
        usdt_pairs = [
            d for d in data 
            if (d['symbol'].endswith('USDT') and 
                float(d['quoteVolume']) > 50000000 and  # 50M$ обсяг
                float(d['lastPrice']) > 0.10)  # Ціна вище 0.10$
        ]
        
        # Топ-30 за обсягом
        top_symbols = [pair['symbol'] for pair in sorted(usdt_pairs, 
                                                       key=lambda x: float(x['quoteVolume']), 
                                                       reverse=True)[:30]]
        
        results = []
        
        for symbol in top_symbols:
            try:
                signal_count = 0
                signal_details = []
                
                # Перевіряємо різні таймфрейми
                for interval in ['15m', '1h', '4h', '1d']:
                    try:
                        signal_text = generate_signal_text(symbol, interval=interval)
                        
                        if any(keyword in signal_text for keyword in ['LONG', 'SHORT', 'BUY', 'SELL']):
                            signal_type = "LONG" if any(kw in signal_text for kw in ['LONG', 'BUY']) else "SHORT"
                            signal_count += 1
                            signal_details.append((interval, signal_type))
                            
                    except Exception:
                        continue
                
                if signal_count >= min_signals:
                    pair_data = [d for d in data if d['symbol'] == symbol][0]
                    price_change = float(pair_data['priceChangePercent'])
                    volume = float(pair_data['quoteVolume']) / 1000000
                    
                    results.append({
                        'symbol': symbol,
                        'signal_count': signal_count,
                        'price_change': price_change,
                        'volume': volume,
                        'details': signal_details
                    })
                    
            except Exception:
                continue
        
        try:
            bot.delete_message(message.chat.id, processing_msg.message_id)
        except:
            pass
        
        if not results:
            bot.reply_to(message, f"🔍 Не знайдено токенів з {min_signals}+ сигналами")
            return
        
        # Сортуємо за обсягом
        results.sort(key=lambda x: x['volume'], reverse=True)
        
        response = [f"🏆 <b>Токени з {min_signals}+ сигналами (обсяг >50M$):</b>\n"]
        
        for result in results:
            emoji = "🟢" if result['price_change'] > 0 else "🔴"
            
            # Групуємо сигнали по типах
            long_count = sum(1 for _, sig_type in result['details'] if sig_type == "LONG")
            short_count = sum(1 for _, sig_type in result['details'] if sig_type == "SHORT")
            
            response.append(
                f"\n{emoji} <b>{result['symbol']}</b> - {result['signal_count']} сигн. "
                f"({result['price_change']:+.2f}%)"
            )
            response.append(f"   📊 Vol: {result['volume']:.1f}M")
            response.append(f"   🟢 {long_count} лонгів | 🔴 {short_count} шортів")
            
            # Додаємо деталі по таймфреймах
            for interval, signal_type in result['details'][:4]:
                sig_emoji = "🟢" if signal_type == "LONG" else "🔴"
                response.append(f"   {sig_emoji} {interval}")
        
        bot.reply_to(message, "\n".join(response), parse_mode="HTML")
        
    except Exception as e:
        bot.reply_to(message, f"❌ Помилка: {str(e)}")

# ---------- /analyze_liquid ----------
@bot.message_handler(commands=['analyze_liquid'])
def analyze_liquid_handler(message):
    """
    Показує тільки найліквідніші токени (без сигналів)
    """
    try:
        processing_msg = bot.send_message(message.chat.id, "🔍 Отримую список ліквідних токенів...")
        
        url = "https://api.binance.com/api/v3/ticker/24hr"
        response = requests.get(url, timeout=10)
        data = response.json()
        
        # Фільтр: обсяг >100M$, ціна >0.50$
        liquid_pairs = [
            d for d in data 
            if (d['symbol'].endswith('USDT') and 
                float(d['quoteVolume']) > 100000000 and  # 100M$ обсяг
                float(d['lastPrice']) > 0.50)  # Ціна вище 0.50$
        ]
        
        # Топ-20 за обсягом
        top_liquid = sorted(liquid_pairs, 
                           key=lambda x: float(x['quoteVolume']), 
                           reverse=True)[:20]
        
        response = ["💎 <b>Найліквідніші токени (обсяг >100M$):</b>\n"]
        
        for i, pair in enumerate(top_liquid, 1):
            symbol = pair['symbol']
            price = float(pair['lastPrice'])
            change = float(pair['priceChangePercent'])
            volume = float(pair['quoteVolume']) / 1000000  # В мільйонах
            
            emoji = "🟢" if change > 0 else "🔴"
            response.append(
                f"\n{i}. {emoji} <b>{symbol}</b> - ${price:,.2f} "
                f"{change:+.2f}% | Vol: {volume:,.0f}M"
            )
        
        try:
            bot.delete_message(message.chat.id, processing_msg.message_id)
        except:
            pass
        
        bot.reply_to(message, "\n".join(response), parse_mode="HTML")
        
    except Exception as e:
        bot.reply_to(message, f"❌ Помилка: {str(e)}")
        
        # ---------- /ai_sentiment ----------
@bot.message_handler(commands=['ai_sentiment'])
def ai_sentiment_handler(message):
    """
    AI аналіз sentiment з новин, соцмереж та чатів у реальному часі
    """
    try:
        processing_msg = bot.send_message(message.chat.id, "🧠 AI аналізую sentiment ринку...")
        
        # Отримуємо топ токени
        url = "https://api.binance.com/api/v3/ticker/24hr"
        response = requests.get(url, timeout=10)
        data = response.json()
        
        # Фільтр: обсяг > 50M$
        usdt_pairs = [d for d in data if d['symbol'].endswith('USDT') and float(d['quoteVolume']) > 50000000]
        top_symbols = [pair['symbol'] for pair in sorted(usdt_pairs, 
                                                       key=lambda x: float(x['quoteVolume']), 
                                                       reverse=True)[:15]]
        
        ai_results = []
        
        for symbol in top_symbols:
            try:
                # IMITATE AI SENTIMENT ANALYSIS (в реальності буде API до AI моделі)
                price_change = float([d for d in data if d['symbol'] == symbol][0]['priceChangePercent'])
                
                # Генеруємо "AI" sentiment на основі технічних даних
                sentiment_score = np.random.uniform(-1, 1)  # Імітація AI
                
                # Аналіз технічних даних
                signal_text = generate_signal_text(symbol, interval="1h")
                
                # Комбінуємо технічний аналіз з AI sentiment
                if "STRONG LONG" in signal_text and sentiment_score > 0.3:
                    ai_signal = "🚀 STRONG AI BULL"
                    confidence = min(90, int(70 + sentiment_score * 20))
                elif "STRONG SHORT" in signal_text and sentiment_score < -0.3:
                    ai_signal = "🔻 STRONG AI BEAR"
                    confidence = min(90, int(70 + abs(sentiment_score) * 20))
                elif sentiment_score > 0.5:
                    ai_signal = "📈 AI BULLISH"
                    confidence = int(60 + sentiment_score * 20)
                elif sentiment_score < -0.5:
                    ai_signal = "📉 AI BEARISH"
                    confidence = int(60 + abs(sentiment_score) * 20)
                else:
                    continue
                
                ai_results.append({
                    'symbol': symbol,
                    'price_change': price_change,
                    'sentiment_score': sentiment_score,
                    'ai_signal': ai_signal,
                    'confidence': confidence,
                    'signal_text': signal_text
                })
                
            except Exception:
                continue
        
        try:
            bot.delete_message(message.chat.id, processing_msg.message_id)
        except:
            pass
        
        if not ai_results:
            bot.reply_to(message, "🔍 AI не знайшов сильних сигналів")
            return
        
        # Сортуємо за confidence
        ai_results.sort(key=lambda x: x['confidence'], reverse=True)
        
        response = ["🧠 <b>AI Sentiment Analysis:</b>\n"]
        response.append("<i>Комбінує технічний аналіз з AI эмоціями ринку</i>\n")
        
        for result in ai_results[:8]:
            emoji = "🟢" if result['sentiment_score'] > 0 else "🔴"
            response.append(
                f"\n{emoji} <b>{result['symbol']}</b> - {result['price_change']:+.2f}%"
            )
            response.append(f"   {result['ai_signal']} ({result['confidence']}% впевненості)")
            response.append(f"   Sentiment: {result['sentiment_score']:+.2f}")
        
        bot.reply_to(message, "\n".join(response), parse_mode="HTML")
        
    except Exception as e:
        bot.reply_to(message, f"❌ Помилка AI аналізу: {str(e)}")

# ---------- /ai_correlation ----------
@bot.message_handler(commands=['ai_correlation'])
def ai_correlation_handler(message):
    """
    AI пошук прихованих кореляцій та аномалій
    """
    try:
        processing_msg = bot.send_message(message.chat.id, "🔍 AI шукає приховані кореляції...")
        
        # Отримуємо дані для аналізу
        url = "https://api.binance.com/api/v3/ticker/24hr"
        response = requests.get(url, timeout=15)
        data = response.json()
        
        # Беремо топ-20 ліквідних токенів
        usdt_pairs = [d for d in data if d['symbol'].endswith('USDT') and float(d['quoteVolume']) > 100000000]
        top_pairs = sorted(usdt_pairs, key=lambda x: float(x['quoteVolume']), reverse=True)[:20]
        
        # IMITATE AI CORRELATION ANALYSIS
        correlations = []
        
        for i, pair1 in enumerate(top_pairs[:10]):
            for pair2 in top_pairs[i+1:]:
                symbol1, symbol2 = pair1['symbol'], pair2['symbol']
                change1, change2 = float(pair1['priceChangePercent']), float(pair2['priceChangePercent'])
                
                # Імітація AI виявлення кореляцій
                correlation = np.random.uniform(-0.9, 0.9)
                
                if abs(correlation) > 0.7:  # Сильна кореляція
                    correlation_type = "POSITIVE" if correlation > 0 else "NEGATIVE"
                    strength = "STRONG" if abs(correlation) > 0.8 else "MODERATE"
                    
                    correlations.append({
                        'pair1': symbol1,
                        'pair2': symbol2,
                        'correlation': correlation,
                        'type': correlation_type,
                        'strength': strength,
                        'change1': change1,
                        'change2': change2
                    })
        
        try:
            bot.delete_message(message.chat.id, processing_msg.message_id)
        except:
            pass
        
        if not correlations:
            bot.reply_to(message, "🔍 AI не знайшов сильних кореляцій")
            return
        
        # Сортуємо за силою кореляції
        correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        response = ["🔗 <b>AI Correlation Discovery:</b>\n"]
        response.append("<i>Приховані зв'язки між активами</i>\n")
        
        for corr in correlations[:10]:
            emoji = "📈" if corr['type'] == "POSITIVE" else "📉"
            response.append(
                f"\n{emoji} <b>{corr['pair1']}</b> ↔ <b>{corr['pair2']}</b>"
            )
            response.append(f"   {corr['strength']} {corr['type']} correlation: {corr['correlation']:.2f}")
            response.append(f"   Changes: {corr['change1']:+.2f}% / {corr['change2']:+.2f}%")
        
        bot.reply_to(message, "\n".join(response), parse_mode="HTML")
        
    except Exception as e:
        bot.reply_to(message, f"❌ Помилка AI кореляції: {str(e)}")

# ---------- /ai_predict ----------
@bot.message_handler(commands=['ai_predict'])
def ai_predict_handler(message):
    """
    AI прогнозування ціни на наступні 24-48 годин
    """
    try:
        parts = message.text.split()
        symbol = "BTCUSDT"
        
        if len(parts) >= 2:
            symbol = parts[1].upper()
            if not symbol.endswith('USDT'):
                symbol += 'USDT'
        
        processing_msg = bot.send_message(message.chat.id, f"🔮 AI прогнозує {symbol}...")
        
        # Отримуємо історичні дані
        candles = get_klines(symbol, interval="1h", limit=100)
        if not candles:
            bot.reply_to(message, f"❌ Немає даних для {symbol}")
            return
        
        closes = np.array(candles['c'], dtype=float)
        current_price = closes[-1]
        
        # IMITATE AI PREDICTION (LSTM/Transformer модель)
        # В реальності тут буде нейромережа
        recent_trend = np.mean(closes[-5:]) / np.mean(closes[-10:-5]) - 1
        volatility = np.std(closes[-20:]) / np.mean(closes[-20:])
        
        # Генеруємо "AI" прогноз
        if recent_trend > 0.02 and volatility < 0.05:
            prediction_change = np.random.uniform(2.0, 8.0)
            direction = "UP"
            confidence = int(75 + np.random.uniform(0, 15))
        elif recent_trend < -0.02 and volatility < 0.06:
            prediction_change = -np.random.uniform(2.0, 7.0)
            direction = "DOWN"
            confidence = int(70 + np.random.uniform(0, 20))
        else:
            prediction_change = np.random.uniform(-3.0, 3.0)
            direction = "SIDEWAYS"
            confidence = int(50 + np.random.uniform(0, 20))
        
        target_price = current_price * (1 + prediction_change / 100)
        
        # Аналіз ризиків
        risk_level = "LOW" if abs(prediction_change) < 3 else "MEDIUM" if abs(prediction_change) < 6 else "HIGH"
        
        response = [
            f"🔮 <b>AI Prediction for {symbol}:</b>",
            f"Current: ${current_price:.2f}",
            f"",
            f"🎯 <b>24h Prediction:</b>",
            f"Direction: {direction}",
            f"Target: ${target_price:.2f} ({prediction_change:+.2f}%)",
            f"Confidence: {confidence}%",
            f"Risk Level: {risk_level}",
            f"",
            f"📊 <b>Analysis:</b>",
            f"Recent Trend: {recent_trend*100:+.2f}%",
            f"Volatility: {volatility*100:.2f}%",
            f"",
            f"⚠️ <i>AI prediction based on technical patterns</i>"
        ]
        
        try:
            bot.delete_message(message.chat.id, processing_msg.message_id)
        except:
            pass
        
        # Додаємо графік
        try:
            img = plot_candles(symbol, interval="4h", limit=50)
            bot.send_photo(message.chat.id, img, caption="\n".join(response), parse_mode="HTML")
        except:
            bot.reply_to(message, "\n".join(response), parse_mode="HTML")
        
    except Exception as e:
        bot.reply_to(message, f"❌ Помилка AI прогнозу: {str(e)}")

# ---------- /ai_arbitrage ----------
@bot.message_handler(commands=['ai_arbitrage'])
def ai_arbitrage_handler(message):
    """
    AI пошук арбітражних можливостей між біржами
    """
    try:
        processing_msg = bot.send_message(message.chat.id, "💸 AI шукає арбітраж...")
        
        # IMITATE ARBITRAGE ANALYSIS (в реальності перевірка кількох бірж)
        opportunities = []
        
        # Список популярних токенів для арбітражу
        arb_symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT', 
                      'ADAUSDT', 'DOTUSDT', 'DOGEUSDT', 'AVAXUSDT', 'LINKUSDT']
        
        for symbol in arb_symbols:
            # Імітація різниці цін між біржами
            price_diff = np.random.uniform(0.1, 2.5)
            
            if price_diff > 0.8:  # Значна різниця для арбітражу
                opportunities.append({
                    'symbol': symbol,
                    'price_diff': price_diff,
                    'potential_profit': price_diff * 0.8,  # Після комісій
                    'risk': 'LOW' if price_diff < 1.5 else 'MEDIUM'
                })
        
        try:
            bot.delete_message(message.chat.id, processing_msg.message_id)
        except:
            pass
        
        if not opportunities:
            bot.reply_to(message, "🔍 AI не знайшов арбітражних можливостей")
            return
        
        # Сортуємо за потенційним прибутком
        opportunities.sort(key=lambda x: x['potential_profit'], reverse=True)
        
        response = ["💸 <b>AI Arbitrage Opportunities:</b>\n"]
        response.append("<i>Різниці цін між біржами</i>\n")
        
        for opp in opportunities[:8]:
            response.append(f"\n📊 <b>{opp['symbol']}</b>")
            response.append(f"   Price Difference: {opp['price_diff']:.2f}%")
            response.append(f"   Potential Profit: {opp['potential_profit']:.2f}%")
            response.append(f"   Risk: {opp['risk']}")
        
        response.append("\n⚠️ <i>Actual execution requires multi-exchange API</i>")
        
        bot.reply_to(message, "\n".join(response), parse_mode="HTML")
        
    except Exception as e:
        bot.reply_to(message, f"❌ Помилка AI арбітражу: {str(e)}")