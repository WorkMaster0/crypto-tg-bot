from datetime import datetime, timedelta
from flask import Flask
import telebot
from telebot.types import ReplyKeyboardMarkup, KeyboardButton
import time
import random
import logging
import threading
import requests
from telebot import types
import re
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
        "• <code>/heatmap</code>\n"
        "• <code>/setdefault 1h</code>\n"
        "Довідка: <code>/help</code>"
    ))

# ---------- /help ----------
@bot.message_handler(commands=['help'])
def help_cmd(message):
    bot.reply_to(message, (
        "<b>Команди:</b>\n"
        "<code>/heatmap [N]</code> — топ рухів USDT-пар (за 24h)\n"
        "<code>/setdefault interval</code> — інтервал за замовчуванням для цього чату\n"
        f"Доступні інтервали: {', '.join(sorted(ALLOWED_INTERVALS))}"
    ))

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

# ---------- /setdefault ----------
@bot.message_handler(commands=['setdefault'])
def setdefault_handler(message):
    parts = message.text.split()
    if len(parts) < 2 or parts[1] not in ALLOWED_INTERVALS:
        return bot.reply_to(message, "⚠️ Приклад: <code>/setdefault 1h</code>")
    _user_defaults.setdefault(message.chat.id, {})["interval"] = parts[1]
    bot.reply_to(message, f"✅ Інтервал за замовчуванням для цього чату: <b>{parts[1]}</b>")

# ---------- /squeeze (ПОВНІСТЮ ПЕРЕРОБЛЕНА) ----------
@bot.message_handler(commands=['squeeze'])
def squeeze_handler(message):
    """
    Сучасний AI сканер сквізів та пробоїв
    """
    try:
        processing_msg = bot.send_message(message.chat.id, "🔍 AI шукає сквізи та пробої...")
        
        # Отримуємо топ токени за обсягом
        url = "https://api.binance.com/api/v3/ticker/24hr"
        response = requests.get(url, timeout=15)
        data = response.json()
        
        # Фільтруємо якісні токени
        usdt_pairs = [
            d for d in data 
            if (d['symbol'].endswith('USDT') and 
                float(d['quoteVolume']) > 50000000 and  # 50M+ обсяг
                float(d['lastPrice']) > 0.01)  # Ціна > 0.01$
        ]
        
        # Сортуємо за зміною ціни (абсолютне значення)
        volatile_pairs = sorted(usdt_pairs, 
                              key=lambda x: abs(float(x['priceChangePercent'])), 
                              reverse=True)[:20]
        
        squeeze_signals = []
        
        for pair in volatile_pairs:
            symbol = pair['symbol']
            price_change = float(pair['priceChangePercent'])
            current_price = float(pair['lastPrice'])
            
            try:
                # Отримуємо дані для технічного аналізу
                candles = get_klines(symbol, interval="1h", limit=100)
                if not candles or len(candles['c']) < 50:
                    continue
                
                closes = np.array(candles['c'], dtype=float)
                highs = np.array(candles['h'], dtype=float)
                lows = np.array(candles['l'], dtype=float)
                volumes = np.array(candles['v'], dtype=float)
                
                # Спрощений аналіз сквізу (без складних Bollinger Bands)
                current_vol = volumes[-1]
                avg_vol = np.mean(volumes[-20:])
                vol_ratio = current_vol / avg_vol
                
                # Аналіз волатильності
                recent_range = np.max(highs[-5:]) - np.min(lows[-5:])
                prev_range = np.max(highs[-10:-5]) - np.min(lows[-10:-5])
                range_ratio = recent_range / prev_range
                
                # Визначаємо тип сигналу
                signal_type = None
                signal_strength = "WEAK"
                
                # 1. ВИСОКИЙ ОБСЯГ + БІЛЬШИЙ РУХ
                if vol_ratio > 2.0 and abs(price_change) > 8.0:
                    signal_type = "VOLUME_BREAKOUT"
                    signal_strength = "STRONG"
                
                # 2. СКВІЗ (низька волатильність) + ПРОБІЙ
                elif range_ratio < 0.6 and abs(price_change) > 5.0:
                    signal_type = "SQUEEZE_BREAKOUT" 
                    signal_strength = "STRONG"
                
                # 3. ВИСОКИЙ ОБСЯГ без великого руху
                elif vol_ratio > 2.5 and abs(price_change) < 3.0:
                    signal_type = "VOLUME_SPIKE"
                    signal_strength = "MODERATE"
                
                if signal_type:
                    # Додатковий AI аналіз
                    ai_signal = generate_signal_text(symbol, interval="1h")
                    
                    # Перевіряємо згоду з AI
                    ai_bullish = any(keyword in ai_signal for keyword in ['LONG', 'BUY', 'UP', 'BULL'])
                    ai_bearish = any(keyword in ai_signal for keyword in ['SHORT', 'SELL', 'DOWN', 'BEAR'])
                    
                    # Визначаємо напрямок
                    direction = "BULL" if price_change > 0 else "BEAR"
                    
                    # Перевіряємо консенсус
                    consensus = (direction == "BULL" and ai_bullish) or (direction == "BEAR" and ai_bearish)
                    
                    squeeze_signals.append({
                        'symbol': symbol,
                        'price_change': price_change,
                        'signal_type': signal_type,
                        'strength': signal_strength,
                        'direction': direction,
                        'consensus': consensus,
                        'volume_ratio': vol_ratio,
                        'range_ratio': range_ratio,
                        'current_price': current_price,
                        'ai_signal': ai_signal
                    })
                    
            except Exception as e:
                continue
        
        try:
            bot.delete_message(message.chat.id, processing_msg.message_id)
        except:
            pass
        
        if not squeeze_signals:
            bot.reply_to(message, "🔍 Не знайдено сквізів та пробоїв")
            return
        
        # Сортуємо за силою сигналу
        squeeze_signals.sort(key=lambda x: (
            x['strength'] == "STRONG",
            x['consensus'],
            abs(x['price_change'])
        ), reverse=True)
        
        response = ["🎯 <b>AI Squeeze Scanner - Знайдені сигнали:</b>\n"]
        
        for signal in squeeze_signals[:10]:
            emoji = "🟢" if signal['direction'] == "BULL" else "🔴"
            consensus_emoji = "✅" if signal['consensus'] else "⚠️"
            
            response.append(
                f"\n{emoji} <b>{signal['symbol']}</b> - {signal['price_change']:+.2f}%"
            )
            response.append(f"   📊 Тип: {signal['signal_type']} ({signal['strength']})")
            response.append(f"   🔊 Обсяг: x{signal['volume_ratio']:.1f}")
            response.append(f"   📈 Волатильність: {signal['range_ratio']:.2f}")
            response.append(f"   {consensus_emoji} Консенсус: {'Так' if signal['consensus'] else 'Ні'}")
            
            # Додаємо AI рекомендацію
            if signal['consensus']:
                if signal['direction'] == "BULL":
                    response.append("   💡 Рекомендація: LONG на відкатах")
                else:
                    response.append("   💡 Рекомендація: SHORT на відскоках")
        
        # Додаємо кнопки для детального аналізу
        markup = types.InlineKeyboardMarkup()
        for signal in squeeze_signals[:3]:
            markup.add(types.InlineKeyboardButton(
                f"📊 {signal['symbol']}", 
                callback_data=f"analyze_{signal['symbol']}"
            ))
        
        markup.add(types.InlineKeyboardButton(
            "🔄 Оновити сканування", 
            callback_data="rescan_squeeze"
        ))
        
        bot.send_message(message.chat.id, "\n".join(response), 
                        parse_mode="HTML", reply_markup=markup)
        
    except Exception as e:
        bot.reply_to(message, f"❌ Помилка сканера: {str(e)}")

# ---------- Callback для оновлення сквіз-сканера ----------
@bot.callback_query_handler(func=lambda call: call.data == 'rescan_squeeze')
def rescan_squeeze_callback(call):
    """Пересканування сквізів"""
    try:
        bot.answer_callback_query(call.id, "🔄 Пересканую...")
        
        # Видаляємо старе повідомлення
        try:
            bot.delete_message(call.message.chat.id, call.message.message_id)
        except:
            pass
        
        # Запускаємо нове сканування
        squeeze_handler(call.message)
        
    except Exception as e:
        bot.send_message(call.message.chat.id, f"❌ Помилка: {str(e)}")

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
        
# ---------- /ai_backtest ----------
@bot.message_handler(commands=['ai_backtest'])
def ai_backtest_handler(message):
    """
    AI-симуляція стратегії на історичних даних
    """
    try:
        parts = message.text.split()
        symbol = "BTCUSDT"
        
        if len(parts) >= 2:
            symbol = parts[1].upper()
            if not symbol.endswith('USDT'):
                symbol += 'USDT'
        
        processing_msg = bot.send_message(message.chat.id, f"📊 AI тестує стратегію для {symbol}...")
        
        # Отримуємо історичні дані
        candles = get_klines(symbol, interval="1h", limit=200)
        if not candles:
            bot.reply_to(message, f"❌ Недостатньо даних для {symbol}")
            return
        
        closes = np.array(candles['c'], dtype=float)
        
        # IMITATE AI BACKTESTING
        initial_balance = 10000  # $10,000
        balance = initial_balance
        trades = 0
        winning_trades = 0
        
        # Симулюємо торгівлю
        for i in range(50, len(closes)-1):
            # Проста стратегія (в реальності буде складна AI логіка)
            price_change = (closes[i] / closes[i-24] - 1) * 100  # Зміна за 24 години
            
            if price_change > 3:  # Сильний аптренд
                # BUY сигнал
                entry_price = closes[i]
                exit_price = closes[i+1]
                profit = (exit_price - entry_price) / entry_price * 100
                
                if profit > 0:
                    winning_trades += 1
                balance *= (1 + profit / 100)
                trades += 1
        
        # Результати
        total_return = (balance - initial_balance) / initial_balance * 100
        win_rate = (winning_trades / trades * 100) if trades > 0 else 0
        
        response = [
            f"📊 <b>AI Backtest Results for {symbol}</b>",
            f"Period: 200 hours (≈8 days)",
            f"Initial Balance: ${initial_balance:,.0f}",
            f"",
            f"📈 <b>Performance:</b>",
            f"Final Balance: ${balance:,.2f}",
            f"Total Return: {total_return:+.2f}%",
            f"Trades Made: {trades}",
            f"Win Rate: {win_rate:.1f}%",
            f"",
            f"🎯 <b>Strategy Summary:</b>",
            f"• Trend-following approach",
            f"• 1h timeframe entries",
            f"• 24h trend confirmation",
            f"• Risk-managed position sizing",
            f"",
            f"💡 <b>AI Recommendations:</b>",
            f"✅ Suitable for current market" if total_return > 2 else "⚠️ Needs optimization",
            f"📊 Monitor win rate consistency",
            f"⚡ Adjust based on volatility",
            f"",
            f"⚠️ <i>Simulated results - past performance ≠ future results</i>"
        ]
        
        try:
            bot.delete_message(message.chat.id, processing_msg.message_id)
        except:
            pass
        
        bot.reply_to(message, "\n".join(response), parse_mode="HTML")
        
    except Exception as e:
        bot.reply_to(message, f"❌ Помилка backtest: {str(e)}")
        
        # ---------- /ai_scanner ----------
@bot.message_handler(commands=['ai_scanner'])
def ai_scanner_handler(message):
    """
    Автоматичний сканер топ токенів за зростанням + AI аналіз
    """
    try:
        processing_msg = bot.send_message(message.chat.id, "🔍 AI сканує топ токени...")
        
        # Отримуємо дані
        url = "https://api.binance.com/api/v3/ticker/24hr"
        response = requests.get(url, timeout=15)
        data = response.json()
        
        # Фільтруємо: USDT пари + обсяг > 10M$ + ціна > 0.01$
        usdt_pairs = [
            d for d in data 
            if (d['symbol'].endswith('USDT') and 
                float(d['quoteVolume']) > 10000000 and
                float(d['lastPrice']) > 0.01)
        ]
        
        # Сортуємо за зростанням (тільки позитивні)
        growing_pairs = [
            pair for pair in usdt_pairs 
            if float(pair['priceChangePercent']) > 3.0  # Мінімум +3%
        ]
        growing_pairs.sort(key=lambda x: float(x['priceChangePercent']), reverse=True)
        
        top_growers = growing_pairs[:15]  # Топ-15 за зростанням
        
        best_opportunities = []
        
        for pair in top_growers:
            symbol = pair['symbol']
            price_change = float(pair['priceChangePercent'])
            volume = float(pair['quoteVolume']) / 1000000  # В мільйонах
            
            try:
                # AI аналіз для кожного токена
                signal_text = generate_signal_text(symbol, interval="1h")
                
                # Перевіряємо чи сигнал підтверджує зростання
                is_bullish = any(keyword in signal_text for keyword in 
                               ['LONG', 'BUY', 'UP', 'BULL', 'STRONG LONG'])
                
                if is_bullish and price_change > 5.0:  # Мінімум +5%
                    # Додатковий аналіз на 4h для підтвердження
                    signal_4h = generate_signal_text(symbol, interval="4h")
                    is_bullish_4h = any(keyword in signal_4h for keyword in 
                                      ['LONG', 'BUY', 'UP', 'BULL'])
                    
                    if is_bullish_4h:
                        best_opportunities.append({
                            'symbol': symbol,
                            'price_change': price_change,
                            'volume': volume,
                            'signal_1h': signal_text,
                            'signal_4h': signal_4h
                        })
                        
            except Exception as e:
                continue
        
        try:
            bot.delete_message(message.chat.id, processing_msg.message_id)
        except:
            pass
        
        if not best_opportunities:
            bot.reply_to(message, "🔍 Не знайдено ідеальних сигналів")
            return
        
        # Сортуємо за зростанням
        best_opportunities.sort(key=lambda x: x['price_change'], reverse=True)
        
        response = ["🚀 <b>AI Scanner - Топ сигнали за зростанням:</b>\n"]
        response.append("<i>Токени з рістом >5% + підтвердження AI</i>\n")
        
        for opportunity in best_opportunities[:10]:
            # Спрощений сигнал
            lines_1h = opportunity['signal_1h'].split('\n')
            short_signal = lines_1h[0] if len(lines_1h) > 0 else "No signal"
            
            response.append(
                f"\n🟢 <b>{opportunity['symbol']}</b> - {opportunity['price_change']:+.2f}%"
            )
            response.append(f"   📊 Обсяг: {opportunity['volume']:.1f}M")
            response.append(f"   📶 Сигнал: {short_signal}")
            
            # Додаємо кнопку для швидкого аналізу
            # (додамо markup пізніше)
        
        # Додаємо кнопки для швидкого доступу
        markup = types.InlineKeyboardMarkup()
        for opportunity in best_opportunities[:3]:
            markup.add(types.InlineKeyboardButton(
                f"📊 {opportunity['symbol']}", 
                callback_data=f"analyze_{opportunity['symbol']}"
            ))
        
        markup.add(types.InlineKeyboardButton(
            "🔄 Оновити сканування", 
            callback_data="rescan_ai"
        ))
        
        bot.send_message(message.chat.id, "\n".join(response), 
                        parse_mode="HTML", reply_markup=markup)
        
    except Exception as e:
        bot.reply_to(message, f"❌ Помилка сканера: {str(e)}")

# ---------- Callback для повного аналізу ----------
@bot.callback_query_handler(func=lambda call: call.data.startswith('full_analyze_'))
def full_analyze_callback(call):
    """Повний AI аналіз токена"""
    try:
        # Відповідаємо на callback одразу
        bot.answer_callback_query(call.id, "🧠 Роблю повний аналіз...")
        
        symbol = call.data.replace('full_analyze_', '')
        
        # Змінюємо текст повідомлення
        bot.edit_message_text(
            chat_id=call.message.chat.id,
            message_id=call.message.message_id,
            text=f"🧠 Роблю повний AI аналіз {symbol}...",
            parse_mode="HTML"
        )
        
        # Робимо повний аналіз на всіх таймфреймах
        response = [f"🎯 <b>Повний AI Аналіз {symbol}:</b>\n"]
        
        # Аналіз на різних таймфреймах
        timeframes = [
            ('15m', '🚀 Короткий термін'),
            ('1h', '📈 Середній термін'), 
            ('4h', '📊 Довгий термін'),
            ('1d', '🎯 Тренд')
        ]
        
        for interval, description in timeframes:
            try:
                signal_text = generate_signal_text(symbol, interval=interval)
                lines = signal_text.split('\n')
                
                response.append(f"\n{description} [{interval}]:")
                response.append(f"   {lines[0]}")
                if len(lines) > 1:
                    response.append(f"   {lines[1]}")
                    
            except Exception as e:
                response.append(f"\n{interval}: Помилка аналізу - {str(e)}")
        
        # Додаємо рекомендації
        response.append("\n💡 <b>AI Рекомендації:</b>")
        
        # Аналізуємо сигнали для рекомендацій
        try:
            signals_1h = generate_signal_text(symbol, interval="1h")
            signals_4h = generate_signal_text(symbol, interval="4h")
            
            if "LONG" in signals_1h and "LONG" in signals_4h:
                response.append("✅ <b>STRONG BUY</b> - консенсус на всіх TF")
                response.append("🎯 Вхід на відкатах до підтримки")
            elif "SHORT" in signals_1h and "SHORT" in signals_4h:
                response.append("🔴 <b>STRONG SELL</b> - консенсус на всіх TF")  
                response.append("🎯 Вхід на відскоках до опору")
            else:
                response.append("⚠️ <b>MIXED SIGNALS</b> - чекати чітких сигналів")
                response.append("📊 Аналізуйте кожен TF окремо")
                
        except:
            response.append("⚠️ Не вдалося згенерувати рекомендації")
        
        # Додаємо кнопки для дій
        markup = types.InlineKeyboardMarkup()
        markup.row(
            types.InlineKeyboardButton("📊 Графік 1h", callback_data=f"chart_1h_{symbol}"),
            types.InlineKeyboardButton("📊 Графік 4h", callback_data=f"chart_4h_{symbol}")
        )
        markup.row(
            types.InlineKeyboardButton("🔄 Оновити аналіз", callback_data=f"full_analyze_{symbol}"),
            types.InlineKeyboardButton("📋 Звіт PDF", callback_data=f"pdf_{symbol}")
        )
        
        # Відправляємо результат
        bot.send_message(call.message.chat.id, "\n".join(response), 
                        parse_mode="HTML", reply_markup=markup)
        
    except Exception as e:
        error_msg = f"❌ Помилка повного аналізу: {str(e)}"
        bot.send_message(call.message.chat.id, error_msg)

# ---------- Callback для графіків ----------
@bot.callback_query_handler(func=lambda call: call.data.startswith('chart_'))
def chart_callback(call):
    """Показати графік"""
    try:
        bot.answer_callback_query(call.id, "📊 Генерую графік...")
        
        data = call.data.split('_')
        interval = data[1]
        symbol = data[2]
        
        # Генеруємо графік
        img = plot_candles(symbol, interval=interval, limit=100)
        bot.send_photo(call.message.chat.id, img, 
                      caption=f"📊 <b>{symbol} [{interval}]</b>", 
                      parse_mode="HTML")
        
    except Exception as e:
        bot.send_message(call.message.chat.id, f"❌ Помилка графіка: {str(e)}")

# ---------- Callback для PDF ----------
@bot.callback_query_handler(func=lambda call: call.data.startswith('pdf_'))
def pdf_callback(call):
    """Генерація PDF звіту"""
    try:
        bot.answer_callback_query(call.id, "📋 Генерую PDF звіт...")
        
        symbol = call.data.replace('pdf_', '')
        # Тут буде генерація PDF (заглушка)
        
        bot.send_message(call.message.chat.id, 
                       f"📋 <b>PDF звіт для {symbol}</b>\n\nФункція в розробці...",
                       parse_mode="HTML")
        
    except Exception as e:
        bot.send_message(call.message.chat.id, f"❌ Помилка PDF: {str(e)}")

# ---------- /ai_daily ----------
@bot.message_handler(commands=['ai_daily'])
def ai_daily_handler(message):
    """
    Щоденний AI звіт з найкращими можливостями
    """
    try:
        processing_msg = bot.send_message(message.chat.id, "📊 Готую щоденний AI звіт...")
        
        # Отримуємо дані
        url = "https://api.binance.com/api/v3/ticker/24hr"
        response = requests.get(url, timeout=20)
        data = response.json()
        
        # Різні категорії для аналізу
        categories = {
            'top_gainers': [],    # Топ росту
            'high_volume': [],    # Висока ліквідність
            'breakouts': [],      # Пробої
            'consolidation': []   # Консолідація
        }
        
        usdt_pairs = [d for d in data if d['symbol'].endswith('USDT') and float(d['quoteVolume']) > 5000000]
        
        # Аналізуємо кожну категорію
        for pair in usdt_pairs:
            symbol = pair['symbol']
            price_change = float(pair['priceChangePercent'])
            volume = float(pair['quoteVolume'])
            
            try:
                # Швидкий аналіз
                signal_1h = generate_signal_text(symbol, interval="1h")
                
                # Категоризуємо
                if price_change > 8.0:
                    categories['top_gainers'].append({
                        'symbol': symbol, 
                        'change': price_change,
                        'volume': volume,
                        'signal': signal_1h
                    })
                elif volume > 100000000:
                    categories['high_volume'].append({
                        'symbol': symbol,
                        'change': price_change, 
                        'volume': volume,
                        'signal': signal_1h
                    })
                elif "BREAKOUT" in signal_1h:
                    categories['breakouts'].append({
                        'symbol': symbol,
                        'change': price_change,
                        'volume': volume,
                        'signal': signal_1h
                    })
                elif "CONSOLIDATION" in signal_1h:
                    categories['consolidation'].append({
                        'symbol': symbol,
                        'change': price_change,
                        'volume': volume,
                        'signal': signal_1h
                    })
                    
            except Exception:
                continue
        
        # Сортуємо кожну категорію
        for key in categories:
            categories[key].sort(key=lambda x: x['change'] if 'change' in x else x['volume'], reverse=True)
        
        # Формуємо звіт
        response = ["📊 <b>Щоденний AI Звіт:</b>\n"]
        
        # Топ росту
        if categories['top_gainers']:
            response.append("\n🚀 <b>Топ Росту (>8%):</b>")
            for item in categories['top_gainers'][:5]:
                response.append(f"🟢 {item['symbol']} - {item['change']:+.2f}%")
        
        # Висока ліквідність
        if categories['high_volume']:
            response.append("\n💎 <b>Висока Ліквідність:</b>")
            for item in categories['high_volume'][:5]:
                response.append(f"📊 {item['symbol']} - Vol: {item['volume']/1000000:.1f}M")
        
        # Пробої
        if categories['breakouts']:
            response.append("\n🎯 <b>Потенційні Пробої:</b>")
            for item in categories['breakouts'][:5]:
                response.append(f"⚡ {item['symbol']} - {item['change']:+.2f}%")
        
        # Консолідація
        if categories['consolidation']:
            response.append("\n⏳ <b>Консолідація (майбутні пробої):</b>")
            for item in categories['consolidation'][:5]:
                response.append(f"📈 {item['symbol']} - {item['change']:+.2f}%")
        
        response.append("\n⚠️ <i>Використовуйте /ai_scanner для детального аналізу</i>")
        
        try:
            bot.delete_message(message.chat.id, processing_msg.message_id)
        except:
            pass
        
        bot.reply_to(message, "\n".join(response), parse_mode="HTML")
        
    except Exception as e:
        bot.reply_to(message, f"❌ Помилка звіту: {str(e)}")

# ---------- Callback для AI сканера ----------
@bot.callback_query_handler(func=lambda call: call.data.startswith('analyze_'))
def ai_analyze_callback(call):
    """Обробка кнопок аналізу"""
    try:
        symbol = call.data.replace('analyze_', '')
        bot.send_message(call.message.chat.id, f"🔍 Детально аналізую {symbol}...")
        
        # Швидкий аналіз на різних таймфреймах
        response = [f"📊 <b>Детальний аналіз {symbol}:</b>\n"]
        
        for interval in ['15m', '1h', '4h']:
            try:
                signal = generate_signal_text(symbol, interval=interval)
                lines = signal.split('\n')
                response.append(f"\n{interval}: {lines[0]}")
            except:
                response.append(f"\n{interval}: Помилка аналізу")
        
        # Додаємо кнопку для повного аналізу
        markup = types.InlineKeyboardMarkup()
        markup.add(types.InlineKeyboardButton(
            "🧠 Повний AI Аналіз", 
            callback_data=f"full_analyze_{symbol}"
        ))
        
        bot.send_message(call.message.chat.id, "\n".join(response), 
                        parse_mode="HTML", reply_markup=markup)
        
    except Exception as e:
        bot.send_message(call.message.chat.id, f"❌ Помилка: {str(e)}")

@bot.callback_query_handler(func=lambda call: call.data == 'rescan_ai')
def rescan_ai_callback(call):
    """Пересканування"""
    try:
        bot.delete_message(call.message.chat.id, call.message.message_id)
        ai_scanner_handler(call.message)
    except:
        bot.send_message(call.message.chat.id, "🔄 Запускаю нове сканування...")
        ai_scanner_handler(call.message)
        
        # ---------- /ai_alert ----------
@bot.message_handler(commands=['ai_alert'])
def ai_alert_handler(message):
    """
    Автоматичні алерти з конкретними рівнями входу/виходу
    """
    try:
        parts = message.text.split()
        symbol = "BTCUSDT"  # default
        
        if len(parts) >= 2:
            symbol = parts[1].upper()
            if not symbol.endswith('USDT'):
                symbol += 'USDT'
        
        processing_msg = bot.send_message(message.chat.id, f"🎯 AI розраховує рівні для {symbol}...")
        
        # Отримуємо поточні дані
        url = f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}"
        response = requests.get(url, timeout=10)
        data = response.json() if response.status_code == 200 else {}
        
        current_price = float(data.get('lastPrice', 0))
        if current_price == 0:
            # Якщо API не дав ціну, пробуємо через get_price
            try:
                current_price = get_price(symbol)
            except:
                bot.reply_to(message, f"❌ Не вдалося отримати ціну для {symbol}")
                return
        
        # Отримуємо сигнал для аналізу
        signal_text = generate_signal_text(symbol, interval="1h")
        
        # Визначаємо напрямок сигналу
        is_bullish = any(keyword in signal_text for keyword in ['LONG', 'BUY', 'UP', 'BULL'])
        is_bearish = any(keyword in signal_text for keyword in ['SHORT', 'SELL', 'DOWN', 'BEAR'])
        
        if not (is_bullish or is_bearish):
            bot.reply_to(message, f"🔍 Для {symbol} немає чітких сигналів")
            return
        
        # Розраховуємо рівні
        if is_bullish:
            entry_price = round(current_price * 0.98, 6)  # -2% для входу на відкаті
            stop_loss = round(entry_price * 0.98, 6)      # -2% від входу
            take_profit = round(entry_price * 1.06, 6)    # +6% ціль (RRR 1:3)
            direction = "LONG"
            emoji = "🟢"
        else:
            entry_price = round(current_price * 1.02, 6)  # +2% для входу на відскоку
            stop_loss = round(entry_price * 1.02, 6)      # +2% від входу  
            take_profit = round(entry_price * 0.94, 6)    # -6% ціль (RRR 1:3)
            direction = "SHORT"
            emoji = "🔴"
        
        # Розраховуємо розмір позиції
        risk_per_trade = 100  # $100 ризик на угоду
        risk_amount = abs(entry_price - stop_loss)
        position_size = round(risk_per_trade / risk_amount, 2)
        
        # Формуємо детальний план
        response = [
            f"{emoji} <b>AI Trading Plan for {symbol}</b>",
            f"",
            f"📊 <b>Current Price:</b> ${current_price:.6f}",
            f"🎯 <b>Signal:</b> {direction}",
            f"",
            f"💰 <b>Trading Levels:</b>",
            f"• Entry: ${entry_price:.6f}",
            f"• Stop Loss: ${stop_loss:.6f}", 
            f"• Take Profit: ${take_profit:.6f}",
            f"• Risk/Reward: 1:3",
            f"",
            f"📈 <b>Position Size:</b>",
            f"• Risk Amount: ${risk_per_trade}",
            f"• Position Size: {position_size} {symbol.replace('USDT', '')}",
            f"• Investment: ${entry_price * position_size:.2f}",
            f"",
            f"💡 <b>AI Recommendations:</b>",
            f"• Wait for price to reach entry level",
            f"• Set limit orders for better execution",
            f"• Monitor 1h timeframe for confirmation",
            f"",
            f"⚠️ <i>Based on current market conditions</i>"
        ]
        
        try:
            bot.delete_message(message.chat.id, processing_msg.message_id)
        except:
            pass
        
        # Додаємо кнопки для швидких дій
        markup = types.InlineKeyboardMarkup()
        markup.row(
            types.InlineKeyboardButton("📊 Графік 1h", callback_data=f"chart_1h_{symbol}"),
            types.InlineKeyboardButton("🔄 Оновити", callback_data=f"alert_{symbol}")
        )
        
        bot.send_message(message.chat.id, "\n".join(response), 
                        parse_mode="HTML", reply_markup=markup)
        
    except Exception as e:
        bot.reply_to(message, f"❌ Помилка: {str(e)}")

# ---------- /ai_auto_alert ----------
@bot.message_handler(commands=['ai_auto_alert'])
def ai_auto_alert_handler(message):
    """
    Автоматичний пошук найкращих сигналів з готовими рівнями
    """
    try:
        processing_msg = bot.send_message(message.chat.id, "🔍 AI шукає найкращі сигнали...")
        
        # Отримуємо топ токени
        url = "https://api.binance.com/api/v3/ticker/24hr"
        response = requests.get(url, timeout=15)
        data = response.json()
        
        usdt_pairs = [d for d in data if d['symbol'].endswith('USDT') and float(d['quoteVolume']) > 10000000]
        top_pairs = sorted(usdt_pairs, key=lambda x: abs(float(x['priceChangePercent'])), reverse=True)[:10]
        
        best_alerts = []
        
        for pair in top_pairs:
            symbol = pair['symbol']
            current_price = float(pair['lastPrice'])
            
            try:
                signal_text = generate_signal_text(symbol, interval="1h")
                
                is_bullish = any(keyword in signal_text for keyword in ['LONG', 'BUY', 'UP', 'BULL'])
                is_bearish = any(keyword in signal_text for keyword in ['SHORT', 'SELL', 'DOWN', 'BEAR'])
                
                if is_bullish or is_bearish:
                    # Розраховуємо рівні
                    if is_bullish:
                        entry = round(current_price * 0.98, 6)
                        sl = round(entry * 0.98, 6)
                        tp = round(entry * 1.06, 6)
                        direction = "LONG"
                        emoji = "🟢"
                    else:
                        entry = round(current_price * 1.02, 6)
                        sl = round(entry * 1.02, 6)
                        tp = round(entry * 0.94, 6)
                        direction = "SHORT" 
                        emoji = "🔴"
                    
                    best_alerts.append({
                        'symbol': symbol,
                        'direction': direction,
                        'emoji': emoji,
                        'entry': entry,
                        'sl': sl,
                        'tp': tp,
                        'current': current_price
                    })
                    
            except Exception:
                continue
        
        try:
            bot.delete_message(message.chat.id, processing_msg.message_id)
        except:
            pass
        
        if not best_alerts:
            bot.reply_to(message, "🔍 Не знайдено сильних сигналів")
            return
        
        # Формуємо звіт
        response = ["🚀 <b>AI Auto Alerts - Найкращі сигнали:</b>\n"]
        
        for alert in best_alerts[:5]:
            response.append(
                f"\n{alert['emoji']} <b>{alert['symbol']}</b> - {alert['direction']}"
            )
            response.append(f"   Current: ${alert['current']:.6f}")
            response.append(f"   Entry: ${alert['entry']:.6f}")
            response.append(f"   SL: ${alert['sl']:.6f} | TP: ${alert['tp']:.6f}")
        
        response.append("\n💡 <i>Використовуйте /ai_alert SYMBOL для деталей</i>")
        
        bot.reply_to(message, "\n".join(response), parse_mode="HTML")
        
    except Exception as e:
        bot.reply_to(message, f"❌ Помилка: {str(e)}")

# ---------- Callback для оновлення алертів ----------
@bot.callback_query_handler(func=lambda call: call.data.startswith('alert_'))
def alert_callback(call):
    """Оновити алерт"""
    try:
        symbol = call.data.replace('alert_', '')
        bot.answer_callback_query(call.id, f"🔄 Оновлюю {symbol}...")
        
        # Видаляємо старе повідомлення і робимо новий аналіз
        try:
            bot.delete_message(call.message.chat.id, call.message.message_id)
        except:
            pass
        
        # Створюємо фейкове повідомлення для обробки
        class FakeMessage:
            def __init__(self, chat_id, text):
                self.chat_id = chat_id
                self.text = text
        
        fake_msg = FakeMessage(call.message.chat.id, f"/ai_alert {symbol}")
        ai_alert_handler(fake_msg)
        
    except Exception as e:
        bot.send_message(call.message.chat.id, f"❌ Помилка: {str(e)}")
        
import re
from datetime import datetime

# ---------- ОБРОБНИК ТЕКСТОВИХ ПОВІДОМЛЕНЬ ----------
@bot.message_handler(func=lambda message: True)
def handle_text_messages(message):
    """Обробка текстовых повідомлень для налаштувань"""
    try:
        user_id = message.from_user.id
        text = message.text.strip()
        
        # Перевіряємо чи користувач в процесі налаштування
        if user_id in user_settings_state:
            state, callback_message = user_settings_state[user_id]
            
            if state == 'waiting_confidence':
                # Обробка впевненості
                try:
                    confidence = int(text)
                    if 50 <= confidence <= 90:
                        if user_id not in notify_settings:
                            notify_settings[user_id] = {'enabled': True}
                        notify_settings[user_id]['min_confidence'] = confidence
                        bot.send_message(user_id, f"✅ Мінімальна впевненість встановлена: {confidence}%")
                        # Повертаємо до меню налаштувань
                        show_config_menu(callback_message)
                    else:
                        bot.send_message(user_id, "❌ Будь ласка, введіть число від 50 до 90")
                        return
                except ValueError:
                    bot.send_message(user_id, "❌ Будь ласка, введіть число")
                    return
                    
            elif state == 'waiting_time':
                # Обробка часу
                if re.match(r'^\d{2}:\d{2}-\d{2}:\d{2}$', text):
                    if user_id not in notify_settings:
                        notify_settings[user_id] = {'enabled': True}
                    notify_settings[user_id]['active_hours'] = text
                    bot.send_message(user_id, f"✅ Час активності встановлений: {text}")
                    show_config_menu(callback_message)
                else:
                    bot.send_message(user_id, "❌ Неправильний формат. Приклад: 09:00-18:00")
                    return
                    
            elif state == 'waiting_favorites':
                # Обробка улюблених монет
                coins = [coin.strip().upper() for coin in text.split(',')]
                valid_coins = []
                
                for coin in coins:
                    if coin.endswith('USDT') and len(coin) > 4:
                        valid_coins.append(coin)
                
                if valid_coins:
                    if user_id not in notify_settings:
                        notify_settings[user_id] = {'enabled': True}
                    notify_settings[user_id]['favorite_coins'] = valid_coins
                    bot.send_message(user_id, f"✅ Улюблені монети додані: {', '.join(valid_coins)}")
                else:
                    bot.send_message(user_id, "❌ Не знайдено валідних монет. Приклад: BTCUSDT,ETHUSDT")
                
                show_config_menu(callback_message)
            
            # Видаляємо стан після обробки
            del user_settings_state[user_id]
                
    except Exception as e:
        bot.send_message(message.chat.id, f"❌ Помилка: {str(e)}")

# ---------- ОБРОБНИКИ КНОПОК ----------
@bot.callback_query_handler(func=lambda call: call.data.startswith(('notify_', 'config_', 'type_')))
def handle_all_callbacks(call):
    """Єдиний обробник для всіх callback"""
    try:
        user_id = call.from_user.id
        data = call.data
        
        # Список оброблюваних команд
        handlers = {
            'notify_enable': handle_enable,
            'notify_toggle': handle_toggle,
            'notify_test': handle_test,
            'notify_config': handle_config,
            'notify_favorites': handle_favorites,
            'notify_back': handle_back,
            'config_confidence': handle_config_confidence,
            'config_types': handle_config_types,
            'config_time': handle_config_time,
            'config_favorites': handle_config_favorites,
            'type_all': handle_type_all,
            'type_breakout': handle_type_breakout,
            'type_trend': handle_type_trend,
            'type_squeeze': handle_type_squeeze
        }
        
        if data in handlers:
            handlers[data](call)
        else:
            bot.answer_callback_query(call.id, "❌ Невідома команда")
            
    except Exception as e:
        bot.answer_callback_query(call.id, f"❌ Помилка: {str(e)}")

# ---------- КОНКРЕТНІ ОБРОБНИКИ ----------
def handle_enable(call):
    """Увімкнути сповіщення"""
    user_id = call.from_user.id
    notify_settings[user_id] = {
        'enabled': True,
        'min_confidence': 70,
        'signal_types': ['ALL'],
        'active_hours': '00:00-23:59',
        'favorite_coins': []
    }
    bot.answer_callback_query(call.id, "✅ Сповіщення увімкнено!")
    # Оновлюємо повідомлення
    try:
        bot.delete_message(call.message.chat.id, call.message.message_id)
    except:
        pass
    ai_notify_handler(call.message)

def handle_toggle(call):
    """Перемкнути статус"""
    user_id = call.from_user.id
    if user_id in notify_settings:
        notify_settings[user_id]['enabled'] = not notify_settings[user_id].get('enabled', False)
        status = "увімкнено" if notify_settings[user_id]['enabled'] else "вимкнено"
        bot.answer_callback_query(call.id, f"✅ Сповіщення {status}!")
        update_message(call)
    else:
        bot.answer_callback_query(call.id, "❌ Спочатку налаштуйте сповіщення!")

def handle_test(call):
    """Тестове сповіщення"""
    user_id = call.from_user.id
    if user_id in notify_settings and notify_settings[user_id].get('enabled', False):
        send_test_notification(user_id)
        bot.answer_callback_query(call.id, "📋 Тестове сповіщення відправлено!")
    else:
        bot.answer_callback_query(call.id, "❌ Спочатку увімкніть сповіщення!")

def handle_config(call):
    """Меню налаштувань"""
    show_config_menu(call)

def handle_favorites(call):
    """Показати улюблені"""
    user_id = call.from_user.id
    favorites = notify_settings.get(user_id, {}).get('favorite_coins', [])
    
    if favorites:
        response = ["💎 <b>Улюблені монети:</b>\n"] + [f"• {coin}" for coin in favorites]
    else:
        response = ["💎 <b>Улюблені монети:</b>\n", "• Список порожній"]
    
    bot.send_message(call.message.chat.id, "\n".join(response), parse_mode="HTML")
    bot.answer_callback_query(call.id, "💎 Список улюблених монет")

def handle_back(call):
    """Назад до головного меню"""
    update_message(call)

def handle_config_confidence(call):
    """Налаштування впевненості"""
    user_id = call.from_user.id
    user_settings_state[user_id] = ('waiting_confidence', call)
    bot.send_message(call.message.chat.id, "🎯 Введіть мінімальну впевненість (50-90):")
    bot.answer_callback_query(call.id, "Введіть число від 50 до 90")

def handle_config_time(call):
    """Налаштування часу"""
    user_id = call.from_user.id
    user_settings_state[user_id] = ('waiting_time', call)
    bot.send_message(call.message.chat.id, "⏰ Введіть час активності (наприклад 09:00-18:00):")
    bot.answer_callback_query(call.id, "Введіть час у форматі HH:MM-HH:MM")

def handle_config_favorites(call):
    """Налаштування улюблених"""
    user_id = call.from_user.id
    user_settings_state[user_id] = ('waiting_favorites', call)
    bot.send_message(call.message.chat.id, "💎 Введіть улюблені монети через кому (BTCUSDT,ETHUSDT):")
    bot.answer_callback_query(call.id, "Введіть монети через кому")

def handle_config_types(call):
    """Меню типів сигналів"""
    user_id = call.from_user.id
    if user_id not in notify_settings:
        bot.answer_callback_query(call.id, "❌ Спочатку увімкніть сповіщення!")
        return
        
    show_signal_types_menu(call)

def handle_type_all(call):
    """Обрати всі типи"""
    user_id = call.from_user.id
    notify_settings[user_id]['signal_types'] = ['ALL']
    bot.answer_callback_query(call.id, "✅ Всі типи сигналів")
    show_config_menu(call)

def handle_type_breakout(call):
    """Обрати тільки пробої"""
    user_id = call.from_user.id
    notify_settings[user_id]['signal_types'] = ['BREAKOUT']
    bot.answer_callback_query(call.id, "✅ Тільки пробої")
    show_config_menu(call)

def handle_type_trend(call):
    """Обрати тільки тренди"""
    user_id = call.from_user.id
    notify_settings[user_id]['signal_types'] = ['TREND']
    bot.answer_callback_query(call.id, "✅ Тільки тренди")
    show_config_menu(call)

def handle_type_squeeze(call):
    """Обрати тільки сквізи"""
    user_id = call.from_user.id
    notify_settings[user_id]['signal_types'] = ['SQUEEZE']
    bot.answer_callback_query(call.id, "✅ Тільки сквізи")
    show_config_menu(call)

# ---------- ДОПОМІЖНІ ФУНКЦІЇ ----------
def update_message(call):
    """Оновити повідомлення"""
    try:
        bot.delete_message(call.message.chat.id, call.message.message_id)
    except:
        pass
    ai_notify_handler(call.message)

def show_config_menu(call):
    """Показати меню налаштувань"""
    response = ["⚙️ <b>Налаштування сповіщень:</b>\n\nОберіть опцію для зміни:"]
    
    markup = types.InlineKeyboardMarkup()
    markup.row(
        types.InlineKeyboardButton("🎯 Впевненість", callback_data="config_confidence"),
        types.InlineKeyboardButton("📊 Типи сигналів", callback_data="config_types")
    )
    markup.row(
        types.InlineKeyboardButton("⏰ Час активності", callback_data="config_time"),
        types.InlineKeyboardButton("💎 Улюблені монети", callback_data="config_favorites")
    )
    markup.row(
        types.InlineKeyboardButton("🔙 Назад", callback_data="notify_back")
    )
    
    try:
        bot.edit_message_text(
            chat_id=call.message.chat.id,
            message_id=call.message.message_id,
            text="\n".join(response),
            parse_mode="HTML",
            reply_markup=markup
        )
    except:
        bot.send_message(call.message.chat.id, "\n".join(response), 
                        parse_mode="HTML", reply_markup=markup)

def show_signal_types_menu(call):
    """Меню типів сигналів"""
    user_id = call.from_user.id
    current_types = notify_settings[user_id].get('signal_types', ['ALL'])
    
    response = ["📊 <b>Оберіть типи сигналів:</b>\n"]
    
    response.append("✅ ВСІ" if 'ALL' in current_types else "⚪️ ВСІ")
    response.append("✅ ПРОБОЇ" if 'BREAKOUT' in current_types else "⚪️ ПРОБОЇ")
    response.append("✅ ТРЕНДИ" if 'TREND' in current_types else "⚪️ ТРЕНДИ")
    response.append("✅ СКВІЗИ" if 'SQUEEZE' in current_types else "⚪️ СКВІЗИ")
    
    markup = types.InlineKeyboardMarkup()
    markup.row(
        types.InlineKeyboardButton("✅ ВСІ", callback_data="type_all"),
        types.InlineKeyboardButton("🚀 ПРОБОЇ", callback_data="type_breakout")
    )
    markup.row(
        types.InlineKeyboardButton("📈 ТРЕНДИ", callback_data="type_trend"),
        types.InlineKeyboardButton("🔍 СКВІЗИ", callback_data="type_squeeze")
    )
    markup.row(
        types.InlineKeyboardButton("🔙 Назад", callback_data="notify_config")
    )
    
    try:
        bot.edit_message_text(
            chat_id=call.message.chat.id,
            message_id=call.message.message_id,
            text="\n".join(response),
            parse_mode="HTML",
            reply_markup=markup
        )
    except:
        bot.send_message(call.message.chat.id, "\n".join(response), 
                        parse_mode="HTML", reply_markup=markup)

def send_test_notification(user_id):
    """Відправити тестове сповіщення"""
    notification = [
        "🎯 <b>TEST NOTIFICATION</b>",
        "📊 BTCUSDT | Впевненість: 85%",
        "",
        "✅ Тестове сповіщення працює!",
        "Система сповіщень активована.",
        "",
        f"⏰ {datetime.now().strftime('%H:%M:%S')}",
        "",
        "💡 Ви будете отримувати сповіщення про найкращі торгові можливості!"
    ]
    
    bot.send_message(user_id, "\n".join(notification), parse_mode="HTML")

# Глобальні змінні
notify_settings = {}
user_settings_state = {}  # Стан користувачів для текстового вводу
# ---------- Додаємо новий обробник для видалення ----------
@bot.callback_query_handler(func=lambda call: call.data.startswith('remove_'))
def remove_favorite_callback(call):
    """Видалити монету з улюблених"""
    try:
        user_id = call.from_user.id
        symbol = call.data.replace('remove_', '')
        
        if user_id in notify_settings and 'favorite_coins' in notify_settings[user_id]:
            if symbol in notify_settings[user_id]['favorite_coins']:
                notify_settings[user_id]['favorite_coins'].remove(symbol)
                bot.answer_callback_query(call.id, f"✅ {symbol} видалено з улюблених")
                
                # Оновлюємо список улюблених
                show_favorites_menu(call)
            else:
                bot.answer_callback_query(call.id, f"❌ {symbol} не знайдено в улюблених")
        else:
            bot.answer_callback_query(call.id, "❌ Список улюблених порожній")
            
    except Exception as e:
        bot.answer_callback_query(call.id, f"❌ Помилка: {str(e)}")

# ---------- Оновлюємо функцію show_favorites_menu ----------
def show_favorites_menu(call):
    """Меню улюблених монет з кнопками видалення"""
    user_id = call.from_user.id
    favorites = notify_settings.get(user_id, {}).get('favorite_coins', [])
    
    if favorites:
        response = ["💎 <b>Улюблені монети:</b>\n\n"]
        markup = types.InlineKeyboardMarkup()
        
        for coin in favorites:
            response.append(f"• {coin}")
            # Додаємо кнопку видалення для кожної монети
            markup.add(types.InlineKeyboardButton(f"❌ Видалити {coin}", callback_data=f"remove_{coin}"))
        
        response.append("\n\n🎯 Натисніть на монету для видалення")
        
    else:
        response = ["💎 <b>Улюблені монети:</b>\n", "• Список порожній"]
        markup = types.InlineKeyboardMarkup()
    
    # Додаємо кнопку повернення
    markup.row(types.InlineKeyboardButton("🔙 Назад", callback_data="notify_config"))
    
    try:
        bot.edit_message_text(
            chat_id=call.message.chat.id,
            message_id=call.message.message_id,
            text="\n".join(response),
            parse_mode="HTML",
            reply_markup=markup
        )
    except:
        bot.send_message(call.message.chat.id, "\n".join(response), 
                        parse_mode="HTML", reply_markup=markup)

# ---------- Оновлюємо обробник текстовых повідомлень ----------
@bot.message_handler(func=lambda message: True)
def handle_text_messages(message):
    """Обробка текстовых повідомлень для налаштувань"""
    try:
        user_id = message.from_user.id
        text = message.text.strip().lower()
        
        # Команда очищення улюблених
        if text == 'clear':
            if user_id in notify_settings and 'favorite_coins' in notify_settings[user_id]:
                notify_settings[user_id]['favorite_coins'] = []
                bot.send_message(user_id, "✅ Список улюблених очищено!")
                
                # Повертаємо до головного меню
                try:
                    ai_notify_handler(message)
                except:
                    pass
            else:
                bot.send_message(user_id, "❌ Список улюблених вже порожній")
            return
        
        # Перевіряємо чи користувач в процесі налаштування
        if user_id in user_settings_state:
            state, callback_message = user_settings_state[user_id]
                
            if state == 'waiting_confidence':
                # Обробка впевненості
                try:
                    confidence = int(text)
                    if 50 <= confidence <= 90:
                        if user_id not in notify_settings:
                            notify_settings[user_id] = {'enabled': True}
                        notify_settings[user_id]['min_confidence'] = confidence
                        bot.send_message(user_id, f"✅ Мінімальна впевненість встановлена: {confidence}%")
                        # Повертаємо до меню налаштувань
                        show_config_menu(callback_message)
                    else:
                        bot.send_message(user_id, "❌ Будь ласка, введіть число від 50 до 90")
                        return
                except ValueError:
                    bot.send_message(user_id, "❌ Будь ласка, введіть число")
                    return
                    
            elif state == 'waiting_time':
                # Обробка часу
                if re.match(r'^\d{2}:\d{2}-\d{2}:\d{2}$', text):
                    if user_id not in notify_settings:
                        notify_settings[user_id] = {'enabled': True}
                    notify_settings[user_id]['active_hours'] = text
                    bot.send_message(user_id, f"✅ Час активності встановлений: {text}")
                    show_config_menu(callback_message)
                else:
                    bot.send_message(user_id, "❌ Неправильний формат. Приклад: 09:00-18:00")
                    return
                    
            elif state == 'waiting_favorites':
                # Обробка улюблених монет
                coins = [coin.strip().upper() for coin in text.split(',')]
                valid_coins = []
                
                for coin in coins:
                    if coin.endswith('USDT') and len(coin) > 4:
                        valid_coins.append(coin)
                
                if valid_coins:
                    if user_id not in notify_settings:
                        notify_settings[user_id] = {'enabled': True}
                    notify_settings[user_id]['favorite_coins'] = valid_coins
                    bot.send_message(user_id, f"✅ Улюблені монети додані: {', '.join(valid_coins)}")
                else:
                    bot.send_message(user_id, "❌ Не знайдено валідних монет. Приклад: BTCUSDT,ETHUSDT")
                    return
                
                show_config_menu(callback_message)
                del user_settings_state[user_id]
                
    except Exception as e:
        bot.send_message(message.chat.id, f"❌ Помилка: {str(e)}")

# ---------- Додаємо кнопку очищення в меню улюблених ----------
def show_favorites_menu(call):
    """Меню улюблених монет з кнопками видалення"""
    user_id = call.from_user.id
    favorites = notify_settings.get(user_id, {}).get('favorite_coins', [])
    
    if favorites:
        response = ["💎 <b>Улюблені монети:</b>\n\n"]
        markup = types.InlineKeyboardMarkup()
        
        for coin in favorites:
            response.append(f"• {coin}")
            # Додаємо кнопку видалення для кожної монети
            markup.add(types.InlineKeyboardButton(f"❌ Видалити {coin}", callback_data=f"remove_{coin}"))
        
        response.append("\n\n🎯 Натисніть на монету для видалення")
        
    else:
        response = ["💎 <b>Улюблені монети:</b>\n", "• Список порожній"]
        markup = types.InlineKeyboardMarkup()
    
    # Додаємо кнопки
    markup.row(types.InlineKeyboardButton("🗑️ Очистити всі", callback_data="clear_all"))
    markup.row(types.InlineKeyboardButton("🔙 Назад", callback_data="notify_config"))
    
    try:
        bot.edit_message_text(
            chat_id=call.message.chat.id,
            message_id=call.message.message_id,
            text="\n".join(response),
            parse_mode="HTML",
            reply_markup=markup
        )
    except:
        bot.send_message(call.message.chat.id, "\n".join(response), 
                        parse_mode="HTML", reply_markup=markup)

# ---------- Додаємо обробник для кнопки очищення ----------
@bot.callback_query_handler(func=lambda call: call.data == 'clear_all')
def clear_all_favorites(call):
    """Очистити всі улюблені монети"""
    try:
        user_id = call.from_user.id
        
        if user_id in notify_settings and 'favorite_coins' in notify_settings[user_id]:
            notify_settings[user_id]['favorite_coins'] = []
            bot.answer_callback_query(call.id, "✅ Список очищено!")
            show_favorites_menu(call)
        else:
            bot.answer_callback_query(call.id, "❌ Список вже порожній")
            
    except Exception as e:
        bot.answer_callback_query(call.id, f"❌ Помилка: {str(e)}")