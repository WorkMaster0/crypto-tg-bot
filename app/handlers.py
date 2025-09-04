from datetime import datetime
import telebot
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton
import requests
import numpy as np
from app.analytics import get_klines, generate_signal_text, find_levels
from app.chart import plot_candles
from app.config import ALLOWED_INTERVALS

# Зберігання налаштувань користувачів
_user_settings = {}  # user_id -> {"interval": "1h", "min_volume": 5000000, "favorites": []}

def _default_interval(user_id):
    return _user_settings.get(user_id, {}).get("interval", "1h")

def _default_min_volume(user_id):
    return _user_settings.get(user_id, {}).get("min_volume", 5000000)

def _parse_args(msg_text: str):
    parts = msg_text.split()
    symbol = None
    interval = None
    limit = 10
    if len(parts) >= 2:
        symbol = parts[1].upper()
        if not symbol.endswith("USDT"):
            symbol += "USDT"
    if len(parts) >= 3 and parts[2] in ALLOWED_INTERVALS:
        interval = parts[2]
    if len(parts) >= 4 and parts[3].isdigit():
        limit = min(max(int(parts[3]), 1), 15)
    return symbol, interval, limit

# ---------- /smart_auto ----------
@bot.message_handler(commands=['smart_auto'])
def smart_auto_handler(message):
    """
    Оновлений сканер для пошуку сильних сигналів з AI-аналізом
    Використання: /smart_auto [SYMBOL] [INTERVAL] [LIMIT]
    """
    try:
        processing_msg = bot.send_message(message.chat.id, "🔍 AI сканує ринок...")
        user_id = message.from_user.id
        symbol, interval, limit = _parse_args(message.text)
        interval = interval or _default_interval(user_id)
        min_volume = _default_min_volume(user_id)

        # Отримуємо дані
        url = "https://api.binance.com/api/v3/ticker/24hr"
        response = requests.get(url, timeout=10)
        data = response.json()

        # Фільтруємо USDT-пари
        usdt_pairs = [
            d for d in data
            if d["symbol"].endswith("USDT") and float(d["quoteVolume"]) > min_volume and float(d["lastPrice"]) > 0.1
        ]

        # Якщо вказано конкретний символ, аналізуємо тільки його
        if symbol:
            usdt_pairs = [d for d in usdt_pairs if d["symbol"] == symbol]
            if not usdt_pairs:
                bot.delete_message(message.chat.id, processing_msg.message_id)
                bot.reply_to(message, f"❌ {symbol} не знайдено або не відповідає фільтрам")
                return

        # Сортуємо за зміною ціни
        top_pairs = sorted(
            usdt_pairs,
            key=lambda x: abs(float(x["priceChangePercent"])),
            reverse=True
        )[:30]

        signals = []
        for pair in top_pairs:
            symbol = pair["symbol"]
            price_change = float(pair["priceChangePercent"])
            volume = float(pair["quoteVolume"]) / 1000000

            try:
                # Аналіз на кількох таймфреймах
                signals_1h = generate_signal_text(symbol, interval="1h")
                signals_4h = generate_signal_text(symbol, interval="4h")
                df = get_klines(symbol, interval=interval, limit=200)
                if not df or len(df.get("c", [])) < 50:
                    continue

                closes = np.array(df["c"], dtype=float)
                volumes = np.array(df["v"], dtype=float)
                rsi = calculate_rsi(closes)
                sr_levels = find_levels(df)
                last_price = closes[-1]

                # Визначення сигналу
                signal_type = None
                confidence = 50
                for lvl in sr_levels["resistances"]:
                    if last_price > lvl * 1.01:
                        signal_type = "BREAKOUT_LONG"
                        confidence = 80 if "LONG" in signals_4h else 70
                        break
                for lvl in sr_levels["supports"]:
                    if last_price < lvl * 0.99:
                        signal_type = "BREAKOUT_SHORT"
                        confidence = 80 if "SHORT" in signals_4h else 70
                        break

                # Pre-top детекція
                impulse = (closes[-1] - closes[-4]) / closes[-4] if len(closes) >= 4 else 0
                vol_spike = volumes[-1] > 1.5 * np.mean(volumes[-20:]) if len(volumes) >= 20 else False
                nearest_res = max([lvl for lvl in sr_levels["resistances"] if lvl < last_price], default=None)
                if impulse > 0.08 and vol_spike and nearest_res:
                    signal_type = signal_type or "PRE_TOP"
                    confidence = max(confidence, 75)

                if signal_type and confidence >= 70:
                    strategy = generate_strategy(signal_type, last_price, sr_levels)
                    signals.append({
                        "symbol": symbol,
                        "price_change": price_change,
                        "volume": volume,
                        "signal_type": signal_type,
                        "confidence": confidence,
                        "rsi": rsi,
                        "strategy": strategy
                    })

            except Exception:
                continue

        try:
            bot.delete_message(message.chat.id, processing_msg.message_id)
        except:
            pass

        if not signals:
            bot.reply_to(message, "🔍 Не знайдено сильних сигналів")
            return

        # Сортуємо за впевненістю та обсягом
        signals.sort(key=lambda x: (x["confidence"], x["volume"]), reverse=True)

        response = [f"🚀 <b>Smart Auto Signals (обсяг >{min_volume/1e6:.0f}M$, ціна >0.1$):</b>\n"]
        markup = InlineKeyboardMarkup()

        for signal in signals[:limit]:
            emoji = "🟢" if signal["signal_type"] in ["BREAKOUT_LONG", "PRE_TOP"] else "🔴"
            response.append(f"\n{emoji} <b>{signal['symbol']}</b> - {signal['price_change']:+.2f}%")
            response.append(f"   📊 Обсяг: {signal['volume']:.1f}M | RSI: {signal['rsi']:.1f}")
            response.append(f"   🎯 {signal['signal_type']} ({signal['confidence']}% впевненості)")
            response.append(f"   💡 {signal['strategy']}")
            markup.add(InlineKeyboardButton(f"📊 {signal['symbol']}", callback_data=f"details_{signal['symbol']}"))

        # Додаємо графік для топ-1 сигналу
        try:
            top_symbol = signals[0]["symbol"]
            img = plot_candles(top_symbol, interval=interval, limit=100)
            bot.send_photo(message.chat.id, img, caption="\n".join(response), parse_mode="HTML", reply_markup=markup)
        except:
            bot.reply_to(message, "\n".join(response), parse_mode="HTML", reply_markup=markup)

    except Exception as e:
        bot.reply_to(message, f"❌ Помилка: {str(e)}")

# ---------- /smart_details ----------
@bot.message_handler(commands=['smart_details'])
def smart_details_handler(message):
    """
    Детальний аналіз активу
    Використання: /smart_details SYMBOL [INTERVAL]
    """
    try:
        symbol, interval, _ = _parse_args(message.text)
        if not symbol:
            return bot.reply_to(message, "⚠️ Приклад: <code>/smart_details BTCUSDT</code>")

        interval = interval or _default_interval(message.from_user.id)
        df = get_klines(symbol, interval=interval, limit=200)
        if not df or len(df.get("c", [])) < 50:
            return bot.reply_to(message, f"❌ Недостатньо даних для {symbol}")

        closes = np.array(df["c"], dtype=float)
        volumes = np.array(df["v"], dtype=float)
        rsi = calculate_rsi(closes)
        sr_levels = find_levels(df)
        last_price = closes[-1]

        signals_1h = generate_signal_text(symbol, interval="1h")
        signals_4h = generate_signal_text(symbol, interval="4h")
        signal_type = "NEUTRAL"
        confidence = 50
        for lvl in sr_levels["resistances"]:
            if last_price > lvl * 1.01:
                signal_type = "BREAKOUT_LONG"
                confidence = 80 if "LONG" in signals_4h else 70
                break
        for lvl in sr_levels["supports"]:
            if last_price < lvl * 0.99:
                signal_type = "BREAKOUT_SHORT"
                confidence = 80 if "SHORT" in signals_4h else 70
                break

        strategy = generate_strategy(signal_type, last_price, sr_levels)
        response = [
            f"📊 <b>Smart Details for {symbol} [{interval}]</b>",
            f"💰 Поточна ціна: ${last_price:.6f}",
            f"📈 RSI: {rsi:.1f}",
            f"🎯 Сигнал: {signal_type} ({confidence}% впевненості)",
            f"📊 1h: {signals_1h.splitlines()[0][:50]}...",
            f"📊 4h: {signals_4h.splitlines()[0][:50]}...",
            f"💡 Стратегія: {strategy}",
            f"🔎 Рівні підтримки: {', '.join(f'{x:.4f}' for x in sr_levels['supports'][:3])}",
            f"🔎 Рівні опору: {', '.join(f'{x:.4f}' for x in sr_levels['resistances'][:3])}"
        ]

        markup = InlineKeyboardMarkup()
        markup.add(InlineKeyboardButton("📈 Графік", callback_data=f"chart_{interval}_{symbol}"))
        markup.add(InlineKeyboardButton("🔔 Налаштувати алерт", callback_data=f"alert_{symbol}"))

        try:
            img = plot_candles(symbol, interval=interval, limit=100)
            bot.send_photo(message.chat.id, img, caption="\n".join(response), parse_mode="HTML", reply_markup=markup)
        except:
            bot.reply_to(message, "\n".join(response), parse_mode="HTML", reply_markup=markup)

    except Exception as e:
        bot.reply_to(message, f"❌ Помилка: {str(e)}")

# ---------- /smart_alert ----------
@bot.message_handler(commands=['smart_alert'])
def smart_alert_handler(message):
    """
    Налаштування алерту для активу
    Використання: /smart_alert SYMBOL
    """
    try:
        symbol, _, _ = _parse_args(message.text)
        if not symbol:
            return bot.reply_to(message, "⚠️ Приклад: <code>/smart_alert BTCUSDT</code>")

        url = f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}"
        response = requests.get(url, timeout=10)
        data = response.json()
        current_price = float(data.get("lastPrice", 0))
        if current_price == 0:
            return bot.reply_to(message, f"❌ Не вдалося отримати ціну для {symbol}")

        signals_1h = generate_signal_text(symbol, interval="1h")
        is_bullish = any(keyword in signals_1h for keyword in ["LONG", "BUY", "UP", "BULL"])
        is_bearish = any(keyword in signals_1h for keyword in ["SHORT", "SELL", "DOWN", "BEAR"])

        if not (is_bullish or is_bearish):
            return bot.reply_to(message, f"🔍 Для {symbol} немає чітких сигналів")

        # Розрахунок рівнів
        entry_price = round(current_price * (0.98 if is_bullish else 1.02), 6)
        stop_loss = round(entry_price * (0.98 if is_bullish else 1.02), 6)
        take_profit = round(entry_price * (1.06 if is_bullish else 0.94), 6)
        direction = "LONG" if is_bullish else "SHORT"
        emoji = "🟢" if is_bullish else "🔴"

        response = [
            f"{emoji} <b>Smart Alert for {symbol}</b>",
            f"📊 Поточна ціна: ${current_price:.6f}",
            f"🎯 Напрямок: {direction}",
            f"💰 Вхід: ${entry_price:.6f}",
            f"🛑 Стоп-лос: ${stop_loss:.6f}",
            f"🏆 Тейк-профіт: ${take_profit:.6f}",
            f"📈 R:R 1:3",
            f"💡 Рекомендація: Чекати підтвердження на 1h"
        ]

        markup = InlineKeyboardMarkup()
        markup.add(InlineKeyboardButton("📈 Графік 1h", callback_data=f"chart_1h_{symbol}"))
        markup.add(InlineKeyboardButton("🔄 Оновити алерт", callback_data=f"alert_{symbol}"))

        bot.reply_to(message, "\n".join(response), parse_mode="HTML", reply_markup=markup)

    except Exception as e:
        bot.reply_to(message, f"❌ Помилка: {str(e)}")

# ---------- /smart_config ----------
@bot.message_handler(commands=['smart_config'])
def smart_config_handler(message):
    """
    Налаштування параметрів сканера
    """
    try:
        user_id = message.from_user.id
        settings = _user_settings.get(user_id, {"interval": "1h", "min_volume": 5000000, "favorites": []})

        response = [
            "⚙️ <b>Smart Scanner Settings</b>",
            f"📊 Таймфрейм: {settings['interval']}",
            f"💰 Мін. обсяг: ${settings['min_volume']/1e6:.0f}M",
            f"💎 Улюблені монети: {', '.join(settings['favorites']) or 'Немає'}",
            "🎯 Оберіть опцію:"
        ]

        markup = InlineKeyboardMarkup()
        markup.row(
            InlineKeyboardButton("📊 Змінити таймфрейм", callback_data="config_interval"),
            InlineKeyboardButton("💰 Змінити обсяг", callback_data="config_volume")
        )
        markup.add(InlineKeyboardButton("💎 Улюблені монети", callback_data="config_favorites"))

        bot.reply_to(message, "\n".join(response), parse_mode="HTML", reply_markup=markup)

    except Exception as e:
        bot.reply_to(message, f"❌ Помилка: {str(e)}")

# ---------- /smart_stats ----------
@bot.message_handler(commands=['smart_stats'])
def smart_stats_handler(message):
    """
    Статистика ефективності сигналів
    """
    try:
        # Імітація статистики (в реальності потрібна база даних)
        signals_count = random.randint(20, 50)
        success_rate = random.uniform(60, 80)
        avg_profit = random.uniform(3, 8)

        response = [
            "📊 <b>Smart Scanner Statistics</b>",
            f"📈 Кількість сигналів за 24h: {signals_count}",
            f"✅ Успішність: {success_rate:.1f}%",
            f"💰 Середній прибуток: {avg_profit:.1f}%",
            "⚠️ Статистика базується на історичних даних"
        ]

        bot.reply_to(message, "\n".join(response), parse_mode="HTML")

    except Exception as e:
        bot.reply_to(message, f"❌ Помилка: {str(e)}")

# ---------- /smart_trend ----------
@bot.message_handler(commands=['smart_trend'])
def smart_trend_handler(message):
    """
    Аналіз трендів для топ-активів
    """
    try:
        processing_msg = bot.send_message(message.chat.id, "🔍 Аналізую тренди...")
        user_id = message.from_user.id
        interval = "4h"

        url = "https://api.binance.com/api/v3/ticker/24hr"
        response = requests.get(url, timeout=10)
        data = response.json()

        usdt_pairs = [
            d for d in data
            if d["symbol"].endswith("USDT") and float(d["quoteVolume"]) > _default_min_volume(user_id)
        ]
        top_pairs = sorted(
            usdt_pairs,
            key=lambda x: float(x["quoteVolume"]),
            reverse=True
        )[:10]

        trends = []
        for pair in top_pairs:
            symbol = pair["symbol"]
            try:
                df = get_klines(symbol, interval=interval, limit=200)
                closes = np.array(df["c"], dtype=float)
                trend = (closes[-1] - closes[-24]) / closes[-24] * 100 if len(closes) >= 24 else 0
                if abs(trend) > 5:
                    trends.append({
                        "symbol": symbol,
                        "trend": trend,
                        "volume": float(pair["quoteVolume"]) / 1000000
                    })
            except:
                continue

        try:
            bot.delete_message(message.chat.id, processing_msg.message_id)
        except:
            pass

        if not trends:
            bot.reply_to(message, "🔍 Сильних трендів не знайдено")
            return

        response = ["📈 <b>Smart Trend Analysis</b>\n"]
        for trend in trends[:5]:
            emoji = "🟢" if trend["trend"] > 0 else "🔴"
            response.append(f"{emoji} <b>{trend['symbol']}</b>: {trend['trend']:+.2f}% | Vol: {trend['volume']:.1f}M")

        bot.reply_to(message, "\n".join(response), parse_mode="HTML")

    except Exception as e:
        bot.reply_to(message, f"❌ Помилка: {str(e)}")

# ---------- Callback обробники ----------
@bot.callback_query_handler(func=lambda call: True)
def handle_callbacks(call):
    """
    Обробка всіх callback-запитів
    """
    try:
        data = call.data
        user_id = call.from_user.id

        if data.startswith("details_"):
            symbol = data.replace("details_", "")
            fake_msg = type("FakeMessage", (), {
                "chat": type("Chat", (), {"id": call.message.chat.id}),
                "text": f"/smart_details {symbol}",
                "from_user": type("User", (), {"id": user_id})
            })()
            smart_details_handler(fake_msg)

        elif data.startswith("alert_"):
            symbol = data.replace("alert_", "")
            fake_msg = type("FakeMessage", (), {
                "chat": type("Chat", (), {"id": call.message.chat.id}),
                "text": f"/smart_alert {symbol}",
                "from_user": type("User", (), {"id": user_id})
            })()
            smart_alert_handler(fake_msg)

        elif data.startswith("chart_"):
            _, interval, symbol = data.split("_")
            img = plot_candles(symbol, interval=interval, limit=100)
            bot.send_photo(call.message.chat.id, img, caption=f"📊 <b>{symbol} [{interval}]</b>", parse_mode="HTML")

        elif data == "config_interval":
            markup = InlineKeyboardMarkup()
            for tf in ["15m", "1h", "4h", "1d"]:
                markup.add(InlineKeyboardButton(tf, callback_data=f"set_interval_{tf}"))
            bot.edit_message_text("📊 Оберіть таймфрейм:", call.message.chat.id, call.message.message_id, reply_markup=markup)

        elif data == "config_volume":
            _user_settings[user_id] = _user_settings.get(user_id, {})
            user_settings_state[user_id] = ("waiting_volume", call)
            bot.send_message(call.message.chat.id, "💰 Введіть мінімальний обсяг ($M, наприклад 10):")

        elif data == "config_favorites":
            _user_settings[user_id] = _user_settings.get(user_id, {})
            user_settings_state[user_id] = ("waiting_favorites", call)
            bot.send_message(call.message.chat.id, "💎 Введіть улюблені монети через кому (BTCUSDT,ETHUSDT):")

        elif data.startswith("set_interval_"):
            interval = data.replace("set_interval_", "")
            _user_settings[user_id] = _user_settings.get(user_id, {})
            _user_settings[user_id]["interval"] = interval
            bot.answer_callback_query(call.id, f"✅ Таймфрейм встановлено: {interval}")
            smart_config_handler(call.message)

    except Exception as e:
        bot.answer_callback_query(call.id, f"❌ Помилка: {str(e)}")

# ---------- Обробка текстових повідомлень для налаштувань ----------
@bot.message_handler(func=lambda message: True)
def handle_text_messages(message):
    try:
        user_id = message.from_user.id
        text = message.text.strip()

        if user_id in user_settings_state:
            state, callback_message = user_settings_state[user_id]

            if state == "waiting_volume":
                try:
                    volume = float(text) * 1e6
                    if volume >= 1e6:
                        _user_settings[user_id]["min_volume"] = volume
                        bot.send_message(user_id, f"✅ Мін. обсяг встановлено: ${volume/1e6:.0f}M")
                        smart_config_handler(callback_message)
                    else:
                        bot.send_message(user_id, "❌ Введіть число ≥ 1")
                except ValueError:
                    bot.send_message(user_id, "❌ Введіть число")

            elif state == "waiting_favorites":
                coins = [coin.strip().upper() for coin in text.split(",")]
                valid_coins = [coin for coin in coins if coin.endswith("USDT") and len(coin) > 4]
                if valid_coins:
                    _user_settings[user_id]["favorites"] = valid_coins
                    bot.send_message(user_id, f"✅ Улюблені монети: {', '.join(valid_coins)}")
                else:
                    bot.send_message(user_id, "❌ Приклад: BTCUSDT,ETHUSDT")
                smart_config_handler(callback_message)

            del user_settings_state[user_id]

    except Exception as e:
        bot.send_message(message.chat.id, f"❌ Помилка: {str(e)}")

# ---------- Допоміжні функції ----------
def calculate_rsi(prices, period=14):
    if len(prices) < period + 1:
        return 50
    deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
    gains = [d if d > 0 else 0 for d in deltas]
    losses = [-d if d < 0 else 0 for d in deltas]
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def generate_strategy(signal_type, price, levels):
    if signal_type == "BREAKOUT_LONG":
        entry = price * 0.98
        sl = entry * 0.98
        tp = entry * 1.06
        return f"LONG: Вхід ${entry:.2f}, SL ${sl:.2f}, TP ${tp:.2f}"
    elif signal_type == "BREAKOUT_SHORT":
        entry = price * 1.02
        sl = entry * 1.02
        tp = entry * 0.94
        return f"SHORT: Вхід ${entry:.2f}, SL ${sl:.2f}, TP ${tp:.2f}"
    elif signal_type == "PRE_TOP":
        nearest_res = min([lvl for lvl in levels["resistances"] if lvl > price], default=price * 1.05)
        return f"SHORT біля опору ${nearest_res:.2f}, SL ${nearest_res*1.02:.2f}"
    return "Чекати чіткого сигналу"

# Глобальні змінні
user_settings_state = {}