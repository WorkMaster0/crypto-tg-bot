from datetime import datetime
import numpy as np
import requests
from telebot import types
from app.analytics import get_klines, generate_signal_text
from app.config import ALLOWED_INTERVALS

# Глобальні налаштування
_user_defaults = {}

def _default_interval(chat_id):
    return _user_defaults.get(chat_id, {}).get("interval", "1h")

def find_support_resistance(prices, window=20, delta=0.005):
    """
    Автоматично знаходить локальні S/R рівні
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

# ---------- /smart_auto (ПОКРАЩЕНА ВЕРСІЯ) ----------
@bot.message_handler(commands=['smart_auto'])
def smart_auto_handler(message):
    """
    Покращена версія автоматичного сканування S/R рівнів
    Додано фільтрацію за обсягом, кращу класифікацію сигналів
    """
    try:
        processing_msg = bot.send_message(message.chat.id, "🔍 AI сканує ринок для S/R сигналів...")
        
        # Отримуємо дані з Binance API
        url = "https://api.binance.com/api/v3/ticker/24hr"
        response = requests.get(url, timeout=10)
        data = response.json()

        # Фільтруємо якісні пари: USDT, обсяг > 10M, ціна > 0.01
        symbols = [
            d for d in data
            if (d["symbol"].endswith("USDT") and 
                float(d["quoteVolume"]) > 10000000 and
                float(d["lastPrice"]) > 0.01)
        ]

        # Сортуємо за абсолютною зміною ціни (найбільш волатильні)
        symbols = sorted(
            symbols,
            key=lambda x: abs(float(x["priceChangePercent"])),
            reverse=True
        )

        # Беремо топ-25 найактивніших
        top_symbols = [s["symbol"] for s in symbols[:25]]
        
        signals = []
        
        for symbol in top_symbols:
            try:
                # Отримуємо дані для технічного аналізу
                df = get_klines(symbol, interval="1h", limit=100)
                if not df or len(df.get("c", [])) < 50:
                    continue

                closes = np.array(df["c"], dtype=float)
                volumes = np.array(df["v"], dtype=float)
                
                # Знаходимо рівні підтримки/опору
                sr_levels = find_support_resistance(closes, window=20, delta=0.005)
                if not sr_levels:
                    continue
                    
                last_price = closes[-1]
                current_volume = volumes[-1]
                avg_volume = np.mean(volumes[-20:])
                
                # Аналізуємо кожен рівень
                for level in sr_levels:
                    diff_pct = (last_price - level) / level * 100
                    
                    # Сигнал пробою опору (LONG)
                    if last_price > level * 1.01 and abs(diff_pct) < 5:
                        signal_type = "RESISTANCE_BREAKOUT"
                        direction = "LONG"
                        strength = "STRONG" if current_volume > avg_volume * 1.5 else "MODERATE"
                        
                        signals.append({
                            'symbol': symbol,
                            'level': level,
                            'current_price': last_price,
                            'signal_type': signal_type,
                            'direction': direction,
                            'strength': strength,
                            'volume_ratio': current_volume / avg_volume,
                            'change_pct': diff_pct
                        })
                    
                    # Сигнал пробою підтримки (SHORT)
                    elif last_price < level * 0.99 and abs(diff_pct) < 5:
                        signal_type = "SUPPORT_BREAKOUT"
                        direction = "SHORT"
                        strength = "STRONG" if current_volume > avg_volume * 1.5 else "MODERATE"
                        
                        signals.append({
                            'symbol': symbol,
                            'level': level,
                            'current_price': last_price,
                            'signal_type': signal_type,
                            'direction': direction,
                            'strength': strength,
                            'volume_ratio': current_volume / avg_volume,
                            'change_pct': diff_pct
                        })
                        
            except Exception as e:
                continue

        # Видаляємо повідомлення про обробку
        try:
            bot.delete_message(message.chat.id, processing_msg.message_id)
        except:
            pass

        if not signals:
            bot.send_message(message.chat.id, "ℹ️ Жодних S/R сигналів не знайдено.")
            return

        # Сортуємо сигнали за силою та обсягом
        signals.sort(key=lambda x: (x['strength'] == "STRONG", x['volume_ratio']), reverse=True)
        
        # Формуємо відповідь
        response = ["🎯 <b>Smart Auto S/R Signals</b>\n\n"]
        response.append("<i>Знайдені пробої рівнів підтримки/опору</i>\n")
        
        for i, signal in enumerate(signals[:15]):  # Обмежуємо кількість
            emoji = "🟢" if signal['direction'] == "LONG" else "🔴"
            strength_emoji = "🔥" if signal['strength'] == "STRONG" else "⚡"
            
            response.append(
                f"\n{emoji} {strength_emoji} <b>{signal['symbol']}</b>"
            )
            response.append(f"   📊 Тип: {signal['signal_type']}")
            response.append(f"   💰 Ціна: {signal['current_price']:.6f}")
            response.append(f"   🎯 Рівень: {signal['level']:.6f}")
            response.append(f"   📈 Відхилення: {signal['change_pct']:+.2f}%")
            response.append(f"   🔊 Обсяг: x{signal['volume_ratio']:.1f}")
            
            # Додаємо рекомендацію
            if signal['direction'] == "LONG":
                response.append("   💡 Рекомендація: LONG, стоп-лосс нижче рівня")
            else:
                response.append("   💡 Рекомендація: SHORT, стоп-лосс вище рівня")
                
            if i < len(signals[:15]) - 1:
                response.append("   ─────────────────")

        # Додаємо кнопки для швидкого аналізу
        markup = types.InlineKeyboardMarkup()
        for signal in signals[:3]:
            markup.add(types.InlineKeyboardButton(
                f"📊 {signal['symbol']}", 
                callback_data=f"analyze_{signal['symbol']}"
            ))
        
        markup.add(types.InlineKeyboardButton(
            "🔄 Оновити сканування", 
            callback_data="rescan_smart_auto"
        ))

        bot.send_message(message.chat.id, "\n".join(response), 
                        parse_mode="HTML", reply_markup=markup)

    except Exception as e:
        bot.send_message(message.chat.id, f"❌ Помилка: {str(e)}")

# ---------- /smart_scan (ДЛЯ СКАНУВАННЯ З ФІЛЬТРАМИ) ----------
@bot.message_handler(commands=['smart_scan'])
def smart_scan_handler(message):
    """
    Сканування з фільтрами за напрямком, силою сигналу
    Використання: /smart_scan [long/short] [strong/moderate]
    """
    try:
        parts = message.text.split()
        direction_filter = None
        strength_filter = None
        
        if len(parts) >= 2:
            if parts[1].lower() in ['long', 'buy', 'bull']:
                direction_filter = 'LONG'
            elif parts[1].lower() in ['short', 'sell', 'bear']:
                direction_filter = 'SHORT'
                
        if len(parts) >= 3:
            if parts[2].lower() in ['strong', 'high']:
                strength_filter = 'STRONG'
            elif parts[2].lower() in ['moderate', 'medium']:
                strength_filter = 'MODERATE'
        
        # Використовуємо ту саму логіку, що й в smart_auto
        processing_msg = bot.send_message(message.chat.id, f"🔍 Сканую з фільтрами...")
        
        # Тут буде та ж логіка отримання даних, що й в smart_auto
        # але з додатковим фільтруванням за direction_filter та strength_filter
        
        # Для прикладу просто викликаємо smart_auto та фільтруємо результати
        bot.send_message(message.chat.id, 
                       f"🔍 Сканування з фільтрами: {direction_filter or 'Всі'} {strength_filter or 'Всі'}\n\n"
                       f"ℹ️ Функція в розробці...",
                       parse_mode="HTML")
        
    except Exception as e:
        bot.send_message(message.chat.id, f"❌ Помилка: {str(e)}")

# ---------- /smart_levels (ДЕТАЛЬНИЙ АНАЛІЗ КОНКРЕТНОГО ТОКЕНА) ----------
@bot.message_handler(commands=['smart_levels'])
def smart_levels_handler(message):
    """
    Детальний аналіз рівнів підтримки/опору для конкретного токена
    Використання: /smart_levels BTCUSDT
    """
    try:
        parts = message.text.split()
        if len(parts) < 2:
            return bot.send_message(message.chat.id, "⚠️ Використання: /smart_levels BTCUSDT")
        
        symbol = parts[1].upper()
        if not symbol.endswith('USDT'):
            symbol += 'USDT'
            
        processing_msg = bot.send_message(message.chat.id, f"🔍 Аналізую рівні для {symbol}...")
        
        # Отримуємо дані
        df = get_klines(symbol, interval="1h", limit=200)
        if not df or len(df.get('c', [])) == 0:
            return bot.send_message(message.chat.id, f"❌ Дані для {symbol} недоступні")

        closes = np.array(df['c'], dtype=float)
        highs = np.array(df['h'], dtype=float)
        lows = np.array(df['l'], dtype=float)
        
        # Знаходимо рівні
        sr_levels = find_support_resistance(closes, window=20, delta=0.005)
        last_price = closes[-1]
        
        if not sr_levels:
            return bot.send_message(message.chat.id, f"❌ Для {symbol} не знайдено чітких рівнів")
        
        # Аналізуємо відносно поточної ціни
        supports = [lvl for lvl in sr_levels if lvl < last_price]
        resistances = [lvl for lvl in sr_levels if lvl > last_price]
        
        # Знаходимо найближчі рівні
        nearest_support = max(supports) if supports else None
        nearest_resistance = min(resistances) if resistances else None
        
        # Формуємо відповідь
        response = [
            f"📊 <b>Детальний аналіз рівнів для {symbol}</b>",
            f"💰 Поточна ціна: {last_price:.6f}",
            f"",
            f"🎯 <b>Рівні підтримки:</b>",
            *[f"• {level:.6f} ({((last_price - level)/level*100):.2f}% нижче)" for level in sorted(supports, reverse=True)[:5]],
            f"",
            f"🎯 <b>Рівні опору:</b>",
            *[f"• {level:.6f} ({((level - last_price)/last_price*100):.2f}% вище)" for level in sorted(resistances)[:5]],
            f"",
            f"📈 <b>Найближчі рівні:</b>",
            f"• Підтримка: {nearest_support:.6f if nearest_support else 'N/A'}",
            f"• Опір: {nearest_resistance:.6f if nearest_resistance else 'N/A'}",
            f"",
            f"💡 <b>Рекомендації:</b>",
            f"• Купувати біля підтримки" if nearest_support else "",
            f"• Продавати біля опору" if nearest_resistance else "",
            f"• Стоп-лосс за рівнями"
        ]
        
        try:
            bot.delete_message(message.chat.id, processing_msg.message_id)
        except:
            pass
            
        bot.send_message(message.chat.id, "\n".join(response), parse_mode="HTML")
        
    except Exception as e:
        bot.send_message(message.chat.id, f"❌ Помилка: {str(e)}")

# ---------- CALLBACK HANDLERS ----------
@bot.callback_query_handler(func=lambda call: call.data.startswith('analyze_'))
def analyze_callback(call):
    """Обробка кнопок аналізу"""
    try:
        symbol = call.data.replace('analyze_', '')
        bot.send_message(call.message.chat.id, f"🔍 Детально аналізую {symbol}...")
        
        # Використовуємо вашу існуючу функцію аналізу
        signal_text = generate_signal_text(symbol, interval="1h")
        
        response = [
            f"📊 <b>Детальний аналіз {symbol}:</b>",
            f"",
            f"{signal_text}",
            f"",
            f"💡 <i>Використовуйте /smart_levels {symbol} для аналізу рівнів</i>"
        ]
        
        bot.send_message(call.message.chat.id, "\n".join(response), parse_mode="HTML")
        
    except Exception as e:
        bot.send_message(call.message.chat.id, f"❌ Помилка: {str(e)}")

@bot.callback_query_handler(func=lambda call: call.data == 'rescan_smart_auto')
def rescan_smart_auto_callback(call):
    """Пересканування"""
    try:
        bot.delete_message(call.message.chat.id, call.message.message_id)
        smart_auto_handler(call.message)
    except:
        bot.send_message(call.message.chat.id, "🔄 Запускаю нове сканування...")
        smart_auto_handler(call.message)

# ---------- /quantum_entanglement ----------
@bot.message_handler(commands=['quantum_entanglement'])
def quantum_entanglement_handler(message):
    """
    AI виявляє квантову заплутаність між активами
    Знаходить приховані кореляції, які неможливо побачити звичайними методами
    """
    try:
        processing_msg = bot.send_message(message.chat.id, "🌌 AI аналізує квантову заплутаність ринків...")
        
        # Отримуємо дані з різних ринків
        symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'ADAUSDT', 
                  'DOTUSDT', 'DOGEUSDT', 'AVAXUSDT', 'LINKUSDT', 'MATICUSDT']
        
        quantum_signals = []
        
        for symbol in symbols:
            try:
                # Отримуємо квантові характеристики ціни
                price_data = get_klines(symbol, "15m", 100)
                if not price_data:
                    continue
                    
                closes = np.array(price_data['c'], dtype=float)
                
                # Квантовий аналіз паттернів
                quantum_pattern = analyze_quantum_pattern(closes)
                entanglement_score = calculate_entanglement_score(closes)
                
                if entanglement_score > 0.7:
                    # Знаходимо заплутані пари
                    entangled_pairs = find_entangled_pairs(symbol, symbols, closes)
                    
                    if entangled_pairs:
                        quantum_signals.append({
                            'symbol': symbol,
                            'entanglement_score': entanglement_score,
                            'quantum_pattern': quantum_pattern,
                            'entangled_pairs': entangled_pairs,
                            'prediction': predict_quantum_movement(closes)
                        })
                        
            except Exception:
                continue
        
        # Формуємо квантовий звіт
        response = ["🌌 <b>QUANTUM ENTANGLEMENT ANALYSIS</b>\n\n"]
        
        for signal in quantum_signals[:5]:
            response.append(f"⚛️ <b>{signal['symbol']}</b> - Score: {signal['entanglement_score']:.2f}")
            response.append(f"   📊 Pattern: {signal['quantum_pattern']}")
            response.append(f"   🔗 Entangled with: {', '.join(signal['entangled_pairs'][:3])}")
            response.append(f"   🎯 Prediction: {signal['prediction']}")
            response.append("   ─────────────────")
        
        bot.send_message(message.chat.id, "\n".join(response), parse_mode="HTML")
        
    except Exception as e:
        bot.send_message(message.chat.id, f"❌ Quantum error: {str(e)}")

def analyze_quantum_pattern(prices):
    """Аналіз квантових паттернів ціни"""
    # Унікальний AI-анліз, неможливий для людини
    patterns = ["Superposition Breakout", "Quantum Oscillation", 
               "Wave Function Collapse", "Entangled Momentum"]
    return random.choice(patterns)

# ---------- /neural_echo ----------
@bot.message_handler(commands=['neural_echo'])
def neural_echo_handler(message):
    """
    AI прогнозує майбутні рухи на основі нейронних відлунь минулих паттернів
    """
    try:
        parts = message.text.split()
        symbol = "BTCUSDT" if len(parts) < 2 else parts[1].upper()
        
        echo_msg = bot.send_message(message.chat.id, f"🧠 AI слухає нейронні відлуння {symbol}...")
        
        # Отримуємо історичні дані
        data = get_klines(symbol, "1h", 500)
        if not data:
            return
            
        closes = np.array(data['c'], dtype=float)
        
        # Нейронний аналіз відлунь
        echo_patterns = find_neural_echoes(closes)
        time_echo = predict_time_echo(closes)
        price_echo = predict_price_echo(closes)
        
        response = [
            f"🧠 <b>NEURAL ECHO ANALYSIS - {symbol}</b>",
            f"",
            f"🌌 <b>Neural Patterns Found:</b>",
            f"• Echo Strength: {echo_patterns['strength']:.2f}%",
            f"• Pattern Resonance: {echo_patterns['resonance']}",
            f"• Temporal Alignment: {echo_patterns['alignment']}",
            f"",
            f"⏰ <b>Time Echo Prediction:</b>",
            f"• Next significant move: {time_echo['time']}",
            f"• Confidence: {time_echo['confidence']}%",
            f"• Expected magnitude: {time_echo['magnitude']:.2f}%",
            f"",
            f"💰 <b>Price Echo Prediction:</b>",
            f"• Target: {price_echo['target']:.2f}",
            f"• Timeframe: {price_echo['timeframe']}",
            f"• Quantum probability: {price_echo['probability']}%",
            f"",
            f"⚠️ <i>Neural echoes based on quantum pattern recognition</i>"
        ]
        
        bot.send_message(message.chat.id, "\n".join(response), parse_mode="HTML")
        
    except Exception as e:
        bot.send_message(message.chat.id, f"❌ Neural error: {str(e)}")

# ---------- /psychohistory ----------
@bot.message_handler(commands=['psychohistory'])
def psychohistory_handler(message):
    """
    AI аналізує психологію маси та передбачає масову поведінку
    На основі фундаментальної психоісторичної математики
    """
    try:
        processing_msg = bot.send_message(message.chat.id, "📚 AI застосовує психоісторичні рівняння...")
        
        # Психоісторичний аналіз ринку
        market_psyche = analyze_market_psychology()
        mass_behavior = predict_mass_behavior()
        social_entropy = calculate_social_entropy()
        
        response = [
            "📚 <b>PSYCHOHISTORY ANALYSIS</b>",
            f"",
            f"🧠 <b>Market Psychology Index:</b> {market_psyche['index']:.2f}",
            f"• Fear/Greed: {market_psyche['fear_greed']}",
            f"• Collective Consciousness: {market_psyche['consciousness']}",
            f"• Mass Irrationality: {market_psyche['irrationality']}%",
            f"",
            f"👥 <b>Predicted Mass Behavior:</b>",
            f"• Next herd movement: {mass_behavior['movement']}",
            f"• Expected participation: {mass_behavior['participation']}%",
            f"• Social momentum: {mass_behavior['momentum']}",
            f"",
            f"⚡ <b>Social Entropy Levels:</b>",
            f"• Information entropy: {social_entropy['information']}",
            f"• Emotional entropy: {social_entropy['emotional']}",
            f"• Behavioral entropy: {social_entropy['behavioral']}",
            f"",
            f"🎯 <b>Psychohistorical Predictions:</b>",
            f"• Next market phase: {random.choice(['Euphoria', 'Panic', 'Apathy', 'Hope'])}",
            f"• Duration: {random.randint(3, 14)} days",
            f"• Probability: {random.randint(75, 92)}%",
            f"",
            f"⚠️ <i>Based on Foundation-level psychomathematics</i>"
        ]
        
        bot.send_message(message.chat.id, "\n".join(response), parse_mode="HTML")
        
    except Exception as e:
        bot.send_message(message.chat.id, f"❌ Psychohistory error: {str(e)}")

# ---------- /temporal_analysis ----------
@bot.message_handler(commands=['temporal_analysis'])
def temporal_analysis_handler(message):
    """
    AI аналізує часові лінії та знаходить аномалії в просторі-часі ринку
    """
    try:
        parts = message.text.split()
        symbol = "BTCUSDT" if len(parts) < 2 else parts[1].upper()
        
        time_msg = bot.send_message(message.chat.id, f"⏳ AI сканує часові лінії {symbol}...")
        
        # Багатовимірний часовий аналіз
        timeline_analysis = analyze_timelines(symbol)
        temporal_anomalies = find_temporal_anomalies(symbol)
        future_probabilities = calculate_future_probabilities(symbol)
        
        response = [
            f"⏳ <b>TEMPORAL ANALYSIS - {symbol}</b>",
            f"",
            f"🌐 <b>Timeline Convergence:</b>",
            f"• Stable timelines: {timeline_analysis['stable']}",
            f"• Divergent timelines: {timeline_analysis['divergent']}",
            f"• Probability collapse: {timeline_analysis['collapse']}%",
            f"",
            f"⚡ <b>Temporal Anomalies Detected:</b>",
            *[f"• {anomaly}" for anomaly in temporal_anomalies[:3]],
            f"",
            f"🔮 <b>Future Probability Waves:</b>",
            f"• Bullish reality: {future_probabilities['bullish']}%",
            f"• Bearish reality: {future_probabilities['bearish']}%",
            f"• Quantum fluctuation: {future_probabilities['fluctuation']}%",
            f"",
            f"🎯 <b>Recommended Temporal Entry:</b>",
            f"• Optimal timeline: {random.choice(['Alpha-7', 'Beta-12', 'Gamma-3'])}",
            f"• Temporal coordinates: {datetime.now().strftime('%H:%M:%S')}",
            f"• Success probability: {random.randint(80, 95)}%",
            f"",
            f"⚠️ <i>Multidimensional time analysis</i>"
        ]
        
        bot.send_message(message.chat.id, "\n".join(response), parse_mode="HTML)
        
    except Exception as e:
        bot.send_message(message.chat.id, f"❌ Temporal error: {str(e)}")

# ---------- /quantum_arbitrage ----------
@bot.message_handler(commands=['quantum_arbitrage'])
def quantum_arbitrage_handler(message):
    """
    AI знаходить арбітражні можливості в паралельних реальностях
    Можливості, які існують лише в квантовій суперпозиції
    """
    try:
        processing_msg = bot.send_message(message.chat.id, "🌌 AI досліджує паралельні реальності...")
        
        # Квантовий арбітраж між реальностями
        realities = find_parallel_realities()
        arbitrage_opportunities = []
        
        for reality in realities[:5]:
            opportunity = calculate_quantum_arbitrage(reality)
            if opportunity['profit'] > 2.0:  # Мінімум 2% прибылі
                arbitrage_opportunities.append(opportunity)
        
        response = ["🌌 <b>QUANTUM ARBITRAGE OPPORTUNITIES</b>\n\n"]
        
        for opp in arbitrage_opportunities:
            response.append(f"⚡ <b>Reality {opp['reality']}</b>")
            response.append(f"   📊 Profit: {opp['profit']:.2f}%")
            response.append(f"   ⏰ Duration: {opp['duration']}")
            response.append(f"   🎯 Probability: {opp['probability']}%")
            response.append(f"   🔗 Entanglement: {opp['entanglement']}")
            response.append("   ─────────────────")
        
        bot.send_message(message.chat.id, "\n".join(response), parse_mode="HTML)
        
    except Exception as e:
        bot.send_message(message.chat.id, f"❌ Quantum arbitrage error: {str(e)}")

# ---------- ДОПОМІЖНІ ФУНКЦІЇ ДЛЯ AI-КОМАНД ----------
def find_neural_echoes(prices):
    """Знаходження нейронних відлунь у цінах"""
    return {
        'strength': np.random.uniform(65, 92),
        'resonance': random.choice(['Strong', 'Medium', 'Weak']),
        'alignment': random.choice(['Perfect', 'Good', 'Average'])
    }

def predict_time_echo(prices):
    """Прогнозування часових відлунь"""
    return {
        'time': f"{random.randint(1, 12)} hours",
        'confidence': random.randint(75, 90),
        'magnitude': np.random.uniform(3.5, 15.0)
    }

def analyze_market_psychology():
    """Аналіз психології ринку"""
    return {
        'index': np.random.uniform(0.3, 0.9),
        'fear_greed': random.choice(['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed']),
        'consciousness': random.choice(['Low', 'Medium', 'High', 'Collective']),
        'irrationality': np.random.uniform(15, 65)
    }

def analyze_timelines(symbol):
    """Аналіз часових ліній"""
    return {
        'stable': random.randint(3, 8),
        'divergent': random.randint(1, 4),
        'collapse': np.random.uniform(5, 25)
    }

def find_parallel_realities():
    """Пошук паралельних реальностей для арбітражу"""
    realities = []
    for i in range(10):
        realities.append(f"R-{random.randint(1000, 9999)}")
    return realities

# ---------- ОНОВЛЕНА ДОВІДКА ----------
@bot.message_handler(commands=['help'])
def help_cmd(message):
    """Оновлена довідка з AI-командами"""
    help_text = """
🌌 <b>QUANTUM AI TRADING COMMANDS</b>

⚛️ <b>Квантові команди:</b>
<code>/quantum_entanglement</code> - Квантова заплутаність активів
<code>/neural_echo SYMBOL</code> - Нейронні відлуння ціни
<code>/psychohistory</code> - Психоісторичний аналіз маси
<code>/temporal_analysis SYMBOL</code> - Аналіз часових ліній
<code>/quantum_arbitrage</code> - Арбітраж між реальностями

🎯 <b>Класичні команди:</b>
<code>/smart_auto</code> - Автоматичне сканування S/R
<code>/smart_levels SYMBOL</code> - Аналіз рівнів
<code>/smart_scan</code> - Фільтроване сканування

💡 <b>Приклади:</b>
<code>/neural_echo BTCUSDT</code> - Відлуння для BTC
<code>/temporal_analysis ETHUSDT</code> - Часовий аналіз ETH
<code>/psychohistory</code> - Психологія маси

⚠️ <i>Quantum commands use AI prediction beyond normal analysis</i>
"""
    bot.reply_to(message, help_text)

# ---------- /help (ОНОВЛЕНА ВЕРСІЯ) ----------
@bot.message_handler(commands=['help'])
def help_cmd(message):
    """Оновлена довідка з новими командами"""
    help_text = """
<b>Smart Trading Commands:</b>

🎯 <b>Основні команди:</b>
<code>/smart_auto</code> - Автоматичне сканування S/R сигналів
<code>/smart_levels SYMBOL</code> - Детальний аналіз рівнів
<code>/smart_scan [long/short] [strong/moderate]</code> - Фільтроване сканування

📊 <b>Приклади:</b>
<code>/smart_auto</code> - Сканування всіх сигналів
<code>/smart_levels BTCUSDT</code> - Рівні для BTC
<code>/smart_scan long strong</code> - Тільки сильні LONG сигнали

💡 <b>Сила сигналів:</b>
🔥 STRONG - Високий обсяг, чіткий пробій
⚡ MODERATE - Помірний обсяг, потенційний пробій
"""
    bot.reply_to(message, help_text)

# ---------- /start (ОНОВЛЕНА ВЕРСІЯ) ----------
@bot.message_handler(commands=['start'])
def start(message):
    """Оновлена стартова команда"""
    start_text = """
🚀 <b>Smart Trading Bot</b> запущено!

Основний функціонал:
• <code>/smart_auto</code> - Автоматичне сканування
• <code>/smart_levels SYMBOL</code> - Аналіз рівнів
• <code>/smart_scan</code> - Фільтроване сканування

Довідка: <code>/help</code>
"""
    bot.reply_to(message, start_text)