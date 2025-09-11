import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
import asyncio
from flask import Flask, request, jsonify
import threading
import json
from typing import Dict, List, Optional, Tuple
import logging
import math
import os
import re
import heapq
from collections import deque

# Налаштування логування
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedPumpDumpBot:
    def __init__(self, token: str):
        self.token = token
        self.app = Application.builder().token(token).build()
        self.flask_app = Flask(__name__)
        self.setup_flask_routes()
        
        # Історичні дані для аналізу трендів
        self.historical_data = deque(maxlen=1000)
        self.market_metrics = {}
        
        # Списки для фільтрації
        self.garbage_symbols = {
            'USD', 'EUR', 'GBP', 'JPY', 'CNY', 'RUB', 'TRY', 'BRL', 'KRW', 'AUD',
            'USDC', 'USDT', 'BUSD', 'DAI', 'TUSD', 'PAX', 'UST', 'HUSD', 'GUSD',
            'BVND', 'IDRT', 'BIDR', 'BKRW', 'NGN', 'UAH', 'VAI', 'SUSD', 'USDN'
        }
        
        # Регулярний вираз для фільтрації сміття
        self.garbage_pattern = re.compile(
            r'.*[0-9]+[LNS].*|.*UP|DOWN|BEAR|BULL|HALF|QUARTER.*|.*[0-9]{3,}.*',
            re.IGNORECASE
        )
        
        # Пороги для пампу (оптимізовані для памп/дамп стратегії)
        self.pump_thresholds = {
            'volume_ratio': 2.8,           # Зростання об'єму
            'price_change_1h': 7.0,        # Зміна ціни за 1 годину
            'price_change_5m': 2.5,        # Зміна ціни за 5 хвилин
            'price_change_24h': 15.0,      # Зміна ціни за 24 години
            'rsi_threshold': 62,           # RSI рівень
            'buy_pressure_ratio': 1.3,     # Співвідношення купівлі/продажу
            'macd_signal': 0.0008,         # MACD сигнал
            'min_volume': 50000,           # Мінімальний об'єм в USDT
            'max_volume': 5000000,         # Максимальний об'єм (фільтр великих монет)
            'market_cap_max': 100000000,   # Макс капіталізація
            'liquidity_score': 0.7,        # Мінімальна ліквідність
            'volatility_ratio': 2.0        # Співвідношення волатильності
        }
        
        # Пороги для дампу
        self.dump_thresholds = {
            'volume_ratio': 2.5,
            'price_change_1h': -6.0,
            'price_change_5m': -2.0,
            'price_change_24h': -12.0,
            'rsi_threshold': 38,
            'sell_pressure_ratio': 1.4,
            'macd_signal': -0.0007,
            'min_volume': 50000,
            'max_volume': 5000000,
            'market_cap_max': 100000000,
            'liquidity_score': 0.7,
            'volatility_ratio': 1.8
        }
        
        self.coin_blacklist = set()
        self.last_signals = {}
        self.performance_stats = {
            'total_scans': 0,
            'signals_found': 0,
            'success_rate': 0.0,
            'avg_scan_time': 0.0
        }
        
        self.setup_handlers()
        
    def setup_flask_routes(self):
        @self.flask_app.route('/webhook', methods=['POST'])
        def webhook():
            data = request.json
            return self.handle_webhook(data)
            
        @self.flask_app.route('/stats', methods=['GET'])
        def stats():
            return jsonify({
                'performance': self.performance_stats,
                'last_signals_count': len(self.last_signals),
                'settings': {
                    'pump': self.pump_thresholds,
                    'dump': self.dump_thresholds
                }
            })
            
        @self.flask_app.route('/update_settings', methods=['POST'])
        def update_settings():
            data = request.json
            self.update_settings(data)
            return jsonify({'status': 'success'})
            
        @self.flask_app.route('/health', methods=['GET'])
        def health():
            return jsonify({
                'status': 'healthy', 
                'timestamp': datetime.now().isoformat(),
                'uptime': time.time() - self.start_time
            })
    
    def setup_handlers(self):
        self.app.add_handler(CommandHandler("start", self.start))
        self.app.add_handler(CommandHandler("scan", self.scan_command))
        self.app.add_handler(CommandHandler("settings", self.settings_command))
        self.app.add_handler(CommandHandler("blacklist", self.blacklist_command))
        self.app.add_handler(CommandHandler("debug", self.debug_command))
        self.app.add_handler(CommandHandler("whale", self.whale_command))
        self.app.add_handler(CommandHandler("topsignals", self.top_signals_command))
        self.app.add_handler(CommandHandler("performance", self.performance_command))
        self.app.add_handler(CommandHandler("analysis", self.market_analysis_command))
        self.app.add_handler(CallbackQueryHandler(self.button_handler))
        
        self.start_time = time.time()

    def is_garbage_symbol(self, symbol: str) -> bool:
        """Перевіряє чи символ є сміттям"""
        symbol = symbol.upper()
        
        if symbol in self.garbage_symbols:
            return True
        
        if self.garbage_pattern.match(symbol):
            return True
        
        if len(symbol) > 12:
            return True
        
        if any(char.isdigit() for char in symbol[1:-1]):
            return True
            
        if symbol.endswith(('UP', 'DOWN', 'BULL', 'BEAR')):
            return True
            
        return False

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        keyboard = [
            [InlineKeyboardButton("🔍 Сканувати зараз", callback_data="scan_now")],
            [InlineKeyboardButton("⚙️ Налаштування", callback_data="settings")],
            [InlineKeyboardButton("📊 Статистика", callback_data="stats")],
            [InlineKeyboardButton("🚫 Чорний список", callback_data="blacklist")],
            [InlineKeyboardButton("🐋 Whale Alert", callback_data="whale_alert")],
            [InlineKeyboardButton("📈 ТОП сигнали", callback_data="top_signals")],
            [InlineKeyboardButton("📋 Аналіз ринку", callback_data="market_analysis")],
            [InlineKeyboardButton("🏆 Продуктивність", callback_data="performance")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            "🤖 **ULTIMATE PUMP & DUMP DETECTOR**\n\n"
            "🚀 *Найрозумніший бот для виявлення пампів та дампів*\n\n"
            "✨ **Унікальні можливості:**\n"
            "• 🎯 AI-детекція аномалій об'єму\n"
            "• 📊 Мультитаймфреймний аналіз\n"
            "• 🐋 Whale order detection\n"
            "• 🔮 Прогнозування трендів\n"
            "• 📈 RSI + MACD + Bollinger Bands\n"
            "• 🌊 Liquidity analysis\n"
            "• ⚡ Real-time alerts\n"
            "• 📱 Smart notifications\n\n"
            "💎 *Створено AI для максимальної ефективності*",
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )

    async def get_market_data(self, symbol: str) -> Optional[Dict]:
        try:
            # Перевірка чи символ не сміття
            if self.is_garbage_symbol(symbol):
                return None
            
            # Отримання даних з Binance
            url = f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}USDT"
            response = requests.get(url, timeout=8)
            data = response.json()
            
            if 'code' in data:
                return None
            
            quote_volume = float(data['quoteVolume'])
            price_change = float(data['priceChangePercent'])
            
            # Фільтрація за об'ємом та зміною ціни
            if (quote_volume < self.pump_thresholds['min_volume'] or 
                quote_volume > self.pump_thresholds['max_volume'] or
                abs(price_change) < 5.0):  # Мінімальна зміна ціни
                return None
            
            # Отримання кляйнсів
            klines_data = {}
            timeframes = {
                '5m': '&interval=5m&limit=100',
                '15m': '&interval=15m&limit=100',
                '1h': '&interval=1h&limit=50',
                '4h': '&interval=4h&limit=25'
            }
            
            for tf, params in timeframes.items():
                klines_url = f"https://api.binance.com/api/v3/klines?symbol={symbol}USDT{params}"
                klines_response = requests.get(klines_url, timeout=8)
                klines_data[tf] = klines_response.json()
            
            # Order book data
            orderbook_url = f"https://api.binance.com/api/v3/depth?symbol={symbol}USDT&limit=15"
            orderbook_response = requests.get(orderbook_url, timeout=8)
            orderbook_data = orderbook_response.json()
            
            # Trades data
            trades_url = f"https://api.binance.com/api/v3/trades?symbol={symbol}USDT&limit=20"
            trades_response = requests.get(trades_url, timeout=8)
            trades_data = trades_response.json()
            
            market_data = {
                'symbol': symbol,
                'price': float(data['lastPrice']),
                'volume': float(data['volume']),
                'quote_volume': quote_volume,
                'price_change_24h': price_change,
                'price_change': float(data['priceChange']),
                'high': float(data['highPrice']),
                'low': float(data['lowPrice']),
                'trades': len(trades_data),
                'klines': klines_data,
                'orderbook': orderbook_data,
                'trades_data': trades_data,
                'timestamp': datetime.now().isoformat()
            }
            
            return market_data
            
        except Exception as e:
            logger.error(f"Помилка отримання даних для {symbol}: {e}")
            return None

    def calculate_advanced_indicators(self, klines_data: List) -> Dict:
        """Розширені технічні індикатори"""
        try:
            closes = np.array([float(x[4]) for x in klines_data])
            highs = np.array([float(x[2]) for x in klines_data])
            lows = np.array([float(x[3]) for x in klines_data])
            volumes = np.array([float(x[5]) for x in klines_data])
            
            # Базові індикатори
            rsi = self.calculate_rsi(closes)
            macd = self.calculate_macd(closes)
            upper_bb, middle_bb, lower_bb = self.calculate_bollinger_bands(closes)
            
            # Розширені індикатори
            volatility = np.std(closes[-20:]) / np.mean(closes[-20:]) if len(closes) >= 20 else 0
            volume_velocity = np.mean(np.diff(volumes[-10:])) if len(volumes) >= 10 else 0
            price_acceleration = np.mean(np.diff(np.diff(closes[-10:]))) if len(closes) >= 12 else 0
            
            # Аналіз тренду
            short_trend = self.calculate_trend_strength(closes, 10)
            medium_trend = self.calculate_trend_strength(closes, 20)
            
            # Об'ємний аналіз
            volume_sma = self.calculate_sma(volumes, 20)
            volume_ratio = volumes[-1] / volume_sma if volume_sma > 0 else 1
            
            return {
                'rsi': rsi,
                'macd': macd,
                'bb_upper': upper_bb,
                'bb_middle': middle_bb,
                'bb_lower': lower_bb,
                'volatility': volatility,
                'volume_velocity': volume_velocity,
                'price_acceleration': price_acceleration,
                'short_trend': short_trend,
                'medium_trend': medium_trend,
                'volume_ratio': volume_ratio,
                'current_price': closes[-1],
                'price_5m_ago': closes[-6] if len(closes) >= 6 else closes[0],
                'price_1h_ago': closes[-12] if len(closes) >= 12 else closes[0],
                'price_4h_ago': closes[-48] if len(closes) >= 48 else closes[0]
            }
            
        except Exception as e:
            logger.error(f"Помилка розрахунку індикаторів: {e}")
            return {}

    def calculate_trend_strength(self, prices: np.ndarray, period: int) -> float:
        """Розрахунок сили тренду"""
        if len(prices) < period:
            return 0.0
        
        returns = np.diff(prices[-period:]) / prices[-period:-1]
        trend_strength = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        return trend_strength

    def detect_volume_anomaly(self, volumes: np.ndarray) -> bool:
        """Детекція аномальних об'ємів з використанням ML"""
        try:
            if len(volumes) < 20:
                return False
            
            # Використання Isolation Forest для аномалій
            volumes_reshaped = volumes[-20:].reshape(-1, 1)
            scaler = StandardScaler()
            volumes_scaled = scaler.fit_transform(volumes_reshaped)
            
            model = IsolationForest(contamination=0.1, random_state=42)
            predictions = model.fit_predict(volumes_scaled)
            
            # Останній об'єм є аномалією
            return predictions[-1] == -1
            
        except Exception as e:
            logger.error(f"Помилка детекції аномалій: {e}")
            return False

    async def scan_for_pump_dump(self):
        """Основна функція сканування для памп/дамп"""
        try:
            start_time = time.time()
            
            # Отримання топ монет за зміною ціни за 24г
            response = requests.get("https://api.binance.com/api/v3/ticker/24hr", timeout=10)
            all_data = response.json()
            
            # Фільтрація USDT пар та сортування за зміною ціни (абсолютне значення)
            usdt_pairs = [
                x for x in all_data 
                if x['symbol'].endswith('USDT') 
                and not self.is_garbage_symbol(x['symbol'].replace('USDT', ''))
            ]
            
            # Сортування за абсолютною зміною ціни (найбільші рухи)
            sorted_by_change = sorted(
                usdt_pairs,
                key=lambda x: abs(float(x['priceChangePercent'])),
                reverse=True
            )[:50]  # Топ 50 за зміною ціни
            
            results = {'pump': [], 'dump': []}
            
            for coin in sorted_by_change:
                symbol = coin['symbol'].replace('USDT', '')
                
                if symbol in self.coin_blacklist:
                    continue
                    
                market_data = await self.get_market_data(symbol)
                if not market_data:
                    continue
                
                # Аналіз технічних індикаторів
                indicators = self.calculate_advanced_indicators(market_data['klines']['15m'])
                
                # Детекція пампу
                pump_score = self.calculate_pump_score(market_data, indicators)
                if pump_score > 0.65:  # Високий поріг для якості
                    results['pump'].append({
                        'symbol': symbol,
                        'score': pump_score,
                        'price': market_data['price'],
                        'volume': market_data['quote_volume'],
                        'change_24h': market_data['price_change_24h'],
                        'indicators': indicators
                    })
                
                # Детекція дампу
                dump_score = self.calculate_dump_score(market_data, indicators)
                if dump_score > 0.65:
                    results['dump'].append({
                        'symbol': symbol,
                        'score': dump_score,
                        'price': market_data['price'],
                        'volume': market_data['quote_volume'],
                        'change_24h': market_data['price_change_24h'],
                        'indicators': indicators
                    })
            
            # Сортування результатів
            for key in results:
                results[key].sort(key=lambda x: x['score'], reverse=True)
            
            scan_time = time.time() - start_time
            self.performance_stats['total_scans'] += 1
            self.performance_stats['signals_found'] += len(results['pump']) + len(results['dump'])
            self.performance_stats['avg_scan_time'] = (
                self.performance_stats['avg_scan_time'] * (self.performance_stats['total_scans'] - 1) + scan_time
            ) / self.performance_stats['total_scans']
            
            return results
            
        except Exception as e:
            logger.error(f"Помилка сканування: {e}")
            return {'pump': [], 'dump': []}

    def calculate_pump_score(self, market_data: Dict, indicators: Dict) -> float:
        """Розширений розрахунок скору для пампу"""
        score = 0.0
        weights = {
            'volume_ratio': 0.25,
            'price_change': 0.20,
            'rsi': 0.15,
            'buy_pressure': 0.20,
            'volatility': 0.10,
            'trend': 0.10
        }
        
        try:
            # Аналіз об'єму
            if indicators.get('volume_ratio', 1) > self.pump_thresholds['volume_ratio']:
                score += weights['volume_ratio']
            
            # Аналіз ціни
            price_change_5m = ((indicators['current_price'] - indicators['price_5m_ago']) / 
                             indicators['price_5m_ago']) * 100
            if price_change_5m > self.pump_thresholds['price_change_5m']:
                score += weights['price_change']
            
            # RSI
            if indicators.get('rsi', 50) > self.pump_thresholds['rsi_threshold']:
                score += weights['rsi']
            
            # Аналіз стакану
            ob_analysis = self.analyze_orderbook(market_data['orderbook'])
            if ob_analysis['buy_pressure'] > self.pump_thresholds['buy_pressure_ratio']:
                score += weights['buy_pressure']
            
            # Волатильність
            if indicators.get('volatility', 0) > 0.02:
                score += weights['volatility']
            
            # Тренд
            if indicators.get('short_trend', 0) > 0:
                score += weights['trend']
                
        except Exception as e:
            logger.error(f"Помилка розрахунку pump score: {e}")
        
        return min(score, 1.0)

    # Аналогічна функція для calculate_dump_score...

    async def send_advanced_alert(self, context: ContextTypes.DEFAULT_TYPE, signal_data: Dict, signal_type: str):
        """Розширені сповіщення"""
        symbol = signal_data['symbol']
        
        emoji = "🚀" if signal_type == 'pump' else "📉"
        title = "ПОТЕНЦІЙНИЙ ПАМП" if signal_type == 'pump' else "ПОТЕНЦІЙНИЙ ДАМП"
        
        message = f"{emoji} **{title}** {emoji}\n\n"
        message += f"🔸 **Монета:** `{symbol}`\n"
        message += f"🔸 **Ціна:** ${signal_data['price']:.6f}\n"
        message += f"🔸 **24h Зміна:** {signal_data['change_24h']:+.1f}%\n"
        message += f"🔸 **Об'єм:** {signal_data['volume']:,.0f} USDT\n"
        message += f"🔸 **Впевненість:** {signal_data['score']:.1%}\n"
        message += f"🔸 **RSI:** {signal_data['indicators'].get('rsi', 50):.1f}\n"
        message += f"🔸 **Волатильність:** {signal_data['indicators'].get('volatility', 0):.2%}\n\n"
        
        message += "📊 **Сигнали:**\n"
        if signal_data['score'] > 0.8:
            message += "🎯 *Високоякісний сигнал*\n"
        elif signal_data['score'] > 0.65:
            message += "✅ *Середня якість*\n"
        else:
            message += "⚠️ *Низька якість*\n"
        
        message += f"\n⏰ *{datetime.now().strftime('%H:%M:%S')}*"
        
        # Збереження сигналу
        self.last_signals[f"{symbol}_{datetime.now().timestamp()}"] = {
            'type': signal_type,
            'time': datetime.now().isoformat(),
            'data': signal_data
        }
        
        await context.bot.send_message(
            chat_id=context.job.chat_id,
            text=message,
            parse_mode='Markdown'
        )

    # Додаткові команди та функції...

    async def market_analysis_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда для аналізу ринку"""
        analysis_msg = "📈 **Аналіз ринку**\n\n"
        
        # Отримання загальної інформації про ринок
        try:
            response = requests.get("https://api.binance.com/api/v3/ticker/24hr", timeout=10)
            data = response.json()
            
            total_volume = sum(float(x['quoteVolume']) for x in data if 'USDT' in x['symbol'])
            avg_change = np.mean([float(x['priceChangePercent']) for x in data if 'USDT' in x['symbol']])
            
            analysis_msg += f"• 📊 Загальний об'єм: {total_volume:,.0f} USDT\n"
            analysis_msg += f"• 📈 Середня зміна: {avg_change:+.2f}%\n"
            analysis_msg += f"• 🔍 Монет у моніторингу: {len(self.last_signals)}\n"
            analysis_msg += f"• ⚡ Останнє оновлення: {datetime.now().strftime('%H:%M')}\n\n"
            
            analysis_msg += "🎯 *Рекомендації:*\n"
            if avg_change > 2.0:
                analysis_msg += "• 📈 Риск пампів збільшений\n"
            elif avg_change < -2.0:
                analysis_msg += "• 📉 Риск дампів збільшений\n"
            else:
                analysis_msg += "• ⚖️ Ринок стабільний\n"
                
        except Exception as e:
            analysis_msg += "❌ Помилка аналізу ринку\n"
        
        await update.message.reply_text(analysis_msg, parse_mode='Markdown')

    async def performance_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда статистики продуктивності"""
        perf_msg = "🏆 **Статистика продуктивності**\n\n"
        perf_msg += f"• 📊 Всього сканувань: {self.performance_stats['total_scans']}\n"
        perf_msg += f"• 📈 Знайдено сигналів: {self.performance_stats['signals_found']}\n"
        perf_msg += f"• ⚡ Середній час сканування: {self.performance_stats['avg_scan_time']:.2f}s\n"
        perf_msg += f"• 🎯 Успішність: {self.performance_stats.get('success_rate', 0):.1%}\n"
        perf_msg += f"• ⏰ Аптайм: {timedelta(seconds=int(time.time() - self.start_time))}\n\n"
        
        perf_msg += "📈 *Останні 10 сигналів:*\n"
        recent_signals = list(self.last_signals.values())[-10:]
        for i, signal in enumerate(recent_signals, 1):
            perf_msg += f"{i}. {signal['data']['symbol']} - {signal['type']} - {signal['data']['score']:.1%}\n"
        
        await update.message.reply_text(perf_msg, parse_mode='Markdown')

    # Решта функцій...

    def run(self):
        """Запуск бота"""
        print("🤖 Запуск ULTIMATE PUMP/DUMP BOT...")
        print("🎯 Версія: 2.0 (AI Enhanced)")
        print("📊 Спеціалізація: Памп/дамп стратегії")
        print("⚡ Фільтрація: Топ монети за зміною ціни")
        print("💎 Створено з використанням передових AI технологій")
        
        # Запуск Flask
        flask_thread = threading.Thread(target=self.run_flask, daemon=True)
        flask_thread.start()
        
        self.app.run_polling()

# Використання
if __name__ == "__main__":
    TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not TOKEN:
        print("❌ Помилка: TELEGRAM_BOT_TOKEN не встановлено")
        exit(1)
    
    bot = AdvancedPumpDumpBot(TOKEN)
    bot.run()