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
            'USDC', 'USDT', 'BUSD', 'DAI', 'TUSD', 'PAX', 'UST', 'HUSD', 'GUSD'
        }
        
        # Популярні монети для пріоритетного сканування
        self.popular_coins = {
            'BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'ADA', 'AVAX', 'DOT', 'DOGE', 'SHIB',
            'MATIC', 'LTC', 'LINK', 'ATOM', 'UNI', 'XMR', 'ETC', 'XLM', 'BCH', 'FIL'
        }
        
        # Унікальні умови для PRE-TOP DETECT
        self.pump_thresholds = {
            'volume_ratio': 3.5,
            'price_change_1h': 8.0,
            'price_change_5m': 4.0,
            'rsi_overbought': 72,
            'rsi_divergence': True,
            'macd_momentum': 0.001,
            'min_volume': 100000,
            'max_volume': 5000000,
            'buy_pressure_ratio': 1.8,
            'volatility_ratio': 2.2,
            'price_acceleration': 0.0005
        }
        
        # Унікальні умови для PRE-TOP DETECT (дамп)
        self.dump_thresholds = {
            'volume_ratio': 2.8,
            'price_rejection': -2.5,
            'rsi_divergence_bearish': True,
            'rsi_overbought': 70,
            'macd_reversal': -0.0008,
            'sell_pressure_ratio': 1.6,
            'support_break': True,
            'wick_ratio': 0.4,
            'orderbook_imbalance': 0.3
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
        self.start_time = time.time()
        
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
        self.app.add_handler(CommandHandler("analysis", self.market_analysis_command))
        self.app.add_handler(CallbackQueryHandler(self.button_handler))

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        keyboard = [
            [InlineKeyboardButton("🔍 Сканувати PRE-TOP", callback_data="scan_now")],
            [InlineKeyboardButton("⚙️ Налаштування", callback_data="settings")],
            [InlineKeyboardButton("📊 Статистика", callback_data="stats")],
            [InlineKeyboardButton("🚫 Чорний список", callback_data="blacklist")],
            [InlineKeyboardButton("📈 Аналіз ринку", callback_data="market_analysis")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            "🤖 **PRE-TOP DETECT BOT**\n\n"
            "🎯 *Спеціалізація: виявлення точок розвороту після пампу*\n\n"
            "✨ **Унікальні сигнали:**\n"
            "• 📊 RSI дивергенції\n"
            "• 📉 Аналіз рівнів resistance\n"
            "• 🎯 Volume-price divergence\n"
            "• ⚡ MACD reversal patterns\n"
            "• 📍 Wick analysis (верхні тіні)\n"
            "• 🌊 Order book imbalance\n"
            "• 🔄 Momentum slowdown\n\n"
            "💎 *Точний вхід в дамп після пампу*",
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )

    def is_garbage_symbol(self, symbol: str) -> bool:
        """Перевіряє чи символ є сміттям"""
        symbol = symbol.upper()
        
        if symbol in self.popular_coins:
            return False
            
        if symbol in self.garbage_symbols:
            return True
        
        if len(symbol) > 8:
            return True
        
        if any(char.isdigit() for char in symbol):
            return True
            
        if symbol.endswith(('UP', 'DOWN', 'BULL', 'BEAR', 'USD', 'EUR')):
            return True
            
        return False

    async def get_market_data(self, symbol: str) -> Optional[Dict]:
        try:
            if self.is_garbage_symbol(symbol):
                return None
            
            # Отримуємо детальні дані
            url = f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}USDT"
            response = requests.get(url, timeout=8)
            data = response.json()
            
            if 'code' in data:
                return None
            
            quote_volume = float(data['quoteVolume'])
            price_change = float(data['priceChangePercent'])
            current_price = float(data['lastPrice'])
            
            # Фільтр за об'ємом
            if quote_volume < self.pump_thresholds['min_volume']:
                return None
            
            # Отримуємо детальні клайнси для аналізу
            klines_data = {}
            timeframes = {
                '5m': '&interval=5m&limit=100',
                '15m': '&interval=15m&limit=100',
                '1h': '&interval=1h&limit=50',
                '4h': '&interval=4h&limit=25'
            }
            
            for tf, params in timeframes.items():
                try:
                    klines_url = f"https://api.binance.com/api/v3/klines?symbol={symbol}USDT{params}"
                    klines_response = requests.get(klines_url, timeout=5)
                    klines_data[tf] = klines_response.json()
                except:
                    continue
            
            # Order book data для аналізу imbalance
            try:
                orderbook_url = f"https://api.binance.com/api/v3/depth?symbol={symbol}USDT&limit=20"
                orderbook_response = requests.get(orderbook_url, timeout=5)
                orderbook_data = orderbook_response.json()
            except:
                orderbook_data = {'bids': [], 'asks': []}
            
            market_data = {
                'symbol': symbol,
                'price': current_price,
                'volume': float(data['volume']),
                'quote_volume': quote_volume,
                'price_change_24h': price_change,
                'high': float(data['highPrice']),
                'low': float(data['lowPrice']),
                'price_change': float(data['priceChange']),
                'klines': klines_data,
                'orderbook': orderbook_data,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Дані для {symbol}: ціна ${current_price}, зміна {price_change:.1f}%")
            return market_data
            
        except Exception as e:
            logger.error(f"Помилка отримання даних для {symbol}: {e}")
            return None

    def calculate_advanced_indicators(self, klines_data: List) -> Dict:
        """Розширені технічні індикатори для PRE-TOP detection"""
        try:
            closes = np.array([float(x[4]) for x in klines_data])
            highs = np.array([float(x[2]) for x in klines_data])
            lows = np.array([float(x[3]) for x in klines_data])
            volumes = np.array([float(x[5]) for x in klines_data])
            
            # Базові індикатори
            rsi = self.calculate_rsi(closes)
            macd = self.calculate_macd(closes)
            
            # Розширені індикатори для pre-top detect
            price_acceleration = self.calculate_price_acceleration(closes)
            volume_price_divergence = self.calculate_volume_price_divergence(closes, volumes)
            wick_analysis = self.calculate_wick_analysis(highs, lows, closes)
            resistance_levels = self.find_resistance_levels(highs, closes)
            
            return {
                'rsi': rsi,
                'macd': macd,
                'price_acceleration': price_acceleration,
                'volume_divergence': volume_price_divergence,
                'wick_ratio': wick_analysis,
                'resistance_levels': resistance_levels,
                'current_price': closes[-1],
                'price_5m_ago': closes[-6] if len(closes) >= 6 else closes[0],
                'price_1h_ago': closes[-12] if len(closes) >= 12 else closes[0],
                'high_24h': np.max(highs) if len(highs) > 0 else closes[-1],
                'low_24h': np.min(lows) if len(lows) > 0 else closes[-1]
            }
            
        except Exception as e:
            logger.error(f"Помилка розрахунку індикаторів: {e}")
            return {}

    def calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Розрахунок RSI"""
        if len(prices) < period + 1:
            return 50.0
        
        try:
            deltas = np.diff(prices)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            avg_gain = np.mean(gains[-period:])
            avg_loss = np.mean(losses[-period:])
            
            if avg_loss == 0:
                return 100.0
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
        except Exception as e:
            return 50.0

    def calculate_macd(self, prices: np.ndarray, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> float:
        """Розрахунок MACD"""
        if len(prices) < slow_period:
            return 0.0
        
        try:
            # EMA calculation
            def ema(data, period):
                return pd.Series(data).ewm(span=period).mean().iloc[-1]
            
            fast_ema = ema(prices, fast_period)
            slow_ema = ema(prices, slow_period)
            macd_line = fast_ema - slow_ema
            
            return macd_line
        except Exception as e:
            return 0.0

    def calculate_price_acceleration(self, prices: np.ndarray) -> float:
        """Розрахунок прискорення ціни"""
        if len(prices) < 10:
            return 0.0
        
        try:
            # Друга похідна (прискорення)
            first_derivative = np.diff(prices[-10:])
            second_derivative = np.diff(first_derivative)
            acceleration = np.mean(second_derivative) if len(second_derivative) > 0 else 0
            return acceleration
        except:
            return 0.0

    def calculate_volume_price_divergence(self, prices: np.ndarray, volumes: np.ndarray) -> float:
        """Аналіз дивергенції ціни та об'єму"""
        if len(prices) < 20 or len(volumes) < 20:
            return 0.0
        
        try:
            # Кореляція між зміною ціни та об'ємом
            price_changes = np.diff(prices[-20:])
            volume_changes = np.diff(volumes[-20:])
            
            if len(price_changes) == len(volume_changes) and len(price_changes) > 1:
                correlation = np.corrcoef(price_changes, volume_changes)[0, 1]
                return correlation
            return 0.0
        except:
            return 0.0

    def calculate_wick_analysis(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> float:
        """Аналіз верхніх тіней (wick)"""
        if len(highs) < 10 or len(lows) < 10 or len(closes) < 10:
            return 0.0
        
        try:
            # Відношення верхніх тіней до тіла свічки
            recent_highs = highs[-10:]
            recent_lows = lows[-10:]
            recent_closes = closes[-10:]
            
            upper_wicks = recent_highs - np.maximum(recent_closes, recent_lows)
            bodies = np.abs(recent_closes - recent_lows)
            
            wick_ratio = np.mean(upper_wicks / np.where(bodies > 0, bodies, 1))
            return wick_ratio
        except:
            return 0.0

    def find_resistance_levels(self, highs: np.ndarray, closes: np.ndarray, tolerance: float = 0.02) -> int:
        """Пошук рівнів resistance"""
        if len(highs) < 20:
            return 0
        
        try:
            # Знаходимо локальні максимуми
            resistance_levels = 0
            current_price = closes[-1]
            
            for i in range(5, len(highs)-5):
                if highs[i] == np.max(highs[i-5:i+5]):
                    # Перевіряємо чи цей рівень ще актуальний
                    if abs(highs[i] - current_price) / current_price < tolerance:
                        resistance_levels += 1
            
            return resistance_levels
        except:
            return 0

    def analyze_orderbook_imbalance(self, orderbook: Dict) -> float:
        """Аналіз imbalance в стакані"""
        try:
            bids = np.array([float(bid[1]) for bid in orderbook['bids'][:10]])
            asks = np.array([float(ask[1]) for ask in orderbook['asks'][:10]])
            
            total_bids = np.sum(bids)
            total_asks = np.sum(asks)
            
            if total_bids + total_asks > 0:
                imbalance = (total_bids - total_asks) / (total_bids + total_asks)
                return imbalance
            return 0.0
        except:
            return 0.0

    def detect_rsi_divergence(self, prices: np.ndarray, rsi_values: np.ndarray) -> bool:
        """Детекція RSI дивергенції"""
        if len(prices) < 20 or len(rsi_values) < 20:
            return False
        
        try:
            # Перевірка медвежчої дивергенції
            price_high = np.max(prices[-10:])
            rsi_high = np.max(rsi_values[-10:])
            
            price_prev_high = np.max(prices[-20:-10])
            rsi_prev_high = np.max(rsi_values[-20:-10])
            
            # Медвежа дивергенція: ціна робить новий high, а RSI - ні
            bearish_divergence = (price_high > price_prev_high) and (rsi_high < rsi_prev_high)
            
            return bearish_divergence
        except:
            return False

    async def scan_for_pre_top_signals(self):
        """Сканування для PRE-TOP detection"""
        try:
            start_time = time.time()
            
            # Отримуємо топ монети за зміною ціни
            response = requests.get("https://api.binance.com/api/v3/ticker/24hr", timeout=10)
            all_data = response.json()
            
            # Фільтруємо USDT пари з хорошою зміною ціни
            usdt_pairs = [
                x for x in all_data 
                if x['symbol'].endswith('USDT') 
                and float(x['priceChangePercent']) > 5.0  # Мінімум +5% зміна
                and not self.is_garbage_symbol(x['symbol'].replace('USDT', ''))
            ]
            
            # Сортуємо за зміною ціни
            sorted_by_change = sorted(
                usdt_pairs,
                key=lambda x: float(x['priceChangePercent']),
                reverse=True
            )[:30]  # Топ 30 за зростанням
            
            results = {'pre_top': []}
            
            for coin in sorted_by_change:
                symbol = coin['symbol'].replace('USDT', '')
                
                if symbol in self.coin_blacklist:
                    continue
                    
                market_data = await self.get_market_data(symbol)
                if not market_data:
                    continue
                
                # Аналізуємо 15-хвилинні графіки для точного виявлення
                indicators = self.calculate_advanced_indicators(market_data['klines']['15m'])
                
                # Детекція PRE-TOP сигналів
                pre_top_score = self.calculate_pre_top_score(market_data, indicators)
                if pre_top_score > 0.6:
                    results['pre_top'].append({
                        'symbol': symbol,
                        'score': pre_top_score,
                        'price': market_data['price'],
                        'change_24h': market_data['price_change_24h'],
                        'indicators': indicators,
                        'signals': self.get_pre_top_signals(market_data, indicators)
                    })
            
            # Сортуємо за score
            results['pre_top'].sort(key=lambda x: x['score'], reverse=True)
            
            scan_time = time.time() - start_time
            self.performance_stats['total_scans'] += 1
            self.performance_stats['signals_found'] += len(results['pre_top'])
            self.performance_stats['avg_scan_time'] = (
                self.performance_stats['avg_scan_time'] * (self.performance_stats['total_scans'] - 1) + scan_time
            ) / self.performance_stats['total_scans']
            
            return results
            
        except Exception as e:
            logger.error(f"Помилка сканування: {e}")
            return {'pre_top': []}

    def calculate_pre_top_score(self, market_data: Dict, indicators: Dict) -> float:
        """Розрахунок скору для PRE-TOP detection"""
        score = 0.0
        signals = []
        
        try:
            # 1. RSI дивергенція (30%)
            rsi_values = [self.calculate_rsi(np.array([float(x[4]) for x in market_data['klines']['15m'][-i:]])) 
                         for i in range(20, 10, -1)]
            
            if self.detect_rsi_divergence(
                np.array([float(x[4]) for x in market_data['klines']['15m'][-20:]]),
                np.array(rsi_values)
            ):
                score += 0.3
                signals.append("RSI дивергенція")
            
            # 2. Верхні тіні (20%)
            if indicators.get('wick_ratio', 0) > 0.3:
                score += 0.2
                signals.append("Великі верхні тіні")
            
            # 3. Order book imbalance (20%)
            ob_imbalance = self.analyze_orderbook_imbalance(market_data['orderbook'])
            if ob_imbalance < -0.2:  # Сильний продавлений тиск
                score += 0.2
                signals.append("Imbalance в стакані")
            
            # 4. Сповільнення моментуму (15%)
            if indicators.get('price_acceleration', 0) < 0:
                score += 0.15
                signals.append("Сповільнення моментуму")
            
            # 5. Volume-price divergence (15%)
            if indicators.get('volume_divergence', 0) < 0:
                score += 0.15
                signals.append("Дивергенція об'єму")
                
        except Exception as e:
            logger.error(f"Помилка розрахунку pre-top score: {e}")
        
        return min(score, 1.0)

    def get_pre_top_signals(self, market_data: Dict, indicators: Dict) -> List[str]:
        """Отримання списку сигналів"""
        signals = []
        
        try:
            # RSI
            if indicators.get('rsi', 50) > 70:
                signals.append(f"RSI {indicators['rsi']:.1f} (перекупленість)")
            
            # MACD
            if indicators.get('macd', 0) < 0:
                signals.append("MACD негативний")
            
            # Рівні resistance
            if indicators.get('resistance_levels', 0) > 0:
                signals.append(f"{indicators['resistance_levels']} рівнів resistance")
            
            # Ціна біля high
            current_price = indicators.get('current_price', 0)
            high_24h = indicators.get('high_24h', 0)
            if high_24h > 0 and (high_24h - current_price) / high_24h < 0.02:
                signals.append("Біля 24h high")
                
        except:
            pass
        
        return signals

    async def scan_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда сканування"""
        await update.message.reply_text("🔍 Шукаю PRE-TOP сигнали...")
        results = await self.scan_for_pre_top_signals()
        
        message = "🎯 **PRE-TOP SIGNALS**\n\n"
        
        if not results['pre_top']:
            message += "⏳ Сигналів не знайдено. Чекайте наступного сканування.\n"
        else:
            for i, signal in enumerate(results['pre_top'][:5], 1):
                message += f"{i}. **{signal['symbol']}** - {signal['score']:.1%}\n"
                message += f"   📈 Зміна: {signal['change_24h']:+.1f}%\n"
                message += f"   📊 Сигнали: {', '.join(signal['signals'][:3])}\n"
                message += f"   💰 Ціна: ${signal['price']:.6f}\n\n"
        
        await update.message.reply_text(message, parse_mode='Markdown')

    # Решта команд залишаються аналогічними...
    async def settings_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        settings_msg = "⚙️ **Налаштування PRE-TOP Detection**\n\n"
        settings_msg += "🎯 **Основні параметри:**\n"
        settings_msg += f"• RSI перекупленість: {self.pump_thresholds['rsi_overbought']}+\n"
        settings_msg += f"• Мін. об'єм: {self.pump_thresholds['min_volume']:,.0f} USDT\n"
        settings_msg += f"• Price acceleration: {self.pump_thresholds['price_acceleration']}\n"
        settings_msg += f"• Volume ratio: {self.pump_thresholds['volume_ratio']}x\n\n"
        settings_msg += "📉 **Сигнали розвороту:**\n"
        settings_msg += "• RSI дивергенція\n• Великі верхні тіні\n• Order book imbalance\n• Співільнення моментуму"
        
        await update.message.reply_text(settings_msg, parse_mode='Markdown')

    async def button_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()
        
        if query.data == "scan_now":
            await self.scan_command(update, context)
        elif query.data == "settings":
            await self.settings_command(update, context)
        # ... інші обробники кнопок

    def run(self):
        print("🤖 Запуск PRE-TOP DETECT BOT...")
        print("🎯 Спеціалізація: точки розвороту після пампу")
        print("📊 Сигнали: RSI дивергенція, верхні тіні, order book imbalance")
        print("💎 Версія: 3.0 (Pre-Top Detection)")
        
        self.app.run_polling()

if __name__ == "__main__":
    TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not TOKEN:
        print("❌ Помилка: TELEGRAM_BOT_TOKEN не встановлено")
        exit(1)
    
    bot = AdvancedPumpDumpBot(TOKEN)
    bot.run()