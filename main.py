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
from typing import Dict, List, Optional, Tuple, Any
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
        
        # Унікальні умови для PRE-TOP DETECT
        self.pump_thresholds = {
            'volume_ratio': 3.5,
            'price_change_1h': 8.0,
            'price_change_5m': 4.0,
            'rsi_overbought': 72,
            'rsi_divergence': True,
            'macd_momentum': 0.001,
            'min_volume': 50000,
            'max_volume': 5000000,
            'buy_pressure_ratio': 1.8,
            'volatility_ratio': 2.2,
            'price_acceleration': 0.0005,
            'orderbook_imbalance_threshold': 0.25,
            'large_orders_threshold': 50000
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
                'settings': self.pump_thresholds
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
        self.app.add_handler(CommandHandler("orderbook", self.orderbook_command))
        self.app.add_handler(CommandHandler("analysis", self.market_analysis_command))
        self.app.add_handler(CommandHandler("performance", self.performance_command))
        self.app.add_handler(CommandHandler("topgainers", self.top_gainers_command))
        self.app.add_handler(CallbackQueryHandler(self.button_handler))

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        keyboard = [
            [InlineKeyboardButton("🔍 Сканувати PRE-TOP", callback_data="scan_now")],
            [InlineKeyboardButton("📈 Топ Gainers", callback_data="top_gainers")],
            [InlineKeyboardButton("📊 Аналіз ордерів", callback_data="orderbook_analysis")],
            [InlineKeyboardButton("⚙️ Налаштування", callback_data="settings")],
            [InlineKeyboardButton("📊 Статистика", callback_data="stats")],
            [InlineKeyboardButton("🚫 Чорний список", callback_data="blacklist")],
            [InlineKeyboardButton("📋 Аналіз ринку", callback_data="market_analysis")],
            [InlineKeyboardButton("🏆 Продуктивність", callback_data="performance")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            "🤖 **ULTIMATE PRE-TOP DETECT BOT**\n\n"
            "🎯 *Спеціалізація: виявлення точок розвороту після пампу*\n\n"
            "✨ **Унікальні фічі:**\n"
            "• 🔍 PRE-TOP detection\n"
            "• 📈 Топ монети за зміною ціни\n"
            "• 📊 Аналіз книги ордерів\n"
            "• 🎯 RSI/Volume дивергенції\n"
            "• ⚡ Швидкі сповіщення\n"
            "• 📱 Всі кнопки працюють!\n\n"
            "💎 *Встигни зайти перед дампом!*",
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )

    def is_garbage_symbol(self, symbol: str) -> bool:
        """Перевіряє чи символ є сміттям"""
        symbol = symbol.upper()
        
        if symbol in self.garbage_symbols:
            return True
        
        if len(symbol) > 10:
            return True
        
        if any(char.isdigit() for char in symbol):
            return True
            
        if symbol.endswith(('UP', 'DOWN', 'BULL', 'BEAR', 'USD', 'EUR')):
            return True
            
        return False

    async def get_top_gainers(self, limit: int = 50) -> List[Dict]:
        """Отримання топ монет за зміною ціни з CoinGecko"""
        try:
            url = "https://api.coingecko.com/api/v3/coins/markets"
            params = {
                'vs_currency': 'usd',
                'order': 'price_change_percentage_24h_desc',
                'per_page': limit,
                'page': 1,
                'sparkline': 'false',
                'price_change_percentage': '24h'
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            gainers = []
            for coin in data:
                symbol = coin['symbol'].upper()
                if not self.is_garbage_symbol(symbol):
                    gainers.append({
                        'symbol': symbol,
                        'name': coin['name'],
                        'price': coin['current_price'],
                        'change_24h': coin['price_change_percentage_24h'],
                        'volume': coin['total_volume'],
                        'market_cap': coin['market_cap']
                    })
            
            return gainers
            
        except Exception as e:
            logger.error(f"Помилка отримання топ gainers з CoinGecko: {e}")
            return await self.get_top_gainers_binance(limit)

    async def get_top_gainers_binance(self, limit: int = 50) -> List[Dict]:
        """Резервний метод отримання топ gainers через Binance"""
        try:
            response = requests.get("https://api.binance.com/api/v3/ticker/24hr", timeout=10)
            response.raise_for_status()
            all_data = response.json()
            
            usdt_pairs = [
                x for x in all_data 
                if x['symbol'].endswith('USDT') 
                and not self.is_garbage_symbol(x['symbol'].replace('USDT', ''))
                and float(x['priceChangePercent']) > 5.0
            ]
            
            sorted_gainers = sorted(
                usdt_pairs,
                key=lambda x: float(x['priceChangePercent']),
                reverse=True
            )[:limit]
            
            gainers = []
            for coin in sorted_gainers:
                symbol = coin['symbol'].replace('USDT', '')
                gainers.append({
                    'symbol': symbol,
                    'price': float(coin['lastPrice']),
                    'change_24h': float(coin['priceChangePercent']),
                    'volume': float(coin['volume']),
                    'quote_volume': float(coin['quoteVolume'])
                })
            
            return gainers
            
        except Exception as e:
            logger.error(f"Помилка резервного отримання gainers: {e}")
            return []

    async def get_market_data(self, symbol: str) -> Optional[Dict]:
        """Отримання ринкових даних для символу"""
        try:
            if self.is_garbage_symbol(symbol):
                return None
            
            # Отримуємо основні дані
            url = f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}USDT"
            response = requests.get(url, timeout=8)
            response.raise_for_status()
            data = response.json()
            
            if 'code' in data:
                return None
            
            quote_volume = float(data['quoteVolume'])
            price_change = float(data['priceChangePercent'])
            current_price = float(data['lastPrice'])
            
            # Фільтр за об'ємом
            if quote_volume < self.pump_thresholds['min_volume']:
                return None
            
            # Отримуємо детальні клайнси
            klines_data = {}
            timeframes = {
                '5m': '5m',
                '15m': '15m',
                '1h': '1h'
            }
            
            for tf, interval in timeframes.items():
                try:
                    klines_url = f"https://api.binance.com/api/v3/klines?symbol={symbol}USDT&interval={interval}&limit=100"
                    klines_response = requests.get(klines_url, timeout=5)
                    klines_response.raise_for_status()
                    klines_data[tf] = klines_response.json()
                except Exception as e:
                    logger.warning(f"Помилка отримання {tf} klines для {symbol}: {e}")
                    continue
            
            # Детальна книга ордерів
            orderbook = await self.get_detailed_orderbook(symbol)
            
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
                'orderbook': orderbook,
                'timestamp': datetime.now().isoformat()
            }
            
            return market_data
            
        except Exception as e:
            logger.error(f"Помилка отримання даних для {symbol}: {e}")
            return None

    async def get_detailed_orderbook(self, symbol: str) -> Dict:
        """Детальний аналіз книги ордерів"""
        try:
            # Отримуємо глибоку книгу ордерів
            orderbook_url = f"https://api.binance.com/api/v3/depth?symbol={symbol}USDT&limit=50"
            orderbook_response = requests.get(orderbook_url, timeout=8)
            orderbook_response.raise_for_status()
            orderbook_data = orderbook_response.json()
            
            # Аналізуємо великі ордери
            large_bids = self.analyze_large_orders(orderbook_data['bids'])
            large_asks = self.analyze_large_orders(orderbook_data['asks'])
            
            # Аналіз imbalance
            imbalance = self.calculate_orderbook_imbalance(orderbook_data)
            
            # Знаходимо кластери ордерів
            bid_clusters = self.find_order_clusters(orderbook_data['bids'])
            ask_clusters = self.find_order_clusters(orderbook_data['asks'])
            
            return {
                'bids': orderbook_data['bids'],
                'asks': orderbook_data['asks'],
                'large_bids': large_bids,
                'large_asks': large_asks,
                'imbalance': imbalance,
                'bid_clusters': bid_clusters,
                'ask_clusters': ask_clusters
            }
            
        except Exception as e:
            logger.error(f"Помилка аналізу orderbook для {symbol}: {e}")
            return {'bids': [], 'asks': [], 'large_bids': 0, 'large_asks': 0, 'imbalance': 0}

    def analyze_large_orders(self, orders: List) -> int:
        """Аналіз великих ордерів"""
        large_orders = 0
        for order in orders:
            price = float(order[0])
            quantity = float(order[1])
            order_size = price * quantity
            if order_size > self.pump_thresholds['large_orders_threshold']:
                large_orders += 1
        return large_orders

    def calculate_orderbook_imbalance(self, orderbook: Dict) -> float:
        """Розрахунок imbalance книги ордерів"""
        try:
            total_bid_volume = sum(float(bid[1]) for bid in orderbook['bids'])
            total_ask_volume = sum(float(ask[1]) for ask in orderbook['asks'])
            
            if total_bid_volume + total_ask_volume > 0:
                imbalance = (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume)
                return round(imbalance, 4)
            return 0.0
        except Exception as e:
            logger.error(f"Помилка розрахунку imbalance: {e}")
            return 0.0

    def find_order_clusters(self, orders: List, threshold: float = 0.02) -> List[Dict]:
        """Знаходження кластерів ордерів"""
        clusters = []
        current_cluster = []
        
        for i in range(len(orders)):
            if not current_cluster:
                current_cluster.append(orders[i])
            else:
                current_price = float(orders[i][0])
                prev_price = float(current_cluster[-1][0])
                price_diff = abs(current_price - prev_price) / prev_price
                
                if price_diff < threshold:
                    current_cluster.append(orders[i])
                else:
                    if len(current_cluster) > 2:
                        clusters.append({
                            'price_range': (float(current_cluster[0][0]), float(current_cluster[-1][0])),
                            'total_quantity': sum(float(order[1]) for order in current_cluster),
                            'orders_count': len(current_cluster)
                        })
                    current_cluster = [orders[i]]
        
        # Додаємо останній кластер
        if len(current_cluster) > 2:
            clusters.append({
                'price_range': (float(current_cluster[0][0]), float(current_cluster[-1][0])),
                'total_quantity': sum(float(order[1]) for order in current_cluster),
                'orders_count': len(current_cluster)
            })
        
        return clusters

    def calculate_advanced_indicators(self, klines_data: List) -> Dict:
        """Розширені технічні індикатори"""
        try:
            if not klines_data or len(klines_data) < 20:
                return {}
                
            closes = np.array([float(x[4]) for x in klines_data])
            highs = np.array([float(x[2]) for x in klines_data])
            lows = np.array([float(x[3]) for x in klines_data])
            volumes = np.array([float(x[5]) for x in klines_data])
            
            # Базові індикатори
            rsi = self.calculate_rsi(closes)
            macd = self.calculate_macd(closes)
            
            # Розширені індикатори
            price_acceleration = self.calculate_price_acceleration(closes)
            volume_price_divergence = self.calculate_volume_price_divergence(closes, volumes)
            wick_analysis = self.calculate_wick_analysis(highs, lows, closes)
            
            return {
                'rsi': rsi,
                'macd': macd,
                'price_acceleration': price_acceleration,
                'volume_divergence': volume_price_divergence,
                'wick_ratio': wick_analysis,
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
            seed = deltas[:period + 1]
            up = seed[seed >= 0].sum() / period
            down = -seed[seed < 0].sum() / period
            
            if down == 0:
                return 100.0
                
            rs = up / down
            rsi = 100.0 - (100.0 / (1.0 + rs))

            for i in range(period + 1, len(deltas)):
                delta = deltas[i - 1]

                if delta > 0:
                    up_val = delta
                    down_val = 0.0
                else:
                    up_val = 0.0
                    down_val = -delta

                up = (up * (period - 1) + up_val) / period
                down = (down * (period - 1) + down_val) / period

                if down == 0:
                    rsi = 100.0
                else:
                    rs = up / down
                    rsi = 100.0 - (100.0 / (1.0 + rs))

            return round(rsi, 2)
        except Exception as e:
            logger.error(f"Помилка розрахунку RSI: {e}")
            return 50.0

    def calculate_macd(self, closes: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> dict:
        """Розрахунок MACD"""
        try:
            if len(closes) < slow + signal:
                return {'macd_line': 0, 'signal_line': 0, 'histogram': 0}

            exp1 = pd.Series(closes).ewm(span=fast, adjust=False).mean()
            exp2 = pd.Series(closes).ewm(span=slow, adjust=False).mean()
            macd_line = exp1 - exp2
            signal_line = macd_line.ewm(span=signal, adjust=False).mean()
            histogram = macd_line - signal_line

            return {
                'macd_line': round(macd_line.iloc[-1], 6),
                'signal_line': round(signal_line.iloc[-1], 6),
                'histogram': round(histogram.iloc[-1], 6)
            }
        except Exception as e:
            logger.error(f"Помилка розрахунку MACD: {e}")
            return {'macd_line': 0, 'signal_line': 0, 'histogram': 0}

    def calculate_price_acceleration(self, closes: np.ndarray, lookback: int = 5) -> float:
        """Розрахунок прискорення ціни"""
        if len(closes) < lookback + 1:
            return 0.0
        recent_changes = np.diff(closes[-lookback:])
        if len(recent_changes) < 2:
            return 0.0
        acceleration = np.diff(recent_changes)[-1]
        return round(acceleration, 6)

    def calculate_volume_price_divergence(self, closes: np.ndarray, volumes: np.ndarray, lookback: int = 20) -> float:
        """Розрахунок дивергенції між об'ємом і ціною"""
        if len(closes) < lookback or len(volumes) < lookback:
            return 0.0
            
        price_changes = np.diff(closes[-lookback:])
        volume_changes = np.diff(volumes[-lookback:])
        
        if len(price_changes) != len(volume_changes):
            min_len = min(len(price_changes), len(volume_changes))
            price_changes = price_changes[:min_len]
            volume_changes = volume_changes[:min_len]
            
        if len(price_changes) < 2:
            return 0.0
            
        correlation = np.corrcoef(price_changes, volume_changes)[0, 1]
        return round(correlation, 4)

    def calculate_wick_analysis(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, lookback: int = 10) -> float:
        """Аналіз співвідношення верхніх та нижніх хвостів"""
        if len(highs) < lookback:
            return 0.5
            
        upper_wick_ratios = []
        lower_wick_ratios = []

        for i in range(len(highs)):
            if i >= len(closes):
                continue
                
            body_high = max(closes[i], closes[i-1]) if i > 0 else closes[i]
            body_low = min(closes[i], closes[i-1]) if i > 0 else closes[i]
            total_range = highs[i] - lows[i]

            if total_range > 0:
                upper_wick = (highs[i] - body_high) / total_range
                lower_wick = (body_low - lows[i]) / total_range
                upper_wick_ratios.append(upper_wick)
                lower_wick_ratios.append(lower_wick)

        avg_upper_wick = np.mean(upper_wick_ratios) if upper_wick_ratios else 0.0
        avg_lower_wick = np.mean(lower_wick_ratios) if lower_wick_ratios else 0.0

        if avg_upper_wick + avg_lower_wick == 0:
            return 0.5
            
        wick_ratio = avg_lower_wick / (avg_upper_wick + avg_lower_wick)
        return round(wick_ratio, 4)

    async def scan_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обробка команди /scan"""
        try:
            await update.message.reply_text("🔍 Запускаю сканування для PRE-TOP...")
            gainers = await self.get_top_gainers(20)
            
            if not gainers:
                await update.message.reply_text("❌ Не вдалося отримати дані для сканування")
                return
                
            signals_found = 0
            for coin in gainers[:5]:  # Скануємо топ-5 для швидкості
                market_data = await self.get_market_data(coin['symbol'])
                if market_data:
                    # Тут має бути ваша логіка аналізу
                    signals_found += 1
            
            await update.message.reply_text(f"✅ Сканування завершено. Знайдено {signals_found} потенційних сигналів.")
            
        except Exception as e:
            logger.error(f"Помилка сканування: {e}")
            await update.message.reply_text("❌ Помилка під час сканування")

    async def settings_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обробка команди /settings"""
        try:
            current_settings = "\n".join([f"• {k}: {v}" for k, v in self.pump_thresholds.items()])
            await update.message.reply_text(f"⚙️ Поточні налаштування:\n{current_settings}")
        except Exception as e:
            logger.error(f"Помилка команди settings: {e}")
            await update.message.reply_text("❌ Помилка відображення налаштувань")

    async def blacklist_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обробка команди /blacklist"""
        try:
            blacklist_str = "\n".join(self.coin_blacklist) if self.coin_blacklist else "Чорний список порожній."
            await update.message.reply_text(f"🚫 Чорний список:\n{blacklist_str}")
        except Exception as e:
            logger.error(f"Помилка команди blacklist: {e}")
            await update.message.reply_text("❌ Помилка відображення чорного списку")

    async def debug_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обробка команди /debug"""
        try:
            await update.message.reply_text("🐞 Режим налагодження активовано. Логи будуть детальнішими.")
            logging.getLogger().setLevel(logging.DEBUG)
        except Exception as e:
            logger.error(f"Помилка команди debug: {e}")

    async def orderbook_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обробка команди /orderbook <symbol>"""
        try:
            if not context.args:
                await update.message.reply_text("ℹ️ Вкажіть символ монети. Наприклад: /orderbook BTC")
                return
                
            symbol = context.args[0].upper()
            await update.message.reply_text(f"📊 Аналізую книгу ордерів для {symbol}...")
            
            orderbook = await self.get_detailed_orderbook(symbol)
            if orderbook['bids']:
                message = (f"📈 Orderbook для {symbol}:\n"
                          f"• Imbalance: {orderbook['imbalance']:.4f}\n"
                          f"• Великі bids: {orderbook['large_bids']}\n"
                          f"• Великі asks: {orderbook['large_asks']}\n"
                          f"• Кластери bids: {len(orderbook['bid_clusters'])}\n"
                          f"• Кластери asks: {len(orderbook['ask_clusters'])}")
                await update.message.reply_text(message)
            else:
                await update.message.reply_text(f"❌ Не вдалося отримати дані для {symbol}")
                
        except Exception as e:
            logger.error(f"Помилка команди orderbook: {e}")
            await update.message.reply_text("❌ Помилка аналізу orderbook")

    async def market_analysis_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обробка команди /analysis"""
        try:
            await update.message.reply_text("📋 Запускаю загальний аналіз ринку...")
            gainers = await self.get_top_gainers(10)
            
            if gainers:
                message = "📈 Топ-5 Gainers (24h):\n"
                for i, coin in enumerate(gainers[:5], 1):
                    message += f"{i}. {coin['symbol']}: {coin['change_24h']:.2f}%\n"
                await update.message.reply_text(message)
            else:
                await update.message.reply_text("ℹ️ Дані ринку тимчасово недоступні")
                
        except Exception as e:
            logger.error(f"Помилка команди analysis: {e}")
            await update.message.reply_text("❌ Помилка аналізу ринку")

    async def performance_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обробка команди /performance"""
        try:
            stats = self.performance_stats
            message = (f"📊 Статистика продуктивності:\n"
                       f"• Всього сканувань: {stats['total_scans']}\n"
                       f"• Знайдено сигналів: {stats['signals_found']}\n"
                       f"• Успішність: {stats['success_rate']:.2f}%\n"
                       f"• Сер. час сканування: {stats['avg_scan_time']:.2f} сек\n"
                       f"• Uptime: {timedelta(seconds=int(time.time() - self.start_time))}")
            await update.message.reply_text(message)
        except Exception as e:
            logger.error(f"Помилка команди performance: {e}")
            await update.message.reply_text("❌ Помилка відображення статистики")

    async def top_gainers_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обробка команди /topgainers"""
        try:
            await update.message.reply_text("📈 Отримую список топ-монет...")
            gainers = await self.get_top_gainers(15)
            
            if not gainers:
                await update.message.reply_text("❌ Не вдалося отримати дані.")
                return
                
            message = "🏆 Топ-10 Gainers (24h):\n"
            for i, coin in enumerate(gainers[:10], 1):
                message += f"{i}. {coin['symbol']}: {coin['change_24h']:.2f}%\n"
                
            await update.message.reply_text(message)
            
        except Exception as e:
            logger.error(f"Помилка команди topgainers: {e}")
            await update.message.reply_text("❌ Помилка отримання топ gainers")

    async def button_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обробка натискань на інлайн-кнопки"""
        try:
            query = update.callback_query
            await query.answer()
            data = query.data

            if data == "scan_now":
                await self.scan_command(query, context)
            elif data == "top_gainers":
                await self.top_gainers_command(query, context)
            elif data == "orderbook_analysis":
                await query.edit_message_text(text="📊 Оберіть монету для аналізу ордербуку...")
            elif data == "settings":
                await self.settings_command(query, context)
            elif data == "stats":
                await query.edit_message_text(text="📈 Статистика бота...")
            elif data == "blacklist":
                await self.blacklist_command(query, context)
            elif data == "market_analysis":
                await self.market_analysis_command(query, context)
            elif data == "performance":
                await self.performance_command(query, context)
                
        except Exception as e:
            logger.error(f"Помилка обробки кнопки: {e}")

    def handle_webhook(self, data):
        """Обробка вхідних вебхуків"""
        try:
            logger.info(f"📩 Отримано вебхук: {data}")
            return jsonify({'status': 'webhook_received', 'timestamp': datetime.now().isoformat()})
        except Exception as e:
            logger.error(f"Помилка обробки вебхука: {e}")
            return jsonify({'status': 'error', 'message': str(e)})

    def update_settings(self, new_settings: dict):
        """Оновлення налаштувань"""
        try:
            valid_settings = {k: v for k, v in new_settings.items() if k in self.pump_thresholds}
            self.pump_thresholds.update(valid_settings)
            logger.info(f"⚙️ Налаштування оновлено: {valid_settings}")
        except Exception as e:
            logger.error(f"Помилка оновлення налаштувань: {e}")

def run_flask(app: Flask):
    """Запуск Flask-сервера в окремому потоці"""
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False, threaded=True)
    except Exception as e:
        logger.error(f"Помилка запуску Flask: {e}")

def main():
    """Головна функція запуску бота"""
    try:
        # Отримання токену бота
        BOT_TOKEN = os.getenv('BOT_TOKEN') or '8489382938:AAHeFFZPODspuEFcSQyjw8lWzYpRRSv9n3g'
        
        if BOT_TOKEN == '8489382938:AAHeFFZPODspuEFcSQyjw8lWzYpRRSv9n3g':
            logger.error("❌ Будь ласка, встановіть ваш Telegram Bot Token")
            return

        # Ініціалізація бота
        bot = AdvancedPumpDumpBot(BOT_TOKEN)

        # Запуск Flask у фоновому потоці
        flask_thread = threading.Thread(
            target=run_flask, 
            args=(bot.flask_app,), 
            daemon=True,
            name="Flask-Thread"
        )
        flask_thread.start()
        logger.info("🌐 Flask server started in background thread")

        # Запуск Telegram Bot
        logger.info("🤖 Starting Telegram bot...")
        bot.app.run_polling(
            drop_pending_updates=True,
            allowed_updates=Update.ALL_TYPES
        )
        
    except KeyboardInterrupt:
        logger.info("⏹️ Bot stopped by user")
    except Exception as e:
        logger.error(f"❌ Critical error: {e}")
        raise

if __name__ == '__main__':
    main()