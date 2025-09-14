import matplotlib
matplotlib.use('Agg')
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
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor
import talib
from scipy import stats
import ccxt
import aiohttp
import warnings
warnings.filterwarnings('ignore')

# Детальне налаштування логування
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot_debug.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class UltimatePumpDumpDetector:
    def __init__(self, token: str):
        self.token = token
        self.app = Application.builder().token(token).build()
        
        # Покращене підключення до бірж через CCXT
        logger.info("Ініціалізація підключення до Binance...")
        try:
            self.exchange = ccxt.binance({
                'enableRateLimit': True,
                'timeout': 30000,
                'options': {
                    'defaultType': 'spot',
                    'adjustForTimeDifference': True,
                }
            })
            logger.info("Підключення до Binance ініціалізовано")
        except Exception as e:
            logger.error(f"Помилка ініціалізації Binance: {e}")
            self.exchange = None
        
        # Розширений чорний список
        self.garbage_symbols = self._load_garbage_symbols()
        self.coin_blacklist = set()
        
        # Динамічні параметри для виявлення (знижені пороги для тестування)
        self.detection_params = {
            'volume_spike_threshold': 1.2,
            'price_acceleration_min': 0.001,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'orderbook_imbalance_min': 0.1,
            'large_order_threshold': 10000,
            'min_volume_usdt': 10000,
            'max_volume_usdt': 2000000,
            'price_change_5m_min': 0.5,
            'wick_ratio_threshold': 0.3,
            'market_cap_filter': 1000000,
            'liquidity_score_min': 0.3,
            'pump_probability_threshold': 0.3,
            'dump_probability_threshold': 0.3,
            'whale_volume_threshold': 10000,
            'volatility_spike_threshold': 1.5,
            'min_daily_change': 2.0
        }
        
        # Тривожні сигнали та історія
        self.live_signals = deque(maxlen=100)
        self.market_anomalies = defaultdict(list)
        self.performance_metrics = {
            'total_scans': 0,
            'signals_triggered': 0,
            'success_rate': 0.0,
            'avg_response_time': 0.0,
            'false_positives': 0,
            'pump_signals_detected': 0,
            'dump_signals_detected': 0
        }
        
        # Пул потоків для паралельного обчислення
        self.executor = ThreadPoolExecutor(max_workers=8)
        self.setup_handlers()
        
        # Ініціалізація даних
        self.market_data_cache = {}
        self.last_update_time = time.time()

    def _load_garbage_symbols(self):
        """Розширений список непотрібних символів"""
        base = {
            'USD', 'EUR', 'GBP', 'JPY', 'CNY', 'RUB', 'TRY', 'BRL', 'KRW', 'AUD',
            'USDC', 'USDT', 'BUSD', 'DAI', 'TUSD', 'PAX', 'UST', 'HUSD', 'GUSD',
            'PAXG', 'WBTC', 'BTCB', 'ETHB', 'BNB', 'HT', 'OKB', 'LEO', 'LINK',
            'XRP', 'ADA', 'DOT', 'DOGE', 'SHIB', 'MATIC', 'SOL', 'AVAX', 'FTM',
            'SXP', 'CHZ', 'VET', 'THETA', 'FTT', 'BTT', 'WIN', 'TRX', 'BCH',
            'LTC', 'EOS', 'XLM', 'XMR', 'XTZ', 'ZEC', 'DASH', 'ETC', 'NEO'
        }
        
        patterns = {
            'UP', 'DOWN', 'BULL', 'BEAR', 'LONG', 'SHORT', 'HALF', 'FULL',
            'HEDGE', 'DOOM', 'MOON', 'SUN', 'EARTH', 'MARS', 'PLUTO',
            '3L', '3S', '2L', '2S', '1L', '1S', '5L', '5S'
        }
        
        return base.union(patterns)

    def setup_handlers(self):
        """Оновлені обробники команд"""
        handlers = [
            CommandHandler("start", self.start_command),
            CommandHandler("scan", self.deep_scan_command),
            CommandHandler("pump_radar", self.pump_radar_command),
            CommandHandler("dump_radar", self.dump_radar_command),
            CommandHandler("liquidity_scan", self.liquidity_scan_command),
            CommandHandler("whale_watch", self.whale_watch_command),
            CommandHandler("volatility_alert", self.volatility_alert_command),
            CommandHandler("ai_risk_scan", self.ai_risk_scan_command),
            CommandHandler("settings", self.settings_command),
            CommandHandler("blacklist", self.blacklist_command),
            CommandHandler("performance", self.performance_command),
            CommandHandler("quick_scan", self.quick_scan_command),
            CommandHandler("emergency", self.emergency_scan),
            CommandHandler("debug", self.debug_command),
            CommandHandler("test", self.test_command),
            CommandHandler("test_symbol", self.test_symbol_command),
            CallbackQueryHandler(self.advanced_button_handler)
        ]
        
        for handler in handlers:
            self.app.add_handler(handler)

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Оновлене меню"""
        keyboard = [
            [InlineKeyboardButton("🚨 PUMP RADAR", callback_data="pump_radar"),
             InlineKeyboardButton("📉 DUMP RADAR", callback_data="dump_radar")],
            [InlineKeyboardButton("🐋 WHALE WATCH", callback_data="whale_watch"),
             InlineKeyboardButton("💧 LIQUIDITY SCAN", callback_data="liquidity_scan")],
            [InlineKeyboardButton("⚡ VOLATILITY ALERTS", callback_data="volatility_alerts"),
             InlineKeyboardButton("🤖 AI RISK SCAN", callback_data="ai_risk_scan")],
            [InlineKeyboardButton("🔍 DEEP SCAN", callback_data="deep_scan"),
             InlineKeyboardButton("⚡ QUICK SCAN", callback_data="quick_scan")],
            [InlineKeyboardButton("⚙️ SETTINGS", callback_data="settings"),
             InlineKeyboardButton("📈 PERFORMANCE", callback_data="performance")],
            [InlineKeyboardButton("🚫 BLACKLIST", callback_data="blacklist"),
             InlineKeyboardButton("🔄 UPDATE", callback_data="update")]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            "🤖 **ULTIMATE PUMP/DUMP DETECTOR v4.2**\n\n"
            "🎯 *Спеціалізація: виявлення маніпуляцій на топ гейнерах*\n\n"
            "✨ **Ексклюзивні фічі:**\n"
            "• 🚨 Детектор пампів на активних монетах\n"
            "• 📉 Виявлення дампів серед топ гейнерів\n"
            "• 🐋 Відстеження китів на волатильних парах\n"
            "• 💧 Аналіз ліквідності ринків\n"
            "• ⚡ Сигнали волатильності\n"
            "• 🤖 AI аналіз ризиків\n\n"
            "💎 *Фокус на монетах з найбільшим зростанням!*",
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )

    async def test_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Тестова команда для перевірки роботи"""
        logger.info("Тестова команда викликана")
        await update.message.reply_text("🧪 Тестую роботу бота...")
        
        # Тест 1: Перевірка підключення до мережі
        network_ok = await self.check_network_connection()
        await update.message.reply_text(f"📡 Мережа: {'✅' if network_ok else '❌'}")
        
        # Тест 2: Перевірка підключення до Binance
        exchange_ok = await self.check_exchange_connection()
        await update.message.reply_text(f"📊 Binance: {'✅' if exchange_ok else '❌'}")
        
        # Тест 3: Отримання топ гейнерів
        try:
            gainers = await self.get_top_gainers(limit=5)
            await update.message.reply_text(f"📈 Топ гейнери: {', '.join([s.replace('/USDT', '') for s in gainers])}")
        except Exception as e:
            await update.message.reply_text(f"📈 Топ гейнери: ❌ ({str(e)})")

    async def test_symbol_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Тестування конкретного символу"""
        try:
            if not context.args:
                await update.message.reply_text("Вкажіть символ, наприклад: /test_symbol BTC/USDT")
                return
            
            symbol = context.args[0].upper()
            if not symbol.endswith('/USDT'):
                symbol += '/USDT'
            
            await update.message.reply_text(f"🔍 Тестую символ {symbol}...")
            
            # Отримуємо всі дані
            market_data = await self.get_market_data(symbol)
            orderbook = await self.get_orderbook_depth(symbol)
            klines = await self.get_klines(symbol, '5m', 20)
            
            if not all([market_data, orderbook, klines]):
                await update.message.reply_text("❌ Не вдалося отримати дані для символу")
                return
            
            # Аналіз
            tech = self.technical_analysis(klines)
            ob_analysis = self.orderbook_analysis(orderbook)
            volume_analysis = self.volume_analysis(klines, market_data)
            
            pump_prob = self.calculate_pump_probability(tech, ob_analysis, volume_analysis)
            dump_prob = self.calculate_dump_probability(tech, ob_analysis, volume_analysis)
            
            response = (
                f"📊 **РЕЗУЛЬТАТИ ТЕСТУ ДЛЯ {symbol}:**\n\n"
                f"💰 Ціна: ${market_data['close']:.6f}\n"
                f"📈 Зміна 24h: {market_data['percentage']:.2f}%\n"
                f"📊 Об'єм: ${market_data['volume']:,.0f}\n\n"
                f"📊 **ТЕХНІЧНИЙ АНАЛІЗ:**\n"
                f"• RSI: {tech['rsi']:.1f}\n"
                f"• MACD Hist: {tech['macd_hist']:.6f}\n"
                f"• Волатильність: {tech['volatility']:.2f}%\n"
                f"• Прискорення ціни: {tech['price_acceleration']:.4f}\n\n"
                f"📊 **СТАКАН:**\n"
                f"• Imbalance: {ob_analysis['imbalance']:.3f}\n"
                f"• Великі покупки: {ob_analysis['large_bids']}\n"
                f"• Великі продажі: {ob_analysis['large_asks']}\n\n"
                f"📊 **ОБ'ЄМИ:**\n"
                f"• Спайк об'ємів: {volume_analysis['volume_spike_ratio']:.2f}x\n"
                f"• Кореляція ціна/об'єм: {volume_analysis['volume_price_correlation']:.2f}\n\n"
                f"🚨 **ЙМОВІРНОСТІ:**\n"
                f"• Pump: {pump_prob:.2%}\n"
                f"• Dump: {dump_prob:.2%}\n"
            )
            
            await update.message.reply_text(response, parse_mode='Markdown')
            
        except Exception as e:
            await update.message.reply_text(f"❌ Помилка тесту: {e}")

    async def check_network_connection(self) -> bool:
        """Перевірка мережевого з'єднання"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get('https://api.binance.com/api/v3/ping', timeout=10) as response:
                    return response.status == 200
        except Exception as e:
            logger.error(f"Помилка мережі: {e}")
            return False

    async def check_exchange_connection(self) -> bool:
        """Перевірка підключення до біржі"""
        if not self.exchange:
            logger.error("Біржа не ініціалізована")
            return False
        
        try:
            # Проста перевірка пінга
            await asyncio.get_event_loop().run_in_executor(
                self.executor, lambda: self.exchange.fetch_status()
            )
            return True
        except Exception as e:
            logger.error(f"Помилка підключення до біржі: {e}")
            return False

    async def fetch_ticker_async(self, symbol: str):
        """Асинхронне отримання ticker даних"""
        try:
            return await asyncio.get_event_loop().run_in_executor(
                self.executor, lambda: self.exchange.fetch_ticker(symbol)
            )
        except Exception as e:
            logger.error(f"Помилка отримання ticker для {symbol}: {e}")
            return None

    async def fetch_order_book_async(self, symbol: str, limit: int = 20):
        """Асинхронне отримання стакану"""
        try:
            return await asyncio.get_event_loop().run_in_executor(
                self.executor, lambda: self.exchange.fetch_order_book(symbol, limit)
            )
        except Exception as e:
            logger.error(f"Помилка отримання стакану для {symbol}: {e}")
            return {'bids': [], 'asks': [], 'symbol': symbol}

    async def fetch_ohlcv_async(self, symbol: str, timeframe: str = '5m', limit: int = 50):
        """Асинхронне отримання історичних даних"""
        try:
            return await asyncio.get_event_loop().run_in_executor(
                self.executor, lambda: self.exchange.fetch_ohlcv(symbol, timeframe, limit)
            )
        except Exception as e:
            logger.error(f"Помилка отримання ohlcv для {symbol}: {e}")
            return []

    async def get_market_data(self, symbol: str) -> Optional[Dict]:
        """Отримання ринкових даних"""
        logger.debug(f"Отримання даних для {symbol}")
        try:
            if not self.exchange:
                logger.error("Біржа не ініціалізована")
                return None
                
            ticker = await self.fetch_ticker_async(symbol)
            if not ticker:
                return None
                
            result = self.parse_ticker_data(ticker, symbol)
            logger.debug(f"Дані отримані для {symbol}: {result['close']}")
            return result
            
        except Exception as e:
            logger.error(f"Не вдалося отримати дані для {symbol}: {e}")
            return None

    def parse_ticker_data(self, ticker: Dict, symbol: str) -> Dict:
        """Парсинг даних ticker"""
        try:
            return {
                'symbol': symbol,
                'open': float(ticker.get('open', 0)),
                'high': float(ticker.get('high', 0)),
                'low': float(ticker.get('low', 0)),
                'close': float(ticker.get('last', ticker.get('close', 0))),
                'volume': float(ticker.get('quoteVolume', 0)),
                'percentage': float(ticker.get('percentage', 0))
            }
        except Exception as e:
            logger.error(f"Помилка парсингу ticker для {symbol}: {e}")
            return {
                'symbol': symbol,
                'open': 0, 'high': 0, 'low': 0, 'close': 0, 'volume': 0, 'percentage': 0
            }

    async def get_orderbook_depth(self, symbol: str, limit: int = 20) -> Optional[Dict]:
        """Отримання глибини ринку"""
        logger.debug(f"Отримання стакану для {symbol}")
        try:
            if not self.exchange:
                return {'bids': [], 'asks': [], 'symbol': symbol}
                
            orderbook = await self.fetch_order_book_async(symbol, limit)
            orderbook['symbol'] = symbol
            return orderbook
        except Exception as e:
            logger.error(f"Помилка отримання стакану для {symbol}: {e}")
            return {'bids': [], 'asks': [], 'symbol': symbol}

    async def get_klines(self, symbol: str, timeframe: str = '5m', limit: int = 50) -> List:
        """Отримання історичних даних"""
        logger.debug(f"Отримання klines для {symbol}")
        try:
            if not self.exchange:
                return []
                
            klines = await self.fetch_ohlcv_async(symbol, timeframe, limit)
            return klines
        except Exception as e:
            logger.error(f"Помилка отримання klines для {symbol}: {e}")
            return []

    async def get_top_gainers(self, limit: int = 50) -> List[str]:
        """Отримання топ гейнерів (спрощена версія)"""
        try:
            if not self.exchange:
                return self.get_fallback_symbols(limit)
            
            # Отримуємо всі tickers
            tickers = await asyncio.get_event_loop().run_in_executor(
                self.executor, lambda: self.exchange.fetch_tickers()
            )
            
            # Фільтруємо USDT пари з достатнім об'ємом
            usdt_pairs = []
            for symbol, ticker in tickers.items():
                if (symbol.endswith('/USDT') and 
                    ticker.get('quoteVolume', 0) > self.detection_params['min_volume_usdt'] and
                    not self.is_garbage_symbol(symbol)):
                    usdt_pairs.append((symbol, ticker.get('percentage', 0)))
            
            # Сортуємо за зміною ціни
            usdt_pairs.sort(key=lambda x: abs(x[1]), reverse=True)
            
            return [pair[0] for pair in usdt_pairs[:limit]]
            
        except Exception as e:
            logger.error(f"Помилка отримання топ гейнерів: {e}")
            return self.get_fallback_symbols(limit)

    async def get_high_volume_symbols(self, limit: int = 30) -> List[str]:
        """Отримання монет з високим об'ємом (для whale watching)"""
        try:
            if not self.exchange:
                return self.get_fallback_symbols(limit)
                
            await asyncio.get_event_loop().run_in_executor(
                self.executor, self.exchange.load_markets
            )
            
            usdt_pairs = [
                symbol for symbol, market in self.exchange.markets.items()
                if symbol.endswith('/USDT') and market.get('active', False)
                and market.get('quoteVolume', 0) > self.detection_params['min_volume_usdt']
                and not self.is_garbage_symbol(symbol)
            ]
            
            # Сортуємо за об'ємом
            usdt_pairs.sort(key=lambda x: self.exchange.markets[x].get('quoteVolume', 0), reverse=True)
            return usdt_pairs[:limit]
            
        except Exception as e:
            logger.error(f"Помилка отримання high volume symbols: {e}")
            return self.get_fallback_symbols(5)

    def get_fallback_symbols(self, limit: int) -> List[str]:
        """Резервний список популярних символів"""
        popular_symbols = [
            'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT',
            'ADA/USDT', 'DOGE/USDT', 'MATIC/USDT', 'DOT/USDT', 'LTC/USDT',
            'AVAX/USDT', 'LINK/USDT', 'ATOM/USDT', 'XMR/USDT', 'ETC/USDT'
        ]
        return popular_symbols[:limit]

    def is_garbage_symbol(self, symbol: str) -> bool:
        """Перевірка чи символ є непотрібним"""
        try:
            symbol_clean = symbol.upper().replace('/USDT', '')
            
            # Головні криптовалюти не є сміттям
            major_coins = {'BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'ADA', 'DOGE', 
                          'MATIC', 'DOT', 'LTC', 'AVAX', 'LINK', 'ATOM'}
            if symbol_clean in major_coins:
                return False
                
            if symbol_clean in self.garbage_symbols:
                return True
            
            if len(symbol_clean) > 10:
                return True
            
            if any(char.isdigit() for char in symbol_clean):
                return True
                
            garbage_patterns = ['UP', 'DOWN', 'BULL', 'BEAR', 'LONG', 'SHORT', 
                               'HEDGE', 'DOOM', 'MOON', '3L', '3S', '2L', '2S']
            if any(pattern in symbol_clean for pattern in garbage_patterns):
                return True
                
            # Фільтр для дуже дивних назв
            if re.match(r'^[0-9]+$', symbol_clean):
                return True
                
            return False
        except:
            return False

    def technical_analysis(self, klines: List) -> Dict:
        """Технічний аналіз"""
        try:
            if len(klines) < 10:
                return {'rsi': 50, 'macd_hist': 0, 'volatility': 0, 'price_acceleration': 0, 'trend_strength': 0.5}
            
            closes = np.array([float(k[4]) for k in klines])
            highs = np.array([float(k[2]) for k in klines])
            lows = np.array([float(k[3]) for k in klines])
            
            # RSI
            rsi = talib.RSI(closes, timeperiod=14)[-1] if len(closes) >= 14 else 50
            
            # MACD
            macd, macd_signal, _ = talib.MACD(closes)
            macd_hist = (macd - macd_signal)[-1] if len(macd) > 0 else 0
            
            # Волотильність
            volatility = talib.ATR(highs, lows, closes, timeperiod=14)[-1] / closes[-1] * 100 if len(closes) >= 14 else 0
            
            # Прискорення ціни
            if len(closes) >= 6:
                price_acceleration = np.polyfit(range(6), closes[-6:], 1)[0] / closes[-6] * 100
            else:
                price_acceleration = 0
            
            # Сила тренду
            trend_strength = self.calculate_trend_strength(closes)
            
            return {
                'rsi': round(rsi, 2),
                'macd_hist': round(macd_hist, 6),
                'volatility': round(volatility, 2),
                'price_acceleration': round(price_acceleration, 4),
                'trend_strength': trend_strength
            }
        except Exception as e:
            logger.error(f"Помилка технічного аналізу: {e}")
            return {'rsi': 50, 'macd_hist': 0, 'volatility': 0, 'price_acceleration': 0, 'trend_strength': 0.5}

    def orderbook_analysis(self, orderbook: Dict) -> Dict:
        """Аналіз стану ордерів"""
        try:
            bids = orderbook.get('bids', [])
            asks = orderbook.get('asks', [])
            
            total_bid = sum(float(bid[1]) for bid in bids[:20])
            total_ask = sum(float(ask[1]) for ask in asks[:20])
            
            imbalance = (total_bid - total_ask) / (total_bid + total_ask) if (total_bid + total_ask) > 0 else 0
            
            large_bids = sum(1 for bid in bids if float(bid[0]) * float(bid[1]) > self.detection_params['large_order_threshold'])
            large_asks = sum(1 for ask in asks if float(ask[0]) * float(ask[1]) > self.detection_params['large_order_threshold'])
            
            return {
                'imbalance': round(imbalance, 4),
                'large_bids': large_bids,
                'large_asks': large_asks,
                'total_bid_volume': total_bid,
                'total_ask_volume': total_ask
            }
        except Exception as e:
            logger.error(f"Помилка аналізу orderbook: {e}")
            return {'imbalance': 0, 'large_bids': 0, 'large_asks': 0, 'total_bid_volume': 0, 'total_ask_volume': 0}

    def volume_analysis(self, klines: List, market_data: Dict) -> Dict:
        """Аналіз об'ємів торгів"""
        try:
            if len(klines) < 10:
                return {
                    'volume_spike_ratio': 1.0,
                    'volume_price_correlation': 0,
                    'current_volume': market_data.get('volume', 0),
                    'average_volume': market_data.get('volume', 0)
                }
            
            volumes = np.array([float(k[5]) for k in klines])
            closes = np.array([float(k[4]) for k in klines])
            
            # Спайк об'ємів
            avg_volume = np.mean(volumes[:-5]) if len(volumes) > 5 else volumes[0]
            current_volume = volumes[-1] if len(volumes) > 0 else 0
            volume_spike = current_volume / avg_volume if avg_volume > 0 else 1
            
            # Кореляція ціна-об'єм
            if len(closes) > 10:
                price_changes = np.diff(closes[-10:])
                volume_changes = np.diff(volumes[-10:])
                if len(price_changes) == len(volume_changes) and len(price_changes) > 1:
                    correlation = np.corrcoef(price_changes, volume_changes)[0, 1]
                else:
                    correlation = 0
            else:
                correlation = 0
            
            return {
                'volume_spike_ratio': round(volume_spike, 2),
                'volume_price_correlation': round(correlation, 2),
                'current_volume': current_volume,
                'average_volume': avg_volume
            }
        except Exception as e:
            logger.error(f"Помилка аналізу об'ємів: {e}")
            return {'volume_spike_ratio': 1, 'volume_price_correlation': 0, 'current_volume': 0, 'average_volume': 0}

    def calculate_trend_strength(self, prices: np.ndarray) -> float:
        """Розрахунок сили тренду"""
        if len(prices) < 10:
            return 0.5
        
        try:
            x = np.arange(len(prices))
            slope, _, r_value, _, _ = stats.linregress(x, prices)
            trend_strength = abs(r_value ** 2)
            direction = 1 if slope > 0 else -1
            return round(trend_strength * direction, 3)
        except:
            return 0.5

    def calculate_price_change(self, klines: List, minutes: int) -> float:
        """Розрахунок зміни ціни за вказаний період"""
        if len(klines) < minutes + 1:
            return 0.0
        
        try:
            current_price = float(klines[-1][4])
            past_price = float(klines[-minutes-1][4])
            return ((current_price - past_price) / past_price) * 100 if past_price != 0 else 0.0
        except:
            return 0.0

    def calculate_pump_probability(self, tech: Dict, orderbook: Dict, volume: Dict) -> float:
        """Розрахунок ймовірності пампу"""
        try:
            weights = {
                'rsi': 0.15,
                'volume_spike': 0.25,
                'ob_imbalance': 0.20,
                'price_accel': 0.20,
                'whale_orders': 0.20
            }
            
            score = (
                (1.0 if tech['rsi'] < 40 else 0.6 if tech['rsi'] < 50 else 0.3) * weights['rsi'] +
                min(volume['volume_spike_ratio'] / 3.0, 1.0) * weights['volume_spike'] +
                (orderbook['imbalance'] + 1.0) / 2.0 * weights['ob_imbalance'] +
                min(abs(tech['price_acceleration']) / 0.01, 1.0) * weights['price_accel'] +
                min(orderbook['large_bids'] / 5.0, 1.0) * weights['whale_orders']
            )
            
            return round(score, 4)
        except:
            return 0.3

    def calculate_dump_probability(self, tech: Dict, orderbook: Dict, volume: Dict) -> float:
        """Розрахунок ймовірності дампу"""
        try:
            weights = {
                'rsi': 0.25,
                'volume_divergence': 0.20,
                'ob_imbalance': 0.20,
                'whale_sells': 0.20,
                'volatility': 0.15
            }
            
            score = (
                (1.0 if tech['rsi'] > 70 else 0.6 if tech['rsi'] > 60 else 0.3) * weights['rsi'] +
                (1.0 - min(volume['volume_price_correlation'], 1.0)) * weights['volume_divergence'] +
                (1.0 - (orderbook['imbalance'] + 1.0) / 2.0) * weights['ob_imbalance'] +
                min(orderbook['large_asks'] / 5.0, 1.0) * weights['whale_sells'] +
                min(tech['volatility'] / 0.05, 1.0) * weights['volatility']
            )
            
            return round(score, 4)
        except:
            return 0.3

    def calculate_pump_confidence(self, tech: Dict, orderbook: Dict, market_data: Dict) -> float:
        """Розрахунок впевненості в пампі"""
        try:
            confidence = 0
            
            if tech['rsi'] < 45:
                confidence += 25
            elif tech['rsi'] < 55:
                confidence += 15
            
            if orderbook['imbalance'] > 0.2:
                confidence += 30
            elif orderbook['imbalance'] > 0.1:
                confidence += 15
            
            if orderbook['large_bids'] >= 3:
                confidence += 20
            elif orderbook['large_bids'] >= 1:
                confidence += 10
            
            if tech['price_acceleration'] > 0.005:
                confidence += 25
            
            return min(confidence, 100)
        except:
            return 0

    def calculate_dump_confidence(self, tech: Dict, orderbook: Dict, market_data: Dict) -> float:
        """Розрахунок впевненості в дампі"""
        try:
            confidence = 0
            
            if tech['rsi'] > 75:
                confidence += 30
            elif tech['rsi'] > 65:
                confidence += 20
            
            if orderbook['imbalance'] < -0.2:
                confidence += 25
            elif orderbook['imbalance'] < -0.1:
                confidence += 15
            
            if orderbook['large_asks'] >= 3:
                confidence += 25
            elif orderbook['large_asks'] >= 1:
                confidence += 15
            
            if tech['volatility'] > 8:
                confidence += 20
            
            return min(confidence, 100)
        except:
            return 0

    def analyze_large_orders(self, orderbook: Dict) -> List[Dict]:
        """Аналіз великих ордерів"""
        large_orders = []
        threshold = self.detection_params['large_order_threshold']
        
        try:
            for bid in orderbook.get('bids', [])[:10]:
                price, amount = float(bid[0]), float(bid[1])
                order_size = price * amount
                
                if order_size > threshold:
                    large_orders.append({
                        'symbol': orderbook.get('symbol', 'UNKNOWN').replace('/USDT', ''),
                        'order_size': order_size,
                        'is_buy': True,
                        'price': price,
                        'amount': amount,
                        'market_impact': (amount / (amount + orderbook.get('total_bid_volume', 1))) * 100,
                        'timestamp': datetime.now().strftime('%H:%M:%S')
                    })
            
            for ask in orderbook.get('asks', [])[:10]:
                price, amount = float(ask[0]), float(ask[1])
                order_size = price * amount
                
                if order_size > threshold:
                    large_orders.append({
                        'symbol': orderbook.get('symbol', 'UNKNOWN').replace('/USDT', ''),
                        'order_size': order_size,
                        'is_buy': False,
                        'price': price,
                        'amount': amount,
                        'market_impact': (amount / (amount + orderbook.get('total_ask_volume', 1))) * 100,
                        'timestamp': datetime.now().strftime('%H:%M:%S')
                    })
        except Exception as e:
            logger.error(f"Помилка аналізу великих ордерів: {e}")
        
        return large_orders

    def calculate_liquidity_score(self, orderbook: Dict) -> float:
        """Розрахунок оцінки ліквідності"""
        try:
            bid_volume = orderbook.get('total_bid_volume', 0)
            ask_volume = orderbook.get('total_ask_volume', 0)
            total_volume = bid_volume + ask_volume
            
            if total_volume == 0:
                return 0.0
            
            volume_score = min(total_volume / 1000000, 1.0)
            
            if orderbook['bids'] and orderbook['asks']:
                best_bid = float(orderbook['bids'][0][0])
                best_ask = float(orderbook['asks'][0][0])
                spread = (best_ask - best_bid) / best_bid
                spread_score = 1.0 - min(spread / 0.01, 1.0)
            else:
                spread_score = 0.5
            
            return round((volume_score * 0.6 + spread_score * 0.4), 3)
        except:
            return 0.5

    def calculate_volatility(self, klines: List) -> float:
        """Розрахунок волатильності"""
        try:
            if len(klines) < 10:
                return 0.0
                
            closes = np.array([float(k[4]) for k in klines])
            returns = np.diff(closes) / closes[:-1]
            volatility = np.std(returns) * 100 * np.sqrt(365)
            return round(volatility, 2)
        except:
            return 0.0

    def calculate_orderbook_imbalance(self, orderbook: Dict) -> float:
        """Розрахунок імбалансу orderbook"""
        try:
            bids = orderbook.get('bids', [])
            asks = orderbook.get('asks', [])
            
            total_bid = sum(float(bid[1]) for bid in bids)
            total_ask = sum(float(ask[1]) for ask in asks)
            
            if total_bid + total_ask == 0:
                return 0.0
            
            return (total_bid - total_ask) / (total_bid + total_ask)
        except:
            return 0.0

    def quick_pump_check(self, market_data: Dict, orderbook: Dict) -> bool:
        """Швидка перевірка на потенційний pump"""
        try:
            if market_data['volume'] < self.detection_params['min_volume_usdt']:
                return False
            
            imbalance = self.calculate_orderbook_imbalance(orderbook)
            if abs(imbalance) < self.detection_params['orderbook_imbalance_min']:
                return False
            
            large_bids = sum(1 for bid in orderbook.get('bids', []) 
                            if float(bid[0]) * float(bid[1]) > self.detection_params['large_order_threshold'])
            
            return large_bids >= 2 and imbalance > 0
        except:
            return False

    def format_signal_message(self, analysis: Dict, index: int) -> str:
        """Форматування повідомлення про сигнал"""
        try:
            symbol = analysis['symbol'].replace('/USDT', '')
            
            return (
                f"{index}. **{symbol}**\n"
                f"   💰 Ціна: ${analysis['price']:.6f}\n"
                f"   📊 Об'єм: ${analysis['volume_usdt']:,.0f}\n"
                f"   📈 Зміна: {analysis['percentage']:+.2f}%\n"
                f"   🚨 Pump ймовірність: {analysis['pump_probability']:.2%}\n"
                f"   📉 Dump ймовірність: {analysis['dump_probability']:.2%}\n"
                f"   📍 RSI: {analysis['technical_indicators']['rsi']:.1f}\n"
                f"   ⚖️ Imbalance: {analysis['orderbook_metrics']['imbalance']:.3f}\n\n"
            )
        except:
            return f"{index}. Помилка форматування сигналу\n\n"

    async def analyze_symbol(self, symbol: str) -> Dict:
        """Комплексний аналіз символу з детальним логуванням"""
        try:
            logger.debug(f"Початок аналізу {symbol}")
            
            if not await self.check_network_connection():
                logger.warning(f"Немає мережевого з'єднання для {symbol}")
                return {}
            
            market_data = await self.get_market_data(symbol)
            if not market_data:
                logger.warning(f"Не вдалося отримати market data для {symbol}")
                return {}
                
            logger.debug(f"Market data для {symbol}: {market_data}")
            
            if market_data['volume'] < self.detection_params['min_volume_usdt']:
                logger.debug(f"Об'єм занадто малий для {symbol}: {market_data['volume']}")
                return {}
            
            if self.is_garbage_symbol(symbol):
                logger.debug(f"Символ у сміттєвому списку: {symbol}")
                return {}
            
            orderbook = await self.get_orderbook_depth(symbol, 30)
            klines = await self.get_klines(symbol, '5m', 50)
            
            logger.debug(f"Orderbook для {symbol}: {len(orderbook.get('bids', []))} bids, {len(orderbook.get('asks', []))} asks")
            logger.debug(f"Klines для {symbol}: {len(klines)} записів")
            
            if not klines or len(klines) < 10:
                logger.warning(f"Недостатньо даних klines для {symbol}")
                return {}
            
            tech_analysis = self.technical_analysis(klines)
            ob_analysis = self.orderbook_analysis(orderbook)
            volume_analysis = self.volume_analysis(klines, market_data)
            
            logger.debug(f"Технічний аналіз для {symbol}: {tech_analysis}")
            logger.debug(f"Аналіз стакану для {symbol}: {ob_analysis}")
            logger.debug(f"Аналіз об'ємів для {symbol}: {volume_analysis}")
            
            pump_prob = self.calculate_pump_probability(tech_analysis, ob_analysis, volume_analysis)
            dump_prob = self.calculate_dump_probability(tech_analysis, ob_analysis, volume_analysis)
            
            logger.debug(f"Ймовірності для {symbol}: Pump={pump_prob}, Dump={dump_prob}")
            
            return {
                'symbol': symbol,
                'price': market_data['close'],
                'volume_usdt': market_data['volume'],
                'percentage': market_data['percentage'],
                'pump_probability': pump_prob,
                'dump_probability': dump_prob,
                'technical_indicators': tech_analysis,
                'orderbook_metrics': ob_analysis,
                'volume_metrics': volume_analysis,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Критична помилка аналізу {symbol}: {e}")
            return {}

    async def analyze_pump_potential(self, symbol: str) -> Dict:
        """Спеціалізований аналіз для пампу"""
        try:
            market_data = await self.get_market_data(symbol)
            orderbook = await self.get_orderbook_depth(symbol)
            klines = await self.get_klines(symbol, '5m', 20)
            
            if not all([market_data, orderbook, klines]):
                return {}
            
            tech = self.technical_analysis(klines)
            ob_analysis = self.orderbook_analysis(orderbook)
            
            confidence = self.calculate_pump_confidence(tech, ob_analysis, market_data)
            
            return {
                'symbol': symbol.replace('/USDT', ''),
                'pump_confidence': confidence,
                'price_change_5m': self.calculate_price_change(klines, 5),
                'volume_usdt': market_data['volume'],
                'whale_orders': ob_analysis['large_bids'],
                'price_acceleration': tech['price_acceleration']
            }
            
        except Exception as e:
            return {}

    async def analyze_dump_potential(self, symbol: str) -> Dict:
        """Спеціалізований аналіз для дампу"""
        try:
            market_data = await self.get_market_data(symbol)
            orderbook = await self.get_orderbook_depth(symbol)
            klines = await self.get_klines(symbol, '5m', 20)
            
            if not all([market_data, orderbook, klines]):
                return {}
            
            tech = self.technical_analysis(klines)
            ob_analysis = self.orderbook_analysis(orderbook)
            
            confidence = self.calculate_dump_confidence(tech, ob_analysis, market_data)
            
            return {
                'symbol': symbol.replace('/USDT', ''),
                'dump_confidence': confidence,
                'max_gain': market_data.get('percentage', 0),
                'whale_sells': ob_analysis['large_asks'],
                'rsi': tech['rsi']
            }
            
        except Exception as e:
            return {}

    async def detect_whale_activity(self) -> List[Dict]:
        """Виявлення активності китів"""
        try:
            symbols = await self.get_high_volume_symbols(limit=10)
            whale_activities = []
            
            for symbol in symbols:
                try:
                    orderbook = await self.get_orderbook_depth(symbol, 20)
                    if orderbook:
                        large_orders = self.analyze_large_orders(orderbook)
                        whale_activities.extend(large_orders)
                    await asyncio.sleep(0.1)
                except Exception as e:
                    continue
            
            return whale_activities[:10]
        except Exception as e:
            return []

    async def deep_scan_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Глибинне сканування топ гейнерів"""
        try:
            if not await self.check_exchange_connection():
                await update.message.reply_text("❌ Немає підключення до біржі")
                return
            
            msg = await update.message.reply_text("🔍 Запускаю сканування топ гейнерів...")
            
            # Отримуємо топ гейнери замість активних символів
            symbols = await self.get_top_gainers(limit=20)
            results = []
            
            for symbol in symbols:
                analysis = await self.analyze_symbol(symbol)
                if analysis and (analysis['pump_probability'] > 0.5 or analysis['dump_probability'] > 0.5):
                    results.append(analysis)
                await asyncio.sleep(0.2)
            
            self.performance_metrics['total_scans'] += len(symbols)
            
            if results:
                results.sort(key=lambda x: max(x['pump_probability'], x['dump_probability']), reverse=True)
                self.performance_metrics['signals_triggered'] += len(results)
                
                response = "🚨 **ЗНАЙДЕНО СИГНАЛИ НА ТОП ГЕЙНЕРАХ:**\n\n"
                for i, res in enumerate(results[:5], 1):
                    response += self.format_signal_message(res, i)
                
                response += f"📊 Знайдено {len(results)} сигналів з {len(symbols)} монет"
                await msg.edit_text(response, parse_mode='Markdown')
            else:
                await msg.edit_text("ℹ️ Сильних сигналів не знайдено. Спробуйте пізніше.")
                
        except Exception as e:
            logger.error(f"Помилка глибинного сканування: {e}")
            await update.message.reply_text("❌ Помилка сканування")

    async def pump_radar_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Детектор пампів серед топ гейнерів"""
        try:
            if not await self.check_exchange_connection():
                await update.message.reply_text("❌ Немає підключення до біржі")
                return
            
            msg = await update.message.reply_text("🚨 АКТИВУЮ PUMP RADAR для топ гейнерів...")
            
            symbols = await self.get_top_gainers(limit=15)
            pump_candidates = []
            
            for symbol in symbols:
                analysis = await self.analyze_pump_potential(symbol)
                if analysis and analysis['pump_confidence'] > 60:
                    pump_candidates.append(analysis)
                await asyncio.sleep(0.2)
            
            if pump_candidates:
                pump_candidates.sort(key=lambda x: x['pump_confidence'], reverse=True)
                self.performance_metrics['pump_signals_detected'] += len(pump_candidates)
                
                response = "🔥 **РИЗИК PUMP НА ГЕЙНЕРАХ:**\n\n"
                for i, candidate in enumerate(pump_candidates[:3], 1):
                    response += (
                        f"{i}. **{candidate['symbol']}** - {candidate['pump_confidence']}% впевненість\n"
                        f"   📈 Зміна: {candidate['price_change_5m']:.2f}% (5m)\n"
                        f"   💰 Об'єм: ${candidate['volume_usdt']:,.0f}\n"
                        f"   🐋 Китові ордери: {candidate['whale_orders']}\n"
                        f"   ⚡ Прискорення: {candidate['price_acceleration']:.4f}\n\n"
                    )
                
                await msg.edit_text(response, parse_mode='Markdown')
            else:
                await msg.edit_text("✅ Немає pump-сигналів серед топ гейнерів")
                
        except Exception as e:
            await update.message.reply_text("❌ Помилка pump radar")

    async def dump_radar_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Детектор дампів серед топ гейнерів"""
        try:
            if not await self.check_exchange_connection():
                await update.message.reply_text("❌ Немає підключення до біржі")
                return
            
            msg = await update.message.reply_text("📉 АКТИВУЮ DUMP RADAR для топ гейнерів...")
            
            symbols = await self.get_top_gainers(limit=15)
            dump_candidates = []
            
            for symbol in symbols:
                analysis = await self.analyze_dump_potential(symbol)
                if analysis and analysis['dump_confidence'] > 55:
                    dump_candidates.append(analysis)
                await asyncio.sleep(0.2)
            
            if dump_candidates:
                dump_candidates.sort(key=lambda x: x['dump_confidence'], reverse=True)
                self.performance_metrics['dump_signals_detected'] += len(dump_candidates)
                
                response = "⚠️ **РИЗИК DUMP НА ГЕЙНЕРАХ:**\n\n"
                for i, candidate in enumerate(dump_candidates[:3], 1):
                    response += (
                        f"{i}. **{candidate['symbol']}** - {candidate['dump_confidence']}% впевненість\n"
                        f"   📉 Макс. зміна: {candidate['max_gain']:.2f}%\n"
                        f"   🐋 Китові продажі: {candidate['whale_sells']}\n"
                        f"   📍 RSI: {candidate['rsi']:.1f}\n\n"
                    )
                
                await msg.edit_text(response, parse_mode='Markdown')
            else:
                await msg.edit_text("✅ Немає dump-сигналів серед топ гейнерів")
                
        except Exception as e:
            await update.message.reply_text("❌ Помилка dump radar")

    async def whale_watch_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Моніторинг китів"""
        try:
            if not await self.check_exchange_connection():
                await update.message.reply_text("❌ Немає підключення до біржі")
                return
            
            msg = await update.message.reply_text("🐋 ВІДСТЕЖУЮ КИТІВ...")
            
            whale_activity = await self.detect_whale_activity()
            
            if whale_activity:
                response = "🐋 **АКТИВНІСТЬ КИТІВ:**\n\n"
                for i, activity in enumerate(whale_activity[:5], 1):
                    response += (
                        f"{i}. **{activity['symbol']}**\n"
                        f"   💰 Розмір ордера: ${activity['order_size']:,.0f}\n"
                        f"   📊 Тип: {'КУПІВЛЯ' if activity['is_buy'] else 'ПРОДАЖ'}\n"
                        f"   ⚖️ Вплив на ринок: {activity['market_impact']:.2f}%\n"
                        f"   🕒 Час: {activity['timestamp']}\n\n"
                    )
                
                await msg.edit_text(response, parse_mode='Markdown')
            else:
                await msg.edit_text("✅ Значної активності китів не виявлено")
                
        except Exception as e:
            await update.message.reply_text("❌ Помилка моніторингу китів")

    async def liquidity_scan_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Сканування ліквідності"""
        try:
            if not await self.check_exchange_connection():
                await update.message.reply_text("❌ Немає підключення до біржі")
                return
            
            msg = await update.message.reply_text("💧 АНАЛІЗУЮ ЛІКВІДНІСТЬ...")
            
            symbols = await self.get_top_gainers(limit=15)
            liquidity_data = []
            
            for symbol in symbols:
                orderbook = await self.get_orderbook_depth(symbol)
                if orderbook:
                    liquidity_score = self.calculate_liquidity_score(orderbook)
                    liquidity_data.append({
                        'symbol': symbol.replace('/USDT', ''),
                        'liquidity_score': liquidity_score
                    })
                await asyncio.sleep(0.1)
            
            if liquidity_data:
                liquidity_data.sort(key=lambda x: x['liquidity_score'], reverse=True)
                
                response = "💧 **ТОП ЛІКВІДНІСТЬ СЕРЕД ГЕЙНЕРІВ:**\n\n"
                for i, data in enumerate(liquidity_data[:5], 1):
                    response += f"{i}. **{data['symbol']}** - Score: {data['liquidity_score']:.3f}\n\n"
                
                await msg.edit_text(response, parse_mode='Markdown')
            else:
                await msg.edit_text("❌ Не вдалося отримати дані ліквідності")
                
        except Exception as e:
            await update.message.reply_text("❌ Помилка аналізу ліквідності")

    async def volatility_alert_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Сигнали волатильності"""
        try:
            if not await self.check_exchange_connection():
                await update.message.reply_text("❌ Немає підключення до біржі")
                return
            
            msg = await update.message.reply_text("⚡ ШУКАЮ ВОЛАТИЛЬНІСТЬ...")
            
            symbols = await self.get_top_gainers(limit=15)
            volatile_symbols = []
            
            for symbol in symbols:
                klines = await self.get_klines(symbol, '5m', 20)
                if klines:
                    volatility = self.calculate_volatility(klines)
                    if volatility > self.detection_params['volatility_spike_threshold']:
                        volatile_symbols.append({
                            'symbol': symbol.replace('/USDT', ''),
                            'volatility': volatility,
                            'price': float(klines[-1][4]) if klines else 0
                        })
                await asyncio.sleep(0.1)
            
            if volatile_symbols:
                volatile_symbols.sort(key=lambda x: x['volatility'], reverse=True)
                
                response = "⚡ **ВИСОКА ВОЛАТИЛЬНІСТЬ СЕРЕД ГЕЙНЕРАХ:**\n\n"
                for i, data in enumerate(volatile_symbols[:5], 1):
                    response += (
                        f"{i}. **{data['symbol']}** - Волатильність: {data['volatility']:.2f}%\n"
                        f"   💰 Ціна: ${data['price']:.6f}\n\n"
                    )
                
                await msg.edit_text(response, parse_mode='Markdown')
            else:
                await msg.edit_text("✅ Волатильність в межах норми")
                
        except Exception as e:
            await update.message.reply_text("❌ Помилка аналізу волатильності")

    async def ai_risk_scan_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """AI аналіз ризиків"""
        try:
            if not await self.check_exchange_connection():
                await update.message.reply_text("❌ Немає підключення до біржі")
                return
            
            msg = await update.message.reply_text("🤖 AI АНАЛІЗ РИЗИКІВ...")
            
            symbols = await self.get_top_gainers(limit=10)
            risk_assessments = []
            
            for symbol in symbols:
                try:
                    analysis = await self.analyze_symbol(symbol)
                    if analysis:
                        risk_score = self.calculate_ai_risk_score(analysis)
                        risk_assessments.append({
                            'symbol': symbol.replace('/USDT', ''),
                            'risk_score': risk_score,
                            'pump_prob': analysis['pump_probability'],
                            'dump_prob': analysis['dump_probability'],
                            'price': analysis['price'],
                            'daily_change': analysis['percentage']
                        })
                    await asyncio.sleep(0.1)
                except:
                    continue
            
            if risk_assessments:
                risk_assessments.sort(key=lambda x: x['risk_score'], reverse=True)
                
                response = "🤖 **AI РЕЙТИНГ РИЗИКІВ НА ГЕЙНЕРАХ:**\n\n"
                for i, risk in enumerate(risk_assessments[:5], 1):
                    risk_level = "🔴 ВИСОКИЙ" if risk['risk_score'] > 70 else "🟡 СЕРЕДНІЙ" if risk['risk_score'] > 40 else "🟢 НИЗЬКИЙ"
                    response += (
                        f"{i}. **{risk['symbol']}** - {risk_level}\n"
                        f"   📊 Ризик: {risk['risk_score']}%\n"
                        f"   📈 Добова зміна: {risk['daily_change']:+.2f}%\n"
                        f"   🚨 Pump: {risk['pump_prob']:.2%}\n"
                        f"   📉 Dump: {risk['dump_prob']:.2%}\n"
                        f"   💰 Ціна: ${risk['price']:.6f}\n\n"
                    )
                
                await msg.edit_text(response, parse_mode='Markdown')
            else:
                await msg.edit_text("✅ Ризики в межах норми")
                
        except Exception as e:
            await update.message.reply_text("❌ Помилка AI аналізу")

    def calculate_ai_risk_score(self, analysis: Dict) -> float:
        """Розрахунок AI оцінки ризику"""
        try:
            pump_prob = analysis['pump_probability']
            dump_prob = analysis['dump_probability']
            volatility = analysis['technical_indicators']['volatility']
            imbalance = abs(analysis['orderbook_metrics']['imbalance'])
            daily_change = abs(analysis['percentage'])
            
            risk_score = (
                pump_prob * 0.25 +
                dump_prob * 0.25 +
                min(volatility / 10, 1.0) * 0.20 +
                min(imbalance / 0.5, 1.0) * 0.15 +
                min(daily_change / 20, 1.0) * 0.15
            ) * 100
            
            return min(round(risk_score, 1), 100)
        except:
            return 50.0

    async def quick_scan_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Швидке сканування"""
        try:
            if not await self.check_exchange_connection():
                await update.message.reply_text("❌ Немає підключення до біржі")
                return
            
            msg = await update.message.reply_text("⚡ ШВИДКЕ СКАНУВАННЯ ГЕЙНЕРІВ...")
            
            symbols = await self.get_top_gainers(limit=10)
            quick_signals = []
            
            for symbol in symbols:
                market_data = await self.get_market_data(symbol)
                orderbook = await self.get_orderbook_depth(symbol)
                
                if market_data and orderbook:
                    is_potential = self.quick_pump_check(market_data, orderbook)
                    if is_potential:
                        quick_signals.append({
                            'symbol': symbol.replace('/USDT', ''),
                            'price': market_data['close'],
                            'volume': market_data['volume'],
                            'change': market_data['percentage'],
                            'imbalance': self.calculate_orderbook_imbalance(orderbook)
                        })
                await asyncio.sleep(0.1)
            
            if quick_signals:
                response = "⚡ **ШВИДКІ СИГНАЛИ НА ГЕЙНЕРАХ:**\n\n"
                for i, signal in enumerate(quick_signals[:5], 1):
                    response += (
                        f"{i}. **{signal['symbol']}**\n"
                        f"   💰 Ціна: ${signal['price']:.6f}\n"
                        f"   📈 Зміна: {signal['change']:.2f}%\n"
                        f"   📊 Об'єм: ${signal['volume']:,.0f}\n"
                        f"   ⚖️ Imbalance: {signal['imbalance']:.3f}\n\n"
                    )
                
                await msg.edit_text(response, parse_mode='Markdown')
            else:
                await msg.edit_text("✅ Швидких сигналів не знайдено")
                
        except Exception as e:
            await update.message.reply_text("❌ Помилка швидкого сканування")

    async def emergency_scan(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Екстрене сканування"""
        try:
            if not await self.check_exchange_connection():
                await update.message.reply_text("❌ Немає підключення до біржі")
                return
            
            msg = await update.message.reply_text("🚨 ЕКСТРЕНЕ СКАНУВАННЯ ГЕЙНЕРІВ...")
            
            symbols = await self.get_top_gainers(limit=10)
            critical_signals = []
            
            for symbol in symbols:
                market_data = await self.get_market_data(symbol)
                if market_data and abs(market_data['percentage']) > 15:
                    orderbook = await self.get_orderbook_depth(symbol, 20)
                    if orderbook:
                        imbalance = self.calculate_orderbook_imbalance(orderbook)
                        
                        critical_signals.append({
                            'symbol': symbol.replace('/USDT', ''),
                            'price': market_data['close'],
                            'change': market_data['percentage'],
                            'volume': market_data['volume'],
                            'imbalance': imbalance,
                            'is_pump': market_data['percentage'] > 0
                        })
                await asyncio.sleep(0.1)
            
            if critical_signals:
                response = "🚨 **КРИТИЧНІ ЗМІНИ НА ГЕЙНЕРАХ:**\n\n"
                for i, signal in enumerate(critical_signals, 1):
                    signal_type = "PUMP" if signal['is_pump'] else "DUMP"
                    response += (
                        f"{i}. **{signal['symbol']}** - {signal_type}\n"
                        f"   📈 Зміна: {signal['change']:+.2f}%\n"
                        f"   💰 Ціна: ${signal['price']:.6f}\n"
                        f"   ⚖️ Imbalance: {signal['imbalance']:.3f}\n\n"
                    )
                
                await msg.edit_text(response, parse_mode='Markdown')
            else:
                await msg.edit_text("✅ Критичних змін не виявлено")
                
        except Exception as e:
            await update.message.reply_text("❌ Помилка екстреного сканування")

    async def debug_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Діагностична команда"""
        try:
            network_ok = await self.check_network_connection()
            exchange_ok = await self.check_exchange_connection()
            test_symbol = 'BTC/USDT'
            
            market_data = await self.get_market_data(test_symbol)
            orderbook = await self.get_orderbook_depth(test_symbol)
            klines = await self.get_klines(test_symbol, '5m', 5)
            
            # Отримуємо топ гейнери
            gainers = await self.get_top_gainers(5)
            
            debug_info = f"""
🔧 **ДІАГНОСТИКА:**

📡 Мережа: {'✅' if network_ok else '❌'}
📊 Біржа: {'✅' if exchange_ok else '❌'}
📈 Топ гейнери: {len(gainers)} монет

💰 BTC Ціна: {market_data.get('close', 'N/A') if market_data else 'N/A'}
📊 Об'єм: {market_data.get('volume', 'N/A') if market_data else 'N/A'}
⚖️ Imbalance: {self.calculate_orderbook_imbalance(orderbook) if orderbook else 'N/A'}

📊 Топ 5 гейнерів:
"""
            for i, symbol in enumerate(gainers[:5], 1):
                debug_info += f"{i}. {symbol.replace('/USDT', '')}\n"
            
            await update.message.reply_text(debug_info, parse_mode='Markdown')
            
        except Exception as e:
            await update.message.reply_text(f"❌ Помилка діагностики: {e}")

    async def settings_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Налаштування"""
        try:
            settings_text = "⚙️ **НАЛАШТУВАННЯ:**\n\n"
            for key, value in self.detection_params.items():
                settings_text += f"• {key}: {value}\n"
            
            await update.message.reply_text(settings_text, parse_mode='Markdown')
        except Exception as e:
            await update.message.reply_text("❌ Помилка налаштувань")

    async def blacklist_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Чорний список"""
        try:
            if context.args:
                symbol = context.args[0].upper()
                if symbol in self.coin_blacklist:
                    self.coin_blacklist.remove(symbol)
                    await update.message.reply_text(f"✅ {symbol} видалено з чорного списку")
                else:
                    self.coin_blacklist.add(symbol)
                    await update.message.reply_text(f"✅ {symbol} додано до чорного списку")
            else:
                if self.coin_blacklist:
                    blacklist_text = "🚫 **ЧОРНИЙ СПИСОК:**\n" + "\n".join(self.coin_blacklist)
                else:
                    blacklist_text = "📝 Чорний список порожній"
                await update.message.reply_text(blacklist_text, parse_mode='Markdown')
                
        except Exception as e:
            await update.message.reply_text("❌ Помилка чорного списку")

    async def performance_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Статистика"""
        try:
            total = self.performance_metrics['total_scans']
            signals = self.performance_metrics['signals_triggered']
            success_rate = (signals / total * 100) if total > 0 else 0
            
            stats_text = (
                "📈 **СТАТИСТИКА:**\n\n"
                f"• Загальна кількість сканувань: {total}\n"
                f"• Знайдено сигналів: {signals}\n"
                f"• Pump сигналів: {self.performance_metrics['pump_signals_detected']}\n"
                f"• Dump сигналів: {self.performance_metrics['dump_signals_detected']}\n"
                f"• Успішність: {success_rate:.1f}%\n"
            )
            
            await update.message.reply_text(stats_text, parse_mode='Markdown')
        except Exception as e:
            await update.message.reply_text("❌ Помилка статистики")

    async def advanced_button_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обробник кнопок"""
        query = update.callback_query
        await query.answer()
        
        try:
            if query.data == "pump_radar":
                await self.pump_radar_command(query, context)
            elif query.data == "dump_radar":
                await self.dump_radar_command(query, context)
            elif query.data == "whale_watch":
                await self.whale_watch_command(query, context)
            elif query.data == "liquidity_scan":
                await self.liquidity_scan_command(query, context)
            elif query.data == "volatility_alerts":
                await self.volatility_alert_command(query, context)
            elif query.data == "ai_risk_scan":
                await self.ai_risk_scan_command(query, context)
            elif query.data == "deep_scan":
                await self.deep_scan_command(query, context)
            elif query.data == "quick_scan":
                await self.quick_scan_command(query, context)
            elif query.data == "settings":
                await self.settings_command(query, context)
            elif query.data == "performance":
                await self.performance_command(query, context)
            elif query.data == "blacklist":
                await self.blacklist_command(query, context)
            elif query.data == "update":
                await query.edit_message_text("🔄 Оновлюю дані...")
                await asyncio.sleep(1)
                await self.start_command(query, context)
                
        except Exception as e:
            await query.edit_message_text("❌ Помилка обробки запиту")

    async def run(self):
        """Запуск бота"""
        try:
            logger.info("🤖 Запускаю Ultimate Pump/Dump Detector...")
            await self.app.initialize()
            await self.app.start()
            await self.app.updater.start_polling()
            
            logger.info("✅ Бот успішно запущено! Очікую команди...")
            
            # Запускаємо фонові tasks
            asyncio.create_task(self.background_monitoring())
            
            # Головний цикл
            while True:
                await asyncio.sleep(3600)
            
        except Exception as e:
            logger.error(f"❌ Помилка запуску бота: {e}")
            raise

    async def background_monitoring(self):
        """Фоновий моніторинг ринку"""
        while True:
            try:
                # Оновлюємо кеш даних кожні 5 хвилин
                await self.update_market_data_cache()
                await asyncio.sleep(300)
                
            except Exception as e:
                logger.error(f"Помилка фонового моніторингу: {e}")
                await asyncio.sleep(60)

    async def update_market_data_cache(self):
        """Оновлення кешу ринкових даних"""
        try:
            symbols = await self.get_top_gainers(limit=10)
            for symbol in symbols:
                try:
                    market_data = await self.get_market_data(symbol)
                    if market_data:
                        self.market_data_cache[symbol] = {
                            'data': market_data,
                            'timestamp': time.time()
                        }
                    await asyncio.sleep(0.1)
                except Exception as e:
                    continue
                    
            logger.info("✅ Кеш ринкових даних оновлено")
        except Exception as e:
            logger.error(f"Помилка оновлення кешу: {e}")

    def cleanup_old_cache(self):
        """Очищення застарілого кешу"""
        current_time = time.time()
        old_keys = []
        
        for symbol, data in self.market_data_cache.items():
            if current_time - data['timestamp'] > 600:
                old_keys.append(symbol)
        
        for key in old_keys:
            del self.market_data_cache[key]
        
        if old_keys:
            logger.info(f"🧹 Очищено {len(old_keys)} застарілих записів кешу")

    def save_state(self):
        """Збереження стану бота"""
        try:
            state = {
                'coin_blacklist': list(self.coin_blacklist),
                'performance_metrics': self.performance_metrics,
                'detection_params': self.detection_params,
                'last_update': time.time()
            }
            
            with open('bot_state.json', 'w') as f:
                json.dump(state, f, indent=2)
            
            logger.info("💾 Стан бота збережено")
        except Exception as e:
            logger.error(f"❌ Помилка збереження стану: {e}")

    def load_state(self):
        """Завантаження стану бота"""
        try:
            if os.path.exists('bot_state.json'):
                with open('bot_state.json', 'r') as f:
                    state = json.load(f)
                
                self.coin_blacklist = set(state.get('coin_blacklist', []))
                self.performance_metrics.update(state.get('performance_metrics', {}))
                self.detection_params.update(state.get('detection_params', {}))
                
                logger.info("📂 Стан бота завантажено")
        except Exception as e:
            logger.error(f"❌ Помилка завантаження стану: {e}")

def main():
    """Головна функція запуску бота"""
    try:
        BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
        
        if not BOT_TOKEN:
            logger.error("❌ Будь ласка, встановіть TELEGRAM_BOT_TOKEN у змінних оточення")
            return
        
        # Створюємо та запускаємо бота
        bot = UltimatePumpDumpDetector(BOT_TOKEN)
        
        # Завантажуємо збережений стан
        bot.load_state()
        
        logger.info("🚀 Запускаю бота...")
        
        # Запускаємо бота
        bot.app.run_polling()
        
    except KeyboardInterrupt:
        logger.info("⏹️ Зупинка бота...")
        # Зберігаємо стан перед виходом
        bot.save_state()
    except Exception as e:
        logger.error(f"❌ Критична помилка: {e}")
        try:
            bot.save_state()
        except:
            pass
        raise

if __name__ == '__main__':
    # Додаткові налаштування для стабільності
    logging.getLogger('ccxt').setLevel(logging.WARNING)
    logging.getLogger('telegram').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)
    
    # Запуск головної функції
    main()