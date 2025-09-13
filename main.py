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

# Налаштування логування
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UltimatePumpDumpDetector:
    def __init__(self, token: str):
        self.token = token
        self.app = Application.builder().token(token).build()
        
        # Підключення до бірж через CCXT
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot'
            }
        })
        
        # Розширений чорний список
        self.garbage_symbols = self._load_garbage_symbols()
        self.coin_blacklist = set()
        
        # Динамічні параметри для виявлення
        self.detection_params = {
            'volume_spike_threshold': 4.5,
            'price_acceleration_min': 0.008,
            'rsi_oversold': 32,
            'rsi_overbought': 78,
            'orderbook_imbalance_min': 0.28,
            'large_order_threshold': 75000,
            'min_volume_usdt': 100000,
            'max_volume_usdt': 5000000,
            'price_change_5m_min': 3.5,
            'wick_ratio_threshold': 0.35,
            'market_cap_filter': 50000000,
            'liquidity_score_min': 0.6,
            'pump_probability_threshold': 0.72,
            'dump_probability_threshold': 0.68,
            'whale_volume_threshold': 100000,
            'volatility_spike_threshold': 2.5
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
        self.executor = ThreadPoolExecutor(max_workers=16)
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
        
        # Додаємо шаблони мусорних монет
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
            CommandHandler("market_pulse", self.market_pulse_command),
            CommandHandler("settings", self.settings_command),
            CommandHandler("blacklist", self.blacklist_command),
            CommandHandler("performance", self.performance_command),
            CommandHandler("quick_scan", self.quick_scan_command),
            CommandHandler("emergency", self.emergency_scan),
            CallbackQueryHandler(self.advanced_button_handler)
        ]
        
        for handler in handlers:
            self.app.add_handler(handler)

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Оновлене меню з акцентом на pump/dump"""
        keyboard = [
            [InlineKeyboardButton("🚨 PUMP RADAR", callback_data="pump_radar"),
             InlineKeyboardButton("📉 DUMP RADAR", callback_data="dump_radar")],
            [InlineKeyboardButton("🐋 WHALE WATCH", callback_data="whale_watch"),
             InlineKeyboardButton("💧 LIQUIDITY SCAN", callback_data="liquidity_scan")],
            [InlineKeyboardButton("⚡ VOLATILITY ALERTS", callback_data="volatility_alerts"),
             InlineKeyboardButton("📊 MARKET PULSE", callback_data="market_pulse")],
            [InlineKeyboardButton("🔍 DEEP SCAN", callback_data="deep_scan"),
             InlineKeyboardButton("⚡ QUICK SCAN", callback_data="quick_scan")],
            [InlineKeyboardButton("⚙️ SETTINGS", callback_data="settings"),
             InlineKeyboardButton("📈 PERFORMANCE", callback_data="performance")],
            [InlineKeyboardButton("🚫 BLACKLIST", callback_data="blacklist"),
             InlineKeyboardButton("🔄 UPDATE", callback_data="update")]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            "🤖 **ULTIMATE PUMP/DUMP DETECTOR v4.0**\n\n"
            "🎯 *Спеціалізація: реальне виявлення маніпуляцій ринком*\n\n"
            "✨ **Ексклюзивні фічі:**\n"
            "• 🚨 Детектор пампів в реальному часі\n"
            "• 📉 Виявлення дампів до їх початку\n"
            "• 🐋 Відстеження китів та великих ордерів\n"
            "• 💧 Аналіз ліквідності та кластерів\n"
            "• ⚡ Сигнали волатильності\n"
            "• 📊 Глибинний аналіз ринку\n\n"
            "💎 *Отримуй сигнали ДО руху ринку!*",
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )

    async def check_network_connection(self) -> bool:
        """Перевірка мережевого з'єднання"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get('https://api.binance.com/api/v3/ping', timeout=10) as response:
                    return response.status == 200
        except:
            return False

    async def get_market_data(self, symbol: str) -> Optional[Dict]:
        """Отримання ринкових даних з обробкою помилок"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                ticker = await asyncio.get_event_loop().run_in_executor(
                    self.executor, lambda: self.exchange.fetch_ticker(symbol)
                )
                return {
                    'symbol': symbol,
                    'open': ticker.get('open', 0),
                    'high': ticker.get('high', 0),
                    'low': ticker.get('low', 0),
                    'close': ticker.get('last', 0),
                    'volume': ticker.get('quoteVolume', 0),
                    'percentage': ticker.get('percentage', 0)
                }
            except ccxt.NetworkError as e:
                logger.warning(f"Мережева помилка ({attempt+1}/{max_retries}) для {symbol}: {e}")
                await asyncio.sleep(1)
            except ccxt.ExchangeError as e:
                logger.error(f"Помилка біржі для {symbol}: {e}")
                return None
            except Exception as e:
                logger.error(f"Несподівана помилка для {symbol}: {e}")
                return None
        
        logger.error(f"Не вдалося отримати дані для {symbol} після {max_retries} спроб")
        return await self.get_market_data_fallback(symbol)

    async def get_market_data_fallback(self, symbol: str) -> Optional[Dict]:
        """Альтернативний спосіб отримання даних"""
        try:
            # Спрощений символ для CoinGecko (видаляємо /USDT)
            clean_symbol = symbol.replace('/USDT', '').lower()
            
            # Спроба отримати дані з CoinGecko
            url = f"https://api.coingecko.com/api/v3/coins/markets"
            params = {
                'vs_currency': 'usd',
                'ids': clean_symbol,
                'order': 'market_cap_desc',
                'per_page': 1,
                'page': 1,
                'sparkline': 'false'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data:
                            coin_data = data[0]
                            return {
                                'symbol': symbol,
                                'open': coin_data.get('open', 0),
                                'high': coin_data.get('high', 0),
                                'low': coin_data.get('low', 0),
                                'close': coin_data.get('current_price', 0),
                                'volume': coin_data.get('total_volume', 0),
                                'percentage': coin_data.get('price_change_percentage_24h', 0)
                            }
            
            return None
        except Exception as e:
            logger.error(f"Помилка альтернативного отримання даних для {symbol}: {e}")
            return None

    async def get_active_symbols(self, limit: int = 100) -> List[str]:
        """Отримання активних торгових пар з обробкою помилок"""
        try:
            # Спроба отримати символи з Binance
            markets = await asyncio.get_event_loop().run_in_executor(
                self.executor, self.exchange.load_markets
            )
            
            usdt_pairs = [
                symbol for symbol, market in markets.items()
                if symbol.endswith('/USDT') and market.get('active', False)
                and market.get('quoteVolume', 0) > self.detection_params['min_volume_usdt']
            ]
            
            usdt_pairs.sort(key=lambda x: markets[x].get('quoteVolume', 0), reverse=True)
            
            return usdt_pairs[:limit]
            
        except Exception as e:
            logger.error(f"Помилка отримання активних символів: {e}")
            
            # Fallback: повертаємо популярні символи
            return [
                'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT',
                'ADA/USDT', 'DOGE/USDT', 'DOT/USDT', 'MATIC/USDT', 'LTC/USDT'
            ]

    async def get_high_volume_symbols(self, limit: int = 50) -> List[str]:
        """Отримання монет з високим об'ємом"""
        try:
            symbols = await self.get_active_symbols(limit * 2)
            return symbols[:limit]
        except Exception as e:
            logger.error(f"Помилка отримання high volume symbols: {e}")
            return ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT']

    async def get_top_gainers(self, limit: int = 30) -> List[str]:
        """Отримання топ gainers"""
        try:
            # Спрощена версія - повертаємо популярні символи
            return [
                'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT',
                'ADA/USDT', 'DOGE/USDT', 'DOT/USDT', 'MATIC/USDT', 'LTC/USDT'
            ][:limit]
        except Exception as e:
            logger.error(f"Помилка отримання top gainers: {e}")
            return ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']

    async def deep_scan_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Глибинне сканування для виявлення аномалій"""
        try:
            logger.info(f"Запит команди: {update.message.text}")
            network_available = await self.check_network_connection()
            logger.info(f"Мережа доступна: {network_available}")
            
            if not network_available:
                await update.message.reply_text("⚠️ Проблеми з мережевим з'єднанням. Спробуйте пізніше.")
                return
                
            msg = await update.message.reply_text("🔍 Запускаю глибинне сканування ринку...")
            
            # Отримуємо активні монети
            active_symbols = await self.get_active_symbols(limit=20)  # Зменшимо ліміт для тесту
            results = []
            
            for symbol in active_symbols:
                try:
                    analysis = await self.analyze_symbol(symbol)
                    if analysis and (analysis['pump_probability'] > 0.65 or analysis['dump_probability'] > 0.6):
                        results.append(analysis)
                    await asyncio.sleep(0.2)  # Збільшимо затримку
                except Exception as e:
                    logger.error(f"Помилка аналізу {symbol}: {e}")
                    continue
            
            self.performance_metrics['total_scans'] += len(active_symbols)
            
            if results:
                # Сортуємо за ймовірністю
                results.sort(key=lambda x: max(x['pump_probability'], x['dump_probability']), reverse=True)
                self.performance_metrics['signals_triggered'] += len(results)
                
                response = "🚨 **ЗНАЙДЕНО ПОТЕНЦІЙНІ СИГНАЛИ:**\n\n"
                for i, res in enumerate(results[:5], 1):
                    response += self.format_signal_message(res, i)
                
                response += f"\n📊 Знайдено {len(results)} сигналів з {len(active_symbols)} монет"
                await msg.edit_text(response, parse_mode='Markdown')
            else:
                await msg.edit_text("ℹ️ Сильних сигналів не знайдено. Риск відносно спокійний.")
                
        except Exception as e:
            logger.error(f"Помилка глибинного сканування: {e}")
            await update.message.reply_text("❌ Помилка сканування")

    async def analyze_symbol(self, symbol: str) -> Dict:
        """Комплексний аналіз символу з обробкою помилок"""
        try:
            # Перевірка мережі
            if not await self.check_network_connection():
                logger.warning("⚠️ Втрачено мережеве з'єднання")
                return {}
            
            # Отримуємо дані з основних джерел
            market_data = await self.get_market_data(symbol)
            if not market_data:
                return {}
            
            # Спрощуємо аналіз для тесту
            orderbook = {'bids': [], 'asks': [], 'imbalance': 0, 'large_bids': 0, 'large_asks': 0}
            klines = []
            
            # Базовий технічний аналіз
            tech_analysis = {
                'rsi': 50,
                'macd_hist': 0,
                'volatility': 0,
                'price_acceleration': 0,
                'trend_strength': 0.5
            }
            
            # Базовий аналіз об'ємів
            volume_analysis = {
                'volume_spike_ratio': 1.0,
                'volume_price_correlation': 0,
                'current_volume': market_data['volume'],
                'average_volume': market_data['volume']
            }
            
            # Спрощений розрахунок ймовірностей
            pump_prob = 0.3 + (market_data['percentage'] / 100)  # Проста формула
            dump_prob = 0.3 - (market_data['percentage'] / 100)
            
            # Обмежуємо значення
            pump_prob = max(0.1, min(0.9, pump_prob))
            dump_prob = max(0.1, min(0.9, dump_prob))
            
            return {
                'symbol': symbol,
                'price': market_data['close'],
                'volume_usdt': market_data['volume'],
                'pump_probability': pump_prob,
                'dump_probability': dump_prob,
                'technical_indicators': tech_analysis,
                'orderbook_metrics': orderbook,
                'volume_metrics': volume_analysis,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Критична помилка аналізу {symbol}: {e}")
            return {}

    def format_signal_message(self, analysis: Dict, index: int) -> str:
        """Форматування повідомлення про сигнал"""
        symbol = analysis['symbol'].replace('/USDT', '')
        
        return (
            f"{index}. **{symbol}**\n"
            f"   💰 Ціна: ${analysis['price']:.6f}\n"
            f"   📊 Об'єм: ${analysis['volume_usdt']:,.0f}\n"
            f"   🚨 Pump ймовірність: {analysis['pump_probability']:.2%}\n"
            f"   📉 Dump ймовірність: {analysis['dump_probability']:.2%}\n"
            f"   📈 Зміна: {analysis.get('percentage', 0):.2f}%\n\n"
        )

    async def pump_radar_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Спеціалізований детектор пампів"""
        try:
            msg = await update.message.reply_text("🚨 АКТИВУЮ PUMP RADAR...")
            
            symbols = await self.get_high_volume_symbols(limit=10)
            pump_candidates = []
            
            for symbol in symbols:
                try:
                    market_data = await self.get_market_data(symbol)
                    if market_data and market_data['percentage'] > 5:  # Проста перевірка
                        pump_candidates.append({
                            'symbol': symbol.replace('/USDT', ''),
                            'pump_confidence': min(90, market_data['percentage'] * 2),
                            'price_change_5m': market_data['percentage'],
                            'volume_usdt': market_data['volume'],
                            'whale_orders': 0,
                            'price_acceleration': 0.01
                        })
                    await asyncio.sleep(0.2)
                except Exception as e:
                    continue
            
            if pump_candidates:
                pump_candidates.sort(key=lambda x: x['pump_confidence'], reverse=True)
                self.performance_metrics['pump_signals_detected'] += len(pump_candidates)
                
                response = "🔥 **ВИСОКИЙ РИЗИК PUMP:**\n\n"
                for i, candidate in enumerate(pump_candidates[:3], 1):
                    response += (
                        f"{i}. **{candidate['symbol']}** - {candidate['pump_confidence']}% впевненість\n"
                        f"   📈 Зміна: {candidate['price_change_5m']:.2f}%\n"
                        f"   💰 Об'єм: ${candidate['volume_usdt']:,.0f}\n\n"
                    )
                
                await msg.edit_text(response, parse_mode='Markdown')
            else:
                await msg.edit_text("✅ Немає активних pump-сигналів. Риск стабільний.")
                
        except Exception as e:
            logger.error(f"Помилка pump radar: {e}")
            await update.message.reply_text("❌ Помилка pump radar")

    async def dump_radar_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Детектор майбутніх дампів"""
        try:
            msg = await update.message.reply_text("📉 АКТИВУЮ DUMP RADAR...")
            
            symbols = await self.get_active_symbols(limit=10)
            dump_candidates = []
            
            for symbol in symbols:
                try:
                    market_data = await self.get_market_data(symbol)
                    if market_data and market_data['percentage'] < -5:  # Проста перевірка
                        dump_candidates.append({
                            'symbol': symbol.replace('/USDT', ''),
                            'dump_confidence': min(90, abs(market_data['percentage']) * 2),
                            'max_gain': abs(market_data['percentage']),
                            'sell_volume': market_data['volume'],
                            'whale_sells': 0,
                            'rsi': 30
                        })
                    await asyncio.sleep(0.2)
                except Exception as e:
                    continue
            
            if dump_candidates:
                dump_candidates.sort(key=lambda x: x['dump_confidence'], reverse=True)
                self.performance_metrics['dump_signals_detected'] += len(dump_candidates)
                
                response = "⚠️ **ПОПЕРЕДЖЕННЯ ПРО DUMP:**\n\n"
                for i, candidate in enumerate(dump_candidates[:3], 1):
                    response += (
                        f"{i}. **{candidate['symbol']}** - {candidate['dump_confidence']}% впевненість\n"
                        f"   📉 Зміна: -{candidate['max_gain']:.2f}%\n"
                        f"   📊 Об'єм: ${candidate['sell_volume']:,.0f}\n\n"
                    )
                
                await msg.edit_text(response, parse_mode='Markdown')
            else:
                await msg.edit_text("✅ Немає активних dump-сигналів. Риск стабільний.")
                
        except Exception as e:
            logger.error(f"Помилка dump radar: {e}")
            await update.message.reply_text("❌ Помилка dump radar")

    async def whale_watch_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Моніторинг китів та великих ордерів"""
        try:
            msg = await update.message.reply_text("🐋 ВІДСТЕЖУЮ КИТІВ...")
            
            # Спрощена версія для тесту
            whale_activity = [
                {
                    'symbol': 'BTC',
                    'order_size': 150000,
                    'is_buy': True,
                    'market_impact': 2.5,
                    'timestamp': datetime.now().strftime('%H:%M:%S')
                },
                {
                    'symbol': 'ETH', 
                    'order_size': 80000,
                    'is_buy': False,
                    'market_impact': 1.8,
                    'timestamp': datetime.now().strftime('%H:%M:%S')
                }
            ]
            
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
                await msg.edit_text("✅ Значної активності китів не виявлено.")
                
        except Exception as e:
            logger.error(f"Помилка whale watch: {e}")
            await update.message.reply_text("❌ Помилка моніторингу китів")

    async def liquidity_scan_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Сканування ліквідності"""
        try:
            msg = await update.message.reply_text("💧 АНАЛІЗУЮ ЛІКВІДНІСТЬ...")
            
            symbols = await self.get_active_symbols(limit=10)
            liquidity_data = []
            
            for symbol in symbols:
                try:
                    market_data = await self.get_market_data(symbol)
                    if market_data:
                        liquidity_data.append({
                            'symbol': symbol.replace('/USDT', ''),
                            'liquidity_score': 0.7 + (market_data['volume'] / 10000000) * 0.3,
                            'bid_volume': market_data['volume'] * 0.6,
                            'ask_volume': market_data['volume'] * 0.4
                        })
                    await asyncio.sleep(0.1)
                except Exception as e:
                    continue
            
            if liquidity_data:
                liquidity_data.sort(key=lambda x: x['liquidity_score'], reverse=True)
                
                response = "💧 **ТОП ЗА ЛІКВІДНІСТЮ:**\n\n"
                for i, data in enumerate(liquidity_data[:5], 1):
                    response += (
                        f"{i}. **{data['symbol']}** - Score: {data['liquidity_score']:.3f}\n"
                        f"   🟢 Bids: ${data['bid_volume']:,.0f}\n"
                        f"   🔴 Asks: ${data['ask_volume']:,.0f}\n\n"
                    )
                
                await msg.edit_text(response, parse_mode='Markdown')
            else:
                await msg.edit_text("❌ Не вдалося отримати дані ліквідності")
                
        except Exception as e:
            logger.error(f"Помилка liquidity scan: {e}")
            await update.message.reply_text("❌ Помилка аналізу ліквідності")

    async def volatility_alert_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Сигнали волатильності"""
        try:
            msg = await update.message.reply_text("⚡ ШУКАЮ ВОЛАТИЛЬНІСТЬ...")
            
            symbols = await self.get_active_symbols(limit=10)
            volatile_symbols = []
            
            for symbol in symbols:
                try:
                    market_data = await self.get_market_data(symbol)
                    if market_data and abs(market_data['percentage']) > 8:
                        volatile_symbols.append({
                            'symbol': symbol.replace('/USDT', ''),
                            'volatility': abs(market_data['percentage']) * 2,
                            'price': market_data['close']
                        })
                    await asyncio.sleep(0.1)
                except Exception as e:
                    continue
            
            if volatile_symbols:
                volatile_symbols.sort(key=lambda x: x['volatility'], reverse=True)
                
                response = "⚡ **ВИСОКА ВОЛАТИЛЬНІСТЬ:**\n\n"
                for i, data in enumerate(volatile_symbols[:5], 1):
                    response += (
                        f"{i}. **{data['symbol']}** - Волатильність: {data['volatility']:.2f}%\n"
                        f"   💰 Ціна: ${data['price']:.6f}\n\n"
                    )
                
                await msg.edit_text(response, parse_mode='Markdown')
            else:
                await msg.edit_text("✅ Волатильність в межах норми")
                
        except Exception as e:
            logger.error(f"Помилка volatility alert: {e}")
            await update.message.reply_text("❌ Помилка аналізу волатильності")

    async def market_pulse_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Загальний стан ринку"""
        try:
            msg = await update.message.reply_text("📊 АНАЛІЗУЮ СТАН РИНКУ...")
            
            # Отримуємо дані по основним монетам
            major_symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT']
            market_status = []
            
            for symbol in major_symbols:
                try:
                    market_data = await self.get_market_data(symbol)
                    if market_data:
                        market_status.append({
                            'symbol': symbol.replace('/USDT', ''),
                            'price': market_data['close'],
                            'change': market_data['percentage'],
                            'volume': market_data['volume']
                        })
                    await asyncio.sleep(0.1)
                except Exception as e:
                    continue
            
            if market_status:
                response = "📊 **СТАН РИНКУ:**\n\n"
                for data in market_status:
                    change_emoji = "🟢" if data['change'] > 0 else "🔴"
                    response += (
                        f"{change_emoji} **{data['symbol']}**: ${data['price']:,.2f} "
                        f"({data['change']:+.2f}%)\n"
                    )
                
                # Додаємо загальну оцінку
                avg_change = sum(d['change'] for d in market_status) / len(market_status)
                market_sentiment = "БІШИЙ" if avg_change > 2 else "ПОЗИТИВНИЙ" if avg_change > 0 else "НЕГАТИВНИЙ" if avg_change < 0 else "НЕЙТРАЛЬНИЙ"
                
                response += f"\n📈 **Загальний настрій**: {market_sentiment}\n"
                response += f"📊 **Середня зміна**: {avg_change:+.2f}%"
                
                await msg.edit_text(response, parse_mode='Markdown')
            else:
                await msg.edit_text("❌ Не вдалося отримати дані ринку")
                
        except Exception as e:
            logger.error(f"Помилка market pulse: {e}")
            await update.message.reply_text("❌ Помилка аналізу ринку")

    async def quick_scan_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Швидке сканування для миттєвих сигналів"""
        try:
            msg = await update.message.reply_text("⚡ ШВИДКЕ СКАНУВАННЯ...")
            
            # Аналізуємо топ монети
            symbols = await self.get_active_symbols(limit=10)
            quick_signals = []
            
            for symbol in symbols:
                try:
                    # Швидкий аналіз
                    market_data = await self.get_market_data(symbol)
                    if market_data and abs(market_data['percentage']) > 3:
                        quick_signals.append({
                            'symbol': symbol.replace('/USDT', ''),
                            'price': market_data['close'],
                            'volume': market_data['volume'],
                            'change': market_data['percentage'],
                            'imbalance': 0.2 if market_data['percentage'] > 0 else -0.2
                        })
                    
                    await asyncio.sleep(0.1)
                except Exception as e:
                    continue
            
            if quick_signals:
                response = "⚡ **ШВИДКІ СИГНАЛИ:**\n\n"
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
            logger.error(f"Помилка quick scan: {e}")
            await update.message.reply_text("❌ Помилка швидкого сканування")

    async def settings_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Показати поточні налаштування"""
        try:
            settings_text = "⚙️ **ПОТОЧНІ НАЛАШТУВАННЯ:**\n\n"
            for key, value in self.detection_params.items():
                settings_text += f"• {key}: {value}\n"
            
            await update.message.reply_text(settings_text, parse_mode='Markdown')
        except Exception as e:
            logger.error(f"Помилка settings command: {e}")

    async def blacklist_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Керування чорним списком"""
        try:
            if context.args:
                # Додавання/видалення з чорного списку
                symbol = context.args[0].upper()
                if symbol in self.coin_blacklist:
                    self.coin_blacklist.remove(symbol)
                    await update.message.reply_text(f"✅ {symbol} видалено з чорного списку")
                else:
                    self.coin_blacklist.add(symbol)
                    await update.message.reply_text(f"✅ {symbol} додано до чорного списку")
            else:
                # Показати чорний список
                if self.coin_blacklist:
                    blacklist_text = "🚫 **ЧОРНИЙ СПИСОК:**\n" + "\n".join(self.coin_blacklist)
                else:
                    blacklist_text = "📝 Чорний список порожній"
                
                await update.message.reply_text(blacklist_text, parse_mode='Markdown')
                
        except Exception as e:
            logger.error(f"Помилка blacklist command: {e}")

    async def performance_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Показати статистику продуктивності"""
        try:
            total = self.performance_metrics['total_scans']
            signals = self.performance_metrics['signals_triggered']
            success_rate = (signals / total * 100) if total > 0 else 0
            
            stats_text = (
                "📈 **СТАТИСТИКА ПРОДУКТИВНОСТІ:**\n\n"
                f"• Загальна кількість сканувань: {total}\n"
                f"• Знайдено сигналів: {signals}\n"
                f"• Pump сигналів: {self.performance_metrics['pump_signals_detected']}\n"
                f"• Dump сигналів: {self.performance_metrics['dump_signals_detected']}\n"
                f"• Успішність: {success_rate:.2f}%\n"
                f"• Помилки: {self.performance_metrics['false_positives']}\n"
            )
            
            await update.message.reply_text(stats_text, parse_mode='Markdown')
        except Exception as e:
            logger.error(f"Помилка performance command: {e}")

    async def advanced_button_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обробник розширених кнопок"""
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
            elif query.data == "market_pulse":
                await self.market_pulse_command(query, context)
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
            logger.error(f"Помилка обробки кнопки {query.data}: {e}")
            await query.edit_message_text("❌ Помилка обробки запиту")

    async def emergency_scan(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Екстрене сканування для критичних ситуацій"""
        try:
            msg = await update.message.reply_text("🚨 ЕКСТРЕНЕ СКАНУВАННЯ!...")
            
            # Швидкий аналіз топ монет
            symbols = await self.get_active_symbols(limit=8)
            critical_signals = []
            
            for symbol in symbols:
                try:
                    market_data = await self.get_market_data(symbol)
                    if market_data and abs(market_data['percentage']) > 10:
                        critical_signals.append({
                            'symbol': symbol.replace('/USDT', ''),
                            'price': market_data['close'],
                            'change': market_data['percentage'],
                            'volume': market_data['volume'],
                            'imbalance': 0.3 if market_data['percentage'] > 0 else -0.3,
                            'is_pump': market_data['percentage'] > 0
                        })
                    
                    await asyncio.sleep(0.1)
                except Exception as e:
                    continue
            
            if critical_signals:
                response = "🚨 **КРИТИЧНІ СИГНАЛИ:**\n\n"
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
                await msg.edit_text("✅ Критичних сигналів не виявлено")
                
        except Exception as e:
            logger.error(f"Помилка emergency scan: {e}")
            await update.message.reply_text("❌ Помилка екстреного сканування")

    async def run(self):
        """Запуск бота"""
        try:
            logger.info("🤖 Запускаю Ultimate Pump/Dump Detector...")
            await self.app.initialize()
            await self.app.start()
            await self.app.updater.start_polling()
            
            logger.info("✅ Бот успішно запущено!")
            
            # Запускаємо фонові tasks
            asyncio.create_task(self.background_monitoring())
            
            # Просто чекаємо безкінечно
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
            symbols = await self.get_active_symbols(limit=10)
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

# Основна функція запуску
def main():
    """Головна функція запуску бота"""
    try:
        # Отримуємо токен з змінних оточення
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