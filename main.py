import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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
from concurrent.futures import ThreadPoolExecutor, as_completed
import talib
from scipy import stats
import ccxt
import aiohttp
import warnings
warnings.filterwarnings('ignore')

# Детальне налаштування логування
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot_debug.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class UltimatePumpDumpDetector:
    def __init__(self, token: str):
        self.token = token
        self.app = Application.builder().token(token).build()
        
        # Підключення до Binance з оптимізацією
        logger.info("Ініціалізація підключення до Binance...")
        try:
            self.exchange = ccxt.binance({
                'enableRateLimit': True,
                'timeout': 15000,
                'options': {
                    'defaultType': 'spot',
                    'adjustForTimeDifference': True,
                },
                'rateLimit': 100,
            })
            logger.info("Підключення до Binance ініціалізовано")
        except Exception as e:
            logger.error(f"Помилка ініціалізації Binance: {e}")
            self.exchange = None
        
        # Розширений чорний список
        self.garbage_symbols = self._load_garbage_symbols()
        self.coin_blacklist = set()
        
        # Оптимізовані параметри для масштабованого сканування
        self.detection_params = {
            'volume_spike_threshold': 1.8,
            'price_acceleration_min': 0.003,
            'rsi_oversold': 28,
            'rsi_overbought': 72,
            'orderbook_imbalance_min': 0.15,
            'large_order_threshold': 75000,
            'min_volume_usdt': 2000000,  # 2M USDT мінімум для якості
            'max_volume_usdt': 100000000,
            'price_change_5m_min': 1.2,
            'wick_ratio_threshold': 0.25,
            'market_cap_filter': 5000000,
            'liquidity_score_min': 0.5,
            'pump_probability_threshold': 0.65,
            'dump_probability_threshold': 0.65,
            'whale_volume_threshold': 75000,
            'volatility_spike_threshold': 2.2,
            'min_daily_change': 7.0,
            'min_price': 0.0005,
            'max_symbols_per_scan': 150  # Максимум токенів за сканування
        }
        
        # Кеш та оптимізація продуктивності
        self.market_data_cache = {}
        self.symbols_cache = []
        self.last_symbols_update = 0
        self.performance_history = deque(maxlen=1000)
        
        # Статистика продуктивності
        self.performance_metrics = {
            'total_scans': 0,
            'signals_triggered': 0,
            'success_rate': 0.0,
            'avg_scan_time': 0.0,
            'false_positives': 0,
            'pump_signals_detected': 0,
            'dump_signals_detected': 0,
            'profitable_signals': 0,
            'unprofitable_signals': 0,
            'avg_symbols_per_scan': 0
        }
        
        # Пул потоків для паралельного обчислення
        self.executor = ThreadPoolExecutor(max_workers=20)
        self.setup_handlers()
        
        # Ініціалізація та кешування
        self.last_update_time = time.time()

    def _load_garbage_symbols(self):
        """Розширений список непотрібних символів"""
        base = {
            'USD', 'EUR', 'GBP', 'JPY', 'CNY', 'RUB', 'TRY', 'BRL', 'KRW', 'AUD',
            'USDC', 'USDT', 'BUSD', 'DAI', 'TUSD', 'PAX', 'UST', 'HUSD', 'GUSD',
            'PAXG', 'WBTC', 'BTCB', 'ETHB', 'BNB', 'HT', 'PNT', 'LEO', 'LINK',
            'XRP', 'ADA', 'DOT', 'DOGE', 'SHIB', 'VIB', 'SOL', 'AVAX', 'FTM',
            'SXP', 'CHZ', 'OOKI', 'CREAM', 'FTT', 'BTT', 'WIN', 'TRX', 'BCH',
            'LTC', 'HARD', 'XLM', 'XMR', 'XTZ', 'WTC', 'DASH', 'ETC', 'BETA',
            'BURGER', 'FIRO', 'CVP', 'EPX', 'PROS', 'SLF', 'OAX', 'VIDT',
            'WING', 'CLV', 'FOR', 'VITE', 'TROY', 'SFP', 'ALPACA', 'DEXE',
            'FUN', 'TVK', 'BOND', 'TLM', 'REEF', 'TWT', 'LEVER', 'MULTI',
            'GFT', 'DREP', 'PERL', 'UFT', 'BTS', 'STMX', 'CKB', 'CHR',
            'COCOS', 'MBL', 'TCT', 'WRX', 'BEAM', 'VTHO', 'DOCK', 'WAN'
        }
        
        patterns = {
            'UP', 'DOWN', 'BULL', 'BEAR', 'LONG', 'SHORT', 'HALF', 'FULL',
            'HEDGE', 'DOOM', 'MOON', 'SUN', 'EARTH', 'MARS', 'PLUTO',
            '3L', '3S', '2L', '2S', '1L', '1S', '5L', '5S', '10L', '10S'
        }
        
        return base.union(patterns)

    def setup_handlers(self):
        """Оновлені обробники команд"""
        handlers = [
            CommandHandler("start", self.start_command),
            CommandHandler("scan", self.deep_scan_command),
            CommandHandler("mass_scan", self.mass_scan_command),
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
            CommandHandler("scan_stats", self.scan_stats_command),
            CallbackQueryHandler(self.advanced_button_handler)
        ]
        
        for handler in handlers:
            self.app.add_handler(handler)

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Оновлене меню"""
        keyboard = [
            [InlineKeyboardButton("🚨 MASS SCAN 150+", callback_data="mass_scan"),
             InlineKeyboardButton("🔍 DEEP SCAN", callback_data="deep_scan")],
            [InlineKeyboardButton("📊 PUMP RADAR", callback_data="pump_radar"),
             InlineKeyboardButton("📉 DUMP RADAR", callback_data="dump_radar")],
            [InlineKeyboardButton("🐋 WHALE WATCH", callback_data="whale_watch"),
             InlineKeyboardButton("💧 LIQUIDITY SCAN", callback_data="liquidity_scan")],
            [InlineKeyboardButton("⚡ VOLATILITY ALERTS", callback_data="volatility_alerts"),
             InlineKeyboardButton("🤖 AI RISK SCAN", callback_data="ai_risk_scan")],
            [InlineKeyboardButton("📈 SCAN STATS", callback_data="scan_stats"),
             InlineKeyboardButton("⚙️ SETTINGS", callback_data="settings")],
            [InlineKeyboardButton("🚫 BLACKLIST", callback_data="blacklist"),
             InlineKeyboardButton("📊 PERFORMANCE", callback_data="performance")],
            [InlineKeyboardButton("🔄 UPDATE", callback_data="update")]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            "🤖 **ULTIMATE PUMP/DUMP DETECTOR v6.0**\n\n"
            "🎯 *Масштабне сканування 150+ токенів одночасно*\n\n"
            "✨ **Покращені фічі:**\n"
            "• 🔥 Сканування 150+ токенів за раз\n"
            "• ⚡ Паралельна обробка даних\n"
            "• 🚀 Оптимізована продуктивність\n"
            "• 📊 Глибокий аналіз якості\n\n"
            "💎 *Фокус на високоліквідних активах 2M+ USDT!*",
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )

    async def mass_scan_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Масове сканування 150+ токенів"""
        try:
            if not await self.check_exchange_connection():
                await update.message.reply_text("❌ Немає підключення до біржі")
                return
            
            msg = await update.message.reply_text("🔥 ЗАПУСКАЮ МАСОВЕ СКАНУВАННЯ 150+ ТОКЕНІВ...")
            
            start_time = time.time()
            symbols = await self.get_all_qualified_symbols()
            
            if not symbols:
                await msg.edit_text("❌ Не вдалося отримати символи для сканування")
                return
            
            # Обмежуємо кількість символів
            symbols = symbols[:self.detection_params['max_symbols_per_scan']]
            
            results = await self.mass_analyze_symbols(symbols)
            
            scan_time = time.time() - start_time
            self.performance_metrics['total_scans'] += 1
            self.performance_metrics['avg_scan_time'] = (
                self.performance_metrics['avg_scan_time'] * (self.performance_metrics['total_scans'] - 1) + scan_time
            ) / self.performance_metrics['total_scans']
            self.performance_metrics['avg_symbols_per_scan'] = len(symbols)
            
            if results:
                results.sort(key=lambda x: max(x['pump_probability'], x['dump_probability']), reverse=True)
                self.performance_metrics['signals_triggered'] += len(results)
                
                response = (
                    f"🚨 **МАСОВЕ СКАНУВАННЯ ЗАВЕРШЕНО**\n\n"
                    f"📊 Проскановано: {len(symbols)} токенів\n"
                    f"⏱️ Час сканування: {scan_time:.1f}с\n"
                    f"🎯 Знайдено сигналів: {len(results)}\n\n"
                )
                
                # Групуємо сигнали за типом
                pump_signals = [r for r in results if r['pump_probability'] > 0.7]
                dump_signals = [r for r in results if r['dump_probability'] > 0.7]
                
                if pump_signals:
                    response += "🔥 **ТОП PUMP СИГНАЛИ:**\n\n"
                    for i, signal in enumerate(pump_signals[:5], 1):
                        response += self.format_signal_message(signal, i)
                    self.performance_metrics['pump_signals_detected'] += len(pump_signals)
                
                if dump_signals:
                    response += "📉 **ТОП DUMP СИГНАЛИ:**\n\n"
                    for i, signal in enumerate(dump_signals[:5], 1):
                        response += self.format_signal_message(signal, i)
                    self.performance_metrics['dump_signals_detected'] += len(dump_signals)
                
                await msg.edit_text(response, parse_mode='Markdown')
            else:
                await msg.edit_text(
                    f"ℹ️ Масове сканування завершено\n"
                    f"📊 Проскановано: {len(symbols)} токенів\n"
                    f"⏱️ Час: {scan_time:.1f}с\n"
                    f"🎯 Сигналів не знайдено"
                )
                
        except Exception as e:
            logger.error(f"Помилка масового сканування: {e}")
            await update.message.reply_text("❌ Помилка масового сканування")

    async def mass_analyze_symbols(self, symbols: List[str]) -> List[Dict]:
        """Масовий аналіз символів з паралельною обробкою"""
        results = []
        
        # Використовуємо ThreadPoolExecutor для паралельної обробки
        with ThreadPoolExecutor(max_workers=min(20, len(symbols))) as executor:
            # Створюємо футури для кожного символу
            future_to_symbol = {
                executor.submit(self.analyze_symbol, symbol): symbol 
                for symbol in symbols
            }
            
            # Обробляємо результати по мірі їх готовності
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result = future.result(timeout=10.0)
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.debug(f"Помилка аналізу {symbol}: {e}")
                finally:
                    # Невелика затримка для rate limiting
                    time.sleep(0.05)
        
        return results

    async def get_all_qualified_symbols(self) -> List[str]:
        """Отримання всіх якісних символів з кешуванням"""
        current_time = time.time()
        
        # Перевіряємо кеш (оновлюємо кожні 5 хвилин)
        if (self.symbols_cache and 
            current_time - self.last_symbols_update < 300):
            return self.symbols_cache
        
        try:
            if not self.exchange:
                return []
            
            # Завантажуємо всі ринки
            markets = await asyncio.get_event_loop().run_in_executor(
                self.executor, self.exchange.load_markets
            )
            
            qualified_symbols = []
            
            for symbol, market in markets.items():
                if (symbol.endswith('/USDT') and 
                    market.get('active', False) and
                    market.get('quoteVolume', 0) >= self.detection_params['min_volume_usdt'] and
                    market.get('quoteVolume', 0) <= self.detection_params['max_volume_usdt'] and
                    not self.is_garbage_symbol(symbol)):
                    
                    # Додаткова перевірка ціни
                    ticker = await asyncio.get_event_loop().run_in_executor(
                        self.executor, lambda: self.exchange.fetch_ticker(symbol)
                    )
                    
                    if (ticker and ticker.get('last', 0) >= self.detection_params['min_price'] and
                        not self.is_low_quality_symbol(symbol, {'close': ticker.get('last', 0)})):
                        
                        qualified_symbols.append(symbol)
            
            # Сортуємо за об'ємом (найбільш ліквідні спочатку)
            qualified_symbols.sort(key=lambda x: markets[x].get('quoteVolume', 0), reverse=True)
            
            # Оновлюємо кеш
            self.symbols_cache = qualified_symbols
            self.last_symbols_update = current_time
            
            logger.info(f"Знайдено {len(qualified_symbols)} якісних символів")
            return qualified_symbols
            
        except Exception as e:
            logger.error(f"Помилка отримання символів: {e}")
            return []

    def is_low_quality_symbol(self, symbol: str, market_data: Dict) -> bool:
        """Перевірка якості символу"""
        try:
            symbol_clean = symbol.upper().replace('/USDT', '')
            
            # Головні криптовалюти - завжди якісні
            major_coins = {
                'BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'ADA', 'DOGE', 'MATIC', 
                'DOT', 'LTC', 'AVAX', 'LINK', 'ATOM', 'XMR', 'ETC', 'BCH',
                'FIL', 'NEAR', 'ALGO', 'VET', 'EOS', 'XTZ', 'THETA', 'AAVE',
                'MKR', 'COMP', 'SNX', 'CRV', 'SUSHI', 'UNI', 'YFI'
            }
            
            if symbol_clean in major_coins:
                return False
                
            if symbol_clean in self.garbage_symbols:
                return True
            
            # Фільтр довжини назви
            if len(symbol_clean) > 10:
                return True
                
            # Фільтр цифр у назві
            if any(char.isdigit() for char in symbol_clean):
                return True
                
            # Фільтр підозрілих патернів
            suspicious_patterns = {'MOON', 'SUN', 'MARS', 'EARTH', 'PLUTO'}
            if any(pattern in symbol_clean for pattern in suspicious_patterns):
                return True
                
            return False
            
        except:
            return True

    async def analyze_symbol(self, symbol: str) -> Dict:
        """Оптимізований аналіз символу"""
        try:
            # Швидка перевірка якості символу
            if self.is_garbage_symbol(symbol):
                return {}
            
            # Отримуємо дані з кешу або API
            market_data = await self.get_market_data(symbol)
            if not market_data:
                return {}
            
            # Швидка перевірка об'єму
            if market_data['volume'] < self.detection_params['min_volume_usdt']:
                return {}
            
            # Отримуємо додаткові дані паралельно
            orderbook_future = asyncio.create_task(self.get_orderbook_depth(symbol, 20))
            klines_future = asyncio.create_task(self.get_klines(symbol, '5m', 25))
            
            orderbook, klines = await asyncio.gather(orderbook_future, klines_future)
            
            if not klines or len(klines) < 15:
                return {}
            
            # Паралельний аналіз
            tech_analysis = self.technical_analysis(klines)
            ob_analysis = self.orderbook_analysis(orderbook)
            volume_analysis = self.volume_analysis(klines, market_data)
            
            # Розрахунок ймовірностей
            pump_prob = self.calculate_pump_probability(tech_analysis, ob_analysis, volume_analysis)
            dump_prob = self.calculate_dump_probability(tech_analysis, ob_analysis, volume_analysis)
            
            # Перевірка якості сигналу
            if (pump_prob < self.detection_params['pump_probability_threshold'] and 
                dump_prob < self.detection_params['dump_probability_threshold']):
                return {}
            
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
            logger.debug(f"Помилка аналізу {symbol}: {e}")
            return {}

    async def get_market_data(self, symbol: str) -> Optional[Dict]:
        """Оптимізоване отримання ринкових даних з кешуванням"""
        try:
            current_time = time.time()
            
            # Перевірка кешу
            if (symbol in self.market_data_cache and 
                current_time - self.market_data_cache[symbol]['timestamp'] < 30):
                return self.market_data_cache[symbol]['data']
            
            if not self.exchange:
                return None
                
            # Використовуємо thread pool для блокувальних операцій
            ticker = await asyncio.get_event_loop().run_in_executor(
                self.executor, lambda: self.exchange.fetch_ticker(symbol)
            )
            
            if not ticker:
                return None
                
            result = {
                'symbol': symbol,
                'open': float(ticker.get('open', 0)),
                'high': float(ticker.get('high', 0)),
                'low': float(ticker.get('low', 0)),
                'close': float(ticker.get('last', ticker.get('close', 0))),
                'volume': float(ticker.get('quoteVolume', 0)),
                'percentage': float(ticker.get('percentage', 0))
            }
            
            # Оновлення кешу
            self.market_data_cache[symbol] = {
                'data': result,
                'timestamp': current_time
            }
            
            return result
            
        except Exception as e:
            logger.debug(f"Помилка отримання даних для {symbol}: {e}")
            return None

    async def scan_stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Статистика сканування"""
        try:
            stats_text = (
                "📊 **СТАТИСТИКА СКАНУВАННЯ:**\n\n"
                f"• Загальна кількість сканувань: {self.performance_metrics['total_scans']}\n"
                f"• Середня кількість токенів: {self.performance_metrics['avg_symbols_per_scan']:.0f}\n"
                f"• Середній час сканування: {self.performance_metrics['avg_scan_time']:.1f}с\n"
                f"• Знайдено сигналів: {self.performance_metrics['signals_triggered']}\n"
                f"• Pump сигналів: {self.performance_metrics['pump_signals_detected']}\n"
                f"• Dump сигналів: {self.performance_metrics['dump_signals_detected']}\n"
                f"• Успішність: {self.performance_metrics['success_rate']:.1f}%\n\n"
            )
            
            # Додаємо інформацію про кеш
            stats_text += f"🔧 **СИСТЕМА:**\n"
            stats_text += f"• Символів у кеші: {len(self.symbols_cache)}\n"
            stats_text += f"• Даних у кеші: {len(self.market_data_cache)}\n"
            stats_text += f"• Воркерів: {self.executor._max_workers}\n"
            
            await update.message.reply_text(stats_text, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Помилка статистики: {e}")
            await update.message.reply_text("❌ Помилка статистики")

    # Оптимізовані версії інших методів
    async def deep_scan_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Глибинне сканування з оптимізацією"""
        try:
            if not await self.check_exchange_connection():
                await update.message.reply_text("❌ Немає підключення до біржі")
                return
            
            msg = await update.message.reply_text("🔍 ЗАПУСКАЮ ГЛИБИННЕ СКАНУВАННЯ...")
            
            symbols = await self.get_all_qualified_symbols()
            symbols = symbols[:50]  # Обмежуємо для глибинного аналізу
            
            results = await self.mass_analyze_symbols(symbols)
            
            if results:
                results.sort(key=lambda x: max(x['pump_probability'], x['dump_probability']), reverse=True)
                
                response = "🚨 **РЕЗУЛЬТАТИ ГЛИБИННОГО СКАНУВАННЯ:**\n\n"
                for i, res in enumerate(results[:8], 1):
                    response += self.format_signal_message(res, i)
                
                response += f"📊 Знайдено {len(results)} сигналів з {len(symbols)} монет"
                await msg.edit_text(response, parse_mode='Markdown')
            else:
                await msg.edit_text("ℹ️ Сильних сигналів не знайдено")
                
        except Exception as e:
            logger.error(f"Помилка глибинного сканування: {e}")
            await update.message.reply_text("❌ Помилка сканування")

    def format_signal_message(self, analysis: Dict, index: int) -> str:
        """Покращене форматування повідомлення"""
        try:
            symbol = analysis['symbol'].replace('/USDT', '')
            pump_prob = analysis['pump_probability']
            dump_prob = analysis['dump_probability']
            
            signal_emoji = "🔥" if pump_prob > dump_prob else "📉"
            signal_type = "PUMP" if pump_prob > dump_prob else "DUMP"
            signal_prob = max(pump_prob, dump_prob)
            
            return (
                f"{index}. {signal_emoji} **{symbol}** - {signal_type} {signal_prob:.1%}\n"
                f"   💰 Ціна: ${analysis['price']:.6f}\n"
                f"   📊 Об'єм: ${analysis['volume_usdt']:,.0f}\n"
                f"   📈 Зміна: {analysis['percentage']:+.2f}%\n"
                f"   📍 RSI: {analysis['technical_indicators']['rsi']:.1f}\n"
                f"   ⚡ Волатильність: {analysis['technical_indicators']['volatility']:.2f}%\n\n"
            )
        except:
            return f"{index}. Помилка форматування сигналу\n\n"

    # Технічні методи залишаються подібними, але оптимізованими
    def technical_analysis(self, klines: List) -> Dict:
        """Оптимізований технічний аналіз"""
        try:
            if len(klines) < 15:
                return {'rsi': 50, 'macd_hist': 0, 'volatility': 0, 'price_acceleration': 0, 'trend_strength': 0.5}
            
            closes = np.array([float(k[4]) for k in klines[-15:]])  # Тільки останні 15 свечок
            highs = np.array([float(k[2]) for k in klines[-15:]])
            lows = np.array([float(k[3]) for k in klines[-15:]])
            
            # Швидкий RSI
            rsi = talib.RSI(closes, timeperiod=14)[-1] if len(closes) >= 14 else 50
            
            # Спрощений MACD
            macd, macd_signal, _ = talib.MACD(closes)
            macd_hist = (macd - macd_signal)[-1] if len(macd) > 0 else 0
            
            # Волатильність
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
            logger.debug(f"Помилка технічного аналізу: {e}")
            return {'rsi': 50, 'macd_hist': 0, 'volatility': 0, 'price_acceleration': 0, 'trend_strength': 0.5}

    # Інші методи залишаються аналогічними, але оптимізованими
    async def run(self):
        """Запуск бота з оптимізацією"""
        try:
            logger.info("🤖 Запускаю Ultimate Pump/Dump Detector v6.0...")
            
            # Попереднє завантаження символів
            logger.info("⏳ Попереднє завантаження символів...")
            await self.get_all_qualified_symbols()
            
            await self.app.initialize()
            await self.app.start()
            await self.app.updater.start_polling()
            
            logger.info("✅ Бот успішно запущено! Очікую команди...")
            
            # Фонові tasks
            asyncio.create_task(self.background_monitoring())
            
            while True:
                await asyncio.sleep(3600)
            
        except Exception as e:
            logger.error(f"❌ Помилка запуску бота: {e}")
            raise

    async def background_monitoring(self):
        """Фоновий моніторинг та оновлення"""
        while True:
            try:
                # Оновлюємо кеш даних
                await self.update_market_data_cache()
                await asyncio.sleep(300)
                
            except Exception as e:
                logger.error(f"Помилка фонового моніторингу: {e}")
                await asyncio.sleep(60)

    async def update_market_data_cache(self):
        """Оновлення кешу ринкових даних"""
        try:
            symbols = await self.get_all_qualified_symbols()
            symbols = symbols[:100]  # Оновлюємо топ 100
            
            for symbol in symbols:
                try:
                    await self.get_market_data(symbol)
                    await asyncio.sleep(0.1)
                except Exception as e:
                    continue
                    
            logger.info(f"✅ Кеш оновлено: {len(self.market_data_cache)} записів")
        except Exception as e:
            logger.error(f"Помилка оновлення кешу: {e}")

def main():
    """Головна функція запуску бота"""
    try:
        BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
        
        if not BOT_TOKEN:
            logger.error("❌ Будь ласка, встановіть TELEGRAM_BOT_TOKEN у змінних оточення")
            return
        
        bot = UltimatePumpDumpDetector(BOT_TOKEN)
        logger.info("🚀 Запускаю бота...")
        bot.app.run_polling()
        
    except KeyboardInterrupt:
        logger.info("⏹️ Зупинка бота...")
    except Exception as e:
        logger.error(f"❌ Критична помилка: {e}")
        raise

if __name__ == '__main__':
    # Оптимізація логування
    logging.getLogger('ccxt').setLevel(logging.WARNING)
    logging.getLogger('telegram').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    
    # Запуск
    main()