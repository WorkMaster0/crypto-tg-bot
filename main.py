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
        
        # Підключення до Binance
        logger.info("Ініціалізація підключення до Binance...")
        try:
            self.exchange = ccxt.binance({
                'enableRateLimit': True,
                'timeout': 20000,
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
        
        # Параметри для виявлення
        self.detection_params = {
            'volume_spike_threshold': 1.8,
            'price_acceleration_min': 0.003,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'orderbook_imbalance_min': 0.15,
            'large_order_threshold': 50000,
            'min_volume_usdt': 1000000,
            'max_volume_usdt': 50000000,
            'price_change_5m_min': 1.0,
            'wick_ratio_threshold': 0.25,
            'market_cap_filter': 1000000,
            'liquidity_score_min': 0.4,
            'pump_probability_threshold': 0.6,
            'dump_probability_threshold': 0.6,
            'whale_volume_threshold': 50000,
            'volatility_spike_threshold': 2.0,
            'min_daily_change': 5.0,
            'min_price': 0.0005,
            'max_symbols_per_scan': 100
        }
        
        # Кеш та оптимізація
        self.market_data_cache = {}
        self.symbols_cache = []
        self.last_symbols_update = 0
        self.performance_history = deque(maxlen=1000)
        
        # Статистика
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
        
        # Пул потоків
        self.executor = ThreadPoolExecutor(max_workers=15)
        self.setup_handlers()
        
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
            [InlineKeyboardButton("🚨 MASS SCAN 100+", callback_data="mass_scan"),
             InlineKeyboardButton("🔍 DEEP SCAN", callback_data="deep_scan")],
            [InlineKeyboardButton("📊 PUMP RADAR", callback_data="pump_radar"),
             InlineKeyboardButton("📉 DUMP RADAR", callback_data="dump_radar")],
            [InlineKeyboardButton("🐋 WHALE WATCH", callback_data="whale_watch"),
             InlineKeyboardButton("💧 LIQUIDITY SCAN", callback_data="liquidity_scan")],
            [InlineKeyboardButton("⚡ VOLATILITY", callback_data="volatility_alerts"),
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
            "🎯 *Професійний аналіз ринку в реальному часі*\n\n"
            "✨ **Можливості:**\n"
            "• 📊 Сканування 100+ токенів одночасно\n"
            "• ⚡ Миттєве виявлення аномалій\n"
            "• 🎯 Точні сигнали на основі даних\n"
            "• 🔍 Глибинний технічний аналіз\n\n"
            "💎 _Фокус на якісних активах з високою ліквідністю_",
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )

    async def mass_scan_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Масове сканування 100+ токенів"""
        try:
            if not await self.check_exchange_connection():
                await update.message.reply_text("❌ Немає підключення до біржі")
                return
            
            msg = await update.message.reply_text("🔥 ЗАПУСКАЮ МАСОВЕ СКАНУВАННЯ 100+ ТОКЕНІВ...")
            
            start_time = time.time()
            symbols = await self.get_all_qualified_symbols()
            
            if not symbols:
                await msg.edit_text("❌ Не вдалося отримати символи для сканування")
                return
            
            symbols = symbols[:self.detection_params['max_symbols_per_scan']]
            
            results = await self.mass_analyze_symbols(symbols)
            
            scan_time = time.time() - start_time
            self.performance_metrics['total_scans'] += 1
            self.performance_metrics['avg_scan_time'] = scan_time
            self.performance_metrics['avg_symbols_per_scan'] = len(symbols)
            
            if results:
                results.sort(key=lambda x: max(x['pump_probability'], x['dump_probability']), reverse=True)
                self.performance_metrics['signals_triggered'] += len(results)
                
                response = (
                    f"🚨 **РЕЗУЛЬТАТИ МАСОВОГО СКАНУВАННЯ**\n\n"
                    f"📊 Проскановано: {len(symbols)} токенів\n"
                    f"⏱️ Час: {scan_time:.1f} секунд\n"
                    f"🎯 Знайдено сигналів: {len(results)}\n\n"
                )
                
                # Групуємо сигнали
                strong_signals = [r for r in results if max(r['pump_probability'], r['dump_probability']) > 0.75]
                medium_signals = [r for r in results if 0.6 <= max(r['pump_probability'], r['dump_probability']) <= 0.75]
                
                if strong_signals:
                    response += "🔥 **СИЛЬНІ СИГНАЛИ:**\n\n"
                    for i, signal in enumerate(strong_signals[:5], 1):
                        response += self.format_signal_message(signal, i)
                
                if medium_signals and not strong_signals:
                    response += "⚠️ **СЕРЕДНІ СИГНАЛИ:**\n\n"
                    for i, signal in enumerate(medium_signals[:5], 1):
                        response += self.format_signal_message(signal, i)
                
                await msg.edit_text(response, parse_mode='Markdown')
            else:
                await msg.edit_text(
                    f"📊 Масове сканування завершено\n"
                    f"• Токенів: {len(symbols)}\n"
                    f"• Час: {scan_time:.1f}с\n"
                    f"• Сигналів: 0\n\n"
                    f"ℹ️ Сильних сигналів не виявлено"
                )
                
        except Exception as e:
            logger.error(f"Помилка масового сканування: {e}")
            await update.message.reply_text("❌ Помилка масового сканування")

    async def mass_analyze_symbols(self, symbols: List[str]) -> List[Dict]:
        """Масовий аналіз символів"""
        results = []
        
        with ThreadPoolExecutor(max_workers=min(15, len(symbols))) as executor:
            future_to_symbol = {
                executor.submit(self.analyze_symbol, symbol): symbol 
                for symbol in symbols
            }
            
            for future in as_completed(future_to_symbol):
                try:
                    result = future.result(timeout=8.0)
                    if result:
                        results.append(result)
                except Exception as e:
                    symbol = future_to_symbol[future]
                    logger.debug(f"Помилка аналізу {symbol}: {e}")
                finally:
                    time.sleep(0.03)
        
        return results

    async def get_all_qualified_symbols(self) -> List[str]:
        """Отримання якісних символів"""
        current_time = time.time()
        
        if (self.symbols_cache and 
            current_time - self.last_symbols_update < 300):
            return self.symbols_cache
        
        try:
            if not self.exchange:
                return []
            
            markets = await asyncio.get_event_loop().run_in_executor(
                self.executor, self.exchange.load_markets
            )
            
            qualified_symbols = []
            
            for symbol, market in markets.items():
                if (symbol.endswith('/USDT') and 
                    market.get('active', False) and
                    market.get('quoteVolume', 0) >= self.detection_params['min_volume_usdt'] and
                    not self.is_garbage_symbol(symbol)):
                    
                    qualified_symbols.append(symbol)
            
            # Сортуємо за об'ємом
            qualified_symbols.sort(key=lambda x: markets[x].get('quoteVolume', 0), reverse=True)
            
            self.symbols_cache = qualified_symbols
            self.last_symbols_update = current_time
            
            logger.info(f"Знайдено {len(qualified_symbols)} якісних символів")
            return qualified_symbols
            
        except Exception as e:
            logger.error(f"Помилка отримання символів: {e}")
            return []

    def is_garbage_symbol(self, symbol: str) -> bool:
        """Перевірка символу на сміття"""
        try:
            symbol_clean = symbol.upper().replace('/USDT', '')
            
            # Основні криптовалюти
            major_coins = {
                'BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'ADA', 'DOGE', 'MATIC', 
                'DOT', 'LTC', 'AVAX', 'LINK', 'ATOM', 'XMR', 'ETC', 'BCH',
                'FIL', 'NEAR', 'ALGO', 'VET', 'EOS', 'XTZ', 'THETA', 'AAVE',
                'MKR', 'COMP', 'SNX', 'CRV', 'SUSHI', 'UNI', 'YFI', 'FTM',
                'EGLD', 'ONE', 'ZEC', 'DASH', 'QTUM', 'ZIL', 'SC', 'ICX'
            }
            
            if symbol_clean in major_coins:
                return False
                
            if symbol_clean in self.garbage_symbols:
                return True
            
            if len(symbol_clean) > 10:
                return True
                
            if any(char.isdigit() for char in symbol_clean):
                return True
                
            suspicious_patterns = {'MOON', 'SUN', 'MARS', 'EARTH', 'PLUTO'}
            if any(pattern in symbol_clean for pattern in suspicious_patterns):
                return True
                
            return False
            
        except:
            return True

    async def analyze_symbol(self, symbol: str) -> Dict:
        """Аналіз символу"""
        try:
            if self.is_garbage_symbol(symbol):
                return {}
            
            market_data = await self.get_market_data(symbol)
            if not market_data or market_data['volume'] < self.detection_params['min_volume_usdt']:
                return {}
            
            # Паралельне отримання даних
            orderbook_future = asyncio.create_task(self.get_orderbook_depth(symbol, 20))
            klines_future = asyncio.create_task(self.get_klines(symbol, '5m', 20))
            
            orderbook, klines = await asyncio.gather(orderbook_future, klines_future)
            
            if not klines or len(klines) < 15:
                return {}
            
            # Аналіз
            tech_analysis = self.technical_analysis(klines)
            ob_analysis = self.orderbook_analysis(orderbook)
            volume_analysis = self.volume_analysis(klines, market_data)
            
            pump_prob = self.calculate_pump_probability(tech_analysis, ob_analysis, volume_analysis)
            dump_prob = self.calculate_dump_probability(tech_analysis, ob_analysis, volume_analysis)
            
            # Перевірка якості сигналу
            if max(pump_prob, dump_prob) < 0.6:
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
        """Отримання ринкових даних"""
        try:
            current_time = time.time()
            
            if (symbol in self.market_data_cache and 
                current_time - self.market_data_cache[symbol]['timestamp'] < 30):
                return self.market_data_cache[symbol]['data']
            
            if not self.exchange:
                return None
                
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
            
            self.market_data_cache[symbol] = {
                'data': result,
                'timestamp': current_time
            }
            
            return result
            
        except Exception as e:
            logger.debug(f"Помилка отримання даних для {symbol}: {e}")
            return None

    def technical_analysis(self, klines: List) -> Dict:
        """Технічний аналіз"""
        try:
            if len(klines) < 15:
                return {'rsi': 50, 'macd_hist': 0, 'volatility': 0, 'price_acceleration': 0, 'trend_strength': 0.5}
            
            closes = np.array([float(k[4]) for k in klines[-15:]])
            highs = np.array([float(k[2]) for k in klines[-15:]])
            lows = np.array([float(k[3]) for k in klines[-15:]])
            
            # RSI
            rsi = talib.RSI(closes, timeperiod=14)[-1] if len(closes) >= 14 else 50
            
            # MACD
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

    def orderbook_analysis(self, orderbook: Dict) -> Dict:
        """Аналіз стакану"""
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
            logger.error(f"Помилка аналізу стакану: {e}")
            return {'imbalance': 0, 'large_bids': 0, 'large_asks': 0, 'total_bid_volume': 0, 'total_ask_volume': 0}

    def volume_analysis(self, klines: List, market_data: Dict) -> Dict:
        """Аналіз об'ємів"""
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
            
            avg_volume = np.mean(volumes[:-5]) if len(volumes) > 5 else volumes[0]
            current_volume = volumes[-1] if len(volumes) > 0 else 0
            volume_spike = current_volume / avg_volume if avg_volume > 0 else 1
            
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

    def calculate_pump_probability(self, tech: Dict, orderbook: Dict, volume: Dict) -> float:
        """Розрахунок ймовірності пампу"""
        try:
            weights = {
                'rsi': 0.25,
                'volume_spike': 0.30,
                'ob_imbalance': 0.20,
                'price_accel': 0.15,
                'volatility': 0.10
            }
            
            score = (
                (1.0 if tech['rsi'] < 35 else 0.7 if tech['rsi'] < 45 else 0.3) * weights['rsi'] +
                min(volume['volume_spike_ratio'] / 2.0, 1.0) * weights['volume_spike'] +
                max(min(orderbook.get('imbalance', 0) + 0.5, 1.0), 0.0) * weights['ob_imbalance'] +
                min(abs(tech['price_acceleration']) / 0.01, 1.0) * weights['price_accel'] +
                min(tech['volatility'] / 15.0, 1.0) * weights['volatility']
            )
            
            return round(score, 4)
        except:
            return 0.3

    def calculate_dump_probability(self, tech: Dict, orderbook: Dict, volume: Dict) -> float:
        """Розрахунок ймовірності дампу"""
        try:
            weights = {
                'rsi': 0.30,
                'volume_divergence': 0.25,
                'ob_imbalance': 0.20,
                'volatility': 0.15,
                'price_accel': 0.10
            }
            
            score = (
                (1.0 if tech['rsi'] > 75 else 0.7 if tech['rsi'] > 65 else 0.3) * weights['rsi'] +
                (1.0 - min(max(volume['volume_price_correlation'], -1.0), 1.0)) * weights['volume_divergence'] +
                max(min(-orderbook.get('imbalance', 0) + 0.5, 1.0), 0.0) * weights['ob_imbalance'] +
                min(tech['volatility'] / 20.0, 1.0) * weights['volatility'] +
                min(abs(tech['price_acceleration']) / 0.008, 1.0) * weights['price_accel']
            )
            
            return round(score, 4)
        except:
            return 0.3

    def format_signal_message(self, analysis: Dict, index: int) -> str:
        """Форматування повідомлення про сигнал"""
        try:
            symbol = analysis['symbol'].replace('/USDT', '')
            pump_prob = analysis['pump_probability']
            dump_prob = analysis['dump_probability']
            
            if pump_prob > dump_prob:
                signal_emoji = "🔥"
                signal_type = "PUMP"
                signal_strength = self.get_strength_description(pump_prob)
            else:
                signal_emoji = "📉"
                signal_type = "DUMP"
                signal_strength = self.get_strength_description(dump_prob)
            
            return (
                f"{index}. {signal_emoji} **{symbol}** - {signal_type} ({signal_strength})\n"
                f"   💰 Ціна: ${analysis['price']:.6f}\n"
                f"   📈 Добова зміна: {analysis['percentage']:+.2f}%\n"
                f"   📊 Об'єм: ${analysis['volume_usdt']:,.0f}\n"
                f"   📍 RSI: {analysis['technical_indicators']['rsi']:.1f}\n"
                f"   ⚡ Волатильність: {analysis['technical_indicators']['volatility']:.1f}%\n\n"
            )
        except:
            return f"{index}. Помилка форматування сигналу\n\n"

    def get_strength_description(self, probability: float) -> str:
        """Опис сили сигналу"""
        if probability > 0.8:
            return "ДУЖЕ СИЛЬНИЙ"
        elif probability > 0.7:
            return "СИЛЬНИЙ"
        elif probability > 0.6:
            return "ПОМІРНИЙ"
        else:
            return "СЛАБКИЙ"

    async def pump_radar_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Детектор пампів"""
        try:
            if not await self.check_exchange_connection():
                await update.message.reply_text("❌ Немає підключення до біржі")
                return
            
            msg = await update.message.reply_text("🚨 АКТИВУЮ PUMP RADAR...")
            
            symbols = await self.get_all_qualified_symbols()
            symbols = symbols[:30]
            
            pump_candidates = []
            
            for symbol in symbols:
                analysis = await self.analyze_symbol(symbol)
                if analysis and analysis['pump_probability'] > 0.65:
                    pump_candidates.append(analysis)
                await asyncio.sleep(0.1)
            
            if pump_candidates:
                pump_candidates.sort(key=lambda x: x['pump_probability'], reverse=True)
                
                response = "🔥 **PUMP СИГНАЛИ:**\n\n"
                for i, candidate in enumerate(pump_candidates[:5], 1):
                    response += self.format_signal_message(candidate, i)
                
                await msg.edit_text(response, parse_mode='Markdown')
            else:
                await msg.edit_text("✅ Потужних pump-сигналів не знайдено")
                
        except Exception as e:
            logger.error(f"Помилка pump radar: {e}")
            await update.message.reply_text("❌ Помилка pump radar")

    async def dump_radar_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Детектор дампів"""
        try:
            if not await self.check_exchange_connection():
                await update.message.reply_text("❌ Немає підключення до біржі")
                return
            
            msg = await update.message.reply_text("📉 АКТИВУЮ DUMP RADAR...")
            
            symbols = await self.get_all_qualified_symbols()
            symbols = symbols[:30]
            
            dump_candidates = []
            
            for symbol in symbols:
                analysis = await self.analyze_symbol(symbol)
                if analysis and analysis['dump_probability'] > 0.65:
                    dump_candidates.append(analysis)
                await asyncio.sleep(0.1)
            
            if dump_candidates:
                dump_candidates.sort(key=lambda x: x['dump_probability'], reverse=True)
                
                response = "📉 **DUMP СИГНАЛИ:**\n\n"
                for i, candidate in enumerate(dump_candidates[:5], 1):
                    response += self.format_signal_message(candidate, i)
                
                await msg.edit_text(response, parse_mode='Markdown')
            else:
                await msg.edit_text("✅ Потужних dump-сигналів не знайдено")
                
        except Exception as e:
            logger.error(f"Помилка dump radar: {e}")
            await update.message.reply_text("❌ Помилка dump radar")

    async def deep_scan_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Глибинне сканування"""
        try:
            if not await self.check_exchange_connection():
                await update.message.reply_text("❌ Немає підключення до біржі")
                return
            
            msg = await update.message.reply_text("🔍 ЗАПУСКАЮ ГЛИБИННЕ СКАНУВАННЯ...")
            
            symbols = await self.get_all_qualified_symbols()
            symbols = symbols[:50]
            
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
                f"• Dump сигналів: {self.performance_metrics['dump_signals_detected']}\n\n"
            )
            
            stats_text += f"🔧 **СИСТЕМА:**\n"
            stats_text += f"• Символів у кеші: {len(self.symbols_cache)}\n"
            stats_text += f"• Даних у кеші: {len(self.market_data_cache)}\n"
            
            await update.message.reply_text(stats_text, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Помилка статистики: {e}")
            await update.message.reply_text("❌ Помилка статистики")

    async def check_exchange_connection(self) -> bool:
        """Перевірка підключення до біржі"""
        if not self.exchange:
            return False
        
        try:
            await asyncio.get_event_loop().run_in_executor(
                self.executor, self.exchange.fetch_status
            )
            return True
        except Exception as e:
            logger.error(f"Помилка підключення до біржі: {e}")
            return False

    async def check_network_connection(self) -> bool:
        """Перевірка мережевого з'єднання"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get('https://api.binance.com/api/v3/ping', timeout=10) as response:
                    return response.status == 200
        except Exception as e:
            logger.error(f"Помилка мережі: {e}")
            return False

    async def get_orderbook_depth(self, symbol: str, limit: int = 20) -> Dict:
        """Отримання глибини ринку"""
        try:
            if not self.exchange:
                return {'bids': [], 'asks': [], 'symbol': symbol}
                
            orderbook = await asyncio.get_event_loop().run_in_executor(
                self.executor, lambda: self.exchange.fetch_order_book(symbol, limit)
            )
            return orderbook
        except Exception as e:
            logger.error(f"Помилка стакану для {symbol}: {e}")
            return {'bids': [], 'asks': [], 'symbol': symbol}

    async def get_klines(self, symbol: str, timeframe: str = '5m', limit: int = 20) -> List:
        """Отримання історичних даних"""
        try:
            if not self.exchange:
                return []
                
            klines = await asyncio.get_event_loop().run_in_executor(
                self.executor, lambda: self.exchange.fetch_ohlcv(symbol, timeframe, limit)
            )
            return klines
        except Exception as e:
            logger.error(f"Помилка klines для {symbol}: {e}")
            return []

    async def whale_watch_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Моніторинг китів"""
        try:
            await update.message.reply_text("🐋 Функція в розробці...")
        except Exception as e:
            logger.error(f"Помилка whale watch: {e}")

    async def liquidity_scan_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Сканування ліквідності"""
        try:
            await update.message.reply_text("💧 Функція в розробці...")
        except Exception as e:
            logger.error(f"Помилка liquidity scan: {e}")

    async def volatility_alert_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Сигнали волатильності"""
        try:
            await update.message.reply_text("⚡ Функція в розробці...")
        except Exception as e:
            logger.error(f"Помилка volatility alert: {e}")

    async def ai_risk_scan_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """AI аналіз ризиків"""
        try:
            await update.message.reply_text("🤖 Функція в розробці...")
        except Exception as e:
            logger.error(f"Помилка AI risk scan: {e}")

    async def quick_scan_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Швидке сканування"""
        try:
            await update.message.reply_text("⚡ Функція в розробці...")
        except Exception as e:
            logger.error(f"Помилка quick scan: {e}")

    async def emergency_scan(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Екстрене сканування"""
        try:
            await update.message.reply_text("🚨 Функція в розробці...")
        except Exception as e:
            logger.error(f"Помилка emergency scan: {e}")

    async def test_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Тестова команда"""
        try:
            await update.message.reply_text("🧪 Тестую роботу бота...")
            
            network_ok = await self.check_network_connection()
            exchange_ok = await self.check_exchange_connection()
            
            response = (
                f"📡 Мережа: {'✅' if network_ok else '❌'}\n"
                f"📊 Біржа: {'✅' if exchange_ok else '❌'}\n"
                f"🔧 Статус: 🟢 ПРАЦЮЄ"
            )
            
            await update.message.reply_text(response)
            
        except Exception as e:
            logger.error(f"Помилка тесту: {e}")
            await update.message.reply_text("❌ Помилка тесту")

    async def test_symbol_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Тестування символу"""
        try:
            symbol = context.args[0].upper() if context.args else 'BTC/USDT'
            if not symbol.endswith('/USDT'):
                symbol += '/USDT'
            
            await update.message.reply_text(f"🔍 Тестую {symbol}...")
            
            analysis = await self.analyze_symbol(symbol)
            if analysis:
                response = (
                    f"📊 **{symbol.replace('/USDT', '')}:**\n"
                    f"💰 Ціна: ${analysis['price']:.6f}\n"
                    f"📈 Зміна: {analysis['percentage']:+.2f}%\n"
                    f"📊 Об'єм: ${analysis['volume_usdt']:,.0f}\n"
                    f"🚨 Pump: {analysis['pump_probability']:.1%}\n"
                    f"📉 Dump: {analysis['dump_probability']:.1%}"
                )
            else:
                response = "❌ Не вдалося проаналізувати символ"
            
            await update.message.reply_text(response, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Помилка тесту символу: {e}")
            await update.message.reply_text("❌ Помилка тесту символу")

    async def settings_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Налаштування"""
        try:
            settings_text = "⚙️ **НАЛАШТУВАННЯ:**\n\n"
            for key, value in self.detection_params.items():
                settings_text += f"• {key}: {value}\n"
            
            await update.message.reply_text(settings_text, parse_mode='Markdown')
        except Exception as e:
            logger.error(f"Помилка налаштувань: {e}")
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
            logger.error(f"Помилка чорного списку: {e}")
            await update.message.reply_text("❌ Помилка чорного списку")

    async def performance_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Статистика"""
        try:
            total = self.performance_metrics['total_scans']
            signals = self.performance_metrics['signals_triggered']
            success_rate = (signals / total * 100) if total > 0 else 0
            
            stats_text = (
                "📈 **СТАТИСТИКА:**\n\n"
                f"• Загальна кількість скануваń: {total}\n"
                f"• Знайдено сигналів: {signals}\n"
                f"• Pump сигналів: {self.performance_metrics['pump_signals_detected']}\n"
                f"• Dump сигналів: {self.performance_metrics['dump_signals_detected']}\n"
                f"• Успішність: {success_rate:.1f}%\n"
            )
            
            await update.message.reply_text(stats_text, parse_mode='Markdown')
        except Exception as e:
            logger.error(f"Помилка статистики: {e}")
            await update.message.reply_text("❌ Помилка статистики")

    async def advanced_button_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обробник кнопок"""
        query = update.callback_query
        await query.answer()
        
        try:
            if query.data == "mass_scan":
                await self.mass_scan_command(query, context)
            elif query.data == "deep_scan":
                await self.deep_scan_command(query, context)
            elif query.data == "pump_radar":
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
            elif query.data == "scan_stats":
                await self.scan_stats_command(query, context)
            elif query.data == "settings":
                await self.settings_command(query, context)
            elif query.data == "blacklist":
                await self.blacklist_command(query, context)
            elif query.data == "performance":
                await self.performance_command(query, context)
            elif query.data == "update":
                await query.edit_message_text("🔄 Оновлюю дані...")
                await asyncio.sleep(1)
                await self.start_command(query, context)
                
        except Exception as e:
            await query.edit_message_text("❌ Помилка обробки запиту")

    async def debug_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Діагностична команда"""
    try:
        network_ok = await self.check_network_connection()
        exchange_ok = await self.check_exchange_connection()
        
        debug_info = f"""
🔧 **ДІАГНОСТИКА СИСТЕМИ:**

📡 Мережа: {'✅' if network_ok else '❌'}
📊 Біржа: {'✅' if exchange_ok else '❌'}
📈 Символів у кеші: {len(self.symbols_cache)}
💾 Даних у кеші: {len(self.market_data_cache)}
⚡ Воркерів: {self.executor._max_workers}

📊 **СТАТИСТИКА:**
• Сканувань: {self.performance_metrics['total_scans']}
• Сигналів: {self.performance_metrics['signals_triggered']}
• Успішність: {self.performance_metrics['success_rate']:.1f}%
"""

        await update.message.reply_text(debug_info, parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"Помилка діагностики: {e}")
        await update.message.reply_text(f"❌ Помилка діагностики: {e}")
    
    async def run(self):
        """Запуск бота"""
        try:
            logger.info("🤖 Запускаю Ultimate Pump/Dump Detector v6.0...")
            
            # Попереднє завантаження
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
        """Фоновий моніторинг"""
        while True:
            try:
                await self.update_market_data_cache()
                await asyncio.sleep(300)
                
            except Exception as e:
                logger.error(f"Помилка фонового моніторингу: {e}")
                await asyncio.sleep(60)

    async def update_market_data_cache(self):
        """Оновлення кешу"""
        try:
            symbols = await self.get_all_qualified_symbols()
            symbols = symbols[:50]
            
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
    """Головна функція"""
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