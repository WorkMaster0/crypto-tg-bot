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

    async def deep_scan_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Глибинне сканування для виявлення аномалій"""
        try:
            msg = await update.message.reply_text("🔍 Запускаю глибинне сканування ринку...")
            
            # Отримуємо активні монети
            active_symbols = await self.get_active_symbols(limit=80)
            results = []
            
            for symbol in active_symbols:
                try:
                    analysis = await self.analyze_symbol(symbol)
                    if analysis and (analysis['pump_probability'] > 0.65 or analysis['dump_probability'] > 0.6):
                        results.append(analysis)
                    await asyncio.sleep(0.1)  # Rate limiting
                except Exception as e:
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

    async def pump_radar_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Спеціалізований детектор пампів"""
        try:
            msg = await update.message.reply_text("🚨 АКТИВУЮ PUMP RADAR...")
            
            symbols = await self.get_high_volume_symbols(limit=50)
            pump_candidates = []
            
            for symbol in symbols:
                try:
                    analysis = await self.analyze_pump_potential(symbol)
                    if analysis and analysis['pump_confidence'] > 70:
                        pump_candidates.append(analysis)
                    await asyncio.sleep(0.1)
                except Exception as e:
                    continue
            
            if pump_candidates:
                pump_candidates.sort(key=lambda x: x['pump_confidence'], reverse=True)
                self.performance_metrics['pump_signals_detected'] += len(pump_candidates)
                
                response = "🔥 **ВИСОКИЙ РИЗИК PUMP:**\n\n"
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
                await msg.edit_text("✅ Немає активних pump-сигналів. Риск стабільний.")
                
        except Exception as e:
            logger.error(f"Помилка pump radar: {e}")
            await update.message.reply_text("❌ Помилка pump radar")

    async def dump_radar_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Детектор майбутніх дампів"""
        try:
            msg = await update.message.reply_text("📉 АКТИВУЮ DUMP RADAR...")
            
            symbols = await self.get_top_gainers(limit=30)
            dump_candidates = []
            
            for symbol in symbols:
                try:
                    analysis = await self.analyze_dump_potential(symbol)
                    if analysis and analysis['dump_confidence'] > 65:
                        dump_candidates.append(analysis)
                    await asyncio.sleep(0.1)
                except Exception as e:
                    continue
            
            if dump_candidates:
                dump_candidates.sort(key=lambda x: x['dump_confidence'], reverse=True)
                self.performance_metrics['dump_signals_detected'] += len(dump_candidates)
                
                response = "⚠️ **ПОПЕРЕДЖЕННЯ ПРО DUMP:**\n\n"
                for i, candidate in enumerate(dump_candidates[:3], 1):
                    response += (
                        f"{i}. **{candidate['symbol']}** - {candidate['dump_confidence']}% впевненість\n"
                        f"   📉 Макс. зміна: {candidate['max_gain']:.2f}%\n"
                        f"   📊 Об'єм продажів: ${candidate['sell_volume']:,.0f}\n"
                        f"   🐋 Китові продажі: {candidate['whale_sells']}\n"
                        f"   📍 RSI: {candidate['rsi']:.1f}\n\n"
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
                await msg.edit_text("✅ Значної активності китів не виявлено.")
                
        except Exception as e:
            logger.error(f"Помилка whale watch: {e}")
            await update.message.reply_text("❌ Помилка моніторингу китів")

    async def analyze_symbol(self, symbol: str) -> Dict:
        """Комплексний аналіз символу"""
        try:
            # Отримуємо дані з різних джерел
            market_data = await self.get_market_data(symbol)
            orderbook = await self.get_orderbook_depth(symbol)
            klines = await self.get_klines(symbol, '5m', 100)
            
            if not all([market_data, orderbook, klines]):
                return {}
            
            # Технічний аналіз
            tech_analysis = self.technical_analysis(klines)
            # Аналіз orderbook
            ob_analysis = self.orderbook_analysis(orderbook)
            # Аналіз об'ємів
            volume_analysis = self.volume_analysis(klines, market_data)
            
            # Розрахунок ймовірностей
            pump_prob = self.calculate_pump_probability(
                tech_analysis, ob_analysis, volume_analysis
            )
            
            dump_prob = self.calculate_dump_probability(
                tech_analysis, ob_analysis, volume_analysis
            )
            
            return {
                'symbol': symbol,
                'price': market_data['close'],
                'volume_usdt': market_data['volume'],
                'pump_probability': pump_prob,
                'dump_probability': dump_prob,
                'technical_indicators': tech_analysis,
                'orderbook_metrics': ob_analysis,
                'volume_metrics': volume_analysis,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Помилка аналізу {symbol}: {e}")
            return {}

    def calculate_pump_probability(self, tech: Dict, orderbook: Dict, volume: Dict) -> float:
        """Розрахунок ймовірності пампу"""
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

    def calculate_dump_probability(self, tech: Dict, orderbook: Dict, volume: Dict) -> float:
        """Розрахунок ймовірності дампу"""
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

    def technical_analysis(self, klines: List) -> Dict:
        """Поглиблений технічний аналіз"""
        try:
            closes = np.array([float(k[4]) for k in klines])
            highs = np.array([float(k[2]) for k in klines])
            lows = np.array([float(k[3]) for k in klines])
            volumes = np.array([float(k[5]) for k in klines])
            
            # RSI
            rsi = talib.RSI(closes, timeperiod=14)[-1] if len(closes) >= 15 else 50
            
            # MACD
            macd, macd_signal, _ = talib.MACD(closes)
            macd_hist = macd - macd_signal
            
            # Волотильність
            volatility = talib.ATR(highs, lows, closes, timeperiod=14)[-1] / closes[-1] * 100
            
            # Прискорення ціни
            if len(closes) >= 6:
                price_acceleration = np.polyfit(range(6), closes[-6:], 1)[0] / closes[-6] * 100
            else:
                price_acceleration = 0
            
            return {
                'rsi': round(rsi, 2),
                'macd_hist': round(macd_hist[-1], 6) if len(macd_hist) > 0 else 0,
                'volatility': round(volatility, 2),
                'price_acceleration': round(price_acceleration, 4),
                'trend_strength': self.calculate_trend_strength(closes)
            }
        except Exception as e:
            return {'rsi': 50, 'macd_hist': 0, 'volatility': 0, 'price_acceleration': 0, 'trend_strength': 0}

    def orderbook_analysis(self, orderbook: Dict) -> Dict:
        """Детальний аналіз стану ордерів"""
        try:
            bids = orderbook.get('bids', [])
            asks = orderbook.get('asks', [])
            
            # Розрахунок імбалансу
            total_bid = sum(float(bid[1]) for bid in bids)
            total_ask = sum(float(ask[1]) for ask in asks)
            imbalance = (total_bid - total_ask) / (total_bid + total_ask) if (total_bid + total_ask) > 0 else 0
            
            # Великі ордери
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
            return {'imbalance': 0, 'large_bids': 0, 'large_asks': 0, 'total_bid_volume': 0, 'total_ask_volume': 0}

    def volume_analysis(self, klines: List, market_data: Dict) -> Dict:
        """Аналіз об'ємів торгів"""
        try:
            volumes = np.array([float(k[5]) for k in klines])
            closes = np.array([float(k[4]) for k in klines])
            
            # Спайк об'ємів
            avg_volume = np.mean(volumes[:-5])
            current_volume = volumes[-1]
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
            return {'volume_spike_ratio': 1, 'volume_price_correlation': 0, 'current_volume': 0, 'average_volume': 0}

    async def get_active_symbols(self, limit: int = 100) -> List[str]:
        """Отримання активних торгових пар"""
        try:
            markets = await asyncio.get_event_loop().run_in_executor(
                self.executor, self.exchange.load_markets
            )
            
            # Фільтруємо USDT пари з достатнім об'ємом
            usdt_pairs = [
                symbol for symbol, market in markets.items()
                if symbol.endswith('/USDT') and market.get('active', False)
                and market.get('quoteVolume', 0) > self.detection_params['min_volume_usdt']
                and market.get('quoteVolume', 0) < self.detection_params['max_volume_usdt']
            ]
            
            # Сортуємо за об'ємом
            usdt_pairs.sort(key=lambda x: markets[x].get('quoteVolume', 0), reverse=True)
            
            return usdt_pairs[:limit]
            
        except Exception as e:
            logger.error(f"Помилка отримання активних символів: {e}")
            return []

    async def get_high_volume_symbols(self, limit: int = 50) -> List[str]:
        """Отримання монет з високим об'ємом"""
        try:
            markets = await asyncio.get_event_loop().run_in_executor(
                self.executor, self.exchange.load_markets
            )
            
            usdt_pairs = [
                symbol for symbol, market in markets.items()
                if symbol.endswith('/USDT') and market.get('active', False)
                and market.get('quoteVolume', 0) > self.detection_params['min_volume_usdt'] * 2
            ]
            
            # Сортуємо за об'ємом
            usdt_pairs.sort(key=lambda x: markets[x].get('quoteVolume', 0), reverse=True)
            
            return usdt_pairs[:limit]
            
        except Exception as e:
            logger.error(f"Помилка отримання high volume symbols: {e}")
            return []

    async def get_top_gainers(self, limit: int = 30) -> List[str]:
        """Отримання топ gainers"""
        try:
            markets = await asyncio.get_event_loop().run_in_executor(
                self.executor, self.exchange.load_markets
            )
            
            # Фільтруємо та сортуємо за зміною ціни
            usdt_pairs = [
                (symbol, market) for symbol, market in markets.items()
                if symbol.endswith('/USDT') and market.get('active', False)
                and market.get('percentage', 0) is not None
            ]
            
            usdt_pairs.sort(key=lambda x: x[1].get('percentage', 0), reverse=True)
            
            return [pair[0] for pair in usdt_pairs[:limit]]
            
        except Exception as e:
            logger.error(f"Помилка отримання top gainers: {e}")
            return []

    def format_signal_message(self, analysis: Dict, index: int) -> str:
        """Форматування повідомлення про сигнал"""
        symbol = analysis['symbol'].replace('/USDT', '')
        
        return (
            f"{index}. **{symbol}**\n"
            f"   💰 Ціна: ${analysis['price']:.6f}\n"
            f"   📊 Об'єм: ${analysis['volume_usdt']:,.0f}\n"
            f"   🚨 Pump ймовірність: {analysis['pump_probability']:.2%}\n"
            f"   📉 Dump ймовірність: {analysis['dump_probability']:.2%}\n"
            f"   📈 RSI: {analysis['technical_indicators']['rsi']:.1f}\n"
            f"   ⚖️ Imbalance: {analysis['orderbook_metrics']['imbalance']:.3f}\n\n"
        )

    async def analyze_pump_potential(self, symbol: str) -> Dict:
        """Спеціалізований аналіз для пампу"""
        try:
            market_data = await self.get_market_data(symbol)
            orderbook = await self.get_orderbook_depth(symbol)
            klines = await self.get_klines(symbol, '5m', 50)
            
            if not all([market_data, orderbook, klines]):
                return {}
            
            # Аналіз технічних індикаторів
            tech = self.technical_analysis(klines)
            ob_analysis = self.orderbook_analysis(orderbook)
            
            # Розрахунок впевненості в пампі
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
            logger.error(f"Помилка аналізу pump potential для {symbol}: {e}")
            return {}

    async def analyze_dump_potential(self, symbol: str) -> Dict:
        """Спеціалізований аналіз для дампу"""
        try:
            market_data = await self.get_market_data(symbol)
            orderbook = await self.get_orderbook_depth(symbol)
            klines = await self.get_klines(symbol, '5m', 50)
            
            if not all([market_data, orderbook, klines]):
                return {}
            
            tech = self.technical_analysis(klines)
            ob_analysis = self.orderbook_analysis(orderbook)
            
            # Розрахунок впевненості в дампі
            confidence = self.calculate_dump_confidence(tech, ob_analysis, market_data)
            
            return {
                'symbol': symbol.replace('/USDT', ''),
                'dump_confidence': confidence,
                'max_gain': market_data.get('percentage', 0),
                'sell_volume': ob_analysis['total_ask_volume'],
                'whale_sells': ob_analysis['large_asks'],
                'rsi': tech['rsi']
            }
            
        except Exception as e:
            logger.error(f"Помилка аналізу dump potential для {symbol}: {e}")
            return {}

    def calculate_pump_confidence(self, tech: Dict, orderbook: Dict, market_data: Dict) -> float:
        """Розрахунок впевненості в пампі"""
        confidence = 0
        
        # RSI нижче 50 - позитивно для пампу
        if tech['rsi'] < 45:
            confidence += 25
        elif tech['rsi'] < 55:
            confidence += 15
        
        # Сильний імбаланс на покупки
        if orderbook['imbalance'] > 0.2:
            confidence += 30
        elif orderbook['imbalance'] > 0.1:
            confidence += 15
        
        # Великі ордери на покупки
        if orderbook['large_bids'] >= 3:
            confidence += 20
        elif orderbook['large_bids'] >= 1:
            confidence += 10
        
        # Прискорення ціни
        if tech['price_acceleration'] > 0.005:
            confidence += 25
        
        return min(confidence, 100)

    def calculate_dump_confidence(self, tech: Dict, orderbook: Dict, market_data: Dict) -> float:
        """Розрахунок впевненості в дампі"""
        confidence = 0
        
        # RSI вище 70 - ризик дампу
        if tech['rsi'] > 75:
            confidence += 30
        elif tech['rsi'] > 65:
            confidence += 20
        
        # Імбаланс на продажі
        if orderbook['imbalance'] < -0.2:
            confidence += 25
        elif orderbook['imbalance'] < -0.1:
            confidence += 15
        
        # Великі ордери на продаж
        if orderbook['large_asks'] >= 3:
            confidence += 25
        elif orderbook['large_asks'] >= 1:
            confidence += 15
        
        # Висока волотильність
        if tech['volatility'] > 8:
            confidence += 20
        
        return min(confidence, 100)

    async def detect_whale_activity(self) -> List[Dict]:
        """Виявлення активності китів"""
        try:
            symbols = await self.get_high_volume_symbols(limit=20)
            whale_activities = []
            
            for symbol in symbols:
                try:
                    orderbook = await self.get_orderbook_depth(symbol)
                    if not orderbook:
                        continue
                    
                    # Аналізуємо великі ордери
                    large_orders = self.analyze_large_orders(orderbook)
                    if large_orders:
                        whale_activities.extend(large_orders)
                    
                    await asyncio.sleep(0.1)
                except Exception as e:
                    continue
            
            return whale_activities[:10]  # Обмежуємо кількість результатів
            
        except Exception as e:
            logger.error(f"Помилка виявлення активності китів: {e}")
            return []

    def analyze_large_orders(self, orderbook: Dict) -> List[Dict]:
        """Аналіз великих ордерів"""
        large_orders = []
        threshold = self.detection_params['large_order_threshold']
        
        # Аналіз bids (покупки)
        for bid in orderbook.get('bids', []):
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
        
        # Аналіз asks (продажі)
        for ask in orderbook.get('asks', []):
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
        
        return large_orders

    def calculate_trend_strength(self, prices: np.ndarray) -> float:
        """Розрахунок сили тренду"""
        if len(prices) < 10:
            return 0.5
        
        # Linear regression для визначення тренду
        x = np.arange(len(prices))
        slope, _, r_value, _, _ = stats.linregress(x, prices)
        
        # Сила тренду based on R-squared
        trend_strength = abs(r_value ** 2)
        
        # Напрямок тренду
        direction = 1 if slope > 0 else -1
        
        return round(trend_strength * direction, 3)

    def calculate_price_change(self, klines: List, minutes: int) -> float:
        """Розрахунок зміни ціни за вказаний період"""
        if len(klines) < minutes + 1:
            return 0.0
        
        current_price = float(klines[-1][4])
        past_price = float(klines[-minutes-1][4])
        
        if past_price == 0:
            return 0.0
        
        return ((current_price - past_price) / past_price) * 100

    async def get_market_data(self, symbol: str) -> Optional[Dict]:
        """Отримання ринкових даних"""
        try:
            ticker = await asyncio.get_event_loop().run_in_executor(
                self.executor, self.exchange.fetch_ticker, symbol
            )
            return {
                'symbol': symbol,
                'open': ticker['open'],
                'high': ticker['high'],
                'low': ticker['low'],
                'close': ticker['last'],
                'volume': ticker['quoteVolume'],
                'percentage': ticker['percentage']
            }
        except Exception as e:
            logger.error(f"Помилка отримання market data для {symbol}: {e}")
            return None

    async def get_orderbook_depth(self, symbol: str, limit: int = 100) -> Optional[Dict]:
        """Отримання глибини ринку"""
        try:
            orderbook = await asyncio.get_event_loop().run_in_executor(
                self.executor, self.exchange.fetch_order_book, symbol, limit
            )
            orderbook['symbol'] = symbol
            return orderbook
        except Exception as e:
            logger.error(f"Помилка отримання orderbook для {symbol}: {e}")
            return None

    async def get_klines(self, symbol: str, timeframe: str = '5m', limit: int = 100) -> List:
        """Отримання історичних даних"""
        try:
            klines = await asyncio.get_event_loop().run_in_executor(
                self.executor, self.exchange.fetch_ohlcv, symbol, timeframe, limit
            )
            return klines
        except Exception as e:
            logger.error(f"Помилка отримання klines для {symbol}: {e}")
            return []

    async def liquidity_scan_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Сканування ліквідності"""
        try:
            msg = await update.message.reply_text("💧 АНАЛІЗУЮ ЛІКВІДНІСТЬ...")
            
            symbols = await self.get_active_symbols(limit=40)
            liquidity_data = []
            
            for symbol in symbols:
                try:
                    orderbook = await self.get_orderbook_depth(symbol)
                    if orderbook:
                        liquidity_score = self.calculate_liquidity_score(orderbook)
                        liquidity_data.append({
                            'symbol': symbol.replace('/USDT', ''),
                            'liquidity_score': liquidity_score,
                            'bid_volume': orderbook.get('total_bid_volume', 0),
                            'ask_volume': orderbook.get('total_ask_volume', 0)
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

    def calculate_liquidity_score(self, orderbook: Dict) -> float:
        """Розрахунок оцінки ліквідності"""
        try:
            bid_volume = orderbook.get('total_bid_volume', 0)
            ask_volume = orderbook.get('total_ask_volume', 0)
            total_volume = bid_volume + ask_volume
            
            if total_volume == 0:
                return 0.0
            
            # Базова оцінка на основі об'єму
            volume_score = min(total_volume / 1000000, 1.0)  # Нормалізуємо до 1.0
            
            # Оцінка на основі спреду
            if orderbook['bids'] and orderbook['asks']:
                best_bid = float(orderbook['bids'][0][0])
                best_ask = float(orderbook['asks'][0][0])
                spread = (best_ask - best_bid) / best_bid
                spread_score = 1.0 - min(spread / 0.01, 1.0)  # Менше 1% спреду - добре
            else:
                spread_score = 0.5
            
            # Фінальна оцінка
            return round((volume_score * 0.6 + spread_score * 0.4), 3)
            
        except Exception as e:
            return 0.0

    async def volatility_alert_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Сигнали волатильності"""
        try:
            msg = await update.message.reply_text("⚡ ШУКАЮ ВОЛАТИЛЬНІСТЬ...")
            
            symbols = await self.get_active_symbols(limit=50)
            volatile_symbols = []
            
            for symbol in symbols:
                try:
                    klines = await self.get_klines(symbol, '5m', 20)
                    if klines:
                        volatility = self.calculate_volatility(klines)
                        if volatility > self.detection_params['volatility_spike_threshold']:
                            volatile_symbols.append({
                                'symbol': symbol.replace('/USDT', ''),
                                'volatility': volatility,
                                'price': float(klines[-1][4])
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

    def calculate_volatility(self, klines: List) -> float:
        """Розрахунок волатильності"""
        try:
            closes = np.array([float(k[4]) for k in klines])
            returns = np.diff(closes) / closes[:-1]
            volatility = np.std(returns) * 100 * np.sqrt(365)  # Річна волатильність в %
            return round(volatility, 2)
        except Exception as e:
            return 0.0

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
            
            # Аналізуємо топ 20 монет за об'ємом
            symbols = await self.get_high_volume_symbols(limit=20)
            quick_signals = []
            
            for symbol in symbols:
                try:
                    # Швидкий аналіз без глибоких перевірок
                    market_data = await self.get_market_data(symbol)
                    orderbook = await self.get_orderbook_depth(symbol, 50)
                    
                    if market_data and orderbook:
                        # Швидкі перевірки для pump/dump
                        is_potential = self.quick_pump_check(market_data, orderbook)
                        if is_potential:
                            quick_signals.append({
                                'symbol': symbol.replace('/USDT', ''),
                                'price': market_data['close'],
                                'volume': market_data['volume'],
                                'change': market_data['percentage'],
                                'imbalance': self.calculate_orderbook_imbalance(orderbook)
                            })
                    
                    await asyncio.sleep(0.05)  # Дуже швидкий аналіз
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

    def quick_pump_check(self, market_data: Dict, orderbook: Dict) -> bool:
        """Швидка перевірка на потенційний pump"""
        # Перевірка об'єму
        if market_data['volume'] < self.detection_params['min_volume_usdt']:
            return False
        
        # Перевірка імбалансу
        imbalance = self.calculate_orderbook_imbalance(orderbook)
        if abs(imbalance) < self.detection_params['orderbook_imbalance_min']:
            return False
        
        # Перевірка великих ордерів
        large_bids = sum(1 for bid in orderbook.get('bids', []) 
                        if float(bid[0]) * float(bid[1]) > self.detection_params['large_order_threshold'])
        
        return large_bids >= 2 and imbalance > 0

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
        except Exception as e:
            return 0.0

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
                f"• Середній час відгуку: {self.performance_metrics['avg_response_time']:.2f}с"
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
                # Тут може бути логіка оновлення даних
                await asyncio.sleep(1)
                await self.start_command(query, context)
                
        except Exception as e:
            logger.error(f"Помилка обробки кнопки {query.data}: {e}")
            await query.edit_message_text("❌ Помилка обробки запиту")

    def is_garbage_symbol(self, symbol: str) -> bool:
        """Перевірка чи символ є непотрібним"""
        symbol_clean = symbol.upper().replace('/USDT', '').replace('USDT', '')
        
        if symbol_clean in self.garbage_symbols:
            return True
        
        if len(symbol_clean) > 12:
            return True
        
        if any(char.isdigit() for char in symbol_clean):
            return True
            
        garbage_patterns = ['UP', 'DOWN', 'BULL', 'BEAR', 'USD', 'EUR', 'BTC', 'ETH', 'BNB']
        if any(symbol_clean.endswith(pattern) for pattern in garbage_patterns):
            return True
            
        return False

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
            
            # Просто чекаємо безкінечно замість idle()
            while True:
                await asyncio.sleep(3600)  # Спимо 1 годину
            
        except Exception as e:
            logger.error(f"❌ Помилка запуску бота: {e}")
            raise

    async def background_monitoring(self):
        """Фоновий моніторинг ринку"""
        while True:
            try:
                # Оновлюємо кеш даних кожні 5 хвилин
                await self.update_market_data_cache()
                await asyncio.sleep(300)  # 5 хвилин
                
            except Exception as e:
                logger.error(f"Помилка фонового моніторингу: {e}")
                await asyncio.sleep(60)

    async def update_market_data_cache(self):
        """Оновлення кешу ринкових даних"""
        try:
            symbols = await self.get_active_symbols(limit=30)
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
            if current_time - data['timestamp'] > 600:  # 10 хвилин
                old_keys.append(symbol)
        
        for key in old_keys:
            del self.market_data_cache[key]
        
        if old_keys:
            logger.info(f"🧹 Очищено {len(old_keys)} застарілих записів кешу")

    async def emergency_shutdown(self):
        """Аварійне вимкнення бота"""
        logger.warning("🛑 Аварійне вимкнення бота...")
        await self.app.stop()
        await self.app.shutdown()
        self.executor.shutdown(wait=False)

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

    async def emergency_scan(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Екстрене сканування для критичних ситуацій"""
        try:
            msg = await update.message.reply_text("🚨 ЕКСТРЕНЕ СКАНУВАННЯ!...")
            
            # Швидкий аналіз топ-10 монет
            symbols = await self.get_high_volume_symbols(limit=10)
            critical_signals = []
            
            for symbol in symbols:
                try:
                    # Дуже швидкий аналіз
                    market_data = await self.get_market_data(symbol)
                    if not market_data:
                        continue
                    
                    # Перевірка на критичні зміни
                    if abs(market_data['percentage']) > 15:  # Дуже різкі зміни
                        orderbook = await self.get_orderbook_depth(symbol, 50)
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
                    
                    await asyncio.sleep(0.05)
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

    async def smart_alert_system(self):
        """Розумна система сповіщень"""
        while True:
            try:
                # Періодична перевірка на критичні ситуації
                symbols = await self.get_active_symbols(limit=15)
                
                for symbol in symbols:
                    try:
                        market_data = await self.get_market_data(symbol)
                        if market_data and abs(market_data['percentage']) > 20:
                            # Знайдено критичну зміну - можна додати сповіщення
                            logger.warning(f"🚨 Критична зміна: {symbol} - {market_data['percentage']}%")
                            
                    except Exception as e:
                        continue
                    
                    await asyncio.sleep(0.1)
                
                await asyncio.sleep(60)  # Перевіряємо кожну хвилину
                
            except Exception as e:
                logger.error(f"Помилка smart alert system: {e}")
                await asyncio.sleep(30)

    def calculate_market_cap(self, symbol: str, price: float) -> Optional[float]:
        """Спрощений розрахунок ринкової капіталізації"""
        try:
            # Для спрощення, використовуємо об'єм як проксі
            markets = self.exchange.load_markets()
            market_info = markets.get(symbol, {})
            
            if market_info and 'info' in market_info:
                circulating_supply = float(market_info['info'].get('circulatingSupply', 0))
                if circulating_supply > 0:
                    return price * circulating_supply
            
            return None
        except Exception as e:
            return None

    async def risk_assessment(self, symbol: str) -> Dict:
        """Оцінка ризику для монети"""
        try:
            market_data = await self.get_market_data(symbol)
            if not market_data:
                return {'risk_level': 'UNKNOWN', 'score': 0}
            
            risk_score = 0
            risk_factors = []
            
            # Волатильність
            if abs(market_data['percentage']) > 25:
                risk_score += 30
                risk_factors.append('Висока волатильність')
            
            # Об'єм торгів
            if market_data['volume'] > 10000000:  # Дуже високий об'єм
                risk_score += 20
                risk_factors.append('Надмірний обсяг торгів')
            
            # Визначення рівня ризику
            if risk_score >= 40:
                risk_level = 'ВИСОКИЙ'
            elif risk_score >= 20:
                risk_level = 'ПОМІРНИЙ'
            else:
                risk_level = 'НИЗЬКИЙ'
            
            return {
                'risk_level': risk_level,
                'score': risk_score,
                'factors': risk_factors,
                'symbol': symbol.replace('/USDT', '')
            }
            
        except Exception as e:
            return {'risk_level': 'UNKNOWN', 'score': 0, 'factors': []}

# Глобальні утиліти
def format_large_number(number: float) -> str:
    """Форматування великих чисел"""
    if number >= 1_000_000:
        return f"{number/1_000_000:.2f}M"
    elif number >= 1_000:
        return f"{number/1_000:.2f}K"
    else:
        return f"{number:.2f}"

def calculate_percentage_change(old_price: float, new_price: float) -> float:
    """Розрахунок відсоткової зміни"""
    if old_price == 0:
        return 0.0
    return ((new_price - old_price) / old_price) * 100

def is_market_open() -> bool:
    """Перевірка чи ринок відкритий (проста реалізація)"""
    now = datetime.now()
    # Припускаємо, що ринок відкритий 24/7 для криптовалют
    return True

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
        
        # Додаємо emergency команду
        bot.app.add_handler(CommandHandler("emergency", bot.emergency_scan))
        
        logger.info("🚀 Запускаю бота...")
        
        # Запускаємо бота в асинхронному режимі
        import asyncio
        asyncio.run(bot.run())
        
    except KeyboardInterrupt:
        logger.info("⏹️ Зупинка бота...")
        # Зберігаємо стан перед виходом
        bot.save_state()
    except Exception as e:
        logger.error(f"❌ Критична помилка: {e}")
        # Спроба зберегти стан навіть при помилці
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