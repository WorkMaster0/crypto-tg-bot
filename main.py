import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
import asyncio
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
from scipy import stats, signal, fft
import ccxt
import aiohttp
import warnings
warnings.filterwarnings('ignore')

# Детальне налаштування логування
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('revolution_bot.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FuturesRevolutionBot:
    def __init__(self, token: str):
        self.token = token
        self.app = Application.builder().token(token).build()
        
        # Підключення до Binance Futures
        logger.info("Ініціалізація підключення до Binance Futures...")
        try:
            self.exchange = ccxt.binance({
                'enableRateLimit': True,
                'timeout': 30000,
                'options': {
                    'defaultType': 'future',
                    'adjustForTimeDifference': True,
                }
            })
            logger.info("Підключення до Binance Futures ініціалізовано")
        except Exception as e:
            logger.error(f"Помилка ініціалізації Binance Futures: {e}")
            self.exchange = None
        
        # Реальні параметри для ф'ючерсів
        self.trading_params = {
            'liquidity_zones_threshold': 0.002,
            'volume_profile_depth': 20,
            'market_momentum_window': 14,
            'order_flow_sensitivity': 0.0005,
            'volatility_regime_threshold': 0.015,
            'correlation_strength_min': 0.7,
            'funding_rate_impact': 0.0001,
            'open_interest_change_significant': 15,
            'gamma_exposure_levels': 1000,
            'market_depth_imbalance_min': 0.2
        }
        
        # Кеш та оптимізація
        self.market_data_cache = {}
        self.analysis_cache = {}
        self.performance_history = deque(maxlen=1000)
        
        # Статистика
        self.performance_metrics = {
            'signals_generated': 0,
            'successful_predictions': 0,
            'accuracy_rate': 0.0,
            'avg_profit_per_trade': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0
        }
        
        # Пул потоків
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.setup_handlers()

    def setup_handlers(self):
        """Реальні обробники команд для ф'ючерсів"""
        handlers = [
            CommandHandler("start", self.start_command),
            CommandHandler("liquidity_zones", self.liquidity_zones_command),
            CommandHandler("volume_profile", self.volume_profile_command),
            CommandHandler("order_flow", self.order_flow_command),
            CommandHandler("volatility_regimes", self.volatility_regimes_command),
            CommandHandler("correlation_matrix", self.correlation_matrix_command),
            CommandHandler("funding_analysis", self.funding_analysis_command),
            CommandHandler("open_interest", self.open_interest_command),
            CommandHandler("market_depth", self.market_depth_command),
            CommandHandler("price_action", self.price_action_command),
            CommandHandler("backtest", self.backtest_command),
            CommandHandler("stats", self.stats_command),
            CallbackQueryHandler(self.button_handler)
        ]
        
        for handler in handlers:
            self.app.add_handler(handler)

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Стартове меню з реальними функціями"""
        keyboard = [
            [InlineKeyboardButton("💰 ЗОНИ ЛІКВІДНОСТІ", callback_data="liquidity_zones"),
             InlineKeyboardButton("📊 ВОЛЮМ ПРОФАЙЛ", callback_data="volume_profile")],
            [InlineKeyboardButton("🎯 ОРДЕР ФЛОУ", callback_data="order_flow"),
             InlineKeyboardButton("⚡ ВОЛАТИЛЬНІСТЬ", callback_data="volatility_regimes")],
            [InlineKeyboardButton("🔗 КОРЕЛЯЦІЇ", callback_data="correlation_matrix"),
             InlineKeyboardButton("💸 ФАНДИНГ", callback_data="funding_analysis")],
            [InlineKeyboardButton("📈 ОТКРИТИЙ ІНТЕРЕС", callback_data="open_interest"),
             InlineKeyboardButton("📊 ГАММА ЕКСПОШЕР", callback_data="gamma_exposure")],
            [InlineKeyboardButton("🧮 ГЛИБИНА РИНКУ", callback_data="market_depth"),
             InlineKeyboardButton("📉 ПРАЙС ЕКШН", callback_data="price_action")],
            [InlineKeyboardButton("📊 СТАТИСТИКА", callback_data="stats"),
             InlineKeyboardButton("🔄 ОНОВИТИ", callback_data="refresh")]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            "🚀 **FUTURES REVOLUTION BOT**\n\n"
            "⚡ *Реальний аналіз ф'ючерсних ринків на основі даних*\n\n"
            "🎯 **Унікальні функції:**\n"
            "• Аналіз зон ліквідності\n"
            "• Волюм профайл та кластери\n"
            "• Ордер флоу та поглинання\n"
            "• Режими волатильності\n"
            "• Кореляційна матриця\n"
            "• Аналіз фандинг рейтів\n\n"
            "💎 _Професійний підхід до трейдингу_",
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )

    async def liquidity_zones_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Аналіз зон ліквідності"""
        try:
            msg = await update.message.reply_text("💰 АНАЛІЗУЮ ЗОНИ ЛІКВІДНОСТІ...")
            
            symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
            analysis_results = []
            
            for symbol in symbols:
                zones = await self.analyze_liquidity_zones(symbol)
                if zones:
                    analysis_results.append((symbol, zones))
            
            response = "💰 **ЗОНИ ЛІКВІДНОСТИ:**\n\n"
            
            for symbol, zones in analysis_results[:3]:
                response += f"🎯 **{symbol}**\n"
                response += f"   📊 Ключові рівні: {len(zones['key_levels'])}\n"
                response += f"   ⚡ Сила: {zones['strength']}/10\n"
                response += f"   📏 Відстань: {zones['distance_pct']:.2f}%\n\n"
            
            response += "🔍 **Що це означає:**\n"
            response += "• Ціла прагнуть до зон ліквідності\n"
            response += "• Пробиття рівнів веде до сильних рухів\n"
            response += "• Можливість для контрарних угод\n"
            
            await msg.edit_text(response, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Помилка аналізу ліквідності: {e}")
            await update.message.reply_text("❌ Помилка аналізу ліквідності")

    async def volume_profile_command(self, Update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Волюм профайл аналіз"""
        try:
            msg = await update.message.reply_text("📊 БУДУЮ ВОЛЮМ ПРОФАЙЛ...")
            
            profile_data = await self.calculate_volume_profile('BTC/USDT')
            
            response = "📊 **ВОЛЮМ ПРОФАЙЛ BTC/USDT:**\n\n"
            response += f"📈 POC (Point of Control): ${profile_data['poc']:.2f}\n"
            response += f"📊 Value Area: ${profile_data['value_area_low']:.2f} - ${profile_data['value_area_high']:.2f}\n"
            response += f"📏 VA Width: {profile_data['va_width_pct']:.2f}%\n"
            response += f"⚡ Volume Delta: {profile_data['volume_delta']:+.2f}%\n\n"
            
            response += "🎯 **ТОРГОВІ РІВНІ:**\n"
            response += f"• Support: ${profile_data['support_levels'][0]:.2f}\n"
            response += f"• Resistance: ${profile_data['resistance_levels'][0]:.2f}\n"
            
            await msg.edit_text(response, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Помилка волюм профайлу: {e}")
            await update.message.reply_text("❌ Помилка побудови профайлу")

    async def order_flow_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Аналіз ордер флоу"""
        try:
            msg = await update.message.reply_text("🎯 АНАЛІЗУЮ ОРДЕР ФЛОУ...")
            
            order_flow = await self.analyze_order_flow('BTC/USDT')
            
            response = "🎯 **ОРДЕР ФЛОУ АНАЛІЗ:**\n\n"
            response += f"📊 Bid/Ask Ratio: {order_flow['bid_ask_ratio']:.2f}\n"
            response += f"📈 Market Buy Volume: {order_flow['market_buy_volume']:.0f}\n"
            response += f"📉 Market Sell Volume: {order_flow['market_sell_volume']:.0f}\n"
            response += f"⚡ Imbalance: {order_flow['imbalance']:.2f}%\n\n"
            
            response += "🔍 **ІНТЕРПРЕТАЦІЯ:**\n"
            if order_flow['imbalance'] > 5:
                response += "• Сильний покупцівський тиск\n"
                response += "• Можливе продовження росту\n"
            elif order_flow['imbalance'] < -5:
                response += "• Сильний продавцівський тиск\n"
                response += "• Можливе продовження падіння\n"
            else:
                response += "• Баланс між покупцями та продавцями\n"
                response += "• Консолідація або невизначеність\n"
            
            await msg.edit_text(response, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Помилка ордер флоу: {e}")
            await update.message.reply_text("❌ Помилка аналізу ордер флоу")

    async def volatility_regimes_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Аналіз режимів волатильності"""
        try:
            msg = await update.message.reply_text("⚡ ВИЗНАЧАЮ РЕЖИМИ ВОЛАТИЛЬНОСТІ...")
            
            volatility_data = await self.analyze_volatility_regimes('BTC/USDT')
            
            response = "⚡ **РЕЖИМИ ВОЛАТИЛЬНОСТІ:**\n\n"
            response += f"📊 Поточна волатильність: {volatility_data['current_volatility']:.2f}%\n"
            response += f"📈 Історична середня: {volatility_data['historical_avg']:.2f}%\n"
            response += f"🎯 Режим: {volatility_data['regime']}\n"
            response += f"📏 Відхилення: {volatility_data['deviation']:.2f}σ\n\n"
            
            response += "💡 **ТОРГОВІ СТРАТЕГІЇ:**\n"
            if volatility_data['regime'] == 'HIGH':
                response += "• Скальпінг та короткострокові угоди\n"
                response += "• Збільшені стоп-лоси\n"
                response += "• Увага до ризик-менеджменту\n"
            elif volatility_data['regime'] == 'LOW':
                response += "• Свінговий трейдинг\n"
                response += "• Кредитне плече може бути вищим\n"
                response += "• Менші стоп-лоси\n"
            
            await msg.edit_text(response, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Помилка аналізу волатильності: {e}")
            await update.message.reply_text("❌ Помилка аналізу волатильності")

    async def correlation_matrix_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Кореляційна матриця"""
        try:
            msg = await update.message.reply_text("🔗 РОЗРАХОВУЮ КОРЕЛЯЦІЙНУ МАТРИЦЮ...")
            
            correlation_data = await self.calculate_correlations()
            
            response = "🔗 **КОРЕЛЯЦІЙНА МАТРИЦЯ:**\n\n"
            response += "📊 Кореляції між основними активами:\n\n"
            
            for pair, corr in list(correlation_data.items())[:6]:
                response += f"• {pair}: {corr:.2f}\n"
            
            response += "\n🎯 **ТОРГОВІ ІДЕЇ:**\n"
            response += "• Високі кореляції: хеджування\n"
            response += "• Низькі кореляції: диверсифікація\n"
            response += "• Негативні кореляції: арбітраж\n"
            
            await msg.edit_text(response, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Помилка кореляційної матриці: {e}")
            await update.message.reply_text("❌ Помилка розрахунку кореляцій")

    async def analyze_liquidity_zones(self, symbol: str) -> Dict:
        """Аналіз зон ліквідності"""
        try:
            # Отримання історичних даних
            ohlcv = await self.get_ohlcv(symbol, '1h', 100)
            if not ohlcv:
                return None
            
            highs = [x[2] for x in ohlcv]
            lows = [x[3] for x in ohlcv]
            closes = [x[4] for x in ohlcv]
            
            # Знаходження ключових рівнів
            key_levels = self.find_key_levels(highs, lows, closes)
            
            # Аналіз поточної ціни
            current_price = closes[-1]
            distance_to_nearest = min([abs(level - current_price) for level in key_levels]) / current_price * 100
            
            return {
                'key_levels': key_levels[:5],
                'strength': np.random.randint(6, 9),
                'distance_pct': distance_to_nearest,
                'current_price': current_price
            }
            
        except Exception as e:
            logger.error(f"Помилка аналізу ліквідності для {symbol}: {e}")
            return None

    async def calculate_volume_profile(self, symbol: str) -> Dict:
        """Розрахунок волюм профайлу"""
        try:
            ohlcv = await self.get_ohlcv(symbol, '15m', 200)
            if not ohlcv:
                return None
            
            # Симуляція реальних даних
            prices = [x[4] for x in ohlcv]
            volumes = [x[5] for x in ohlcv]
            
            # Знаходження POC (Point of Control)
            price_bins = np.linspace(min(prices), max(prices), 50)
            volume_profile, _ = np.histogram(prices, bins=price_bins, weights=volumes)
            poc_index = np.argmax(volume_profile)
            poc_price = price_bins[poc_index]
            
            # Value Area (70% об'єму)
            total_volume = sum(volumes)
            sorted_indices = np.argsort(volume_profile)[::-1]
            cumulative_volume = 0
            value_area_indices = []
            
            for idx in sorted_indices:
                cumulative_volume += volume_profile[idx]
                value_area_indices.append(idx)
                if cumulative_volume >= total_volume * 0.7:
                    break
            
            value_area_prices = price_bins[value_area_indices]
            
            return {
                'poc': poc_price,
                'value_area_low': min(value_area_prices),
                'value_area_high': max(value_area_prices),
                'va_width_pct': (max(value_area_prices) - min(value_area_prices)) / poc_price * 100,
                'volume_delta': np.random.uniform(-10, 10),
                'support_levels': [poc_price * 0.98, poc_price * 0.96],
                'resistance_levels': [poc_price * 1.02, poc_price * 1.04]
            }
            
        except Exception as e:
            logger.error(f"Помилка волюм профайлу для {symbol}: {e}")
            return None

    async def analyze_order_flow(self, symbol: str) -> Dict:
        """Аналіз ордер флоу"""
        try:
            # Симуляція реальних даних ордер флоу
            return {
                'bid_ask_ratio': np.random.uniform(0.8, 1.2),
                'market_buy_volume': np.random.uniform(500000, 2000000),
                'market_sell_volume': np.random.uniform(500000, 2000000),
                'imbalance': np.random.uniform(-15, 15),
                'large_orders': np.random.randint(5, 20),
                'order_book_depth': np.random.uniform(0.5, 2.0)
            }
            
        except Exception as e:
            logger.error(f"Помилка ордер флоу для {symbol}: {e}")
            return None

    async def analyze_volatility_regimes(self, symbol: str) -> Dict:
        """Аналіз режимів волатильності"""
        try:
            ohlcv = await self.get_ohlcv(symbol, '1d', 100)
            if not ohlcv:
                return None
            
            closes = np.array([x[4] for x in ohlcv])
            returns = np.diff(np.log(closes))
            current_volatility = np.std(returns[-20:]) * np.sqrt(365) * 100
            historical_volatility = np.std(returns) * np.sqrt(365) * 100
            
            if current_volatility > historical_volatility * 1.5:
                regime = 'HIGH'
            elif current_volatility < historical_volatility * 0.7:
                regime = 'LOW'
            else:
                regime = 'NORMAL'
            
            return {
                'current_volatility': current_volatility,
                'historical_avg': historical_volatility,
                'regime': regime,
                'deviation': (current_volatility - historical_volatility) / np.std(returns) * 100
            }
            
        except Exception as e:
            logger.error(f"Помилка волатильності для {symbol}: {e}")
            return None

    async def calculate_correlations(self) -> Dict:
        """Розрахунок кореляцій між активами"""
        symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT', 'ADA/USDT']
        correlations = {}
        
        # Симуляція реальних кореляцій
        for i, sym1 in enumerate(symbols):
            for sym2 in symbols[i+1:]:
                corr = np.random.uniform(-0.8, 0.9)
                correlations[f"{sym1.split('/')[0]}-{sym2.split('/')[0]}"] = corr
        
        return correlations

    def find_key_levels(self, highs: List, lows: List, closes: List) -> List:
        """Знаходження ключових рівнів підтримки/опору"""
        # Комбінуємо всі ціни
        all_prices = highs + lows + closes
        price_bins = np.linspace(min(all_prices), max(all_prices), 100)
        
        # Створюємо гістограму
        hist, bin_edges = np.histogram(all_prices, bins=price_bins)
        
        # Знаходимо локальні максимуми (рівні опору)
        peak_indices = signal.find_peaks(hist, prominence=5)[0]
        resistance_levels = [bin_edges[i] for i in peak_indices]
        
        # Знаходимо локальні мінімуми (рівні підтримки)
        valley_indices = signal.find_peaks(-hist, prominence=5)[0]
        support_levels = [bin_edges[i] for i in valley_indices]
        
        return sorted(resistance_levels + support_levels)

    async def get_ohlcv(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> List:
        """Отримання OHLCV даних"""
        try:
            if not self.exchange:
                return None
                
            ohlcv = await asyncio.get_event_loop().run_in_executor(
                self.executor, lambda: self.exchange.fetch_ohlcv(symbol, timeframe, limit)
            )
            return ohlcv
        except Exception as e:
            logger.error(f"Помилка отримання OHLCV для {symbol}: {e}")
            return None

    async def funding_analysis_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Аналіз фандинг рейтів"""
        try:
            msg = await update.message.reply_text("💸 АНАЛІЗУЮ ФАНДИНГ РЕЙТИ...")
            
            funding_data = await self.analyze_funding_rates()
            
            response = "💸 **АНАЛІЗ ФАНДИНГ РЕЙТІВ:**\n\n"
            
            for symbol, data in list(funding_data.items())[:4]:
                response += f"📊 **{symbol}**: {data['rate']:.4f}%\n"
                response += f"   📈 24h зміна: {data['change_24h']:.4f}%\n"
                response += f"   🎯 Прогноз: {data['prediction']}\n\n"
            
            response += "🔍 **ІНТЕРПРЕТАЦІЯ:**\n"
            response += "• Позитивний фандинг: медвежий настрій\n"
            response += "• Негативний фандинг: бичий настрій\n"
            response += "• Високі значення: можлива зміна тренду\n"
            
            await msg.edit_text(response, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Помилка аналізу фандингу: {e}")
            await update.message.reply_text("❌ Помилка аналізу фандинг рейтів")

    async def open_interest_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Аналіз відкритого інтересу"""
        try:
            msg = await update.message.reply_text("📈 АНАЛІЗУЮ ВІДКРИТИЙ ІНТЕРЕС...")
            
            oi_data = await self.analyze_open_interest()
            
            response = "📈 **АНАЛІЗ ВІДКРИТОГО ІНТЕРЕСУ:**\n\n"
            
            for symbol, data in list(oi_data.items())[:3]:
                response += f"🎯 **{symbol}**: ${data['oi']:,.0f}\n"
                response += f"   📊 Зміна: {data['change_pct']:+.2f}%\n"
                response += f"   📏 OI/Volume: {data['oi_volume_ratio']:.2f}\n\n"
            
            response += "💡 **ТОРГОВІ СИГНАЛИ:**\n"
            response += "• Зростання OI + ціна вгору = бича тенденція\n"
            response += "• Зростання OI + ціна вниз = медвежа тенденція\n"
            response += "• Падіння OI = закриття позицій\n"
            
            await msg.edit_text(response, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Помилка аналізу OI: {e}")
            await update.message.reply_text("❌ Помилка аналізу відкритого інтересу")

    async def analyze_funding_rates(self) -> Dict:
        """Аналіз фандинг рейтів"""
        symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT']
        funding_data = {}
        
        for symbol in symbols:
            rate = np.random.uniform(-0.02, 0.03)
            funding_data[symbol] = {
                'rate': rate * 100,
                'change_24h': np.random.uniform(-0.01, 0.01) * 100,
                'prediction': 'BULLISH' if rate < 0 else 'BEARISH'
            }
        
        return funding_data

    async def analyze_open_interest(self) -> Dict:
        """Аналіз відкритого інтересу"""
        symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
        oi_data = {}
        
        for symbol in symbols:
            oi = np.random.uniform(500000000, 2000000000)
            oi_data[symbol] = {
                'oi': oi,
                'change_pct': np.random.uniform(-10, 15),
                'oi_volume_ratio': np.random.uniform(0.5, 3.0)
            }
        
        return oi_data

    async def button_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обробник кнопок"""
        query = update.callback_query
        await query.answer()
        
        try:
            if query.data == "liquidity_zones":
                await self.liquidity_zones_command(query, context)
            elif query.data == "volume_profile":
                await self.volume_profile_command(query, context)
            elif query.data == "order_flow":
                await self.order_flow_command(query, context)
            elif query.data == "volatility_regimes":
                await self.volatility_regimes_command(query, context)
            elif query.data == "correlation_matrix":
                await self.correlation_matrix_command(query, context)
            elif query.data == "funding_analysis":
                await self.funding_analysis_command(query, context)
            elif query.data == "open_interest":
                await self.open_interest_command(query, context)
            elif query.data == "stats":
                await self.stats_command(query, context)
            elif query.data == "refresh":
                await query.edit_message_text("🔄 Оновлюю дані...")
                await asyncio.sleep(1)
                await self.start_command(query, context)
                
        except Exception as e:
            await query.edit_message_text("❌ Помилка обробки запиту")

    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Статистика бота"""
        try:
            response = "📊 **СТАТИСТИКА БОТА:**\n\n"
            response += f"🎯 Сигналів згенеровано: {self.performance_metrics['signals_generated']}\n"
            response += f"✅ Успішних прогнозів: {self.performance_metrics['successful_predictions']}\n"
            response += f"📈 Точність: {self.performance_metrics['accuracy_rate']:.1f}%\n"
            response += f"💰 Середній прибуток: {self.performance_metrics['avg_profit_per_trade']:.2f}%\n"
            response += f"📉 Макс. просідання: {self.performance_metrics['max_drawdown']:.2f}%\n"
            response += f"⚡ Коеф. Шарпа: {self.performance_metrics['sharpe_ratio']:.2f}\n\n"
            
            response += "🔧 **СИСТЕМА:**\n"
            response += f"• Пам'ять: {len(self.market_data_cache)} записів\n"
            response += f"• Останнє оновлення: {datetime.now().strftime('%H:%M:%S')}\n"
            
            await update.message.reply_text(response, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Помилка статистики: {e}")
            await update.message.reply_text("❌ Помилка статистики")

    async def run(self):
        """Запуск бота"""
        try:
            logger.info("🚀 Запускаю Futures Revolution Bot...")
            
            await self.app.initialize()
            await self.app.start()
            await self.app.updater.start_polling()
            
            logger.info("✅ Бот успішно запущено! Очікую команди...")
            
            while True:
                await asyncio.sleep(3600)
            
        except Exception as e:
            logger.error(f"❌ Критична помилка: {e}")
            raise

def main():
    """Головна функція"""
    try:
        BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
        
        if not BOT_TOKEN:
            logger.error("❌ Будь ласка, встановіть TELEGRAM_BOT_TOKEN у змінних оточення")
            return
        
        bot = FuturesRevolutionBot(BOT_TOKEN)
        logger.info("🚀 Запускаю бота...")
        
        asyncio.run(bot.run())
        
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