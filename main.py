import pandas as pd
import numpy as np
import ccxt
import asyncio
import time
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
import talib
from scipy import stats, signal
import warnings
warnings.filterwarnings('ignore')

# Налаштування логування
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('real_analysis_bot.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RealAnalysisBot:
    def __init__(self, token: str):
        self.token = token
        self.app = Application.builder().token(token).build()
        
        # Підключення до біржі
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })
        
        # Кеш даних
        self.market_data = {}
        self.last_analysis = {}
        self.analysis_timestamp = {}
        
        self.setup_handlers()
        logger.info("Real Analysis Bot ініціалізовано")

    def setup_handlers(self):
        """Реальні команди аналізу"""
        handlers = [
            CommandHandler("start", self.start_command),
            CommandHandler("analyze", self.analyze_command),
            CommandHandler("scan", self.scan_market_command),
            CommandHandler("levels", self.key_levels_command),
            CommandHandler("volume", self.volume_analysis_command),
            CommandHandler("momentum", self.momentum_analysis_command),
            CommandHandler("correlation", self.correlation_analysis_command),
            CommandHandler("opportunities", self.opportunities_command),
            CallbackQueryHandler(self.handle_callback)
        ]
        
        for handler in handlers:
            self.app.add_handler(handler)

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Стартова команда"""
        keyboard = [
            [InlineKeyboardButton("🔍 Аналізувати ринок", callback_data="analyze"),
             InlineKeyboardButton("📊 Сканувати", callback_data="scan")],
            [InlineKeyboardButton("🎯 Ключові рівні", callback_data="levels"),
             InlineKeyboardButton("📈 Аналіз об'ємів", callback_data="volume")],
            [InlineKeyboardButton("⚡ Моментум", callback_data="momentum"),
             InlineKeyboardButton("🔗 Кореляції", callback_data="correlation")],
            [InlineKeyboardButton("💰 Можливості", callback_data="opportunities")]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            "📊 **РЕАЛЬНИЙ АНАЛІТИЧНИЙ БОТ**\n\n"
            "Аналіз ринку в реальному часі на основі даних\n\n"
            "⚡ *Останній аналіз:*\n"
            f"• Час: {datetime.now().strftime('%H:%M:%S')}\n"
            f"• Статус: 🟢 АКТИВНИЙ\n\n"
            "🎯 *Оберіть опцію аналізу:*",
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )

    async def analyze_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Повний аналіз ринку"""
        try:
            msg = await update.message.reply_text("📊 Запускаю повний аналіз ринку...")
            
            # Отримуємо реальні дані
            btc_data = await self.get_real_time_data('BTC/USDT')
            eth_data = await self.get_real_time_data('ETH/USDT')
            sol_data = await self.get_real_time_data('SOL/USDT')
            
            if not all([btc_data, eth_data, sol_data]):
                await msg.edit_text("❌ Не вдалося отримати дані для аналізу")
                return
            
            # Аналізуємо кожен актив
            btc_analysis = await self.analyze_symbol('BTC/USDT', btc_data)
            eth_analysis = await self.analyze_symbol('ETH/USDT', eth_data)
            sol_analysis = await self.analyze_symbol('SOL/USDT', sol_data)
            
            response = "📊 **РЕЗУЛЬТАТИ АНАЛІЗУ:**\n\n"
            
            for analysis in [btc_analysis, eth_analysis, sol_analysis]:
                if analysis:
                    emoji = "🟢" if analysis['trend'] == 'bullish' else "🔴"
                    response += f"{emoji} **{analysis['symbol']}** - ${analysis['price']:.2f}\n"
                    response += f"   📈 Тренд: {analysis['trend']}\n"
                    response += f"   📊 RSI: {analysis['rsi']:.1f}\n"
                    response += f"   ⚡ Моментум: {analysis['momentum']:.2f}\n"
                    response += f"   💰 Об'єм: ${analysis['volume']:,.0f}\n\n"
            
            # Додаємо загальну оцінку ринку
            market_status = await self.assess_market_status([btc_analysis, eth_analysis, sol_analysis])
            response += f"🌐 **СТАТУС РИНКУ: {market_status}**\n"
            
            await msg.edit_text(response, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Помилка аналізу: {e}")
            await update.message.reply_text("❌ Помилка аналізу ринку")

    async def get_real_time_data(self, symbol: str) -> Optional[Dict]:
        """Отримання реальних даних з біржі"""
        try:
            # Отримуємо останні дані
            ohlcv = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.exchange.fetch_ohlcv(symbol, '1h', 100)
            )
            
            if not ohlcv or len(ohlcv) < 50:
                return None
            
            # Отримуємо поточний ticker
            ticker = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.exchange.fetch_ticker(symbol)
            )
            
            return {
                'symbol': symbol,
                'ohlcv': ohlcv,
                'ticker': ticker,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Помилка отримання даних для {symbol}: {e}")
            return None

    async def analyze_symbol(self, symbol: str, market_data: Dict) -> Optional[Dict]:
        """Реальний аналіз символу"""
        try:
            ohlcv = market_data['ohlcv']
            ticker = market_data['ticker']
            
            closes = np.array([x[4] for x in ohlcv])
            highs = np.array([x[2] for x in ohlcv])
            lows = np.array([x[3] for x in ohlcv])
            volumes = np.array([x[5] for x in ohlcv])
            
            # Технічні індикатори
            rsi = talib.RSI(closes, 14)
            macd, macd_signal, _ = talib.MACD(closes)
            stoch = talib.STOCH(highs, lows, closes)
            atr = talib.ATR(highs, lows, closes, 14)
            
            # Аналіз тренду
            ema20 = talib.EMA(closes, 20)
            ema50 = talib.EMA(closes, 50)
            
            if np.isnan(ema20[-1]) or np.isnan(ema50[-1]):
                return None
            
            # Визначення тренду
            if ema20[-1] > ema50[-1] and closes[-1] > ema20[-1]:
                trend = 'bullish'
            elif ema20[-1] < ema50[-1] and closes[-1] < ema20[-1]:
                trend = 'bearish'
            else:
                trend = 'neutral'
            
            # Аналіз моментуму
            momentum = self.calculate_momentum_strength(closes, volumes)
            
            return {
                'symbol': symbol,
                'price': ticker['last'],
                'trend': trend,
                'rsi': rsi[-1] if not np.isnan(rsi[-1]) else 50,
                'momentum': momentum,
                'volume': np.mean(volumes[-5:]),
                'volatility': atr[-1] / closes[-1] * 100 if not np.isnan(atr[-1]) else 0,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Помилка аналізу {symbol}: {e}")
            return None

    def calculate_momentum_strength(self, prices: np.ndarray, volumes: np.ndarray) -> float:
        """Розрахунок сили моментуму"""
        if len(prices) < 20:
            return 0.0
        
        # Price momentum
        price_change = (prices[-1] - prices[-20]) / prices[-20] * 100
        
        # Volume momentum
        volume_change = (volumes[-1] - np.mean(volumes[-20:-10])) / np.mean(volumes[-20:-10]) * 100
        
        # Combined momentum score
        momentum_score = (price_change * 0.6 + volume_change * 0.4) / 10
        return max(min(momentum_score, 10.0), -10.0)

    async def assess_market_status(self, analyses: List[Dict]) -> str:
        """Оцінка загального стану ринку"""
        if not analyses:
            return "НЕВІДОМИЙ"
        
        bullish_count = sum(1 for a in analyses if a and a['trend'] == 'bullish')
        bearish_count = sum(1 for a in analyses if a and a['trend'] == 'bearish')
        
        if bullish_count >= 2:
            return "БИЧИЙ"
        elif bearish_count >= 2:
            return "МЕДВЕЖИЙ"
        else:
            return "НЕЙТРАЛЬНИЙ"

    async def scan_market_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Сканування ринку на можливості"""
        try:
            msg = await update.message.reply_text("🔍 Сканую ринок на торгові можливості...")
            
            symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT', 'ADA/USDT']
            opportunities = []
            
            for symbol in symbols:
                data = await self.get_real_time_data(symbol)
                if data:
                    analysis = await self.analyze_symbol(symbol, data)
                    if analysis and self.is_trading_opportunity(analysis):
                        opportunities.append(analysis)
            
            if opportunities:
                response = "🎯 **ТОРГОВІ МОЖЛИВОСТІ:**\n\n"
                
                for i, opp in enumerate(opportunities[:5], 1):
                    emoji = "🟢" if opp['trend'] == 'bullish' else "🔴"
                    response += f"{i}. {emoji} **{opp['symbol']}**\n"
                    response += f"   💰 Ціна: ${opp['price']:.2f}\n"
                    response += f"   📈 Тренд: {opp['trend']}\n"
                    response += f"   📊 RSI: {opp['rsi']:.1f}\n"
                    response += f"   ⚡ Моментум: {opp['momentum']:.2f}\n\n"
                
                await msg.edit_text(response, parse_mode='Markdown')
            else:
                await msg.edit_text("📉 Наразі якісних торгових можливостей не знайдено")
                
        except Exception as e:
            logger.error(f"Помилка сканування: {e}")
            await update.message.reply_text("❌ Помилка сканування ринку")

    def is_trading_opportunity(self, analysis: Dict) -> bool:
        """Перевірка чи є аналіз торговою можливістю"""
        if not analysis:
            return False
        
        # Критерії якісної можливості
        rsi_ok = (analysis['rsi'] < 35 and analysis['trend'] == 'bullish') or \
                 (analysis['rsi'] > 65 and analysis['trend'] == 'bearish')
        
        momentum_ok = abs(analysis['momentum']) > 2.0
        volume_ok = analysis['volume'] > 1000000  # Мінімальний об'єм
        
        return rsi_ok and momentum_ok and volume_ok

    async def key_levels_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Ключові рівні підтримки/опору"""
        try:
            msg = await update.message.reply_text("🎯 Визначаю ключові рівні...")
            
            symbol = 'BTC/USDT'
            data = await self.get_real_time_data(symbol)
            
            if not data:
                await msg.edit_text("❌ Не вдалося отримати дані")
                return
            
            levels = self.find_key_levels(data['ohlcv'])
            
            response = f"🎯 **КЛЮЧОВІ РІВНІ {symbol}:**\n\n"
            response += f"📊 Поточна ціна: ${data['ticker']['last']:.2f}\n\n"
            
            response += "🛡️ **ПІДТРИМКА:**\n"
            for level in levels['support'][:3]:
                response += f"• ${level:.2f}\n"
            
            response += "\n📈 **ОПІР:**\n"
            for level in levels['resistance'][:3]:
                response += f"• ${level:.2f}\n"
            
            response += f"\n📏 Відстань до найближчого рівня: {levels['distance_to_nearest']:.2f}%"
            
            await msg.edit_text(response, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Помилка пошуку рівнів: {e}")
            await update.message.reply_text("❌ Помилка аналізу рівнів")

    def find_key_levels(self, ohlcv: List) -> Dict:
        """Знаходження ключових рівнів підтримки/опору"""
        highs = np.array([x[2] for x in ohlcv])
        lows = np.array([x[3] for x in ohlcv])
        closes = np.array([x[4] for x in ohlcv])
        
        # Знаходимо локальні екстремуми
        support_levels = []
        resistance_levels = []
        
        # Простий алгоритм пошуку рівнів
        for i in range(2, len(ohlcv) - 2):
            # Перевірка для підтримки
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                support_levels.append(lows[i])
            
            # Перевірка для опору
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                resistance_levels.append(highs[i])
        
        # Унікальні рівні
        support_levels = sorted(set(support_levels))
        resistance_levels = sorted(set(resistance_levels))
        
        # Відстань до найближчого рівня
        current_price = closes[-1]
        nearest_level = min(
            [abs(price - current_price) for price in support_levels + resistance_levels],
            default=0
        )
        distance_pct = nearest_level / current_price * 100
        
        return {
            'support': support_levels,
            'resistance': resistance_levels,
            'distance_to_nearest': distance_pct
        }

    async def volume_analysis_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Аналіз об'ємів"""
        try:
            msg = await update.message.reply_text("📊 Аналізую об'ємні паттерни...")
            
            symbol = 'BTC/USDT'
            data = await self.get_real_time_data(symbol)
            
            if not data:
                await msg.edit_text("❌ Не вдалося отримати дані")
                return
            
            volume_analysis = self.analyze_volume_patterns(data['ohlcv'])
            
            response = f"📊 **АНАЛІЗ ОБ'ЄМІВ {symbol}:**\n\n"
            response += f"📈 Поточний об'єм: ${volume_analysis['current_volume']:,.0f}\n"
            response += f"📊 Середній об'єм: ${volume_analysis['avg_volume']:,.0f}\n"
            response += f"⚡ Об'ємний тиск: {volume_analysis['volume_pressure']:.2f}\n"
            response += f"🎯 Тренд об'ємів: {volume_analysis['volume_trend']}\n\n"
            
            response += "💡 **ІНТЕРПРЕТАЦІЯ:**\n"
            if volume_analysis['volume_pressure'] > 1.5:
                response += "• Сильний об'ємний тиск\n• Можливість пробою\n"
            elif volume_analysis['volume_pressure'] < 0.7:
                response += "• Низька об'ємна активність\n• Консолідація\n"
            else:
                response += "• Нормальна об'ємна активність\n"
            
            await msg.edit_text(response, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Помилка аналізу об'ємів: {e}")
            await update.message.reply_text("❌ Помилка аналізу об'ємів")

    def analyze_volume_patterns(self, ohlcv: List) -> Dict:
        """Аналіз об'ємних паттернів"""
        volumes = np.array([x[5] for x in ohlcv])
        
        current_volume = volumes[-1]
        avg_volume = np.mean(volumes[-20:])
        volume_pressure = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Визначення тренду об'ємів
        if current_volume > np.mean(volumes[-5:]):
            volume_trend = "ЗРОСТАННЯ"
        elif current_volume < np.mean(volumes[-5:]):
            volume_trend = "СПАДАННЯ"
        else:
            volume_trend = "СТАБІЛЬНІСТЬ"
        
        return {
            'current_volume': current_volume,
            'avg_volume': avg_volume,
            'volume_pressure': volume_pressure,
            'volume_trend': volume_trend
        }

    async def momentum_analysis_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Аналіз моментуму"""
        try:
            msg = await update.message.reply_text("⚡ Аналізую моментум...")
            
            symbol = 'BTC/USDT'
            data = await self.get_real_time_data(symbol)
            
            if not data:
                await msg.edit_text("❌ Не вдалося отримати дані")
                return
            
            momentum_analysis = self.analyze_momentum(data['ohlcv'])
            
            response = f"⚡ **АНАЛІЗ МОМЕНТУМУ {symbol}:**\n\n"
            response += f"📈 Сила моментуму: {momentum_analysis['momentum_strength']:.2f}\n"
            response += f"📊 Напрямок: {momentum_analysis['momentum_direction']}\n"
            response += f"🎯 RSI: {momentum_analysis['rsi']:.1f}\n"
            response += f"📉 Stochastic: {momentum_analysis['stoch']:.1f}\n\n"
            
            response += "🔍 **СИГНАЛИ:**\n"
            if momentum_analysis['rsi'] < 30:
                response += "• Перепроданість (RSI < 30)\n"
            elif momentum_analysis['rsi'] > 70:
                response += "• Перекупленість (RSI > 70)\n"
            
            if momentum_analysis['stoch'] < 20:
                response += "• Перепроданість (Stoch < 20)\n"
            elif momentum_analysis['stoch'] > 80:
                response += "• Перекупленість (Stoch > 80)\n"
            
            await msg.edit_text(response, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Помилка аналізу моментуму: {e}")
            await update.message.reply_text("❌ Помилка аналізу моментуму")

    def analyze_momentum(self, ohlcv: List) -> Dict:
        """Аналіз моментуму"""
        closes = np.array([x[4] for x in ohlcv])
        highs = np.array([x[2] for x in ohlcv])
        lows = np.array([x[3] for x in ohlcv])
        
        # RSI
        rsi = talib.RSI(closes, 14)
        
        # Stochastic
        stoch_k, stoch_d = talib.STOCH(highs, lows, closes)
        
        # MACD
        macd, macd_signal, _ = talib.MACD(closes)
        
        # Визначення напрямку моментуму
        if len(macd) > 1 and macd[-1] > macd_signal[-1]:
            momentum_direction = "БИЧИЙ"
        elif len(macd) > 1 and macd[-1] < macd_signal[-1]:
            momentum_direction = "МЕДВЕЖИЙ"
        else:
            momentum_direction = "НЕЙТРАЛЬНИЙ"
        
        # Сила моментуму
        momentum_strength = abs(macd[-1] - macd_signal[-1]) / np.std(closes) * 100 if len(macd) > 1 else 0
        
        return {
            'rsi': rsi[-1] if not np.isnan(rsi[-1]) else 50,
            'stoch': stoch_k[-1] if not np.isnan(stoch_k[-1]) else 50,
            'momentum_direction': momentum_direction,
            'momentum_strength': momentum_strength
        }

    async def handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обробник callback"""
        query = update.callback_query
        await query.answer()
        
        try:
            if query.data == "analyze":
                await self.analyze_command(query, context)
            elif query.data == "scan":
                await self.scan_market_command(query, context)
            elif query.data == "levels":
                await self.key_levels_command(query, context)
            elif query.data == "volume":
                await self.volume_analysis_command(query, context)
            elif query.data == "momentum":
                await self.momentum_analysis_command(query, context)
            elif query.data == "correlation":
                await self.correlation_analysis_command(query, context)
            elif query.data == "opportunities":
                await self.opportunities_command(query, context)
                
        except Exception as e:
            await query.edit_message_text("❌ Помилка обробки запиту")

    async def correlation_analysis_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Аналіз кореляцій"""
        try:
            msg = await update.message.reply_text("🔗 Аналізую кореляції...")
            
            correlations = await self.calculate_correlations()
            
            response = "🔗 **КОРЕЛЯЦІЙНИЙ АНАЛІЗ:**\n\n"
            
            for pair, corr in correlations.items():
                correlation_str = f"{corr:.2f}"
                if corr > 0.7:
                    emoji = "🔴"
                elif corr < -0.7:
                    emoji = "🟢"
                else:
                    emoji = "⚪"
                
                response += f"{emoji} {pair}: {correlation_str}\n"
            
            response += "\n💡 **ІНТЕРПРЕТАЦІЯ:**\n"
            response += "• > 0.7: Сильна позитивна кореляція\n"
            response += "• < -0.7: Сильна негативна кореляція\n"
            response += "• -0.3 до 0.3: Слабка кореляція\n"
            
            await msg.edit_text(response, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Помилка аналізу кореляцій: {e}")
            await update.message.reply_text("❌ Помилка аналізу кореляцій")

    async def calculate_correlations(self) -> Dict:
        """Розрахунок кореляцій"""
        symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
        correlations = {}
        
        # Отримуємо дані для всіх символів
        data = {}
        for symbol in symbols:
            symbol_data = await self.get_real_time_data(symbol)
            if symbol_data:
                data[symbol] = np.array([x[4] for x in symbol_data['ohlcv']])
        
        # Розраховуємо кореляції
        for i, sym1 in enumerate(symbols):
            for sym2 in symbols[i+1:]:
                if sym1 in data and sym2 in data and len(data[sym1]) == len(data[sym2]):
                    corr = np.corrcoef(data[sym1][-30:], data[sym2][-30:])[0, 1]
                    if not np.isnan(corr):
                        pair_name = f"{sym1.split('/')[0]}-{sym2.split('/')[0]}"
                        correlations[pair_name] = corr
        
        return correlations

    async def opportunities_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Пошук найкращих можливостей"""
        try:
            msg = await update.message.reply_text("💰 Шукаю найкращі можливості...")
            
            symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT']
            best_opportunities = []
            
            for symbol in symbols:
                data = await self.get_real_time_data(symbol)
                if data:
                    analysis = await self.analyze_symbol(symbol, data)
                    if analysis:
                        score = self.calculate_opportunity_score(analysis)
                        if score > 7.0:  # Мінімальний поріг
                            best_opportunities.append((analysis, score))
            
            if best_opportunities:
                best_opportunities.sort(key=lambda x: x[1], reverse=True)
                
                response = "💰 **НАЙКРАЩІ МОЖЛИВОСТІ:**\n\n"
                
                for i, (analysis, score) in enumerate(best_opportunities[:3], 1):
                    emoji = "🟢" if analysis['trend'] == 'bullish' else "🔴"
                    response += f"{i}. {emoji} **{analysis['symbol']}** - Оцінка: {score:.1f}/10\n"
                    response += f"   💰 Ціна: ${analysis['price']:.2f}\n"
                    response += f"   📈 Тренд: {analysis['trend']}\n"
                    response += f"   📊 RSI: {analysis['rsi']:.1f}\n"
                    response += f"   ⚡ Моментум: {analysis['momentum']:.2f}\n\n"
                
                await msg.edit_text(response, parse_mode='Markdown')
            else:
                await msg.edit_text("📉 Наразі найкращих можливостей не знайдено")
                
        except Exception as e:
            logger.error(f"Помилка пошуку можливостей: {e}")
            await update.message.reply_text("❌ Помилка пошуку можливостей")

    def calculate_opportunity_score(self, analysis: Dict) -> float:
        """Розрахунок оцінки можливості"""
        if not analysis:
            return 0.0
        
        # Ваги для різних факторів
        weights = {
            'rsi': 0.3,
            'momentum': 0.3,
            'volume': 0.2,
            'volatility': 0.2
        }
        
        # Нормалізація факторів
        rsi_score = 0
        if analysis['trend'] == 'bullish' and analysis['rsi'] < 35:
            rsi_score = (35 - analysis['rsi']) / 35 * 10
        elif analysis['trend'] == 'bearish' and analysis['rsi'] > 65:
            rsi_score = (analysis['rsi'] - 65) / 35 * 10
        
        momentum_score = min(abs(analysis['momentum']) * 2, 10)
        volume_score = min(analysis['volume'] / 5000000 * 10, 10)  # Нормалізація об'єму
        volatility_score = min(analysis['volatility'] * 100, 10)  # Волатильність у %
        
        # Загальна оцінка
        total_score = (
            rsi_score * weights['rsi'] +
            momentum_score * weights['momentum'] +
            volume_score * weights['volume'] +
            volatility_score * weights['volatility']
        )
        
        return min(total_score, 10.0)

    async def run(self):
        """Запуск бота"""
        try:
            logger.info("🚀 Запускаю Real Analysis Bot...")
            await self.app.initialize()
            await self.app.start()
            await self.app.updater.start_polling()
            
            logger.info("✅ Бот успішно запущено!")
            
            while True:
                await asyncio.sleep(3600)
                
        except Exception as e:
            logger.error(f"❌ Помилка запуску: {e}")
            raise

async def main():
    """Головна функція"""
    try:
        BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
        if not BOT_TOKEN:
            logger.error("Встановіть TELEGRAM_BOT_TOKEN")
            return
        
        bot = RealAnalysisBot(BOT_TOKEN)
        await bot.run()
        
    except KeyboardInterrupt:
        logger.info("Зупинка бота...")
    except Exception as e:
        logger.error(f"Критична помилка: {e}")

if __name__ == '__main__':
    import os
    asyncio.run(main())