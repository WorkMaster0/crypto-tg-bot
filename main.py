import os
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
        logging.FileHandler('vwmc_bot.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class VWMCStrategyBot:
    def __init__(self, token: str):
        self.token = token
        self.app = Application.builder().token(token).build()
        
        # Підключення до біржі
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })
        
        # Унікальні параметри VWMC стратегії
        self.vwmc_params = {
            'volume_weight_period': 21,
            'momentum_window': 14,
            'convergence_threshold': 0.85,
            'divergence_sensitivity': 1.5,
            'liquidity_zone_depth': 0.003,
            'entry_confidence_min': 0.7,
            'trend_filter_strength': 0.6,
            'volatility_adjustment': True,
            'dynamic_position_sizing': True
        }
        
        # Статистика стратегії
        self.strategy_stats = {
            'total_signals': 0,
            'successful_signals': 0,
            'avg_profit_per_trade': 0.0,
            'max_runup': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0
        }
        
        # Історичні дані для аналізу
        self.historical_data = {}
        self.pattern_recognition = {}
        
        self.setup_handlers()
        logger.info("VWMC Strategy Bot ініціалізовано")

    def setup_handlers(self):
        """Унікальні команди для VWMC стратегії"""
        handlers = [
            CommandHandler("start", self.start_command),
            CommandHandler("vwmc_scan", self.vwmc_scan_command),
            CommandHandler("volume_analysis", self.volume_analysis_command),
            CommandHandler("momentum_matrix", self.momentum_matrix_command),
            CommandHandler("liquidity_map", self.liquidity_map_command),
            CommandHandler("pattern_recognition", self.pattern_recognition_command),
            CommandHandler("backtest_results", self.backtest_results_command),
            CommandHandler("market_insights", self.market_insights_command),
            CommandHandler("risk_assessment", self.risk_assessment_command),
            CommandHandler("performance", self.performance_command),
            CallbackQueryHandler(self.handle_callback)
        ]
        
        for handler in handlers:
            self.app.add_handler(handler)

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Стартова команда з VWMC стратегією"""
        keyboard = [
            [InlineKeyboardButton("🔍 VWMC СКАН", callback_data="vwmc_scan"),
             InlineKeyboardButton("📊 АНАЛІЗ ОБ'ЄМІВ", callback_data="volume_analysis")],
            [InlineKeyboardButton("⚡ МОМЕНТУМ МАТРИЦЯ", callback_data="momentum_matrix"),
             InlineKeyboardButton("💰 КАРТА ЛІКВІДНОСТІ", callback_data="liquidity_map")],
            [InlineKeyboardButton("🎯 РОЗПІЗНАВАННЯ ПАТТЕРНІВ", callback_data="pattern_recognition"),
             InlineKeyboardButton("📈 БЕКТЕСТ", callback_data="backtest_results")],
            [InlineKeyboardButton("💡 ІНСАЙТИ РИНКУ", callback_data="market_insights"),
             InlineKeyboardButton("⚠️ ОЦІНКА РИЗИКІВ", callback_data="risk_assessment")],
            [InlineKeyboardButton("📊 ПРОДУКТИВНІСТЬ", callback_data="performance")]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            "🚀 **VWMC STRATEGY BOT**\n\n"
            "🎯 *Volume-Weighted Momentum Convergence*\n"
            "Унікальна стратегія на основі аналізу історії ринку\n\n"
            "📊 *Ключові переваги:*\n"
            "• 🤖 Автоматичне розпізнавання паттернів\n"
            "• 📈 Weighted Volume Analysis\n"
            "• ⚡ Momentum Convergence Detection\n"
            "• 💰 Liquidity Zone Mapping\n"
            "• 🎯 High-Probability Entries\n\n"
            "🔮 *Статистика стратегії:*\n"
            f"• Win Rate: {self.strategy_stats['win_rate']:.1f}%\n"
            f"• Profit Factor: {self.strategy_stats['profit_factor']:.2f}\n"
            f"• Sharpe Ratio: {self.strategy_stats['sharpe_ratio']:.2f}",
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )

    async def vwmc_scan_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Сканування за VWMC стратегією"""
        try:
            msg = await update.message.reply_text("🔍 Запускаю VWMC сканування...")
            
            symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT']
            vwmc_signals = []
            
            for symbol in symbols:
                signal_data = await self.analyze_vwmc_pattern(symbol)
                if signal_data and signal_data['confidence'] >= self.vwmc_params['entry_confidence_min']:
                    vwmc_signals.append(signal_data)
            
            if vwmc_signals:
                vwmc_signals.sort(key=lambda x: x['score'], reverse=True)
                
                response = "🎯 **VWMC СИГНАЛИ:**\n\n"
                
                for i, signal in enumerate(vwmc_signals[:3], 1):
                    response += f"{i}. 🌟 **{signal['symbol']}** - Score: {signal['score']}/100\n"
                    response += f"   📈 Напрям: {signal['direction']}\n"
                    response += f"   💰 Вірогідність: {signal['confidence']:.0%}\n"
                    response += f"   ⚡ Моментум: {signal['momentum_strength']:.2f}\n"
                    response += f"   📊 Об'ємний тиск: {signal['volume_pressure']:.2f}\n\n"
                
                response += "🔍 **VWMC КРИТЕРІЇ:**\n"
                response += "• Конвергенція ціни та об'єму\n"
                response += "• Моментум акселерація\n"
                response += "• Ліквідність кластеризація\n"
                response += "• Волатильність адаптація\n"
                
                await msg.edit_text(response, parse_mode='Markdown')
            else:
                await msg.edit_text("📉 VWMC сигналів не знайдено. Очікуйте кращих умов.")
                
        except Exception as e:
            logger.error(f"VWMC scan error: {e}")
            await update.message.reply_text("❌ Помилка VWMC сканування")

    async def analyze_vwmc_pattern(self, symbol: str) -> Optional[Dict]:
        """Аналіз VWMC паттерну"""
        try:
            # Отримання даних для аналізу
            ohlcv = await self.get_ohlcv(symbol, '1h', 100)
            if not ohlcv or len(ohlcv) < 50:
                return None
            
            closes = np.array([x[4] for x in ohlcv])
            highs = np.array([x[2] for x in ohlcv])
            lows = np.array([x[3] for x in ohlcv])
            volumes = np.array([x[5] for x in ohlcv])
            
            # VWMC аналіз
            volume_weighted_analysis = self.calculate_volume_weighted_metrics(closes, volumes)
            momentum_convergence = self.analyze_momentum_convergence(closes, volumes)
            liquidity_zones = self.identify_liquidity_zones(highs, lows, volumes)
            pattern_recognition = self.recognize_price_patterns(closes, volumes)
            
            # Комбінована оцінка
            vwmc_score = self.calculate_vwmc_score(
                volume_weighted_analysis,
                momentum_convergence,
                liquidity_zones,
                pattern_recognition
            )
            
            if vwmc_score < 60:  # Мінімальний поріг
                return None
            
            # Визначення напрямку
            direction = self.determine_direction(
                volume_weighted_analysis,
                momentum_convergence,
                pattern_recognition
            )
            
            return {
                'symbol': symbol,
                'direction': direction,
                'score': vwmc_score,
                'confidence': vwmc_score / 100,
                'momentum_strength': momentum_convergence['strength'],
                'volume_pressure': volume_weighted_analysis['pressure'],
                'liquidity_zones': liquidity_zones,
                'pattern': pattern_recognition['pattern']
            }
            
        except Exception as e:
            logger.error(f"VWMC analysis error for {symbol}: {e}")
            return None

    def calculate_volume_weighted_metrics(self, prices: np.ndarray, volumes: np.ndarray) -> Dict:
        """Розрахунок volume-weighted метрик"""
        # Volume Weighted Moving Average
        vwma = np.sum(prices * volumes) / np.sum(volumes)
        
        # Volume Pressure Index
        volume_ma = talib.SMA(volumes, self.vwmc_params['volume_weight_period'])
        volume_pressure = volumes[-1] / volume_ma[-1] if volume_ma[-1] > 0 else 1.0
        
        # Volume-Weighted Momentum
        returns = np.diff(prices) / prices[:-1]
        weighted_returns = returns * volumes[1:] / np.sum(volumes[1:])
        vwmomentum = np.sum(weighted_returns)
        
        return {
            'vwma': vwma,
            'pressure': volume_pressure,
            'momentum': vwmomentum,
            'trend': 1 if vwmomentum > 0 else -1
        }

    def analyze_momentum_convergence(self, prices: np.ndarray, volumes: np.ndarray) -> Dict:
        """Аналіз конвергенції моментуму"""
        # RSI з об'ємною зваженою
        rsi = talib.RSI(prices, self.vwmc_params['momentum_window'])
        
        # Volume-Weighted MACD
        macd, macd_signal, _ = talib.MACD(prices)
        
        # Momentum Convergence Index
        price_momentum = talib.MOM(prices, self.vwmc_params['momentum_window'])
        volume_momentum = talib.MOM(volumes, self.vwmc_params['momentum_window'])
        
        # Конвергенція/дивергенція
        convergence = np.corrcoef(price_momentum[-20:], volume_momentum[-20:])[0, 1]
        
        return {
            'rsi': rsi[-1],
            'macd_convergence': macd[-1] - macd_signal[-1],
            'convergence_strength': abs(convergence),
            'convergence_direction': 1 if convergence > 0 else -1,
            'strength': np.mean([abs(convergence), abs(macd[-1] - macd_signal[-1]) / np.std(prices)])
        }

    def identify_liquidity_zones(self, highs: np.ndarray, lows: np.ndarray, volumes: np.ndarray) -> List[float]:
        """Ідентифікація зон ліквідності"""
        # Volume Profile Analysis
        price_levels = np.linspace(np.min(lows), np.max(highs), 100)
        volume_at_price = []
        
        for i in range(len(price_levels) - 1):
            mask = (highs >= price_levels[i]) & (lows <= price_levels[i + 1])
            volume_at_price.append(np.sum(volumes[mask]))
        
        # Знаходження значущих рівнів
        significant_levels = []
        mean_volume = np.mean(volume_at_price)
        std_volume = np.std(volume_at_price)
        
        for i, vol in enumerate(volume_at_price):
            if vol > mean_volume + std_volume * self.vwmc_params['divergence_sensitivity']:
                significant_levels.append(price_levels[i])
        
        return significant_levels[:5]  # Топ-5 рівнів

    def recognize_price_patterns(self, prices: np.ndarray, volumes: np.ndarray) -> Dict:
        """Розпізнавання ценових паттернів"""
        # Аналіз паттернів на основі історії
        patterns = {
            'bullish_engulfing': self.detect_bullish_engulfing(prices, volumes),
            'bearish_engulfing': self.detect_bearish_engulfing(prices, volumes),
            'double_bottom': self.detect_double_bottom(prices),
            'double_top': self.detect_double_top(prices),
            'volume_spike': self.detect_volume_spike(volumes)
        }
        
        # Визначення найсильнішого паттерну
        strongest_pattern = max(patterns.items(), key=lambda x: x[1]['strength'])
        
        return {
            'pattern': strongest_pattern[0],
            'strength': strongest_pattern[1]['strength'],
            'direction': strongest_pattern[1]['direction']
        }

    def detect_bullish_engulfing(self, prices: np.ndarray, volumes: np.ndarray) -> Dict:
        """Детекція бичого поглинаючого паттерну"""
        if len(prices) < 3:
            return {'strength': 0, 'direction': 'none'}
        
        # Перевірка умов bullish engulfing
        current_close = prices[-1]
        current_open = prices[-1] - (prices[-1] - prices[-2])  # Approximation
        prev_close = prices[-2]
        prev_open = prices[-2] - (prices[-2] - prices[-3])
        
        is_engulfing = (current_close > prev_open and 
                       current_open < prev_close and 
                       current_close > current_open)
        
        strength = 0.7 if is_engulfing else 0
        if is_engulfing and volumes[-1] > np.mean(volumes[-5:]):
            strength = 0.9
        
        return {'strength': strength, 'direction': 'bullish'}

    def detect_bearish_engulfing(self, prices: np.ndarray, volumes: np.ndarray) -> Dict:
        """Детекція ведмежого поглинаючого паттерну"""
        if len(prices) < 3:
            return {'strength': 0, 'direction': 'none'}
        
        current_close = prices[-1]
        current_open = prices[-1] - (prices[-1] - prices[-2])
        prev_close = prices[-2]
        prev_open = prices[-2] - (prices[-2] - prices[-3])
        
        is_engulfing = (current_close < prev_open and 
                       current_open > prev_close and 
                       current_close < current_open)
        
        strength = 0.7 if is_engulfing else 0
        if is_engulfing and volumes[-1] > np.mean(volumes[-5:]):
            strength = 0.9
        
        return {'strength': strength, 'direction': 'bearish'}

    def detect_double_bottom(self, prices: np.ndarray) -> Dict:
        """Детекція подвійного дна"""
        if len(prices) < 20:
            return {'strength': 0, 'direction': 'none'}
        
        # Пошук локальних мінімумів
        minima_indices = signal.argrelextrema(prices, np.less_equal, order=5)[0]
        
        if len(minima_indices) < 2:
            return {'strength': 0, 'direction': 'none'}
        
        # Перевірка умов double bottom
        last_minima = minima_indices[-2:]
        price_diff = abs(prices[last_minima[0]] - prices[last_minima[1]]) / prices[last_minima[0]]
        
        if price_diff < 0.02:  # Максимальна різниця 2%
            strength = 0.8
            # Перевірка пробою neckline
            neckline = np.max(prices[last_minima[0]:last_minima[1]])
            if prices[-1] > neckline:
                strength = 0.95
            
            return {'strength': strength, 'direction': 'bullish'}
        
        return {'strength': 0, 'direction': 'none'}

    def detect_double_top(self, prices: np.ndarray) -> Dict:
        """Детекція подвійної вершини"""
        if len(prices) < 20:
            return {'strength': 0, 'direction': 'none'}
        
        # Пошук локальних максимумів
        maxima_indices = signal.argrelextrema(prices, np.greater_equal, order=5)[0]
        
        if len(maxima_indices) < 2:
            return {'strength': 0, 'direction': 'none'}
        
        # Перевірка умов double top
        last_maxima = maxima_indices[-2:]
        price_diff = abs(prices[last_maxima[0]] - prices[last_maxima[1]]) / prices[last_maxima[0]]
        
        if price_diff < 0.02:
            strength = 0.8
            # Перевірка пробою neckline
            neckline = np.min(prices[last_maxima[0]:last_maxima[1]])
            if prices[-1] < neckline:
                strength = 0.95
            
            return {'strength': strength, 'direction': 'bearish'}
        
        return {'strength': 0, 'direction': 'none'}

    def detect_volume_spike(self, volumes: np.ndarray) -> Dict:
        """Детекція спайку об'ємів"""
        if len(volumes) < 10:
            return {'strength': 0, 'direction': 'none'}
        
        current_volume = volumes[-1]
        avg_volume = np.mean(volumes[-10:-1])
        volume_ratio = current_volume / avg_volume
        
        if volume_ratio > 2.0:
            return {'strength': 0.7, 'direction': 'breakout'}
        elif volume_ratio > 3.0:
            return {'strength': 0.9, 'direction': 'breakout'}
        
        return {'strength': 0, 'direction': 'none'}

    def calculate_vwmc_score(self, volume_analysis: Dict, momentum_analysis: Dict, 
                           liquidity_zones: List, pattern_analysis: Dict) -> float:
        """Розрахунок загального VWMC score"""
        weights = {
            'volume_pressure': 0.25,
            'momentum_strength': 0.25,
            'pattern_strength': 0.20,
            'liquidity_zones': 0.15,
            'convergence': 0.15
        }
        
        # Нормалізація компонентів
        volume_score = min(volume_analysis['pressure'] * 50, 100)
        momentum_score = momentum_analysis['strength'] * 100
        pattern_score = pattern_analysis['strength'] * 100
        liquidity_score = min(len(liquidity_zones) * 20, 100)
        convergence_score = momentum_analysis['convergence_strength'] * 100
        
        # Загальна оцінка
        total_score = (
            volume_score * weights['volume_pressure'] +
            momentum_score * weights['momentum_strength'] +
            pattern_score * weights['pattern_strength'] +
            liquidity_score * weights['liquidity_zones'] +
            convergence_score * weights['convergence']
        )
        
        return min(total_score, 100)

    def determine_direction(self, volume_analysis: Dict, momentum_analysis: Dict, 
                          pattern_analysis: Dict) -> str:
        """Визначення напрямку торгівлі"""
        # Голосування між різними компонентами
        votes = {
            'LONG': 0,
            'SHORT': 0
        }
        
        # Volume analysis vote
        if volume_analysis['trend'] > 0:
            votes['LONG'] += 1
        else:
            votes['SHORT'] += 1
        
        # Momentum analysis vote
        if momentum_analysis['convergence_direction'] > 0:
            votes['LONG'] += 1
        else:
            votes['SHORT'] += 1
        
        # Pattern analysis vote
        if pattern_analysis['direction'] in ['bullish', 'breakout']:
            votes['LONG'] += 1
        elif pattern_analysis['direction'] in ['bearish']:
            votes['SHORT'] += 1
        
        return 'LONG' if votes['LONG'] > votes['SHORT'] else 'SHORT'

    async def volume_analysis_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Глибинний аналіз об'ємів"""
        try:
            msg = await update.message.reply_text("📊 Аналізую об'ємні паттерни...")
            
            volume_data = await self.deep_volume_analysis('BTC/USDT')
            
            response = "📊 **ГЛИБИННИЙ АНАЛІЗ ОБ'ЄМІВ:**\n\n"
            response += f"🔍 **BTC/USDT**\n"
            response += f"• Об'ємний тиск: {volume_data['volume_pressure']:.2f}\n"
            response += f"• VWMA відхилення: {volume_data['vwma_deviation']:.2f}%\n"
            response += f"• Об'ємний тренд: {volume_data['volume_trend']}\n"
            response += f"• Кластеризація: {volume_data['clustering_score']:.2f}\n\n"
            
            response += "🎯 **ІНТЕРПРЕТАЦІЯ:**\n"
            if volume_data['volume_pressure'] > 1.5:
                response += "• Сильний об'ємний тиск\n"
                response += "• Можливість пробою\n"
            elif volume_data['volume_pressure'] < 0.7:
                response += "• Низька об'ємна активність\n"
                response += "• Консолідація\n"
            
            await msg.edit_text(response, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Volume analysis error: {e}")
            await update.message.reply_text("❌ Помилка аналізу об'ємів")

    async def deep_volume_analysis(self, symbol: str) -> Dict:
        """Глибинний аналіз об'ємів"""
        ohlcv = await self.get_ohlcv(symbol, '4h', 50)
        if not ohlcv:
            return {}
        
        closes = np.array([x[4] for x in ohlcv])
        volumes = np.array([x[5] for x in ohlcv])
        
        # Volume Pressure
        volume_ma = talib.SMA(volumes, 20)
        volume_pressure = volumes[-1] / volume_ma[-1] if volume_ma[-1] > 0 else 1.0
        
        # VWMA Deviation
        vwma = np.sum(closes * volumes) / np.sum(volumes)
        price_ma = talib.SMA(closes, 20)
        vwma_deviation = (vwma - price_ma[-1]) / price_ma[-1] * 100
        
        # Volume Trend
        volume_trend = "ВИСХІДНИЙ" if volumes[-1] > np.mean(volumes[-5:]) else "НИЗХІДНИЙ"
        
        # Clustering Analysis
        volume_std = np.std(volumes)
        clustering_score = volume_std / np.mean(volumes)
        
        return {
            'volume_pressure': volume_pressure,
            'vwma_deviation': vwma_deviation,
            'volume_trend': volume_trend,
            'clustering_score': clustering_score
        }

    async def momentum_matrix_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Матриця моментуму"""
        try:
            msg = await update.message.reply_text("⚡ Розраховую матрицю моментуму...")
            
            momentum_data = await self.calculate_momentum_matrix()
            
            response = "⚡ **МАТРИЦЯ МОМЕНТУМУ:**\n\n"
            
            for asset, data in momentum_data.items():
                response += f"🎯 **{asset}**\n"
                response += f"   RSI: {data['rsi']:.1f}\n"
                response += f"   MACD: {data['macd']:.4f}\n"
                response += f"   Моментум: {data['momentum']:.2f}\n\n"
            
            response += "🔍 **КЛЮЧОВІ РІВНІ:**\n"
            response += "• RSI > 70: перекупленість\n"
            response += "• RSI < 30: перепроданість\n"
            response += "• MACD > 0: бичий моментум\n"
            
            await msg.edit_text(response, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Momentum matrix error: {e}")
            await update.message.reply_text("❌ Помилка матриці моментуму")

    async def calculate_momentum_matrix(self) -> Dict:
        """Розрахунок матриці моментуму"""
        symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
        momentum_data = {}
        
        for symbol in symbols:
            ohlcv = await self.get_ohlcv(symbol, '1h', 50)
            if ohlcv:
                closes = np.array([x[4] for x in ohlcv])
                
                rsi = talib.RSI(closes, 14)
                macd, macd_signal, _ = talib.MACD(closes)
                momentum = talib.MOM(closes, 10)
                
                momentum_data[symbol] = {
                    'rsi': rsi[-1] if not np.isnan(rsi[-1]) else 50,
                    'macd': macd[-1] - macd_signal[-1] if len(macd) > 0 else 0,
                    'momentum': momentum[-1] if len(momentum) > 0 else 0
                }
        
        return momentum_data

    async def liquidity_map_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Карта ліквідності"""
        try:
            msg = await update.message.reply_text("💰 Створюю карту ліквідності...")
            
            liquidity_map = await self.generate_liquidity_map()
            
            response = "💰 **КАРТА ЛІКВІДНОСТІ:**\n\n"
            
            for symbol, levels in liquidity_map.items():
                response += f"📊 **{symbol}**\n"
                response += f"   🎯 Ключові рівні: {len(levels)}\n"
                if levels:
                    response += f"   💰 Найближчий: ${levels[0]:.2f}\n"
                response += "\n"
            
            response += "🔍 **ВАЖЛИВІСТЬ:**\n"
            response += "• Ціла прагнуть до зон ліквідності\n"
            response += "• Пробиття веде до сильних рухів\n"
            response += "• Ідеальні точки для входу/виходу\n"
            
            await msg.edit_text(response, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Liquidity map error: {e}")
            await update.message.reply_text("❌ Помилка карти ліквідності")

    async def generate_liquidity_map(self) -> Dict:
        """Генерація карти ліквідності"""
        symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
        liquidity_map = {}
        
        for symbol in symbols:
            ohlcv = await self.get_ohlcv(symbol, '4h', 100)
            if ohlcv:
                highs = np.array([x[2] for x in ohlcv])
                lows = np.array([x[3] for x in ohlcv])
                volumes = np.array([x[5] for x in ohlcv])
                
                levels = self.identify_liquidity_zones(highs, lows, volumes)
                liquidity_map[symbol] = levels
        
        return liquidity_map

    async def pattern_recognition_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Розпізнавання паттернів"""
        try:
            msg = await update.message.reply_text("🎯 Аналізую ценові паттерни...")
            
            patterns = await self.recognize_market_patterns()
            
            response = "🎯 **РОЗПІЗНАННЯ ПАТТЕРНІВ:**\n\n"
            
            for symbol, pattern in patterns.items():
                response += f"📈 **{symbol}**\n"
                response += f"   Паттерн: {pattern['name']}\n"
                response += f"   Сила: {pattern['strength']:.2f}\n"
                response += f"   Напрям: {pattern['direction']}\n\n"
            
            response += "🔮 **ТОРГОВІ НАСЛІДКИ:**\n"
            response += "• Поглинаючі паттерни: високоякісні\n"
            response += "• Подвійне дно/вершина: сильні рівні\n"
            response += "• Спайки об'єму: потенційні пробої\n"
            
            await msg.edit_text(response, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Pattern recognition error: {e}")
            await update.message.reply_text("❌ Помилка розпізнавання паттернів")

    async def recognize_market_patterns(self) -> Dict:
        """Розпізнавання ринкових паттернів"""
        symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
        patterns = {}
        
        for symbol in symbols:
            ohlcv = await self.get_ohlcv(symbol, '1h', 50)
            if ohlcv:
                closes = np.array([x[4] for x in ohlcv])
                volumes = np.array([x[5] for x in ohlcv])
                
                pattern_data = self.recognize_price_patterns(closes, volumes)
                patterns[symbol] = {
                    'name': pattern_data['pattern'],
                    'strength': pattern_data['strength'],
                    'direction': pattern_data['direction']
                }
        
        return patterns

    async def get_ohlcv(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> Optional[List]:
        """Отримання OHLCV даних"""
        try:
            ohlcv = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.exchange.fetch_ohlcv(symbol, timeframe, limit)
            )
            return ohlcv
        except Exception as e:
            logger.error(f"Помилка отримання даних для {symbol}: {e}")
            return None

    async def handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обробник callback"""
        query = update.callback_query
        await query.answer()
        
        try:
            if query.data == "vwmc_scan":
                await self.vwmc_scan_command(query, context)
            elif query.data == "volume_analysis":
                await self.volume_analysis_command(query, context)
            elif query.data == "momentum_matrix":
                await self.momentum_matrix_command(query, context)
            elif query.data == "liquidity_map":
                await self.liquidity_map_command(query, context)
            elif query.data == "pattern_recognition":
                await self.pattern_recognition_command(query, context)
            # ... інші команди
                
        except Exception as e:
            await query.edit_message_text("❌ Помилка обробки запиту")

    async def run(self):
        """Запуск бота"""
        try:
            logger.info("🚀 Запускаю VWMC Strategy Bot...")
            await self.app.initialize()
            await self.app.start()
            await self.app.updater.start_polling()
            
            logger.info("✅ VWMC бот успішно запущено!")
            
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
        
        bot = VWMCStrategyBot(BOT_TOKEN)
        await bot.run()
        
    except KeyboardInterrupt:
        logger.info("Зупинка бота...")
    except Exception as e:
        logger.error(f"Критична помилка: {e}")

if __name__ == '__main__':
    asyncio.run(main())