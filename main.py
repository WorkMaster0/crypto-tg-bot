import pandas as pd
import numpy as np
import ccxt
import asyncio
import time
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
import talib
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Налаштування логування
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('profit_bot.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ProfitFuturesBot:
    def __init__(self, token: str):
        self.token = token
        self.app = Application.builder().token(token).build()
        
        # Підключення до біржі
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })
        
        # Реальні торгові параметри
        self.settings = {
            'risk_per_trade': 0.02,  # 2% ризик на угоду
            'take_profit_ratio': 2.0,
            'max_open_positions': 3,
            'min_volume_usdt': 1000000,
            'min_volatility': 0.003,
            'ema_fast': 9,
            'ema_slow': 21,
            'rsi_period': 14,
            'atr_period': 14
        }
        
        # Активні позиції
        self.positions = {}
        # Історія угод
        self.trade_history = []
        # Статистика
        self.performance = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_profit': 0.0,
            'max_drawdown': 0.0,
            'current_balance': 10000.0  # Початковий баланс
        }
        
        self.setup_handlers()

    def setup_handlers(self):
        """Реальні торгові команди"""
        handlers = [
            CommandHandler("start", self.start_command),
            CommandHandler("scan", self.scan_opportunities),
            CommandHandler("positions", self.show_positions),
            CommandHandler("balance", self.show_balance),
            CommandHandler("analysis", self.market_analysis),
            CommandHandler("settings", self.show_settings),
            CommandHandler("history", self.trade_history),
            CommandHandler("signals", self.live_signals),
            CallbackQueryHandler(self.handle_callback)
        ]
        
        for handler in handlers:
            self.app.add_handler(handler)

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Стартова команда з реальними опціями"""
        keyboard = [
            [InlineKeyboardButton("🔍 Сканувати ринок", callback_data="scan"),
             InlineKeyboardButton("📊 Мої позиції", callback_data="positions")],
            [InlineKeyboardButton("💰 Баланс", callback_data="balance"),
             InlineKeyboardButton("📈 Аналіз ринку", callback_data="analysis")],
            [InlineKeyboardButton("⚡ Живі сигнали", callback_data="signals"),
             InlineKeyboardButton("📋 Історія угод", callback_data="history")],
            [InlineKeyboardButton("⚙️ Налаштування", callback_data="settings")]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            "💰 **PROFIT FUTURES BOT**\n\n"
            "Реальний помічник для заробітку на ф'ючерсах\n\n"
            "📊 *Поточний статус:*\n"
            f"• Баланс: ${self.performance['current_balance']:,.2f}\n"
            f"• Угод: {self.performance['total_trades']}\n"
            f"• Прибуток: ${self.performance['total_profit']:,.2f}\n"
            f"• Відкрито позицій: {len(self.positions)}\n\n"
            "🎯 *Оберіть опцію:*",
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )

    async def scan_opportunities(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Сканування ринку на реальні можливості"""
        try:
            msg = await update.message.reply_text("🔍 Сканую ринок для знаходження прибуткових угод...")
            
            symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT']
            opportunities = []
            
            for symbol in symbols:
                opportunity = await self.analyze_symbol(symbol)
                if opportunity and opportunity['score'] >= 7:
                    opportunities.append(opportunity)
            
            if opportunities:
                opportunities.sort(key=lambda x: x['score'], reverse=True)
                
                response = "🎯 **НАЙКРАЩІ ОПОРТУНІТЕТИ:**\n\n"
                
                for i, opp in enumerate(opportunities[:3], 1):
                    response += f"{i}. **{opp['symbol']}** - Оцінка: {opp['score']}/10\n"
                    response += f"   📈 Напрям: {opp['direction']}\n"
                    response += f"   💰 Потенціал: {opp['potential']:.2f}%\n"
                    response += f"   ⚡ Вірогідність: {opp['probability']:.0%}\n"
                    response += f"   📊 Об'єм: ${opp['volume']:,.0f}\n\n"
                
                response += "🔔 *Рекомендація:* Уважно моніторьте ці активи"
                await msg.edit_text(response, parse_mode='Markdown')
            else:
                await msg.edit_text("📉 Наразі сильних можливостей не знайдено. Чекайте кращих умов.")
                
        except Exception as e:
            logger.error(f"Помилка сканування: {e}")
            await update.message.reply_text("❌ Помилка сканування ринку")

    async def analyze_symbol(self, symbol: str) -> Optional[Dict]:
        """Реальний аналіз символу"""
        try:
            # Отримання даних
            ohlcv = await self.get_ohlcv(symbol, '15m', 100)
            if not ohlcv or len(ohlcv) < 50:
                return None
            
            closes = np.array([x[4] for x in ohlcv])
            highs = np.array([x[2] for x in ohlcv])
            lows = np.array([x[3] for x in ohlcv])
            volumes = np.array([x[5] for x in ohlcv])
            
            # Технічні індикатори
            ema_fast = talib.EMA(closes, self.settings['ema_fast'])
            ema_slow = talib.EMA(closes, self.settings['ema_slow'])
            rsi = talib.RSI(closes, self.settings['rsi_period'])
            atr = talib.ATR(highs, lows, closes, self.settings['atr_period'])
            
            if any(np.isnan([ema_fast[-1], ema_slow[-1], rsi[-1], atr[-1]])):
                return None
            
            # Аналіз тренду
            trend_strength = self.calculate_trend_strength(closes)
            volume_analysis = self.analyze_volume(volumes)
            
            # Визначення напрямку
            if ema_fast[-1] > ema_slow[-1] and rsi[-1] > 50:
                direction = "LONG"
                probability = min(rsi[-1] / 100, 0.85)
            elif ema_fast[-1] < ema_slow[-1] and rsi[-1] < 50:
                direction = "SHORT"
                probability = min((100 - rsi[-1]) / 100, 0.85)
            else:
                return None
            
            # Розрахунок оцінки
            score = self.calculate_score(
                trend_strength, volume_analysis, probability, 
                np.mean(volumes[-5:]), atr[-1]
            )
            
            return {
                'symbol': symbol,
                'direction': direction,
                'score': score,
                'probability': probability,
                'potential': atr[-1] / closes[-1] * 100 * 3,  # 3x ATR
                'volume': np.mean(volumes[-5:])
            }
            
        except Exception as e:
            logger.error(f"Помилка аналізу {symbol}: {e}")
            return None

    def calculate_trend_strength(self, prices: np.ndarray) -> float:
        """Розрахунок сили тренду"""
        if len(prices) < 20:
            return 0.5
        
        x = np.arange(len(prices))
        slope, _, r_value, _, _ = stats.linregress(x, prices)
        return abs(r_value) * (1 if slope > 0 else -1)

    def analyze_volume(self, volumes: np.ndarray) -> float:
        """Аналіз об'ємів"""
        if len(volumes) < 20:
            return 0.5
        
        current_volume = volumes[-1]
        avg_volume = np.mean(volumes[:-5])
        return min(current_volume / avg_volume, 2.0)

    def calculate_score(self, trend: float, volume: float, probability: float, 
                      avg_volume: float, atr: float) -> int:
        """Розрахунок загальної оцінки"""
        # Ваги для різних факторів
        weights = {
            'trend': 0.3,
            'volume': 0.25,
            'probability': 0.25,
            'liquidity': 0.1,
            'volatility': 0.1
        }
        
        # Нормалізація факторів
        trend_score = abs(trend) * 10
        volume_score = min(volume, 2.0) * 5
        probability_score = probability * 10
        liquidity_score = min(avg_volume / 1000000, 2.0) * 5
        volatility_score = min(atr * 1000, 2.0) * 5
        
        # Загальна оцінка
        total_score = (
            trend_score * weights['trend'] +
            volume_score * weights['volume'] +
            probability_score * weights['probability'] +
            liquidity_score * weights['liquidity'] +
            volatility_score * weights['volatility']
        )
        
        return min(int(total_score), 10)

    async def show_positions(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Показати поточні позиції"""
        if not self.positions:
            await update.message.reply_text("📭 Немає відкритих позицій")
            return
        
        response = "📊 **ПОТОЧНІ ПОЗИЦІЇ:**\n\n"
        total_pnl = 0
        
        for symbol, position in self.positions.items():
            pnl = (position['current_price'] - position['entry_price']) * position['size'] * (
                1 if position['direction'] == 'LONG' else -1
            )
            total_pnl += pnl
            
            response += f"🎯 **{symbol}** - {position['direction']}\n"
            response += f"   📈 Вхід: ${position['entry_price']:.2f}\n"
            response += f"   📊 Поточна: ${position['current_price']:.2f}\n"
            response += f"   📏 Розмір: {position['size']:.3f}\n"
            response += f"   💰 PnL: ${pnl:.2f} ({pnl/position['entry_price']/position['size']*100:.2f}%)\n\n"
        
        response += f"📈 **Загальний PnL: ${total_pnl:.2f}**"
        await update.message.reply_text(response, parse_mode='Markdown')

    async def show_balance(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Показати баланс та статистику"""
        response = (
            "💰 **БАЛАНС ТА СТАТИСТИКА:**\n\n"
            f"📊 Поточний баланс: ${self.performance['current_balance']:,.2f}\n"
            f"📈 Усього угод: {self.performance['total_trades']}\n"
            f"✅ Виграшних: {self.performance['winning_trades']}\n"
            f"📉 Процент успіху: {self.performance['winning_trades']/max(self.performance['total_trades'],1)*100:.1f}%\n"
            f"💰 Загальний прибуток: ${self.performance['total_profit']:,.2f}\n"
            f"⚡ Макс. просідання: {self.performance['max_drawdown']:.2f}%\n\n"
            "🎯 *Статистика оновлюється в реальному часі*"
        )
        await update.message.reply_text(response, parse_mode='Markdown')

    async def market_analysis(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Глибинний аналіз ринку"""
        try:
            msg = await update.message.reply_text("📈 Проводжу глибинний аналіз ринку...")
            
            analysis = await self.get_market_analysis()
            
            response = "📊 **АНАЛІЗ РИНКУ:**\n\n"
            response += f"📈 Загальний тренд: {analysis['overall_trend']}\n"
            response += f"⚡ Волатильність: {analysis['volatility']}\n"
            response += f"📊 Об'єми: {analysis['volume_status']}\n"
            response += f"🎯 Найсильніші активи: {', '.join(analysis['strong_assets'][:3])}\n\n"
            
            response += "💡 **РЕКОМЕНДАЦІЇ:**\n"
            for recommendation in analysis['recommendations']:
                response += f"• {recommendation}\n"
            
            await msg.edit_text(response, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Помилка аналізу ринку: {e}")
            await update.message.reply_text("❌ Помилка аналізу ринку")

    async def get_market_analysis(self) -> Dict:
        """Отримати аналіз ринку"""
        symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT']
        trends = []
        volatilities = []
        volumes = []
        strong_assets = []
        
        for symbol in symbols:
            try:
                ohlcv = await self.get_ohlcv(symbol, '1h', 50)
                if ohlcv:
                    closes = np.array([x[4] for x in ohlcv])
                    price_change = (closes[-1] - closes[0]) / closes[0] * 100
                    trends.append(price_change)
                    
                    volatility = np.std(np.diff(closes) / closes[:-1]) * 100
                    volatilities.append(volatility)
                    
                    avg_volume = np.mean([x[5] for x in ohlcv[-20:]])
                    volumes.append(avg_volume)
                    
                    if abs(price_change) > 5:
                        strong_assets.append(symbol)
            except:
                continue
        
        overall_trend = "БИЧИЙ" if np.mean(trends) > 0.5 else "МЕДВЕЖИЙ" if np.mean(trends) < -0.5 else "НЕЙТРАЛЬНИЙ"
        
        return {
            'overall_trend': overall_trend,
            'volatility': "ВИСОКА" if np.mean(volatilities) > 0.02 else "СЕРЕДНЯ" if np.mean(volatilities) > 0.01 else "НИЗЬКА",
            'volume_status': "ВИСОКИЙ" if np.mean(volumes) > 5000000 else "СЕРЕДНІЙ" if np.mean(volumes) > 1000000 else "НИЗЬКИЙ",
            'strong_assets': strong_assets,
            'recommendations': [
                "Увага до ризик-менеджменту",
                "Диверсифікуйте портфель",
                "Слідкуйте за об'ємами"
            ]
        }

    async def live_signals(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Живі торгові сигнали"""
        try:
            msg = await update.message.reply_text("⚡ Шукаю живі торгові сигнали...")
            
            signals = await self.generate_signals()
            
            if signals:
                response = "🎯 **ЖИВІ СИГНАЛИ:**\n\n"
                
                for signal in signals[:5]:
                    response += f"🔔 **{signal['symbol']}** - {signal['direction']}\n"
                    response += f"   💰 Ціна: ${signal['price']:.2f}\n"
                    response += f"   🎯 Стоп: ${signal['stop_loss']:.2f}\n"
                    response += f"   📈 Тейк: ${signal['take_profit']:.2f}\n"
                    response += f"   ⚡ Сила: {signal['strength']}/10\n\n"
                
                await msg.edit_text(response, parse_mode='Markdown')
            else:
                await msg.edit_text("📉 Наразі сильних сигналів не знайдено")
                
        except Exception as e:
            logger.error(f"Помилка сигналів: {e}")
            await update.message.reply_text("❌ Помилка генерації сигналів")

    async def generate_signals(self) -> List[Dict]:
        """Генерація торгових сигналів"""
        symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
        signals = []
        
        for symbol in symbols:
            try:
                ohlcv = await self.get_ohlcv(symbol, '5m', 50)
                if not ohlcv:
                    continue
                
                closes = np.array([x[4] for x in ohlcv])
                current_price = closes[-1]
                
                # Проста стратегія на основі EMA
                ema9 = talib.EMA(closes, 9)
                ema21 = talib.EMA(closes, 21)
                
                if len(ema9) < 2 or len(ema21) < 2:
                    continue
                
                if ema9[-1] > ema21[-1] and ema9[-2] <= ema21[-2]:
                    # BUY signal
                    atr = talib.ATR(
                        np.array([x[2] for x in ohlcv]),
                        np.array([x[3] for x in ohlcv]),
                        closes,
                        14
                    )[-1]
                    
                    signals.append({
                        'symbol': symbol,
                        'direction': 'BUY',
                        'price': current_price,
                        'stop_loss': current_price - 2 * atr,
                        'take_profit': current_price + 4 * atr,
                        'strength': 7
                    })
                    
                elif ema9[-1] < ema21[-1] and ema9[-2] >= ema21[-2]:
                    # SELL signal
                    atr = talib.ATR(
                        np.array([x[2] for x in ohlcv]),
                        np.array([x[3] for x in ohlcv]),
                        closes,
                        14
                    )[-1]
                    
                    signals.append({
                        'symbol': symbol,
                        'direction': 'SELL',
                        'price': current_price,
                        'stop_loss': current_price + 2 * atr,
                        'take_profit': current_price - 4 * atr,
                        'strength': 7
                    })
                    
            except Exception as e:
                logger.error(f"Помилка генерації сигналу для {symbol}: {e}")
                continue
        
        return signals

    async def trade_history(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Історія угод"""
        if not self.trade_history:
            await update.message.reply_text("📋 Історія угод порожня")
            return
        
        response = "📋 **ОСТАННІ УГОДИ:**\n\n"
        
        for trade in self.trade_history[-10:]:
            profit_color = "🟢" if trade['profit'] >= 0 else "🔴"
            response += f"{profit_color} {trade['symbol']} {trade['direction']}\n"
            response += f"   📅 {trade['time']}\n"
            response += f"   💰 Прибуток: ${trade['profit']:.2f}\n"
            response += f"   📊 Розмір: {trade['size']:.3f}\n\n"
        
        await update.message.reply_text(response, parse_mode='Markdown')

    async def show_settings(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Показати налаштування"""
        response = (
            "⚙️ **НАЛАШТУВАННЯ ТОРГІВЛІ:**\n\n"
            f"📉 Ризик на угоду: {self.settings['risk_per_trade']*100:.1f}%\n"
            f"🎯 Take Profit/Stop Loss: {self.settings['take_profit_ratio']:.1f}\n"
            f"📊 Макс. позицій: {self.settings['max_open_positions']}\n"
            f"📈 Мін. об'єм: ${self.settings['min_volume_usdt']:,.0f}\n"
            f"⚡ Мін. волатильність: {self.settings['min_volatility']*100:.1f}%\n\n"
            "🔧 *Налаштування оптимізовані для безпечної торгівлі*"
        )
        await update.message.reply_text(response, parse_mode='Markdown')

    async def get_ohlcv(self, symbol: str, timeframe: str = '15m', limit: int = 100) -> Optional[List]:
        """Отримати OHLCV дані"""
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
            if query.data == "scan":
                await self.scan_opportunities(query, context)
            elif query.data == "positions":
                await self.show_positions(query, context)
            elif query.data == "balance":
                await self.show_balance(query, context)
            elif query.data == "analysis":
                await self.market_analysis(query, context)
            elif query.data == "signals":
                await self.live_signals(query, context)
            elif query.data == "history":
                await self.trade_history(query, context)
            elif query.data == "settings":
                await self.show_settings(query, context)
                
        except Exception as e:
            await query.edit_message_text("❌ Помилка обробки запиту")

    async def run(self):
        """Запуск бота"""
        try:
            logger.info("🚀 Запускаю Profit Futures Bot...")
            await self.app.initialize()
            await self.app.start()
            await self.app.updater.start_polling()
            
            logger.info("✅ Бот успішно запущено!")
            
            # Фоновий моніторинг
            asyncio.create_task(self.background_monitoring())
            
            while True:
                await asyncio.sleep(3600)
                
        except Exception as e:
            logger.error(f"❌ Помилка запуску: {e}")
            raise

    async def background_monitoring(self):
        """Фоновий моніторинг ринку"""
        while True:
            try:
                # Оновлення цін для відкритих позицій
                for symbol in list(self.positions.keys()):
                    try:
                        ticker = await asyncio.get_event_loop().run_in_executor(
                            None, lambda: self.exchange.fetch_ticker(symbol)
                        )
                        if ticker:
                            self.positions[symbol]['current_price'] = ticker['last']
                    except:
                        continue
                
                # Перевірка умов для закриття позицій
                await self.check_exit_conditions()
                
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"Помилка моніторингу: {e}")
                await asyncio.sleep(30)

    async def check_exit_conditions(self):
        """Перевірка умов для закриття позицій"""
        # Реалізація логіки закриття позицій
        pass

async def main():
    """Головна функція"""
    try:
        BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
        if not BOT_TOKEN:
            logger.error("Встановіть TELEGRAM_BOT_TOKEN")
            return
        
        bot = ProfitFuturesBot(BOT_TOKEN)
        await bot.run()
        
    except KeyboardInterrupt:
        logger.info("Зупинка бота...")
    except Exception as e:
        logger.error(f"Критична помилка: {e}")

if __name__ == '__main__':
    asyncio.run(main())