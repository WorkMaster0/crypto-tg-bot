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
from typing import Dict, List, Optional, Tuple
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
            'USDC', 'USDT', 'BUSD', 'DAI', 'TUSD', 'PAX', 'UST', 'HUSD', 'GUSD',
            'BVND', 'IDRT', 'BIDR', 'BKRW', 'NGN', 'UAH', 'VAI', 'SUSD', 'USDN'
        }
        
        # Регулярний вираз для фільтрації сміття
        self.garbage_pattern = re.compile(
            r'.*[0-9]+[LNS].*|.*UP|DOWN|BEAR|BULL|HALF|QUARTER.*|.*[0-9]{3,}.*',
            re.IGNORECASE
        )
        
        # Пороги для пампу
        self.pump_thresholds = {
            'volume_ratio': 2.8,
            'price_change_1h': 7.0,
            'price_change_5m': 2.5,
            'price_change_24h': 15.0,
            'rsi_threshold': 62,
            'buy_pressure_ratio': 1.3,
            'macd_signal': 0.0008,
            'min_volume': 100000,           # Збільшено мінімальний об'єм
            'max_volume': 10000000,         # Збільшено максимальний об'єм
            'min_market_cap': 1000000,      # Мінімальна капіталізація $1M
            'max_market_cap': 500000000,    # Максимальна капіталізація $500M
            'liquidity_score': 0.7,
            'volatility_ratio': 2.0
        }
        
        # Пороги для дампу
        self.dump_thresholds = {
            'volume_ratio': 2.5,
            'price_change_1h': -6.0,
            'price_change_5m': -2.0,
            'price_change_24h': -12.0,
            'rsi_threshold': 38,
            'sell_pressure_ratio': 1.4,
            'macd_signal': -0.0007,
            'min_volume': 100000,           # Збільшено мінімальний об'єм
            'max_volume': 10000000,         # Збільшено максимальний об'єм
            'min_market_cap': 1000000,      # Мінімальна капіталізація $1M
            'max_market_cap': 500000000,    # Максимальна капіталізація $500M
            'liquidity_score': 0.7,
            'volatility_ratio': 1.8
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
                'settings': {
                    'pump': self.pump_thresholds,
                    'dump': self.dump_thresholds
                }
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
        self.app.add_handler(CommandHandler("whale", self.whale_command))
        self.app.add_handler(CommandHandler("topsignals", self.top_signals_command))
        self.app.add_handler(CommandHandler("performance", self.performance_command))
        self.app.add_handler(CommandHandler("analysis", self.market_analysis_command))
        self.app.add_handler(CallbackQueryHandler(self.button_handler))

    def is_garbage_symbol(self, symbol: str) -> bool:
        """Перевіряє чи символ є сміттям"""
        symbol = symbol.upper()
        
        if symbol in self.garbage_symbols:
            return True
        
        if self.garbage_pattern.match(symbol):
            return True
        
        if len(symbol) > 12:
            return True
        
        if any(char.isdigit() for char in symbol[1:-1]):
            return True
            
        if symbol.endswith(('UP', 'DOWN', 'BULL', 'BEAR')):
            return True
            
        return False

    async def estimate_market_cap(self, symbol: str, current_price: float, volume: float) -> float:
        """Оцінка ринкової капіталізації"""
        try:
            # Для більших монет використовуємо більш консервативний множник
            multiplier = 20  # Для монет з капою 1M-500M
            
            # Корекція множника в залежності від об'єму
            if volume > 5000000:  # Об'єм понад 5M USDT
                multiplier = 15
            elif volume > 2000000:  # Об'єм понад 2M USDT
                multiplier = 18
            elif volume < 500000:  # Об'єм менше 500K USDT
                multiplier = 25
                
            estimated_cap = current_price * volume * multiplier
            return estimated_cap
            
        except Exception as e:
            logger.error(f"Помилка оцінки капіталізації для {symbol}: {e}")
            return 0

    async def get_market_data(self, symbol: str) -> Optional[Dict]:
        try:
            if self.is_garbage_symbol(symbol):
                return None
            
            url = f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}USDT"
            response = requests.get(url, timeout=8)
            data = response.json()
            
            if 'code' in data:
                return None
            
            quote_volume = float(data['quoteVolume'])
            price_change = float(data['priceChangePercent'])
            current_price = float(data['lastPrice'])
            
            # Оцінка капіталізації
            market_cap = await self.estimate_market_cap(symbol, current_price, quote_volume)
            
            # Фільтрація за капіталізацією (1M - 500M)
            if (market_cap < self.pump_thresholds['min_market_cap'] or 
                market_cap > self.pump_thresholds['max_market_cap']):
                logger.info(f"Пропускаємо {symbol} через капіталізацію: ${market_cap:,.0f}")
                return None
            
            # Фільтрація за об'ємом та зміною ціни
            if (quote_volume < self.pump_thresholds['min_volume'] or 
                quote_volume > self.pump_thresholds['max_volume'] or
                abs(price_change) < 3.0):  # Зменшено мінімальну зміну ціни
                return None
            
            klines_data = {}
            timeframes = {
                '5m': '&interval=5m&limit=100',
                '15m': '&interval=15m&limit=100',
                '1h': '&interval=1h&limit=50'
            }
            
            for tf, params in timeframes.items():
                klines_url = f"https://api.binance.com/api/v3/klines?symbol={symbol}USDT{params}"
                klines_response = requests.get(klines_url, timeout=8)
                klines_data[tf] = klines_response.json()
            
            orderbook_url = f"https://api.binance.com/api/v3/depth?symbol={symbol}USDT&limit=10"
            orderbook_response = requests.get(orderbook_url, timeout=8)
            orderbook_data = orderbook_response.json()
            
            market_data = {
                'symbol': symbol,
                'price': current_price,
                'volume': float(data['volume']),
                'quote_volume': quote_volume,
                'market_cap': market_cap,
                'price_change_24h': price_change,
                'price_change': float(data['priceChange']),
                'high': float(data['highPrice']),
                'low': float(data['lowPrice']),
                'klines': klines_data,
                'orderbook': orderbook_data,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Дані для {symbol}: капа ${market_cap:,.0f}, ціна ${current_price}, зміна {price_change:.1f}%")
            return market_data
            
        except Exception as e:
            logger.error(f"Помилка отримання даних для {symbol}: {e}")
            return None

    # Решта методів залишаються незмінними (calculate_rsi, calculate_ema, calculate_macd, etc.)
    # ... [всі технічні функції] ...

    async def scan_for_pump_dump(self):
        """Основна функція сканування"""
        try:
            start_time = time.time()
            
            response = requests.get("https://api.binance.com/api/v3/ticker/24hr", timeout=10)
            all_data = response.json()
            
            usdt_pairs = [
                x for x in all_data 
                if x['symbol'].endswith('USDT') 
                and not self.is_garbage_symbol(x['symbol'].replace('USDT', ''))
            ]
            
            # Сортуємо за абсолютною зміною ціни (найбільші рухи)
            sorted_by_change = sorted(
                usdt_pairs,
                key=lambda x: abs(float(x['priceChangePercent'])),
                reverse=True
            )[:50]  # Топ 50 за зміною ціни
            
            results = {'pump': [], 'dump': []}
            
            for coin in sorted_by_change:
                symbol = coin['symbol'].replace('USDT', '')
                
                if symbol in self.coin_blacklist:
                    continue
                    
                market_data = await self.get_market_data(symbol)
                if not market_data:
                    continue
                
                indicators = self.calculate_advanced_indicators(market_data['klines']['15m'])
                
                pump_score = self.calculate_pump_score(market_data, indicators)
                if pump_score > 0.65:
                    results['pump'].append({
                        'symbol': symbol,
                        'score': pump_score,
                        'price': market_data['price'],
                        'volume': market_data['quote_volume'],
                        'market_cap': market_data['market_cap'],
                        'change_24h': market_data['price_change_24h'],
                        'indicators': indicators
                    })
                
                dump_score = self.calculate_dump_score(market_data, indicators)
                if dump_score > 0.65:
                    results['dump'].append({
                        'symbol': symbol,
                        'score': dump_score,
                        'price': market_data['price'],
                        'volume': market_data['quote_volume'],
                        'market_cap': market_data['market_cap'],
                        'change_24h': market_data['price_change_24h'],
                        'indicators': indicators
                    })
            
            for key in results:
                results[key].sort(key=lambda x: x['score'], reverse=True)
            
            scan_time = time.time() - start_time
            self.performance_stats['total_scans'] += 1
            self.performance_stats['signals_found'] += len(results['pump']) + len(results['dump'])
            self.performance_stats['avg_scan_time'] = (
                self.performance_stats['avg_scan_time'] * (self.performance_stats['total_scans'] - 1) + scan_time
            ) / self.performance_stats['total_scans']
            
            return results
            
        except Exception as e:
            logger.error(f"Помилка сканування: {e}")
            return {'pump': [], 'dump': []}

    # Решта методів залишаються незмінними (calculate_pump_score, calculate_dump_score, etc.)
    # ... [всі інші функції] ...

    async def scan_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда сканування"""
        await update.message.reply_text("🔍 Запускаю сканування топ монет за зміною ціни...")
        results = await self.scan_for_pump_dump()
        
        message = "📊 **Результати сканування**\n\n"
        
        if not results['pump'] and not results['dump']:
            message += "ℹ️ Сигналів не знайдено. Спробуйте пізніше.\n"
        else:
            for signal_type in ['pump', 'dump']:
                if results[signal_type]:
                    message += f"{'🚀' if signal_type == 'pump' else '📉'} **{signal_type.upper()}:**\n"
                    for i, signal in enumerate(results[signal_type][:5], 1):
                        message += f"{i}. `{signal['symbol']}` - {signal['score']:.1%} "
                        message += f"(капа: ${signal['market_cap']:,.0f}, зміна: {signal['change_24h']:+.1f}%)\n"
                    message += "\n"
        
        await update.message.reply_text(message, parse_mode='Markdown')

    async def settings_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда налаштувань"""
        settings_msg = "⚙️ **Поточні налаштування**\n\n"
        
        settings_msg += "🚀 **Pump Detection:**\n"
        settings_msg += f"• Мін. капіталізація: ${self.pump_thresholds['min_market_cap']:,.0f}\n"
        settings_msg += f"• Макс. капіталізація: ${self.pump_thresholds['max_market_cap']:,.0f}\n"
        settings_msg += f"• Мін. об'єм: {self.pump_thresholds['min_volume']:,.0f} USDT\n"
        settings_msg += f"• Volume ratio: {self.pump_thresholds['volume_ratio']}x\n"
        settings_msg += f"• Зміна ціни 5m: {self.pump_thresholds['price_change_5m']}%\n"
        
        settings_msg += "\n📉 **Dump Detection:**\n"
        settings_msg += f"• Мін. капіталізація: ${self.dump_thresholds['min_market_cap']:,.0f}\n"
        settings_msg += f"• Макс. капіталізація: ${self.dump_thresholds['max_market_cap']:,.0f}\n"
        settings_msg += f"• Мін. об'єм: {self.dump_thresholds['min_volume']:,.0f} USDT\n"
        settings_msg += f"• Volume ratio: {self.dump_thresholds['volume_ratio']}x\n"
        settings_msg += f"• Зміна ціни 5m: {self.dump_thresholds['price_change_5m']}%\n"
        
        await update.message.reply_text(settings_msg, parse_mode='Markdown')

    async def blacklist_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда чорного списку"""
        if context.args:
            coin = context.args[0].upper()
            if coin in self.coin_blacklist:
                self.coin_blacklist.remove(coin)
                await update.message.reply_text(f"✅ {coin} видалено з чорного списку")
            else:
                self.coin_blacklist.add(coin)
                await update.message.reply_text(f"✅ {coin} додано до чорного списку")
        else:
            blacklist_msg = "🚫 **Чорний список:**\n"
            if self.coin_blacklist:
                blacklist_msg += "\n".join(f"• {coin}" for coin in sorted(self.coin_blacklist))
            else:
                blacklist_msg += "Порожній"
            await update.message.reply_text(blacklist_msg)

    async def debug_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда дебагу"""
        await update.message.reply_text("🛠️ Режим дебагу...")
        
        test_symbol = "BTC"
        market_data = await self.get_market_data(test_symbol)
        
        if market_data:
            indicators = self.calculate_advanced_indicators(market_data['klines']['15m'])
            
            message = f"🔧 **Дебаг для {test_symbol}:**\n"
            message += f"• Ціна: ${market_data['price']}\n"
            message += f"• Об'єм: {market_data['quote_volume']:,.0f} USDT\n"
            message += f"• 24h зміна: {market_data['price_change_24h']:.1f}%\n"
            message += f"• RSI: {indicators.get('rsi', 0):.1f}\n"
            message += f"• Volume ratio: {indicators.get('volume_ratio', 0):.2f}x"
            
            await update.message.reply_text(message, parse_mode='Markdown')
        else:
            await update.message.reply_text("❌ Не вдалося отримати дані")

    async def whale_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда whale alert"""
        await update.message.reply_text(
            "🐋 **Whale Alert System**\n\n"
            "• Мінімальний ордер: $50,000\n"
            "• Моніторинг: Топ 20 монет\n"
            "• Сповіщення: Real-time\n\n"
            "_Функція в активній розробці_",
            parse_mode='Markdown'
        )

    async def top_signals_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда топ сигналів"""
        if not self.last_signals:
            await update.message.reply_text("📊 Ще немає збережених сигналів")
            return
        
        message = "📈 **ТОП останніх сигналів**\n\n"
        
        recent_signals = list(self.last_signals.values())[-10:]
        for i, signal in enumerate(recent_signals, 1):
            message += f"{i}. {signal['data']['symbol']} - {signal['type']} - {signal['data']['score']:.1%}\n"
        
        await update.message.reply_text(message)

    async def performance_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда продуктивності"""
        perf_msg = "🏆 **Статистика продуктивності**\n\n"
        perf_msg += f"• 📊 Всього сканувань: {self.performance_stats['total_scans']}\n"
        perf_msg += f"• 📈 Знайдено сигналів: {self.performance_stats['signals_found']}\n"
        perf_msg += f"• ⚡ Середній час: {self.performance_stats['avg_scan_time']:.2f}s\n"
        perf_msg += f"• ⏰ Аптайм: {timedelta(seconds=int(time.time() - self.start_time))}\n"
        
        await update.message.reply_text(perf_msg, parse_mode='Markdown')

    async def market_analysis_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда аналізу ринку"""
        analysis_msg = "📈 **Аналіз ринку**\n\n"
        
        try:
            response = requests.get("https://api.binance.com/api/v3/ticker/24hr", timeout=10)
            data = response.json()
            
            total_volume = sum(float(x['quoteVolume']) for x in data if 'USDT' in x['symbol'])
            avg_change = np.mean([float(x['priceChangePercent']) for x in data if 'USDT' in x['symbol']])
            
            analysis_msg += f"• 📊 Загальний об'єм: {total_volume:,.0f} USDT\n"
            analysis_msg += f"• 📈 Середня зміна: {avg_change:+.2f}%\n"
            analysis_msg += f"• 🔍 Монет у моніторингу: {len(self.last_signals)}\n"
            
        except Exception as e:
            analysis_msg += "❌ Помилка аналізу ринку\n"
        
        await update.message.reply_text(analysis_msg, parse_mode='Markdown')

    def handle_webhook(self, data: Dict) -> str:
        """Обробка вебхук запитів"""
        try:
            if data.get('type') == 'manual_scan':
                results = asyncio.run(self.scan_for_pump_dump())
                return jsonify(results)
            elif data.get('type') == 'update_settings':
                self.update_settings(data.get('settings', {}))
                return jsonify({'status': 'success'})
            return jsonify({'error': 'Unknown webhook type'})
        except Exception as e:
            return jsonify({'error': str(e)})

    def update_settings(self, new_settings: Dict):
        """Оновлення налаштувань"""
        if 'pump' in new_settings:
            self.pump_thresholds.update(new_settings['pump'])
        if 'dump' in new_settings:
            self.dump_thresholds.update(new_settings['dump'])

    async def button_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()
        
        if query.data == "scan_now":
            await query.edit_message_text("🔍 Сканую топ монети за зміною ціни...")
            results = await self.scan_for_pump_dump()
            
            message = "📊 **Результати сканування**\n\n"
            
            if not results['pump'] and not results['dump']:
                message += "ℹ️ Сигналів не знайдено\n"
            else:
                for signal_type in ['pump', 'dump']:
                    if results[signal_type]:
                        message += f"{'🚀' if signal_type == 'pump' else '📉'} **{signal_type.upper()}:**\n"
                        for i, signal in enumerate(results[signal_type][:3], 1):
                            message += f"{i}. `{signal['symbol']}` - {signal['score']:.1%}\n"
                        message += "\n"
            
            await query.edit_message_text(message, parse_mode='Markdown')
            
        elif query.data == "settings":
            settings_msg = "⚙️ **Поточні налаштування**\n\n"
            settings_msg += "🚀 **Pump Detection:**\n"
            for k, v in self.pump_thresholds.items():
                settings_msg += f"• {k}: {v}\n"
            settings_msg += "\n📉 **Dump Detection:**\n"
            for k, v in self.dump_thresholds.items():
                settings_msg += f"• {k}: {v}\n"
            await query.edit_message_text(settings_msg, parse_mode='Markdown')
            
        elif query.data == "stats":
            stats_msg = "📊 **Статистика бота:**\n\n"
            stats_msg += f"• Останні сигнали: {len(self.last_signals)}\n"
            stats_msg += f"• Чорний список: {len(self.coin_blacklist)} монет\n"
            stats_msg += f"• Усього сканувань: {self.performance_stats['total_scans']}\n"
            await query.edit_message_text(stats_msg, parse_mode='Markdown')
            
        elif query.data == "blacklist":
            if self.coin_blacklist:
                blacklist_msg = "🚫 **Чорний список:**\n"
                blacklist_msg += "\n".join(f"• {coin}" for coin in sorted(self.coin_blacklist))
            else:
                blacklist_msg = "✅ Чорний список порожній"
            await query.edit_message_text(blacklist_msg)
            
        elif query.data == "whale_alert":
            await query.edit_message_text("🐋 **Whale Alert**\n\nМоніторинг великих ордерів...", parse_mode='Markdown')
            
        elif query.data == "top_signals":
            if not self.last_signals:
                await query.edit_message_text("📊 Ще немає сигналів")
                return
            message = "📈 **ТОП сигналів:**\n\n"
            recent_signals = list(self.last_signals.values())[-5:]
            for i, signal in enumerate(recent_signals, 1):
                message += f"{i}. {signal['data']['symbol']} - {signal['type']} - {signal['data']['score']:.1%}\n"
            await query.edit_message_text(message)
            
        elif query.data == "market_analysis":
            await self.market_analysis_command(update, context)
            
        elif query.data == "performance":
            await self.performance_command(update, context)

    def run_flask(self):
        """Запуск Flask сервера"""
        port = int(os.environ.get('PORT', 5000))
        self.flask_app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)

        def run(self):
        """Запуск бота"""
        print("🤖 Запуск ULTIMATE PUMP/DUMP BOT...")
        print("🎯 Спеціалізація: Памп/дамп стратегії")
        print("📊 Сканування: Топ монети за зміною ціни")
        print("💰 Капіталізація: $1M - $500M")
        print("💎 Версія: 2.1 (Large Cap Edition)")
        
        flask_thread = threading.Thread(target=self.run_flask, daemon=True)
        flask_thread.start()
        
        self.app.run_polling()

# Використання
if __name__ == "__main__":
    TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not TOKEN:
        print("❌ Помилка: TELEGRAM_BOT_TOKEN не встановлено")
        exit(1)
    
    bot = AdvancedPumpDumpBot(TOKEN)
    bot.run()