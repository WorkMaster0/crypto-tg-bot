import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
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

# Налаштування логування
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedPumpDumpBot:
    def __init__(self, token: str):
        self.token = token
        self.app = Application.builder().token(token).build()
        self.flask_app = Flask(__name__)
        self.setup_flask_routes()
        
        # Списки для фільтрації
        self.popular_coins = {'BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'ADA', 'AVAX', 'DOT', 'DOGE', 'SHIB', 
                             'MATIC', 'LTC', 'LINK', 'ATOM', 'UNI', 'XMR', 'ETC', 'XLM', 'BCH', 'FIL',
                             'APT', 'ARB', 'OP', 'SUI', 'SEI', 'NEAR', 'ALGO', 'FTM', 'AAVE', 'COMP',
                             'MKR', 'SNX', 'CRV', 'SUSHI', '1INCH', 'LDO', 'RUNE', 'INJ', 'IMX', 'RNDR'}
        
        self.garbage_symbols = {'USD', 'EUR', 'GBP', 'JPY', 'CNY', 'RUB', 'TRY', 'BRL', 'KRW', 'AUD',
                               'USDC', 'USDT', 'BUSD', 'DAI', 'TUSD', 'PAX', 'UST', 'HUSD', 'GUSD',
                               'BVND', 'IDRT', 'BIDR', 'BKRW', 'NGN', 'UAH', 'VAI', 'SUSD', 'USDN'}
        
        # Регулярний вираз для фільтрації сміття
        self.garbage_pattern = re.compile(r'.*[0-9]+[LNS].*|.*UP|DOWN|BEAR|BULL|HALF|QUARTER.*', re.IGNORECASE)
        
        # Пороги для пампу
        self.pump_thresholds = {
            'volume_ratio': 3.0,
            'price_change_1h': 5.0,
            'price_change_5m': 2.0,
            'rsi_threshold': 60,
            'buy_pressure_ratio': 1.2,
            'macd_signal': 0.0005,
            'min_volume': 100000  # Мінімальний об'єм в USDT
        }
        
        # Пороги для дампу
        self.dump_thresholds = {
            'volume_ratio': 2.5,
            'price_change_1h': -4.0,
            'price_change_5m': -1.5,
            'rsi_threshold': 40,
            'sell_pressure_ratio': 1.5,
            'macd_signal': -0.0005,
            'min_volume': 100000  # Мінімальний об'єм в USDT
        }
        
        self.coin_blacklist = set()
        self.last_signals = {}
        self.whale_alert_cooldown = {}
        self.setup_handlers()
        
    def setup_flask_routes(self):
        @self.flask_app.route('/webhook', methods=['POST'])
        def webhook():
            data = request.json
            return self.handle_webhook(data)
            
        @self.flask_app.route('/stats', methods=['GET'])
        def stats():
            return jsonify({
                'last_signals': self.last_signals,
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
            return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})
    
    def setup_handlers(self):
        self.app.add_handler(CommandHandler("start", self.start))
        self.app.add_handler(CommandHandler("scan", self.scan_command))
        self.app.add_handler(CommandHandler("settings", self.settings_command))
        self.app.add_handler(CommandHandler("blacklist", self.blacklist_command))
        self.app.add_handler(CommandHandler("debug", self.debug_command))
        self.app.add_handler(CommandHandler("whale", self.whale_command))
        self.app.add_handler(CommandHandler("topsignals", self.top_signals_command))
        self.app.add_handler(CallbackQueryHandler(self.button_handler))
    
    def is_garbage_symbol(self, symbol: str) -> bool:
        """Перевіряє чи символ є сміттям"""
        if symbol in self.garbage_symbols:
            return True
        
        if self.garbage_pattern.match(symbol):
            return True
        
        # Фільтр довгих та дивних символів
        if len(symbol) > 10:
            return True
        
        # Фільтр символів з цифрами посеред назви
        if any(char.isdigit() for char in symbol[1:-1]):
            return True
            
        return False

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        keyboard = [
            [InlineKeyboardButton("🔍 Сканувати зараз", callback_data="scan_now")],
            [InlineKeyboardButton("⚙️ Налаштування", callback_data="settings")],
            [InlineKeyboardButton("📊 Статистика", callback_data="stats")],
            [InlineKeyboardButton("🚫 Чорний список", callback_data="blacklist")],
            [InlineKeyboardButton("🐋 Whale Alert", callback_data="whale_alert")],
            [InlineKeyboardButton("📈 ТОП сигнали", callback_data="top_signals")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            "🤖 Advanced Pump & Dump Detect Bot\n\n"
            "Можливості:\n"
            "• 🚀 Детекція пампів\n"
            "• 📉 Детекція дампів\n"
            "• 🐋 Whale transactions monitoring\n"
            "• 📊 RSI & MACD аналіз\n"
            "• 🔄 Volume anomaly detection\n"
            "• 🌊 Order book analysis\n"
            "• ⚡ Real-time alerts\n"
            "• 🌐 Webhook integration",
            reply_markup=reply_markup
        )

    async def get_market_data(self, symbol: str) -> Optional[Dict]:
        try:
            logger.info(f"Отримання даних для {symbol}")
            
            # Перевірка чи символ не сміття
            if self.is_garbage_symbol(symbol):
                logger.info(f"Пропускаємо сміття: {symbol}")
                return None
            
            # Основні дані
            url = f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}USDT"
            response = requests.get(url, timeout=10)
            data = response.json()
            
            # Перевірка на помилку API
            if 'code' in data:
                logger.warning(f"API помилка для {symbol}: {data}")
                return None
            
            # Перевірка мінімального об'єму
            quote_volume = float(data['quoteVolume'])
            if quote_volume < self.pump_thresholds['min_volume']:
                logger.info(f"Пропускаємо {symbol} через низький об'єм: {quote_volume:,.0f} USDT")
                return None
            
            # Клайнси для різних таймфреймів
            timeframes = {
                '5m': '&interval=5m&limit=100',
                '1h': '&interval=1h&limit=50',
                '15m': '&interval=15m&limit=100'
            }
            
            klines_data = {}
            for tf, params in timeframes.items():
                klines_url = f"https://api.binance.com/api/v3/klines?symbol={symbol}USDT{params}"
                klines_response = requests.get(klines_url, timeout=10)
                klines_data[tf] = klines_response.json()
            
            # Order book data
            orderbook_url = f"https://api.binance.com/api/v3/depth?symbol={symbol}USDT&limit=10"
            orderbook_response = requests.get(orderbook_url, timeout=10)
            orderbook_data = orderbook_response.json()
            
            market_data = {
                'symbol': symbol,
                'price': float(data['lastPrice']),
                'volume': float(data['volume']),
                'price_change': float(data['priceChangePercent']),
                'high': float(data['highPrice']),
                'low': float(data['lowPrice']),
                'quote_volume': quote_volume,
                'klines': klines_data,
                'orderbook': orderbook_data
            }
            
            logger.info(f"Дані отримані для {symbol}: price={market_data['price']}, volume={market_data['quote_volume']:,.0f} USDT")
            return market_data
            
        except Exception as e:
            logger.error(f"Помилка отримання даних для {symbol}: {e}")
            return None

    # Інші методи (calculate_rsi, calculate_ema, calculate_macd, etc.) залишаються незмінними
    # ... [тут всі технічні функції з попереднього коду] ...

    async def scan_top_coins(self, scan_type: str = 'both'):
        try:
            logger.info(f"Початок сканування типу: {scan_type}")
            
            response = requests.get("https://api.binance.com/api/v3/ticker/24hr", timeout=15)
            all_data = response.json()
            
            usdt_pairs = [x for x in all_data if x['symbol'].endswith('USDT')]
            
            # Фільтрація та сортування по об'єму
            filtered_pairs = []
            for pair in usdt_pairs:
                symbol = pair['symbol'].replace('USDT', '')
                quote_volume = float(pair['quoteVolume'])
                
                # Пропускаємо сміття та низькооб'ємні пари
                if (not self.is_garbage_symbol(symbol) and 
                    symbol not in self.coin_blacklist and
                    quote_volume >= self.pump_thresholds['min_volume']):
                    filtered_pairs.append(pair)
            
            # Сортуємо по об'єму
            sorted_by_volume = sorted(filtered_pairs, 
                                    key=lambda x: float(x['quoteVolume']), 
                                    reverse=True)[:30]  # Топ 30 по об'єму після фільтрації
            
            logger.info(f"Знайдено {len(sorted_by_volume)} після фільтрації (з {len(usdt_pairs)})")
            
            results = {'pump': [], 'dump': []}
            
            for coin in sorted_by_volume:
                symbol = coin['symbol'].replace('USDT', '')
                    
                logger.info(f"Аналіз монети: {symbol}")
                market_data = await self.get_market_data(symbol)
                if not market_data:
                    continue
                
                # Технічні індикатори
                tech_indicators = await self.calculate_technical_indicators(
                    market_data['klines']['5m']
                )
                
                # Детекція пампу
                if scan_type in ['both', 'pump']:
                    pump_result = await self.detect_pump_pattern(market_data, tech_indicators)
                    if pump_result['score'] > 0.4:
                        results['pump'].append({
                            'symbol': symbol,
                            'score': pump_result['score'],
                            'signals': pump_result['signals'],
                            'confidence': pump_result['confidence'],
                            'price': market_data['price'],
                            'volume': market_data['volume'],
                            'quote_volume': market_data['quote_volume'],
                            'price_change_24h': market_data['price_change']
                        })
                        logger.info(f"Знайдено pump сигнал для {symbol}: {pump_result['score']:.2%}")
                
                # Детекція дампу
                if scan_type in ['both', 'dump']:
                    dump_result = await self.detect_dump_pattern(market_data, tech_indicators)
                    if dump_result['score'] > 0.4:
                        results['dump'].append({
                            'symbol': symbol,
                            'score': dump_result['score'],
                            'signals': dump_result['signals'],
                            'confidence': dump_result['confidence'],
                            'price': market_data['price'],
                            'volume': market_data['volume'],
                            'quote_volume': market_data['quote_volume'],
                            'price_change_24h': market_data['price_change']
                        })
                        logger.info(f"Знайдено dump сигнал для {symbol}: {dump_result['score']:.2%}")
            
            # Сортування результатів
            for key in results:
                results[key].sort(key=lambda x: x['score'], reverse=True)
            
            logger.info(f"Сканування завершено. Pump: {len(results['pump'])}, Dump: {len(results['dump'])}")
            return results
            
        except Exception as e:
            logger.error(f"Помилка сканування: {e}")
            return {'pump': [], 'dump': []}

    async def whale_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда для перегляду whale alert"""
        await update.message.reply_text(
            "🐋 Whale Alert System\n\n"
            "Поточні налаштування:\n"
            "• Мінімальний ордер: $50,000\n"
            "• Моніторинг: Топ 20 монет\n"
            "• Сповіщення: Real-time\n\n"
            "Для налаштування використовуйте /settings"
        )

    async def top_signals_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда для перегляду топ сигналів"""
        if not self.last_signals:
            await update.message.reply_text("📊 Ще немає збережених сигналів. Запустіть сканування /scan")
            return
        
        message = "📈 ТОП останніх сигналів:\n\n"
        
        # Групуємо сигнали по типу
        pump_signals = [s for s in self.last_signals.values() if s['type'] == 'pump']
        dump_signals = [s for s in self.last_signals.values() if s['type'] == 'dump']
        
        if pump_signals:
            message += "🚀 PUMP сигнали:\n"
            for i, signal in enumerate(sorted(pump_signals, key=lambda x: x['data']['score'], reverse=True)[:5], 1):
                message += f"{i}. {signal['data']['symbol']} - {signal['data']['score']:.2%} ({signal['time'][11:16]})\n"
            message += "\n"
        
        if dump_signals:
            message += "📉 DUMP сигнали:\n"
            for i, signal in enumerate(sorted(dump_signals, key=lambda x: x['data']['score'], reverse=True)[:5], 1):
                message += f"{i}. {signal['data']['symbol']} - {signal['data']['score']:.2%} ({signal['time'][11:16]})\n"
        
        await update.message.reply_text(message)

    async def button_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()
        
        if query.data == "scan_now":
            await query.edit_message_text("🔍 Сканую топ монети...")
            results = await self.scan_top_coins('both')
            
            message = "📊 Результати сканування:\n\n"
            
            if not results['pump'] and not results['dump']:
                message += "ℹ️ Сигналів не знайдено\n"
                message += "Спробуйте команду /debug для перевірки роботи"
            else:
                for signal_type in ['pump', 'dump']:
                    if results[signal_type]:
                        message += f"{'🚀' if signal_type == 'pump' else '📉'} {signal_type.upper()}:\n"
                        for i, signal in enumerate(results[signal_type][:3], 1):
                            message += f"{i}. {signal['symbol']} - {signal['score']:.2%} (об'єм: {signal['quote_volume']:,.0f} USDT)\n"
                        message += "\n"
                
            await query.edit_message_text(message)
            
        elif query.data == "settings":
            settings_msg = self.get_settings_message()
            await query.edit_message_text(settings_msg)
            
        elif query.data == "stats":
            stats_msg = "📊 Статистика бота:\n\n"
            stats_msg += f"Останні сигнали: {len(self.last_signals)}\n"
            stats_msg += f"Чорний список: {len(self.coin_blacklist)} монет\n"
            stats_msg += f"Популярні монети: {len(self.popular_coins)}\n"
            stats_msg += f"Заблоковано сміття: {len(self.garbage_symbols)} символів\n\n"
            stats_msg += "⚙️ Поточні пороги:\n"
            stats_msg += f"• Мінімальний об'єм: {self.pump_thresholds['min_volume']:,.0f} USDT\n"
            stats_msg += f"• Volume ratio: {self.pump_thresholds['volume_ratio']}x\n"
            stats_msg += f"• Price change: {self.pump_thresholds['price_change_5m']}%"
            
            await query.edit_message_text(stats_msg)
            
        elif query.data == "blacklist":
            if self.coin_blacklist:
                blacklist_msg = "🚫 Чорний список монет:\n"
                blacklist_msg += "\n".join(f"• {coin}" for coin in sorted(self.coin_blacklist))
                blacklist_msg += "\n\nВикористовуйте /blacklist [монета] для додавання/видалення"
            else:
                blacklist_msg = "✅ Чорний список порожній\n\nВикористовуйте /blacklist [монета] для додавання"
            
            await query.edit_message_text(blacklist_msg)
            
        elif query.data == "whale_alert":
            whale_msg = "🐋 Whale Alert System\n\n"
            whale_msg += "Функціональність:\n"
            whale_msg += "• Визначення великих ордерів (>$50K)\n"
            whale_msg += "• Моніторинг стакану топ-монет\n"
            whale_msg += "• Сповіщення про аномальну активність\n\n"
            whale_msg += "Для перегляду деталей використовуйте /whale"
            
            await query.edit_message_text(whale_msg)
            
        elif query.data == "top_signals":
            if not self.last_signals:
                await query.edit_message_text("📊 Ще немає збережених сигналів. Запустіть сканування /scan")
                return
            
            message = "📈 ТОП останніх сигналів:\n\n"
            
            # Групуємо сигнали по типу
            pump_signals = [s for s in self.last_signals.values() if s['type'] == 'pump']
            dump_signals = [s for s in self.last_signals.values() if s['type'] == 'dump']
            
            if pump_signals:
                message += "🚀 ТОП 3 PUMP:\n"
                for i, signal in enumerate(sorted(pump_signals, key=lambda x: x['data']['score'], reverse=True)[:3], 1):
                    message += f"{i}. {signal['data']['symbol']} - {signal['data']['score']:.2%}\n"
                message += "\n"
            
            if dump_signals:
                message += "📉 ТОП 3 DUMP:\n"
                for i, signal in enumerate(sorted(dump_signals, key=lambda x: x['data']['score'], reverse=True)[:3], 1):
                    message += f"{i}. {signal['data']['symbol']} - {signal['data']['score']:.2%}\n"
            
            await query.edit_message_text(message)

    # Решта методів залишаються незмінними
    # ... [тут всі інші функції] ...

    def run(self):
        """Запуск бота"""
        # Запуск Flask в окремому потоці
        flask_thread = threading.Thread(target=self.run_flask, daemon=True)
        flask_thread.start()
        
        print("🤖 Бот запущений...")
        print("🌐 Flask сервер запущений")
        print("📊 Фільтрація сміття активована")
        print("💬 Доступні команди: /start, /scan, /settings, /blacklist, /debug, /whale, /topsignals")
        
        # Запуск Telegram бота
        self.app.run_polling()

# Використання
if __name__ == "__main__":
    TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not TOKEN:
        print("❌ Помилка: TELEGRAM_BOT_TOKEN не встановлено в змінних оточення")
        print("ℹ️ Додайте TELEGRAM_BOT_TOKEN до налаштувань Render")
        exit(1)
    
    bot = AdvancedPumpDumpBot(TOKEN)
    bot.run()