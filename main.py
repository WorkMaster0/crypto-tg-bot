import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
import asyncio
from flask import Flask, request, jsonify
import threading
import talib
import json
from typing import Dict, List, Optional
import logging

# Налаштування логування
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedPumpDumpBot:
    def __init__(self, token: str):
        self.token = token
        self.app = Application.builder().token(token).build()
        self.flask_app = Flask(__name__)
        self.setup_flask_routes()
        
        # Пороги для пампу
        self.pump_thresholds = {
            'volume_ratio': 5.0,
            'price_change_1h': 8.0,
            'price_change_5m': 3.0,
            'market_cap_min': 500000,
            'liquidity_min': 50000,
            'rsi_threshold': 65,
            'buy_pressure_ratio': 1.5
        }
        
        # Пороги для дампу
        self.dump_thresholds = {
            'volume_ratio': 4.0,
            'price_change_1h': -6.0,
            'price_change_5m': -2.5,
            'rsi_threshold': 35,
            'sell_pressure_ratio': 1.8,
            'support_break': True
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
    
    def setup_handlers(self):
        self.app.add_handler(CommandHandler("start", self.start))
        self.app.add_handler(CommandHandler("scan", self.scan_command))
        self.app.add_handler(CommandHandler("settings", self.settings_command))
        self.app.add_handler(CommandHandler("blacklist", self.blacklist_command))
        self.app.add_handler(CallbackQueryHandler(self.button_handler))
    
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
            # Основні дані
            url = f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}USDT"
            response = requests.get(url, timeout=10)
            data = response.json()
            
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
            orderbook_url = f"https://api.binance.com/api/v3/depth?symbol={symbol}USDT&limit=20"
            orderbook_response = requests.get(orderbook_url, timeout=10)
            orderbook_data = orderbook_response.json()
            
            return {
                'symbol': symbol,
                'price': float(data['lastPrice']),
                'volume': float(data['volume']),
                'price_change': float(data['priceChangePercent']),
                'high': float(data['highPrice']),
                'low': float(data['lowPrice']),
                'quote_volume': float(data['quoteVolume']),
                'klines': klines_data,
                'orderbook': orderbook_data
            }
        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {e}")
            return None

    async def calculate_technical_indicators(self, klines_data: List) -> Dict:
        closes = np.array([float(x[4]) for x in klines_data])
        highs = np.array([float(x[2]) for x in klines_data])
        lows = np.array([float(x[3]) for x in klines_data])
        volumes = np.array([float(x[5]) for x in klines_data])
        
        # RSI
        rsi = talib.RSI(closes, timeperiod=14)[-1] if len(closes) >= 14 else 50
        
        # MACD
        macd, macd_signal, macd_hist = talib.MACD(closes)
        macd_value = macd[-1] if len(macd) > 0 else 0
        
        # Bollinger Bands
        upper_bb, middle_bb, lower_bb = talib.BBANDS(closes, timeperiod=20)
        bb_position = (closes[-1] - lower_bb[-1]) / (upper_bb[-1] - lower_bb[-1]) if upper_bb[-1] != lower_bb[-1] else 0.5
        
        # Volume SMA
        volume_sma = talib.SMA(volumes, timeperiod=20)[-1] if len(volumes) >= 20 else np.mean(volumes)
        
        return {
            'rsi': rsi,
            'macd': macd_value,
            'bb_position': bb_position,
            'volume_ratio': volumes[-1] / volume_sma if volume_sma > 0 else 1,
            'current_price': closes[-1],
            'price_5m_ago': closes[-6] if len(closes) >= 6 else closes[0]
        }

    async def detect_pump_pattern(self, market_data: Dict, tech_indicators: Dict) -> Dict:
        score = 0
        signals = []
        
        # Volume analysis
        volume_ratio = tech_indicators['volume_ratio']
        if volume_ratio > self.pump_thresholds['volume_ratio']:
            score += 0.3
            signals.append(f"Volume x{volume_ratio:.1f}")
        
        # Price momentum
        price_change_5m = ((tech_indicators['current_price'] - tech_indicators['price_5m_ago']) / 
                          tech_indicators['price_5m_ago']) * 100
        if price_change_5m > self.pump_thresholds['price_change_5m']:
            score += 0.2
            signals.append(f"+{price_change_5m:.1f}% 5m")
        
        # RSI condition
        if tech_indicators['rsi'] > self.pump_thresholds['rsi_threshold']:
            score += 0.1
            signals.append(f"RSI {tech_indicators['rsi']:.1f}")
        
        # Order book analysis
        ob_analysis = self.analyze_orderbook(market_data['orderbook'])
        if ob_analysis['buy_pressure'] > self.pump_thresholds['buy_pressure_ratio']:
            score += 0.2
            signals.append(f"Buy pressure {ob_analysis['buy_pressure']:.1f}")
        
        # Whale detection
        whale_volume = self.detect_whale_orders(market_data['orderbook'])
        if whale_volume > market_data['quote_volume'] * 0.01:
            score += 0.2
            signals.append(f"Whale order: ${whale_volume:,.0f}")
        
        return {'score': min(score, 1.0), 'signals': signals, 'confidence': 'high' if score > 0.6 else 'medium'}

    async def detect_dump_pattern(self, market_data: Dict, tech_indicators: Dict) -> Dict:
        score = 0
        signals = []
        
        # Volume analysis (selling volume)
        volume_ratio = tech_indicators['volume_ratio']
        if volume_ratio > self.dump_thresholds['volume_ratio']:
            score += 0.3
            signals.append(f"Sell volume x{volume_ratio:.1f}")
        
        # Price decline
        price_change_5m = ((tech_indicators['current_price'] - tech_indicators['price_5m_ago']) / 
                          tech_indicators['price_5m_ago']) * 100
        if price_change_5m < self.dump_thresholds['price_change_5m']:
            score += 0.2
            signals.append(f"{price_change_5m:.1f}% 5m")
        
        # RSI condition (oversold)
        if tech_indicators['rsi'] < self.dump_thresholds['rsi_threshold']:
            score += 0.1
            signals.append(f"RSI {tech_indicators['rsi']:.1f}")
        
        # Order book analysis (sell pressure)
        ob_analysis = self.analyze_orderbook(market_data['orderbook'])
        if ob_analysis['sell_pressure'] > self.dump_thresholds['sell_pressure_ratio']:
            score += 0.2
            signals.append(f"Sell pressure {ob_analysis['sell_pressure']:.1f}")
        
        # Support break detection
        if self.check_support_break(tech_indicators, market_data['klines']['15m']):
            score += 0.2
            signals.append("Support broken")
        
        return {'score': min(score, 1.0), 'signals': signals, 'confidence': 'high' if score > 0.6 else 'medium'}

    def analyze_orderbook(self, orderbook: Dict) -> Dict:
        bids = np.array([float(bid[1]) for bid in orderbook['bids']])
        asks = np.array([float(ask[1]) for ask in orderbook['asks']])
        
        total_bids = np.sum(bids)
        total_asks = np.sum(asks)
        
        buy_pressure = total_bids / total_asks if total_asks > 0 else 1
        sell_pressure = total_asks / total_bids if total_bids > 0 else 1
        
        return {
            'buy_pressure': buy_pressure,
            'sell_pressure': sell_pressure,
            'imbalance': abs(total_bids - total_asks) / (total_bids + total_asks)
        }

    def detect_whale_orders(self, orderbook: Dict) -> float:
        """Detect large individual orders"""
        large_orders = []
        for side in ['bids', 'asks']:
            for order in orderbook[side]:
                order_size = float(order[1]) * float(order[0])
                if order_size > 50000:  # $50k+ considered whale order
                    large_orders.append(order_size)
        return sum(large_orders) if large_orders else 0

    def check_support_break(self, tech_indicators: Dict, klines: List) -> bool:
        """Check if price broke through support level"""
        closes = np.array([float(x[4]) for x in klines])
        if len(closes) < 20:
            return False
        
        # Simple support as recent low
        support_level = np.min(closes[-10:])
        current_price = tech_indicators['current_price']
        
        return current_price < support_level * 0.99  # 1% below support

    async def scan_top_coins(self, scan_type: str = 'both'):
        try:
            response = requests.get("https://api.binance.com/api/v3/ticker/24hr", timeout=10)
            all_data = response.json()
            
            usdt_pairs = [x for x in all_data if x['symbol'].endswith('USDT')]
            sorted_by_volume = sorted(usdt_pairs, 
                                    key=lambda x: float(x['volume']), 
                                    reverse=True)[:100]  # Топ 100 по об'єму
            
            results = {'pump': [], 'dump': []}
            
            for coin in sorted_by_volume:
                symbol = coin['symbol'].replace('USDT', '')
                if symbol in self.coin_blacklist:
                    continue
                    
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
                    if pump_result['score'] > 0.5:
                        results['pump'].append({
                            'symbol': symbol,
                            'score': pump_result['score'],
                            'signals': pump_result['signals'],
                            'confidence': pump_result['confidence'],
                            'price': market_data['price'],
                            'volume': market_data['volume']
                        })
                
                # Детекція дампу
                if scan_type in ['both', 'dump']:
                    dump_result = await self.detect_dump_pattern(market_data, tech_indicators)
                    if dump_result['score'] > 0.5:
                        results['dump'].append({
                            'symbol': symbol,
                            'score': dump_result['score'],
                            'signals': dump_result['signals'],
                            'confidence': dump_result['confidence'],
                            'price': market_data['price'],
                            'volume': market_data['volume']
                        })
            
            # Сортування результатів
            for key in results:
                results[key].sort(key=lambda x: x['score'], reverse=True)
            
            return results
            
        except Exception as e:
            logger.error(f"Scan error: {e}")
            return {'pump': [], 'dump': []}

    async def send_alert(self, context: ContextTypes.DEFAULT_TYPE, signal_data: Dict, signal_type: str):
        symbol = signal_data['symbol']
        
        if signal_type == 'pump':
            message = f"🚀 ПОТЕНЦІЙНИЙ ПАМП\n\n"
            emoji = "🚀"
        else:
            message = f"📉 ПОТЕНЦІЙНИЙ ДАМП\n\n"
            emoji = "📉"
        
        message += f"{emoji} Монета: {symbol}\n"
        message += f"💰 Ціна: ${signal_data['price']:.6f}\n"
        message += f"📊 Впевненість: {signal_data['confidence']}\n"
        message += f"⚡ Score: {signal_data['score']:.2%}\n\n"
        message += "📶 Сигнали:\n"
        
        for signal in signal_data['signals'][:5]:
            message += f"• {signal}\n"
        
        message += f"\n⏰ {datetime.now().strftime('%H:%M:%S')}"
        
        # Збереження останнього сигналу
        self.last_signals[symbol] = {
            'type': signal_type,
            'time': datetime.now().isoformat(),
            'data': signal_data
        }
        
        await context.bot.send_message(
            chat_id=context.job.chat_id,
            text=message,
            parse_mode='HTML'
        )

    async def scan_job(self, context: ContextTypes.DEFAULT_TYPE):
        results = await self.scan_top_coins('both')
        
        # Відправка топ-3 сигналів для кожного типу
        for signal_type in ['pump', 'dump']:
            for signal in results[signal_type][:3]:
                await self.send_alert(context, signal, signal_type)
                
                # Cooldown для монети
                await asyncio.sleep(1)

    def handle_webhook(self, data: Dict) -> str:
        """Обробка вебхук запитів"""
        try:
            if data.get('type') == 'manual_scan':
                # Ручне сканування через вебхук
                results = asyncio.run(self.scan_top_coins(data.get('scan_type', 'both')))
                return jsonify(results)
                
            elif data.get('type') == 'update_settings':
                self.update_settings(data.get('settings', {}))
                return jsonify({'status': 'success'})
                
            return jsonify({'error': 'Unknown webhook type'})
            
        except Exception as e:
            return jsonify({'error': str(e)})

    def update_settings(self, new_settings: Dict):
        """Оновлення налаштувань через вебхук"""
        if 'pump' in new_settings:
            self.pump_thresholds.update(new_settings['pump'])
        if 'dump' in new_settings:
            self.dump_thresholds.update(new_settings['dump'])

    async def button_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()
        
        if query.data == "scan_now":
            await query.edit_message_text("🔍 Сканую топ монети...")
            results = await self.scan_top_coins('both')
            
            message = "📊 Результати сканування:\n\n"
            for signal_type in ['pump', 'dump']:
                message += f"{'🚀' if signal_type == 'pump' else '📉'} {signal_type.upper()}:\n"
                for i, signal in enumerate(results[signal_type][:3], 1):
                    message += f"{i}. {signal['symbol']} - {signal['score']:.2%}\n"
                message += "\n"
                
            await query.edit_message_text(message)
            
        elif query.data == "settings":
            settings_msg = self.get_settings_message()
            await query.edit_message_text(settings_msg)

    def get_settings_message(self) -> str:
        msg = "⚙️ Поточні налаштування:\n\n"
        msg += "🚀 Pump Detection:\n"
        for k, v in self.pump_thresholds.items():
            msg += f"  {k}: {v}\n"
        
        msg += "\n📉 Dump Detection:\n"
        for k, v in self.dump_thresholds.items():
            msg += f"  {k}: {v}\n"
        
        return msg

    def run_flask(self):
        """Запуск Flask сервера"""
        self.flask_app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

    def run(self):
        """Запуск бота"""
        # Запуск Flask в окремому потоці
        flask_thread = threading.Thread(target=self.run_flask, daemon=True)
        flask_thread.start()
        
        print("🤖 Бот запущений...")
        print("🌐 Flask сервер запущений на порті 5000")
        print("📊 Доступні endpoints: /webhook, /stats, /update_settings")
        
        self.app.run_polling()

# Використання
if __name__ == "__main__":
    TOKEN = "8489382938:AAHeFFZPODspuEFcSQyjw8lWzYpRRSv9n3g"
    bot = AdvancedPumpDumpBot(TOKEN)
    bot.run()