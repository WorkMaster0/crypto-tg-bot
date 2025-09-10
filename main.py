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
            'buy_pressure_ratio': 1.5,
            'macd_signal': 0.001
        }
        
        # Пороги для дампу
        self.dump_thresholds = {
            'volume_ratio': 4.0,
            'price_change_1h': -6.0,
            'price_change_5m': -2.5,
            'rsi_threshold': 35,
            'sell_pressure_ratio': 1.8,
            'support_break': True,
            'macd_signal': -0.001
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

    def calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Власна реалізація RSI"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi

    def calculate_ema(self, data: np.ndarray, period: int) -> float:
        """Експоненційна ковзна середня"""
        if len(data) < period:
            return np.mean(data) if len(data) > 0 else 0
        
        weights = np.exp(np.linspace(-1., 0., period))
        weights /= weights.sum()
        
        # Використовуємо згортку для EMA
        ema = np.convolve(data, weights, mode='valid')
        return ema[-1] if len(ema) > 0 else np.mean(data)

    def calculate_macd(self, prices: np.ndarray, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> float:
        """Власна реалізація MACD"""
        if len(prices) < slow_period + signal_period:
            return 0.0
        
        fast_ema = self.calculate_ema(prices, fast_period)
        slow_ema = self.calculate_ema(prices, slow_period)
        macd_line = fast_ema - slow_ema
        
        return macd_line

    def calculate_bollinger_bands(self, prices: np.ndarray, period: int = 20, std_dev: int = 2) -> Tuple[float, float, float]:
        """Власна реалізація Bollinger Bands"""
        if len(prices) < period:
            return prices[-1], prices[-1], prices[-1]
        
        sma = np.mean(prices[-period:])
        std = np.std(prices[-period:])
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return upper_band, sma, lower_band

    def calculate_sma(self, data: np.ndarray, period: int) -> float:
        """Проста ковзна середня"""
        if len(data) < period:
            return np.mean(data) if len(data) > 0 else 0
        return np.mean(data[-period:])

    async def calculate_technical_indicators(self, klines_data: List) -> Dict:
        closes = np.array([float(x[4]) for x in klines_data])
        highs = np.array([float(x[2]) for x in klines_data])
        lows = np.array([float(x[3]) for x in klines_data])
        volumes = np.array([float(x[5]) for x in klines_data])
        
        # RSI
        rsi = self.calculate_rsi(closes)
        
        # MACD
        macd = self.calculate_macd(closes)
        
        # Bollinger Bands
        upper_bb, middle_bb, lower_bb = self.calculate_bollinger_bands(closes)
        bb_position = (closes[-1] - lower_bb) / (upper_bb - lower_bb) if upper_bb != lower_bb else 0.5
        
        # Volume SMA
        volume_sma = self.calculate_sma(volumes, 20)
        volume_ratio = volumes[-1] / volume_sma if volume_sma > 0 else 1
        
        # Price changes
        price_change_1h = ((closes[-1] - closes[-12]) / closes[-12]) * 100 if len(closes) >= 12 else 0
        price_change_5m = ((closes[-1] - closes[-2]) / closes[-2]) * 100 if len(closes) >= 2 else 0
        
        return {
            'rsi': rsi,
            'macd': macd,
            'bb_position': bb_position,
            'volume_ratio': volume_ratio,
            'current_price': closes[-1],
            'price_5m_ago': closes[-2] if len(closes) >= 2 else closes[0],
            'price_1h_ago': closes[-12] if len(closes) >= 12 else closes[0],
            'price_change_1h': price_change_1h,
            'price_change_5m': price_change_5m
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
        if tech_indicators['price_change_5m'] > self.pump_thresholds['price_change_5m']:
            score += 0.2
            signals.append(f"+{tech_indicators['price_change_5m']:.1f}% 5m")
        
        # RSI condition
        if tech_indicators['rsi'] > self.pump_thresholds['rsi_threshold']:
            score += 0.1
            signals.append(f"RSI {tech_indicators['rsi']:.1f}")
        
        # MACD condition
        if tech_indicators['macd'] > self.pump_thresholds['macd_signal']:
            score += 0.1
            signals.append(f"MACD {tech_indicators['macd']:.4f}")
        
        # Order book analysis
        ob_analysis = self.analyze_orderbook(market_data['orderbook'])
        if ob_analysis['buy_pressure'] > self.pump_thresholds['buy_pressure_ratio']:
            score += 0.2
            signals.append(f"Buy pressure {ob_analysis['buy_pressure']:.1f}")
        
        # Whale detection
        whale_volume = self.detect_whale_orders(market_data['orderbook'])
        if whale_volume > 50000:  # $50k+ whale order
            score += 0.1
            signals.append(f"Whale: ${whale_volume:,.0f}")
        
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
        if tech_indicators['price_change_5m'] < self.dump_thresholds['price_change_5m']:
            score += 0.2
            signals.append(f"{tech_indicators['price_change_5m']:.1f}% 5m")
        
        # RSI condition (oversold)
        if tech_indicators['rsi'] < self.dump_thresholds['rsi_threshold']:
            score += 0.1
            signals.append(f"RSI {tech_indicators['rsi']:.1f}")
        
        # MACD condition
        if tech_indicators['macd'] < self.dump_thresholds['macd_signal']:
            score += 0.1
            signals.append(f"MACD {tech_indicators['macd']:.4f}")
        
        # Order book analysis (sell pressure)
        ob_analysis = self.analyze_orderbook(market_data['orderbook'])
        if ob_analysis['sell_pressure'] > self.dump_thresholds['sell_pressure_ratio']:
            score += 0.2
            signals.append(f"Sell pressure {ob_analysis['sell_pressure']:.1f}")
        
        # Support break detection
        if self.check_support_break(tech_indicators, market_data['klines']['15m']):
            score += 0.1
            signals.append("Support broken")
        
        return {'score': min(score, 1.0), 'signals': signals, 'confidence': 'high' if score > 0.6 else 'medium'}

    def analyze_orderbook(self, orderbook: Dict) -> Dict:
        bids = np.array([float(bid[1]) for bid in orderbook['bids'][:10]])  # Топ 10 bid
        asks = np.array([float(ask[1]) for ask in orderbook['asks'][:10]])  # Топ 10 ask
        
        total_bids = np.sum(bids)
        total_asks = np.sum(asks)
        
        buy_pressure = total_bids / total_asks if total_asks > 0 else 1
        sell_pressure = total_asks / total_bids if total_bids > 0 else 1
        
        return {
            'buy_pressure': buy_pressure,
            'sell_pressure': sell_pressure,
            'imbalance': abs(total_bids - total_asks) / (total_bids + total_asks) if (total_bids + total_asks) > 0 else 0
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
        if len(klines) < 20:
            return False
        
        closes = np.array([float(x[4]) for x in klines])
        
        # Support as lowest price in last 10 periods
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
                                    reverse=True)[:50]  # Топ 50 по об'єму
            
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
                            'volume': market_data['volume'],
                            'price_change_24h': market_data['price_change']
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
                            'volume': market_data['volume'],
                            'price_change_24h': market_data['price_change']
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
        message += f"📈 24h change: {signal_data['price_change_24h']:.1f}%\n"
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

    async def scan_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда для ручного сканування"""
        await update.message.reply_text("🔍 Запускаю сканування...")
        results = await self.scan_top_coins('both')
        
        message = "📊 Результати сканування:\n\n"
        for signal_type in ['pump', 'dump']:
            message += f"{'🚀' if signal_type == 'pump' else '📉'} {signal_type.upper()}:\n"
            for i, signal in enumerate(results[signal_type][:3], 1):
                message += f"{i}. {signal['symbol']} - {signal['score']:.2%}\n"
            message += "\n"
            
        await update.message.reply_text(message)

    async def settings_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда для перегляду налаштувань"""
        settings_msg = self.get_settings_message()
        await update.message.reply_text(settings_msg)

    async def blacklist_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда для керування чорним списком"""
        if context.args:
            coin = context.args[0].upper()
            if coin in self.coin_blacklist:
                self.coin_blacklist.remove(coin)
                await update.message.reply_text(f"✅ {coin} видалено з чорного списку")
            else:
                self.coin_blacklist.add(coin)
                await update.message.reply_text(f"✅ {coin} додано до чорного списку")
        else:
            blacklist_msg = "🚫 Чорний список:\n" + "\n".join(self.coin_blacklist) if self.coin_blacklist else "Чорний список порожній"
            await update.message.reply_text(blacklist_msg)

    async def manual_scan_job(self):
        """Ручне сканування без JobQueue"""
        while True:
            try:
                # Тут можна додати логіку для періодичного сканування
                await asyncio.sleep(300)  # Спати 5 хвилин
            except Exception as e:
                logger.error(f"Manual scan error: {e}")
                await asyncio.sleep(60)

    def handle_webhook(self, data: Dict) -> str:
        """Обробка вебхук запитів"""
        try:
            if data.get('type') == 'manual_scan':
                results = asyncio.run(self.scan_top_coins(data.get('scan_type', 'both')))
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
        port = int(os.environ.get('PORT', 5000))
        self.flask_app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)

    def run(self):
        """Запуск бота"""
        # Запуск Flask в окремому потоці
        flask_thread = threading.Thread(target=self.run_flask, daemon=True)
        flask_thread.start()
        
        print("🤖 Бот запущений...")
        print("🌐 Flask сервер запущений")
        print("📊 Доступні endpoints: /webhook, /stats, /update_settings, /health")
        
        # Запуск ручного сканування в окремому потоці
        scan_thread = threading.Thread(target=lambda: asyncio.run(self.manual_scan_job()), daemon=True)
        scan_thread.start()
        
        # Запуск Telegram бота
        self.app.run_polling()

# Використання
if __name__ == "__main__":
    TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "8489382938:AAHeFFZPODspuEFcSQyjw8lWzYpRRSv9n3g")
    bot = AdvancedPumpDumpBot(TOKEN)
    bot.run()