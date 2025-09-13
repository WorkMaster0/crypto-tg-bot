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
import aiohttp
import warnings
warnings.filterwarnings('ignore')

# Налаштування логування
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot_debug.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class UltimatePumpDumpDetector:
    def __init__(self, token: str):
        self.token = token
        self.app = Application.builder().token(token).build()
        
        # Покращене підключення до бірж через CCXT
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'timeout': 30000,
            'options': {
                'defaultType': 'spot',
                'adjustForTimeDifference': True,
            }
        })
        
        # Розширений чорний список
        self.garbage_symbols = self._load_garbage_symbols()
        self.coin_blacklist = set()
        
        # Динамічні параметри для виявлення (знижені пороги для тесту)
        self.detection_params = {
            'volume_spike_threshold': 2.5,
            'price_acceleration_min': 0.004,
            'rsi_oversold': 35,
            'rsi_overbought': 75,
            'orderbook_imbalance_min': 0.18,
            'large_order_threshold': 25000,
            'min_volume_usdt': 50000,
            'max_volume_usdt': 2000000,
            'price_change_5m_min': 2.0,
            'wick_ratio_threshold': 0.35,
            'market_cap_filter': 50000000,
            'liquidity_score_min': 0.5,
            'pump_probability_threshold': 0.55,
            'dump_probability_threshold': 0.50,
            'whale_volume_threshold': 50000,
            'volatility_spike_threshold': 2.0
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
        self.executor = ThreadPoolExecutor(max_workers=8)
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
        """Оновлені обробники команд з debug"""
        handlers = [
            CommandHandler("start", self.start_command),
            CommandHandler("scan", self.deep_scan_command),
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
             InlineKeyboardButton("🤖 AI RISK SCAN", callback_data="ai_risk_scan")],
            [InlineKeyboardButton("🔍 DEEP SCAN", callback_data="deep_scan"),
             InlineKeyboardButton("⚡ QUICK SCAN", callback_data="quick_scan")],
            [InlineKeyboardButton("⚙️ SETTINGS", callback_data="settings"),
             InlineKeyboardButton("📈 PERFORMANCE", callback_data="performance")],
            [InlineKeyboardButton("🚫 BLACKLIST", callback_data="blacklist"),
             InlineKeyboardButton("🔄 UPDATE", callback_data="update")]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            "🤖 **ULTIMATE PUMP/DUMP DETECTOR v4.1**\n\n"
            "🎯 *Спеціалізація: реальне виявлення маніпуляцій ринком*\n\n"
            "✨ **Ексклюзивні фічі:**\n"
            "• 🚨 Детектор пампів в реальному часі\n"
            "• 📉 Виявлення дампів до їх початку\n"
            "• 🐋 Відстеження китів та великих ордерів\n"
            "• 💧 Аналіз ліквідності та кластерів\n"
            "• ⚡ Сигнали волатильності\n"
            "• 🤖 AI аналіз ризиків\n\n"
            "💎 *Отримуй сигнали ДО руху ринку!*",
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )

    async def check_network_connection(self) -> bool:
        """Перевірка мережевого з'єднання"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get('https://api.binance.com/api/v3/ping', timeout=10) as response:
                    return response.status == 200
        except:
            return False

    async def get_market_data(self, symbol: str) -> Optional[Dict]:
        """Покращене отримання ринкових даних"""
        try:
            for attempt in range(3):
                try:
                    ticker = await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(
                            self.executor, 
                            lambda: self.exchange.fetch_ticker(symbol)
                        ),
                        timeout=10.0
                    )
                    return self.parse_ticker_data(ticker, symbol)
                except (asyncio.TimeoutError, ccxt.NetworkError):
                    if attempt == 2:
                        raise
                    await asyncio.sleep(1)
                    
        except Exception as e:
            logger.error(f"Помилка отримання даних для {symbol}: {e}")
            return await self.get_market_data_fallback(symbol)

    def parse_ticker_data(self, ticker: Dict, symbol: str) -> Dict:
        """Парсинг даних ticker"""
        try:
            return {
                'symbol': symbol,
                'open': float(ticker.get('open', 0)),
                'high': float(ticker.get('high', 0)),
                'low': float(ticker.get('low', 0)),
                'close': float(ticker.get('last', ticker.get('close', 0))),
                'volume': float(ticker.get('quoteVolume', 0)),
                'percentage': float(ticker.get('percentage', 0))
            }
        except:
            return {
                'symbol': symbol,
                'open': 0,
                'high': 0,
                'low': 0,
                'close': 0,
                'volume': 0,
                'percentage': 0
            }

    async def get_market_data_fallback(self, symbol: str) -> Optional[Dict]:
        """Альтернативний спосіб отримання даних"""
        try:
            clean_symbol = symbol.replace('/USDT', '').lower()
            
            url = f"https://api.coingecko.com/api/v3/coins/markets"
            params = {
                'vs_currency': 'usd',
                'ids': clean_symbol,
                'order': 'market_cap_desc',
                'per_page': 1,
                'page': 1,
                'sparkline': 'false'
            }
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/json',
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, headers=headers, timeout=15) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data and len(data) > 0:
                            coin_data = data[0]
                            return {
                                'symbol': symbol,
                                'open': coin_data.get('open', 0),
                                'high': coin_data.get('high', 0),
                                'low': coin_data.get('low', 0),
                                'close': coin_data.get('current_price', 0),
                                'volume': coin_data.get('total_volume', 0),
                                'percentage': coin_data.get('price_change_percentage_24h', 0)
                            }
            
            return None
        except Exception as e:
            logger.error(f"Помилка альтернативного отримання даних для {symbol}: {e}")
            return None

    async def get_orderbook_depth(self, symbol: str, limit: int = 50) -> Optional[Dict]:
        """Отримання глибини ринку з обробкою помилок"""
        try:
            orderbook = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    self.executor, 
                    lambda: self.exchange.fetch_order_book(symbol, limit)
                ),
                timeout=10.0
            )
            orderbook['symbol'] = symbol
            return orderbook
        except Exception as e:
            logger.error(f"Помилка отримання orderbook для {symbol}: {e}")
            return {'bids': [], 'asks': [], 'symbol': symbol}

    async def get_klines(self, symbol: str, timeframe: str = '5m', limit: int = 50) -> List:
        """Отримання історичних даних"""
        try:
            klines = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    self.executor, 
                    lambda: self.exchange.fetch_ohlcv(symbol, timeframe, limit)
                ),
                timeout=15.0
            )
            return klines
        except Exception as e:
            logger.error(f"Помилка отримання klines для {symbol}: {e}")
            return []

    async def get_active_symbols(self, limit: int = 50) -> List[str]:
        """Отримання активних торгових пар з покращеною обробкою помилок"""
        try:
            await asyncio.get_event_loop().run_in_executor(
                self.executor, self.exchange.load_markets
            )
            
            if not hasattr(self.exchange, 'markets') or not self.exchange.markets:
                return self.get_fallback_symbols(limit)
            
            usdt_pairs = [
                symbol for symbol, market in self.exchange.markets.items()
                if symbol.endswith('/USDT') and market.get('active', False)
                and market.get('quoteVolume', 0) > self.detection_params['min_volume_usdt']
            ]
            
            usdt_pairs.sort(key=lambda x: self.exchange.markets[x].get('quoteVolume', 0), reverse=True)
            
            return usdt_pairs[:limit]
            
        except Exception as e:
            logger.error(f"Помилка отримання активних символів: {e}")
            return self.get_fallback_symbols(limit)

    def get_fallback_symbols(self, limit: int) -> List[str]:
        """Резервний список популярних символів"""
        popular_symbols = [
            'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT',
            'ADA/USDT', 'AVAX/USDT', 'DOT/USDT', 'MATIC/USDT', 'LINK/USDT',
            'LTC/USDT', 'DOGE/USDT', 'ATOM/USDT', 'XMR/USDT', 'ETC/USDT',
            'BCH/USDT', 'FIL/USDT', 'NEAR/USDT', 'UNI/USDT', 'ALGO/USDT'
        ]
        return popular_symbols[:limit]

    async def get_high_volume_symbols(self, limit: int = 30) -> List[str]:
        """Отримання монет з високим об'ємом"""
        try:
            symbols = await self.get_active_symbols(limit * 2)
            return symbols[:limit]
        except Exception as e:
            logger.error(f"Помилка отримання high volume symbols: {e}")
            return self.get_fallback_symbols(5)

    async def get_top_gainers(self, limit: int = 20) -> List[str]:
        """Отримання топ gainers"""
        try:
            symbols = await self.get_active_symbols(limit * 2)
            return symbols[:limit]
        except Exception as e:
            logger.error(f"Помилка отримання top gainers: {e}")
            return self.get_fallback_symbols(5)

    def technical_analysis(self, klines: List) -> Dict:
        """Поглиблений технічний аналіз"""
        try:
            if len(klines) < 10:
                return {'rsi': 50, 'macd_hist': 0, 'volatility': 0, 'price_acceleration': 0, 'trend_strength': 0.5}
            
            closes = np.array([float(k[4]) for k in klines])
            highs = np.array([float(k[2]) for k in klines])
            lows = np.array([float(k[3]) for k in klines])
            
            # RSI
            rsi = talib.RSI(closes, timeperiod=14)[-1] if len(closes) >= 14 else 50
            
            # MACD
            macd, macd_signal, _ = talib.MACD(closes)
            macd_hist = (macd - macd_signal)[-1] if len(macd) > 0 else 0
            
            # Волотильність
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
            logger.error(f"Помилка технічного аналізу: {e}")
            return {'rsi': 50, 'macd_hist': 0, 'volatility': 0, 'price_acceleration': 0, 'trend_strength': 0.5}

    def orderbook_analysis(self, orderbook: Dict) -> Dict:
        """Детальний аналіз стану ордерів"""
        try:
            bids = orderbook.get('bids', [])
            asks = orderbook.get('asks', [])
            
            total_bid = sum(float(bid[1]) for bid in bids[:20])  # Перші 20 рівнів
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
            logger.error(f"Помилка аналізу orderbook: {e}")
            return {'imbalance': 0, 'large_bids': 0, 'large_asks': 0, 'total_bid_volume': 0, 'total_ask_volume': 0}

    def volume_analysis(self, klines: List, market_data: Dict) -> Dict:
        """Аналіз об'ємів торгів"""
        try:
            if len(klines) < 10:
                return {
                    'volume_spike_ratio': 1.0,
                    'volume_price_correlation': 0,
                    'current_volume': market_data.get('volume', 0),
                    'average_volume': market_data.get('volume', 0)
                }
            
            volumes = np.array([float(k[5]) for k in klines])
            
            avg_volume = np.mean(volumes[:-5]) if len(volumes) > 5 else volumes[0]
            current_volume = volumes[-1] if len(volumes) > 0 else 0
            volume_spike = current_volume / avg_volume if avg_volume > 0 else 1
            
            return {
                'volume_spike_ratio': round(volume_spike, 2),
                'volume_price_correlation': 0,  # Спрощено для тесту
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

    def calculate_price_change(self, klines: List, minutes: int) -> float:
        """Розрахунок зміни ціни за вказаний період"""
        if len(klines) < minutes + 1:
            return 0.0
        
        try:
            current_price = float(klines[-1][4])
            past_price = float(klines[-minutes-1][4])
            return ((current_price - past_price) / past_price) * 100 if past_price != 0 else 0.0
        except:
            return 0.0

    def calculate_pump_probability(self, tech: Dict, orderbook: Dict, volume: Dict) -> float:
        """Розрахунок ймовірності пампу"""
        try:
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
        except:
            return 0.3

    def calculate_dump_probability(self, tech: Dict, orderbook: Dict, volume: Dict) -> float:
        """Розрахунок ймовірності дампу"""
        try:
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
        except:
            return 0.3

    def calculate_pump_confidence(self, tech: Dict, orderbook: Dict, market_data: Dict) -> float:
        """Розрахунок впевненості в пампі"""
        try:
            confidence = 0
            
            if tech['rsi'] < 45:
                confidence += 25
            elif tech['rsi'] < 55:
                confidence += 15
            
            if orderbook['imbalance'] > 0.2:
                confidence += 30
            elif orderbook['imbalance'] > 0.1:
                confidence += 15
            
            if orderbook['large_bids'] >= 3:
                confidence += 20
            elif orderbook['large_bids'] >= 1:
                confidence += 10
            
            if tech['price_acceleration'] > 0.005:
                confidence += 25
            
            return min(confidence, 100)
        except:
            return 0

    def calculate_dump_confidence(self, tech: Dict, orderbook: Dict, market_data: Dict) -> float:
        """Розрахунок впевненості в дампі"""
        try:
            confidence = 0
            
            if tech['rsi'] > 75:
                confidence += 30
            elif tech['rsi'] > 65:
                confidence += 20
            
            if orderbook['imbalance'] < -0.2:
                confidence += 25
            elif orderbook['imbalance'] < -0.1:
                confidence += 15
            
            if orderbook['large_asks'] >= 3:
                confidence += 25
            elif orderbook['large_asks'] >= 1:
                confidence += 15
            
            if tech['volatility'] > 8:
                confidence += 20
            
            return min(confidence, 100)
        except:
            return 0

    def analyze_large_orders(self, orderbook: Dict) -> List[Dict]:
        """Аналіз великих ордерів"""
        large_orders = []
        threshold = self.detection_params['large_order_threshold']
        
        try:
            for bid in orderbook.get('bids', [])[:10]:
                price, amount = float(bid[0]), float(bid[1])
                order_size = price * amount
                
                if order_size > threshold:
                    large_orders.append({
                        'symbol': orderbook.get('symbol', 'UNKNOWN').replace('/USDT', ''),
                        'order_size': order_size,
                        'is_buy': True,
                        'price': price,
                        'amount': amount,
                        'timestamp': datetime.now().strftime('%H:%M:%S')
                    })
            
            for ask in orderbook.get('asks', [])[:10]:
                price, amount = float(ask[0]), float(ask[1])
                order_size = price * amount
                
                if order_size > threshold:
                    large_orders.append({
                        'symbol': orderbook.get('symbol', 'UNKNOWN').replace('/USDT', ''),
                        'order_size': order_size,
                        'is_buy': False,
                        'price': price,
                        'amount': amount,
                        'timestamp': datetime.now().strftime('%H:%M:%S')
                    })
        except Exception as e:
            logger.error(f"Помилка аналізу великих ордерів: {e}")
        
        return large_orders

    def calculate_liquidity_score(self, orderbook: Dict) -> float:
        """Розрахунок оцінки ліквідності"""
        try:
            bids = orderbook.get('bids', [])
            asks = orderbook.get('asks', [])
            
            if not bids or not asks:
                return 0.5
                
            total_bid = sum(float(bid[1]) for bid in bids[:10])
            total_ask = sum(float(ask[1]) for ask in asks[:10])
            total_volume = total_bid + total_ask
            
            if total_volume == 0:
                return 0.5
                
            volume_score = min(total_volume / 500000, 1.0)
            
            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])
            spread = (best_ask - best_bid) / best_bid
            spread_score = 1.0 - min(spread / 0.02, 1.0)
            
            return round((volume_score * 0.6 + spread_score * 0.4), 3)
        except:
            return 0.5

    def calculate_volatility(self, klines: List) -> float:
        """Розрахунок волатильності"""
        try:
            if len(klines) < 10:
                return 0.0
                
            closes = np.array([float(k[4]) for k in klines])
            returns = np.diff(closes) / closes[:-1]
            volatility = np.std(returns) * 100
            return round(volatility, 2)
        except:
            return 0.0

    def calculate_orderbook_imbalance(self, orderbook: Dict) -> float:
        """Розрахунок імбалансу orderbook"""
        try:
            bids = orderbook.get('bids', [])
            asks = orderbook.get('asks', [])
            
            total_bid = sum(float(bid[1]) for bid in bids[:10])
            total_ask = sum(float(ask[1]) for ask in asks[:10])
            
            if total_bid + total_ask == 0:
                return 0.0
            
            return (total_bid - total_ask) / (total_bid + total_ask)
        except:
            return 0.0

    def quick_pump_check(self, market_data: Dict, orderbook: Dict) -> bool:
        """Швидка перевірка на потенційний pump"""
        try:
            if market_data['volume'] < self.detection_params['min_volume_usdt']:
                return False
            
            imbalance = self.calculate_orderbook_imbalance(orderbook)
            if abs(imbalance) < self.detection_params['orderbook_imbalance_min']:
                return False
            
            large_bids = sum(1 for bid in orderbook.get('bids', []) 
                            if float(bid[0]) * float(bid[1]) > self.detection_params['large_order_threshold'])
            
            return large_bids >= 1 and imbalance > 0
        except:
            return False

    def is_garbage_symbol(self, symbol: str) -> bool:
        """Перевірка чи символ є непотрібним"""
        try:
            symbol_clean = symbol.upper().replace('/USDT', '').replace('USDT', '')
            
            if symbol_clean in self.garbage_symbols:
                return True
            
            if len(symbol_clean) > 12:
                return True
            
            if any(char.isdigit() for char in symbol_clean):
                return True
                
            garbage_patterns = ['UP', 'DOWN', 'BULL', 'BEAR', 'USD', 'EUR', 'BTC', 'ETH', 'BNB']
            if any(pattern in symbol_clean for pattern in garbage_patterns):
                return True
                
            return False
        except:
            return False

    def format_signal_message(self, analysis: Dict, index: int) -> str:
        """Форматування повідомлення про сигнал"""
        try:
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
        except:
            return f"{index}. Помилка форматування сигналу\n\n"

    async def analyze_symbol(self, symbol: str) -> Dict:
        """Комплексний аналіз символу"""
        try:
            if not await self.check_network_connection():
                return {}
            
            market_data = await self.get_market_data(symbol)
            if not market_data or market_data['volume'] < self.detection_params['min_volume_usdt']:
                return {}
            
            if self.is_garbage_symbol(symbol):
                return {}
            
            orderbook = await self.get_orderbook_depth(symbol, 30)
            klines = await self.get_klines(symbol, '5m', 30)
            
            if not klines or not orderbook:
                return {}
            
            tech_analysis = self.technical_analysis(klines)
            ob_analysis = self.orderbook_analysis(orderbook)
            volume_analysis = self.volume_analysis(klines, market_data)
            
            pump_prob = self.calculate_pump_probability(tech_analysis, ob_analysis, volume_analysis)
            dump_prob = self.calculate_dump_probability(tech_analysis, ob_analysis, volume_analysis)
            
            return {
                'symbol': symbol,
                'price': market_data['close'],
                'volume_usdt': market_data['volume'],
                'percentage': market_data['percentage'],
                'pump_probability': pump_prob,
                'dump_probability': dump_prob,
                'technical_indicators': tech_analysis,
                'orderbook_metrics': ob_analysis,
                'volume_metrics': volume_analysis
            }
            
        except Exception as e:
            logger.error(f"Помилка аналізу {symbol}: {e}")
            return {}

    async def analyze_pump_potential(self, symbol: str) -> Dict:
        """Спеціалізований аналіз для пампу"""
        try:
            market_data = await self.get_market_data(symbol)
            orderbook = await self.get_orderbook_depth(symbol)
            klines = await self.get_klines(symbol, '5m', 20)
            
            if not all([market_data, orderbook, klines]):
                return {}
            
            tech = self.technical_analysis(klines)
            ob_analysis = self.orderbook_analysis(orderbook)
            
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
            return {}

    async def analyze_dump_potential(self, symbol: str) -> Dict:
        """Спеціалізований аналіз для дампу"""
        try:
            market_data = await self.get_market_data(symbol)
            orderbook = await self.get_orderbook_depth(symbol)
            klines = await self.get_klines(symbol, '5m', 20)
            
            if not all([market_data, orderbook, klines]):
                return {}
            
            tech = self.technical_analysis(klines)
            ob_analysis = self.orderbook_analysis(orderbook)
            
            confidence = self.calculate_dump_confidence(tech, ob_analysis, market_data)
            
            return {
                'symbol': symbol.replace('/USDT', ''),
                'dump_confidence': confidence,
                'max_gain': market_data.get('percentage', 0),
                'whale_sells': ob_analysis['large_asks'],
                'rsi': tech['rsi']
            }
            
        except Exception as e:
            return {}

    async def detect_whale_activity(self) -> List[Dict]:
        """Виявлення активності китів"""
        try:
            symbols = await self.get_high_volume_symbols(limit=8)
            whale_activities = []
            
            for symbol in symbols:
                try:
                    orderbook = await self.get_orderbook_depth(symbol, 20)
                    if orderbook:
                        large_orders = self.analyze_large_orders(orderbook)
                        whale_activities.extend(large_orders)
                    await asyncio.sleep(0.1)
                except:
                    continue
            
            return whale_activities[:8]
        except Exception as e:
            return []

    async def deep_scan_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Глибинне сканування"""
        try:
            if not await self.check_network_connection():
                await update.message.reply_text("⚠️ Проблеми з мережевим з'єднанням")
                return
                
            msg = await update.message.reply_text("🔍 Запускаю глибинне сканування...")
            
            symbols = await self.get_active_symbols(limit=20)
            results = []
            
            for symbol in symbols:
                analysis = await self.analyze_symbol(symbol)
                if analysis and (analysis['pump_probability'] > 0.5 or analysis['dump_probability'] > 0.5):
                    results.append(analysis)
                await asyncio.sleep(0.1)
            
            self.performance_metrics['total_scans'] += len(symbols)
            
            if results:
                results.sort(key=lambda x: max(x['pump_probability'], x['dump_probability']), reverse=True)
                self.performance_metrics['signals_triggered'] += len(results)
                
                response = "🚨 **ЗНАЙДЕНО СИГНАЛИ:**\n\n"
                for i, res in enumerate(results[:5], 1):
                    response += self.format_signal_message(res, i)
                
                response += f"📊 Знайдено {len(results)} сигналів"
                await msg.edit_text(response, parse_mode='Markdown')
            else:
                await msg.edit_text("ℹ️ Сильних сигналів не знайдено")
                
        except Exception as e:
            await update.message.reply_text("❌ Помилка сканування")

    async def pump_radar_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Детектор пампів"""
        try:
            msg = await update.message.reply_text("🚨 АКТИВУЮ PUMP RADAR...")
            
            symbols = await self.get_high_volume_symbols(limit=15)
            pump_candidates = []
            
            for symbol in symbols:
                analysis = await self.analyze_pump_potential(symbol)
                if analysis and analysis['pump_confidence'] > 60:
                    pump_candidates.append(analysis)
                await asyncio.sleep(0.1)
            
            if pump_candidates:
                pump_candidates.sort(key=lambda x: x['pump_confidence'], reverse=True)
                self.performance_metrics['pump_signals_detected'] += len(pump_candidates)
                
                response = "🔥 **РИЗИК PUMP:**\n\n"
                for i, candidate in enumerate(pump_candidates[:3], 1):
                    response += (
                        f"{i}. **{candidate['symbol']}** - {candidate['pump_confidence']}%\n"
                        f"   📈 Зміна: {candidate['price_change_5m']:.2f}%\n"
                        f"   💰 Об'єм: ${candidate['volume_usdt']:,.0f}\n"
                        f"   🐋 Ордери: {candidate['whale_orders']}\n\n"
                    )
                
                await msg.edit_text(response, parse_mode='Markdown')
            else:
                await msg.edit_text("✅ Немає pump-сигналів")
                
        except Exception as e:
            await update.message.reply_text("❌ Помилка pump radar")

    async def dump_radar_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Детектор дампів"""
        try:
            msg = await update.message.reply_text("📉 АКТИВУЮ DUMP RADAR...")
            
            symbols = await self.get_top_gainers(limit=15)
            dump_candidates = []
            
            for symbol in symbols:
                analysis = await self.analyze_dump_potential(symbol)
                if analysis and analysis['dump_confidence'] > 55:
                    dump_candidates.append(analysis)
                await asyncio.sleep(0.1)
            
            if dump_candidates:
                dump_candidates.sort(key=lambda x: x['dump_confidence'], reverse=True)
                self.performance_metrics['dump_signals_detected'] += len(dump_candidates)
                
                response = "⚠️ **РИЗИК DUMP:**\n\n"
                for i, candidate in enumerate(dump_candidates[:3], 1):
                    response += (
                        f"{i}. **{candidate['symbol']}** - {candidate['dump_confidence']}%\n"
                        f"   📉 Зміна: {candidate['max_gain']:.2f}%\n"
                        f"   🐋 Продажі: {candidate['whale_sells']}\n"
                        f"   📍 RSI: {candidate['rsi']:.1f}\n\n"
                    )
                
                await msg.edit_text(response, parse_mode='Markdown')
            else:
                await msg.edit_text("✅ Немає dump-сигналів")
                
        except Exception as e:
            await update.message.reply_text("❌ Помилка dump radar")

    async def whale_watch_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Моніторинг китів"""
        try:
            msg = await update.message.reply_text("🐋 ВІДСТЕЖУЮ КИТІВ...")
            
            whale_activity = await self.detect_whale_activity()
            
            if whale_activity:
                response = "🐋 **АКТИВНІСТЬ КИТІВ:**\n\n"
                for i, activity in enumerate(whale_activity[:5], 1):
                    response += (
                        f"{i}. **{activity['symbol']}**\n"
                        f"   💰 Розмір: ${activity['order_size']:,.0f}\n"
                        f"   📊 Тип: {'КУПІВЛЯ' if activity['is_buy'] else 'ПРОДАЖ'}\n"
                        f"   🕒 Час: {activity['timestamp']}\n\n"
                    )
                
                await msg.edit_text(response, parse_mode='Markdown')
            else:
                await msg.edit_text("✅ Активності китів не виявлено")
                
        except Exception as e:
            await update.message.reply_text("❌ Помилка моніторингу")

    async def liquidity_scan_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Сканування ліквідності"""
        try:
            msg = await update.message.reply_text("💧 АНАЛІЗУЮ ЛІКВІДНІСТЬ...")
            
            symbols = await self.get_active_symbols(limit=15)
            liquidity_data = []
            
            for symbol in symbols:
                orderbook = await self.get_orderbook_depth(symbol)
                if orderbook:
                    liquidity_score = self.calculate_liquidity_score(orderbook)
                    liquidity_data.append({
                        'symbol': symbol.replace('/USDT', ''),
                        'liquidity_score': liquidity_score
                    })
                await asyncio.sleep(0.1)
            
            if liquidity_data:
                liquidity_data.sort(key=lambda x: x['liquidity_score'], reverse=True)
                
                response = "💧 **ТОП ЛІКВІДНІСТЬ:**\n\n"
                for i, data in enumerate(liquidity_data[:5], 1):
                    response += f"{i}. **{data['symbol']}** - Score: {data['liquidity_score']:.3f}\n\n"
                
                await msg.edit_text(response, parse_mode='Markdown')
            else:
                await msg.edit_text("❌ Не вдалося отримати дані")
                
        except Exception as e:
            await update.message.reply_text("❌ Помилка аналізу")

    async def volatility_alert_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Сигнали волатильності"""
        try:
            msg = await update.message.reply_text("⚡ ШУКАЮ ВОЛАТИЛЬНІСТЬ...")
            
            symbols = await self.get_active_symbols(limit=15)
            volatile_symbols = []
            
            for symbol in symbols:
                klines = await self.get_klines(symbol, '5m', 15)
                if klines:
                    volatility = self.calculate_volatility(klines)
                    if volatility > self.detection_params['volatility_spike_threshold']:
                        volatile_symbols.append({
                            'symbol': symbol.replace('/USDT', ''),
                            'volatility': volatility,
                            'price': float(klines[-1][4]) if klines else 0
                        })
                await asyncio.sleep(0.1)
            
            if volatile_symbols:
                volatile_symbols.sort(key=lambda x: x['volatility'], reverse=True)
                
                response = "⚡ **ВОЛАТИЛЬНІСТЬ:**\n\n"
                for i, data in enumerate(volatile_symbols[:5], 1):
                    response += (
                        f"{i}. **{data['symbol']}** - {data['volatility']:.2f}%\n"
                        f"   💰 Ціна: ${data['price']:.6f}\n\n"
                    )
                
                await msg.edit_text(response, parse_mode='Markdown')
            else:
                await msg.edit_text("✅ Волатильність в нормі")
                
        except Exception as e:
            await update.message.reply_text("❌ Помилка аналізу")

    async def ai_risk_scan_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Нова функція: AI аналіз ризиків"""
        try:
            msg = await update.message.reply_text("🤖 AI АНАЛІЗ РИЗИКІВ...")
            
            symbols = await self.get_active_symbols(limit=10)
            risk_assessments = []
            
            for symbol in symbols:
                try:
                    analysis = await self.analyze_symbol(symbol)
                    if analysis:
                        risk_score = self.calculate_ai_risk_score(analysis)
                        risk_assessments.append({
                            'symbol': symbol.replace('/USDT', ''),
                            'risk_score': risk_score,
                            'pump_prob': analysis['pump_probability'],
                            'dump_prob': analysis['dump_probability'],
                            'price': analysis['price']
                        })
                    await asyncio.sleep(0.1)
                except:
                    continue
            
            if risk_assessments:
                risk_assessments.sort(key=lambda x: x['risk_score'], reverse=True)
                
                response = "🤖 **AI РЕЙТИНГ РИЗИКІВ:**\n\n"
                for i, risk in enumerate(risk_assessments[:5], 1):
                    risk_level = "🔴 ВИСОКИЙ" if risk['risk_score'] > 70 else "🟡 СЕРЕДНІЙ" if risk['risk_score'] > 40 else "🟢 НИЗЬКИЙ"
                    response += (
                        f"{i}. **{risk['symbol']}** - {risk_level}\n"
                        f"   📊 Ризик: {risk['risk_score']}%\n"
                        f"   🚨 Pump: {risk['pump_prob']:.2%}\n"
                        f"   📉 Dump: {risk['dump_prob']:.2%}\n"
                        f"   💰 Ціна: ${risk['price']:.6f}\n\n"
                    )
                
                await msg.edit_text(response, parse_mode='Markdown')
            else:
                await msg.edit_text("✅ Ризики в межах норми")
                
        except Exception as e:
            await update.message.reply_text("❌ Помилка AI аналізу")

    def calculate_ai_risk_score(self, analysis: Dict) -> float:
        """Розрахунок AI оцінки ризику"""
        try:
            pump_prob = analysis['pump_probability']
            dump_prob = analysis['dump_probability']
            volatility = analysis['technical_indicators']['volatility']
            imbalance = abs(analysis['orderbook_metrics']['imbalance'])
            
            risk_score = (
                pump_prob * 0.3 +
                dump_prob * 0.3 +
                min(volatility / 10, 1.0) * 0.2 +
                min(imbalance / 0.5, 1.0) * 0.2
            ) * 100
            
            return min(round(risk_score, 1), 100)
        except:
            return 50.0

    async def quick_scan_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Швидке сканування"""
        try:
            msg = await update.message.reply_text("⚡ ШВИДКЕ СКАНУВАННЯ...")
            
            symbols = await self.get_high_volume_symbols(limit=10)
            quick_signals = []
            
            for symbol in symbols:
                market_data = await self.get_market_data(symbol)
                orderbook = await self.get_orderbook_depth(symbol)
                
                if market_data and orderbook:
                    is_potential = self.quick_pump_check(market_data, orderbook)
                    if is_potential:
                        quick_signals.append({
                            'symbol': symbol.replace('/USDT', ''),
                            'price': market_data['close'],
                            'volume': market_data['volume'],
                            'change': market_data['percentage']
                        })
                await asyncio.sleep(0.1)
            
            if quick_signals:
                response = "⚡ **ШВИДКІ СИГНАЛИ:**\n\n"
                for i, signal in enumerate(quick_signals[:5], 1):
                    response += (
                        f"{i}. **{signal['symbol']}**\n"
                        f"   💰 Ціна: ${signal['price']:.6f}\n"
                        f"   📈 Зміна: {signal['change']:.2f}%\n"
                        f"   📊 Об'єм: ${signal['volume']:,.0f}\n\n"
                    )
                
                await msg.edit_text(response, parse_mode='Markdown')
            else:
                await msg.edit_text("✅ Швидких сигналів не знайдено")
                
        except Exception as e:
            await update.message.reply_text("❌ Помилка швидкого сканування")

    async def emergency_scan(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Екстрене сканування"""
        try:
            msg = await update.message.reply_text("🚨 ЕКСТРЕНЕ СКАНУВАННЯ...")
            
            symbols = await self.get_high_volume_symbols(limit=8)
            critical_signals = []
            
            for symbol in symbols:
                market_data = await self.get_market_data(symbol)
                if market_data and abs(market_data['percentage']) > 10:
                    critical_signals.append({
                        'symbol': symbol.replace('/USDT', ''),
                        'price': market_data['close'],
                        'change': market_data['percentage'],
                        'volume': market_data['volume'],
                        'is_pump': market_data['percentage'] > 0
                    })
                await asyncio.sleep(0.1)
            
            if critical_signals:
                response = "🚨 **КРИТИЧНІ ЗМІНИ:**\n\n"
                for i, signal in enumerate(critical_signals, 1):
                    signal_type = "PUMP" if signal['is_pump'] else "DUMP"
                    response += (
                        f"{i}. **{signal['symbol']}** - {signal_type}\n"
                        f"   📈 Зміна: {signal['change']:+.2f}%\n"
                        f"   💰 Ціна: ${signal['price']:.6f}\n\n"
                    )
                
                await msg.edit_text(response, parse_mode='Markdown')
            else:
                await msg.edit_text("✅ Критичних змін не виявлено")
                
        except Exception as e:
            await update.message.reply_text("❌ Помилка екстреного сканування")

    async def debug_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Діагностична команда"""
        try:
            network_ok = await self.check_network_connection()
            test_symbol = 'BTC/USDT'
            
            market_data = await self.get_market_data(test_symbol)
            orderbook = await self.get_orderbook_depth(test_symbol)
            klines = await self.get_klines(test_symbol, '5m', 5)
            
            debug_info = f"""
🔧 **ДІАГНОСТИКА:**

📡 Мережа: {'✅' if network_ok else '❌'}
📊 Market data: {'✅' if market_data else '❌'}
📋 Orderbook: {'✅' if orderbook and orderbook.get('bids') else '❌'}
📈 Klines: {len(klines) if klines else 0}

💰 BTC Ціна: {market_data.get('close', 'N/A') if market_data else 'N/A'}
📊 Об'єм: {market_data.get('volume', 'N/A') if market_data else 'N/A'}
            """
            
            await update.message.reply_text(debug_info, parse_mode='Markdown')
            
        except Exception as e:
            await update.message.reply_text(f"❌ Помилка діагностики: {e}")

    async def settings_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Налаштування"""
        try:
            settings_text = "⚙️ **НАЛАШТУВАННЯ:**\n\n"
            for key, value in self.detection_params.items():
                settings_text += f"• {key}: {value}\n"
            
            await update.message.reply_text(settings_text, parse_mode='Markdown')
        except Exception as e:
            await update.message.reply_text("❌ Помилка налаштувань")

    async def blacklist_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Чорний список"""
        try:
            if context.args:
                symbol = context.args[0].upper()
                if symbol in self.coin_blacklist:
                    self.coin_blacklist.remove(symbol)
                    await update.message.reply_text(f"✅ {symbol} видалено")
                else:
                    self.coin_blacklist.add(symbol)
                    await update.message.reply_text(f"✅ {symbol} додано")
            else:
                if self.coin_blacklist:
                    blacklist_text = "🚫 **ЧОРНИЙ СПИСОК:**\n" + "\n".join(self.coin_blacklist)
                else:
                    blacklist_text = "📝 Чорний список порожній"
                await update.message.reply_text(blacklist_text, parse_mode='Markdown')
                
        except Exception as e:
            await update.message.reply_text("❌ Помилка чорного списку")

    async def performance_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Статистика"""
        try:
            total = self.performance_metrics['total_scans']
            signals = self.performance_metrics['signals_triggered']
            success_rate = (signals / total * 100) if total > 0 else 0
            
            stats_text = (
                "📈 **СТАТИСТИКА:**\n\n"
                f"• Сканувань: {total}\n"
                f"• Сигналів: {signals}\n"
                f"• Pump: {self.performance_metrics['pump_signals_detected']}\n"
                f"• Dump: {self.performance_metrics['dump_signals_detected']}\n"
                f"• Успішність: {success_rate:.1f}%\n"
            )
            
            await update.message.reply_text(stats_text, parse_mode='Markdown')
        except Exception as e:
            await update.message.reply_text("❌ Помилка статистики")

    async def advanced_button_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обробник кнопок"""
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
            elif query.data == "ai_risk_scan":
                await self.ai_risk_scan_command(query, context)
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
                await query.edit_message_text("🔄 Оновлюю...")
                await asyncio.sleep(1)
                await self.start_command(query, context)
                
        except Exception as e:
            await query.edit_message_text("❌ Помилка обробки")

    async def run(self):
        """Запуск бота"""
        try:
            logger.info("🤖 Запускаю бота...")
            await self.app.initialize()
            await self.app.start()
            await self.app.updater.start_polling()
            
            logger.info("✅ Бот запущено!")
            
            while True:
                await asyncio.sleep(3600)
            
        except Exception as e:
            logger.error(f"❌ Помилка запуску: {e}")
            raise

    def save_state(self):
        """Збереження стану"""
        try:
            state = {
                'coin_blacklist': list(self.coin_blacklist),
                'performance_metrics': self.performance_metrics,
                'last_update': time.time()
            }
            
            with open('bot_state.json', 'w') as f:
                json.dump(state, f)
            
            logger.info("💾 Стан збережено")
        except Exception as e:
            logger.error(f"❌ Помилка збереження: {e}")

    def load_state(self):
        """Завантаження стану"""
        try:
            if os.path.exists('bot_state.json'):
                with open('bot_state.json', 'r') as f:
                    state = json.load(f)
                
                self.coin_blacklist = set(state.get('coin_blacklist', []))
                self.performance_metrics.update(state.get('performance_metrics', {}))
                
                logger.info("📂 Стан завантажено")
        except Exception as e:
            logger.error(f"❌ Помилка завантаження: {e}")

def main():
    """Головна функція"""
    try:
        BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
        
        if not BOT_TOKEN:
            logger.error("❌ Встановіть TELEGRAM_BOT_TOKEN")
            return
        
        bot = UltimatePumpDumpDetector(BOT_TOKEN)
        bot.load_state()
        
        logger.info("🚀 Запускаю бота...")
        bot.app.run_polling()
        
    except KeyboardInterrupt:
        logger.info("⏹️ Зупинка...")
        bot.save_state()
    except Exception as e:
        logger.error(f"❌ Критична помилка: {e}")
        try:
            bot.save_state()
        except:
            pass
        raise

if __name__ == '__main__':
    logging.getLogger('ccxt').setLevel(logging.WARNING)
    logging.getLogger('telegram').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)
    
    main()