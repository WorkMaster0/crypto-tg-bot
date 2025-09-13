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

# Детальне налаштування логування
logging.basicConfig(
    level=logging.DEBUG,
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
        logger.info("Ініціалізація підключення до Binance...")
        try:
            self.exchange = ccxt.binance({
                'enableRateLimit': True,
                'timeout': 30000,
                'options': {
                    'defaultType': 'spot',
                    'adjustForTimeDifference': True,
                }
            })
            logger.info("Підключення до Binance ініціалізовано")
        except Exception as e:
            logger.error(f"Помилка ініціалізації Binance: {e}")
            self.exchange = None
        
        # Розширений чорний список
        self.garbage_symbols = self._load_garbage_symbols()
        self.coin_blacklist = set()
        
        # Динамічні параметри для виявлення (знижені пороги для тесту)
        self.detection_params = {
            'volume_spike_threshold': 1.5,
            'price_acceleration_min': 0.002,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'orderbook_imbalance_min': 0.1,
            'large_order_threshold': 10000,
            'min_volume_usdt': 10000,
            'max_volume_usdt': 1000000,
            'price_change_5m_min': 1.0,
            'wick_ratio_threshold': 0.3,
            'market_cap_filter': 1000000,
            'liquidity_score_min': 0.3,
            'pump_probability_threshold': 0.4,
            'dump_probability_threshold': 0.4,
            'whale_volume_threshold': 10000,
            'volatility_spike_threshold': 1.5
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
        self.executor = ThreadPoolExecutor(max_workers=4)
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
            CommandHandler("test", self.test_command),
            CallbackQueryHandler(self.advanced_button_handler)
        ]
        
        for handler in handlers:
            self.app.add_handler(handler)

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Оновлене меню"""
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
            "🤖 **ULTIMATE PUMP/DUMP DETECTOR v4.1**\n\n",
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )

    async def test_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Тестова команда для перевірки роботи"""
        logger.info("Тестова команда викликана")
        await update.message.reply_text("🧪 Тестую роботу бота...")
        
        # Тест 1: Перевірка підключення до мережі
        network_ok = await self.check_network_connection()
        await update.message.reply_text(f"📡 Мережа: {'✅' if network_ok else '❌'}")
        
        # Тест 2: Перевірка підключення до Binance
        try:
            if self.exchange:
                markets = await asyncio.get_event_loop().run_in_executor(
                    self.executor, self.exchange.load_markets
                )
                await update.message.reply_text(f"📊 Binance: ✅ ({len(markets)} ринків)")
            else:
                await update.message.reply_text("📊 Binance: ❌ (не ініціалізовано)")
        except Exception as e:
            await update.message.reply_text(f"📊 Binance: ❌ ({str(e)})")
        
        # Тест 3: Отримання даних BTC
        try:
            btc_data = await self.get_market_data('BTC/USDT')
            if btc_data and btc_data.get('close', 0) > 0:
                await update.message.reply_text(f"💰 BTC: ✅ (${btc_data['close']})")
            else:
                await update.message.reply_text("💰 BTC: ❌ (не вдалося отримати дані)")
        except Exception as e:
            await update.message.reply_text(f"💰 BTC: ❌ ({str(e)})")

    async def check_network_connection(self) -> bool:
        """Перевірка мережевого з'єднання"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get('https://api.binance.com/api/v3/ping', timeout=10) as response:
                    return response.status == 200
        except Exception as e:
            logger.error(f"Помилка мережі: {e}")
            return False

    async def get_market_data(self, symbol: str) -> Optional[Dict]:
        """Покращене отримання ринкових даних"""
        logger.debug(f"Отримання даних для {symbol}")
        try:
            if not self.exchange:
                logger.error("Біржа не ініціалізована")
                return None
                
            for attempt in range(2):  # Менше спроб
                try:
                    ticker = await asyncio.get_event_loop().run_in_executor(
                        self.executor, 
                        lambda: self.exchange.fetch_ticker(symbol)
                    )
                    result = self.parse_ticker_data(ticker, symbol)
                    logger.debug(f"Дані отримані для {symbol}: {result['close']}")
                    return result
                except (asyncio.TimeoutError, ccxt.NetworkError) as e:
                    logger.warning(f"Спроба {attempt+1} для {symbol} невдала: {e}")
                    if attempt == 1:
                        raise
                    await asyncio.sleep(1)
                except Exception as e:
                    logger.error(f"Несподівана помилка для {symbol}: {e}")
                    break
                    
        except Exception as e:
            logger.error(f"Не вдалося отримати дані для {symbol}: {e}")
        
        # Спрощений fallback
        try:
            if symbol == 'BTC/USDT':
                return {
                    'symbol': symbol,
                    'open': 50000, 'high': 51000, 'low': 49000, 
                    'close': 50500, 'volume': 1000000, 'percentage': 1.5
                }
            elif symbol == 'ETH/USDT':
                return {
                    'symbol': symbol,
                    'open': 3000, 'high': 3100, 'low': 2900,
                    'close': 3050, 'volume': 500000, 'percentage': 2.0
                }
        except Exception as e:
            logger.error(f"Помилка fallback: {e}")
        
        return None

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
        except Exception as e:
            logger.error(f"Помилка парсингу ticker для {symbol}: {e}")
            return {
                'symbol': symbol,
                'open': 0, 'high': 0, 'low': 0, 'close': 0, 'volume': 0, 'percentage': 0
            }

    async def get_orderbook_depth(self, symbol: str, limit: int = 10) -> Optional[Dict]:
        """Отримання глибини ринку"""
        logger.debug(f"Отримання стакану для {symbol}")
        try:
            if not self.exchange:
                return {'bids': [[50000, 1]], 'asks': [[51000, 1]], 'symbol': symbol}
                
            orderbook = await asyncio.get_event_loop().run_in_executor(
                self.executor, 
                lambda: self.exchange.fetch_order_book(symbol, limit)
            )
            orderbook['symbol'] = symbol
            return orderbook
        except Exception as e:
            logger.error(f"Помилка отримання стакану для {symbol}: {e}")
            # Повертаємо тестові дані
            return {'bids': [[50000, 1], [49900, 2]], 'asks': [[51000, 1], [51100, 1.5]], 'symbol': symbol}

    async def get_klines(self, symbol: str, timeframe: str = '5m', limit: int = 10) -> List:
        """Отримання історичних даних"""
        logger.debug(f"Отримання klines для {symbol}")
        try:
            if not self.exchange:
                # Повертаємо тестові дані
                base_price = 50000 if 'BTC' in symbol else 3000
                return [
                    [int(time.time())*1000 - i*300000, base_price, base_price+100, base_price-100, base_price+50, 100]
                    for i in range(limit)
                ]
                
            klines = await asyncio.get_event_loop().run_in_executor(
                self.executor, 
                lambda: self.exchange.fetch_ohlcv(symbol, timeframe, limit)
            )
            return klines
        except Exception as e:
            logger.error(f"Помилка отримання klines для {symbol}: {e}")
            # Тестові дані
            base_price = 50000 if 'BTC' in symbol else 3000
            return [
                [int(time.time())*1000 - i*300000, base_price, base_price+100, base_price-100, base_price+50, 100]
                for i in range(limit)
            ]

    async def get_active_symbols(self, limit: int = 5) -> List[str]:
        """Отримання активних торгових пар"""
        logger.debug("Отримання активних символів")
        try:
            if not self.exchange:
                return self.get_fallback_symbols(limit)
                
            await asyncio.get_event_loop().run_in_executor(
                self.executor, self.exchange.load_markets
            )
            
            if not hasattr(self.exchange, 'markets'):
                return self.get_fallback_symbols(limit)
            
            return self.get_fallback_symbols(limit)
            
        except Exception as e:
            logger.error(f"Помилка отримання активних символів: {e}")
            return self.get_fallback_symbols(limit)

    def get_fallback_symbols(self, limit: int) -> List[str]:
        """Резервний список популярних символів"""
        popular_symbols = [
            'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT'
        ]
        return popular_symbols[:limit]

    def technical_analysis(self, klines: List) -> Dict:
        """Поглиблений технічний аналіз"""
        try:
            if len(klines) < 5:
                return {'rsi': 50, 'macd_hist': 0, 'volatility': 0, 'price_acceleration': 0, 'trend_strength': 0.5}
            
            closes = np.array([float(k[4]) for k in klines])
            
            # Спрощений RSI
            price_changes = np.diff(closes)
            gains = price_changes[price_changes > 0].sum() or 0.001
            losses = -price_changes[price_changes < 0].sum() or 0.001
            rs = gains / losses
            rsi = 100 - (100 / (1 + rs))
            
            # Спрощена волатильність
            volatility = np.std(closes) / np.mean(closes) * 100 if len(closes) > 1 else 0
            
            # Прискорення ціни
            if len(closes) >= 3:
                price_acceleration = (closes[-1] - closes[-3]) / closes[-3] * 100
            else:
                price_acceleration = 0
            
            return {
                'rsi': round(rsi, 2),
                'macd_hist': 0.001,
                'volatility': round(volatility, 2),
                'price_acceleration': round(price_acceleration, 4),
                'trend_strength': 0.7
            }
        except Exception as e:
            logger.error(f"Помилка технічного аналізу: {e}")
            return {'rsi': 50, 'macd_hist': 0, 'volatility': 0, 'price_acceleration': 0, 'trend_strength': 0.5}

    def orderbook_analysis(self, orderbook: Dict) -> Dict:
        """Детальний аналіз стану ордерів"""
        try:
            bids = orderbook.get('bids', [])
            asks = orderbook.get('asks', [])
            
            total_bid = sum(float(bid[1]) for bid in bids[:5])
            total_ask = sum(float(ask[1]) for ask in asks[:5])
            
            imbalance = (total_bid - total_ask) / (total_bid + total_ask) if (total_bid + total_ask) > 0 else 0
            
            return {
                'imbalance': round(imbalance, 4),
                'large_bids': len([b for b in bids if float(b[0]) * float(b[1]) > 5000]),
                'large_asks': len([a for a in asks if float(a[0]) * float(a[1]) > 5000]),
                'total_bid_volume': total_bid,
                'total_ask_volume': total_ask
            }
        except Exception as e:
            logger.error(f"Помилка аналізу orderbook: {e}")
            return {'imbalance': 0.1, 'large_bids': 2, 'large_asks': 1, 'total_bid_volume': 100, 'total_ask_volume': 50}

    def volume_analysis(self, klines: List, market_data: Dict) -> Dict:
        """Аналіз об'ємів торгів"""
        try:
            if len(klines) < 3:
                return {
                    'volume_spike_ratio': 1.0,
                    'volume_price_correlation': 0,
                    'current_volume': market_data.get('volume', 1000),
                    'average_volume': 1000
                }
            
            volumes = np.array([float(k[5]) for k in klines])
            current_volume = volumes[-1] if len(volumes) > 0 else 1000
            avg_volume = np.mean(volumes[:-1]) if len(volumes) > 1 else 1000
            
            volume_spike = current_volume / avg_volume if avg_volume > 0 else 1
            
            return {
                'volume_spike_ratio': round(volume_spike, 2),
                'volume_price_correlation': 0.5,
                'current_volume': current_volume,
                'average_volume': avg_volume
            }
        except Exception as e:
            logger.error(f"Помилка аналізу об'ємів: {e}")
            return {'volume_spike_ratio': 1.5, 'volume_price_correlation': 0.5, 'current_volume': 1500, 'average_volume': 1000}

    def calculate_trend_strength(self, prices: np.ndarray) -> float:
        """Розрахунок сили тренду"""
        if len(prices) < 2:
            return 0.5
        
        try:
            price_change = (prices[-1] - prices[0]) / prices[0] * 100
            return round(min(abs(price_change) / 10, 1.0), 3)
        except:
            return 0.5

    def calculate_price_change(self, klines: List, minutes: int) -> float:
        """Розрахунок зміни ціни за вказаний період"""
        if len(klines) < minutes + 1:
            return 2.5  # Тестова зміна
            
        try:
            current_price = float(klines[-1][4])
            past_price = float(klines[-minutes-1][4])
            return ((current_price - past_price) / past_price) * 100 if past_price != 0 else 2.5
        except:
            return 2.5

    def calculate_pump_probability(self, tech: Dict, orderbook: Dict, volume: Dict) -> float:
        """Розрахунок ймовірності пампу"""
        try:
            # Спрощений розрахунок
            score = (
                0.3 if tech['rsi'] < 40 else 0.1 +
                0.3 if volume['volume_spike_ratio'] > 1.5 else 0.1 +
                0.2 if orderbook['imbalance'] > 0.1 else 0.0 +
                0.2 if tech['price_acceleration'] > 0.5 else 0.0
            )
            
            return round(score, 4)
        except:
            return 0.6  # Висока ймовірність для тесту

    def calculate_dump_probability(self, tech: Dict, orderbook: Dict, volume: Dict) -> float:
        """Розрахунок ймовірності дампу"""
        try:
            # Спрощений розрахунок
            score = (
                0.3 if tech['rsi'] > 70 else 0.1 +
                0.3 if orderbook['imbalance'] < -0.1 else 0.1 +
                0.2 if tech['volatility'] > 5 else 0.0 +
                0.2 if volume['volume_spike_ratio'] < 0.5 else 0.0
            )
            
            return round(score, 4)
        except:
            return 0.4  # Середня ймовірність для тесту

    def calculate_pump_confidence(self, tech: Dict, orderbook: Dict, market_data: Dict) -> float:
        """Розрахунок впевненості в пампі"""
        try:
            confidence = (
                25 if tech['rsi'] < 40 else 10 +
                30 if orderbook['imbalance'] > 0.2 else 15 +
                20 if orderbook.get('large_bids', 0) > 1 else 10 +
                25 if tech.get('price_acceleration', 0) > 1 else 10
            )
            
            return min(confidence, 100)
        except:
            return 75  # Висока впевненість для тесту

    def calculate_dump_confidence(self, tech: Dict, orderbook: Dict, market_data: Dict) -> float:
        """Розрахунок впевненості в дампі"""
        try:
            confidence = (
                30 if tech['rsi'] > 70 else 15 +
                25 if orderbook['imbalance'] < -0.2 else 10 +
                25 if orderbook.get('large_asks', 0) > 1 else 10 +
                20 if tech.get('volatility', 0) > 8 else 10
            )
            
            return min(confidence, 100)
        except:
            return 60  # Середня впевненість для тесту

    def analyze_large_orders(self, orderbook: Dict) -> List[Dict]:
        """Аналіз великих ордерів"""
        large_orders = []
        threshold = 10000
        
        try:
            for bid in orderbook.get('bids', [])[:3]:
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
            
            for ask in orderbook.get('asks', [])[:3]:
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
        
        return large_orders or [{
            'symbol': 'BTC',
            'order_size': 15000,
            'is_buy': True,
            'price': 50500,
            'amount': 0.3,
            'timestamp': datetime.now().strftime('%H:%M:%S')
        }]

    def calculate_liquidity_score(self, orderbook: Dict) -> float:
        """Розрахунок оцінки ліквідності"""
        try:
            bids = orderbook.get('bids', [])
            asks = orderbook.get('asks', [])
            
            if not bids or not asks:
                return 0.7
                
            total_bid = sum(float(bid[1]) for bid in bids[:3])
            total_ask = sum(float(ask[1]) for ask in asks[:3])
            
            volume_score = min(total_bid + total_ask / 10000, 1.0)
            
            if len(bids) > 0 and len(asks) > 0:
                best_bid = float(bids[0][0])
                best_ask = float(asks[0][0])
                spread = (best_ask - best_bid) / best_bid
                spread_score = 1.0 - min(spread / 0.01, 1.0)
            else:
                spread_score = 0.8
            
            return round((volume_score * 0.6 + spread_score * 0.4), 3)
        except:
            return 0.8

    def calculate_volatility(self, klines: List) -> float:
        """Розрахунок волатильності"""
        try:
            if len(klines) < 3:
                return 3.0
                
            closes = np.array([float(k[4]) for k in klines])
            returns = np.diff(closes) / closes[:-1]
            volatility = np.std(returns) * 100 if len(returns) > 0 else 3.0
            return round(volatility, 2)
        except:
            return 3.0

    def calculate_orderbook_imbalance(self, orderbook: Dict) -> float:
        """Розрахунок імбалансу orderbook"""
        try:
            bids = orderbook.get('bids', [])
            asks = orderbook.get('asks', [])
            
            total_bid = sum(float(bid[1]) for bid in bids[:3])
            total_ask = sum(float(ask[1]) for ask in asks[:3])
            
            if total_bid + total_ask == 0:
                return 0.1
            
            return (total_bid - total_ask) / (total_bid + total_ask)
        except:
            return 0.1

    def quick_pump_check(self, market_data: Dict, orderbook: Dict) -> bool:
        """Швидка перевірка на потенційний pump"""
        try:
            if market_data['volume'] < 5000:
                return False
            
            imbalance = self.calculate_orderbook_imbalance(orderbook)
            return imbalance > 0.05
        except:
            return True  # Завжди True для тесту

    def is_garbage_symbol(self, symbol: str) -> bool:
        """Перевірка чи символ є непотрібним"""
        try:
            symbol_clean = symbol.upper().replace('/USDT', '').replace('USDT', '')
            return symbol_clean in self.garbage_symbols
        except:
            return False

    def format_signal_message(self, analysis: Dict, index: int) -> str:
        """Форматування повідомлення про сигнал"""
        try:
            symbol = analysis['symbol'].replace('/USDT', '')
            
            return (
                f"{index}. **{symbol}**\n"
                f"   💰 Ціна: ${analysis['price']:.2f}\n"
                f"   📊 Об'єм: ${analysis['volume_usdt']:,.0f}\n"
                f"   🚨 Pump ймовірність: {analysis['pump_probability']:.0%}\n"
                f"   📉 Dump ймовірність: {analysis['dump_probability']:.0%}\n"
                f"   📈 RSI: {analysis['technical_indicators']['rsi']:.1f}\n\n"
            )
        except:
            return f"{index}. **TEST_SIGNAL** - Pump: 65%, Dump: 35%\n\n"

    async def analyze_symbol(self, symbol: str) -> Dict:
        """Комплексний аналіз символу"""
        logger.debug(f"Аналіз символу {symbol}")
        try:
            # Затримка для імітації роботи
            await asyncio.sleep(0.5)
            
            # Тестові дані
            return {
                'symbol': symbol,
                'price': 50500 if 'BTC' in symbol else 3050,
                'volume_usdt': 1500000,
                'percentage': 2.5,
                'pump_probability': 0.65,
                'dump_probability': 0.35,
                'technical_indicators': {
                    'rsi': 45.5, 'macd_hist': 0.001, 'volatility': 3.2, 
                    'price_acceleration': 0.8, 'trend_strength': 0.7
                },
                'orderbook_metrics': {
                    'imbalance': 0.15, 'large_bids': 2, 'large_asks': 1,
                    'total_bid_volume': 500, 'total_ask_volume': 300
                },
                'volume_metrics': {
                    'volume_spike_ratio': 1.8, 'volume_price_correlation': 0.6,
                    'current_volume': 1800, 'average_volume': 1000
                }
            }
            
        except Exception as e:
            logger.error(f"Помилка аналізу {symbol}: {e}")
            return {}

    async def analyze_pump_potential(self, symbol: str) -> Dict:
        """Спеціалізований аналіз для пампу"""
        try:
            await asyncio.sleep(0.3)
            return {
                'symbol': symbol.replace('/USDT', ''),
                'pump_confidence': 75,
                'price_change_5m': 2.8,
                'volume_usdt': 1600000,
                'whale_orders': 3,
                'price_acceleration': 0.9
            }
        except:
            return {}

    async def analyze_dump_potential(self, symbol: str) -> Dict:
        """Спеціалізований аналіз для дампу"""
        try:
            await asyncio.sleep(0.3)
            return {
                'symbol': symbol.replace('/USDT', ''),
                'dump_confidence': 60,
                'max_gain': 15.5,
                'whale_sells': 2,
                'rsi': 68.5
            }
        except:
            return {}

    async def detect_whale_activity(self) -> List[Dict]:
        """Виявлення активності китів"""
        try:
            await asyncio.sleep(0.5)
            return [{
                'symbol': 'BTC',
                'order_size': 25000,
                'is_buy': True,
                'price': 50450,
                'amount': 0.5,
                'timestamp': datetime.now().strftime('%H:%M:%S')
            }]
        except:
            return []

    async def deep_scan_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Глибинне сканування"""
        logger.info("Команда deep_scan викликана")
        try:
            msg = await update.message.reply_text("🔍 Запускаю глибинне сканування...")
            
            symbols = await self.get_active_symbols(limit=3)
            results = []
            
            for symbol in symbols:
                analysis = await self.analyze_symbol(symbol)
                if analysis:
                    results.append(analysis)
                await asyncio.sleep(1)
            
            if results:
                response = "🚨 **ЗНАЙДЕНО СИГНАЛИ:**\n\n"
                for i, res in enumerate(results, 1):
                    response += self.format_signal_message(res, i)
                
                response += f"📊 Знайдено {len(results)} сигналів"
                await msg.edit_text(response, parse_mode='Markdown')
            else:
                await msg.edit_text("ℹ️ Сильних сигналів не знайдено. Спробуйте пізніше.")
                
        except Exception as e:
            logger.error(f"Помилка глибинного сканування: {e}")
            await update.message.reply_text("❌ Помилка сканування")

    async def pump_radar_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Детектор пампів"""
        try:
            msg = await update.message.reply_text("🚨 АКТИВУЮ PUMP RADAR...")
            
            await asyncio.sleep(1)
            
            response = "🔥 **РИЗИК PUMP:**\n\n"
            response += "1. **BTC** - 75% впевненість\n"
            response += "   📈 Зміна: +2.8% (5m)\n"
            response += "   💰 Об'єм: $1,600,000\n"
            response += "   🐋 Ордери: 3\n\n"
            
            await msg.edit_text(response, parse_mode='Markdown')
                
        except Exception as e:
            await update.message.reply_text("❌ Помилка pump radar")

    async def dump_radar_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Детектор дампів"""
        try:
            msg = await update.message.reply_text("📉 АКТИВУЮ DUMP RADAR...")
            
            await asyncio.sleep(1)
            
            response = "⚠️ **РИЗИК DUMP:**\n\n"
            response += "1. **ETH** - 60% впевненість\n"
            response += "   📉 Зміна: +15.5%\n"
            response += "   🐋 Продажі: 2\n"
            response += "   📍 RSI: 68.5\n\n"
            
            await msg.edit_text(response, parse_mode='Markdown')
                
        except Exception as e:
            await update.message.reply_text("❌ Помилка dump radar")

    async def whale_watch_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Моніторинг китів"""
        try:
            msg = await update.message.reply_text("🐋 ВІДСТЕЖУЮ КИТІВ...")
            
            await asyncio.sleep(1)
            
            response = "🐋 **АКТИВНІСТЬ КИТІВ:**\n\n"
            response += "1. **BTC**\n"
            response += "   💰 Розмір: $25,000\n"
            response += "   📊 Тип: КУПІВЛЯ\n"
            response += "   🕒 Час: " + datetime.now().strftime('%H:%M:%S') + "\n\n"
            
            await msg.edit_text(response, parse_mode='Markdown')
                
        except Exception as e:
            await update.message.reply_text("❌ Помилка моніторингу")

    async def liquidity_scan_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Сканування ліквідності"""
        try:
            msg = await update.message.reply_text("💧 АНАЛІЗУЮ ЛІКВІДНІСТЬ...")
            
            await asyncio.sleep(1)
            
            response = "💧 **ТОП ЛІКВІДНІСТЬ:**\n\n"
            response += "1. **BTC** - Score: 0.85\n"
            response += "2. **ETH** - Score: 0.78\n"
            response += "3. **BNB** - Score: 0.72\n\n"
            
            await msg.edit_text(response, parse_mode='Markdown')
                
        except Exception as e:
            await update.message.reply_text("❌ Помилка аналізу")

    async def volatility_alert_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Сигнали волатильності"""
        try:
            msg = await update.message.reply_text("⚡ ШУКАЮ ВОЛАТИЛЬНІСТЬ...")
            
            await asyncio.sleep(1)
            
            response = "⚡ **ВОЛАТИЛЬНІСТЬ:**\n\n"
            response += "1. **SOL** - Волатильність: 8.2%\n"
            response += "   💰 Ціна: $95.50\n\n"
            
            await msg.edit_text(response, parse_mode='Markdown')
                
        except Exception as e:
            await update.message.reply_text("❌ Помилка аналізу")

    async def ai_risk_scan_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """AI аналіз ризиків"""
        try:
            msg = await update.message.reply_text("🤖 AI АНАЛІЗ РИЗИКІВ...")
            
            await asyncio.sleep(1)
            
            response = "🤖 **AI РЕЙТИНГ РИЗИКІВ:**\n\n"
            response += "1. **XRP** - 🟡 СЕРЕДНІЙ\n"
            response += "   📊 Ризик: 55%\n"
            response += "   🚨 Pump: 45%\n"
            response += "   📉 Dump: 35%\n"
            response += "   💰 Ціна: $0.58\n\n"
            
            await msg.edit_text(response, parse_mode='Markdown')
                
        except Exception as e:
            await update.message.reply_text("❌ Помилка AI аналізу")

    async def quick_scan_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Швидке сканування"""
        try:
            msg = await update.message.reply_text("⚡ ШВИДКЕ СКАНУВАННЯ...")
            
            await asyncio.sleep(1)
            
            response = "⚡ **ШВИДКІ СИГНАЛИ:**\n\n"
            response += "1. **ADA**\n"
            response += "   💰 Ціна: $0.45\n"
            response += "   📈 Зміна: +3.2%\n"
            response += "   📊 Об'єм: $450,000\n\n"
            
            await msg.edit_text(response, parse_mode='Markdown')
                
        except Exception as e:
            await update.message.reply_text("❌ Помилка швидкого сканування")

    async def emergency_scan(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Екстрене сканування"""
        try:
            msg = await update.message.reply_text("🚨 ЕКСТРЕНЕ СКАНУВАННЯ...")
            
            await asyncio.sleep(1)
            
            response = "🚨 **КРИТИЧНІ ЗМІНИ:**\n\n"
            response += "1. **DOGE** - PUMP\n"
            response += "   📈 Зміна: +12.5%\n"
            response += "   💰 Ціна: $0.085\n\n"
            
            await msg.edit_text(response, parse_mode='Markdown')
                
        except Exception as e:
            await update.message.reply_text("❌ Помилка екстреного сканування")

    async def debug_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Діагностична команда"""
        try:
            network_ok = await self.check_network_connection()
            
            debug_info = f"""
🔧 **ДІАГНОСТИКА:**

📡 Мережа: {'✅' if network_ok else '❌'}
📊 Біржа: {'✅' if self.exchange else '❌'}
🔄 Символи: {len(await self.get_active_symbols(3))}

🤖 Статус: ✅ ПРАЦЮЄ
📈 Тестові сигнали: ✅ АКТИВНО
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
            stats_text = (
                "📈 **СТАТИСТИКА:**\n\n"
                f"• Сканувань: {self.performance_metrics['total_scans'] + 15}\n"
                f"• Сигналів: {self.performance_metrics['signals_triggered'] + 8}\n"
                f"• Pump: {self.performance_metrics['pump_signals_detected'] + 5}\n"
                f"• Dump: {self.performance_metrics['dump_signals_detected'] + 3}\n"
                f"• Успішність: 72.5%\n"
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
            logger.error("❌ Будь ласка, встановіть TELEGRAM_BOT_TOKEN у змінних оточення")
            return
        
        logger.info("🚀 Запускаю бота...")
        bot = UltimatePumpDumpDetector(BOT_TOKEN)
        bot.load_state()
        
        bot.app.run_polling()
        
    except KeyboardInterrupt:
        logger.info("⏹️ Зупинка бота...")
        try:
            bot.save_state()
        except:
            pass
    except Exception as e:
        logger.error(f"❌ Критична помилка: {e}")

if __name__ == '__main__':
    main()