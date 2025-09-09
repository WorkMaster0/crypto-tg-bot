import requests
import asyncio
import aiohttp
import time
import hmac
import hashlib
import base64
from datetime import datetime
import logging
import threading
import os
from typing import Dict, List, Optional

# Налаштування логування
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Конфігурація
class Config:
    # Telegram
    TELEGRAM_TOKEN = "8489382938:AAHeFFZPODspuEFcSQyjw8lWzYpRRSv9n3g"
    TELEGRAM_CHAT_ID = "6053907025"
    
    # LBank API
    LBANK_API_KEY = "c21d76ba-3252-4372-b7dd-30ac80428363"
    LBANK_SECRET_KEY = "4EC9D3BB56CBD4C42B9E83F0C7B7C1A9"
    LBANK_BASE_URL = "https://api.lbank.info"
    
    # API Keys для альтернативних джерел
    BIRDEYE_API_KEY = "your_birdeye_api_key_here"  # Отримати на https://birdeye.so/
    DEXVIEW_API_KEY = "your_dexview_api_key_here"  # Отримати на https://dexview.com/
    
    # API URLs
    BIRDEYE_API_URL = "https://public-api.birdeye.so"
    DEXVIEW_API_URL = "https://api.dexview.com"
    
    # Торгові налаштування
    ORDER_VOLUME = 50  # Обсяг в USDT
    PRICE_PREMIUM = 0.0001  # 0.01% вище ринкової ціни
    MIN_TRADE_AMOUNT = 3000  # Мінімальна сума угоди для реакції
    
    # Фільтри токенів
    MIN_MARKET_CAP = 3000000  # 3M$ мінімальна капіталізація
    MAX_MARKET_CAP = 100000000  # 100M$ максимальна капіталізація
    MIN_VOLUME = 500000  # 500K$ мінімальний обсяг
    ALLOWED_CHAINS = ["solana", "bsc"]  # Тільки Solana та BSC
    BLACKLIST_TOKENS = ["shitcoin", "scam", "test", "meme", "fake", "pump", "dump"]

class TelegramClient:
    """Простий клієнт для Telegram без polling"""
    
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{token}"
    
    def send_message(self, text: str, parse_mode: str = None) -> bool:
        """Надсилання повідомлення через Telegram API"""
        try:
            url = f"{self.base_url}/sendMessage"
            payload = {
                'chat_id': self.chat_id,
                'text': text
            }
            if parse_mode:
                payload['parse_mode'] = parse_mode
            
            response = requests.post(url, json=payload, timeout=10)
            return response.status_code == 200
        except Exception as e:
            logging.error(f"Помилка відправки повідомлення: {e}")
            return False

# Ініціалізація клієнтів
telegram_client = TelegramClient(Config.TELEGRAM_TOKEN, Config.TELEGRAM_CHAT_ID)

class TokenFilter:
    """Клас для фільтрації токенів"""
    
    @staticmethod
    def is_token_allowed(token_data: Dict) -> bool:
        """Перевірка чи токен підходить під критерії"""
        try:
            # Перевірка капіталізації
            market_cap = token_data.get('market_cap', 0)
            if not (Config.MIN_MARKET_CAP <= market_cap <= Config.MAX_MARKET_CAP):
                logging.debug(f"Токен {token_data.get('symbol')} не підходить по капіталізації: {market_cap}")
                return False
            
            # Перевірка обсягу торгів
            volume = token_data.get('volume_24h', 0)
            if volume < Config.MIN_VOLUME:
                logging.debug(f"Токен {token_data.get('symbol')} не підходить по обсягу: {volume}")
                return False
            
            # Перевірка блокчейну
            chain = token_data.get('chain', '').lower()
            if chain not in Config.ALLOWED_CHAINS:
                logging.debug(f"Токен {token_data.get('symbol')} не підходить по блокчейну: {chain}")
                return False
            
            # Перевірка чорного списку
            symbol = token_data.get('symbol', '').lower()
            name = token_data.get('name', '').lower()
            
            for blacklisted in Config.BLACKLIST_TOKENS:
                if blacklisted in symbol or blacklisted in name:
                    logging.debug(f"Токен {symbol} в чорному списку: {blacklisted}")
                    return False
            
            # Додаткові перевірки якості
            price_change = abs(token_data.get('price_change_24h', 0))
            if price_change > 50:
                logging.debug(f"Токен {symbol} має завелику волатильність: {price_change}%")
                return False
                
            liquidity = token_data.get('liquidity', 0)
            if liquidity < 100000:
                logging.debug(f"Токен {symbol} має замалу ліквідність: {liquidity}")
                return False
                
            logging.info(f"✅ Токен {symbol} пройшов фільтрацію")
            return True
            
        except Exception as e:
            logging.error(f"Помилка фільтрації токена: {e}")
            return False

class DexDataClient:
    """Клієнт для роботи з альтернативними джерелами даних"""
    
    @staticmethod
    async def get_recent_trades(chain: str, limit: int = 20) -> List[Dict]:
        """Отримання останніх угод з альтернативних джерел"""
        try:
            if chain == "solana":
                return await DexDataClient._get_solana_trades(limit)
            elif chain == "bsc":
                return await DexDataClient._get_bsc_trades(limit)
            else:
                return []
                
        except Exception as e:
            logging.error(f"❌ Помилка отримання угод з {chain}: {e}")
            return []
    
    @staticmethod
    async def _get_solana_trades(limit: int) -> List[Dict]:
        """Отримання угод з Solana через Birdeye API"""
        try:
            url = f"{Config.BIRDEYE_API_URL}/public/trades?sort_by=time&order=desc&limit=100"
            headers = {"X-API-KEY": Config.BIRDEYE_API_KEY}
            
            logging.info("🔗 Запит до Birdeye API для Solana")
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=15) as response:
                    if response.status != 200:
                        logging.warning(f"❌ Birdeye API статус: {response.status}")
                        return []
                    
                    data = await response.json()
                    trades = []
                    
                    for item in data.get('data', {}).get('items', [])[:limit]:
                        try:
                            amount_usd = float(item.get('volume', 0))
                            if amount_usd >= Config.MIN_TRADE_AMOUNT:
                                trades.append({
                                    'chain': 'solana',
                                    'token_address': item.get('token_address', ''),
                                    'token_symbol': item.get('symbol', '').upper(),
                                    'amount_usd': amount_usd,
                                    'price': float(item.get('price', 0)),
                                    'timestamp': item.get('unix_time', 0),
                                    'dex_url': f"https://birdeye.so/token/{item.get('token_address', '')}"
                                })
                        except (ValueError, TypeError) as e:
                            continue
                    
                    logging.info(f"✅ Solana: знайдено {len(trades)} великих угод")
                    return trades
                    
        except Exception as e:
            logging.error(f"❌ Помилка Birdeye API: {e}")
            return []
    
    @staticmethod
    async def _get_bsc_trades(limit: int) -> List[Dict]:
        """Отримання угод з BSC через DexView API"""
        try:
            url = f"{Config.DEXVIEW_API_URL}/v1/trades/recent?chain=bsc&limit=100"
            headers = {"Authorization": f"Bearer {Config.DEXVIEW_API_KEY}"}
            
            logging.info("🔗 Запит до DexView API для BSC")
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=15) as response:
                    if response.status != 200:
                        logging.warning(f"❌ DexView API статус: {response.status}")
                        return []
                    
                    data = await response.json()
                    trades = []
                    
                    for item in data.get('trades', [])[:limit]:
                        try:
                            amount_usd = float(item.get('valueUSD', 0))
                            if amount_usd >= Config.MIN_TRADE_AMOUNT:
                                trades.append({
                                    'chain': 'bsc',
                                    'token_address': item.get('tokenAddress', ''),
                                    'token_symbol': item.get('tokenSymbol', '').upper(),
                                    'amount_usd': amount_usd,
                                    'price': float(item.get('priceUSD', 0)),
                                    'timestamp': item.get('timestamp', 0),
                                    'dex_url': item.get('txUrl', '')
                                })
                        except (ValueError, TypeError) as e:
                            continue
                    
                    logging.info(f"✅ BSC: знайдено {len(trades)} великих угод")
                    return trades
                    
        except Exception as e:
            logging.error(f"❌ Помилка DexView API: {e}")
            return []
    
    @staticmethod
    async def get_token_info(chain: str, token_address: str) -> Optional[Dict]:
        """Отримання детальної інформації про токен"""
        try:
            if chain == "solana":
                return await DexDataClient._get_solana_token_info(token_address)
            elif chain == "bsc":
                return await DexDataClient._get_bsc_token_info(token_address)
            else:
                return None
                
        except Exception as e:
            logging.error(f"❌ Помилка отримання інфо токена: {e}")
            return None
    
    @staticmethod
    async def _get_solana_token_info(token_address: str) -> Optional[Dict]:
        """Отримання інформації про токен Solana"""
        try:
            url = f"{Config.BIRDEYE_API_URL}/public/token?address={token_address}"
            headers = {"X-API-KEY": Config.BIRDEYE_API_KEY}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=10) as response:
                    if response.status != 200:
                        return None
                    
                    data = await response.json()
                    token_data = data.get('data', {})
                    
                    return {
                        'symbol': token_data.get('symbol', '').upper(),
                        'name': token_data.get('name', ''),
                        'price': float(token_data.get('price', 0)),
                        'volume_24h': float(token_data.get('volume24h', 0)),
                        'market_cap': float(token_data.get('marketCap', 0)),
                        'liquidity': float(token_data.get('liquidity', 0)),
                        'price_change_24h': float(token_data.get('priceChange24h', 0)),
                        'chain': 'solana'
                    }
                        
        except Exception as e:
            logging.error(f"❌ Помилка отримання інфо токена Solana: {e}")
            return None
    
    @staticmethod
    async def _get_bsc_token_info(token_address: str) -> Optional[Dict]:
        """Отримання інформації про токен BSC"""
        try:
            url = f"{Config.DEXVIEW_API_URL}/v1/token/info?chain=bsc&address={token_address}"
            headers = {"Authorization": f"Bearer {Config.DEXVIEW_API_KEY}"}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=10) as response:
                    if response.status != 200:
                        return None
                    
                    data = await response.json()
                    token_data = data.get('token', {})
                    
                    return {
                        'symbol': token_data.get('symbol', '').upper(),
                        'name': token_data.get('name', ''),
                        'price': float(token_data.get('priceUSD', 0)),
                        'volume_24h': float(token_data.get('volume24hUSD', 0)),
                        'market_cap': float(token_data.get('marketCapUSD', 0)),
                        'liquidity': float(token_data.get('liquidityUSD', 0)),
                        'price_change_24h': float(token_data.get('priceChange24h', 0)),
                        'chain': 'bsc'
                    }
                        
        except Exception as e:
            logging.error(f"❌ Помилка отримання інфо токена BSC: {e}")
            return None

class LBankClient:
    def __init__(self):
        self.api_key = Config.LBANK_API_KEY
        self.secret_key = Config.LBANK_SECRET_KEY
        self.base_url = Config.LBANK_BASE_URL
        
    def _generate_signature(self, params: Dict) -> str:
        """Генерація підпису для LBank API"""
        query_string = '&'.join([f"{k}={v}" for k, v in sorted(params.items())])
        signature = hmac.new(
            self.secret_key.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).digest()
        return base64.b64encode(signature).decode('utf-8')
    
    async def get_ticker_price(self, symbol: str) -> Optional[float]:
        """Отримання поточної ціни з LBank"""
        try:
            lbank_symbol = f"{symbol}_usdt"
            url = f"{self.base_url}/v2/ticker.do?symbol={lbank_symbol}"
            logging.info(f"🔗 Перевіряю ціну на LBank: {lbank_symbol}")
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    if response.status != 200:
                        logging.warning(f"❌ LBank не відповідає: статус {response.status}")
                        return None
                    
                    data = await response.json()
                    if data.get('result') and 'ticker' in data:
                        price = float(data['ticker'][0])
                        logging.info(f"💰 LBank ціна {symbol}: ${price:.6f}")
                        return price
                    else:
                        logging.warning(f"❌ Токен {symbol} не знайдено на LBank")
                        return None
                        
        except Exception as e:
            logging.error(f"❌ Помилка отримання ціни {symbol}: {e}")
            return None
    
    async def place_limit_order(self, symbol: str, price: float, amount: float) -> Dict:
        """Розміщення лімітного ордера"""
        try:
            lbank_symbol = f"{symbol}_usdt"
            params = {
                'api_key': self.api_key,
                'symbol': lbank_symbol,
                'type': 'buy',
                'price': str(price),
                'amount': str(amount),
                'timestamp': str(int(time.time() * 1000))
            }
            
            params['sign'] = self._generate_signature(params)
            
            logging.info(f"🛒 Розміщую ордер на LBank: {symbol} {amount} по ${price:.6f}")
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/v2/create_order.do",
                    data=params,
                    timeout=10
                ) as response:
                    result = await response.json()
                    if result.get('result'):
                        logging.info(f"✅ Ордер успішно розміщено: {result.get('order_id')}")
                    else:
                        logging.error(f"❌ Помилка розміщення ордера: {result}")
                    return result
                    
        except Exception as e:
            logging.error(f"❌ Помилка розміщення ордера: {e}")
            return {'error': str(e)}

class ArbitrageBot:
    def __init__(self):
        self.lbank_client = LBankClient()
        self.dex_client = DexDataClient()  # Змінено на новий клієнт
        self.token_filter = TokenFilter()
        self.last_processed = {}
        self.is_scanning = False
        
    async def check_apis_connection(self):
        """Перевірка з'єднання з API"""
        try:
            logging.info("🔗 Перевіряю з'єднання з API...")
            
            # Перевірка альтернативних API
            test_trades = await self.dex_client.get_recent_trades("solana", 2)
            logging.info(f"✅ Альтернативні API доступні. Знайдено угод: {len(test_trades)}")
            
            # Перевірка LBank
            test_price = await self.lbank_client.get_ticker_price("BTC")
            if test_price:
                logging.info(f"✅ LBank доступний. Ціна BTC: ${test_price:.2f}")
            else:
                logging.warning("⚠️ LBank не відповідає")
                
            # Перевірка Telegram
            if telegram_client.send_message("✅ Бот успішно запущено та перевіряє зв'язок"):
                logging.info("✅ Telegram доступний")
            else:
                logging.warning("⚠️ Telegram не доступний")
                
        except Exception as e:
            logging.error(f"❌ Помилка перевірки з'єднання: {e}")
    
    # Решта коду залишається незмінною...
    # [Інші методи залишаються без змін]

# Глобальний екземпляр бота
arbitrage_bot = ArbitrageBot()

# Решта коду залишається незмінною...
# [Функції send_telegram_command, run_scanner, Flask сервер залишаються без змін]

if __name__ == "__main__":
    logging.info("🚀 Arbitrage Bot з альтернативними API запущено!")
    logging.info(f"📡 Мережі: {Config.ALLOWED_CHAINS}")
    logging.info(f"💰 Мін. угода: ${Config.MIN_TRADE_AMOUNT}")
    logging.info(f"💵 Обсяг ордера: ${Config.ORDER_VOLUME}")
    logging.info(f"📊 Мін. капіталізація: ${Config.MIN_MARKET_CAP:,}")
    
    # Очищаємо всі попередні webhook
    try:
        logging.info("🧹 Очищаю старі webhook...")
        requests.get(f"https://api.telegram.org/bot{Config.TELEGRAM_TOKEN}/deleteWebhook", timeout=5)
        time.sleep(2)
    except Exception as e:
        logging.warning(f"⚠️ Не вдалося очистити webhook: {e}")
    
    # Запускаємо сканер
    logging.info("📡 Запускаю сканер угод...")
    scanner_thread = threading.Thread(target=run_scanner, daemon=True)
    scanner_thread.start()
    
    # Простий веб-сервер для команд
    from flask import Flask, request
    app = Flask(__name__)
    
    @app.route('/')
    def home():
        return "🤖 Arbitrage Bot is Running! Use /command endpoint to send commands.", 200
    
    @app.route('/command', methods=['POST'])
    def command_handler():
        try:
            data = request.get_json()
            command = data.get('command', '')
            chat_id = data.get('chat_id', Config.TELEGRAM_CHAT_ID)
            
            if command:
                send_telegram_command(command, chat_id)
                return {'status': 'success', 'message': 'Command processed'}
            else:
                return {'status': 'error', 'message': 'No command provided'}, 400
                
        except Exception as e:
            return {'status': 'error', 'message': str(e)}, 500
    
    # Запускаємо Flask
    port = int(os.environ.get('PORT', 10000))
    logging.info(f"🌐 Запускаю веб-сервер на порті {port}")
    app.run(host='0.0.0.0', port=port, debug=False)