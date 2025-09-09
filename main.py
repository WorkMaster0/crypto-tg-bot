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
import json

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
    
    # Безкоштовні API URLs
    GEKKOTERM_API_URL = "https://api.geckoterminal.com/api/v2"
    
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
    
    # Обмеження запитів
    REQUEST_DELAY = 2  # Затримка між запитами в секундах
    MAX_REQUESTS_PER_MINUTE = 30  # Максимальна кількість запитів на хвилину

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
            
            response = requests.post(url, data=payload, timeout=10)
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
    """Клієнт для роботи з безкоштовними API"""
    
    def __init__(self):
        self.last_request_time = 0
        self.request_count = 0
        self.minute_start = time.time()
    
    async def _rate_limit(self):
        """Обмеження кількості запитів"""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        
        # Перевіряємо ліміт запитів на хвилину
        if current_time - self.minute_start >= 60:
            self.request_count = 0
            self.minute_start = current_time
        
        if self.request_count >= Config.MAX_REQUESTS_PER_MINUTE:
            wait_time = 60 - (current_time - self.minute_start)
            logging.warning(f"⏳ Досягнуто ліміт запитів. Очікую {wait_time:.1f} секунд")
            await asyncio.sleep(wait_time)
            self.request_count = 0
            self.minute_start = time.time()
        
        # Затримка між запитами
        if elapsed < Config.REQUEST_DELAY:
            await asyncio.sleep(Config.REQUEST_DELAY - elapsed)
        
        self.last_request_time = time.time()
        self.request_count += 1
    
    async def get_recent_trades(self, chain: str, limit: int = 10) -> List[Dict]:
        """Отримання останніх угод з безкоштовних API"""
        try:
            await self._rate_limit()
            
            if chain == "solana":
                return await self._get_network_trades("solana", limit)
            elif chain == "bsc":
                return await self._get_network_trades("bsc", limit)
            else:
                return []
                
        except Exception as e:
            logging.error(f"❌ Помилка отримання угод з {chain}: {e}")
            return []
    
    async def _get_network_trades(self, network: str, limit: int) -> List[Dict]:
        """Отримання угод з мережі через GeckoTerminal API"""
        try:
            url = f"{Config.GEKKOTERM_API_URL}/networks/{network}/pools"
            
            logging.info(f"🔗 Запит до GeckoTerminal API для {network}")
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=15) as response:
                    if response.status != 200:
                        logging.warning(f"❌ GeckoTerminal API статус: {response.status}")
                        return []
                    
                    data = await response.json()
                    trades = []
                    
                    for pool in data.get('data', [])[:limit]:
                        try:
                            attributes = pool.get('attributes', {})
                            volume_24h = float(attributes.get('volume_usd', {}).get('h24', 0))
                            
                            if volume_24h >= Config.MIN_TRADE_AMOUNT:
                                base_token = pool.get('relationships', {}).get('base_token', {}).get('data', {})
                                token_address = base_token.get('address', '') if base_token else ''
                                
                                trades.append({
                                    'chain': network,
                                    'token_address': token_address,
                                    'token_symbol': attributes.get('base_token_symbol', '').upper(),
                                    'amount_usd': volume_24h,
                                    'price': float(attributes.get('price_usd', 0)),
                                    'timestamp': int(time.time()),
                                    'dex_url': f"https://www.geckoterminal.com/{network}/pools/{pool.get('id', '')}",
                                    'pool_id': pool.get('id', '')
                                })
                        except (ValueError, TypeError) as e:
                            logging.debug(f"Помилка обробки пулу: {e}")
                            continue
                    
                    logging.info(f"✅ {network.upper()}: знайдено {len(trades)} великих пулів")
                    return trades
                    
        except Exception as e:
            logging.error(f"❌ Помилка GeckoTerminal API для {network}: {e}")
            return []
    
    async def get_token_info(self, chain: str, token_address: str, pool_id: str = None) -> Optional[Dict]:
        """Отримання детальної інформації про токен через GeckoTerminal"""
        try:
            await self._rate_limit()
            
            # Спочатку пробуємо отримати інформацію через pool ID
            if pool_id:
                pool_info = await self._get_pool_info(pool_id)
                if pool_info:
                    return pool_info
            
            # Якщо не вдалося, пробуємо через token address
            url = f"{Config.GEKKOTERM_API_URL}/networks/{chain}/tokens/{token_address}"
            
            logging.info(f"🔗 Отримую інфо токена через GeckoTerminal: {chain}/{token_address}")
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    if response.status != 200:
                        logging.warning(f"❌ GeckoTerminal статус: {response.status}")
                        return None
                    
                    data = await response.json()
                    token_data = data.get('data', {}).get('attributes', {})
                    
                    return {
                        'symbol': token_data.get('symbol', '').upper(),
                        'name': token_data.get('name', ''),
                        'price': float(token_data.get('price_usd', 0)),
                        'volume_24h': float(token_data.get('volume_usd', {}).get('h24', 0)),
                        'market_cap': float(token_data.get('fdv_usd', 0)),
                        'liquidity': float(token_data.get('reserve_in_usd', 0)),
                        'price_change_24h': float(token_data.get('price_change_percentage', {}).get('h24', 0)),
                        'chain': chain
                    }
                        
        except Exception as e:
            logging.error(f"❌ Помилка отримання інфо токена: {e}")
            return None
    
    async def _get_pool_info(self, pool_id: str) -> Optional[Dict]:
        """Отримання інформації про пул через GeckoTerminal"""
        try:
            await self._rate_limit()
            
            # Розбираємо pool_id на мережу та адресу
            if '_' in pool_id:
                network, address = pool_id.split('_', 1)
                url = f"{Config.GEKKOTERM_API_URL}/networks/{network}/pools/{address}"
            else:
                return None
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    if response.status != 200:
                        return None
                    
                    data = await response.json()
                    pool_data = data.get('data', {}).get('attributes', {})
                    
                    return {
                        'symbol': pool_data.get('base_token_symbol', '').upper(),
                        'name': pool_data.get('base_token_name', ''),
                        'price': float(pool_data.get('price_usd', 0)),
                        'volume_24h': float(pool_data.get('volume_usd', {}).get('h24', 0)),
                        'market_cap': float(pool_data.get('fdv_usd', 0)),
                        'liquidity': float(pool_data.get('reserve_in_usd', 0)),
                        'price_change_24h': float(pool_data.get('price_change_percentage', {}).get('h24', 0)),
                        'chain': network
                    }
                        
        except Exception as e:
            logging.error(f"Помилка отримання інфо пулу: {e}")
            return None

class LBankClient:
    def __init__(self):
        self.api_key = Config.LBANK_API_KEY
        self.secret_key = Config.LBANK_SECRET_KEY
        self.base_url = Config.LBANK_BASE_URL
        
    def _generate_signature(self, params: Dict) -> str:
        """Генерація підпису для LBank API"""
        try:
            # Формуємо рядок для підпису
            query_string = '&'.join([f"{k}={v}" for k, v in sorted(params.items()) if k != 'sign'])
            signature = hmac.new(
                self.secret_key.encode('utf-8'),
                query_string.encode('utf-8'),
                hashlib.sha256
            ).digest()
            return base64.b64encode(signature).decode('utf-8')
        except Exception as e:
            logging.error(f"Помилка генерації підпису: {e}")
            return ""
    
    async def get_ticker_price(self, symbol: str) -> Optional[float]:
        """Отримання поточної ціни з LBank"""
        try:
            lbank_symbol = f"{symbol.lower()}_usdt"
            url = f"{self.base_url}/v2/ticker.do?symbol={lbank_symbol}"
            logging.info(f"🔗 Перевіряю ціну на LBank: {lbank_symbol}")
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    if response.status != 200:
                        logging.warning(f"❌ LBank не відповідає: статус {response.status}")
                        return None
                    
                    data = await response.json()
                    if data.get('result') and 'data' in data and data['data']:
                        ticker_data = data['data'][0]
                        price = float(ticker_data.get('ticker', {}).get('latest', 0))
                        logging.info(f"💰 LBank ціна {symbol}: ${price:.6f}")
                        return price
                    else:
                        logging.warning(f"❌ Токен {symbol} не знайдено на LBank: {data}")
                        return None
                        
        except Exception as e:
            logging.error(f"❌ Помилка отримання ціни {symbol}: {e}")
            return None
    
    async def place_limit_order(self, symbol: str, price: float, amount: float) -> Dict:
        """Розміщення лімітного ордера"""
        try:
            lbank_symbol = f"{symbol.lower()}_usdt"
            timestamp = str(int(time.time() * 1000))
            
            params = {
                'api_key': self.api_key,
                'symbol': lbank_symbol,
                'type': 'buy',
                'price': str(price),
                'amount': str(amount),
                'timestamp': timestamp
            }
            
            params['sign'] = self._generate_signature(params)
            
            logging.info(f"🛒 Розміщую ордер на LBank: {symbol} {amount} по ${price:.6f}")
            
            headers = {
                'Content-Type': 'application/x-www-form-urlencoded'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/v2/create_order.do",
                    data=params,
                    headers=headers,
                    timeout=15
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
        self.dex_client = DexDataClient()
        self.token_filter = TokenFilter()
        self.last_processed = {}
        self.is_scanning = False
        
    async def check_apis_connection(self):
        """Перевірка з'єднання з API"""
        try:
            logging.info("🔗 Перевіряю з'єднання з API...")
            
            # Перевірка GeckoTerminal API
            test_trades = await self.dex_client.get_recent_trades("solana", 2)
            logging.info(f"✅ GeckoTerminal API доступний. Знайдено пулів: {len(test_trades)}")
            
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
    
    async def start_auto_scan(self):
        """Запуск автоматичного сканування угод"""
        self.is_scanning = True
        logging.info("🔍 Запуск автосканування пулів на Solana та BSC")
        
        # Перевірка з'єднання
        await self.check_apis_connection()
        
        # Відправляємо повідомлення про запуск
        telegram_client.send_message(
            "🤖 Бот запущено! Початок сканування пулів...",
            parse_mode="Markdown"
        )
        
        scan_count = 0
        while self.is_scanning:
            try:
                scan_count += 1
                logging.info(f"🔄 Сканування #{scan_count} - перевіряю пули...")
                
                # Отримуємо пули з обох мереж
                tasks = [
                    self.dex_client.get_recent_trades("solana", 5),
                    self.dex_client.get_recent_trades("bsc", 5)
                ]
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                all_trades = []
                
                for i, result in enumerate(results):
                    chain_name = ["Solana", "BSC"][i]
                    if isinstance(result, list):
                        logging.info(f"📊 {chain_name}: знайдено {len(result)} пулів")
                        all_trades.extend(result)
                    elif isinstance(result, Exception):
                        logging.error(f"❌ {chain_name}: помилка - {result}")
                
                logging.info(f"📈 Всього знайдено пулів: {len(all_trades)}")
                
                # Обробляємо кожен великий пул
                processed_count = 0
                for trade in all_trades:
                    if await self.process_trade_signal(trade):
                        processed_count += 1
                    # Додаємо затримку між обробкою пулів
                    await asyncio.sleep(1)
                
                logging.info(f"✅ Оброблено пулів: {processed_count}/{len(all_trades)}")
                
                # Пауза між скануваннями
                logging.info("⏸️ Очікую 60 секунд...")
                await asyncio.sleep(60)
                
            except Exception as e:
                logging.error(f"🔥 Критична помилка: {e}")
                await asyncio.sleep(60)
    
    async def stop_auto_scan(self):
        """Зупинка автоматичного сканування"""
        self.is_scanning = False
        logging.info("⏹️ Зупинка автосканування")
        telegram_client.send_message("⏹️ Сканування зупинено!")
    
    async def process_trade_signal(self, trade: Dict) -> bool:
    """Обробка сигналу про великий пул"""
    try:
        token_address = trade['token_address']
        chain = trade['chain']
        symbol = trade['token_symbol']
        pool_id = trade.get('pool_id', '')
        
        # Перевіряємо чи символ не пустий
        if not symbol or symbol == 'UNKNOWN' or symbol.strip() == '':
            logging.warning(f"❌ Пропускаємо пул з пустим символом: {token_address}")
            
            # Спробуємо отримати символ через детальну інформацію про пул
            logging.info(f"🔍 Спроба отримати символ через детальну інформацію пулу...")
            token_info = await self.dex_client.get_token_info(chain, token_address, pool_id)
            
            if token_info and token_info.get('symbol'):
                symbol = token_info['symbol']
                trade['token_symbol'] = symbol
                logging.info(f"✅ Символ знайдено через детальну інформацію: {symbol}")
            else:
                logging.warning(f"❌ Не вдалося отримати символ для токена {token_address}")
                return False
        
        # Уникаємо дублювання обробки
        trade_key = f"{chain}_{token_address}"
        current_time = time.time()
        
        if trade_key in self.last_processed:
            if current_time - self.last_processed[trade_key] < 600:  # 10 хвилин
                logging.debug(f"⏭️ Пропускаємо дубль пулу: {symbol}")
                return False
        
        self.last_processed[trade_key] = current_time
        
        logging.info(f"🔍 Обробляю пул: {symbol} на {chain} з обсягом ${trade['amount_usd']:,.2f}")
        
        # Отримуємо детальну інформацію про токен
        token_info = await self.dex_client.get_token_info(chain, token_address, pool_id)
        if not token_info:
            logging.warning(f"❌ Не вдалося отримати інфо для {symbol}")
            return False
        
        # Перевіряємо чи токен підходить під фільтри
        if not self.token_filter.is_token_allowed(token_info):
            logging.info(f"⏭️ Токен {symbol} не пройшов фільтрацію")
            return False
        
        # Перевіряємо чи токен доступний на LBank
        lbank_price = await self.lbank_client.get_ticker_price(token_info['symbol'])
        if not lbank_price:
            logging.warning(f"❌ Токен {symbol} не знайдено на LBank")
            return False
        
        # Розміщуємо ордер
        order_price = round(lbank_price * (1 + Config.PRICE_PREMIUM), 6)
        order_amount = round(Config.ORDER_VOLUME / order_price, 8)
        
        logging.info(f"🛒 Розміщую ордер: {symbol} {order_amount} по ${order_price:.6f}")
        
        order_result = await self.lbank_client.place_limit_order(
            token_info['symbol'],
            order_price,
            order_amount
        )
        
        # Відправляємо сповіщення
        await self.send_trade_notification(trade, token_info, lbank_price, order_price, order_result)
        
        logging.info(f"✅ Пул успішно оброблено: {symbol}")
        return True
        
    except Exception as e:
        logging.error(f"❌ Помилка обробки пулу: {e}")
        return False
    
    async def send_trade_notification(self, trade: Dict, token_info: Dict, 
                                    market_price: float, order_price: float, order_result: Dict):
        """Відправка сповіщення про угоду"""
        try:
            success = order_result.get('result', False)
            
            message = (
                f"🚀 *Великий пул виявлено!*\n\n"
                f"📊 *Деталі пулу:*\n"
                f"• Токен: `{token_info['symbol']}`\n"
                f"• Мережа: `{trade['chain'].upper()}`\n"
                f"• Обсяг 24h: `${trade['amount_usd']:,.2f}`\n"
                f"• Ціна: `${trade['price']:.6f}`\n\n"
                f"📈 *Ринкові дані:*\n"
                f"• Капіталізація: `${token_info['market_cap']:,.0f}`\n"
                f"• Обсяг 24h: `${token_info['volume_24h']:,.0f}`\n"
                f"• LBank ціна: `${market_price:.6f}`\n"
                f"• Ціна ордера: `${order_price:.6f}`\n"
                f"• Premium: `{Config.PRICE_PREMIUM*100:.2f}%`\n\n"
                f"🛒 *Ордер на LBank:*\n"
                f"• Статус: `{'✅ Успішно' if success else '❌ Помилка'}`\n"
                f"• Час: `{datetime.now().strftime('%H:%M:%S')}`\n\n"
                f"🔗 *[GeckoTerminal]({trade['dex_url']})*"
            )
            
            telegram_client.send_message(message, parse_mode="Markdown")
            
        except Exception as e:
            logging.error(f"❌ Помилка відправки сповіщення: {e}")

# Глобальний екземпляр бота
arbitrage_bot = ArbitrageBot()

def send_telegram_command(command: str, chat_id: str = None):
    """Обробка Telegram команд"""
    try:
        if not chat_id:
            chat_id = Config.TELEGRAM_CHAT_ID
            
        if command == '/start' or command == '/help':
            help_text = (
                "🤖 *GeckoTerminal/LBank Arbitrage Bot*\n\n"
                "Цей бот сканує великі пули на Solana та BSC:\n"
                "• Моніторить GeckoTerminal для пулів >$3000\n"
                "• Фільтрує токени за критеріями\n"
                "• Автоматично купує на LBank\n"
                "• Ціна: +0.01% до ринкової\n\n"
                "⚙️ *Фільтри:*\n"
                f"• Капіталізація: ${Config.MIN_MARKET_CAP:,} - ${Config.MAX_MARKET_CAP:,}\n"
                f"• Обсяг: >${Config.MIN_VOLUME:,}/24h\n"
                f"• Мережі: {', '.join(Config.ALLOWED_CHAINS).upper()}\n"
                f"• Мін. обсяг: ${Config.MIN_TRADE_AMOUNT}\n\n"
                "📊 *Команди:*\n"
                "/scan_start - Почати сканування\n"
                "/scan_stop - Зупинити сканування\n"
                "/status - Статус бота"
            )
            telegram_client.send_message(help_text, parse_mode="Markdown")
            
        elif command == '/scan_start':
            if not arbitrage_bot.is_scanning:
                def start():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(arbitrage_bot.start_auto_scan())
                thread = threading.Thread(target=start, daemon=True)
                thread.start()
                telegram_client.send_message("🔍 Сканування запущено! Шукаю великі пули...")
            else:
                telegram_client.send_message("⚠️ Сканування вже запущено!")
                
        elif command == '/scan_stop':
            async def stop():
                await arbitrage_bot.stop_auto_scan()
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(stop())
            telegram_client.send_message("⏹️ Сканування зупинено!")
            
        elif command == '/status':
            status_text = (
                "📊 *Статус бота:*\n\n"
                f"• Сканування: {'✅ Активне' if arbitrage_bot.is_scanning else '❌ Неактивне'}\n"
                f"• Мережі: {', '.join(Config.ALLOWED_CHAINS).upper()}\n"
                f"• Оброблено пулів: {len(arbitrage_bot.last_processed)}\n"
                f"• Мін. обсяг: ${Config.MIN_TRADE_AMOUNT}\n"
                f"• Обсяг ордера: ${Config.ORDER_VOLUME}\n\n"
                f"⏰ *Час:* {datetime.now().strftime('%H:%M:%S')}"
            )
            telegram_client.send_message(status_text, parse_mode="Markdown")
            
    except Exception as e:
        logging.error(f"❌ Помилка обробки команди: {e}")

def run_scanner():
    """Запуск сканера в окремому потоці"""
    async def start():
        await arbitrage_bot.start_auto_scan()
    
    # Створюємо новий event loop для потоку
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(start())

if __name__ == "__main__":
    logging.info("🚀 Arbitrage Bot з GeckoTerminal API запущено!")
    logging.info(f"📡 Мережі: {Config.ALLOWED_CHAINS}")
    logging.info(f"💰 Мін. обсяг: ${Config.MIN_TRADE_AMOUNT}")
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
    logging.info("📡 Запускаю сканер пулів...")
    scanner_thread = threading.Thread(target=run_scanner, daemon=True)
    scanner_thread.start()
    
    # Простий веб-сервер для команд
    from flask import Flask, request, jsonify
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
                return jsonify({'status': 'success', 'message': 'Command processed'})
            else:
                return jsonify({'status': 'error', 'message': 'No command provided'}), 400
                
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 500
    
    # Запускаємо Flask
    port = int(os.environ.get('PORT', 10000))
    logging.info(f"🌐 Запускаю веб-сервер на порті {port}")
    
    # Запускаємо Flask в окремому потоці
    def run_flask():
        app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)
    
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    
    # Чекаємо завершення потоків
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("⏹️ Зупинка бота...")
        arbitrage_bot.is_scanning = False