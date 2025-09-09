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
    
    # DexScreener API
    DEXSCREENER_API = "https://api.dexscreener.com/latest/dex"
    
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

class DexScreenerClient:
    """Клієнт для роботи з DexScreener API"""
    
    @staticmethod
    async def get_recent_trades(chain: str, limit: int = 20) -> List[Dict]:
        """Отримання останніх угод з DexScreener"""
        try:
            url = f"{Config.DEXSCREENER_API}/transactions/{chain}"
            logging.info(f"🔗 Запит до DexScreener: {chain}")
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=15) as response:
                    if response.status != 200:
                        logging.warning(f"❌ DexScreener {chain} статус: {response.status}")
                        return []
                    
                    data = await response.json()
                    trades = data.get('transactions', [])
                    logging.info(f"📊 {chain}: знайдено {len(trades)} угод")
                    
                    large_trades = []
                    for trade in trades[:limit]:
                        try:
                            amount_usd = float(trade.get('volumeUsd', 0))
                            if amount_usd >= Config.MIN_TRADE_AMOUNT:
                                large_trades.append({
                                    'chain': chain,
                                    'token_address': trade.get('baseToken', {}).get('address', ''),
                                    'token_symbol': trade.get('baseToken', {}).get('symbol', '').upper(),
                                    'amount_usd': amount_usd,
                                    'price': float(trade.get('priceUsd', 0)),
                                    'timestamp': trade.get('timestamp', 0),
                                    'dex_url': trade.get('url', '')
                                })
                                logging.info(f"💎 Велика угода на {chain}: {trade.get('baseToken', {}).get('symbol', '').upper()} - ${amount_usd:,.2f}")
                        except (ValueError, TypeError) as e:
                            logging.debug(f"Помилка обробки угоди: {e}")
                            continue
                    
                    logging.info(f"✅ {chain}: відфільтровано {len(large_trades)} великих угод")
                    return large_trades
                    
        except Exception as e:
            logging.error(f"❌ Помилка отримання угод з {chain}: {e}")
            return []
    
    @staticmethod
    async def get_token_info(chain: str, token_address: str) -> Optional[Dict]:
        """Отримання детальної інформації про токен"""
        try:
            url = f"{Config.DEXSCREENER_API}/tokens/{chain}/{token_address}"
            logging.info(f"🔗 Отримую інфо токена: {chain}/{token_address}")
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    if response.status != 200:
                        logging.warning(f"❌ Не вдалося отримати інфо токена: статус {response.status}")
                        return None
                    
                    data = await response.json()
                    pair = data.get('pair', {})
                    
                    if pair:
                        token_info = {
                            'symbol': pair.get('baseToken', {}).get('symbol', '').upper(),
                            'name': pair.get('baseToken', {}).get('name', ''),
                            'price': float(pair.get('priceUsd', 0)),
                            'volume_24h': float(pair.get('volume', {}).get('h24', 0)),
                            'market_cap': float(pair.get('marketCap', 0)),
                            'liquidity': float(pair.get('liquidity', {}).get('usd', 0)),
                            'price_change_24h': float(pair.get('priceChange', {}).get('h24', 0)),
                            'chain': chain
                        }
                        logging.info(f"📋 Інфо токена {token_info['symbol']}: капіталізація ${token_info['market_cap']:,.0f}, обсяг ${token_info['volume_24h']:,.0f}")
                        return token_info
                    else:
                        logging.warning("❌ Не знайдено даних про токен")
                        return None
                        
        except Exception as e:
            logging.error(f"❌ Помилка отримання інфо токена: {e}")
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
        self.dex_client = DexScreenerClient()
        self.token_filter = TokenFilter()
        self.last_processed = {}
        self.is_scanning = False
        
    async def check_apis_connection(self):
        """Перевірка з'єднання з API"""
        try:
            logging.info("🔗 Перевіряю з'єднання з API...")
            
            # Перевірка DexScreener
            test_trades = await self.dex_client.get_recent_trades("solana", 2)
            logging.info(f"✅ DexScreener доступний. Знайдено угод: {len(test_trades)}")
            
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
        logging.info("🔍 Запуск автосканування угод на Solana та BSC")
        
        # Перевірка з'єднання
        await self.check_apis_connection()
        
        # Відправляємо повідомлення про запуск
        telegram_client.send_message(
            "🤖 Бот запущено! Початок сканування угод...",
            parse_mode="Markdown"
        )
        
        scan_count = 0
        while self.is_scanning:
            try:
                scan_count += 1
                logging.info(f"🔄 Сканування #{scan_count} - перевіряю угоди...")
                
                # Отримуємо угоди з обох мереж
                tasks = [
                    self.dex_client.get_recent_trades("solana"),
                    self.dex_client.get_recent_trades("bsc")
                ]
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                all_trades = []
                
                for i, result in enumerate(results):
                    chain_name = ["Solana", "BSC"][i]
                    if isinstance(result, list):
                        logging.info(f"📊 {chain_name}: знайдено {len(result)} угод")
                        all_trades.extend(result)
                    elif isinstance(result, Exception):
                        logging.error(f"❌ {chain_name}: помилка - {result}")
                
                logging.info(f"📈 Всього знайдено угод: {len(all_trades)}")
                
                # Обробляємо кожну велику угоду
                processed_count = 0
                for trade in all_trades:
                    if await self.process_trade_signal(trade):
                        processed_count += 1
                
                logging.info(f"✅ Оброблено угод: {processed_count}/{len(all_trades)}")
                
                # Пауза між скануваннями
                logging.info("⏸️ Очікую 15 секунд...")
                await asyncio.sleep(15)
                
            except Exception as e:
                logging.error(f"🔥 Критична помилка: {e}")
                await asyncio.sleep(30)
    
    async def stop_auto_scan(self):
        """Зупинка автоматичного сканування"""
        self.is_scanning = False
        logging.info("⏹️ Зупинка автосканування")
        telegram_client.send_message("⏹️ Сканування зупинено!")
    
    async def process_trade_signal(self, trade: Dict) -> bool:
        """Обробка сигналу про велику угоду"""
        try:
            token_address = trade['token_address']
            chain = trade['chain']
            symbol = trade['token_symbol']
            
            # Уникаємо дублювання обробки
            trade_key = f"{chain}_{token_address}"
            current_time = time.time()
            
            if trade_key in self.last_processed:
                if current_time - self.last_processed[trade_key] < 300:
                    logging.debug(f"⏭️ Пропускаємо дубль угоди: {symbol}")
                    return False
            
            self.last_processed[trade_key] = current_time
            
            logging.info(f"🔍 Обробляю угоду: {symbol} на {chain} за ${trade['amount_usd']:,.2f}")
            
            # Отримуємо детальну інформацію про токен
            token_info = await self.dex_client.get_token_info(chain, token_address)
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
            
            logging.info(f"✅ Угода успішно оброблена: {symbol}")
            return True
            
        except Exception as e:
            logging.error(f"❌ Помилка обробки угоди: {e}")
            return False
    
    async def send_trade_notification(self, trade: Dict, token_info: Dict, 
                                    market_price: float, order_price: float, order_result: Dict):
        """Відправка сповіщення про угоду"""
        try:
            success = order_result.get('result', False)
            
            message = (
                f"🚀 *Велика угода виявлена!*\n\n"
                f"📊 *Деталі угоди:*\n"
                f"• Токен: `{token_info['symbol']}`\n"
                f"• Мережа: `{trade['chain'].upper()}`\n"
                f"• Сума: `${trade['amount_usd']:,.2f}`\n"
                f"• Ціна: `${trade['price']:.6f}`\n\n"
                f"📈 *Ринкові дані:*\n"
                f"• Капіталізація: `${token_info['market_cap']:,.0f}`\n"
                f"• Обсяг 24h: `${token_info['volume_24h']:,.0f}`\n"
                f"• LBank цена: `${market_price:.6f}`\n"
                f"• Ціна ордера: `${order_price:.6f}`\n"
                f"• Premium: `{Config.PRICE_PREMIUM*100:.2f}%`\n\n"
                f"🛒 *Ордер на LBank:*\n"
                f"• Статус: `{'✅ Успішно' if success else '❌ Помилка'}`\n"
                f"• Час: `{datetime.now().strftime('%H:%M:%S')}`\n\n"
                f"🔗 *[DEX Screener]({trade['dex_url']})*"
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
                "🤖 *DexScreener/LBank Arbitrage Bot*\n\n"
                "Цей бот сканує великі угоди на Solana та BSC:\n"
                "• Моніторить DexScreener для угод >$3000\n"
                "• Фільтрує токени за критеріями\n"
                "• Автоматично купує на LBank\n"
                "• Ціна: +0.01% до ринкової\n\n"
                "⚙️ *Фільтри:*\n"
                f"• Капіталізація: ${Config.MIN_MARKET_CAP:,} - ${Config.MAX_MARKET_CAP:,}\n"
                f"• Обсяг: >${Config.MIN_VOLUME:,}/24h\n"
                f"• Мережі: {', '.join(Config.ALLOWED_CHAINS).upper()}\n"
                f"• Мін. угода: ${Config.MIN_TRADE_AMOUNT}\n\n"
                "📊 *Команди:*\n"
                "/scan_start - Почати сканування\n"
                "/scan_stop - Зупинити сканування\n"
                "/status - Статус бота"
            )
            telegram_client.send_message(help_text, parse_mode="Markdown")
            
        elif command == '/scan_start':
            if not arbitrage_bot.is_scanning:
                def start():
                    asyncio.run(arbitrage_bot.start_auto_scan())
                thread = threading.Thread(target=start, daemon=True)
                thread.start()
                telegram_client.send_message("🔍 Сканування запущено! Шукаю великі угоди...")
            else:
                telegram_client.send_message("⚠️ Сканування вже запущено!")
                
        elif command == '/scan_stop':
            async def stop():
                await arbitrage_bot.stop_auto_scan()
            asyncio.run(stop())
            telegram_client.send_message("⏹️ Сканування зупинено!")
            
        elif command == '/status':
            status_text = (
                "📊 *Статус бота:*\n\n"
                f"• Сканування: {'✅ Активне' if arbitrage_bot.is_scanning else '❌ Неактивне'}\n"
                f"• Мережі: {', '.join(Config.ALLOWED_CHAINS).upper()}\n"
                f"• Оброблено угод: {len(arbitrage_bot.last_processed)}\n"
                f"• Мін. сума угоди: ${Config.MIN_TRADE_AMOUNT}\n"
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
    asyncio.run(start())

if __name__ == "__main__":
    logging.info("🚀 Arbitrage Bot з DexScreener запущено!")
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