import requests
import telebot
import asyncio
import aiohttp
import json
import time
import hmac
import hashlib
import base64
from datetime import datetime
import logging
from typing import Dict, List, Optional

# Налаштування логування
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Конфігурація
class Config:
    # Telegram
    TELEGRAM_TOKEN = "8307873209:AAGtH1H4scuomk4VCAJeMSSk0dVtRi15xoU"
    TELEGRAM_CHAT_ID = "6053907025"
    
    # LBank API
    LBANK_API_KEY = "c21d76ba-3252-4372-b7dd-30ac80428363"
    LBANK_SECRET_KEY = "4EC9D3BB56CBD4C42B9E83F0C7B7C1A9"
    LBANK_BASE_URL = "https://api.lbank.info"
    
    # DexScreener API
    DEXSCREENER_API = "https://api.dexscreener.com/latest/dex"
    
    # Торгові налаштування
    ORDER_VOLUME = 5000  # Обсяг в USDT
    PRICE_PREMIUM = 0.0001  # 0.01% вище ринкової ціни
    MIN_TRADE_AMOUNT = 3000  # Мінімальна сума угоди для реакції
    
    # Фільтри токенів
    MIN_MARKET_CAP = 3000000  # 3M$ мінімальна капіталізація
    MAX_MARKET_CAP = 100000000  # 100M$ максимальна капіталізація
    ALLOWED_CHAINS = ["solana", "bsc"]  # Тільки Solana та BSC
    BLACKLIST_TOKENS = ["shitcoin", "scam", "test", "meme", "fake", "pump", "dump"]

# Ініціалізація бота
bot = telebot.TeleBot(Config.TELEGRAM_TOKEN)

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
            if TokenFilter.is_low_quality_token(token_data):
                logging.debug(f"Токен {symbol} низької якості")
                return False
                
            logging.info(f"✅ Токен {symbol} пройшов фільтрацію")
            return True
            
        except Exception as e:
            logging.error(f"Помилка фільтрації токена: {e}")
            return False
    
    @staticmethod
    def is_low_quality_token(token_data: Dict) -> bool:
        """Додаткові перевірки якості токена"""
        # Перевірка на надто високу волатильність
        price_change = abs(token_data.get('price_change_24h', 0))
        if price_change > 50:  # Більше 50% зміна ціни за добу
            return True
            
        # Перевірка на дуже низьку ліквідність
        liquidity = token_data.get('liquidity', 0)
        if liquidity < 100000:  # Менше 100K$ ліквідності
            return True
            
        return False

class DexScreenerClient:
    """Клієнт для роботи з DexScreener API"""
    
    @staticmethod
    async def get_recent_trades(chain: str, limit: int = 20) -> List[Dict]:
        """Отримання останніх угод з DexScreener"""
        try:
            url = f"{Config.DEXSCREENER_API}/transactions/{chain}"
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    data = await response.json()
                    
                    large_trades = []
                    for trade in data.get('transactions', [])[:limit]:
                        amount_usd = float(trade.get('volumeUsd', 0))
                        
                        # Фільтруємо тільки великі угоди
                        if amount_usd >= Config.MIN_TRADE_AMOUNT:
                            large_trades.append({
                                'chain': chain,
                                'token_address': trade.get('baseToken', {}).get('address', ''),
                                'token_symbol': trade.get('baseToken', {}).get('symbol', '').upper(),
                                'amount_usd': amount_usd,
                                'price': float(trade.get('priceUsd', 0)),
                                'timestamp': trade.get('timestamp', 0),
                                'tx_hash': trade.get('txnHash', ''),
                                'dex_url': trade.get('url', '')
                            })
                    
                    return large_trades
                    
        except Exception as e:
            logging.error(f"Помилка отримання угод з {chain}: {e}")
            return []
    
    @staticmethod
    async def get_token_info(chain: str, token_address: str) -> Optional[Dict]:
        """Отримання детальної інформації про токен"""
        try:
            url = f"{Config.DEXSCREENER_API}/tokens/{chain}/{token_address}"
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=5) as response:
                    data = await response.json()
                    
                    pair = data.get('pair', {})
                    if pair:
                        return {
                            'symbol': pair.get('baseToken', {}).get('symbol', '').upper(),
                            'name': pair.get('baseToken', {}).get('name', ''),
                            'price': float(pair.get('priceUsd', 0)),
                            'volume_24h': float(pair.get('volume', {}).get('h24', 0)),
                            'market_cap': float(pair.get('marketCap', 0)),
                            'liquidity': float(pair.get('liquidity', {}).get('usd', 0)),
                            'price_change_24h': float(pair.get('priceChange', {}).get('h24', 0)),
                            'chain': chain
                        }
        except Exception as e:
            logging.error(f"Помилка отримання інфо токена: {e}")
        return None

class LBankClient:
    def __init__(self, api_key: str, secret_key: str):
        self.api_key = api_key
        self.secret_key = secret_key
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
            # Конвертуємо символ у формат LBank (наприклад: SOL_USDT)
            lbank_symbol = f"{symbol.split('_')[0]}_usdt"
            
            url = f"{self.base_url}/v2/ticker.do?symbol={lbank_symbol}"
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=2) as response:
                    data = await response.json()
                    if data.get('result') and 'ticker' in data:
                        return float(data['ticker'][0])
        except Exception as e:
            logging.error(f"Помилка отримання ціни {symbol}: {e}")
        return None
    
    async def place_limit_order(self, symbol: str, price: float, amount: float) -> Dict:
        """Розміщення лімітного ордера"""
        try:
            # Конвертуємо символ у формат LBank
            lbank_symbol = f"{symbol.split('_')[0]}_usdt"
            
            params = {
                'api_key': self.api_key,
                'symbol': lbank_symbol,
                'type': 'buy',
                'price': str(price),
                'amount': str(amount),
                'timestamp': str(int(time.time() * 1000))
            }
            
            params['sign'] = self._generate_signature(params)
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/v2/create_order.do",
                    data=params,
                    timeout=3
                ) as response:
                    result = await response.json()
                    return result
                    
        except Exception as e:
            logging.error(f"Помилка розміщення ордера: {e}")
            return {'error': str(e)}

class ArbitrageBot:
    def __init__(self):
        self.lbank_client = LBankClient(Config.LBANK_API_KEY, Config.LBANK_SECRET_KEY)
        self.dex_client = DexScreenerClient()
        self.token_filter = TokenFilter()
        self.last_processed = {}
        self.is_scanning = False
        
    async def start_auto_scan(self):
        """Запуск автоматичного сканування угод"""
        self.is_scanning = True
        logging.info("🔍 Запуск автосканування угод на Solana та BSC")
        
        while self.is_scanning:
            try:
                # Отримуємо угоди з обох мереж паралельно
                solana_trades, bsc_trades = await asyncio.gather(
                    self.dex_client.get_recent_trades("solana"),
                    self.dex_client.get_recent_trades("bsc")
                )
                
                all_trades = solana_trades + bsc_trades
                
                # Обробляємо кожну велику угоду
                for trade in all_trades:
                    await self.process_trade_signal(trade)
                
                # Пауза між скануваннями
                await asyncio.sleep(5)
                
            except Exception as e:
                logging.error(f"Помилка автосканування: {e}")
                await asyncio.sleep(10)
    
    async def stop_auto_scan(self):
        """Зупинка автоматичного сканування"""
        self.is_scanning = False
        logging.info("⏹️ Зупинка автосканування")
    
    async def process_trade_signal(self, trade: Dict):
        """Обробка сигналу про велику угоду"""
        try:
            token_address = trade['token_address']
            chain = trade['chain']
            
            # Уникаємо дублювання обробки
            trade_key = f"{chain}_{token_address}_{trade['timestamp']}"
            if trade_key in self.last_processed:
                return
            
            self.last_processed[trade_key] = time.time()
            
            # Очищаємо стару історію
            self._clean_processed_history()
            
            logging.info(f"🔍 Обробляю угоду: {trade['token_symbol']} на {chain} за ${trade['amount_usd']}")
            
            # Отримуємо детальну інформацію про токен
            token_info = await self.dex_client.get_token_info(chain, token_address)
            if not token_info:
                logging.warning(f"Не вдалося отримати інфо для {trade['token_symbol']}")
                return
            
            # Перевіряємо чи токен підходить під фільтри
            if not self.token_filter.is_token_allowed(token_info):
                return
            
            # Перевіряємо чи токен доступний на LBank
            lbank_price = await self.lbank_client.get_ticker_price(token_info['symbol'])
            if not lbank_price:
                logging.warning(f"Токен {token_info['symbol']} не знайдено на LBank")
                return
            
            # Розміщуємо ордер
            order_price = round(lbank_price * (1 + Config.PRICE_PREMIUM), 6)
            order_amount = round(Config.ORDER_VOLUME / order_price, 8)
            
            order_result = await self.lbank_client.place_limit_order(
                token_info['symbol'],
                order_price,
                order_amount
            )
            
            # Відправляємо сповіщення
            await self.send_trade_notification(trade, token_info, lbank_price, order_price, order_result)
            
        except Exception as e:
            logging.error(f"Помилка обробки угоди: {e}")
    
    def _clean_processed_history(self):
        """Очищення старої історії оброблених угод"""
        current_time = time.time()
        keys_to_remove = []
        
        for key, timestamp in self.last_processed.items():
            if current_time - timestamp > 300:  # 5 хвилин
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.last_processed[key]
    
    async def send_trade_notification(self, trade: Dict, token_info: Dict, 
                                    market_price: float, order_price: float, order_result: Dict):
        """Відправка сповіщення про угоду"""
        try:
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
                f"• Статус: `{'✅ Успішно' if order_result.get('result') else '❌ Помилка'}`\n"
                f"• ID: `{order_result.get('order_id', 'N/A')}`\n"
                f"• Час: `{datetime.now().strftime('%H:%M:%S')}`\n\n"
                f"🔗 *[DEX Screener]({trade['dex_url']})*"
            )
            
            await bot.send_message(Config.TELEGRAM_CHAT_ID, message, parse_mode="Markdown")
            
        except Exception as e:
            logging.error(f"Помилка відправки сповіщення: {e}")

# Глобальний екземпляр бота
arbitrage_bot = ArbitrageBot()

# Команди для Telegram бота
@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
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
    bot.send_message(message.chat.id, help_text, parse_mode="Markdown")

@bot.message_handler(commands=['scan_start'])
def start_scan(message):
    """Почати автоматичне сканування"""
    async def start():
        if not arbitrage_bot.is_scanning:
            await arbitrage_bot.start_auto_scan()
            bot.send_message(message.chat.id, "🔍 Сканування запущено! Шукаю великі угоди...")
        else:
            bot.send_message(message.chat.id, "⚠️ Сканування вже запущено!")
    
    asyncio.create_task(start())

@bot.message_handler(commands=['scan_stop'])
def stop_scan(message):
    """Зупинити автоматичне сканування"""
    async def stop():
        await arbitrage_bot.stop_auto_scan()
        bot.send_message(message.chat.id, "⏹️ Сканування зупинено!")
    
    asyncio.create_task(stop())

@bot.message_handler(commands=['status'])
def show_status(message):
    """Показати статус бота"""
    status_text = (
        "📊 *Статус бота:*\n\n"
        f"• Сканування: {'✅ Активне' if arbitrage_bot.is_scanning else '❌ Неактивне'}\n"
        f"• Мережі: {', '.join(Config.ALLOWED_CHAINS).upper()}\n"
        f"• Оброблено угод: {len(arbitrage_bot.last_processed)}\n"
        f"• Мін. сума угоди: ${Config.MIN_TRADE_AMOUNT}\n"
        f"• Обсяг ордера: ${Config.ORDER_VOLUME}\n\n"
        f"⏰ *Час:* {datetime.now().strftime('%H:%M:%S')}"
    )
    bot.send_message(message.chat.id, status_text, parse_mode="Markdown")

if __name__ == "__main__":
    # Запускаємо Telegram бота
    logging.info("🚀 Arbitrage Bot з DexScreener запущено!")
    logging.info(f"Мережі: {Config.ALLOWED_CHAINS}")
    logging.info(f"Мін. угода: ${Config.MIN_TRADE_AMOUNT}")
    
    bot.infinity_polling()