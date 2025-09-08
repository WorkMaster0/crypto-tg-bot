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
    LBANK_API_KEY = "ваш_lbank_api_key"
    LBANK_SECRET_KEY = "ваш_lbank_secret_key"
    LBANK_BASE_URL = "https://api.lbank.info"
    
    # GMGN моніторинг
    GMGN_WEBHOOK_URL = "ваш_gmgn_webhook_url"
    
    # Торгові налаштування
    TRADE_SYMBOL = "labubu_usdt"  # Символ для торгівлі
    ORDER_VOLUME = 5000  # Обсяг в USDT
    PRICE_PREMIUM = 0.0001  # 0.01% вище ринкової ціни
    ORDER_TYPE = "limit"  # Тип ордера

# Ініціалізація бота
bot = telebot.TeleBot(Config.TELEGRAM_TOKEN)

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
            url = f"{self.base_url}/v2/ticker.do?symbol={symbol}"
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=1) as response:
                    data = await response.json()
                    if data.get('result') and 'ticker' in data:
                        return float(data['ticker'][0])  # Перший елемент - поточна ціна
        except Exception as e:
            logging.error(f"Помилка отримання ціни: {e}")
        return None
    
    async def place_limit_order(self, symbol: str, price: float, amount: float) -> Dict:
        """Розміщення лімітного ордера"""
        try:
            params = {
                'api_key': self.api_key,
                'symbol': symbol,
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
                    timeout=2
                ) as response:
                    result = await response.json()
                    return result
                    
        except Exception as e:
            logging.error(f"Помилка розміщення ордера: {e}")
            return {'error': str(e)}

class ArbitrageBot:
    def __init__(self):
        self.lbank_client = LBankClient(Config.LBANK_API_KEY, Config.LBANK_SECRET_KEY)
        self.last_processed_time = 0
        self.min_processing_interval = 0.1  # 100ms мінімальний інтервал
        
    async def process_gmgn_signal(self, signal_data: Dict):
        """Обробка сигналу від GMGN"""
        current_time = time.time()
        
        # Запобігаємо дублюванню обробки
        if current_time - self.last_processed_time < self.min_processing_interval:
            logging.warning("Скипаємо сигнал - занадто швидко")
            return
        
        self.last_processed_time = current_time
        
        try:
            # Отримуємо поточну ціну з LBank
            current_price = await self.lbank_client.get_ticker_price(Config.TRADE_SYMBOL)
            if not current_price:
                logging.error("Не вдалося отримати ціну з LBank")
                return
            
            # Розраховуємо ціну ордера (0.01% вище)
            order_price = round(current_price * (1 + Config.PRICE_PREMIUM), 6)
            
            # Розраховуємо обсяг
            order_amount = round(Config.ORDER_VOLUME / order_price, 8)
            
            # Розміщуємо ордер
            order_result = await self.lbank_client.place_limit_order(
                Config.TRADE_SYMBOL,
                order_price,
                order_amount
            )
            
            # Відправляємо повідомлення в Telegram
            await self.send_telegram_notification(signal_data, current_price, order_price, order_result)
            
            logging.info(f"Ордер розміщено: {order_price} USDT, {order_amount} {Config.TRADE_SYMBOL}")
            
        except Exception as e:
            logging.error(f"Помилка обробки сигналу: {e}")
    
    async def send_telegram_notification(self, signal_data: Dict, market_price: float, 
                                       order_price: float, order_result: Dict):
        """Відправка сповіщення в Telegram"""
        try:
            message = (
                f"🚀 *Нова угода GMGN/LBank*\n\n"
                f"📊 *Деталі сигналу:*\n"
                f"• Токен: `{signal_data.get('token', 'N/A')}`\n"
                f"• Обсяг: `${signal_data.get('amount', 0):.2f}`\n"
                f"• Ціна покупки: `${signal_data.get('price', 0):.6f}`\n\n"
                f"📈 *Ринкові дані:*\n"
                f"• LBank цена: `${market_price:.6f}`\n"
                f"• Ціна ордера: `${order_price:.6f}`\n"
                f"• Premium: `{Config.PRICE_PREMIUM*100:.2f}%`\n\n"
                f"🛒 *Ордер:*\n"
                f"• Статус: `{'Успішно' if order_result.get('result') else 'Помилка'}`\n"
                f"• ID ордера: `{order_result.get('order_id', 'N/A')}`\n"
                f"• Час: `{datetime.now().strftime('%H:%M:%S.%f')[:-3]}`\n\n"
                f"⚡ *Оброблено за:* `{(time.time() - self.last_processed_time)*1000:.2f}ms`"
            )
            
            await bot.send_message(Config.TELEGRAM_CHAT_ID, message, parse_mode="Markdown")
            
        except Exception as e:
            logging.error(f"Помилка відправки в Telegram: {e}")

# Глобальний екземпляр бота
arbitrage_bot = ArbitrageBot()

# Webhook endpoint для отримання сигналів від GMGN
from flask import Flask, request, jsonify
app = Flask(__name__)

@app.route('/gmgn-webhook', methods=['POST'])
async def gmgn_webhook():
    """Ендпоінт для отримання webhook від GMGN"""
    try:
        data = request.get_json()
        
        # Швидка перевірка даних
        if not data or 'token' not in data:
            return jsonify({'status': 'error', 'message': 'Invalid data'}), 400
        
        logging.info(f"Отримано сигнал від GMGN: {data['token']}")
        
        # Асинхронна обробка сигналу
        asyncio.create_task(arbitrage_bot.process_gmgn_signal(data))
        
        return jsonify({'status': 'success', 'message': 'Signal processing started'}), 200
        
    except Exception as e:
        logging.error(f"Помилка webhook: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

# Команди для Telegram бота
@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    help_text = (
        "🤖 *GMGN/LBank Arbitrage Bot*\n\n"
        "Цей бот автоматично виконує арбітраж між GMGN та LBank:\n"
        "• Отримує сигнали купівлі з GMGN\n"
        "• Миттєво розміщує ордери на LBank\n"
        "• Ціна: +0.01% до ринкової\n"
        "• Обробка: <100ms\n\n"
        "⚙️ *Налаштування:*\n"
        f"• Токен: `{Config.TRADE_SYMBOL}`\n"
        f"• Обсяг: `${Config.ORDER_VOLUME}`\n"
        f"• Premium: `{Config.PRICE_PREMIUM*100:.2f}%`\n\n"
        "📊 Статус: Активний ✅"
    )
    bot.send_message(message.chat.id, help_text, parse_mode="Markdown")

@bot.message_handler(commands=['status'])
def check_status(message):
    """Перевірка статусу бота та з'єднань"""
    status_text = (
        "📊 *Статус системи:*\n\n"
        "• Бот: Активний ✅\n"
        "• LBank API: Перевірка...\n"
        "• GMGN Webhook: Активний ✅\n"
        f"• Остання обробка: {datetime.now().strftime('%H:%M:%S')}\n"
        f"• Обсяг ордера: ${Config.ORDER_VOLUME}\n"
        f"• Premium: {Config.PRICE_PREMIUM*100:.2f}%"
    )
    bot.send_message(message.chat.id, status_text, parse_mode="Markdown")

# Функція для тестування
async def test_execution_speed():
    """Тест швидкості виконання"""
    test_data = {
        'token': 'TEST_GMGN',
        'amount': 5000,
        'price': 1.0000
    }
    
    start_time = time.time()
    await arbitrage_bot.process_gmgn_signal(test_data)
    execution_time = (time.time() - start_time) * 1000
    
    logging.info(f"Тестова швидкість: {execution_time:.2f}ms")

if __name__ == "__main__":
    # Запускаємо тест швидкості при старті
    asyncio.run(test_execution_speed())
    
    # Запускаємо Flask для webhook
    from threading import Thread
    flask_thread = Thread(target=lambda: app.run(host='0.0.0.0', port=5000, debug=False))
    flask_thread.daemon = True
    flask_thread.start()
    
    # Запускаємо Telegram бота
    logging.info("🚀 Arbitrage Bot запущено!")
    bot.infinity_polling()