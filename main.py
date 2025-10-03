import os
import requests
import logging
from flask import Flask, request, jsonify
from binance.client import Client

# ---------------------------------
# Налаштування
# ---------------------------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")

client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

app = Flask(__name__)


# ---------------------------------
# Відправка в Telegram
# ---------------------------------
def send_telegram(text: str):
    try:
        # ділимо повідомлення на шматки до 3500 символів
        chunks = [text[i:i+3500] for i in range(0, len(text), 3500)]
        for chunk in chunks:
            resp = requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                json={"chat_id": CHAT_ID, "text": chunk, "parse_mode": "HTML"},
                timeout=20
            )
            if resp.status_code != 200:
                logging.error(f"Telegram API: {resp.text}")
    except Exception as e:
        logging.error(f"send_telegram error: {e}")


# ---------------------------------
# Аналіз сигналів
# ---------------------------------
def analyze_symbol(symbol: str):
    try:
        klines = client.get_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_15MINUTE, limit=150)
        closes = [float(x[4]) for x in klines]
        high = max(closes)
        low = min(closes)
        last_price = closes[-1]

        signals = []

        # Breakouts (long)
        for lvl in [round(low + (high - low) * p, 4) for p in [0.2, 0.4, 0.5, 0.6, 0.8]]:
            if last_price < lvl:
                signals.append(f"🚀 LONG breakout біля {lvl}")

        # Breakouts (short)
        for lvl in [round(high - (high - low) * p, 4) for p in [0.2, 0.4, 0.5, 0.7]]:
            if last_price > lvl:
                signals.append(f"⚡ SHORT breakout біля {lvl}")

        # Pre-top (якщо ціна майже дійшла до high)
        if abs(last_price - high) / high < 0.01:
            signals.append(f"⚠️ Pre-top біля {round(high, 4)}")

        # Fake breakout (якщо ціна відбилась від рівня)
        if abs(last_price - high) / high < 0.02 and last_price < high:
            signals.append(f"⭐️ Fake breakout біля {round(high, 4)}")

        return signals

    except Exception as e:
        logging.error(f"{symbol} analyze error: {e}")
        return []


# ---------------------------------
# Основна функція
# ---------------------------------
def smart_auto():
    logging.info("Запускаю smart_auto()...")
    try:
        tickers = client.get_ticker()
        logging.info(f"Отримано {len(tickers)} монет з Binance")

        # фільтруємо по USDT і об'єму > 5M
        usdt_pairs = [x for x in tickers if x['symbol'].endswith("USDT") and float(x['quoteVolume']) > 5_000_000]
        logging.info(f"Фільтровані монети (>5M USDT volume): {len(usdt_pairs)}")

        # беремо топ-20 за ростом (% зміна за 24h)
        top_symbols = sorted(usdt_pairs, key=lambda x: float(x['priceChangePercent']), reverse=True)[:20]
        symbols = [x['symbol'] for x in top_symbols]
        logging.info(f"TOP 20 монет за ростом 24h: {symbols}")

        # аналізуємо
        results = []
        for s in symbols:
            sigs = analyze_symbol(s)
            if sigs:
                results.append(f"[SIGNAL] <b>{s}</b> (+{top_symbols[symbols.index(s)]['priceChangePercent']}%):\n" + "\n".join(sigs))
            else:
                results.append(f"[INFO] {s} (+{top_symbols[symbols.index(s)]['priceChangePercent']}%): сигналів нема")

        # формуємо повідомлення
        full_msg = "📊 <b>Smart Auto Scan — TOP 20 росту (24h)</b>\n\n" + "\n\n".join(results)
        send_telegram(full_msg)

    except Exception as e:
        logging.error(f"smart_auto error: {e}")

# ---------------------------------
# Flask routes
# ---------------------------------
@app.route("/")
def home():
    return "Bot is running", 200


@app.route(f"/telegram_webhook/{TELEGRAM_TOKEN}", methods=["POST"])
def telegram_webhook():
    update = request.get_json()
    if not update:
        return jsonify({"ok": False})

    message = update.get("message", {}).get("text", "")
    chat_id = update.get("message", {}).get("chat", {}).get("id", "")

    if str(chat_id) != str(CHAT_ID):
        return jsonify({"ok": True})

    if message == "/smart_auto":
        send_telegram("⚡ Виконую /smart_auto ...")
        smart_auto()

    return jsonify({"ok": True})


# ---------------------------------
# Запуск
# ---------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)