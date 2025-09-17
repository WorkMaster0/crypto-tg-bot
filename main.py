import os
import time
import requests
import logging
from flask import Flask

# === Config ===
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

LEADERBOARD_BASE = "https://www.binance.com/bapi/futures/v1/public"

app = Flask(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

def send_telegram_message(text: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": text, "parse_mode": "Markdown"}
    try:
        r = requests.post(url, json=payload, timeout=10)
        if r.status_code != 200:
            logging.error(f"Telegram error: {r.text}")
    except Exception as e:
        logging.error(f"Telegram send failed: {e}")

def get_top_traders(page=1, size=20):
    url = f"{LEADERBOARD_BASE}/leaderboard/getLeaderboardRank"
    payload = {
        "tradeType": "PERPETUAL",
        "statisticsType": "ROI",
        "periodType": "MONTHLY",  # тепер дивимось PnL за місяць
        "isShared": True,
        "page": page,
        "size": size
    }
    try:
        r = requests.post(url, json=payload, timeout=10)
        return r.json().get("data", {}).get("otherRankList", [])
    except Exception as e:
        logging.error(f"Error fetching traders: {e}")
        return []

def get_trader_positions(encryptedUid: str):
    url = f"{LEADERBOARD_BASE}/leaderboard/getOtherPosition"
    params = {"encryptedUid": encryptedUid, "tradeType": "PERPETUAL"}
    try:
        r = requests.get(url, params=params, timeout=10)
        return r.json().get("data", [])
    except Exception as e:
        logging.error(f"Error fetching positions: {e}")
        return []

def monitor_traders():
    logging.info("🔎 Scanning top traders...")
    traders = get_top_traders(page=1, size=50)
    for t in traders:
        uid = t.get("encryptedUid")
        nick = t.get("nickName")
        win_rate = t.get("winRate", 0)
        total_trades = t.get("totalTrades", 0)
        pnl = t.get("roi", 0)  # ROI % (PnL)

        if not uid or win_rate is None or pnl is None:
            continue

        # === Фільтр ===
        if win_rate >= 93 and total_trades > 50 and pnl > 0:
            positions = get_trader_positions(uid)
            for pos in positions:
                symbol = pos.get("symbol")
                entry = pos.get("entryPrice")
                leverage = pos.get("leverage")
                side = pos.get("side")  # LONG/SHORT

                message = (
                    f"🔥 *CopyTrade Signal*\n\n"
                    f"👤 Trader: *{nick}*\n"
                    f"✅ Winrate: `{win_rate}%`\n"
                    f"💰 PnL (30d): `{pnl:.2f}%`\n"
                    f"📊 Trades: `{total_trades}`\n\n"
                    f"📌 Symbol: `{symbol}`\n"
                    f"💹 Action: *{side}* ({leverage}x)\n"
                    f"📈 Entry Price: `{entry}`\n\n"
                    f"⏰ Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} UTC"
                )

                send_telegram_message(message)

@app.route('/')
def home():
    return "✅ CopyTrading Bot is Live with PnL Filter"

if __name__ == '__main__':
    from threading import Thread

    def loop():
        while True:
            monitor_traders()
            time.sleep(300)  # кожні 5 хв

    Thread(target=loop, daemon=True).start()
    app.run(host="0.0.0.0", port=5000)