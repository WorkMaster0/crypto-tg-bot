import json
import os
import threading
import time
from datetime import datetime
from typing import Dict

from app.bot import bot
from app.config import (
    DEFAULT_INTERVAL, RISK_PCT_DEFAULT, SUBSCRIPTIONS_PATH, SETTINGS_PATH, TIMEZONE
)
from app.analytics import (
    get_klines, get_price, find_support_resistance, generate_signal,
    backtest_level_rsi, heatmap_top_moves, alpha_volatility_compression,
    position_size
)
from app.chart import plot_levels

# --------- Просте «сховище» налаштувань/підписок ---------

_state_lock = threading.Lock()
_subscriptions: Dict[int, Dict[str, str]] = {}  # chat_id -> {symbol: interval}
_user_settings: Dict[int, Dict[str, float]] = {}  # chat_id -> {"risk_pct": float}
_last_signal_sent: Dict[str, float] = {}  # key=f"{chat_id}:{symbol}" -> ts

def _load_json(path, default):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default
    return default

def _save_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def load_state():
    global _subscriptions, _user_settings
    with _state_lock:
        _subscriptions = _load_json(SUBSCRIPTIONS_PATH, {})
        _user_settings = _load_json(SETTINGS_PATH, {})

def save_state():
    with _state_lock:
        _save_json(SUBSCRIPTIONS_PATH, _subscriptions)
        _save_json(SETTINGS_PATH, _user_settings)

load_state()

# --------- Команди ---------

@bot.message_handler(commands=["start"])
def cmd_start(message):
    text = (
        "🚀 <b>Crypto Analysis Bot</b> готовий!\n"
        "Використовуй:\n"
        "• /analyze BTCUSDT [1h]\n"
        "• /levels BTCUSDT [1h]\n"
        "• /signal BTCUSDT [1h]\n"
        "• /backtest BTCUSDT [1h] [400]\n"
        "• /watch add BTCUSDT [1h] | /watch remove BTCUSDT | /watch list\n"
        "• /setrisk 1.0  — ризик % на угоду\n"
        "• /whatif BTCUSDT 65000 63500 1000  — розмір позиції\n"
        "• /heatmap — топ рухи за 24h\n"
        "• /alpha — сканер «стиснення ATR%» (ексклюзив)\n"
    )
    bot.reply_to(message, text)

@bot.message_handler(commands=["analyze"])
def cmd_analyze(message):
    parts = message.text.split()
    if len(parts) < 2:
        bot.reply_to(message, "Приклад: <code>/analyze BTCUSDT 1h</code>")
        return
    symbol = parts[1].upper()
    interval = parts[2] if len(parts) > 2 else DEFAULT_INTERVAL
    try:
        df = get_klines(symbol, interval)
        supports, resistances = find_support_resistance(df)
        sig = generate_signal(df)
        text = (
            f"📊 <b>{symbol} ({interval})</b>\n"
            f"Ціна: <b>{sig['price']:.4f}</b>\n"
            f"RSI: {sig['rsi']:.1f} | ATR: {sig['atr']:.4f}\n"
            f"Підтримки: {', '.join(f'{x:.4f}' for x in supports) if supports else '—'}\n"
            f"Опори: {', '.join(f'{x:.4f}' for x in resistances) if resistances else '—'}\n"
            f"Сигнал: <b>{sig['decision']}</b> ({int(sig['confidence']*100)}%)\n"
            f"Причина: {sig['reason']}"
        )
        bot.reply_to(message, text)
    except Exception as e:
        bot.reply_to(message, f"❌ Помилка: {e}")

@bot.message_handler(commands=["levels"])
def cmd_levels(message):
    parts = message.text.split()
    if len(parts) < 2:
        bot.reply_to(message, "Приклад: <code>/levels BTCUSDT 1h</code>")
        return
    symbol = parts[1].upper()
    interval = parts[2] if len(parts) > 2 else DEFAULT_INTERVAL
    try:
        df = get_klines(symbol, interval)
        supports, resistances = find_support_resistance(df)
        img = plot_levels(df, supports, resistances, title=f"{symbol} {interval}", last_price=df['close'].iloc[-1])
        bot.send_photo(message.chat.id, img, caption=(
            f"🔎 Рівні для <b>{symbol}</b> ({interval})\n"
            f"Підтримки: {', '.join(f'{x:.4f}' for x in supports) if supports else '—'}\n"
            f"Опори: {', '.join(f'{x:.4f}' for x in resistances) if resistances else '—'}"
        ))
    except Exception as e:
        bot.reply_to(message, f"❌ Помилка: {e}")

@bot.message_handler(commands=["signal"])
def cmd_signal(message):
    parts = message.text.split()
    if len(parts) < 2:
        bot.reply_to(message, "Приклад: <code>/signal BTCUSDT 1h</code>")
        return
    symbol = parts[1].upper()
    interval = parts[2] if len(parts) > 2 else DEFAULT_INTERVAL
    try:
        df = get_klines(symbol, interval)
        sig = generate_signal(df)
        direction = "🟩 LONG" if sig["decision"] == "LONG" else ("🟥 SHORT" if sig["decision"] == "SHORT" else "⬜ NEUTRAL")
        text = (
            f"⚡️ <b>Сигнал {symbol} ({interval})</b>\n"
            f"{direction} | Впевненість: <b>{int(sig['confidence']*100)}%</b>\n"
            f"Ціна: {sig['price']:.4f} | RSI: {sig['rsi']:.1f} | ATR: {sig['atr']:.4f}\n"
            f"Ближча підтримка: {sig['near_support']:.4f}" if sig['near_support'] else "Ближча підтримка: —"
        )
        text += "\n"
        text += f"Ближчий опір: {sig['near_resistance']:.4f}" if sig['near_resistance'] else "Ближчий опір: —"
        text += f"\nПричина: {sig['reason']}"
        bot.reply_to(message, text)
    except Exception as e:
        bot.reply_to(message, f"❌ Помилка: {e}")

@bot.message_handler(commands=["backtest"])
def cmd_backtest(message):
    parts = message.text.split()
    if len(parts) < 2:
        bot.reply_to(message, "Приклад: <code>/backtest BTCUSDT 1h 400</code>")
        return
    symbol = parts[1].upper()
    interval = parts[2] if len(parts) > 2 else DEFAULT_INTERVAL
    lookback = int(parts[3]) if len(parts) > 3 else 400
    try:
        df = get_klines(symbol, interval, limit=lookback)
        res = backtest_level_rsi(df)
        text = (
            f"🧪 <b>Backtest {symbol} ({interval})</b>\n"
            f"Угод: {res['trades']}\n"
            f"Winrate: {res['winrate']}%\n"
            f"Сумарний PnL: {res['pnl_pct']}%"
        )
        bot.reply_to(message, text)
    except Exception as e:
        bot.reply_to(message, f"❌ Помилка: {e}")

@bot.message_handler(commands=["setrisk"])
def cmd_setrisk(message):
    parts = message.text.split()
    if len(parts) < 2:
        bot.reply_to(message, "Приклад: <code>/setrisk 1.0</code> (1% ризику)")
        return
    try:
        pct = float(parts[1])
        with _state_lock:
            _user_settings.setdefault(message.chat.id, {})["risk_pct"] = pct
        save_state()
        bot.reply_to(message, f"✅ Ризик на угоду встановлено: {pct}%")
    except Exception:
        bot.reply_to(message, "❌ Вкажи число, наприклад: <code>/setrisk 1.0</code>")

@bot.message_handler(commands=["whatif"])
def cmd_whatif(message):
    parts = message.text.split()
    if len(parts) < 5:
        bot.reply_to(message, "Приклад: <code>/whatif BTCUSDT 65000 63500 1000</code> (entry stop balance)")
        return
    symbol = parts[1].upper()
    entry = float(parts[2]); stop = float(parts[3]); balance = float(parts[4])
    with _state_lock:
        risk_pct = _user_settings.get(message.chat.id, {}).get("risk_pct", RISK_PCT_DEFAULT)
    res = position_size(balance, risk_pct, entry, stop)
    bot.reply_to(
        message,
        f"🧮 <b>WhatIf {symbol}</b>\n"
        f"Баланс: {balance}\nРизик: {risk_pct}% → {res['risk_amount']:.2f}\n"
        f"Розмір позиції: <b>{res['size']:.4f}</b> (у базовій валюті)\n"
        f"Оціночний RR(2x): ~{res['rr']:.2f}"
    )

@bot.message_handler(commands=["heatmap"])
def cmd_heatmap(message):
    try:
        top = heatmap_top_moves()
        lines = [f"{sym}: {chg:+.2f}%" for sym, chg in top]
        bot.reply_to(message, "🌡 <b>24h Heatmap (топ рухи)</b>\n" + "\n".join(lines))
    except Exception as e:
        bot.reply_to(message, f"❌ Помилка: {e}")

@bot.message_handler(commands=["alpha"])
def cmd_alpha(message):
    parts = message.text.split()
    interval = parts[1] if len(parts) > 1 else DEFAULT_INTERVAL
    try:
        res = alpha_volatility_compression(interval=interval)
        lines = [f"{sym}: ATR% ~ {val:.2f}" for sym, val in res]
        bot.reply_to(message, f"🧬 <b>Alpha (Compression) {interval}</b>\n" + "\n".join(lines))
    except Exception as e:
        bot.reply_to(message, f"❌ Помилка: {e}")

@bot.message_handler(commands=["watch"])
def cmd_watch(message):
    """
    /watch add BTCUSDT [1h]
    /watch remove BTCUSDT
    /watch list
    """
    parts = message.text.split()
    if len(parts) < 2:
        bot.reply_to(message, "Приклад: <code>/watch add BTCUSDT 1h</code> | <code>/watch list</code>")
        return
    action = parts[1].lower()
    if action == "add":
        if len(parts) < 3:
            bot.reply_to(message, "Приклад: <code>/watch add BTCUSDT 1h</code>")
            return
        symbol = parts[2].upper()
        interval = parts[3] if len(parts) > 3 else DEFAULT_INTERVAL
        with _state_lock:
            _subscriptions.setdefault(message.chat.id, {})[symbol] = interval
        save_state()
        bot.reply_to(message, f"👀 Додав {symbol} ({interval}) до watch-листа")
    elif action == "remove":
        if len(parts) < 3:
            bot.reply_to(message, "Приклад: <code>/watch remove BTCUSDT</code>")
            return
        symbol = parts[2].upper()
        with _state_lock:
            _subscriptions.setdefault(message.chat.id, {}).pop(symbol, None)
        save_state()
        bot.reply_to(message, f"🗑 Прибрав {symbol} з watch-листа")
    elif action == "list":
        with _state_lock:
            subs = _subscriptions.get(message.chat.id, {})
        if not subs:
            bot.reply_to(message, "Порожній watch-лист. Додай: <code>/watch add BTCUSDT 1h</code>")
        else:
            lines = [f"{s} ({itv})" for s, itv in subs.items()]
            bot.reply_to(message, "📬 Watch-лист:\n" + "\n".join(lines))
    else:
        bot.reply_to(message, "Доступно: <code>add</code> | <code>remove</code> | <code>list</code>")

# --------- Фоновий «наглядач» сигналів ---------

def _watcher_loop():
    while True:
        time.sleep(30)  # перевірка кожні ~30с
        with _state_lock:
            subs_copy = dict(_subscriptions)
        for chat_id, pairs in subs_copy.items():
            for symbol, interval in pairs.items():
                key = f"{chat_id}:{symbol}"
                # анти-спам: не частіше ніж раз на 10 хв на один символ
                if time.time() - _last_signal_sent.get(key, 0) < 600:
                    continue
                try:
                    df = get_klines(symbol, interval)
                    sig = generate_signal(df)
                    if sig["decision"] in ("LONG", "SHORT") and sig["confidence"] >= 0.60:
                        direction = "🟩 LONG" if sig["decision"] == "LONG" else "🟥 SHORT"
                        txt = (
                            f"🔔 <b>Watch сигнал {symbol} ({interval})</b>\n"
                            f"{direction} | Впевненість: <b>{int(sig['confidence']*100)}%</b>\n"
                            f"Ціна: {sig['price']:.4f} | RSI: {sig['rsi']:.1f} | ATR: {sig['atr']:.4f}\n"
                            f"Причина: {sig['reason']}"
                        )
                        bot.send_message(chat_id, txt)
                        _last_signal_sent[key] = time.time()
                except Exception:
                    continue

# запуск потоку-наглядача
threading.Thread(target=_watcher_loop, daemon=True).start()