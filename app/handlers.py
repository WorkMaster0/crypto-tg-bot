import numpy as np
from app.bot import bot
from app.analytics import (
    get_price, get_klines, generate_signal_text, trend_strength_text,
    find_levels, top_movers, position_size, normalize_symbol, find_atr_squeeze  # <-- Додано find_atr_squeeze
)
from app.chart import plot_candles
from app.config import DEFAULT_INTERVAL, ALLOWED_INTERVALS

# просте зберігання налаштувань чату в ОЗП
_user_defaults = {}  # chat_id -> {"interval": "1h"}

def _default_interval(chat_id):
    return _user_defaults.get(chat_id, {}).get("interval", DEFAULT_INTERVAL)

def _parse_args(msg_text: str):
    parts = msg_text.split()
    symbol = None
    interval = None
    if len(parts) >= 2:
        symbol = normalize_symbol(parts[1])
    if len(parts) >= 3 and parts[2] in ALLOWED_INTERVALS:
        interval = parts[2]
    return symbol, interval

# ---------- /start ----------
@bot.message_handler(commands=['start'])
def start(message):
    bot.reply_to(message, (
        "🚀 <b>Crypto Analysis Bot</b> запущено!\n"
        "Використання прикладів:\n"
        "• <code>/price BTCUSDT</code>\n"
        "• <code>/analyze BTCUSDT 1h</code>\n"
        "• <code>/levels BTCUSDT 4h</code>\n"
        "• <code>/chart BTCUSDT 1h</code>\n"
        "• <code>/trend BTCUSDT</code>\n"
        "• <code>/heatmap</code>\n"
        "• <code>/risk 1000 1 65000 64000</code>  (баланс 1000$, ризик 1%, вхід 65000, стоп 64000)\n"
        "• <code>/setdefault 1h</code>\n"
        "Довідка: <code>/help</code>"
    ))

# ---------- /help ----------
@bot.message_handler(commands=['help'])
def help_cmd(message):
    bot.reply_to(message, (
        "<b>Команди:</b>\n"
        "<code>/price SYMBOL</code> — поточна ціна\n"
        "<code>/analyze SYMBOL [interval]</code> — сигнал + рівні S/R\n"
        "<code>/levels SYMBOL [interval]</code> — список рівнів підтримки/опору\n"
        "<code>/chart SYMBOL [interval]</code> — графік з EMA та рівнями\n"
        "<code>/trend SYMBOL [interval]</code> — сила тренду\n"
        "<code>/heatmap [N]</code> — топ рухів USDT-пар (за 24h)\n"
        "<code>/risk balance risk% entry stop</code> — розмір позиції\n"
        "<code>/setdefault interval</code> — інтервал за замовчуванням для цього чату\n"
        f"Доступні інтервали: {', '.join(sorted(ALLOWED_INTERVALS))}"
    ))

# ---------- /price ----------
@bot.message_handler(commands=['price'])
def price_handler(message):
    symbol, interval = _parse_args(message.text)
    if not symbol:
        return bot.reply_to(message, "⚠️ Приклад: <code>/price BTCUSDT</code>")
    try:
        price = get_price(symbol)
        bot.reply_to(message, f"💰 <b>{symbol}</b> = <b>{price:.6f}</b> USDT")
    except Exception as e:
        bot.reply_to(message, f"❌ Помилка: {e}")

# ---------- /levels ----------
@bot.message_handler(commands=['levels'])
def levels_handler(message):
    symbol, interval = _parse_args(message.text)
    if not symbol:
        return bot.reply_to(message, "⚠️ Приклад: <code>/levels BTCUSDT 1h</code>")
    interval = interval or _default_interval(message.chat.id)
    try:
        candles = get_klines(symbol, interval=interval)
        lv = find_levels(candles)
        s = ", ".join(f"{x:.4f}" for x in lv["supports"])
        r = ", ".join(f"{x:.4f}" for x in lv["resistances"])
        bot.reply_to(message, (
            f"🔎 <b>{symbol}</b> [{interval}] Levels\n"
            f"Supports: {s or '—'}\n"
            f"Resistances: {r or '—'}\n"
            f"Nearest S: <b>{lv['near_support']:.4f}</b> | "
            f"Nearest R: <b>{lv['near_resistance']:.4f}</b>\n"
            f"ATR(14): {lv['atr']:.4f} | tol: {lv['tolerance']:.4f}"
        ))
    except Exception as e:
        bot.reply_to(message, f"❌ Помилка: {e}")

# ---------- /analyze ----------
@bot.message_handler(commands=['analyze'])
def analyze_handler(message):
    symbol, interval = _parse_args(message.text)
    if not symbol:
        return bot.reply_to(message, "⚠️ Приклад: <code>/analyze BTCUSDT 1h</code>")
    interval = interval or _default_interval(message.chat.id)
    try:
        text = generate_signal_text(symbol, interval=interval)
        bot.reply_to(message, text)
    except Exception as e:
        bot.reply_to(message, f"❌ Помилка: {e}")

# ---------- /trend ----------
@bot.message_handler(commands=['trend'])
def trend_handler(message):
    symbol, interval = _parse_args(message.text)
    if not symbol:
        return bot.reply_to(message, "⚠️ Приклад: <code>/trend BTCUSDT 4h</code>")
    interval = interval or _default_interval(message.chat.id)
    try:
        candles = get_klines(symbol, interval=interval)
        txt = trend_strength_text(candles)
        bot.reply_to(message, f"📈 <b>{symbol}</b> [{interval}]  {txt}")
    except Exception as e:
        bot.reply_to(message, f"❌ Помилка: {e}")

# ---------- /chart ----------
@bot.message_handler(commands=['chart'])
def chart_handler(message):
    symbol, interval = _parse_args(message.text)
    if not symbol:
        return bot.reply_to(message, "⚠️ Приклад: <code>/chart BTCUSDT 1h</code>")
    interval = interval or _default_interval(message.chat.id)
    try:
        img = plot_candles(symbol, interval=interval, limit=200, with_levels=True)
        bot.send_photo(message.chat.id, img)
    except Exception as e:
        bot.reply_to(message, f"❌ Помилка: {e}")

# ---------- /heatmap ----------
@bot.message_handler(commands=['heatmap'])
def heatmap_handler(message):
    parts = message.text.split()
    try:
        n = int(parts[1]) if len(parts) > 1 else 10
    except:
        n = 10
    try:
        movers = top_movers(limit=min(max(n, 1), 20))
        lines = ["🔥 <b>Top movers (24h, USDT pairs)</b>"]
        for i, (s, chg, qv) in enumerate(movers, 1):
            lines.append(f"{i}. <b>{s}</b>  {chg:+.2f}%  | vol≈{qv/1e6:.2f}M")
        bot.reply_to(message, "\n".join(lines))
    except Exception as e:
        bot.reply_to(message, f"❌ Помилка: {e}")

# ---------- /risk ----------
@bot.message_handler(commands=['risk'])
def risk_handler(message):
    parts = message.text.split()
    if len(parts) < 5:
        return bot.reply_to(message, "⚠️ Приклад: <code>/risk 1000 1 65000 64000</code> (balance risk% entry stop)")
    try:
        balance = float(parts[1])
        risk_pct = float(parts[2])
        entry = float(parts[3])
        stop = float(parts[4])
        res = position_size(balance, risk_pct, entry, stop)
        bot.reply_to(message, (
            f"🧮 Risk: {risk_pct:.2f}% від ${balance:.2f} → ${res['risk_amount']:.2f}\n"
            f"📦 Position size ≈ <b>{res['qty']:.6f}</b> токенів\n"
            f"🎯 1R ≈ {abs(entry - stop):.4f} | 2R TP ≈ {entry + (res['rr_one_tp'] if entry>stop else -res['rr_one_tp']):.4f}"
        ))
    except Exception as e:
        bot.reply_to(message, f"❌ Помилка: {e}")

# ---------- /setdefault ----------
@bot.message_handler(commands=['setdefault'])
def setdefault_handler(message):
    parts = message.text.split()
    if len(parts) < 2 or parts[1] not in ALLOWED_INTERVALS:
        return bot.reply_to(message, "⚠️ Приклад: <code>/setdefault 1h</code>")
    _user_defaults.setdefault(message.chat.id, {})["interval"] = parts[1]
    bot.reply_to(message, f"✅ Інтервал за замовчуванням для цього чату: <b>{parts[1]}</b>")

# ---------- /squeeze ----------
@bot.message_handler(commands=['squeeze'])
def squeeze_scanner(message):
    """Сканує топ пари на стиснення волатильності і аналізує напрям"""
    parts = message.text.split()
    try:
        n = int(parts[1]) if len(parts) > 1 else 5
    except:
        n = 5
    n = max(1, min(n, 10))

    try:
        top_pairs = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 
                     'XRPUSDT', 'ADAUSDT', 'AVAXUSDT', 'DOTUSDT', 
                     'DOGEUSDT', 'LINKUSDT']
        squeeze_list = []

        for pair in top_pairs:
            try:
                ratio = find_atr_squeeze(pair, '1h', 50)
                if ratio < 0.8:
                    squeeze_list.append((pair, ratio))
            except Exception as e:
                print(f"Помилка для {pair}: {e}")
                continue

        squeeze_list.sort(key=lambda x: x[1])

        if squeeze_list:
            lines = [f"🔍 <b>Стиснення волатильності (ATR Squeeze)</b> на 1h:"]
            for i, (pair, ratio) in enumerate(squeeze_list[:n], 1):
                # Отримуємо додаткові дані
                candles = get_klines(pair, interval="1h", limit=100)
                close = np.array(candles["c"], dtype=float)

                rsi_val = rsi(close, 14)[-1]
                ema_fast = ta.trend.ema_indicator(pd.Series(close), 20).iloc[-1]
                ema_slow = ta.trend.ema_indicator(pd.Series(close), 50).iloc[-1]

                # Bollinger Bands
                bb_high = ta.volatility.BollingerBands(pd.Series(close), window=20).bollinger_hband().iloc[-1]
                bb_low  = ta.volatility.BollingerBands(pd.Series(close), window=20).bollinger_lband().iloc[-1]
                last_price = close[-1]

                # Визначаємо напрям
                signal = "⚖️ Нейтрально"
                if last_price > ema_fast > ema_slow and last_price > bb_high:
                    signal = "🟢 Ймовірний ріст (пробій вгору)"
                elif last_price < ema_fast < ema_slow and last_price < bb_low:
                    signal = "🔴 Ймовірне падіння (пробій вниз)"
                elif rsi_val > 65:
                    signal = "⚠️ RSI перекупленість → обережно, може бути відкат вниз"
                elif rsi_val < 35:
                    signal = "⚠️ RSI перепроданість → можливий відскок вгору"

                lines.append(
                    f"{i}. {pair} : ATR Ratio = <code>{ratio:.3f}</code>\n"
                    f"   ➤ RSI = <b>{rsi_val:.1f}</b>\n"
                    f"   ➤ {signal}"
                )

            lines.append("\n💡 ATR Squeeze = сигнал, що скоро буде сильний рух. "
                         "Індикатори вище допомагають визначити ймовірний напрям.")

            bot.send_message(message.chat.id, "\n".join(lines), parse_mode="HTML")

        else:
            bot.send_message(message.chat.id, 
                             "На даний момент сильних стискань не виявлено (всі коефіцієнти > 0.8).")

    except Exception as e:
        bot.send_message(message.chat.id, f"❌ Помилка сканера: {e}")
        
        # ---------- /trap ----------
@bot.message_handler(commands=['trap'])
def trap_scanner(message):
    """Сканує топ пари на пастки ліквідності"""
    top_pairs = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 
                 'XRPUSDT', 'ADAUSDT', 'AVAXUSDT', 'DOTUSDT', 
                 'DOGEUSDT', 'LINKUSDT']

    traps = []
    for pair in top_pairs:
        try:
            signal = detect_liquidity_trap(pair, interval="1h", lookback=50)
            if signal:
                traps.append(signal)
        except Exception as e:
            print(f"Помилка для {pair}: {e}")
            continue

    if traps:
        bot.send_message(message.chat.id, 
                         "🔍 <b>Виявлені пастки ліквідності:</b>\n\n" + "\n".join(traps),
                         parse_mode="HTML")
    else:
        bot.send_message(message.chat.id, 
                         "✅ Пасток ліквідності не знайдено на 1h таймфреймі.")
                        
# ---------- Smart S/R ----------
def find_support_resistance(prices, window=20, delta=0.005):
    sr_levels = []
    for i in range(window, len(prices)-window):
        local_max = max(prices[i-window:i+window+1])
        local_min = min(prices[i-window:i+window+1])
        if prices[i] == local_max:
            if all(abs(prices[i]-lvl)/lvl > delta for lvl in sr_levels):
                sr_levels.append(prices[i])
        elif prices[i] == local_min:
            if all(abs(prices[i]-lvl)/lvl > delta for lvl in sr_levels):
                sr_levels.append(prices[i])
    return sorted(sr_levels)

@bot.message_handler(commands=['smart_sr'])
def smart_sr_handler(message):
    parts = message.text.split()
    if len(parts) < 2:
        return bot.reply_to(message, "⚠️ Використання: /smart_sr BTCUSDT")
    symbol = parts[1].upper()
    
    try:
    df = get_klines(symbol, interval="1h", limit=200)
    if not df or len(df['c']) == 0:
        return bot.send_message(message.chat.id, f"❌ Дані для {symbol} недоступні")

    closes = np.array(df['c'], dtype=float)
    highs = np.array(df['h'], dtype=float)
    lows = np.array(df['l'], dtype=float)
    volumes = np.array(df['v'], dtype=float)
    
    sr_levels = find_support_resistance(closes, window=20, delta=0.005)
    last_price = closes[-1]

    # Перевірка breakout
    signal = "ℹ️ Патерн не знайдено"
    for lvl in sr_levels:
        if last_price > lvl * 1.01:
            signal = f"🚀 LONG breakout: ціна пробила опір {lvl:.4f}"
        elif last_price < lvl * 0.99:
            signal = f"⚡ SHORT breakout: ціна пробила підтримку {lvl:.4f}"

    # Перевірка pre-top / pump
    impulse = (closes[-1] - closes[-4]) / closes[-4]
    vol_spike = volumes[-1] > 1.5 * np.mean(volumes[-20:])
    nearest_resistance = max([lvl for lvl in sr_levels if lvl < last_price], default=None)
    if impulse > 0.08 and vol_spike and nearest_resistance is not None:
        signal += f"\n⚠️ Pre-top detected: можливий short біля {nearest_resistance:.4f}"

    img = plot_candles(symbol, interval="1h", limit=100, sr_levels=sr_levels)
    bot.send_photo(message.chat.id, img, caption=f"<b>{symbol} — Smart S/R Analysis</b>\n\n{signal}", parse_mode="HTML")
    
except Exception as e:
    bot.send_message(message.chat.id, f"❌ Error: {e}")