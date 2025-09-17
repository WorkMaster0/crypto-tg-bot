import os
import logging
from flask import Flask, request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
from binance.client import Client
from telegram import Bot, Update
from telegram.ext import (
    ApplicationBuilder, CommandHandler, ContextTypes, Dispatcher
)
import asyncio
import random
import hashlib

# ---------- Налаштування ----------
BINANCE_API_KEY = "ВАШ_BINANCE_API_KEY"
BINANCE_API_SECRET = "ВАШ_BINANCE_API_SECRET"
TELEGRAM_TOKEN = "ВАШ_TELEGRAM_TOKEN"

client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)
bot = Bot(token=TELEGRAM_TOKEN)

# ---------- Flask app ----------
app = Flask(__name__)

# ---------- Функції отримання даних ----------
def get_symbols():
    info = client.get_ticker()
    symbols = [s['symbol'] for s in info if 'USDT' in s['symbol']]
    return symbols

def get_klines(symbol="BTCUSDT", interval="1h", limit=200):
    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(klines, columns=['open_time','open','high','low','close','volume','close_time',
                                       'quote_asset_volume','number_of_trades','taker_buy_base_asset_volume',
                                       'taker_buy_quote_asset_volume','ignore'])
    df[['open','high','low','close','volume']] = df[['open','high','low','close','volume']].astype(float)
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    return df

# ---------- SMC аналіз ----------
def analyze_smc(df):
    df = df.copy()
    df['prev_high'] = df['high'].shift(1)
    df['prev_low'] = df['low'].shift(1)
    
    df['BOS_up'] = df['high'] > df['prev_high']
    df['BOS_down'] = df['low'] < df['prev_low']
    
    df['OB'] = np.where(df['BOS_up'], df['low'].shift(1),
                        np.where(df['BOS_down'], df['high'].shift(1), np.nan))
    
    df['Signal'] = np.where(df['BOS_up'] & (~df['OB'].isna()), 'BUY',
                         np.where(df['BOS_down'] & (~df['OB'].isna()), 'SELL', None))
    
    df['SL'] = np.where(df['Signal']=='BUY', df['OB']*0.995, np.where(df['Signal']=='SELL', df['OB']*1.005, np.nan))
    df['TP'] = np.where(df['Signal']=='BUY', df['close']*1.01, np.where(df['Signal']=='SELL', df['close']*0.99, np.nan))
    return df

# ---------- Побудова графіка ----------
def plot_chart(df, symbol="BTCUSDT"):
    df_plot = df[['open_time','open','high','low','close','volume']].copy()
    df_plot.set_index('open_time', inplace=True)

    apds = []
    if 'OB' in df.columns:
        apds.append(mpf.make_addplot(df['OB'], type='scatter', markersize=80, marker='s', color='blue'))
    if 'SL' in df.columns:
        apds.append(mpf.make_addplot(df['SL'], type='scatter', markersize=50, marker='x', color='red'))
    if 'TP' in df.columns:
        apds.append(mpf.make_addplot(df['TP'], type='scatter', markersize=80, marker='*', color='green'))

    filename = f"{symbol}_smc.png"
    mpf.plot(
        df_plot, type='candle', style='yahoo',
        title=f"{symbol} - Smart Money Concept",
        addplot=apds if apds else None,
        volume=True, savefig=filename
    )
    return filename

# ---------- Telegram команди ----------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Привіт! Я Smart Money бот.\n"
        "/smc TIMEFRAME - сигнал SMC для всіх USDT токенів\n"
        "/liqmap TIMEFRAME - карта ліквідності\n"
        "/orderflow TIMEFRAME - ордер флоу\n"
        "/ultrasecret - секретна команда"
    )

# ---- SMC для всіх токенів ----
async def smc_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔎 Генерую сигнали для всіх USDT токенів...")
    try:
        interval = context.args[0] if len(context.args) >= 1 else "1h"
        symbols = get_symbols()
        for symbol in symbols[:10]:  # обмежимо до 10 для швидкості
            df = analyze_smc(get_klines(symbol, interval))
            chart_file = plot_chart(df, symbol)
            latest_signal = df.dropna(subset=['Signal']).tail(1)
            if latest_signal.empty:
                continue
            row = latest_signal.iloc[0]
            time_str = row['open_time'].strftime('%Y-%m-%d %H:%M')
            caption = (f"📊 SMC для *{symbol} {interval}*:\n"
                       f"{time_str} | {row['Signal']} | Entry: {row['close']:.2f} | SL: {row['SL']:.2f} | TP: {row['TP']:.2f}")
            with open(chart_file, 'rb') as f:
                await bot.send_photo(chat_id=update.effective_chat.id, photo=f, caption=caption)
    except Exception as e:
        await update.message.reply_text(f"❌ Помилка: {e}")

# ---- LIQUIDITY MAP ----
async def liqmap_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("📡 Генерую карту ліквідності...")
    try:
        interval = context.args[0] if len(context.args) >= 1 else "1h"
        symbols = get_symbols()
        for symbol in symbols[:10]:
            df = get_klines(symbol, interval)
            highs = df['high'].rolling(window=5).max()
            lows = df['low'].rolling(window=5).min()

            plt.figure(figsize=(15,7))
            plt.plot(df['open_time'], df['close'], color='black', label="Close")
            plt.scatter(df['open_time'], highs, color='red', label="Liquidity Above")
            plt.scatter(df['open_time'], lows, color='blue', label="Liquidity Below")
            plt.title(f"{symbol} - Liquidity Pools")
            plt.xlabel("Time")
            plt.ylabel("Price")
            plt.legend()
            filename = f"{symbol}_liqmap.png"
            plt.savefig(filename)
            plt.close()
            with open(filename, 'rb') as f:
                await bot.send_photo(chat_id=update.effective_chat.id, photo=f,
                                     caption=f"🌊 Liquidity Map для *{symbol} {interval}*")
    except Exception as e:
        await update.message.reply_text(f"❌ Помилка: {e}")

# ---- ORDERFLOW ----
async def orderflow_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("📊 Генерую Order Flow...")
    try:
        interval = context.args[0] if len(context.args) >= 1 else "1h"
        symbols = get_symbols()
        for symbol in symbols[:10]:
            df = get_klines(symbol, interval)
            df['buy_vol'] = np.where(df['close'] > df['open'], df['volume'], df['volume']*0.3)
            df['sell_vol'] = np.where(df['close'] < df['open'], df['volume'], df['volume']*0.3)
            df['delta'] = df['buy_vol'] - df['sell_vol']

            plt.figure(figsize=(15,7))
            plt.bar(df['open_time'], df['delta'], color=np.where(df['delta']>0,'green','red'))
            plt.title(f"{symbol} - Order Flow Delta")
            plt.xlabel("Time")
            plt.ylabel("Delta Volume")
            filename = f"{symbol}_orderflow.png"
            plt.savefig(filename)
            plt.close()
            last_delta = df['delta'].iloc[-1]
            trend = "🟢 Покупці домінують" if last_delta>0 else "🔴 Продавці домінують"
            with open(filename, 'rb') as f:
                await bot.send_photo(chat_id=update.effective_chat.id, photo=f,
                                     caption=f"📊 Order Flow для *{symbol} {interval}*\n{trend}")
    except Exception as e:
        await update.message.reply_text(f"❌ Помилка: {e}")

# ---- ULTRASECRET ----
async def ultrasecret(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🕵️ Виконую секретну операцію...")
    try:
        interval = context.args[0] if len(context.args) >= 1 else "1h"
        symbol = "BTCUSDT"
        df = get_klines(symbol, interval, limit=150)
        secret_metric = df["close"].pct_change().rolling(10).std().iloc[-1]*random.uniform(0.8,1.2)
        stamp = hashlib.md5(str(secret_metric).encode()).hexdigest()[:8]
        msg = f"🔒 Ультра-секретний аналіз для {symbol} {interval}\n\n" \
              f"Секретна метрика: {secret_metric:.5f}\n" \
              f"Ідентифікатор: {stamp}\n\n(🧩 Розгадати значення можеш лише сам...)"
        await update.message.reply_text(msg)
    except Exception as e:
        await update.message.reply_text(f"❌ Помилка секретної операції: {e}")

# ---------- Ініціалізація ApplicationBuilder ----------
application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
application.add_handler(CommandHandler("start", start))
application.add_handler(CommandHandler("smc", smc_command))
application.add_handler(CommandHandler("liqmap", liqmap_command))
application.add_handler(CommandHandler("orderflow", orderflow_command))
application.add_handler(CommandHandler("ultrasecret", ultrasecret))

# ---------- Flask route для вебхука ----------
@app.route(f"/{TELEGRAM_TOKEN}", methods=["POST"])
def webhook():
    update = Update.de_json(request.get_json(force=True), bot)
    dispatcher: Dispatcher = application.dispatcher
    dispatcher.process_update(update)
    return "OK"

# ---------- Встановлення вебхука на старті ----------
@app.before_first_request
def setup_webhook():
    url = "https://dex-tg-bot.onrender.com/" + TELEGRAM_TOKEN
    bot.set_webhook(url)
    logging.info(f"Webhook встановлено на {url}")

# ---------- Запуск Flask ----------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)