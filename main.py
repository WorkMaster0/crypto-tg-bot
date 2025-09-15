"""
telegram_futures_revolution_bot.py
Один файл: Telegram бот для симуляції/бектесту 10 "революційних" стратегій на ф'ючерсах.
Примітка: ордери ОВКЛЮЧАЮТЬСЯ ТІЛЬКИ якщо вручну змінити ENABLE_LIVE = True і прописати ключі.
"""

import os
import asyncio
import logging
from datetime import datetime, timedelta
import io
import csv
import traceback

import ccxt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from telegram import Update, InputFile
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler

# ----------------------------
# Basic config
# ----------------------------
TELE_TOKEN = os.getenv("TELEGRAM_TOKEN", "")  # краще поставити в оточення
ENABLE_LIVE = False  # !!! ЗА МОВЧАННЯМ False (не виконує реальні ордери)

# Биржа для даних (за замовчуванням): Binance (public OHLCV)
EXCHANGE_ID = "binance"  # або інша підтримувана ccxt біржа
EXCHANGE = getattr(ccxt, EXCHANGE_ID)()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------
# Utility indicator functions
# ----------------------------
def ema(series: pd.Series, period: int) -> pd.Series:
    """Проста EMA через pandas. Формула: multiplier = 2/(period+1)"""
    return series.ewm(span=period, adjust=False).mean()

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = pd.concat([
        (df['high'] - df['low']).abs(),
        (df['high'] - df['close'].shift()).abs(),
        (df['low'] - df['close'].shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

# ----------------------------
# Strategy implementations (10)
# Each returns signals DataFrame with 'signal' column: 1 buy, -1 sell, 0 none
# and optionally performance metrics.
# ----------------------------
def strat_ema_crossover(df: pd.DataFrame, params=None):
    """Класичний EMA crossover (швидка/повільна)."""
    if params is None:
        params = {"fast": 9, "slow": 21}
    fast = ema(df['close'], params['fast'])
    slow = ema(df['close'], params['slow'])
    sig = pd.Series(0, index=df.index)
    cross_up = (fast > slow) & (fast.shift() <= slow.shift())
    cross_down = (fast < slow) & (fast.shift() >= slow.shift())
    sig[cross_up] = 1
    sig[cross_down] = -1
    df2 = df.copy()
    df2['fast'] = fast
    df2['slow'] = slow
    df2['signal'] = sig
    return df2

def strat_ema_ribbon_breakout(df: pd.DataFrame, params=None):
    """EMA ribbon breakout: кілька EMA (ribbon) — сигнал при розходженні і пробої вузької стрічки."""
    if params is None:
        params = {"periods": [5,8,13,21,34]}
    emas = {p: ema(df['close'], p) for p in params['periods']}
    ribbon_mean = pd.concat(emas.values(), axis=1).mean(axis=1)
    width = pd.concat(emas.values(), axis=1).std(axis=1)
    sig = pd.Series(0, index=df.index)
    # breakout when price closes above ribbon_mean + k*width
    k = 1.0
    sig[(df['close'] > ribbon_mean + k*width) & (df['close'].shift() <= ribbon_mean.shift()+k*width.shift())] = 1
    sig[(df['close'] < ribbon_mean - k*width) & (df['close'].shift() >= ribbon_mean.shift()-k*width.shift())] = -1
    df2 = df.copy()
    for i,(p,s) in enumerate(emas.items()):
        df2[f'ema_{p}'] = s
    df2['ribbon_mean'] = ribbon_mean
    df2['ribbon_width'] = width
    df2['signal'] = sig
    return df2

def strat_ema_volatility_squeeze(df: pd.DataFrame, params=None):
    """Squeeze: вузький діапазон EMA + низька волатильність -> вибуховий рух."""
    if params is None:
        params = {"ema_short":8, "ema_long":34, "atr_period":14, "atr_thresh_mult":0.6}
    s = ema(df['close'], params['ema_short'])
    l = ema(df['close'], params['ema_long'])
    a = atr(df, params['atr_period'])
    sig = pd.Series(0, index=df.index)
    squeeze = (a < a.rolling(50).mean()*params['atr_thresh_mult'])
    breakout_up = (df['close'] > s) & (s > l)
    breakout_down = (df['close'] < s) & (s < l)
    sig[squeeze & breakout_up] = 1
    sig[squeeze & breakout_down] = -1
    df2 = df.copy()
    df2['s'] = s; df2['l'] = l; df2['atr'] = a; df2['signal'] = sig
    return df2

def strat_ema_rsi_divergence(df: pd.DataFrame, params=None):
    """EMA + RSI divergence (спрощено)."""
    if params is None:
        params = {"ema":21, "rsi_period":14}
    s = ema(df['close'], params['ema'])
    delta = df['close'].diff()
    up = delta.clip(lower=0).rolling(params['rsi_period']).mean()
    down = -delta.clip(upper=0).rolling(params['rsi_period']).mean()
    rsi = 100 - 100/(1 + up/down.replace(0,np.nan))
    sig = pd.Series(0, index=df.index)
    # простий критерій дивергенції: ціна робить новий мін/макс, але RSI — ні.
    price_high = df['close'] > df['close'].shift(1)
    rsi_low_div = (df['close'] < df['close'].shift(1)) & (rsi > rsi.shift(1))
    rsi_high_div = (df['close'] > df['close'].shift(1)) & (rsi < rsi.shift(1))
    sig[rsi_low_div] = 1
    sig[rsi_high_div] = -1
    df2 = df.copy()
    df2['ema'] = s; df2['rsi'] = rsi; df2['signal'] = sig
    return df2

def strat_ema_anchored_mean_reversion(df: pd.DataFrame, params=None):
    """Anchored EMA mean reversion: прив'язка до локальних максимумів/мінімумів."""
    if params is None:
        params = {"ema":34, "lookback":20}
    s = ema(df['close'], params['ema'])
    rolling_max = df['close'].rolling(params['lookback']).max()
    rolling_min = df['close'].rolling(params['lookback']).min()
    sig = pd.Series(0, index=df.index)
    # якщо ціна віддаляється від EMA більше ніж певний процент — входити проти руху
    pct = (df['close'] - s) / s
    sig[pct > 0.03] = -1
    sig[pct < -0.03] = 1
    df2 = df.copy(); df2['ema']=s; df2['signal']=sig
    return df2

def strat_ema_volatility_skew(df: pd.DataFrame, params=None):
    """Volatility skew адаптивний експонент: важчі ваги при високій волатильності."""
    if params is None: params = {"base":10}
    base = params['base']
    atrv = atr(df, 14).fillna(method='bfill')
    # адаптивний span: span = base * (1 + normalized_atr)
    norm = (atrv - atrv.min())/(atrv.max()-atrv.min()+1e-9)
    span = (base * (1 + norm)).round().astype(int).clip(3,200)
    # обчислити EMA по точці: використовуємо перемінний span - будемо приблизно імітувати
    ema_vals = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        p = span.iloc[i]
        ema_vals.iloc[i] = df['close'].iloc[:i+1].ewm(span=p, adjust=False).mean().iloc[-1]
    sig = pd.Series(0, index=df.index)
    sig[(df['close'] > ema_vals*1.01)] = 1
    sig[(df['close'] < ema_vals*0.99)] = -1
    df2 = df.copy(); df2['a_ema']=ema_vals; df2['signal']=sig
    return df2

def strat_quantum_ema(df: pd.DataFrame, params=None):
    """Quantum EMA — художня назва, реалістично: EMA з часовим ваговим згасанням та випадковим шуМом (explorative)."""
    if params is None: params = {"base": 13, "decay":0.999, "noise":0.001}
    base = params['base']; decay = params['decay']; noise = params['noise']
    q_ema = pd.Series(0.0, index=df.index)
    alpha = 2/(base+1)
    q_ema.iloc[0] = df['close'].iloc[0]
    for i in range(1,len(df)):
        a = alpha * (decay ** i)
        q_ema.iloc[i] = a*df['close'].iloc[i] + (1-a)*q_ema.iloc[i-1] + np.random.normal(scale=noise)
    sig = pd.Series(0, index=df.index)
    sig[(df['close'] > q_ema.shift(1)) & (df['close'] > q_ema)] = 1
    sig[(df['close'] < q_ema.shift(1)) & (df['close'] < q_ema)] = -1
    df2 = df.copy(); df2['q_ema']=q_ema; df2['signal']=sig
    return df2

def strat_fractal_ema(df: pd.DataFrame, params=None):
    """Fractal EMA: EMA, що реагує на фрактальні точки (локальні екстремуми)."""
    if params is None: params = {"ema":21, "fractal_look":2}
    s = ema(df['close'], params['ema'])
    n = params['fractal_look']
    is_high = (df['high'] > df['high'].shift(1)) & (df['high'] > df['high'].shift(-1))
    is_low = (df['low'] < df['low'].shift(1)) & (df['low'] < df['low'].shift(-1))
    sig = pd.Series(0, index=df.index)
    sig[is_low & (df['close'] > s)] = 1
    sig[is_high & (df['close'] < s)] = -1
    df2 = df.copy(); df2['ema']=s; df2['signal']=sig
    return df2

def strat_adaptive_ema_atr(df: pd.DataFrame, params=None):
    """Adaptive EMA з ATR як модифікатор ваги (експеримент)."""
    if params is None: params = {"base":20}
    base = params['base']
    a = atr(df, 14).fillna(method='bfill')
    span = (base * (1 + a/a.mean())).round().astype(int).clip(3,200)
    ada = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        p = span.iloc[i]
        ada.iloc[i] = df['close'].iloc[:i+1].ewm(span=p, adjust=False).mean().iloc[-1]
    sig = pd.Series(0, index=df.index)
    sig[(df['close'] > ada)] = 1
    sig[(df['close'] < ada)] = -1
    df2 = df.copy(); df2['ada']=ada; df2['signal']=sig
    return df2

def strat_neural_ema_placeholder(df: pd.DataFrame, params=None):
    """Neural EMA: плейсхолдер для ML-підходу. Тут ми просто імітуємо 'learned' сигнал — на базі ковзного логістичного регресора (very simple)."""
    from sklearn.linear_model import LogisticRegression
    # Простий feature: returns over few horizons
    X = pd.concat([
        df['close'].pct_change(1).fillna(0),
        df['close'].pct_change(3).fillna(0),
        df['close'].pct_change(7).fillna(0)
    ], axis=1).fillna(0)
    # target: наступний день рух
    y = (df['close'].shift(-1) > df['close']).astype(int).fillna(0)
    # тренуємо на першій половині, прогнозуємо на другій
    split = int(len(df)*0.6)
    if split < 50:
        sig = pd.Series(0, index=df.index)
        df2 = df.copy(); df2['signal'] = sig
        return df2
    clf = LogisticRegression(max_iter=200)
    clf.fit(X.iloc[:split], y.iloc[:split])
    preds = clf.predict(X.iloc[split:])
    sig = pd.Series(0, index=df.index)
    sig.iloc[split:] = np.where(preds==1, 1, -1)
    df2 = df.copy(); df2['signal']=sig
    return df2

# --------------------------------------
# Mapping strategy names -> functions
# --------------------------------------
STRATEGIES = {
    "ema_crossover": (strat_ema_crossover, "Класичний EMA crossover (fast/slow)"),
    "ema_ribbon": (strat_ema_ribbon_breakout, "EMA ribbon breakout"),
    "ema_squeeze": (strat_ema_volatility_squeeze, "Volatility squeeze + EMA"),
    "ema_rsi_div": (strat_ema_rsi_divergence, "EMA + RSI divergence"),
    "ema_anchor_revert": (strat_ema_anchored_mean_reversion, "Anchored EMA mean reversion"),
    "ema_vol_skew": (strat_ema_volatility_skew, "EMA з волатильнісним сківом"),
    "quantum_ema": (strat_quantum_ema, "Quantum EMA (експеримент)"),
    "fractal_ema": (strat_fractal_ema, "Fractal EMA"),
    "adaptive_ema": (strat_adaptive_ema_atr, "Adaptive EMA на базі ATR"),
    "neural_ema": (strat_neural_ema_placeholder, "Neural EMA (placeholder ML)")
}

# ----------------------------
# Data fetching
# ----------------------------
async def fetch_ohlcv(symbol: str, timeframe: str, since_days: int = 30, limit=1000):
    """Fetch OHLCV via ccxt (async not guaranteed for all brokers)."""
    # convert days to ms
    since_ms = int((datetime.utcnow() - timedelta(days=since_days)).timestamp() * 1000)
    # try fetch_ohlcv
    try:
        bars = EXCHANGE.fetch_ohlcv(symbol, timeframe=timeframe, since=since_ms, limit=limit)
    except Exception as e:
        # try without since
        bars = EXCHANGE.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    cols = ['timestamp','open','high','low','close','volume']
    df = pd.DataFrame(bars, columns=cols)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

# ----------------------------
# Backtest engine (very simple, position sizing flat)
# ----------------------------
def simple_backtest(df: pd.DataFrame, initial_capital=10000, per_trade_risk=0.01):
    """Простий бектест: входи по 'signal' 1/-1, вихід при протилежному сигналі.
    Передбачається, що торгуємо контрактами з гарантованою ліквідністю — це спрощення.
    Повертає df з колонками equity, positions, trades, pnl.
    """
    df = df.copy()
    df['position'] = 0
    pos = 0
    entry_price = 0.0
    cash = initial_capital
    equity = []
    trades = []
    for i in range(len(df)):
        sig = df['signal'].iloc[i]
        price = df['close'].iloc[i]
        if sig == 1 and pos <= 0:
            # enter long: close short if any
            pos = 1
            entry_price = price
            trades.append(("buy", df.index[i], price))
        elif sig == -1 and pos >= 0:
            pos = -1
            entry_price = price
            trades.append(("sell", df.index[i], price))
        # mark-to-market equity: assume 1 contract scaled by capital*leverage
        equity_val = cash + (price - entry_price) * pos * (initial_capital*0.1)  # simplified
        equity.append(equity_val)
        df['position'].iloc[i] = pos
    df['equity'] = equity
    # basic metrics
    ret = (df['equity'].iloc[-1] - initial_capital)
    drawdown = (df['equity'].cummax() - df['equity']).max()
    return {"df": df, "profit": ret, "max_drawdown": drawdown, "trades": trades}

# ----------------------------
# Telegram command handlers
# ----------------------------
async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    txt = (
        "Привіт! Я бот-симулятор ф'ючерсної революції 🧪.\n"
        "Команди:\n"
        "/strategies - список стратегій\n"
        "/simulate <strategy> <symbol> <tf> <days> - запустити симуляцію\n"
        "/visual <strategy> <symbol> <tf> <days> - графік з сигналами\n"
        "/summon_revolution - прогнати всі стратегії і ранжувати\n"
        "/kneel_forgiveness - якщо хочеш, щоб я став на коліна 😅 (fun + запустить агресивну симуляцію)\n"
        "/risk_report <symbol> <days>\n"
        "Почнемо експерименти. Пам'ятай: це не фінансова порада."
    )
    await update.message.reply_text(txt)

async def strategies_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    lines = []
    for k,(f,desc) in STRATEGIES.items():
        lines.append(f"/simulate {k} SYMBOL TF DAYS  — {desc}")
    await update.message.reply_text("Доступні стратегії:\n" + "\n".join(lines))

async def simulate_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        args = context.args
        if len(args) < 4:
            await update.message.reply_text("Використання: /simulate <strategy> <symbol> <tf> <days>")
            return
        strat_name, symbol, tf, days = args[0], args[1], args[2], int(args[3])
        if strat_name not in STRATEGIES:
            await update.message.reply_text(f"Невідома стратегія {strat_name}. /strategies")
            return
        await update.message.reply_text(f"Завантажую дані для {symbol} ({tf}) за {days} днів...")
        df = await asyncio.get_event_loop().run_in_executor(None, lambda: asyncio.run(fetch_ohlcv_sync(symbol, tf, days)))
        # run strategy
        fn = STRATEGIES[strat_name][0]
        df_strat = fn(df)
        bt = simple_backtest(df_strat)
        txt = f"Результат симуляції {strat_name} на {symbol} за {days} днів:\nProfit: {bt['profit']:.2f}\nMax drawdown: {bt['max_drawdown']:.2f}\nTrades: {len(bt['trades'])}"
        await update.message.reply_text(txt)
    except Exception as e:
        await update.message.reply_text("Помилка при симуляції:\n" + str(e) + "\n" + traceback.format_exc())

# since fetch_ohlcv in this script is sync (ccxt blocking), provide a small sync wrapper
def fetch_ohlcv_sync(symbol, tf, days):
    return asyncio.get_event_loop().run_until_complete(fetch_ohlcv(symbol, tf, days))

async def visual_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        args = context.args
        if len(args) < 4:
            await update.message.reply_text("Використання: /visual <strategy> <symbol> <tf> <days>")
            return
        strat_name, symbol, tf, days = args[0], args[1], args[2], int(args[3])
        if strat_name not in STRATEGIES:
            await update.message.reply_text("Невідома стратегія.")
            return
        await update.message.reply_text("Генерую графік...")
        df = fetch_ohlcv_sync(symbol, tf, days)
        fn = STRATEGIES[strat_name][0]
        df2 = fn(df)
        # малюємо
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(df2.index, df2['close'], label='close')
        if 'fast' in df2.columns and 'slow' in df2.columns:
            ax.plot(df2.index, df2['fast'], label='fast')
            ax.plot(df2.index, df2['slow'], label='slow')
        if 'q_ema' in df2.columns:
            ax.plot(df2.index, df2['q_ema'], label='q_ema')
        # сигнали
        buys = df2[df2['signal'] == 1]
        sells = df2[df2['signal'] == -1]
        ax.scatter(buys.index, buys['close'], marker='^', label='buy', s=60)
        ax.scatter(sells.index, sells['close'], marker='v', label='sell', s=60)
        ax.legend()
        ax.set_title(f"{strat_name} on {symbol}")
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        await update.message.reply_photo(photo=buf)
        plt.close(fig)
    except Exception as e:
        await update.message.reply_text("Помилка при генерації графіку:\n" + str(e))

async def summon_revolution_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Запускаю революцію: тестую всі стратегії (це може зайняти час)...")
    try:
        # мінімальна перевірка: візьмемо BTC/USDT як дефолт
        symbol = "BTC/USDT"
        tf = "1h"
        days = 30
        results = []
        for k,(fn,desc) in STRATEGIES.items():
            await update.message.reply_text(f"Тестую {k} ...")
            df = fetch_ohlcv_sync(symbol, tf, days)
            df2 = fn(df)
            bt = simple_backtest(df2)
            results.append((k, bt['profit'], bt['max_drawdown']))
        # ранжуємо за прибутком
        results_sorted = sorted(results, key=lambda x: x[1], reverse=True)
        txt = "Результати (symbol BTC/USDT, 1h, 30d):\n"
        for r in results_sorted:
            txt += f"{r[0]} — profit: {r[1]:.2f}, maxDD: {r[2]:.2f}\n"
        await update.message.reply_text(txt)
    except Exception as e:
        await update.message.reply_text("Помилка при револьції:\n" + str(e))

async def kneel_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # fun command + agressive sim
    await update.message.reply_text("Стаю на коліна... ОК, запускаю агресивну симуляцію (fun mode).")
    try:
        strat = "ema_crossover"
        df = fetch_ohlcv_sync("BTC/USDT", "1h", 14)
        df2 = STRATEGIES[strat][0](df)
        # модифікатор ризику (агресивний)
        bt = simple_backtest(df2, initial_capital=1000)
        await update.message.reply_text(f"Aggressive sim result: profit {bt['profit']:.2f}, maxDD {bt['max_drawdown']:.2f}")
    except Exception as e:
        await update.message.reply_text("Помилка:\n" + str(e))

async def risk_report_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        args = context.args
        if len(args) < 2:
            await update.message.reply_text("Використання: /risk_report <symbol> <days>")
            return
        symbol, days = args[0], int(args[1])
        df = fetch_ohlcv_sync(symbol, "1h", days)
        vol = df['close'].pct_change().std() * np.sqrt(24*365)  # приблизне annualized
        # max drawdown (price simple)
        series = df['close']
        dd = (series.cummax() - series).max()
        await update.message.reply_text(f"Risk report for {symbol} ({days}d):\nAnnualized vol(approx): {vol:.4f}\nMax price drawdown: {dd:.2f}")
    except Exception as e:
        await update.message.reply_text("Помилка:\n" + str(e))

# ----------------------------
# App setup
# ----------------------------
def build_app(token):
    app = ApplicationBuilder().token(token).build()
    app.add_handler(CommandHandler("start", start_handler))
    app.add_handler(CommandHandler("strategies", strategies_handler))
    app.add_handler(CommandHandler("simulate", simulate_handler))
    app.add_handler(CommandHandler("visual", visual_handler))
    app.add_handler(CommandHandler("summon_revolution", summon_revolution_handler))
    app.add_handler(CommandHandler("kneel_forgiveness", kneel_handler))
    app.add_handler(CommandHandler("risk_report", risk_report_handler))
    return app

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    if not TELE_TOKEN:
        print("Set TELEGRAM_TOKEN env var.")
        exit(1)
    print("Запускаю бота...")
    app = build_app(TELE_TOKEN)
    app.run_polling()