# =========================
# app.py (Part 1)
# =========================

import os
import json
import time
import datetime
import logging
import threading

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from flask import Flask, request
from binance.client import Client
from binance.exceptions import BinanceAPIException

from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb

from telegram import Bot
from telegram.constants import ParseMode

# -------------------------
# Конфіг
# -------------------------

# Твої API ключі Binance
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "your_binance_api_key")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "your_binance_api_secret")

# Telegram
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "your_telegram_bot_token")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "your_chat_id")

# Параметри
TIMEFRAME = "15m"   # ТФ для аналізу
LIMIT = 500         # кількість свічок
SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]  # що скануємо

# Ініціалізація клієнтів
client = Client(api_key=BINANCE_API_KEY, api_secret=BINANCE_API_SECRET)
bot = Bot(token=TELEGRAM_TOKEN)

# Flask API
app = Flask(__name__)

# Логи
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------
# Функції Binance
# -------------------------

def get_klines(symbol, interval=TIMEFRAME, limit=LIMIT):
    """Отримати історію свічок"""
    try:
        data = client.get_klines(symbol=symbol, interval=interval, limit=limit)
        df = pd.DataFrame(data, columns=[
            "time","open","high","low","close","volume","c1","c2","c3","c4","c5","c6"
        ])
        df["time"] = pd.to_datetime(df["time"], unit="ms")
        df["open"] = df["open"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)
        df["close"] = df["close"].astype(float)
        df["volume"] = df["volume"].astype(float)
        return df[["time","open","high","low","close","volume"]]
    except BinanceAPIException as e:
        logger.error(f"Binance error: {e}")
        return None

# -------------------------
# Індикатори
# -------------------------

def ema(df, period=20, column="close"):
    """EMA"""
    return df[column].ewm(span=period, adjust=False).mean()

def atr(df, period=14):
    """ATR"""
    high_low = df["high"] - df["low"]
    high_close = np.abs(df["high"] - df["close"].shift())
    low_close = np.abs(df["low"] - df["close"].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    return true_range.rolling(period).mean()

def obv(df):
    """On-Balance Volume"""
    obv_vals = [0]
    for i in range(1, len(df)):
        if df["close"].iloc[i] > df["close"].iloc[i-1]:
            obv_vals.append(obv_vals[-1] + df["volume"].iloc[i])
        elif df["close"].iloc[i] < df["close"].iloc[i-1]:
            obv_vals.append(obv_vals[-1] - df["volume"].iloc[i])
        else:
            obv_vals.append(obv_vals[-1])
    return pd.Series(obv_vals, index=df.index)

def heikin_ashi(df):
    """Heikin Ashi свічки"""
    ha_df = df.copy()
    ha_df["HA_close"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    ha_df["HA_open"] = (df["open"].shift() + df["close"].shift()) / 2
    ha_df["HA_open"].iloc[0] = (df["open"].iloc[0] + df["close"].iloc[0]) / 2
    ha_df["HA_high"] = ha_df[["HA_open","HA_close","high"]].max(axis=1)
    ha_df["HA_low"] = ha_df[["HA_open","HA_close","low"]].min(axis=1)
    return ha_df[["HA_open","HA_close","HA_high","HA_low"]]
    
    # =========================
# app.py (Part 2)
# =========================

# -------------------------
# Патерни
# -------------------------

def detect_pre_top(df, window=5):
    """
    Виявлення pre-top (різкий стрибок ціни і швидке падіння)
    """
    signals = []
    for i in range(window, len(df)):
        recent = df["close"].iloc[i-window:i]
        max_price = recent.max()
        if df["close"].iloc[i] < max_price * 0.97:  # падіння 3% від локального піку
            signals.append(1)
        else:
            signals.append(0)
    return pd.Series([0]*window + signals, index=df.index)

def detect_volume_spike(df, factor=2):
    """
    Спайк об'єму (в 2 рази більше середнього)
    """
    avg_vol = df["volume"].rolling(20).mean()
    return (df["volume"] > avg_vol * factor).astype(int)

def detect_orderflow(df):
    """
    Спрощений аналіз ордерфлоу:
    Якщо свічка закрилась вище відкриття → покупці сильні
    Якщо нижче → продавці
    """
    return (df["close"] > df["open"]).astype(int) - (df["close"] < df["open"]).astype(int)

# -------------------------
# ML ансамбль
# -------------------------

class EnsembleModel:
    def __init__(self):
        self.rf = RandomForestClassifier(n_estimators=100, random_state=42)
        self.lgb_model = None
        self.trained = False

    def prepare_data(self, df):
        """
        Формуємо фічі для ML
        """
        X = pd.DataFrame()
        X["ema_fast"] = ema(df, 9)
        X["ema_slow"] = ema(df, 21)
        X["atr"] = atr(df, 14)
        X["obv"] = obv(df)
        X["volume"] = df["volume"]
        X["return"] = df["close"].pct_change()
        X = X.fillna(0)

        y = (df["close"].shift(-1) > df["close"]).astype(int)  # 1 = зростання, 0 = падіння
        y = y.fillna(0)
        return X, y

    def fit(self, df):
        X, y = self.prepare_data(df)

        # RandomForest
        self.rf.fit(X, y)

        # LightGBM
        train_data = lgb.Dataset(X, label=y)
        params = {"objective": "binary", "verbosity": -1, "seed": 42}
        self.lgb_model = lgb.train(params, train_data, num_boost_round=50)

        self.trained = True

    def predict(self, df):
        if not self.trained:
            return pd.Series([0]*len(df), index=df.index)

        X, _ = self.prepare_data(df)

        rf_pred = self.rf.predict_proba(X)[:,1]
        lgb_pred = self.lgb_model.predict(X)

        # Ансамбль — середнє
        final_pred = (rf_pred + lgb_pred) / 2
        return pd.Series(final_pred, index=df.index)
        
        # =========================
# app.py (Part 3)
# =========================

# -------------------------
# Auto-Optimizer сигналів
# -------------------------

class SignalOptimizer:
    """
    Визначає ваги сигналів (pre-top, volume spike, orderflow, ML) для confidence score
    """
    def __init__(self, weights=None):
        # ваги: pre_top, vol_spike, orderflow, ml
        self.weights = weights or {"pre_top":0.3, "vol_spike":0.2, "orderflow":0.2, "ml":0.3}

    def optimize(self, df, ensemble_model):
        """
        Простий градієнтний підбір ваг:
        - перебираємо комбінації ваг
        - вибираємо max accuracy на історії
        """
        best_score = 0
        best_weights = self.weights.copy()

        import itertools
        vals = [0.0,0.1,0.2,0.3,0.4,0.5]
        for w_pre, w_vol, w_of, w_ml in itertools.product(vals, repeat=4):
            total = w_pre + w_vol + w_of + w_ml
            if total == 0: continue
            norm = 1/total
            w_dict = {"pre_top":w_pre*norm, "vol_spike":w_vol*norm, "orderflow":w_of*norm, "ml":w_ml*norm}
            pred = self.compute_confidence(df, ensemble_model, w_dict)
            score = ((pred > 0.5) == (df["close"].shift(-1) > df["close"])).mean()
            if score > best_score:
                best_score = score
                best_weights = w_dict
        self.weights = best_weights
        return best_weights, best_score

    def compute_confidence(self, df, ensemble_model, weights=None):
        weights = weights or self.weights
        # поєднуємо всі сигнали
        pre_top_sig = detect_pre_top(df)
        vol_sig = detect_volume_spike(df)
        of_sig = detect_orderflow(df)
        ml_sig = ensemble_model.predict(df)

        conf = (
            pre_top_sig * weights["pre_top"] +
            vol_sig * weights["vol_spike"] +
            of_sig * weights["orderflow"] +
            ml_sig * weights["ml"]
        )
        return conf

# -------------------------
# Backtester
# -------------------------

class Backtester:
    """
    Простий backtest стратегії
    """
    def __init__(self, df, conf_series):
        self.df = df.copy()
        self.conf = conf_series
        self.df["conf"] = conf_series

    def run(self, threshold=0.6):
        """
        Виконуємо backtest
        """
        df = self.df.copy()
        df["position"] = (df["conf"] > threshold).astype(int)
        df["return"] = df["close"].pct_change().shift(-1)
        df["strategy"] = df["position"] * df["return"]
        cum_ret = (1 + df["strategy"].fillna(0)).cumprod()
        total_return = cum_ret.iloc[-1] - 1
        win_rate = (df["strategy"] > 0).mean()
        return {"total_return": total_return, "win_rate": win_rate, "df": df}

    def plot_results(self, df=None, symbol="BTCUSDT"):
        df = df or self.df
        plt.figure(figsize=(12,6))
        plt.plot(df["time"], df["close"], label="Price", color="blue")
        plt.plot(df["time"], df["conf"], label="Confidence", color="orange")
        buy_signals = df[df["position"]==1]
        plt.scatter(buy_signals["time"], buy_signals["close"], marker="^", color="green", label="BUY")
        plt.title(f"Backtest {symbol}")
        plt.xlabel("Time")
        plt.ylabel("Price / Confidence")
        plt.legend()
        plt.tight_layout()
        filename = f"{symbol}_backtest.png"
        plt.savefig(filename)
        plt.close()
        return filename
        
        # =========================
# app.py (Part 4)
# =========================

# -------------------------
# Telegram повідомлення
# -------------------------

def send_telegram_message(text, img_path=None):
    try:
        if img_path:
            with open(img_path, "rb") as f:
                bot.send_photo(chat_id=TELEGRAM_CHAT_ID, photo=f, caption=text, parse_mode=ParseMode.MARKDOWN)
        else:
            bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=text, parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        logger.error(f"Telegram send error: {e}")

# -------------------------
# Flask API для webhook
# -------------------------

@app.route("/scan_now", methods=["GET"])
def scan_now():
    threading.Thread(target=run_scan, daemon=True).start()
    return {"status":"scanning"}

@app.route("/status", methods=["GET"])
def status():
    return {"status":"ok", "time":str(datetime.datetime.utcnow())}

# -------------------------
# Основний запуск сканера
# -------------------------

def run_scan():
    for symbol in SYMBOLS:
        df = get_klines(symbol)
        if df is None or len(df)<50:
            continue

        # ML ансамбль
        model = EnsembleModel()
        model.fit(df)
        conf_series = model.predict(df)

        # Оптимізація сигналів
        optimizer = SignalOptimizer()
        optimizer.optimize(df, model)
        conf_series = optimizer.compute_confidence(df, model)

        # Backtest
        backtester = Backtester(df, conf_series)
        results = backtester.run()
        img_file = backtester.plot_results(df, symbol=symbol)

        # Відправка в Telegram
        text = (
            f"*{symbol} Scan*\n"
            f"Total Return: {results['total_return']*100:.2f}%\n"
            f"Win Rate: {results['win_rate']*100:.2f}%\n"
            f"Last Price: {df['close'].iloc[-1]:.2f}"
        )
        send_telegram_message(text, img_path=img_file)
        time.sleep(1)

# -------------------------
# Flask запуск
# -------------------------

if __name__ == "__main__":
    logger.info("Starting AI Trading Hub")
    threading.Thread(target=run_scan, daemon=True).start()
    app.run(host="0.0.0.0", port=5000)