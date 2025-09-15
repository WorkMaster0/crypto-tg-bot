import os
import asyncio
import logging
import numpy as np
import ccxt
from typing import Dict, Optional, List
from scipy import stats, signal
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, ContextTypes

# Логування
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("TEPC_Bot")

# ====================== КЛАС БОТА ======================
class TEPCBot:
    def __init__(self, token: str):
        self.token = token
        self.app = Application.builder().token(token).build()

        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })

        # Хендлери
        self.app.add_handler(CommandHandler("start", self.start_command))
        self.app.add_handler(CommandHandler("echo_phase", self.echo_phase_command))

    # ----------- START ----------- #
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        keyboard = [
            [InlineKeyboardButton("🔮 Echo Phase", callback_data="echo_phase")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text(
            "🚀 **TEPC STRATEGY BOT**\n\n"
            "🔮 Temporal Echo Phase Convergence\n"
            "Абсолютно нова стратегія для фʼючерсів.\n\n"
            "Команди:\n"
            "• /echo_phase – запуск стратегії",
            reply_markup=reply_markup,
            parse_mode="Markdown"
        )

    # ----------- ОСНОВНА КОМАНДА ----------- #
    async def echo_phase_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        msg = await update.message.reply_text("🔍 Аналізую Echo Phase...")
        try:
            symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
            results = []
            for s in symbols:
                res = await self.analyze_echo_pattern(s, lookbacks=[12, 36, 144], timeframe="1h", limit=300)
                if res:
                    results.append(res)

            if not results:
                await msg.edit_text("📭 Echo Phase: сигналів не знайдено.")
                return

            results.sort(key=lambda x: x["convergence_score"], reverse=True)

            out = "🔮 **ECHO PHASE CONVERGENCE SIGNALS:**\n\n"
            for r in results[:3]:
                out += f"• {r['symbol']} — Score: {r['convergence_score']:.2f}\n"
                out += f"   Phase: {r['phase_direction']}\n"
                out += f"   Echo Level: {r['echo_level']:.2f}\n"
                out += f"   Liquidity Proximity: {r['liquidity_proximity']:.3f}\n\n"

            await msg.edit_text(out, parse_mode="Markdown")

        except Exception as e:
            logger.error(f"Echo phase error: {e}")
            await msg.edit_text("❌ Помилка Echo Phase аналізу")

    # ----------- АНАЛІЗ СИМВОЛУ ----------- #
    async def analyze_echo_pattern(self, symbol: str, lookbacks: list, timeframe: str = "1h", limit: int = 300) -> Optional[Dict]:
        try:
            ohlcv = await self.get_ohlcv(symbol, timeframe, limit)
            if not ohlcv or len(ohlcv) < max(lookbacks) + 50:
                return None

            closes = np.array([x[4] for x in ohlcv])
            highs = np.array([x[2] for x in ohlcv])
            lows = np.array([x[3] for x in ohlcv])
            volumes = np.array([x[5] for x in ohlcv])

            phase_corr = {}
            for lb in lookbacks:
                series = closes[-lb*2:]
                phase_corr[lb] = self.calculate_phase_correlation(series, lb)

            convergence_score = self.calculate_echo_convergence(phase_corr)
            if convergence_score < 0.55:
                return None

            echo_level = self.generate_echo_level(closes, phase_corr)
            liquidity_proximity = self.echo_liquidity_proximity(echo_level, highs, lows, volumes)

            phase_direction = "LONG" if np.mean([v["phase_dir"] for v in phase_corr.values()]) > 0 else "SHORT"

            return {
                "symbol": symbol,
                "convergence_score": convergence_score,
                "phase_map": phase_corr,
                "echo_level": echo_level,
                "liquidity_proximity": liquidity_proximity,
                "phase_direction": phase_direction
            }
        except Exception as e:
            logger.error(f"analyze_echo_pattern error for {symbol}: {e}")
            return None

    # ----------- ФУНКЦІЇ АНАЛІЗУ ----------- #
    def calculate_phase_correlation(self, series: np.ndarray, window: int) -> Dict:
        n = len(series)
        if n < window * 2:
            return {"phase_corr": 0.0, "phase_dir": 0}

        a = series[-window*2:-window]
        b = series[-window:]

        a_d = stats.zscore(a) if np.std(a) != 0 else a - np.mean(a)
        b_d = stats.zscore(b) if np.std(b) != 0 else b - np.mean(b)

        try:
            ha = signal.hilbert(a_d)
            hb = signal.hilbert(b_d)
            phase_a = np.angle(ha)
            phase_b = np.angle(hb)
            sin_diff = np.sin(phase_a - phase_b)
            phase_corr = 1 - (np.std(sin_diff) / 2.0)
            dir_score = np.sign(np.mean(np.diff(b_d)) - np.mean(np.diff(a_d)))
            return {"phase_corr": float(np.clip(phase_corr, 0, 1)), "phase_dir": int(dir_score)}
        except Exception:
            corr = np.corrcoef(a_d, b_d)[0, 1] if np.std(a_d) > 0 and np.std(b_d) > 0 else 0.0
            dir_score = 1 if np.mean(np.diff(b_d)) > np.mean(np.diff(a_d)) else -1
            return {"phase_corr": float(np.clip((corr + 1) / 2, 0, 1)), "phase_dir": dir_score}

    def calculate_echo_convergence(self, phase_map: Dict) -> float:
        vals = [v["phase_corr"] for v in phase_map.values()]
        dirs = [v["phase_dir"] for v in phase_map.values()]
        base = float(np.mean(vals))
        dir_bonus = 0.15 if len(set(dirs)) == 1 else 0.0
        momentum_bonus = 0.10 if base > 0.8 else 0.0
        return float(np.clip(base + dir_bonus + momentum_bonus, 0.0, 1.0))

    def generate_echo_level(self, closes: np.ndarray, phase_map: Dict) -> float:
        weights, centers = [], []
        for lb, v in phase_map.items():
            w = v["phase_corr"] * (1 + abs(v["phase_dir"]))
            centers.append(np.mean(closes[-lb:]))
            weights.append(w)
        if sum(weights) == 0:
            return float(closes[-1])
        return float(np.dot(weights, centers) / sum(weights))

    def echo_liquidity_proximity(self, echo_level: float, highs: np.ndarray, lows: np.ndarray, volumes: np.ndarray) -> float:
        price_levels = np.linspace(np.min(lows), np.max(highs), 200)
        vol_at_levels = []
        for i in range(len(price_levels) - 1):
            mask = (highs >= price_levels[i]) & (lows <= price_levels[i+1])
            vol_at_levels.append(np.sum(volumes[mask]))
        vol_at_levels = np.array(vol_at_levels)
        idx = np.argmin(np.abs(price_levels - echo_level))
        local_vol = vol_at_levels[max(0, idx-2):min(len(vol_at_levels), idx+3)]
        mean_local = float(np.mean(local_vol)) if local_vol.size > 0 else 0.0
        mean_global = float(np.mean(vol_at_levels)) if vol_at_levels.size > 0 else 1.0
        proximity = float(np.clip(1.0 - abs(mean_local - mean_global) / (mean_global + 1e-9), 0.0, 1.0))
        return proximity

    # ----------- OHLCV ----------- #
    async def get_ohlcv(self, symbol: str, timeframe: str = "1h", limit: int = 100) -> Optional[List]:
        try:
            loop = asyncio.get_event_loop()
            ohlcv = await loop.run_in_executor(
                None, lambda: self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            )
            return ohlcv
        except Exception as e:
            logger.error(f"Помилка отримання даних {symbol}: {e}")
            return None

    # ----------- ЗАПУСК ----------- #
    async def run(self):
        logger.info("🚀 Запуск TEPC Strategy Bot...")
        await self.app.initialize()
        await self.app.start()
        await self.app.updater.start_polling()
        logger.info("✅ TEPC Bot працює!")

# ====================== MAIN ======================
async def main():
    BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    if not BOT_TOKEN:
        logger.error("Встановіть TELEGRAM_BOT_TOKEN")
        return
    bot = TEPCBot(BOT_TOKEN)
    await bot.run()

if __name__ == "__main__":
    asyncio.run(main())