import asyncio
import json
import pandas as pd
import websockets
import logging
import os

logger = logging.getLogger("ws-manager")

class WebSocketKlineManager:
    def __init__(self, symbols, interval="15m", state_dir="ws_data"):
        self.symbols = [s.upper() for s in symbols]
        self.interval = interval
        self.state_dir = state_dir
        os.makedirs(state_dir, exist_ok=True)
        self.data = {}

        # завантаження історії з диска
        for sym in self.symbols:
            path = os.path.join(state_dir, f"{sym}.parquet")
            if os.path.exists(path):
                try:
                    df = pd.read_parquet(path)
                    self.data[sym] = df
                    logger.info(f"[DISK] Loaded {len(df)} candles for {sym}")
                except Exception as e:
                    logger.error(f"[DISK] Failed to load {sym}: {e}")
                    self.data[sym] = pd.DataFrame(columns=["open","high","low","close","volume"])
            else:
                self.data[sym] = pd.DataFrame(columns=["open","high","low","close","volume"])

    async def _subscribe(self):
        streams = "/".join([f"{s.lower()}@kline_{self.interval}" for s in self.symbols])
        url = f"wss://stream.binance.com:9443/stream?streams={streams}"
        while True:
            try:
                async with websockets.connect(url, ping_interval=20) as ws:
                    async for msg in ws:
                        data = json.loads(msg)
                        if "data" not in data:
                            continue
                        kline = data["data"]["k"]
                        if not kline["x"]:
                            continue  # свічка не закрилася
                        symbol = kline["s"]
                        ts = pd.to_datetime(kline["t"], unit="ms", utc=True)
                        self.data[symbol].loc[ts] = {
                            "open": float(kline["o"]),
                            "high": float(kline["h"]),
                            "low": float(kline["l"]),
                            "close": float(kline["c"]),
                            "volume": float(kline["v"])
                        }
                        # зберігаємо на диск після кожної свічки
                        self._save_symbol(symbol)
            except Exception as e:
                logger.error(f"[WebSocket] error: {e}, reconnecting in 5s...")
                await asyncio.sleep(5)

    def _save_symbol(self, symbol):
        try:
            path = os.path.join(self.state_dir, f"{symbol}.parquet")
            self.data[symbol].to_parquet(path)
        except Exception as e:
            logger.error(f"[DISK] Failed to save {symbol}: {e}")

    def start(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._subscribe())

    def get_klines(self, symbol, limit=500):
        df = self.data.get(symbol.upper())
        if df is None or len(df) < 10:
            return None
        return df.tail(limit).copy()