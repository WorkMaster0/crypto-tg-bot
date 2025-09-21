import asyncio
import json
import pandas as pd
import websockets
import requests
import logging

logger = logging.getLogger("ws-manager")

class WebSocketKlineManager:
    def __init__(self, symbols, interval="15m"):
        self.symbols = [s.lower() for s in symbols]
        self.interval = interval
        self.data = {s.upper(): pd.DataFrame(columns=["open", "high", "low", "close", "volume"]) for s in symbols}

    def load_history(self, symbol, limit=500):
        """Разове завантаження історії через REST"""
        try:
            url = "https://api.binance.com/api/v3/klines"
            params = {"symbol": symbol.upper(), "interval": self.interval, "limit": limit}
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            raw = r.json()
            df = pd.DataFrame(raw, columns=[
                "open_time","open","high","low","close","volume",
                "close_time","qav","num_trades","taker_base","taker_quote","ignore"
            ])
            df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
            df.set_index("open_time", inplace=True)
            df = df.astype(float)[["open","high","low","close","volume"]]
            self.data[symbol.upper()] = df
            logger.info(f"[REST] Loaded {len(df)} candles for {symbol}")
        except Exception as e:
            logger.error(f"[REST] Failed to load history for {symbol}: {e}")

    async def _subscribe(self):
        streams = "/".join([f"{s}@kline_{self.interval}" for s in self.symbols])
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
                            continue
                        symbol = kline["s"]
                        ts = pd.to_datetime(kline["t"], unit="ms", utc=True)
                        self.data[symbol].loc[ts] = {
                            "open": float(kline["o"]),
                            "high": float(kline["h"]),
                            "low": float(kline["l"]),
                            "close": float(kline["c"]),
                            "volume": float(kline["v"])
                        }
            except Exception as e:
                logger.error(f"[WebSocket] error: {e}, reconnecting in 5s...")
                await asyncio.sleep(5)

    def start(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._subscribe())

    def get_klines(self, symbol, limit=500):
        df = self.data.get(symbol.upper())
        if df is None or len(df) < 10:
            return None
        return df.tail(limit).copy()