import requests

BINANCE_API = "https://api.binance.com/api/v3/ticker/price"

def get_price(symbol: str) -> float:
    response = requests.get(BINANCE_API, params={"symbol": symbol.replace("/", "")})
    data = response.json()
    return float(data["price"])