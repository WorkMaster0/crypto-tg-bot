# crypto-tg-bot

Телеграм-бот з аналітикою крипторинку: рівні підтримки/опору, сигнали Long/Short, графіки, watch-лист, бек-тест.

## Команди
- `/start` — допомога
- `/analyze BTCUSDT [1h]`
- `/levels BTCUSDT [1h]`
- `/signal BTCUSDT [1h]`
- `/backtest BTCUSDT [1h] [400]`
- `/watch add BTCUSDT [1h] | /watch remove BTCUSDT | /watch list`
- `/setrisk 1.0`
- `/whatif BTCUSDT 65000 63500 1000`
- `/heatmap`
- `/alpha` (сканер стискання ATR%)

## Запуск локально
```bash
python -m venv .venv
source .venv/bin/activate  # або .venv\Scripts\activate у Windows
pip install -r requirements.txt
cp .env.example .env  # або створити .env з токеном
python main.py