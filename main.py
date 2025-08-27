import os
import threading
from flask import Flask

# Створюємо Flask-сервер для health-check
app = Flask(__name__)

@app.route("/health")
def health():
    return "OK", 200

def run_web():
    port = int(os.environ.get("PORT", 5000))  # Render автоматично задає PORT
    app.run(host="0.0.0.0", port=port)

# Запускаємо сервер у окремому потоці, щоб не блокував polling
threading.Thread(target=run_web).start()
