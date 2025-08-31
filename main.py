# main.py
import os
import threading
from flask import Flask
from app.bot import bot
import app.handlers

app = Flask(__name__)

@app.route("/")
def home():
    return "✅ Crypto Bot is running!"