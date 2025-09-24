import json
import os

FILE = "src/portfolio/user_data.json"

def load_portfolio():
    if os.path.exists(FILE):
        with open(FILE) as f:
            return json.load(f)
    return {"tickers": []}

def save_portfolio(portfolio):
    with open(FILE, "w") as f:
        json.dump(portfolio, f)

def add_stock(ticker):
    portfolio = load_portfolio()

    if "tickers" not in portfolio:
        portfolio["tickers"] = []

    if ticker not in portfolio["tickers"]:
        portfolio["tickers"].append(ticker)
        save_portfolio(portfolio)
        print(f"Ação {ticker} adicionada à carteira.")
    else:
        print(f"Ação {ticker} já está na carteira.")

def remove_stock(ticker):
    portfolio = load_portfolio()

    if "tickers" in portfolio and ticker in portfolio["tickers"]:
        portfolio["tickers"].remove(ticker)
        save_portfolio(portfolio)
        print(f"Ação {ticker} removida da carteira.")
    else:
        print(f"Ação {ticker} não encontrada na carteira.")

def show_portfolio():
    return load_portfolio()["tickers"]