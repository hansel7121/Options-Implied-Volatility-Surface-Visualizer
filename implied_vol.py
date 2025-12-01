import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import plotly.graph_objects as go

voo = yf.Ticker("VOO")
exps = voo.options

def plot_implied_volatility_strikes(calls):
    chain = voo.option_chain(exps[0])
    if calls:
        plt.plot(chain.calls['strike'], chain.calls['impliedVolatility'])
    else:
        plt.plot(chain.puts['strike'], chain.puts['impliedVolatility'])
    plt.title("Implied Volatility")
    plt.xlabel("Strike")
    plt.ylabel("Implied Volatility")
    plt.show()

def plot_implied_volatility_expirations():
    ivs = []
    current_price = voo.history(period="1d")["Close"].iloc[-1]

    for date in exps[:12]:
        chain = voo.option_chain(date)
        c = chain.calls
        atm_call = c.iloc[(c['strike'] - current_price).abs().argsort()[:1]]
        call_iv = atm_call['impliedVolatility'].values[0]

        ivs.append({'date': date, 'call_iv': call_iv})

    df = pd.DataFrame(ivs)
    df['date'] = pd.to_datetime(df['date'])
    plt.plot(df['date'], df['call_iv'], marker='o', linestyle='-', color='b', label='Call IV')
    plt.title("Implied Volatility Term Structure: VOO (ATM Calls)")
    plt.xlabel("Expiration Date")
    plt.ylabel("Implied Volatility")
    plt.legend()
    plt.show()

plot_implied_volatility_strikes(True)
plot_implied_volatility_expirations()

