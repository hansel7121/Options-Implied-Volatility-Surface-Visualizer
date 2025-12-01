import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from scipy.optimize import brentq
import matplotlib.pyplot as plt


def black_scholes_price(sigma, S, K, T, r, q, option_type="call"):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    return price


def calculate_iv(market_price, S, K, T, r, q, option_type="call"):
    def objective(sigma):
        return black_scholes_price(sigma, S, K, T, r, q, option_type) - market_price

    try:
        return brentq(objective, 0.001, 3.0)
    except:
        return np.nan


def plot_implied_volatility_strikes():
    chain = voo.option_chain(exps[0])
    plt.plot(chain.calls["strike"], chain.calls["impliedVolatility"])
    plt.plot(chain.puts["strike"], chain.puts["impliedVolatility"])
    plt.title("Implied Volatility")
    plt.legend(["Calls", "Puts"])
    plt.xlabel("Strike")
    plt.ylabel("Implied Volatility")
    plt.show()


ticker = "VOO"
voo = yf.Ticker(ticker)
current_price = voo.history(period="1d")["Close"].iloc[-1]
exps = voo.options
risk_free_rate = 0.046
dividend_yield = 0.013
ivs = []

for exp in exps[:12]:
    chain = voo.option_chain(exp)
    days_to_exp = (pd.to_datetime(exp) - pd.to_datetime("today")).days
    t_normalized = days_to_exp / 365

    calls = chain.calls
    puts = chain.puts

    atm_calls = calls.iloc[(calls["strike"] - current_price).abs().argsort()[:1]].iloc[
        0
    ]
    atm_puts = puts.iloc[(puts["strike"] - current_price).abs().argsort()[:1]].iloc[0]

    call_price = (atm_calls["bid"] + atm_calls["ask"]) / 2
    put_price = (atm_puts["bid"] + atm_puts["ask"]) / 2

    call_iv = calculate_iv(
        call_price,
        current_price,
        atm_calls["strike"],
        t_normalized,
        risk_free_rate,
        dividend_yield,
        option_type="call",
    )
    put_iv = calculate_iv(
        put_price,
        current_price,
        atm_puts["strike"],
        t_normalized,
        risk_free_rate,
        dividend_yield,
        option_type="put",
    )

    ivs.append((days_to_exp, call_iv, put_iv))

ivs = pd.DataFrame(ivs, columns=["days_to_exp", "call_iv", "put_iv"])

plot_implied_volatility_strikes()

plt.plot(ivs["days_to_exp"], ivs["call_iv"], label="Call IV")
plt.plot(ivs["days_to_exp"], ivs["put_iv"], label="Put IV")
plt.xlabel("Days to Expiration")
plt.ylabel("Implied Volatility")
plt.legend()
plt.show()
