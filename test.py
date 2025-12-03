import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from scipy.optimize import brentq
import matplotlib.pyplot as plt
import plotly.graph_objects as go


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


def plot_implied_volatility_strikes(expiration):
    chain = voo.option_chain(expiration)
    calls = chain.calls
    puts = chain.puts
    call_ivs = []
    put_ivs = []
    for i in range(len(calls)):
        call_iv = calculate_iv((calls["bid"].iloc[i] + calls["ask"].iloc[i]) / 2, current_price, calls["strike"].iloc[i], (pd.to_datetime(expiration) - pd.to_datetime("today")).days / 365, risk_free_rate, dividend_yield, option_type="call")
        call_ivs.append((calls["strike"].iloc[i], call_iv))
    for i in range(len(puts)):
        put_iv = calculate_iv((puts["bid"].iloc[i] + puts["ask"].iloc[i]) / 2, current_price, puts["strike"].iloc[i], (pd.to_datetime(expiration) - pd.to_datetime("today")).days / 365, risk_free_rate, dividend_yield, option_type="put")
        put_ivs.append((puts["strike"].iloc[i], put_iv))
    call_ivs = pd.DataFrame(call_ivs, columns=["strike", "iv"])
    put_ivs = pd.DataFrame(put_ivs, columns=["strike", "iv"])
    plt.plot(call_ivs["strike"], call_ivs["iv"], label="Call IV")
    plt.plot(put_ivs["strike"], put_ivs["iv"], label="Put IV")
    plt.title("Implied Volatility")
    plt.legend()
    plt.xlabel("Strike")
    plt.ylabel("Implied Volatility")
    plt.show()

def plot_implied_volatility_expirations(strike_price):
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


    plt.plot(ivs["days_to_exp"], ivs["call_iv"], label="Call IV")
    plt.plot(ivs["days_to_exp"], ivs["put_iv"], label="Put IV")
    plt.xlabel("Days to Expiration")
    plt.ylabel("Implied Volatility")
    plt.legend()
    plt.show()

def get_strikes():
    chain = voo.option_chain(exps[0])
    calls = chain.calls
    return calls["strike"].unique()

def volatility_surface(strikes):
    surface = []
    count = 0
    i = 0
    while count < 8:
        i += 1
        chain = voo.option_chain(exps[i])
        calls = chain.calls
        days_to_exp = (pd.to_datetime(exps[i]) - pd.to_datetime("today")).days
        if days_to_exp < 30:
            continue
        count += 1
        t_normalized = days_to_exp / 365
        calls['mid_price'] = (calls['bid'] + calls['ask']) / 2

        for strike in strikes:
            contract_row = calls[calls['strike'] == strike]
            if not contract_row.empty:
                iv = calculate_iv(contract_row['mid_price'].iloc[0], current_price, strike, t_normalized, risk_free_rate, dividend_yield, option_type="call")
                surface.append({"strike": strike, "expiration": exps[i], "iv": iv})
            else:
                pass
    surface = pd.DataFrame(surface)
    surface = surface.pivot(index="strike", columns="expiration", values="iv")
    surface = surface.interpolate(method='linear', axis=1)
    surface.to_csv('vol_surface.csv')
    return surface

def plot_surface(surface):
    surface.columns = pd.to_datetime(surface.columns)
    x_data = surface.columns
    y_data = surface.index
    z_data = surface.values

    fig = go.Figure(data=[go.Surface(z=z_data, x=x_data, y=y_data)])
    fig.update_layout(
        title='Volatility Surface',
        scene=dict(
            xaxis_title='Expiration',
            yaxis_title='Strike',
            zaxis_title='Implied Volatility'
        ),
    )
    fig.show(renderer="browser")

ticker = "VOO"
voo = yf.Ticker(ticker)
current_price = voo.history(period="1d")["Close"].iloc[-1]
exps = voo.options
risk_free_rate = 0.046
dividend_yield = 0.013
strikes = get_strikes()
surface = volatility_surface(strikes)
plot_surface(surface)

plot_implied_volatility_strikes(exps[0])
plot_implied_volatility_expirations(current_price)
