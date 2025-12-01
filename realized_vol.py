import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

voo = yf.download("VOO", period="1y", auto_adjust=True)["Close"]
voo_returns = voo.pct_change().dropna()


voo_returns.plot()
plt.title("VOO Returns")
plt.show()

histogram = voo_returns.hist(bins=50, figsize=(10, 5))
plt.title("VOO Returns Histogram")
plt.show()

print("Mean %: ", voo_returns.mean() * 100)
print("STDEV %: ", voo_returns.std() * 100)
print("Skewness: ", voo_returns.skew())
print("Kurtosis: ", voo_returns.kurt())
