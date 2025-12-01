import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

# 1. Setup
ticker = "VOO"
voo = yf.Ticker(ticker)

# Get current price to find the "At-The-Money" (ATM) strike
hist = voo.history(period="1d")
current_price = hist['Close'].iloc[-1]
print(f"Current Price: ${current_price:.2f}")

exps = voo.options
term_structure = []

print("Fetching data...")

# 2. Loop through ALL expirations
for date in exps:
    try:
        # Fetch the chain
        chain = voo.option_chain(date)
        # We usually use PUTS for volatility analysis (fear gauge)
        options = chain.puts

        # Filter for liquidity (optional but good practice)
        options = options[options['volume'] > 0]

        if options.empty:
            continue

        # 3. Find the ATM Option
        # Calculate difference between Strike and Current Price
        # Sort by that difference and take the top 1 (closest)
        atm_option = options.iloc[(options['strike'] - current_price).abs().argsort()[:1]]

        # Extract the IV
        iv = atm_option['impliedVolatility'].values[0]

        term_structure.append({'date': date, 'iv': iv})

    except Exception as e:
        # Sometimes data is missing for specific dates
        pass

# 3. Create DataFrame
df = pd.DataFrame(term_structure)

# Convert string dates to datetime objects so matplotlib handles them correctly
df['date'] = pd.to_datetime(df['date'])

# 4. Plot
plt.figure(figsize=(10, 6))
plt.plot(df['date'], df['iv'], marker='o', linestyle='-', color='b')

plt.title(f"Volatility Term Structure: {ticker} (ATM Puts)")
plt.xlabel("Expiration Date")
plt.ylabel("Implied Volatility")
plt.grid(True)
plt.show()