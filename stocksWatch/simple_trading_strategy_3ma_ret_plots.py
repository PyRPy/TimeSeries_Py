import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Download data
data = yf.download('AAPL', start='2020-01-01', end='2025-04-11')
data = data[['Close']].copy()

# Calculate moving averages
data['SMA20'] = data['Close'].rolling(window=20).mean()
data['SMA50'] = data['Close'].rolling(window=50).mean()
data['SMA200'] = data['Close'].rolling(window=200).mean()  # Long-term trend

# Generate trading signal
data['Signal'] = 0
data.loc[data['SMA20'] > data['SMA50'], 'Signal'] = 1
data['Position'] = data['Signal'].shift(1)

# Calculate returns
data['Daily Return'] = data['Close'].pct_change()
data['Strategy Return'] = data['Position'] * data['Daily Return']
data['Cumulative Market Return'] = (1 + data['Daily Return']).cumprod()
data['Cumulative Strategy Return'] = (1 + data['Strategy Return']).cumprod()

# Create subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# Plot 1: Price + SMAs
ax1.plot(data['Close'], label='Close Price', alpha=0.6)
ax1.plot(data['SMA20'], label='SMA20', linestyle='--', color='orange')
ax1.plot(data['SMA50'], label='SMA50', linestyle='--', color='purple')
ax1.plot(data['SMA200'], label='SMA200 (Trend)', linestyle='--', color='green')
ax1.set_title('AAPL Close Price with SMA20, SMA50 & SMA200')
ax1.legend()
ax1.grid(True)

# Plot 2: Cumulative Returns
ax2.plot(data['Cumulative Market Return'], label='Market Return')
ax2.plot(data['Cumulative Strategy Return'], label='Strategy Return', linestyle='--')
ax2.set_title('Cumulative Returns: Market vs Strategy')
ax2.legend()
ax2.grid(True)

# Final layout
plt.tight_layout()
plt.show()
