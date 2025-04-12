import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Download 6+ years of data to calculate 5-year MA
data = yf.download('AAPL', start='2019-01-01', end='2025-04-11')
data = data[['Close']].copy()

# Resample to weekly (Friday close)
weekly = data.resample('W-FRI').last()

# 1-Year MA = 52-week SMA (weekly data)
weekly['MA_1Y'] = weekly['Close'].rolling(window=52).mean()

# 5-Year MA = 260-week SMA
weekly['MA_5Y'] = weekly['Close'].rolling(window=260).mean()

# Plot
plt.figure(figsize=(14, 6))
plt.plot(weekly['Close'], label='Weekly Close', alpha=0.6)
plt.plot(weekly['MA_1Y'], label='1-Year MA (52W)', linestyle='--', color='orange')
plt.plot(weekly['MA_5Y'], label='5-Year MA (260W)', linestyle='--', color='green')
plt.title('AAPL Weekly Close with 1-Year and 5-Year Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
