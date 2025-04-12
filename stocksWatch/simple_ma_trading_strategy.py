import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Download data
data = yf.download('AAPL', start='2023-01-01', end='2024-12-31')
data = data[['Close']].copy()

# Calculate SMAs
data['SMA20'] = data['Close'].rolling(20).mean()
data['SMA50'] = data['Close'].rolling(50).mean()

# Generate Buy/Sell signals
data['Signal'] = 0
data.loc[data['SMA20'] > data['SMA50'], 'Signal'] = 1  # Buy/Hold
data.loc[data['SMA20'] < data['SMA50'], 'Signal'] = 0  # Sell/Exit

# Lag the signal to simulate taking the action the next day
data['Position'] = data['Signal'].shift(1)

# Calculate returns
data['Daily Return'] = data['Close'].pct_change()
data['Strategy Return'] = data['Position'] * data['Daily Return']

# Cumulative returns
data['Cumulative Market Return'] = (1 + data['Daily Return']).cumprod()
data['Cumulative Strategy Return'] = (1 + data['Strategy Return']).cumprod()

# Plot performance
plt.figure(figsize=(12,6))
plt.plot(data['Cumulative Market Return'], label='Buy & Hold Market')
plt.plot(data['Cumulative Strategy Return'], label='SMA Strategy')
plt.title("SMA Crossover Backtest")
plt.legend()
plt.grid(True)
plt.show()

# Print summary
print("Final Strategy Return:", round(data['Cumulative Strategy Return'].iloc[-1], 2))
print("Final Market Return:", round(data['Cumulative Market Return'].iloc[-1], 2))
