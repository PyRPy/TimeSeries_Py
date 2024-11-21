import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# load the data set CO2
df = sm.datasets.co2.load_pandas().data
df.plot(figsize = (20,10))
# plt.show()
print(df.head())
print(df.dtypes)

# stationary or not
adfuller_result = adfuller(df.dropna())
pvalue = adfuller_result[1]
print(adfuller_result)
if pvalue < 0.05:
    print("stationary")
else:
    print("non-stationary")

# after one order of differencing
adfuller_result = adfuller(df.dropna().diff().dropna())
pvalue = adfuller_result[1]
print(adfuller_result)
if pvalue < 0.05:
    print("stationary")
else:
    print("non-stationary")

# ACF/PACF lag = 600
fig, (ax1,ax2) = plt.subplots(figsize = (20,10), ncols = 2)
plot_acf(df.diff().dropna(), ax = ax1, lags = 600)
plot_pacf(df.diff().dropna(), ax = ax2, lags = 600)
# plt.show()

# ARIMA
train, test = train_test_split(df, test_size = 0.2,random_state = 13, shuffle = False)
p, d, q = 1,1,1
fcst = []
for step in range(test.shape[0]):
    try:
        arima = ARIMA(train, order = (p,d,q))
        arima_final = arima.fit()
        prediction = arima_final.forecast(steps = 1)
        fcst.append(prediction[0])
        train = train._append(pd.Series(test.iloc[step]))
    except:
        error = -99999
        print("error")
        fcst.append(error)
        tmp = test.iloc[step]
        tmp[0] = error
        train = train._append(pd.Series(tmp))

# plot results for comparison
test["fcst"] = fcst
test.plot(figsize = (20,10))
plt.show()
