
# coding: utf-8

# In[ ]:


# Autoregressive Moving Average (ARMA): Sunspots data


# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm

from statsmodels.graphics.api import qqplot


# In[8]:


import warnings
warnings.filterwarnings("ignore")


# In[2]:


# Sunspots Data
print(sm.datasets.sunspots.NOTE)


# In[3]:


dta = sm.datasets.sunspots.load_pandas().data
dta.head()


# In[4]:


dta.index = pd.Index(sm.tsa.datetools.dates_from_range('1700', '2008'))
del dta["YEAR"]
dta.head()


# In[5]:


dta.plot(figsize = (12, 4))


# In[ ]:


# ACF and PACF plots


# In[6]:


fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(dta.values.squeeze(), lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(dta, lags=40, ax=ax2)


# In[9]:


# fit a AR(2) model
arma_mod20 = sm.tsa.statespace.SARIMAX(dta, order=(2,0,0), trend='c').fit(disp=False)
print(arma_mod20.params)


# In[10]:


arma_mod30 = sm.tsa.statespace.SARIMAX(dta, order=(3,0,0), trend='c').fit(disp=False)


# In[11]:


print(arma_mod20.aic, arma_mod20.bic, arma_mod20.hqic)


# In[12]:


print(arma_mod30.params)


# In[ ]:


# model comparison and selection


# In[13]:


print(arma_mod30.aic, arma_mod30.bic, arma_mod30.hqic)


# In[ ]:


# residuals


# In[14]:


sm.stats.durbin_watson(arma_mod30.resid)


# In[15]:


fig = plt.figure(figsize=(12,4))
ax = fig.add_subplot(111)
ax = plt.plot(arma_mod30.resid)


# In[17]:


resid = arma_mod30.resid
stats.normaltest(resid)


# In[18]:


fig = plt.figure(figsize=(12,4))
ax = fig.add_subplot(111)
fig = qqplot(resid, line='q', ax=ax, fit=True)


# In[19]:


fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(resid, lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(resid, lags=40, ax=ax2)


# In[20]:


r,q,p = sm.tsa.acf(resid, fft=True, qstat=True)
data = np.c_[range(1,41), r[1:], q, p]
table = pd.DataFrame(data, columns=['lag', "AC", "Q", "Prob(>Q)"])
print(table.set_index('lag'))
# This indicates a lack of fit.


# In[ ]:


# predictions


# In[21]:


predict_sunspots = arma_mod30.predict(start='1990', end='2012', dynamic=True)


# In[22]:


fig, ax = plt.subplots(figsize=(12, 8))
dta.loc['1950':].plot(ax=ax)
predict_sunspots.plot(ax=ax, style='r')


# In[23]:


def mean_forecast_err(y, yhat):
    return y.sub(yhat).mean()


# In[24]:


mean_forecast_err(dta.SUNACTIVITY, predict_sunspots)


# In[ ]:


# https://www.statsmodels.org/dev/examples/notebooks/generated/statespace_arma_0.html

