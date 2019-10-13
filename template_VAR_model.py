
# coding: utf-8

# In[ ]:


# A Multivariate Time Series Guide to Forecasting and Modeling 


# In[68]:


#import required packages
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from math import sqrt
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings("ignore")


# In[58]:


#read the data
df = pd.read_csv("AirQualityUCI2.csv")

#check the dtypes
df.dtypes


# In[59]:


df.head()


# In[60]:


# convert index to have datetime
df['Date_Time'] = df.Date + ' ' + df.Time
df.head()


# In[61]:


df['Date_Time'] = pd.to_datetime(df.Date_Time , format = '%m/%d/%Y %H:%M:%S')


# In[62]:


data = df.drop(['Date', 'Time', 'Date_Time'], axis=1)

data.index = df.Date_Time


# In[63]:


data.head()


# In[64]:


data.columns


# In[65]:



data = data.drop(['Unnamed: 15', 'Unnamed: 16'], axis = 1)


# In[66]:


data = data.dropna()


# In[80]:


# plot the data
data.plot(subplots = True, figsize = (12, 16))


# In[67]:


#checking stationarity
from statsmodels.tsa.vector_ar.vecm import coint_johansen
#since the test works for only 12 variables, I have randomly dropped
#in the next iteration, I would drop another and check the eigenvalues
johan_test_temp = data.drop([ 'CO(GT)'], axis=1)
coint_johansen(johan_test_temp,-1,1).eig


# In[69]:


#creating the train and validation set
train = data[:int(0.8*(len(data)))]
valid = data[int(0.8*(len(data))):]

#fit the model
from statsmodels.tsa.vector_ar.var_model import VAR

model = VAR(endog=train)
model_fit = model.fit()


# In[70]:


train.head()


# In[71]:


# make prediction on validation
prediction = model_fit.forecast(model_fit.y, steps=len(valid))


# In[72]:


#converting predictions to dataframe
pred = pd.DataFrame(index=range(0,len(prediction)),columns=[cols])
for j in range(0,13):
    for i in range(0, len(prediction)):
       pred.iloc[i][j] = prediction[i][j]

#check rmse
for i in cols:
    print('rmse value for', i, 'is : ', sqrt(mean_squared_error(pred[i], valid[i])))


# In[73]:


#make final predictions
model = VAR(endog=data)
model_fit = model.fit()
yhat = model_fit.forecast(model_fit.y, steps=1)
print(yhat)


# In[45]:


# https://www.analyticsvidhya.com/blog/2018/09/multivariate-time-series-guide-forecasting-modeling-python-codes/?utm_source=DataCamp.com&utm_medium=Community&utm_campaign=News

