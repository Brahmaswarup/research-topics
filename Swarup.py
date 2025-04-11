#!/usr/bin/env python
# coding: utf-8

# # ARIMA -  Johnson & Johnson Sales Prediction

# In[16]:


from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools
from itertools import product
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = [10, 7.5]
from scipy.stats import boxcox
import statsmodels.api as sm
import pmdarima as pm


# In[26]:


df = pd.read_csv('jj.csv')


# In[27]:


df.head(10)


# In[28]:


df['date'] = pd.to_datetime(df['date'])


# In[29]:


plt.plot(df['date'], df['data'])
plt.title('Johnson & Johnson Sales over Time')
plt.xlabel('Timeline')
plt.ylabel('Sales')
plt.show()


# In[7]:


df.describe()


# In[8]:


fig, ax = plt.subplots(1, 2, figsize=(15, 5))
plot_acf(df['data'], ax=ax[0], lags=40)
ax[0].set_title('Autocorrelation Function')
plot_pacf(df['data'], ax=ax[1], lags=40, method='ywm')
ax[1].set_title('Partial Autocorrelation Function')
plt.show()


# In[9]:


adf_result = adfuller(df['data'])
print(f"ADF Statistic: {adf_result[0]}")
print(f"p-value: {adf_result[1]}")


# A p-value greater than 0.05 suggests non-stationarity

# In[11]:


data = df['data'].values
log_data = np.log(data)
stationary_data = np.diff(log_data)
stationary_data_positive = stationary_data + 1
transformed_data, lambda_value = boxcox(stationary_data_positive)
print("Lambda value from Box-Cox transformation:", lambda_value)


# In[13]:


model = pm.auto_arima(transformed_data, start_p=0, d=1, start_q=0, max_p=8, max_d=1, max_q=8, seasonal=False,
                      trace=False, error_action='ignore', suppress_warnings=True)


# In[14]:


print("Best ARIMA model order:", model.order)
print("AIC value for the best model:", model.aic())


# In[17]:


model = ARIMA(data, order=(4, 1, 1))
model_fit = model.fit()
print(model_fit.summary())
model_fit.plot_diagnostics()


# In[22]:


predictions_int = model_fit.get_forecast(steps=24)
forecasted_values = predictions_int.predicted_mean
conf_int = predictions_int.conf_int()

idx = np.arange(len(data), len(data) + 24)

plt.plot(data, color='blue', label='Original Data')
plt.plot(idx, forecasted_values, color='orange', label='Forecast')
plt.fill_between(idx, conf_int[:, 0], conf_int[:, 1], color='yellow', alpha=0.3, label='Confidence Intervals')
plt.title('Forecast of Johnson & Johnson Sales')
plt.xlabel('Time')
plt.ylabel('Sales')
plt.legend()
plt.show()


# # ARIMA -  Amazon Sales Prediction

# In[99]:


df = pd.read_csv('AMZN.csv')


# In[100]:


df.head()


# In[101]:


df['Date'] = pd.to_datetime(df['Date'])


# In[102]:


plt.plot(df['Date'], df['Close'])
plt.title('Amazon stock price over Time')
plt.xlabel('Timeline')
plt.ylabel('Closing Price')
plt.show()


# In[103]:


df.describe()


# In[104]:


fig, ax = plt.subplots(1, 2, figsize=(15, 5))
plot_acf(df['Close'], ax=ax[0], lags=40)
ax[0].set_title('Autocorrelation Function')
plot_pacf(df['Close'], ax=ax[1], lags=40, method='ywm')
ax[1].set_title('Partial Autocorrelation Function')
plt.show()


# In[105]:


adf_result = adfuller(df['Close'])
print(f"ADF Statistic: {adf_result[0]}")
print(f"p-value: {adf_result[1]}")


# A p-value greater than 0.05 suggests non-stationarity

# In[106]:


data = df['Close'].values
log_data = np.log(data)
stationary_data = np.diff(log_data)
stationary_data_positive = stationary_data + 1
transformed_data, lambda_value = boxcox(stationary_data_positive)
print("Lambda value from Box-Cox transformation:", lambda_value)


# In[107]:


model = pm.auto_arima(transformed_data, start_p=0, d=1, start_q=0, max_p=8, max_d=1, max_q=8, seasonal=False,
                      trace=False, error_action='ignore', suppress_warnings=True)


# In[108]:


print("Best ARIMA model order:", model.order)
print("AIC value for the best model:", model.aic())


# In[109]:


model = ARIMA(data, order=(8, 1, 2))
model_fit = model.fit()
print(model_fit.summary())
model_fit.plot_diagnostics()


# In[112]:


forecast_result = model_fit.get_forecast(steps=504)
forecast_mean = forecast_result.predicted_mean
forecast_ci = forecast_result.conf_int()
last_date = df['Date'].iloc[-1]
forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=504, freq='B')


# In[116]:


plt.plot(df['Date'], df['Close'], label='Historical')
plt.plot(forecast_dates, forecast_mean, color='orange', label='Forecast (Next 24 Months)')
plt.fill_between(forecast_dates, forecast_ci[:, 0], forecast_ci[:, 1], color='orange', alpha=0.3)
plt.title('Forecast of Amazon Sales')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()


# # RNN - GRU

# In[71]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, BatchNormalization


# In[72]:


df = pd.read_csv('AMZN.csv')


# In[73]:


df.head()


# In[74]:


x = df[['Open', 'High', 'Low', 'Adj Close', 'Volume']]; y = df['Close']


# In[75]:


from sklearn.preprocessing import StandardScaler


# In[76]:


scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)


# In[77]:


from sklearn.model_selection import train_test_split


# In[82]:


x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.3)


# In[83]:


print("Training data shape:", x_train.shape)
print("Testing data shape:", x_test.shape)


# In[84]:


x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


# In[85]:


print("Training data shape:", x_train.shape)
print("Testing data shape:", x_test.shape)


# In[89]:


model = Sequential()

# Layer 1: GRU + BatchNorm
model.add(GRU(units=64, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(BatchNormalization())

# Layer 2: GRU
model.add(GRU(units=64, return_sequences=False))

model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=16, activation='relu'))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()


# In[90]:


history = model.fit(x_train, y_train, epochs=50)


# In[92]:


loss = history.history['loss']
plt.plot(loss, label='Training Loss')
plt.title('Training Loss Plot')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.show()


# In[96]:


df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

last_known_inputs = x_scaled[-1].reshape(1, x.shape[1], 1)

future_preds = []

current_input = last_known_inputs.copy()

for _ in range(504):
    next_pred = model.predict(current_input, verbose=0)[0][0]
    future_preds.append(next_pred)
    new_input_flat = current_input.reshape(-1)
    new_input_flat[3] = next_pred
    new_input_scaled = new_input_flat.reshape(1, -1)
    current_input = new_input_scaled.reshape(1, x.shape[1], 1)


# In[97]:


last_date = df['Date'].iloc[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=504, freq='B')


# In[98]:


plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Close'], label='Historical')
plt.plot(future_dates, future_preds, label='Forecast (24 months)', color='orange')
plt.title('Forecast of Amazon Sales')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()


# In[ ]:




