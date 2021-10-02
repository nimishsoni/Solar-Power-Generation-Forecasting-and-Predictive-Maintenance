#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go


# ## Data Import and Preprocessing

# In[2]:


data_folder_location = 'C:/Users/E0514808/Documents/Learning/Kaggle Competitions/Solar Power Generation/Data/'

# Import Plant generation data and weather senson data
plant1_generation_data = pd.read_csv(
    data_folder_location + 'Plant_1_Generation_Data.csv', index_col=False)
plant2_generation_data = pd.read_csv(
    data_folder_location + 'Plant_2_Generation_Data.csv', index_col=False)

# Import weather sensor data
plant1_weather_sensor_data = pd.read_csv(
    data_folder_location + 'Plant_1_Weather_Sensor_Data.csv', index_col=False)
plant2_weather_sensor_data = pd.read_csv(
    data_folder_location + 'Plant_2_Weather_Sensor_Data.csv', index_col=False)


# In[3]:


plant1_generation_data['DATE_TIME'] = pd.to_datetime(
    plant1_generation_data['DATE_TIME'], dayfirst=True)
plant1_weather_sensor_data['DATE_TIME'] = pd.to_datetime(
    plant1_weather_sensor_data['DATE_TIME'], dayfirst=True)


# In[4]:


plant1_generation_data


# In[5]:


plant1_inv_list = list(plant1_generation_data['SOURCE_KEY'].unique())
plant1_inv_list


# In[6]:


p1_i1_generation_data = plant1_generation_data[plant1_generation_data['SOURCE_KEY']
                                               == plant1_inv_list[0]]
mask = ((plant1_weather_sensor_data['DATE_TIME'] >= min(p1_i1_generation_data["DATE_TIME"])) & (
    plant1_weather_sensor_data['DATE_TIME'] <= max(p1_i1_generation_data["DATE_TIME"])))
weather_filtered = plant1_weather_sensor_data.loc[mask]


# In[9]:


merged_p1_i1 = p1_i1_generation_data.merge(
    weather_filtered, on="DATE_TIME", how='left')
merged_p1_i1 = merged_p1_i1[['DATE_TIME', 'AC_POWER',
                             'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION']]
merged_p1_i1


# ## Anomaly detection using LSTM Autoencoder

# In[12]:


from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Input, Dropout, Dense, LSTM, TimeDistributed, RepeatVector
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers


# In[10]:


merged_p1_i1_timestamp = merged_p1_i1[["DATE_TIME"]]
merged_p1_i1_features = merged_p1_i1[[
    "AC_POWER", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]]


# In[11]:


train_size = .7
train = merged_p1_i1_features.loc[:merged_p1_i1_features.shape[0]*train_size]
test = merged_p1_i1_features.loc[merged_p1_i1_features.shape[0]*train_size:]


# In[13]:


scaler = MinMaxScaler()
X_train = scaler.fit_transform(train)
X_test = scaler.transform(test)
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")


# In[14]:


def autoencoder_model(X):
    inputs = Input(shape=(X.shape[1], X.shape[2]))
    L1 = LSTM(16, activation='relu', return_sequences=True,
              kernel_regularizer=regularizers.l2(0.00))(inputs)
    L2 = LSTM(4, activation='relu', return_sequences=False)(L1)
    L3 = RepeatVector(X.shape[1])(L2)
    L4 = LSTM(4, activation='relu', return_sequences=True)(L3)
    L5 = LSTM(16, activation='relu', return_sequences=True)(L4)
    output = TimeDistributed(Dense(X.shape[2]))(L5)
    model = Model(inputs=inputs, outputs=output)
    return model


# In[15]:


model = autoencoder_model(X_train)
model.compile(optimizer='adam', loss='mae')
model.summary()


# In[16]:


epochs = 100
batch = 10
history = model.fit(X_train, X_train, epochs=epochs,
                    batch_size=batch, validation_split=.2, verbose=0).history


# In[17]:


fig = go.Figure()

fig.add_trace(go.Scatter(x=[x for x in range(len(history['loss']))], y=history['loss'],
                         mode='lines',
                         name='loss'))

fig.add_trace(go.Scatter(x=[x for x in range(len(history['val_loss']))], y=history['val_loss'],
                         mode='lines',
                         name='validation loss'))

fig.update_layout(title="Autoencoder error loss over epochs",
                  yaxis=dict(title="Loss"),
                  xaxis=dict(title="Epoch"))

fig.show()


# In[18]:


X_pred = model.predict(X_train)
X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[2])
X_pred = scaler.inverse_transform(X_pred)
X_pred = pd.DataFrame(X_pred, columns=train.columns)


# In[19]:


scores = pd.DataFrame()
scores['AC_train'] = train['AC_POWER']
scores["AC_predicted"] = X_pred["AC_POWER"]
scores['loss_mae'] = (scores['AC_train']-scores['AC_predicted']).abs()


# In[20]:


fig = go.Figure(data=[go.Histogram(x=scores['loss_mae'])])
fig.update_layout(title="Error distribution",
                  xaxis=dict(
                      title="Error delta between predicted and real data [AC Power]"),
                  yaxis=dict(title="Data point counts"))
fig.show()


# In[25]:


X_pred = model.predict(X_test)
X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[2])
X_pred = scaler.inverse_transform(X_pred)
X_pred = pd.DataFrame(X_pred, columns=train.columns)
X_pred.index = test.index


# In[26]:


scores = X_pred
scores['datetime'] = merged_p1_i1_timestamp.loc[1893:]
scores['real AC'] = test['AC_POWER']
scores["loss_mae"] = (scores['real AC'] - scores['AC_POWER']).abs()
scores['Threshold'] = 200
scores['Anomaly'] = np.where(scores["loss_mae"] > scores["Threshold"], 1, 0)


# In[27]:


fig = go.Figure()
fig.add_trace(go.Scatter(x=scores['datetime'],
                         y=scores['loss_mae'],
                         name="Loss"))
fig.add_trace(go.Scatter(x=scores['datetime'],
                         y=scores['Threshold'],
                         name="Threshold"))

fig.update_layout(title="Error Timeseries and Threshold",
                  xaxis=dict(title="DateTime"),
                  yaxis=dict(title="Loss"))
fig.show()


# In[28]:


scores['Anomaly'].value_counts()


# In[29]:


anomalies = scores[scores['Anomaly'] == 1][['real AC']]
anomalies = anomalies.rename(columns={'real AC': 'anomalies'})
scores = scores.merge(anomalies, left_index=True, right_index=True, how='left')


# In[30]:


fig = go.Figure()

fig.add_trace(go.Scatter(x=scores["datetime"], y=scores["real AC"],
                         mode='lines',
                         name='AC Power'))

fig.add_trace(go.Scatter(x=scores["datetime"], y=scores["anomalies"],
                         name='Anomaly',
                         mode='markers',
                         marker=dict(color="red",
                                     size=11,
                                     line=dict(color="red",
                                               width=2))))

fig.update_layout(title_text="Anomalies Detected LSTM Autoencoder")

fig.show()


# In[ ]:




