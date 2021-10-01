#!/usr/bin/env python
# coding: utf-8

# # Prediction of Solar Power generation using Linear Regression and XG Boost

# In[1]:


from sklearn.preprocessing import MinMaxScaler
from keras.layers import LSTM
from keras.layers import Dense
from keras.models import Sequential
import keras
from xgboost import plot_importance
import xgboost as xgb
from sklearn.utils import check_array
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import datasets, linear_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data_folder_location = 'C:/Users/E0514808/Documents/Learning/Kaggle Competitions/Solar Power Generation/Data/'

# Import Plant generation data and weather senson data
plant1_generation_data = pd.read_csv(
    data_folder_location +
    'Plant_1_Generation_Data.csv',
    index_col=False)
plant2_generation_data = pd.read_csv(
    data_folder_location +
    'Plant_2_Generation_Data.csv',
    index_col=False)

# Import weather sensor data
plant1_weather_sensor_data = pd.read_csv(
    data_folder_location +
    'Plant_1_Weather_Sensor_Data.csv',
    index_col=False)
plant2_weather_sensor_data = pd.read_csv(
    data_folder_location +
    'Plant_2_Weather_Sensor_Data.csv',
    index_col=False)


# In[3]:


# Retaining relevant data
plant2_generation_Time = plant2_generation_data.groupby(
    ['DATE_TIME'], as_index=False).sum()
plant2_generation_Time = plant2_generation_Time[[
    'DATE_TIME', 'DC_POWER', 'AC_POWER', 'DAILY_YIELD']]


# In[4]:


# Retaining relevant data
plant2_weather_sensor_data1 = plant2_weather_sensor_data.drop(
    ['PLANT_ID', 'SOURCE_KEY'], axis=1)

# Merge plant 1 solar generation data and weather data
merged_data_plant2 = pd.merge(
    plant2_generation_Time,
    plant2_weather_sensor_data1,
    how='inner',
    on='DATE_TIME')


# ## AC Power Output prediction of Plant

# In[23]:


plt.style.use('fivethirtyeight')


# In[6]:


target = merged_data_plant2['AC_POWER']
features = merged_data_plant2[['IRRADIATION', 'AMBIENT_TEMPERATURE']]


# In[9]:


# Split the data in to train and test sets
#scaler = MinMaxScaler(feature_range=(0, 1))
#features_norm = scaler.fit_transform(features)
#target_norm = scaler.fit_transform(target)
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.3, random_state=5)


# ## Prediction using Linear Regression

# In[11]:


lm = linear_model.LinearRegression()
model_lm = lm.fit(X_train, y_train)
pred_y_test_lm = lm.predict(X_test)


# In[42]:


# Calculate MSE, MAE and MAPE for Predicted output to quantify model error
R2_lm = r2_score(y_test, pred_y_test_lm)
mse_lm = mean_squared_error(y_test, pred_y_test_lm, squared=False)
mae_lm = mean_absolute_error(y_test, pred_y_test_lm)


def mean_absolute_scaled_error(y_true, y_pred, y_train):
    e_t = y_true - y_pred
    scale = mean_absolute_error(y_train[1:], y_train[:-1])
    return np.mean(np.abs(e_t / scale))


mase_lm = mean_absolute_scaled_error(y_test, pred_y_test_lm, y_train)

print('R2 using Linear Regression:', R2_lm, '  '
      'RMSE using Linear Regression:', mse_lm, '\n '
      'MAE using Linear Regression:', mae_lm, '  '
      'MASE using Linear Regression:', mase_lm)


# ## Prediction using XG Boost Model

# In[13]:


config = xgb.get_config()
config


# In[14]:


# Fit XGB model on the Solar Generation Dataset
model_xgb = xgb.XGBRegressor(n_estimators=50)
model_xgb.fit(X_train, y_train,
              eval_set=[(X_train, y_train),
                        (X_test, y_test)],
              early_stopping_rounds=50,
              verbose=False)


# In[15]:


# Plot features by Significance for forecasting
_ = plot_importance(model_xgb, height=0.5)


# In[16]:


pred_y_test_xgb = model_xgb.predict(X_test)


# In[43]:


# Calculate MSE, MAE and MAPE for Predicted output to quantify model error
R2_xgb = r2_score(y_test, pred_y_test_xgb)
mse_xgb = mean_squared_error(y_test, pred_y_test_xgb, squared=False)
mae_xgb = mean_absolute_error(y_test, pred_y_test_xgb)
mase_xgb = mean_absolute_scaled_error(y_test, pred_y_test_xgb, y_train)

print(
    'R2 using XGB:',
    R2_xgb,
    '  '
    'RMSE using XGB:',
    mse_xgb,
    '\n '
    'MAE using XGB:',
    mae_xgb,
    '  '
    'MASE using XGB:',
    mase_xgb)


# ## XGB with Gridsearch Parameter Selection

# In[36]:


# Define Pipeline and Parammeter grid
pipeline = Pipeline([
    ('model', model_xgb)
])

param_grid = {
    'model__max_depth': [2, 3, 5, 7],
    'model__n_estimators': [10, 50, 100],
    'model__learning_rate': [0.02, 0.05, 0.1, 0.3],
    'model__min_child_weight': [0.5, 1, 2]
}

grid = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)


# In[37]:


# Fit the model
grid.fit(X_train, y_train)


# In[38]:


# Print the Best parameters for the model identified using Gridsearch
print(f"Best parameters: {grid.best_params_}")


# In[39]:


# Predict using Gridsearch
pred_y_test_xgb_grid = grid.predict(X_test)


# In[41]:


# Calculate MSE, MAE and MAPE for Predicted output to quantify model error
R2_xgb_grid = r2_score(y_test, pred_y_test_xgb_grid)
mse_xgb_grid = mean_squared_error(y_test, pred_y_test_xgb_grid, squared=False)
mae_xgb_grid = mean_absolute_error(y_test, pred_y_test_xgb_grid)
mase_xgb_grid = mean_absolute_scaled_error(
    y_test, pred_y_test_xgb_grid, y_train)

print(
    'R2 using XGB_grid:',
    R2_xgb_grid,
    '  '
    'RMSE using XGB_grid:',
    mse_xgb_grid,
    '\n '
    'MAE using XGB_grid:',
    mae_xgb_grid,
    '  '
    'MASE using XGB_grid:',
    mase_xgb_grid)


# ## Prediction using LSTM

# In[18]:


# In[20]:


# Split the normalized data in to train and test sets
scaler = MinMaxScaler(feature_range=(0, 1))
features_norm = scaler.fit_transform(features)
#target_norm = scaler.fit_transform(target)
X_train1, X_test1, y_train1, y_test1 = train_test_split(
    features_norm, target, test_size=0.3, random_state=5)


# In[ ]:
