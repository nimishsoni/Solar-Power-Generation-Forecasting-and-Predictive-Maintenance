#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Data Import and Exploration

# In[3]:


data_folder_location = 'C:/Users/E0514808/Documents/Learning/Kaggle Competitions/Solar Power Generation/Data/'


# In[4]:


# Import Plant generation data and weather senson data
plant1_generation_data = pd.read_csv(
    data_folder_location +
    'Plant_1_Generation_Data.csv',
    index_col=False)
plant2_generation_data = pd.read_csv(
    data_folder_location +
    'Plant_2_Generation_Data.csv',
    index_col=False)


# In[5]:


# Print plant generation data table
print(
    "Plant1 Generation Data Table --- -------",
    "\n Table Shape",
    plant1_generation_data.shape,
    "\n Table\n",
    plant1_generation_data.head(10))
#print("\n Plant2 Generation Data Table --- -------","\n Table Shape", plant2_generation_data.shape,"\n Table\n", plant2_generation_data.head(10))


# In[6]:


# Import weather sensor data
plant1_weather_sensor_data = pd.read_csv(
    data_folder_location +
    'Plant_1_Weather_Sensor_Data.csv',
    index_col=False)
plant2_weather_sensor_data = pd.read_csv(
    data_folder_location +
    'Plant_2_Weather_Sensor_Data.csv',
    index_col=False)


# In[7]:


# Print plant weather sensor data table
print(
    "Plant1 Weather Sensor Data Table --- -------",
    "\n Table Shape",
    plant1_weather_sensor_data.shape,
    "\n Table\n",
    plant1_weather_sensor_data.head(10))
#print("\n Plant2 Weather Sensor Data Table --- -------","\n Table Shape", plant2_weather_sensor_data.shape,"\n Table\n", plant2_weather_sensor_data.head(10))


# In[8]:


# Statistics of Plant 1 Generation Data
plant1_generation_data.describe()


# In[9]:


# Statistics of Plant 2 Generation Data
plant1_weather_sensor_data.describe()


# In[10]:


# Null values in all the dataset
print("Total null values in plant 1 generation data is {}".format(
    plant1_generation_data.isnull().sum().sum()))
print("Total null values in plant 2 generation data is {}".format(
    plant2_generation_data.isnull().sum().sum()))
print("Total null values in plant 1 weather data is {}".format(
    plant1_weather_sensor_data.isnull().sum().sum()))
print("Total null values in plant 2 weather data is {}".format(
    plant1_weather_sensor_data.isnull().sum().sum()))


# In[11]:


# Print Number of Inverters in each Plant
print("Total inverters in plant 1 generation data are {}".format(
    plant1_generation_data['SOURCE_KEY'].nunique()))
print("Total inverters in plant 2 generation data are {}".format(
    plant2_generation_data['SOURCE_KEY'].nunique()))


# In[12]:


# Average DC and AC Power from Plant 1 Inverters in Descending order
plant1_generation_data.groupby(['SOURCE_KEY'])['DC_POWER', 'AC_POWER'].mean(
).sort_values(by=['DC_POWER', 'AC_POWER'], ascending=False)


# In[13]:


# Average DC and AC Power from Plant 2 Inverters in Descending order
plant2_generation_data.groupby(['SOURCE_KEY'])['DC_POWER', 'AC_POWER'].mean(
).sort_values(by=['DC_POWER', 'AC_POWER'], ascending=False)


# In[14]:


# Group plant 1 DC Power Generation by hour and minutes
times = pd.DatetimeIndex(plant1_generation_data.DATE_TIME)
plant1_group_time = plant1_generation_data.groupby(
    [times.hour, times.minute]).DC_POWER.mean()


# In[15]:


# Average DC Output power for plant 1 through the day
plant1_group_time.plot(figsize=(20, 5))
plt.title('Total 34 day Average DC Power Output for Plant 1 with time of the day')
plt.ylabel('DC Power (kW)')


# In[16]:


# Comupte Avergade DC Power output for each inverter during time (Hour and
# minute) of day
plant1_group_inv_time = plant1_generation_data.groupby(
    [times.hour, times.minute, 'SOURCE_KEY']).DC_POWER.mean().unstack()
plant1_group_inv_time


# In[17]:


# Plot and compare DC Power Supplier to Plant 1 inverters
fig, ax = plt.subplots(ncols=3, nrows=1, dpi=200, figsize=(20, 5))
ax[0].set_title('DC Power Supplied 1st 7 Inverters')
ax[1].set_title('DC Power Supplied to next 8 Inverters')
ax[2].set_title('DC Power Supplied to last 7 Inverters')
ax[0].set_ylabel('DC POWER (kW)')


plant1_group_inv_time.iloc[:, 0:7].plot(ax=ax[0], linewidth=5)
plant1_group_inv_time.iloc[:, 7:15].plot(ax=ax[1], linewidth=5)
plant1_group_inv_time.iloc[:, 15:22].plot(ax=ax[2], linewidth=5)


# In[18]:


# Compute and plot inverter efficiency for plant 1
plant1_group_inv = plant1_generation_data.groupby(['SOURCE_KEY']).mean()
plant1_group_inv['Inv_Efficiency'] = plant1_group_inv['AC_POWER'] * \
    100 / plant1_group_inv['DC_POWER']

plant1_group_inv['Inv_Efficiency'].plot(figsize=(15, 5), style='o--')
plt.axhline(
    plant1_group_inv['Inv_Efficiency'].mean(),
    linestyle='--',
    color='green')
plt.title('Plant 1 Inverter Efficiency Plot', size=20)
plt.ylabel('% Efficiency')


# In[19]:


# Compute and plot inverter efficiency for plant 2
plant2_group_inv = plant2_generation_data.groupby(['SOURCE_KEY']).mean()
plant2_group_inv['Inv_Efficiency'] = plant2_group_inv['AC_POWER'] * \
    100 / plant2_group_inv['DC_POWER']

plant2_group_inv['Inv_Efficiency'].plot(figsize=(15, 5), style='o--')
plt.axhline(
    plant2_group_inv['Inv_Efficiency'].mean(),
    linestyle='--',
    color='green')
plt.title('Plant 2 Inverter Efficiency Plot', size=20)
plt.ylabel('% Efficiency')


# In[20]:


plant2_generation_Time = plant2_generation_data.groupby(
    ['DATE_TIME'], as_index=False).sum()
plant2_generation_Time


# In[21]:


# Retaining relevant data
plant2_generation_Time = plant2_generation_Time[[
    'DATE_TIME', 'DC_POWER', 'AC_POWER', 'DAILY_YIELD']]
plant2_generation_Time


# In[24]:


# Retaining relevant data
plant2_weather_sensor_data1 = plant2_weather_sensor_data.drop(
    ['PLANT_ID', 'SOURCE_KEY'], axis=1)


# In[25]:


# Merge plant 1 solar generation data and weather data
merged_data = pd.merge(
    plant2_generation_Time,
    plant2_weather_sensor_data1,
    how='inner',
    on='DATE_TIME')


# In[26]:


sns.pairplot(merged_data[['DC_POWER',
                          'AC_POWER',
                          'DAILY_YIELD',
                          'AMBIENT_TEMPERATURE',
                          'MODULE_TEMPERATURE',
                          'IRRADIATION']])


# In[27]:


merged_data_num = merged_data[['DC_POWER',
                               'AC_POWER',
                               'DAILY_YIELD',
                               'AMBIENT_TEMPERATURE',
                               'MODULE_TEMPERATURE',
                               'IRRADIATION']]
corr = merged_data_num.corr()

fig_dims = (2, 2)
sns.heatmap(round(corr, 2), annot=True, mask=(np.triu(corr, +1)))


# As expected DC Power generated from Solar panel has a very high positive
# sorrelation with Irradiation, Module and Ambient temperature.

# In[ ]:
