# Solar-Power-Generation-Analysis-and-Predictive-Maintenance
This project covers analysis for solar power deneration data, prediction and predictive Maintenance using Kaggle Dataset provided here: https://www.kaggle.com/anikannal/solar-power-generation-data. The power generation datasets are gathered at the inverter level - each inverter has multiple lines of solar panels attached to it. The sensor data is gathered at a plant level - single array of sensors optimally placed at the plant.

## Motivation
Through this project we are trying to answer the following:
1. Can we predict the power generation for next couple of days? - this allows for better grid management
2. Can we identify the need for panel cleaning/maintenance?
3. Can we identify faulty or suboptimally performing equipment?

## Summary of the Project
Following are the main components of the project

## File Descriptions
Solar Descriptive Analytics.ipynb: Python notebook for analyzing historical data for plant 1 and 2 and compare power generation from 22 inverters
Solar Power Prediction.ipynb: Python notebook for training and evaluating performance of linear regression and XG Boost model for predicting power generation. The dataset is divided in to 70% training and 30% test data set for this. The metrics used for comparing and evaluating the performance of models are: R2 score, Mean square error and mean absolute scaled error. 

## Pre-requisites and Dependencies
Python
VsCode
Jupyter Notebook
Pandas
Numpy
Matplotlib
Seaborn
Sklearn model selections
Sklearn.metrics
Sklearn
