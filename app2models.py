#!/usr/bin/env python
# coding: utf-8

# In[3]:
# I still want to put what each models are based on whether bagging or hyperparametric or base or activation function

#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)
from sklearn.svm import SVR
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


# --------------------------------------- FUNCTION DEFINITIONS --------------------------------------- #

def load_data(filepath):
    index_names = ['unit_number', 'time_cycles']
    setting_names = ['setting_1', 'setting_2', 'setting_3']
    sensor_names = ['s_{}'.format(i+1) for i in range(0,21)]
    col_names = index_names + setting_names + sensor_names
    df = pd.read_csv(filepath, sep='\s+', header=None, index_col=False, names=col_names)
    return df

def process_data_for_regression(df):
    df.sort_values(['unit_number', 'time_cycles'], inplace=True)
    max_cycle_df = df.groupby('unit_number')['time_cycles'].max().reset_index()
    max_cycle_df.columns = ['unit_number', 'max_cycle']
    df = df.merge(max_cycle_df, on='unit_number', how='left')
    df['RUL'] = df['max_cycle'] - df['time_cycles']
    df.drop('max_cycle', axis=1, inplace=True)
    return df

# --------------------------------------- STREAMLIT APP START --------------------------------------- #

# Main title
st.title("Predictive Maintenance Dataset Explorer and Predictor")

# Subtitle and details
st.write("### NASA Turbofan Jet Engine Data Set")
st.write("Run to Failure Degradation Simulation")

st.write("### Prediction Goal")
st.write("The goal is to predict the remaining useful life (RUL) of each engine in the test dataset. RUL is the equivalent of the number of flights remaining for the engine after the last datapoint in the test dataset.")
uploaded_file = st.sidebar.file_uploader("Upload your training dataset (CSV file)", type="csv")

if uploaded_file:
    df_train = pd.read_csv(uploaded_file)
    st.sidebar.success("Training dataset uploaded successfully!")
else:
    df_train = load_data('train_FD001.txt')

uploaded_test_file = st.sidebar.file_uploader("Upload your test dataset (CSV file)", type="csv")
if uploaded_test_file:
    df_test = pd.read_csv(uploaded_test_file)
else:
    df_test = load_data('test_FD001.txt')


# Create df_test_last_cycle for both Linear Regression and Random Forest:
df_test_last_cycle = df_test.groupby('unit_number').last().reset_index()
df_rul = pd.read_csv('RUL_FD001.txt', names=['RUL'])  # assuming you have this file, adjust path if needed
df_test_last_cycle['RUL'] = df_rul.values

# --------------------------------------- EXPLORATORY DATA ANALYSIS --------------------------------------- #

page_selection = st.sidebar.radio("Select a Page", ["Exploratory Data Analysis", "Linear Regression", "Random Forest", "Support Vector Regression", "XGBoost Regression", "LSTM Regression", "Dense Neural Network"])


if page_selection == "Exploratory Data Analysis":
    st.title("Exploratory Data Analysis (EDA) of the Dataset")
    
    st.write("### General Overview")
    st.write(df_train.head())
    st.write(f"Number of Rows in Training Data: {df_train.shape[0]}")
    st.write(f"Number of Columns in Training Data: {df_train.shape[1]}")
    
    # Display basic statistics about the dataset
    st.write("### Basic Statistics")
    st.write(df_train.describe())
    
    df_test = load_data('test_FD001.txt')
    df_rul = pd.read_csv('RUL_FD001.txt', names=['RUL'])

    # Process the training data
    df_train = process_data_for_regression(df_train)

    # Get the last time step for each engine from test dataset
    df_test_last_cycle = df_test.groupby('unit_number').last().reset_index()
    df_test_last_cycle['RUL'] = df_rul.values

    
    # Display histogram of Remaining Useful Life (RUL)
    st.write("### Distribution of Remaining Useful Life (RUL)")
    fig, ax = plt.subplots()
    df_train['RUL'].hist(ax=ax, bins=100)
    ax.set_xlabel("RUL")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)
    
    # Display correlation heatmap
    st.write("### Correlation Heatmap")
    corr = df_train.corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    cax = ax.matshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(cax)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    st.pyplot(fig)
    
    # Display box plot for sensor readings
    st.write("### Box Plot for Sensor Readings")
    sensor_cols = [f's_{i}' for i in range(1, 22)]
    fig, ax = plt.subplots(figsize=(12, 6))
    df_train[sensor_cols].boxplot(ax=ax)
    ax.set_title("Box plot of Sensor Readings")
    ax.set_xlabel("Sensor")
    ax.set_ylabel("Value")
    ax.set_xticklabels(sensor_cols, rotation=90)
    st.pyplot(fig)
    
# --------------------------------------- LINEAR REGRESSION PAGE --------------------------------------- #

if page_selection == "Linear Regression":

    st.write("### Training Data (Linear Regression)")
    st.write(df_train.head())
    st.write(f"Number of Rows in Training Data: {df_train.shape[0]}")
    st.write(f"Number of Columns in Training Data: {df_train.shape[1]}")

    # Slider Inputs
    st.sidebar.header("Custom Prediction Inputs (LR)")
    st.sidebar.write("Adjust the sliders based on input data:")

    setting_min = float(df_train['setting_1'].min())
    setting_max = float(df_train['setting_1'].max()) + 0.01 if setting_min == float(df_train['setting_1'].max()) else float(df_train['setting_1'].max())
    setting_1 = st.sidebar.slider('Setting 1', setting_min, setting_max)
    setting_2 = st.sidebar.slider('Setting 2', setting_min, setting_max)
    setting_3 = st.sidebar.slider('Setting 3', setting_min, setting_max)

    sensors = {}
    for i in range(1, 22):
        sensor_min = float(df_train[f's_{i}'].min())
        sensor_max = sensor_min + 0.01 if sensor_min == float(df_train[f's_{i}'].max()) else float(df_train[f's_{i}'].max())
        sensors[f's_{i}'] = st.sidebar.slider(f'Sensor {i}', sensor_min, sensor_max)

    if st.sidebar.button('Predict RUL using Custom Input (LR)'):
        if 'lr_model' in globals():
            custom_input_data = [[setting_1, setting_2, setting_3] + list(sensors.values())]
            custom_scaled_input = scaler.transform(custom_input_data)
            custom_prediction = lr_model.predict(custom_scaled_input)
            st.sidebar.write(f"Predicted RUL (LR): {custom_prediction[0]:.2f}")
        
st.write("### Linear Regression Model Training")

if st.button('Train and Predict using Linear Regression Model'):
    # Loading test dataset and RUL dataset
    df_test = load_data('test_FD001.txt')
    df_rul = pd.read_csv('RUL_FD001.txt', names=['RUL'])

    # Process the training data
    df_train = process_data_for_regression(df_train)

    # Get the last time step for each engine from test dataset
    df_test_last_cycle = df_test.groupby('unit_number').last().reset_index()
    df_test_last_cycle['RUL'] = df_rul.values

    # Prepare training data
    X_train = df_train.drop(['RUL', 'unit_number', 'time_cycles'], axis=1)
    y_train = df_train['RUL']

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Prepare test data
    X_test = df_test_last_cycle.drop(['RUL', 'unit_number', 'time_cycles'], axis=1)
    y_test = df_test_last_cycle['RUL']
    X_test_scaled = scaler.transform(X_test)

    # Train the model
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)

    # Predictions
    y_train_pred = lr_model.predict(X_train_scaled)
    y_test_pred = lr_model.predict(X_test_scaled)

    # Calculate RMSE and R2 score for training data
    rmse_train_lr = np.sqrt(mean_squared_error(y_train, y_train_pred))
    r2_train_lr = r2_score(y_train, y_train_pred)

    # Calculate RMSE and R2 score for test data
    rmse_test_lr = np.sqrt(mean_squared_error(y_test, y_test_pred))
    r2_test_lr = r2_score(y_test, y_test_pred)

    st.write(f"Root Mean Squared Error on Training set: {rmse_train_lr}")
    st.write(f"R2 Score on Training set: {r2_train_lr}")

    st.write(f"Root Mean Squared Error on Test set: {rmse_test_lr}")
    st.write(f"R2 Score on Test set: {r2_test_lr}")

    # Plotting actual vs predicted RUL for test data
    st.write("### Actual vs Predicted RUL for Test Data")
    fig, ax = plt.subplots()
    ax.bar(y_test.index, y_test, label="Actual RUL", color='blue', width=0.4)
    ax.bar(y_test.index+0.4, y_test_pred, label="Predicted RUL", color='red', width=0.4)
    ax.set_xlabel("Engine Unit Number")
    ax.set_ylabel("RUL")
    ax.legend()

    st.pyplot(fig)



# --------------------------------------- RANDOM FOREST PAGE --------------------------------------- #
# Function to process data for Random Forest
def process_data_for_rf(df_train, df_test):
    # data processing code here
    # just call the 'process_data_for_regression' function as it seems suitable
    df_train = process_data_for_regression(df_train)
    
    # MinMaxScaler code, so integrating that:
    scaler = MinMaxScaler()
    features_to_scale = [col for col in df_train.columns if col not in ['unit_number', 'time_cycles', 'RUL']]
    scaler.fit(df_train[features_to_scale])
    df_train[features_to_scale] = scaler.transform(df_train[features_to_scale])
    df_test[features_to_scale] = scaler.transform(df_test[features_to_scale])

    return df_train, df_test, scaler
if page_selection == "Random Forest":
    st.write("### Training Data (Random Forest)")
    st.write(df_train.head())
    st.write(f"Number of Rows in Training Data: {df_train.shape[0]}")
    st.write(f"Number of Columns in Training Data: {df_train.shape[1]}")

    # Process Data
    df_train, df_test, scaler = process_data_for_rf(df_train, df_test)

    # Split Data for Training and Validation
    train_size = int(len(df_train) * 0.80)
    train_df, val_df = df_train[0:train_size], df_train[train_size:len(df_train)]
    X_train = train_df.drop('RUL', axis=1)
    y_train = train_df['RUL']
    X_val = val_df.drop('RUL', axis=1)
    y_val = val_df['RUL']

    # Slider Inputs for Custom Prediction
    st.sidebar.header("Custom Prediction Inputs (RF)")
    st.sidebar.write("Adjust the sliders based on input data:")
    setting_min = float(df_train['setting_1'].min())
    setting_max = float(df_train['setting_1'].max()) + 0.01 if setting_min == float(df_train['setting_1'].max()) else float(df_train['setting_1'].max())
    setting_1 = st.sidebar.slider('Setting 1', setting_min, setting_max)
    setting_2 = st.sidebar.slider('Setting 2', setting_min, setting_max)
    setting_3 = st.sidebar.slider('Setting 3', setting_min, setting_max)

    sensors = {}
    for i in range(1, 22):
        sensor_min = float(df_train[f's_{i}'].min())
        sensor_max = sensor_min + 0.01 if sensor_min == float(df_train[f's_{i}'].max()) else float(df_train[f's_{i}'].max())
        sensors[f's_{i}'] = st.sidebar.slider(f'Sensor {i}', sensor_min, sensor_max)

    # Train and Predict using Random Forest
    
st.write("### Random Forest Regressor Model Training")

if st.button('Train and Predict using Random Forest'):
    param_grid = {
        'n_estimators': [101],
        'max_features': ['sqrt'],
        'max_depth': [10],
        'min_samples_split': [5],
        'min_samples_leaf': [1]
    }
    rf_reg = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=rf_reg, param_grid=param_grid, 
                           scoring='neg_mean_squared_error', cv=5, 
                           verbose=2, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    st.write(f"Best parameters: {best_params}")

    best_rf_reg = RandomForestRegressor(**best_params)
    best_rf_reg.fit(X_train, y_train)

    y_val_pred = best_rf_reg.predict(X_val)

    rf_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    r2 = r2_score(y_val, y_val_pred)

    st.write(f"Random Forest RMSE after hyperparameter tuning: {rf_rmse}")
    st.write(f"Random Forest R^2 after hyperparameter tuning: {r2}")

    # Plotting actual vs predicted RUL for validation data
    st.write("### Actual vs Predicted RUL for Validation Data")
    fig, ax = plt.subplots()
    ax.scatter(y_val.index, y_val, label="Actual RUL", color='blue')
    ax.scatter(y_val.index, y_val_pred, label="Predicted RUL", color='red', marker='x')
    ax.set_xlabel("Engine Unit Number")
    ax.set_ylabel("RUL")
    ax.legend()

    st.pyplot(fig)
# Bar plot for actual vs predicted RUL for validation data

    # Custom Prediction using RF model
    if st.sidebar.button('Predict RUL using Custom Input (RF)'):
        if 'best_rf_reg' in locals():
            custom_input_data = [[setting_1, setting_2, setting_3] + list(sensors.values())]
            custom_prediction = best_rf_reg.predict(custom_input_data)
            st.sidebar.write(f"Predicted RUL (RF): {custom_prediction[0]:.2f}")
            

# --------------------------------------- Support Vector Regressor --------------------------------------- #            

if page_selection == "Support Vector Regression":

    st.write("### Training Data (Support Vector Regressor)")
    st.write(df_train.head())
    st.write(f"Number of Rows in Training Data: {df_train.shape[0]}")
    st.write(f"Number of Columns in Training Data: {df_train.shape[1]}")

    # Slider Inputs
    st.sidebar.header("Custom Prediction Inputs (SVR)")
    st.sidebar.write("Adjust the sliders based on input data:")

    setting_min = float(df_train['setting_1'].min())
    setting_max = float(df_train['setting_1'].max()) + 0.01 if setting_min == float(df_train['setting_1'].max()) else float(df_train['setting_1'].max())
    setting_1 = st.sidebar.slider('Setting 1', setting_min, setting_max)
    setting_2 = st.sidebar.slider('Setting 2', setting_min, setting_max)
    setting_3 = st.sidebar.slider('Setting 3', setting_min, setting_max)

    sensors = {}
    for i in range(1, 22):
        sensor_min = float(df_train[f's_{i}'].min())
        sensor_max = sensor_min + 0.01 if sensor_min == float(df_train[f's_{i}'].max()) else float(df_train[f's_{i}'].max())
        sensors[f's_{i}'] = st.sidebar.slider(f'Sensor {i}', sensor_min, sensor_max)

    if st.sidebar.button('Predict RUL using Custom Input (SVR)'):
        if 'svr_model' in globals():
            custom_input_data = [[setting_1, setting_2, setting_3] + list(sensors.values())]
            custom_scaled_input = scaler.transform(custom_input_data)
            custom_prediction = lr_model.predict(custom_scaled_input)
            st.sidebar.write(f"Predicted RUL (SVR): {custom_prediction[0]:.2f}")
        
st.write("### Support Vector Regressor Model Training")

if st.button('Train and Predict using Support Vector Regression Model'):
    # Loading test dataset and RUL dataset
    df_test = load_data('test_FD001.txt')
    df_rul = pd.read_csv('RUL_FD001.txt', names=['RUL'])

    # Process the training data
    df_train = process_data_for_regression(df_train)

    # Get the last time step for each engine from test dataset
    df_test_last_cycle = df_test.groupby('unit_number').last().reset_index()
    df_test_last_cycle['RUL'] = df_rul.values

    # Prepare training data
    X_train = df_train.drop(['RUL', 'unit_number', 'time_cycles'], axis=1)
    y_train = df_train['RUL']

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Prepare test data
    X_test = df_test_last_cycle.drop(['RUL', 'unit_number', 'time_cycles'], axis=1)
    y_test = df_test_last_cycle['RUL']
    X_test_scaled = scaler.transform(X_test)

    # Train the model
    # Initialize the base model
    base_model_svr = SVR(kernel='rbf')

    # Initialize the bagging regressor
    svr_model = BaggingRegressor(estimator=base_model_svr, n_estimators=5)

    # Fit the model to the training data
    svr_model.fit(X_train_scaled, y_train)

    

    # Predictions
    y_train_pred = svr_model.predict(X_train_scaled)
    y_test_pred = svr_model.predict(X_test_scaled)

    # Calculate RMSE and R2 score for training data
    rmse_train_svr = np.sqrt(mean_squared_error(y_train, y_train_pred))
    r2_train_svr = r2_score(y_train, y_train_pred)

    # Calculate RMSE and R2 score for test data
    rmse_test_svr = np.sqrt(mean_squared_error(y_test, y_test_pred))
    r2_test_svr = r2_score(y_test, y_test_pred)

    st.write(f"Root Mean Squared Error on Training set: {rmse_train_svr}")
    st.write(f"R2 Score on Training set: {r2_train_svr}")

    st.write(f"Root Mean Squared Error on Test set: {rmse_test_svr}")
    st.write(f"R2 Score on Test set: {r2_test_svr}")

    # Plotting actual vs predicted RUL for test data
    st.write("### Actual vs Predicted RUL for Test Data")
    fig, ax = plt.subplots()
    ax.bar(y_test.index, y_test, label="Actual RUL", color='blue', width=0.4)
    ax.bar(y_test.index+0.4, y_test_pred, label="Predicted RUL", color='red', width=0.4)
    ax.set_xlabel("Engine Unit Number")
    ax.set_ylabel("RUL")
    ax.legend()

    st.pyplot(fig)
    
    
# --------------------------------------- XGBoost Regression (XGB) --------------------------------------- #            

# Function to process data for XGBoost
def process_data_for_xgb(df_train, df_test):
    # data processing code here
    # call the 'process_data_for_regression' function as it seems suitable
    df_train = process_data_for_regression(df_train)
    
    # MinMaxScaler code, so integrating that:
    scaler = MinMaxScaler()
    features_to_scale = [col for col in df_train.columns if col not in ['unit_number', 'time_cycles', 'RUL']]
    scaler.fit(df_train[features_to_scale])
    df_train[features_to_scale] = scaler.transform(df_train[features_to_scale])
    df_test[features_to_scale] = scaler.transform(df_test[features_to_scale])

    return df_train, df_test, scaler
if page_selection == "XGBoost Regression":
    st.write("### Training Data (Random Forest)")
    st.write(df_train.head())
    st.write(f"Number of Rows in Training Data: {df_train.shape[0]}")
    st.write(f"Number of Columns in Training Data: {df_train.shape[1]}")

    # Process Data
    df_train, df_test, scaler = process_data_for_xgb(df_train, df_test)

    # Split Data for Training and Validation
    train_size = int(len(df_train) * 0.80)
    train_df, val_df = df_train[0:train_size], df_train[train_size:len(df_train)]
    X_train = train_df.drop('RUL', axis=1)
    y_train = train_df['RUL']
    X_val = val_df.drop('RUL', axis=1)
    y_val = val_df['RUL']

    # Slider Inputs for Custom Prediction
    st.sidebar.header("Custom Prediction Inputs (XGB)")
    st.sidebar.write("Adjust the sliders based on input data:")
    setting_min = float(df_train['setting_1'].min())
    setting_max = float(df_train['setting_1'].max()) + 0.01 if setting_min == float(df_train['setting_1'].max()) else float(df_train['setting_1'].max())
    setting_1 = st.sidebar.slider('Setting 1', setting_min, setting_max)
    setting_2 = st.sidebar.slider('Setting 2', setting_min, setting_max)
    setting_3 = st.sidebar.slider('Setting 3', setting_min, setting_max)

    sensors = {}
    for i in range(1, 22):
        sensor_min = float(df_train[f's_{i}'].min())
        sensor_max = sensor_min + 0.01 if sensor_min == float(df_train[f's_{i}'].max()) else float(df_train[f's_{i}'].max())
        sensors[f's_{i}'] = st.sidebar.slider(f'Sensor {i}', sensor_min, sensor_max)

    # Train and Predict using XGBoost
    
st.write("### XGBoost Regressor Model Training")

if st.button('Train and Predict using XGBoost'):
    params = {'colsample_bytree': 0.5, 'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 100, 'subsample': 1}
    
    xgb_reg =XGBRegressor(random_state=42)
    
    xgb_reg = XGBRegressor(**params)
    xgb_reg.fit(X_train, y_train)

    # Make predictions on the validation set
    y_val_pred = xgb_reg.predict(X_val)

    # Calculate RMSE
    xgb_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))

    # Calculate R^2
    xgb_r2 = r2_score(y_val, y_val_pred)


    st.write(f"XGBoost RMSE after hyperparameter tuning: {xgb_rmse}")
    st.write(f"XGBoost R^2 after hyperparameter tuning: {xgb_r2}")

    # Plotting actual vs predicted RUL for validation data
    st.write("### Actual vs Predicted RUL for Validation Data")
    fig, ax = plt.subplots()
    ax.scatter(y_val.index, y_val, label="Actual RUL", color='blue')
    ax.scatter(y_val.index, y_val_pred, label="Predicted RUL", color='red', marker='x')
    ax.set_xlabel("Engine Unit Number")
    ax.set_ylabel("RUL")
    ax.legend()

    st.pyplot(fig)
# Bar plot for actual vs predicted RUL for validation data

    # Custom Prediction using XGB model
    if st.sidebar.button('Predict RUL using Custom Input (XGB)'):
        if 'XGB_reg' in locals():
            custom_input_data = [[setting_1, setting_2, setting_3] + list(sensors.values())]
            custom_prediction = XGB_reg.predict(custom_input_data)
            st.sidebar.write(f"Predicted RUL (XGB): {custom_prediction[0]:.2f}")
 
# --------------------------------------- Long Short Term Memory (LSTM) --------------------------------------- #
          
if page_selection == "LSTM Regression":
    st.write("### Training Data (LSTM)")
    st.write(df_train.head())
    st.write(f"Number of Rows in Training Data: {df_train.shape[0]}")
    st.write(f"Number of Columns in Training Data: {df_train.shape[1]}")
    
 # Slider Inputs
    st.sidebar.header("Custom Prediction Inputs (LSTM)")
    st.sidebar.write("Adjust the sliders based on input data:")

    setting_min = float(df_train['setting_1'].min())
    setting_max = float(df_train['setting_1'].max()) + 0.01 if setting_min == float(df_train['setting_1'].max()) else float(df_train['setting_1'].max())
    setting_1 = st.sidebar.slider('Setting 1', setting_min, setting_max)
    setting_2 = st.sidebar.slider('Setting 2', setting_min, setting_max)
    setting_3 = st.sidebar.slider('Setting 3', setting_min, setting_max)

    sensors = {}
    for i in range(1, 22):
        sensor_min = float(df_train[f's_{i}'].min())
        sensor_max = sensor_min + 0.01 if sensor_min == float(df_train[f's_{i}'].max()) else float(df_train[f's_{i}'].max())
        sensors[f's_{i}'] = st.sidebar.slider(f'Sensor {i}', sensor_min, sensor_max)

    if st.sidebar.button('Predict RUL using Custom Input (LR)'):
        if 'lr_model' in globals():
            custom_input_data = [[setting_1, setting_2, setting_3] + list(sensors.values())]
            custom_scaled_input = scaler.transform(custom_input_data)
            custom_prediction = lr_model.predict(custom_scaled_input)
            st.sidebar.write(f"Predicted RUL (LR): {custom_prediction[0]:.2f}")
    
st.write("### Long Short Term Memory Model Training")

if st.button('Train and Predict using Long Short Term Memory Deep Learning Model'):
    # Loading test dataset and RUL dataset
    df_test = load_data('test_FD001.txt')
    df_rul = pd.read_csv('RUL_FD001.txt', names=['RUL'])

    # Process the training data
    df_train = process_data_for_regression(df_train)

    # Get the last time step for each engine from test dataset
    df_test_last_cycle = df_test.groupby('unit_number').last().reset_index()
    df_test_last_cycle['RUL'] = df_rul.values

    # Prepare training data
    X_train = df_train.drop(['RUL', 'unit_number', 'time_cycles'], axis=1)
    y_train = df_train['RUL']

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Prepare test data
    X_test = df_test_last_cycle.drop(['RUL', 'unit_number', 'time_cycles'], axis=1)
    y_test = df_test_last_cycle['RUL']
    X_test_scaled = scaler.transform(X_test)

    # Reshape input to be 3D [samples, timesteps, features]
    X_train_scaled_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
    X_test_scaled_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

    # Train the LSTM model
    model_lstm = Sequential()
    model_lstm.add(LSTM(26, input_shape=(X_train_scaled_lstm.shape[1], X_train_scaled_lstm.shape[2])))
    model_lstm.add(Dense(1, activation='softplus'))
    model_lstm.compile(loss='mse')
    model_lstm.fit(X_train_scaled_lstm, y_train, epochs=10, batch_size=32, verbose=2, shuffle=False)

    # Predictions
    y_train_pred = model_lstm.predict(X_train_scaled_lstm)
    y_test_pred = model_lstm.predict(X_test_scaled_lstm)
    
    y_test_pred = y_test_pred.ravel() # Convert y_test_pred to 1D array

    # Calculate RMSE and R2 score for training data
    rmse_train_lstm = np.sqrt(mean_squared_error(y_train, y_train_pred))
    r2_train_lstm = r2_score(y_train, y_train_pred)

    # Calculate RMSE and R2 score for test data
    rmse_test_lstm = np.sqrt(mean_squared_error(y_test, y_test_pred))
    r2_test_lstm = r2_score(y_test, y_test_pred)

    st.write(f"LSTM Train RMSE: {rmse_train_lstm}")
    st.write(f"LSTM Train R2 Score: {r2_train_lstm}")
    st.write(f"LSTM Test RMSE: {rmse_test_lstm}")
    st.write(f"LSTM Test R2 Score: {r2_test_lstm}")

    # Plotting actual vs predicted RUL for test data
    st.write("### Actual vs Predicted RUL for Test Data")
    fig, ax = plt.subplots()
    ax.bar(y_test.index, y_test, label="Actual RUL", color='blue', width=0.4)
    ax.bar(y_test.index+0.4, y_test_pred, label="Predicted RUL", color='red', width=0.4)
    ax.set_xlabel("Engine Unit Number")
    ax.set_ylabel("RUL")
    ax.legend()
    st.pyplot(fig)
    
    
# --------------------------------------- Dense Neural Network (DNN) --------------------------------------- #
          
if page_selection == "Dense Neural Network":
    st.write("### Training Data (DNN)")
    st.write(df_train.head())
    st.write(f"Number of Rows in Training Data: {df_train.shape[0]}")
    st.write(f"Number of Columns in Training Data: {df_train.shape[1]}")
    
 # Slider Inputs
    st.sidebar.header("Custom Prediction Inputs (DNN)")
    st.sidebar.write("Adjust the sliders based on input data:")

    setting_min = float(df_train['setting_1'].min())
    setting_max = float(df_train['setting_1'].max()) + 0.01 if setting_min == float(df_train['setting_1'].max()) else float(df_train['setting_1'].max())
    setting_1 = st.sidebar.slider('Setting 1', setting_min, setting_max)
    setting_2 = st.sidebar.slider('Setting 2', setting_min, setting_max)
    setting_3 = st.sidebar.slider('Setting 3', setting_min, setting_max)

    sensors = {}
    for i in range(1, 22):
        sensor_min = float(df_train[f's_{i}'].min())
        sensor_max = sensor_min + 0.01 if sensor_min == float(df_train[f's_{i}'].max()) else float(df_train[f's_{i}'].max())
        sensors[f's_{i}'] = st.sidebar.slider(f'Sensor {i}', sensor_min, sensor_max)

    if st.sidebar.button('Predict RUL using Custom Input (LR)'):
        if 'lr_model' in globals():
            custom_input_data = [[setting_1, setting_2, setting_3] + list(sensors.values())]
            custom_scaled_input = scaler.transform(custom_input_data)
            custom_prediction = lr_model.predict(custom_scaled_input)
            st.sidebar.write(f"Predicted RUL (LR): {custom_prediction[0]:.2f}")
    
st.write("### Dense Neural Network Model Training")

if st.button('Train and Predict using Dense Neural Network Deep Learning Model'):
    # Loading test dataset and RUL dataset
    df_test = load_data('test_FD001.txt')
    df_rul = pd.read_csv('RUL_FD001.txt', names=['RUL'])

    # Process the training data
    df_train = process_data_for_regression(df_train)

    # Get the last time step for each engine from test dataset
    df_test_last_cycle = df_test.groupby('unit_number').last().reset_index()
    df_test_last_cycle['RUL'] = df_rul.values

    # Prepare training data
    X_train = df_train.drop(['RUL', 'unit_number', 'time_cycles'], axis=1)
    y_train = df_train['RUL']

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Prepare test data
    X_test = df_test_last_cycle.drop(['RUL', 'unit_number', 'time_cycles'], axis=1)
    y_test = df_test_last_cycle['RUL']
    X_test_scaled = scaler.transform(X_test)

    # Train the LSTM model
     # Create the model
    model_dnn = Sequential()
    model_dnn.add(Dense(32, input_dim=X_train_scaled.shape[1]))  # Hidden layer with activation
    model_dnn.add(Dense(1, activation='linear'))  # Output layer with activation

    # Compile the model
    model_dnn.compile(loss='mean_squared_error')
    
    # Fit the model to the training data
    model_dnn.fit(X_train_scaled, y_train, epochs=10, batch_size=32, verbose=1)

    # Predictions
    y_train_pred = model_dnn.predict(X_train_scaled).flatten()
    y_test_pred = model_dnn.predict(X_test_scaled).flatten()
    
#     # Predict RUL on the training data
#     y_train_pred_nn = model.predict(X_train_scaled).flatten()

#     # Predict RUL on the test data
#     y_test_pred_nn = model.predict(X_test_scaled).flatten()

    
#     y_test_pred = y_test_pred.ravel() # Convert y_test_pred to 1D array

    # Calculate RMSE and R2 score for training data
    rmse_train_dnn = np.sqrt(mean_squared_error(y_train, y_train_pred))
    r2_train_dnn = r2_score(y_train, y_train_pred)

    # Calculate RMSE and R2 score for test data
    rmse_test_dnn = np.sqrt(mean_squared_error(y_test, y_test_pred))
    r2_test_dnn = r2_score(y_test, y_test_pred)

    st.write(f"LSTM Train RMSE: {rmse_train_dnn}")
    st.write(f"LSTM Train R2 Score: {r2_train_dnn}")
    st.write(f"LSTM Test RMSE: {rmse_test_dnn}")
    st.write(f"LSTM Test R2 Score: {r2_test_dnn}")

    # Plotting actual vs predicted RUL for test data
    st.write("### Actual vs Predicted RUL for Test Data")
    fig, ax = plt.subplots()
    ax.bar(y_test.index, y_test, label="Actual RUL", color='blue', width=0.4)
    ax.bar(y_test.index+0.4, y_test_pred, label="Predicted RUL", color='red', width=0.4)
    ax.set_xlabel("Engine Unit Number")
    ax.set_ylabel("RUL")
    ax.legend()
    st.pyplot(fig)


# In[ ]:
