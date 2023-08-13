#!/usr/bin/env python
# coding: utf-8

# In[3]:


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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

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

st.title("Predictive Maintenance Dataset Explorer and Predictor")

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

# Add SVR to the page selection sidebar
page_selection = st.sidebar.radio("Select a Page", ["Linear Regression", "Random Forest", "Support Vector Regression"])
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



# Necessary imports
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

# ...[rest of your existing imports and codes]...

# --------------------------------------- RANDOM FOREST PAGE --------------------------------------- #
# --------------------------------------- RANDOM FOREST PAGE --------------------------------------- #
# --------------------------------------- RANDOM FOREST PAGE --------------------------------------- #
# Function to process data for Random Forest
def process_data_for_rf(df_train, df_test):
    # Your data processing code here
    # I'll just call the 'process_data_for_regression' function as it seems suitable
    df_train = process_data_for_regression(df_train)
    
    # You also had some MinMaxScaler code, so integrating that:
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

if page_selection == "Support Vector Regressor using Bagging Method":

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
    
#     # Custom Prediction using RF model
#     if st.sidebar.button('Predict RUL using Custom Input (SVR)'):
#         if 'best_rf_reg' in locals():
#             custom_input_data = [[setting_1, setting_2, setting_3] + list(sensors.values())]
#             custom_prediction = best_rf_reg.predict(custom_input_data)
#             st.sidebar.write(f"Predicted RUL (RF): {custom_prediction[0]:.2f}")



# In[ ]:




