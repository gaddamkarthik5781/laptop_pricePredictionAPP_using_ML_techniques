import streamlit as st
import joblib
import pickle
import numpy as np

# Load the pipeline and dataframe
try:
    pipeline = joblib.load('RandomForestRegressor_pipeline.joblib')
    with open('dataframe.pkl', 'rb') as f:
        df = pickle.load(f, encoding='latin1')
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

st.title("Laptop Price Prediction")

# Collect user input
company = st.selectbox('Brand', df['Company'].unique())
type_L = st.selectbox('Type', df['TypeName'].unique())
ram = st.selectbox('RAM (in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])
weight = st.number_input('Weight of the Laptop')
touchscreen = st.selectbox('TouchScreen', ['No', 'Yes'])
ips = st.selectbox('IPS', ['No', 'Yes'])
screen_size = st.number_input('Screen Size')
resolution = st.selectbox('Screen Resolution',
                          ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1600',
                           '2560x1440', '2304x1440'])
cpu = st.selectbox('CPU', df['Cpu_Brand'].unique())
hdd = st.selectbox('HDD (in GB)', [0, 128, 256, 512, 1024, 2048])
ssd = st.selectbox('SSD (in GB)', [0, 8, 128, 256, 512, 1024])
gpu = st.selectbox('GPU', df['Gpu_brand'].unique())
os = st.selectbox('OS', df['OS'].unique())

if st.button('Predict Price'):
    touchscreen = 1 if touchscreen == 'Yes' else 0
    ips = 1 if ips == 'Yes' else 0
    X_res, Y_res = map(int, resolution.split('x'))
    ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / screen_size
    query = np.array([company, type_L, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os])
    query = query.reshape(1, 12)

    # Debugging output for query
    st.write("Query data:", query)

    try:
        predicted_price = int(np.exp(pipeline.predict(query)[0]))
        st.title(f"The Laptop predicted price of this configuration could be between Rs {predicted_price-1001}/- to Rs {predicted_price+1999}/-")
    except Exception as e:
        st.error(f"Prediction error: {e}")
