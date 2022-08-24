
# Import Libraries

import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np

# Load Dataset (This Time We Load CarShares Dataset)

st.title('Explore The Dataset of Car Shares from Plotly')
st.text('In this APP, I will try to explore Car Shares dataset using interactive data visualization library called Plotly.')
df = px.data.carshare()
st.write(df.head())

# Features
st.subheader('Displaying The Features of Car Shares Dataset')
st.write(df.columns)

#Summary Stat (For Exploratory Data Analysis)

st.subheader('Describe The Car Shares Dataset')
st.write(df.describe())

# Data Management

peak_option = df['peak_hour'].unique().tolist()

peak = st.selectbox('Which Peak Hour Value You Want to use in our dataset', peak_option, 0)

df = df[df['peak_hour']==peak]


# Plotting The Car Shares Dataset

figure = px.scatter(df,
                   x='centroid_lat', 
                    y='centroid_lon',
                    size='car_hours',
                    color='car_hours',
                   log_x=True,
                   size_max=55,
                   range_x=[100,1000],
                   range_y=[10,50])


st.write(figure)
