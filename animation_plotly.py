
# Import Libraries

import streamlit as st
import plotly.express as px
import numpy as np
import pandas as pd

# Load Datset

st.title('GapMinder APP')
st.text('Explore the dataset of Gapminder using one of the famous library of datasets called plotly')
df = px.data.gapminder()
st.write(df.head())

# Display Features

st.subheader('Displaying The Features of GapMinder Dataset')
st.write(df.columns)


# Summary Stat (Exploratory Data Analysis)

st.subheader('Analyse The Data By Describing Dataset')
st.write(df.describe())



#Data Management

year_option = df['year'].unique().tolist()
# year = st.selectbox('Which Year of Dataset You want to plot?', year_option, 0)
# # df= df[df['year']==year]

# Plotting The Graph
fig = px.scatter(df, x='gdpPercap',
                 y='continent',
                 size='pop', 
                 color='country',
                 hover_name='country',
                 animation_frame='year',
                 animation_group='country',
                 log_x=True, 
                 size_max=55,
                 range_x=[100,10000],
                 range_y=[20,90]
                )

st.write(fig)
