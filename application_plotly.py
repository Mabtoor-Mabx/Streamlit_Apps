
#Import Libraries

import streamlit as st
import plotly.express as px
import pandas as pd

#Load Dataset

st.title('Explore GapMinder Dataset with Plotly Express')
st.text('In this APP, I will try to explore gapminder dataset using interactive data visualization library called Plotly.')
df = px.data.gapminder()
st.write(df)

#Displaying Feature 

st.subheader('Features in GapMinder Dataset')
st.write(df.columns)


# Summary Stat (For Exploratory Data Analysis)

st.subheader('Describing The Dataset')

st.write(df.describe())

# Data Management

year_option = df['year'].unique().tolist()

year = st.selectbox('Which Year Of Dataset you want to plot?', year_option,0)

df = df[df['year']==year]


#Plotting

fig = px.scatter(df,x='gdpPercap',
                 y='lifeExp',
                 size='pop',
                 color='country',
                 hover_name='country',
                 log_x=True,
                 size_max=55,
                 range_x=[100,10000],
                 range_y=[20,90])

fig.update_layout(width=700)

st.write(fig)

