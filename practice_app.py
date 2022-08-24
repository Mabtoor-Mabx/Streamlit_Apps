import streamlit as st
import seaborn as sns
import pandas as pd

# Making Containers

header = st.container()
data_set = st.container()
features = st.container()
footer = st.container()


# Header

st.title('Planets Dataset')
st.text('Planet: Understanding the Amazon from Space')


# Dataset 

st.header('About Dataset')
st.subheader('Description')
st.text('For one or another reason, it is impossible to load the initial "Planet: Understanding the Amazon from Space" dataset into the kernel. That is why this dataset was created. Refer to the initial competition full description')
st.subheader('Content')
st.text('I have added only the csv file using seaborn. They are enough to achieve fairly accurate results')

df = sns.load_dataset('planets')
st.write(df.head())

# Display  Features

st.bar_chart(df['method'].value_counts())
st.line_chart(df['distance'].value_counts())
st.area_chart(df['method'].value_counts())


