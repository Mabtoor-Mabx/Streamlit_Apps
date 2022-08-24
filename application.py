# Import Libraries

from this import d
import streamlit as st
import seaborn as sns
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Making Containers

header = st.container()
dataset = st.container()
feature = st.container()
model_training = st.container()


#Header 
with header:
    st.title('Penguins APP')
    st.write('In this Application , We Describe MAE, MSE and R2-Score of Penguin App using Machine Learning Algorithm')


#Dataset
with dataset:
    st.subheader('The Dataset is avalible in Seaborn Library')
    df = sns.load_dataset('penguins')
    df = df.dropna()
    st.write(df.head())
    st.subheader('Bar Chart of Species in Dataset')
    st.bar_chart(df['species'].value_counts())
    st.subheader('Bar Chart of Gender in Dataset')
    st.bar_chart(df['sex'].value_counts())
    st.subheader('Line Chart of Bill Length')
    st.line_chart(df['bill_length_mm'].sample(100))


# Features
with feature:
    st.subheader('Features of Penguin Dataset')
    st.write(df.columns)

# Model Training
with model_training:

    # Making Columns
    value, display = st.columns(2)
    
    #Display Column of Input

    max_depth = value.slider('What is The Value of Bill Length', min_value=20, max_value=120, value=20, step=6)

    # N-Estimators

    n_estimator = value.selectbox('How Many Tress You Should Have in Random Forest', options=[100,200,300,400,500,'No Limit'])

    # Input Features

    input_features = value.selectbox('Which Feature We should Use', options=['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g'])



    # Machine Learning Model

    if n_estimator =='No Limit':
        model = RandomForestRegressor(max_depth=max_depth)
    else:
        model = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimator)




    # Define X and Y

    X= df[[input_features]]
    Y = df[['bill_depth_mm']]

    # Fit The Model

    model.fit(X, Y)
    predict = model.predict(Y)


    #Display Metrics 
    display.subheader('Mean Absolute Error of Model is :')
    display.write(mean_absolute_error(Y, predict))
    display.subheader('Mean Square Error of the Model is:')
    display.write(mean_squared_error(Y, predict))
    display.subheader('R2-Value of The Model is :')
    display.write(r2_score(Y,predict))






