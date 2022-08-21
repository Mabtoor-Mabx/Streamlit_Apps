# Import Libraries

import streamlit as st
import seaborn as sns
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Making Containers

header = st.container()
data_sets = st.container()
features = st.container()
model_training = st.container()


# Header

with header:
    st.title('Titanic App! My First APP')
    st.text('We are Creating Titanic App in which we used titanic dataset and Using Machine Learning Algorithm.')

#Datasets

with data_sets:
    st.subheader('We Are Loading Dataset. The Dataset is Loading Using Seaborn')
    df = sns.load_dataset('titanic')
    df = df.dropna()
    st.write(df.head())
    st.subheader('This Bar Chart Describe The Gender in Titanic Dataset')
    st.bar_chart(df['sex'].value_counts())
    st.subheader('Now We are Displaying Some classes in our dataset')
    st.bar_chart(df['class'].value_counts())
    st.subheader('Now We Display Some Random Ages of Our Dataset')
    st.bar_chart(df['age'].sample(15))

#Features

with features:
    st.subheader('These are The Features of Titanic Dataset')
    st.markdown('**Passenger Id**: and id given to each traveler on the boat')
    st.markdown('**Pclass**: the passenger class. It has three possible values: 1,2,3 first, second and third class')
    st.markdown('**The Name of the passenger**')
    st.markdown('**Sex**')
    st.markdown('**Age**')
    st.markdown('**SibSp:** number of siblings and spouses traveling with the passenger')
    st.markdown('**Parch:** number of parents and children traveling with the passenger')
    st.markdown('**The ticket number**')
    st.markdown('**The ticket Fare**')
    st.markdown('**The cabin number**')
    st.markdown('**The embarkation**. This describe three possible areas of the Titanic from which the people embark. Three possible values S,C,Q')

 
with model_training:

    # Making Columns
    input, display = st.columns(2)

    # Add List of Features

    # input.write(df.columns())

    # Display Column of Input Values
    max_depth =input.slider('How Many Peoples You have in Dataset', min_value=20, max_value=100, value=20, step=5)

    # N-Estimatos (Because We Are Using Random Forest Machine Learning Algorithm)

    n_estimators = input.selectbox('How Many Trees You Should Have in Dataset', options=[100,200,300,'No Limit'])

    # Input Features For User

    input_features= input.text_input('Which Features We Should Use?', value='fare')

    #Machine Learning Model

    if n_estimators  == 'No Limit':
        model = RandomForestRegressor(max_depth=max_depth)
    else:
        model = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)  


    # Define Input X and Y

    X=df[[input_features]]
    y=df[['fare']]

    # Fit The Model

    model.fit(X,y)
    predict = model.predict(y)

    #Display Metrics

    display.subheader('Mean Absolute Error of Model is:')
    display.write(mean_absolute_error(y, predict))
    display.subheader('Mean Square Error of Model is: ')
    display.write(mean_squared_error(y,predict))
    display.subheader('R2 Value of the Model is :')
    display.write(r2_score(y,predict))


    


