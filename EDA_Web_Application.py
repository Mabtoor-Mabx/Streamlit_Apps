
# Import Libraries
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report


# Header of The APP

st.markdown('''
# **EDA WEB Application**

**Explore And Describe Dataset and Perform EDA Functionality By Uploading The Dataset**
''')


# Upload The File From PC

with st.sidebar.header('Upload Your Dataset.(.csv .txt)'):
    uploaded_file = st.sidebar.file_uploader('Upload Your File. Some File Might Not work', type=['csv', 'txt'])
    st.sidebar.markdown('[Example CSV and TXT file](https://raw.githubusercontent.com/codebasics/py/master/pandas/11_melt/weather.csv)')

    
# Profiling Report For Pandas

if uploaded_file is not None:
    @st.cache # Cache Function is used to save dataset and make it possible for run EDA WebAPP Fast
    
    # Create A function to load CSV file
    
    def load_csv():
        csv = pd.read_csv(uploaded_file)
        return csv
    df = load_csv()
    pr = ProfileReport(df, explorative=True)
    st.header('**1-Input Data Frame**')
    st.write(df)
    st.write('---')
    st.header('**2-Profiling Report With Pandas**')
    st_profile_report(pr)

else:
    
    st.info('Waiting To Upload .CSV or .TXT File')
    
    # Create Sample Data Button If You have no Dataset
    
    if st.button('Press This Button To Perform EDA on Sample Data'):
        
        # Create Function To Create Sample Data and Load it to perform EDA Function
        
        def load_csv():
            data = pd.DataFrame(np.random.rand(100,5), columns=['Apple', 'Bat', 'Cat', 'Dog', 'Eye'])
            return data
        df = load_csv()
        pr = ProfileReport(df, explorative=True)
        st.header('**1-Input Data Frame**')
        st.write(df)
        st.write('---')
        st.header('**2-Profiling Report With Pandas**')
        st_profile_report(pr)
        st.subheader('Bar Chart Of Sample Dataset')
        st.bar_chart(df.head())
        st.subheader('Line Chart Of Sample Dataset')
        st.line_chart(df.head())
        

        





