
# import Libraries

import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report


# Header 

st.markdown('''
# **EDA WEB APP**
**In This Application, You can Analyze & Describe Data & Perform All  EDA Operations Easily by Just Uploading The Dataset**
''')


#Upload File From PC


with st.sidebar.header('Upload Your Dataset(.csv)'):
    uploaded_file = st.sidebar.file_uploader('Upload Your File', type=['csv', 'txt', 'xlsx'])
    df = sns.load_dataset('titanic')
    st.sidebar.markdown("[ExampleCSVFile](https://raw.githubusercontent.com/codebasics/py/master/pandas/11_melt/weather.csv)")
    
    
# Profiling Report For Pandas

if uploaded_file is not None:
    @st.cache
    def load_csv():
        csv = pd.read_csv(uploaded_file)
        return csv
    df= load_csv()
    pr =ProfileReport(df, explorative=True)
    st.header('1- **Input Data Frame**')
    st.write(df)
    st.write('---')
    st.header("2- **Profiling Report With Pandas**")
    st_profile_report(pr)
else:
    st.info('Waiting To Upload CSV File')
    if st.button('Press This Button To Use Example Dataset'):
        # Sample Dataset
        def load_csv():
            a = pd.DataFrame(np.random.rand(100,5), columns=['apple', 'ball', 'carrot', 'donkey', 'elephant'])
            return a
        df = load_csv()
        pr = ProfileReport(df, explorative=True)
        st.header("1- **Input Data Frame**")
        st.write(df)
        st.write('---')
        st.header("2- **Profiling Report With Pandas**")
        st_profile_report(pr)
            
        
        
    
