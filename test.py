import streamlit as st
import seaborn as sns

df = sns.load_dataset('iris')

st.header('Welcome To Iris Dataset')
st.text('This Dataset is loaded in seaborn and dataset is called iris dataset.')

st.header('Data Set Information')

st.write('This is perhaps the best known database to be found in the pattern recognition literature. Fishers paper is a classic in the field and is referenced frequently to this day. (See Duda & Hart, for example.) The data set contains 3 classes of 50 instances each, where each class refers to a type of iris plant. One class is linearly separable from the other 2; the latter are NOT linearly separable from each other')

st.header('Dataset')
st.write(df.head())


