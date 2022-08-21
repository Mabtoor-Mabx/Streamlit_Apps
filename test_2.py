import streamlit as st
import seaborn as sns

st.header('Corn or Maize Leaf Disease Dataset')
st.text('Artificial Intelligence based classification of diseases in maize/corn plants')
st.header('About Dataset')
st.text('A dataset for classification of corn or maize plant leaf diseases')
st.header('Note')
st.text('This dataset has been made using the popular PlantVillage and PlantDoc datasets. During the formation of the dataset certain images have been removed which were not found to be useful. The original authors reserve right to the respective datasets. If you use this dataset in your academic research, please credit the authors.')

sns.get_dataset_names()

dataset = sns.load_dataset('dots')

st.write(dataset)