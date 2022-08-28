
# Import Libraries

import streamlit as st
from PIL import Image

# Heading 

st.header('Add Image, Audio and Video in Streamlit')

# Add Image
st.subheader('Cat Image')
images = Image.open('cat.jpg')
st.image(images)

# Add Video
st.subheader('Cat Video')
videos = open('cat.mp4','rb')
st.video(videos)

# Add Audio

st.subheader('Cat Audio')
audios = open('cat.mp3', 'rb')
st.audio(audios)
