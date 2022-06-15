import streamlit as st
from PIL import Image
import os
import json
#from resnet_model import ResnetModel
import numpy as np
import pandas as pd
from fastai.vision.all import (
    load_learner,
    PILImage,
    Resize,
)
from fastai.vision.all import *
from fastai.vision.widgets import *
import pathlib
temp = pathlib.PosixPath
pathlib.WindowsPath = pathlib.PosixPath #if run on window and not streamlit cloud pathlib.PosixPath = pathlib.WindowsPath

model = load_learner("deploy/tytc_resnet34_fastai_R8.pkl") # load model

st.title("Is that a Supra?!") #Title
st.markdown('"Is that a Supra?!" is a project that will help you identify a Toyota car\'s model from the image you upload.') #information
st.markdown("Please upload your image of Toyota car or use the sample images on the left sidebar.") #information

sample_path = ("deploy/sample_images") #folder sameple images
file_name = os.listdir(sample_path) #file name from folder
sample_image = st.sidebar.selectbox(   #create selectbox sidebar
    'Sample images',
    (file_name))

file = st.file_uploader("Upload your image") #upload file
if file is None:
    img = PILImage.create(os.path.join(sample_path, sample_image))
    st.title("Here is the sample image") #display sample image
    st.image(img)

else:
    img = PILImage.create(file)
    st.title("Here is the image you've selected") #display selected image
    st.image(img)

a, b, c = model.predict(img) #predict model

if a in ['supra']: #easter egg
    st.success(f"Wow! That is a **{a}**  with the probability of **{c[b]*100:.02f}**%") #result display
    st.balloons()

else:
    st.success(f"The car model is **{a}**  with the probability of **{c[b]*100:.02f}**%") #result display

#st.markdown('https://www.youtube.com/watch?v=8sgycukafqQ&list=RD2miAJe2OE8U&index=2')




