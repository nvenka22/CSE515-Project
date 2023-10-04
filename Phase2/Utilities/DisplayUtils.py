import os
import cv2
from torchvision.models import resnet50
from torchvision.datasets import Caltech101
import pandas as pd
import numpy as np
import torch
import torchvision.transforms as transforms
from scipy.stats import moment
from PIL import Image
import warnings
warnings.filterwarnings("ignore")
from scipy.spatial.distance import cosine
from tqdm import tqdm
from scipy.stats import skew
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics.pairwise import cosine_similarity

import streamlit as st
from pathlib import Path

def to_base64(image):
    from io import BytesIO
    import base64

    image_pil = Image.fromarray(image)
    buffered = BytesIO()
    image_pil.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def display_image_centered(image,idx):
    left_co, cent_co,last_co = st.columns(3)
    with cent_co:
            st.image(image=image, caption="ImageID: "+idx,channels="BGR", width = 300)
    
def display_feature_vector(vector, heading):

    with st.expander(heading):
        st.write(vector)
    
def display_color_moments(color_moments):
    
    with st.expander("Query Image Color Moments"):
        st.write(color_moments.tolist())
    
    
    
def display_hog(hog_descriptor, cell_size=(30, 10)):

    with st.expander("Query Image HOG Descriptor"):
        st.write(hog_descriptor)
    
    
def display_images(images, indices, similarity_scores, rows, cols):
    
    k = len(images)

    buckets = [[],[],[],[],[]]

    for i in range(0,k,5):

        if i<k:
            buckets[0].append([images[i],indices[i],similarity_scores[i]])

        if i+1<k:
            buckets[1].append([images[i+1],indices[i+1],similarity_scores[i+1]])

        if i+2<k:
            buckets[2].append([images[i+2],indices[i+2],similarity_scores[i+2]])

        if i+3<k:
            buckets[3].append([images[i+3],indices[i+3],similarity_scores[i+3]])

        if i+4<k:
            buckets[4].append([images[i+4],indices[i+4],similarity_scores[i+4]])


    with st.container():
        
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:

            for data in buckets[0]:
                st.image(data[0],caption = "Image ID: "+str(data[1])+" Similarity Score: "+str(data[2]))

        with col2:

            for data in buckets[1]:
                st.image(data[0],caption = "Image ID: "+str(data[1])+" Similarity Score: "+str(data[2]))

        with col3:

            for data in buckets[2]:
                st.image(data[0],caption = "Image ID: "+str(data[1])+" Similarity Score: "+str(data[2]))

        with col4:

            for data in buckets[3]:
                st.image(data[0],caption = "Image ID: "+str(data[1])+" Similarity Score: "+str(data[2]))

        with col5:

            for data in buckets[4]:
                st.image(data[0],caption = "Image ID: "+str(data[1])+" Similarity Score: "+str(data[2]))



def show_ksimilar(k_similar,collection):
    images = []
    indices = []
    similarity_scores = []
    count = len(k_similar.keys())
    rows = int(count/5)
    if (rows*5)<count: rows+=1
    cols = 5
    for index in k_similar.keys():
        document = collection.find_one({'_id': int(index)})
        images.append(cv2.cvtColor(np.array(document['image'], dtype=np.uint8), cv2.COLOR_BGR2RGB))
        indices.append(int(index))
        similarity_scores.append(k_similar[str(index)])
        
    display_images(images, indices, similarity_scores, rows, cols)    