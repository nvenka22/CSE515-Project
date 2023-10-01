#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 19:22:18 2023

@author: nikhilvr
"""

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
from tqdm.notebook import tqdm
from scipy.stats import skew
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics.pairwise import cosine_similarity
from ipywidgets import interact, widgets
from IPython.display import display, Markdown, HTML
from IPython.display import clear_output

import streamlit as st

from Utilities.DisplayUtils import *
from FeatureDescriptors.FeatureDescriptorUtils import *
from FeatureDescriptors.SimilarityScoreUtils import *
from MongoDB.MongoDBUtils import *

#utils=Task0Utility()

caltech101 = Caltech101("~/CSE 515 - Multimedia Web Databases/",download=True)
collection = connect_to_db('image_features')

#print(caltech101.__getitem__(index = 0)[0] ) #max index 8676

dataset_mean_values = [0, 0, 0]
dataset_std_dev_values = [0, 0, 0]

for idx in range(len(caltech101)):
    img = caltech101.__getitem__(index=idx)[0]
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img = img / 255.0  # Normalize pixel values to [0, 1]

    # Calculate mean and standard deviation for each channel
    for i in range(3):
        dataset_mean_values[i] += np.mean(img[:,:,i])
        dataset_std_dev_values[i] += np.std(img[:,:,i])

# Calculate final mean and standard deviation values
dataset_size = len(caltech101)
dataset_mean_values = [val / dataset_size for val in dataset_mean_values]
dataset_std_dev_values = [val / dataset_size for val in dataset_std_dev_values]

# print(f'Mean values: {dataset_mean_values}')
# print(f'Standard deviation values: {dataset_std_dev_values}')

#Calculate required feature space vectors for each image
#push_dataset_to_mongodb(dataset = caltech101)

idx = st.number_input('Enter ImageID',placeholder="Type a number...",format = "%d",min_value=0,max_value=8676)

document = collection.find_one({'_id': idx})
image = np.array(document['image'], dtype=np.uint8)

if idx!=None:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("")
    
    with col2:
        st.write("Query Image:")
        st.image(image=image, caption="ImageID: "+str(idx),channels="BGR", width = 300)
        queryksimilar(idx, 5,collection,caltech101)
        
        
    
    with col3:
        st.write("")