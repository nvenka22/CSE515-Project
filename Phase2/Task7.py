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
from pathlib import Path
import pickle

import streamlit as st

from Utilities.DisplayUtils import *
from FeatureDescriptors.FeatureDescriptorUtils import *
from FeatureDescriptors.SimilarityScoreUtils import *
from MongoDB.MongoDBUtils import *

mod_path = Path(__file__).parent.parent

caltech101 = Caltech101(str(mod_path) + "/caltech101",download=True)

if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False

dbName = "CSE515-MWD-Vaishnavi-ProjectPhase2"
odd_feature_collection = connect_to_db(dbName,'image_features_odd')
feature_collection = connect_to_db(dbName,'image_features')
similarity_collection = connect_to_db(dbName,'image_similarities')

idx = st.number_input('Enter ImageID',placeholder="Type a number...",format = "%d",min_value=0,max_value=8676)
k = st.number_input('Enter k for similar images',placeholder="Type a number...",format = "%d",min_value=1,max_value=8676)

dimred = st.selectbox(
        "Select Dimensionality Reduction Technique",
        ("SVD", "NNMF", "LDA","k-Means"),
        label_visibility=st.session_state.visibility,
        disabled=st.session_state.disabled,
    )
uploaded_file = st.file_uploader("Choose an image file", type=['png', 'jpeg', 'jpg'])

if st.button("Run", type="primary"):
    with st.container():    
    	get_simlar_ls()    	
elif st.button("Run for uploaded image", type="primary") and uploaded_file is not None:
    with st.container():    
        get_simlar_ls_img()     
else:
    st.write("")