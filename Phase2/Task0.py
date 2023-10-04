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
odd_feature_collection = connect_to_db('CSE515-MWD-Kesudh_Giri-ProjectPhase2','image_features_odd')
feature_collection = connect_to_db('CSE515-MWD-Kesudh_Giri-ProjectPhase2','image_features')
similarity_collection = connect_to_db('CSE515-MWD-Kesudh_Giri-ProjectPhase2','image_similarities')

idx = st.number_input('Enter ImageID',placeholder="Type a number...",format = "%d",min_value=0,max_value=8676)
k = st.number_input('Enter k for similar images',placeholder="Type a number...",format = "%d",min_value=1,max_value=8676)

uploaded_file = st.file_uploader("Choose an image file", type=['png', 'jpeg', 'jpg'])

if st.button("Run", type="primary") and uploaded_file is None:
    with st.container():    
        queryksimilar(idx, k,odd_feature_collection,feature_collection,similarity_collection,caltech101)
elif st.button("Run for uploaded image", type="primary") and uploaded_file is not None:
    with st.container():
        
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)

        st.markdown("Query Image")
        
        image = cv2.cvtColor(opencv_image,cv2.COLOR_RGB2BGR)
        image = cv2.resize(image, dsize=(300, 100), interpolation=cv2.INTER_AREA)
        left_co, cent_co,last_co = st.columns(3)
        with cent_co:
            st.image(image, channels="BGR")
        
        queryksimilar_newimg(image, k,odd_feature_collection,feature_collection,similarity_collection,caltech101)
else:
    st.write("")