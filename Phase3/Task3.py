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

dbName = "CSE515-MWD-ProjectPhase2-Final"
odd_feature_collection = connect_to_db(dbName,'image_features_odd')
feature_collection = connect_to_db(dbName,'image_features')
similarity_collection = connect_to_db(dbName,'image_similarities')

option = st.selectbox(
        "Select Classifier",
        ("Decision Tree", "PPR","k-Means"),
        label_visibility=st.session_state.visibility,
        disabled=st.session_state.disabled,
    )

if option == "k-Means":
    k = st.number_input('Enter k for k-Means classifier',placeholder="Type a number...",format = "%d",min_value=1,max_value=4338)
    teleport_prob = 0
elif option=="PPR":
    k = 0
    teleport_prob = st.number_input('Enter teleportation probability for PageRank',placeholder="Type a number...",format="%f",min_value=0.0,max_value=1.0)
else:
    k = 0
    teleport_prob = 0

if st.button("Run", type="primary"):
    with st.spinner('Calculating...'):
        with st.container():    
            classifier(option,feature_collection,odd_feature_collection,similarity_collection,caltech101,k=k,teleport_prob = teleport_prob)
else:
    st.write("")