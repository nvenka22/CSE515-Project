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
from st_aggrid import AgGrid

from Utilities.DisplayUtils import *
from FeatureDescriptors.FeatureDescriptorUtils import *
from FeatureDescriptors.SimilarityScoreUtils import *
from MongoDB.MongoDBUtils import *
from sys import argv

if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False

dbName = "CSE515-MWD-ProjectPhase2-Final"
odd_feature_collection = connect_to_db(dbName,'image_features_odd')
feature_collection = connect_to_db(dbName,'image_features')
similarity_collection = connect_to_db(dbName,'image_similarities')

if len(argv)==5:

    num_layers = int(argv[1])
    num_hashes = int(argv[2])
    query_image = int(argv[3])
    t = int(argv[4])

    with st.container():

        nearest_indices, unique_indices, lsh, distances, hash_values = lsh_search(feature_collection,odd_feature_collection,num_layers, num_hashes,query_image,t)

    feedback = {}

    feedback_mapping = {'Very Relevant (R+)':3, 'Relevant (R)':2, 'Irrelevant (I)':1,'Very Irrelevant (I-)':0}

    for index in nearest_indices:
        feedback[index] = -1

    user_feedback = st.text_area("Feedback - Enter comma separated values from 0-3 to Indicate a Relevance Score for each image in the result, where 3 is for Very Relevant (R+), 2 if for Relevant (R), 1 is for Irrelevant (I), and 0 is for Very Irrelevant (I-)")

    option = st.selectbox(
            "Select Relevance Feedback Model",
            ("SVM", "Probabilistic"),
            label_visibility=st.session_state.visibility,
            disabled=st.session_state.disabled,
        )

    if st.button('Submit Feedback',type="primary"):
        with st.expander("Feedback"):
            feedback_vals = user_feedback.split(",")

            idx = 0
            for index in nearest_indices:
                feedback[index] = int(feedback_vals[idx].strip())
                idx+=1
            relevance_feedback(feedback,distances,unique_indices,hash_values,feature_collection)