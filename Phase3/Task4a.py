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


dbName = "CSE515-MWD-ProjectPhase2-Final"
odd_feature_collection = connect_to_db(dbName,'image_features_odd')
feature_collection = connect_to_db(dbName,'image_features')
similarity_collection = connect_to_db(dbName,'image_similarities')

num_layers = st.number_input('Enter number of layers for LSH',placeholder="Type a number...",format = "%d",min_value=1,max_value=1000)
num_hashes = st.number_input('Enter number of hashes for LSH',placeholder="Type a number...",format = "%d",min_value=1,max_value=1000)

if st.button("Run", type="primary"):
    with st.spinner('Calculating...'):
        with st.container():    
            lsh = lsh_calc(feature_collection,num_layers, num_hashes)
            # Display the buckets
            for i in range(1,num_layers+1):
                lsh.display_buckets(i)
else:
    st.write("")