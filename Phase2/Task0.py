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
from pathlib import Path

import streamlit as st

from Utilities.DisplayUtils import *
from FeatureDescriptors.FeatureDescriptorUtils import *
from FeatureDescriptors.SimilarityScoreUtils import *
from MongoDB.MongoDBUtils import *

mod_path = Path(__file__).parent.parent

caltech101 = Caltech101(str(mod_path) + "/caltech101",download=True)
collection = connect_to_db('CSE515-MWD-Kesudh_Giri-ProjectPhase1','image_features')

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

idx = st.number_input('Enter ImageID',placeholder="Type a number...",format = "%d",min_value=0,max_value=8676)

if st.button("Run", type="primary"):
    document = collection.find_one({'_id': idx})
    print(document['_id'])
    image = np.array(document['image'], dtype=np.uint8)
    st.write("Query Image:")
    st.image(image=image, caption="ImageID: "+str(idx),channels="BGR", width = 300)
    queryksimilar(idx, 5,collection,caltech101)
else:
    st.write("")