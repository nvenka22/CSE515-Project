#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 20:08:46 2023

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


#Color Moments
def featurenormalize(feature_vector):
    # Normalize the feature vector to have unit length
    if np.isnan(feature_vector).any():
        # Handle the presence of NaN values (e.g., replace with zeros)
        feature_vector = np.nan_to_num(feature_vector)

    # Calculate the L2 norm of the feature_vector
    norm = np.linalg.norm(feature_vector)
    return feature_vector / norm if norm != 0 else feature_vector

def euclidean_distance_calculator(x, y):
    return np.linalg.norm(x - y)

def similarity_score_color_moments(moments1, moments2):
    # Normalize the feature vectors
    moments1_normalized = featurenormalize(moments1)
    moments2_normalized = featurenormalize(moments2)
    
    # Calculate the Euclidean Distance
    similarity = euclidean_distance_calculator(moments1_normalized, moments2_normalized)

    return round(similarity,5)



#Similarity for HOG

def cosine_similarity_calculator(descriptor1, descriptor2, normalize=True):
    if normalize:
        descriptor1 = featurenormalize(descriptor1)
        descriptor2 = featurenormalize(descriptor2)
    
    # Reshape descriptors as needed
    descriptor1 = np.reshape(descriptor1, (1, -1))
    descriptor2 = np.reshape(descriptor2, (1, -1))

    similarity = cosine_similarity(descriptor1, descriptor2)
    return similarity[0][0]

def similarity_score_hog(descriptor1, descriptor2):
    return round(cosine_similarity_calculator([descriptor1], [descriptor2]),5)

#Cosine Similarity for Avgpool
def similarity_score_avgpool(descriptor1, descriptor2):
    # Calculate Cosine Similarity for avgpool descriptors
    return round(cosine_similarity_calculator([descriptor1], [descriptor2]),5)


#Layer3

def similarity_score_layer3(descriptor1, descriptor2):
    # Calculate Euclidean distance between two layer3 descriptors
    desc1_normalized = featurenormalize(descriptor1)
    desc2_normalized = featurenormalize(descriptor2)

    # Calculate the Euclidean Distance
    similarity = euclidean_distance_calculator(desc1_normalized, desc2_normalized)

    return round(similarity,5)

#FC

def similarity_score_fc(descriptor1, descriptor2):
    # Convert to NumPy arrays if they are of type torch.Tensor
    if isinstance(descriptor1, torch.Tensor):
        descriptor1 = descriptor1.numpy()
    if isinstance(descriptor2, torch.Tensor):
        descriptor2 = descriptor2.numpy()

    # Calculate Cosine Similarity for fc descriptors
    return round(cosine_similarity_calculator([descriptor1], [descriptor2]), 5)

def similarity_calculator(index,collection,dataset):
    imagedata1 = collection.find_one({'_id': index})
    similarities = {
        "color_moments": {},
        "hog_descriptor": {},
        "avgpool_descriptor": {},
        "layer3_descriptor": {},
        "fc_descriptor": {}
    }
    
    for cmpidx in tqdm(range(len(dataset))):
        
        imagedata2 = collection.find_one({"_id": cmpidx})
        
        # Corrected the use of 'not' instead of '!'
        color_moments_similarity = similarity_score_color_moments(imagedata1["color_moments"], imagedata2["color_moments"])
        histogram_similarity = similarity_score_hog(imagedata1["hog_descriptor"], imagedata2["hog_descriptor"])
        avgpool_similarity = similarity_score_avgpool(imagedata1["avgpool_descriptor"], imagedata2["avgpool_descriptor"])
        layer3_similarity = similarity_score_layer3(imagedata1["layer3_descriptor"], imagedata2["layer3_descriptor"]) 
        fc_similarity = similarity_score_fc(imagedata1["fc_descriptor"], imagedata2["fc_descriptor"])
        if not np.isnan(color_moments_similarity):
            similarities["color_moments"][cmpidx] = color_moments_similarity
        else: 
            similarities["color_moments"][cmpidx] = 1
        similarities["hog_descriptor"][cmpidx] =  histogram_similarity
        similarities["avgpool_descriptor"][cmpidx] =  avgpool_similarity
        similarities["layer3_descriptor"][cmpidx] = layer3_similarity
        similarities["fc_descriptor"][cmpidx] =  fc_similarity
    
    return similarities