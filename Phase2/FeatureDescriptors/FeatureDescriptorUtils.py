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

from FeatureDescriptors.SimilarityScoreUtils import *
from Utilities.DisplayUtils import *
import streamlit as st
from pathlib import Path

def color_moments_calculator(image):
    
    if image.shape[2] == 2:
        # If the image has 2 channels (e.g., grayscale), convert it to 3 channels (RGB)
        print("2 channels")
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Resize image to 300x100
    image = cv2.resize(image, (300, 100))
    
    # Partition the image into 10x10 grid
    rows = np.vsplit(image, 10)
    cells = [np.hsplit(row, 10) for row in rows]
    cells = np.array(cells).reshape(-1, 100, 3)

    # Initialize feature descriptor array
    color_moments = np.zeros((10, 10, 3, 3))

    # Compute color moments for each cell
    for row in range(10):
        for col in range(10):
            for ch in range(3):
                channel = cells[row*10 + col][:, ch]
                mean = np.mean(channel)
                std_dev = np.std(channel)
                skewness = skew(channel)
                color_moments[row, col, ch] = [mean, std_dev, skewness]

    return color_moments

    
#HOG Calculations

def hog_calculator(image):
    # Convert the image to grayscale
    image_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize image to 300x100
    image_resized = cv2.resize(image_grayscale, (300, 100))

    # Define the cell size
    cell_size = (30, 10)

    # Initialize the HOG descriptor array
    hog = []

    # Compute HOG descriptor for each cell
    for row in range(10):
        for col in range(10):
            # Extract the cell
            cell = image_resized[row*cell_size[1]:(row+1)*cell_size[1], col*cell_size[0]:(col+1)*cell_size[0]]

            # Calculate gradients using Sobel
            gradient_x = cv2.Sobel(cell, cv2.CV_64F, 1, 0, ksize=5)
            gradient_y = cv2.Sobel(cell, cv2.CV_64F, 0, 1, ksize=5)

            # Calculate gradient magnitude and orientation
            magnitude, angle = cv2.cartToPolar(gradient_x, gradient_y, angleInDegrees=True)

            # Calculate the histogram for this cell
            hist, _ = np.histogram(angle, bins=9, range=(0, 180), weights=magnitude)

            # Append the histogram values to the HOG descriptor
            hog.extend(hist)

    return np.array(hog)

#ResNet Calculator
resnet_model = resnet50(pretrained=True)
def avgpool_calculator(image):
    
    # Define a hook to get the output of the "avgpool" layer
    output_hook = None
    def hook_fn(module, input, output):
        nonlocal output_hook
        output_hook = output

    # Attach the hook to the "avgpool" layer
    resnet_model.avgpool.register_forward_hook(hook_fn)

    # Define a function to preprocess the image
    def preprocess_image(image):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=dataset_mean_values, std=dataset_std_dev_values),
        ])
        return transform(image).unsqueeze(0)

    # Define a function to get the 2048-dimensional vector
    def get_2048_dimensional_vector(image):
        input_tensor = preprocess_image(image)
        resnet_model.eval()
        with torch.no_grad():
            _ = resnet_model(input_tensor)
        return output_hook.squeeze().numpy()

    # Get the 2048-dimensional vector
    vector_2048 = get_2048_dimensional_vector(image)

    # Reduce dimensions to 1024 by averaging adjacent entries
    vector_1024 = [(v1 + v2) / 2 for v1, v2 in zip(vector_2048[::2], vector_2048[1::2])]

    return vector_1024  

def layer3_calculator(image):
    
    # Define a hook to get the output of the "layer3" layer
    output_hook = None
    def hook_fn(module, input, output):
        nonlocal output_hook
        output_hook = output

    # Attach the hook to the "layer3" layer
    resnet_model.layer3.register_forward_hook(hook_fn)

    # Define a function to preprocess the image
    def preprocess_image(image):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=dataset_mean_values, std=dataset_std_dev_values),
        ])
        return transform(image).unsqueeze(0)

    # Define a function to get the 1024x14x14 tensor
    def get_1024x14x14_tensor(image):
        input_tensor = preprocess_image(image)
        resnet_model.eval()
        with torch.no_grad():
            _ = resnet_model(input_tensor)
        return output_hook.squeeze()

    # Get the 1024x14x14 tensor
    tensor_1024x14x14 = get_1024x14x14_tensor(image)

    # Convert the tensor to a 1024-dimensional vector by averaging each 14x14 slice
    vector_1024 = torch.mean(tensor_1024x14x14.view(1024, -1), dim=1).numpy()

    return vector_1024

#ResNet FC 1000

def fc_calculator(image):
    
    # Define a hook to get the output of the "fc" layer
    output_hook = None
    def hook_fn(module, input, output):
        nonlocal output_hook
        output_hook = output

    # Attach the hook to the "fc" layer
    resnet_model.fc.register_forward_hook(hook_fn)

    # Define a function to preprocess the image
    def preprocess_image(image):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=dataset_mean_values, std=dataset_std_dev_values),
        ])
        return transform(image).unsqueeze(0)

    # Define a function to get the 1000-dimensional tensor
    def get_1000_dimensional_tensor(image):
        input_tensor = preprocess_image(image)
        resnet_model.eval()
        with torch.no_grad():
            _ = resnet_model(input_tensor)
        return output_hook.squeeze()

    # Get the 1000-dimensional tensor
    tensor_1000 = get_1000_dimensional_tensor(image)

    return tensor_1000

def descriptor_calculator(image, idx,caltech101):
    color_moments = color_moments_calculator(image)
    hog_descriptor = hog_calculator(image)
    avgpool_descriptor = avgpool_calculator(image)
    layer3_descriptor = layer3_calculator(image)
    fc_descriptor = fc_calculator(image)
    return {
        '_id': idx,
        'label': caltech101.__getitem__(index=idx)[1],
        'image': image.tolist() if isinstance(image, np.ndarray) else image,  # Convert the image to a list for storage
        'color_moments': color_moments.tolist() if isinstance(color_moments, np.ndarray) else color_moments,
        'hog_descriptor': hog_descriptor.tolist() if isinstance(hog_descriptor, np.ndarray) else hog_descriptor,
        'avgpool_descriptor': avgpool_descriptor.tolist() if isinstance(avgpool_descriptor, np.ndarray) else avgpool_descriptor,
        'layer3_descriptor': layer3_descriptor.tolist() if isinstance(layer3_descriptor, np.ndarray) else layer3_descriptor,
        'fc_descriptor': fc_descriptor.tolist()
    }
    
def queryksimilar(index,k,odd_feature_collection,feature_collection,similarity_collection,dataset):
    
    similarity_scores = similarity_calculator(index,odd_feature_collection,feature_collection,similarity_collection,dataset)
    color_moments_similar = dict(sorted(similarity_scores["color_moments"].items(), key = lambda x: x[1])[:k])
    hog_similar = dict(sorted(similarity_scores["hog_descriptor"].items(), key = lambda x: x[1])[-k:])
    avgpool_similar = dict(sorted(similarity_scores["avgpool_descriptor"].items(), key = lambda x: x[1])[-k:])
    layer3_similar = dict(sorted(similarity_scores["layer3_descriptor"].items(), key = lambda x: x[1])[:k])
    fc_similar = dict(sorted(similarity_scores["fc_descriptor"].items(), key = lambda x: x[1])[-k:])
    imagedata = feature_collection.find_one({'_id': index})
    
    image = np.array(imagedata['image'], dtype=np.uint8)
    st.markdown("Query Image")
    display_image_centered(np.array(image),index)
    st.markdown("Query Image Color Moments")
    display_color_moments(np.array(imagedata['color_moments']))
    st.markdown("Query Image HOG Descriptor")
    display_hog(imagedata['hog_descriptor'])
    st.markdown("Query Image Avgpool Descriptor")
    display_feature_vector(imagedata['avgpool_descriptor'])
    st.markdown("Query Image Layer3 Descriptor")
    display_feature_vector(imagedata['layer3_descriptor'])
    st.markdown("Query Image FC Descriptor")
    display_feature_vector(imagedata['fc_descriptor'])
    st.markdown('Color Moments - Euclidean Distance')
    show_ksimilar(color_moments_similar,feature_collection)
    st.markdown('Histograms of Oriented Gradients(HOG) - Cosine Similarity')
    show_ksimilar(hog_similar,feature_collection)
    st.markdown('ResNet-AvgPool-1024 - Cosine Similarity')
    show_ksimilar(avgpool_similar,feature_collection)
    st.markdown('ResNet-Layer3-1024 - Euclidean Distance')
    show_ksimilar(layer3_similar,feature_collection)
    st.markdown('ResNet-FC-1000 - Cosine Similarity')
    show_ksimilar(fc_similar,feature_collection)
    return similarity_scores

dataset_size = 8677
dataset_mean_values = [0.5021372281891864, 0.5287581550675707, 0.5458470856851454]
dataset_std_dev_values = [0.24773670511666424, 0.24607509728422117, 0.24912913964278197]