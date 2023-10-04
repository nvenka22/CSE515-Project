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
    st.image(image=image, caption="ImageID: "+str(idx),channels="BGR", width = 300)
    
def display_feature_vector(vector):
    """
    plt.figure(figsize=(10,5))
    plt.bar(range(len(vector)), vector)
    plt.title('Feature Vector Visualization')
    plt.xlabel('Dimension')
    plt.ylabel('Value')
    plt.show()"""
    st.write(vector)
    
def display_color_moments(color_moments):
    
    """# Display a total of 100 subplots
    
    fig, axs = plt.subplots(10, 10, figsize=(30, 30))
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    
    custom_handles = [Line2D([0], [0], color='w', label='M - (Mean)', markerfacecolor='red', markersize=1),
                      Line2D([0], [0], color='w', label='St - (Std Dev)', markerfacecolor='green', markersize=1),
                      Line2D([0], [0], color='w', label='(Sk - Skewness)', markerfacecolor='blue', markersize=1)]

    for row in range(10):
        for col in range(10):
            moments = color_moments[row, col]
            ax = axs[row, col]

            positions = np.array([1, 2, 3])  # Adjust the positions of the groups

            # Group 1: Mean (Red, Green, Blue)
            ax.bar(positions, moments[:, 0], width=1, color=['red', 'green', 'blue'], alpha=0.5, label='M')

            # Group 2: Std Deviation (Red, Green, Blue)
            ax.bar(positions + 4, moments[:, 1], width=1, color=['red', 'green', 'blue'], alpha=0.5, label='St')

            # Group 3: Skewness (Red, Green, Blue)
            ax.bar(positions + 8, moments[:, 2], width=1, color=['red', 'green', 'blue'], alpha=0.5, label='Sk')

            ax.set_xticks([2, 6, 10])
            ax.set_xticklabels(['M', 'St', 'Sk'])
            ax.set_title(f'Cell ({row+1}, {col+1})')

    # Add legend at top right
    axs[0, -1].legend(handles=custom_handles, loc='upper right', bbox_to_anchor=(1.1, 1.5), frameon=False)

    plt.show()"""

    st.write(color_moments.tolist())
    
    
    
def display_hog(hog_descriptor, cell_size=(30, 10)):
    """# Create a black canvas
    background = np.zeros((100, 300), dtype=np.uint8)

    for row in range(10):
        for col in range(10):
            # Calculate the center of the cell
            center = (col*cell_size[0] + cell_size[0]//2, row*cell_size[1] + cell_size[1]//2)

            # Calculate the endpoint of the gradient vector based on the HOG descriptor
            angle = (hog_descriptor[row*10 + col] * 20) - 90
            length = min(cell_size) // 2
            endpoint = (int(center[0] + length * np.cos(np.radians(angle))),
                        int(center[1] + length * np.sin(np.radians(angle))))

            # Draw a line on the canvas
            cv2.line(background, center, endpoint, 255, 1)

    # Display the visualization using matplotlib
    plt.imshow(background, cmap='gray')
    plt.axis('off')
    plt.show()"""
    st.write(hog_descriptor)
    
    
def display_images(images, indices, similarity_scores, rows, cols):
    
    """k = len(images)

    # Create a grid of subplots
    fig, axes = plt.subplots(rows, cols, figsize=(12, 8))
    fig.subplots_adjust(wspace=0.3, hspace=0.5)

    for i, ax in enumerate(axes.flat):
        if i < k:
            ax.imshow(images[i], cmap='gray')  # Assuming grayscale images
            ax.axis('off')
            ax.set_title(f"Index: {indices[i]}", fontsize=10)
            ax.text(
                0.5, -0.15, f"Similarity Score: {similarity_scores[i]:.5f}",
                transform=ax.transAxes,
                fontsize=10,
                ha='center'
            )
        else:
            ax.axis('off')

    plt.show()"""

def show_ksimilar(k_similar,collection):
    images = []
    indices = []
    similarity_scores = []
    count = len(k_similar.keys())
    rows = int(count/5)
    if (rows*5)<count: rows+=1
    cols = 5
    st.write(k_similar)
    for index in k_similar.keys():
        document = collection.find_one({'_id': int(index)})
        images.append(cv2.cvtColor(np.array(document['image'], dtype=np.uint8), cv2.COLOR_BGR2RGB))
        indices.append(int(index))
        similarity_scores.append(k_similar[str(index)])
        
    #display_images(images, indices, similarity_scores, rows, cols)    