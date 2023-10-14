import os
import cv2
from torchvision.models import resnet50
from torchvision.datasets import Caltech101
from torch.autograd import Variable
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
from sklearn.decomposition import TruncatedSVD, NMF, LatentDirichletAllocation
from sklearn.cluster import KMeans
from sklearn import preprocessing as p
import pickle
from tensorly.decomposition import parafac

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

def fc_calculator_2(image):
    
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
            transforms.Resize((256)),
            transforms.CenterCrop(224),
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

    activation = torch.nn.Softmax()
    output_tensor = activation(tensor_1000)

    return output_tensor

def resnet_features(image):
    # Remove the final classification layer
    print("in")
    resnet50 = torch.nn.Sequential(*list(resnet_model.children())[:-13])
    # Set the model to evaluation mode
    resnet50.eval()

    # Define the preprocessing transformations
    def preprocess_img(image):
        preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return preprocess(image).unsqueeze(0)

    tensor_1000 = preprocess_img(image)
    print(type(tensor_1000))
    print("Shape of output tensor: ", tensor_1000.shape)

    with torch.no_grad():
        features = resnet50(tensor_1000)
    
    print("Features shape: ", features.shape)

    return features.squeeze().numpy()

    
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
    

def queryksimilar(index,k,odd_feature_collection,feature_collection,similarity_collection,dataset,feature_space = None):
    
    similarity_scores = similarity_calculator(index,odd_feature_collection,feature_collection,similarity_collection,dataset)
    color_moments_similar = dict(sorted(similarity_scores["color_moments"].items(), key = lambda x: x[1])[:k])
    hog_similar = dict(sorted(similarity_scores["hog_descriptor"].items(), key = lambda x: x[1],reverse = True)[:k])
    avgpool_similar = dict(sorted(similarity_scores["avgpool_descriptor"].items(), key = lambda x: x[1],reverse=True)[:k])
    layer3_similar = dict(sorted(similarity_scores["layer3_descriptor"].items(), key = lambda x: x[1])[:k])
    fc_similar = dict(sorted(similarity_scores["fc_descriptor"].items(), key = lambda x: x[1],reverse=True)[:k])
    
    if index%2==0:
        imagedata = feature_collection.find_one({'_id': index})
    else:
        imagedata = odd_feature_collection.find_one({'_id': index})
    
    image = np.array(imagedata['image'], dtype=np.uint8)

    if feature_space == None:
        st.markdown("Query Image")
        display_image_centered(np.array(image),str(index))
        display_color_moments(np.array(imagedata['color_moments']))
        display_hog(imagedata['hog_descriptor'])
        display_feature_vector(imagedata['avgpool_descriptor'],"Query Image Avgpool Descriptor")
        display_feature_vector(imagedata['layer3_descriptor'],"Query Image Layer3 Descriptor")
        display_feature_vector(imagedata['fc_descriptor'],"Query Image FC Descriptor")
        st.markdown('Color Moments - Euclidean Distance')
        show_ksimilar(color_moments_similar,feature_collection,"Distance Score: ")
        st.markdown('Histograms of Oriented Gradients(HOG) - Cosine Similarity')
        show_ksimilar(hog_similar,feature_collection,"Similarity Score:")
        st.markdown('ResNet-AvgPool-1024 - Cosine Similarity')
        show_ksimilar(avgpool_similar,feature_collection,"Similarity Score:")
        st.markdown('ResNet-Layer3-1024 - Euclidean Distance')
        show_ksimilar(layer3_similar,feature_collection, "Distance Score: ")
        st.markdown('ResNet-FC-1000 - Cosine Similarity')
        show_ksimilar(fc_similar,feature_collection,"Similarity Score:")

    elif feature_space == "Color Moments":
        st.markdown("Query Image")
        display_image_centered(np.array(image),str(index))
        display_color_moments(np.array(imagedata['color_moments']))
        st.markdown('Color Moments - Euclidean Distance')
        show_ksimilar(color_moments_similar,feature_collection,"Distance Score: ")

    elif feature_space == "Histograms of Oriented Gradients(HOG)":
        st.markdown("Query Image")
        display_image_centered(np.array(image),str(index))
        display_hog(imagedata['hog_descriptor'])
        st.markdown('Histograms of Oriented Gradients(HOG) - Cosine Similarity')
        show_ksimilar(hog_similar,feature_collection,"Similarity Score:")

    elif feature_space == "ResNet-AvgPool-1024":
        st.markdown("Query Image")
        display_image_centered(np.array(image),str(index))
        display_feature_vector(imagedata['avgpool_descriptor'],"Query Image Avgpool Descriptor")
        st.markdown('ResNet-AvgPool-1024 - Cosine Similarity')
        show_ksimilar(avgpool_similar,feature_collection,"Similarity Score:")

    elif feature_space == "ResNet-Layer3-1024":
        st.markdown("Query Image")
        display_image_centered(np.array(image),str(index))
        display_feature_vector(imagedata['layer3_descriptor'],"Query Image Layer3 Descriptor")
        st.markdown('ResNet-Layer3-1024 - Euclidean Distance')
        show_ksimilar(layer3_similar,feature_collection, "Distance Score: ")

    elif feature_space == "ResNet-FC-1000":
        st.markdown("Query Image")
        display_image_centered(np.array(image),str(index))
        display_feature_vector(imagedata['fc_descriptor'],"Query Image FC Descriptor")
        st.markdown('ResNet-FC-1000 - Cosine Similarity')
        show_ksimilar(fc_similar,feature_collection,"Similarity Score:")


    return similarity_scores

def queryksimilar_newimg(image, k,odd_feature_collection,feature_collection,similarity_collection,dataset,feature_space = None):

    color_moments = color_moments_calculator(image)
    hog_descriptor = hog_calculator(image)
    avgpool_descriptor = avgpool_calculator(image)
    layer3_descriptor = layer3_calculator(image)
    fc_descriptor = fc_calculator(image)

    imagedata = {
        'image': image.tolist() if isinstance(image, np.ndarray) else image,  # Convert the image to a list for storage
        'color_moments': color_moments.tolist() if isinstance(color_moments, np.ndarray) else color_moments,
        'hog_descriptor': hog_descriptor.tolist() if isinstance(hog_descriptor, np.ndarray) else hog_descriptor,
        'avgpool_descriptor': avgpool_descriptor.tolist() if isinstance(avgpool_descriptor, np.ndarray) else avgpool_descriptor,
        'layer3_descriptor': layer3_descriptor.tolist() if isinstance(layer3_descriptor, np.ndarray) else layer3_descriptor,
        'fc_descriptor': fc_descriptor.tolist()
    }
    similarity_scores = similarity_calculator_newimg(imagedata,odd_feature_collection,feature_collection,similarity_collection,dataset)
    color_moments_similar = dict(sorted(similarity_scores["color_moments"].items(), key = lambda x: x[1])[:k])
    hog_similar = dict(sorted(similarity_scores["hog_descriptor"].items(), key = lambda x: x[1])[-k:])
    avgpool_similar = dict(sorted(similarity_scores["avgpool_descriptor"].items(), key = lambda x: x[1])[-k:])
    layer3_similar = dict(sorted(similarity_scores["layer3_descriptor"].items(), key = lambda x: x[1])[:k])
    fc_similar = dict(sorted(similarity_scores["fc_descriptor"].items(), key = lambda x: x[1])[-k:])

    if feature_space == None:
        display_color_moments(np.array(imagedata['color_moments']))
        display_hog(imagedata['hog_descriptor'])
        display_feature_vector(imagedata['avgpool_descriptor'],"Query Image Avgpool Descriptor")
        display_feature_vector(imagedata['layer3_descriptor'],"Query Image Layer3 Descriptor")
        display_feature_vector(imagedata['fc_descriptor'],"Query Image FC Descriptor")
        st.markdown('Color Moments - Euclidean Distance')
        show_ksimilar(color_moments_similar,feature_collection,"Distance Score: ")
        st.markdown('Histograms of Oriented Gradients(HOG) - Cosine Similarity')
        show_ksimilar(hog_similar,feature_collection,"Similarity Score:")
        st.markdown('ResNet-AvgPool-1024 - Cosine Similarity')
        show_ksimilar(avgpool_similar,feature_collection,"Similarity Score:")
        st.markdown('ResNet-Layer3-1024 - Euclidean Distance')
        show_ksimilar(layer3_similar,feature_collection, "Distance Score: ")
        st.markdown('ResNet-FC-1000 - Cosine Similarity')
        show_ksimilar(fc_similar,feature_collection,"Similarity Score:")

    elif feature_space == "Color Moments":
        display_color_moments(np.array(imagedata['color_moments']))
        st.markdown('Color Moments - Euclidean Distance')
        show_ksimilar(color_moments_similar,feature_collection,"Distance Score: ")

    elif feature_space == "Histograms of Oriented Gradients(HOG)":
        display_hog(imagedata['hog_descriptor'])
        st.markdown('Histograms of Oriented Gradients(HOG) - Cosine Similarity')
        show_ksimilar(hog_similar,feature_collection,"Similarity Score:")

    elif feature_space == "ResNet-AvgPool-1024":
        display_feature_vector(imagedata['avgpool_descriptor'],"Query Image Avgpool Descriptor")
        st.markdown('ResNet-AvgPool-1024 - Cosine Similarity')
        show_ksimilar(avgpool_similar,feature_collection,"Similarity Score:")

    elif feature_space == "ResNet-Layer3-1024":
        display_feature_vector(imagedata['layer3_descriptor'],"Query Image Layer3 Descriptor")
        st.markdown('ResNet-Layer3-1024 - Euclidean Distance')
        show_ksimilar(layer3_similar,feature_collection, "Distance Score: ")

    elif feature_space == "ResNet-FC-1000":
        display_feature_vector(imagedata['fc_descriptor'],"Query Image FC Descriptor")
        st.markdown('ResNet-FC-1000 - Cosine Similarity')
        show_ksimilar(fc_similar,feature_collection,"Similarity Score:")


    return similarity_scores

def manual_svd(X):
    # Convert X to a NumPy array
    X = np.array(X)

    # Compute covariance matrix
    cov = np.dot(X.T, X)

    # Eigenvalue decomposition of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(cov)

    # Replace NaN values with 0
    eigenvalues = np.nan_to_num(eigenvalues)
    eigenvectors = np.nan_to_num(eigenvectors)

    # Sort eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Compute the singular values and their reciprocals
    singular_values = np.sqrt(eigenvalues)
    reciprocals_singular_values = np.where(singular_values != 0, 1/singular_values, 0)

    # Replace NaN values with 0
    singular_values = np.nan_to_num(singular_values)
    reciprocals_singular_values = np.nan_to_num(reciprocals_singular_values)

    # Compute U and V matrices
    U = np.dot(X, eigenvectors)
    V = np.dot(eigenvectors, np.diag(singular_values))

    # Replace NaN values with 0
    U = np.nan_to_num(U)
    V = np.nan_to_num(V)

    return U, V, reciprocals_singular_values

def reduce_dimensionality(feature_model, k, technique):
    if technique == 'SVD':
        U, V, sigma_inv = manual_svd(feature_model)

        # Take the first k columns of U and V
        latent_semantics = np.dot(U[:, :k], V[:k, :])

        return latent_semantics
    elif technique == 'NNMF':
        reducer = NMF(n_components=k)
    elif technique == 'LDA':
        reducer = LatentDirichletAllocation(n_components=k)
    elif technique == 'k-Means':
        reducer = KMeans(n_clusters=k)
    else:
        raise ValueError("Invalid dimensionality reduction technique")

    latent_semantics = reducer.fit_transform(feature_model)
    return latent_semantics

def get_top_k_latent_semantics(latent_semantics, k):
    top_k_indices = np.argsort(latent_semantics.sum(axis=0))[::-1][:k]
    return top_k_indices
    
def list_imageID_weight_pairs(top_k_indices, latent_semantics):
    imageID_weight_pairs = list(zip(top_k_indices, latent_semantics[:, top_k_indices]))
    imageID_weight_pairs.sort(key=lambda x: np.mean(x[1]), reverse=True)
    return imageID_weight_pairs

def ls1(feature_model,k,dimred,feature_collection):

    mod_path = Path(__file__).parent.parent
    output_file = str(mod_path)+"/LatentSemantics/"

    feature_descriptors = []

    if feature_model == "Color Moments":
        obj = feature_collection.find({},{"color_moments":1})
        output_file += "latent_semantics_1_color_moments_"+dimred+"_"+str(k)+"_output.pkl"
        for doc in obj:
            fetchedarray = doc['color_moments']
            cmarray = []

            for row in range(0,10):
                for col in range(0,10):
                    for channel in fetchedarray[row][col]:
                        cmarray.append(channel[0])
                        cmarray.append(channel[1])
                        cmarray.append(channel[2])

            cmarray = [0 if pd.isna(x) else x for x in cmarray]
            feature_descriptors.append(cmarray)
        feature_descriptors = np.array(feature_descriptors)

    elif feature_model == "Histograms of Oriented Gradients(HOG)":
        obj = feature_collection.find({},{"hog_descriptor":1})
        output_file += "latent_semantics_1_hog_descriptor_"+dimred+"_"+str(k)+"_output.pkl"
        for doc in obj:
            hogarray = doc['hog_descriptor']
            hogarray = [0 if pd.isna(x) else x for x in hogarray]
            feature_descriptors.append(hogarray)
        feature_descriptors = np.array(feature_descriptors)

    elif feature_model == "ResNet-AvgPool-1024":
        obj = feature_collection.find({},{"avgpool_descriptor":1})
        output_file += "latent_semantics_1_avgpool_descriptor_"+dimred+"_"+str(k)+"_output.pkl"
        for doc in obj:
            avgpoolarray = doc['avgpool_descriptor']
            avgpoolarray = [0 if pd.isna(x) else x for x in avgpoolarray]
            feature_descriptors.append(avgpoolarray)
        feature_descriptors = np.array(feature_descriptors)

    elif feature_model == "ResNet-Layer3-1024":
        obj = feature_collection.find({},{"layer3_descriptor":1})
        output_file += "latent_semantics_1_layer3_descriptor_"+dimred+"_"+str(k)+"_output.pkl"
        for doc in obj:
            layer3array = doc['layer3_descriptor']
            layer3array = [0 if pd.isna(x) else x for x in layer3array]
            feature_descriptors.append(layer3array)
        feature_descriptors = np.array(feature_descriptors)

    elif feature_model == "ResNet-FC-1000":
        obj = feature_collection.find({},{"fc_descriptor":1})
        output_file += "latent_semantics_1_fc_descriptor_"+dimred+"_"+str(k)+"_output.pkl"
        for doc in obj:
            fcarray = doc['fc_descriptor']
            fcarray = [0 if pd.isna(x) else x for x in fcarray]
            feature_descriptors.append(fcarray)
        feature_descriptors = np.array(feature_descriptors)

    min_max_scaler = p.MinMaxScaler() 
    feature_descriptors = min_max_scaler.fit_transform(feature_descriptors)
    latent_semantics = reduce_dimensionality(feature_descriptors, k, dimred)
    top_k_indices = get_top_k_latent_semantics(latent_semantics, k)

    pickle.dump((top_k_indices, latent_semantics), open(output_file, 'wb+'))

    imageID_weight_pairs = list_imageID_weight_pairs(top_k_indices, latent_semantics)

    with st.container():
        rank = 1
        for imageID, weight in imageID_weight_pairs:
            st.markdown("Rank: "+str(rank))
            with st.expander("Image ID: "+str(imageID)+" weights:"):
                st.write(weight.tolist())
            rank+=1

def get_class_name(label):
    data = {
	0: "Faces", 1: "Faces_easy", 2: "Leopards", 3: "Motorbikes", 4: "accordion", 5: "airplanes", 6: "anchor", 7: "ant", 8: "barrel", 9: "bass",
	10: "beaver", 11: "binocular", 12: "bonsai", 13: "brain", 14: "brontosaurus", 15: "buddha", 16: "butterfly", 17: "camera", 18: "cannon", 19: "car_side",
	20: "ceiling_fan", 21: "cellphone", 22: "chair", 23: "chandelier", 24: "cougar_body", 25: "cougar_face", 26: "crab", 27: "crayfish", 28: "crocodile", 29: "crocodile_head",
	30: "cup", 31: "dalmatian", 32: "dollar_bill", 33: "dolphin", 34: "dragonfly", 35: "electric_guitar", 36: "elephant", 37: "emu", 38: "euphonium", 39: "ewer",
	40: "ferry", 41: "flamingo", 42: "flamingo_head", 43: "garfield", 44: "gerenuk", 45: "gramophone", 46: "grand_piano", 47: "hawksbill", 48: "headphone", 49: "hedgehog",
	50: "helicopter", 51: "ibis", 52: "inline_skate", 53: "joshua_tree", 54: "kangaroo", 55: "ketch", 56: "lamp", 57: "laptop", 58: "llama", 59: "lobster",
	60: "lotus", 61: "mandolin", 62: "mayfly", 63: "menorah", 64: "metronome", 65: "minaret", 66: "nautilus", 67: "octopus", 68: "okapi", 69: "pagoda",
	70: "panda", 71: "pigeon", 72: "pizza", 73: "platypus", 74: "pyramid", 75: "revolver", 76: "rhino", 77: "rooster", 78: "saxophone", 79: "schooner",
	80: "scissors", 81: "scorpion", 82: "seahorse", 83: "snoopy", 84: "soccer_ball", 85: "stapler", 86: "starfish", 87: "stegosaurus", 88: "stop_sign", 89: "strawberry",
	90: "sunflower", 91: "tick", 92: "trilobite", 93: "umbrella", 94: "watch", 95: "water_lilly", 96: "wheelchair", 97: "wild_cat", 98: "windsor_chair", 99: "wrench", 100: "yin_yang"}

    return data[label]

def perform_cp_decomposition(feature_tensor, k):
    weights, factors = parafac(feature_tensor, rank=k)
    return weights, factors

def ls2(feature_model,k,feature_collection):

    mod_path = Path(__file__).parent.parent
    output_file = str(mod_path)+"/LatentSemantics/"

    label_factors = []

    if feature_model == "Color Moments":

        output_file += "latent_semantics_2_color_moments_"+str(k)+"_output.pkl"

        images = []
        feature_descriptors = []
        labels = []

        for index in tqdm(range(0,dataset_size,2)):

            doc = feature_collection.find_one({'_id': index})

            fetchedarray = doc['color_moments']
            
            cmarray = []

            for row in range(0,10):
                for col in range(0,10):
                    for channel in fetchedarray[row][col]:
                        cmarray.append(channel[0])
                        cmarray.append(channel[1])
                        cmarray.append(channel[2])

            cmarray = [0 if pd.isna(x) else x for x in cmarray]

            image = np.array(doc['image'], dtype=np.uint8)
            resized_image = cv2.resize(np.array(image), (300, 100))
            image_array = np.array(resized_image)
            #print(image_array.shape)
            cmarray = np.array(cmarray)
            label = doc['label']

            images.append(image_array)
            feature_descriptors.append(cmarray)
            labels.append(label)

        # Convert lists to numpy arrays
        images_array = np.array(images)
        images_array = images_array.reshape(images_array.shape[0], -1)

        feature_descriptors_array = np.array(feature_descriptors)
        labels_array = np.array(labels).reshape(-1,1)

        print(images_array.shape)
        print(feature_descriptors_array.shape)
        print(labels_array.shape)

        # Stack the arrays to construct the three-modal tensor
        cp_tensor = np.concatenate([images_array, feature_descriptors_array, labels_array], axis=1)
        cp_tensor = cp_tensor.transpose(0, 2, 1).reshape(cp_tensor.shape[0], 900, 100)

        print(cp_tensor.shape)

        weights,factors = perform_cp_decomposition(cp_tensor, k)

    """elif feature_model == "Histograms of Oriented Gradients(HOG)":
            
                    output_file += "latent_semantics_2_hog_descriptor_"+str(k)+"_output.pkl"
            
                    for index in range(0,dataset_size,2):
            
                        doc = feature_collection.find_one({'_id': index})
            
                        hogarray = doc['hog_descriptor']
            
                        hogarray = [0 if pd.isna(x) else x for x in hogarray]
                        
                        cp_tensor = np.hstack((doc['image'], hogarray, np.array([doc['label']])))
            
                        weights,factors = perform_cp_decomposition(cp_tensor, k)
            
                        label_factors.append(factors[2])
            
                elif feature_model == "ResNet-AvgPool-1024":
            
                    output_file += "latent_semantics_2_avgpool_descriptor_"+str(k)+"_output.pkl"
            
                    for index in range(0,dataset_size,2):
            
                        doc = feature_collection.find_one({'_id': index})
            
                        avgpoolarray = doc['avgpool_descriptor']
            
                        avgpoolarray = [0 if pd.isna(x) else x for x in avgpoolarray]
                        
                        cp_tensor = np.hstack((doc['image'], avgpoolarray, np.array([doc['label']])))
            
                        weights,factors = perform_cp_decomposition(cp_tensor, k)
            
                        label_factors.append(factors[2])
            
            
                elif feature_model == "ResNet-Layer3-1024":
            
                    output_file += "latent_semantics_2_layer3_descriptor_"+str(k)+"_output.pkl"
            
                    for index in range(0,dataset_size,2):
            
                        doc = feature_collection.find_one({'_id': index})
            
                        layer3array = doc['layer3_descriptor']
            
                        layer3array = [0 if pd.isna(x) else x for x in layer3array]
                        
                        cp_tensor = np.hstack((doc['image'], layer3array, np.array([doc['label']])))
            
                        weights,factors = perform_cp_decomposition(cp_tensor, k)
            
                        label_factors.append(factors[2])
            
            
                elif feature_model == "ResNet-FC-1000":
            
                    output_file += "latent_semantics_2_fc_descriptor_"+str(k)+"_output.pkl"
            
                    for index in range(0,dataset_size,2):
            
                        doc = feature_collection.find_one({'_id': index})
            
                        fcarray = doc['fc_descriptor']
            
                        fcarray = [0 if pd.isna(x) else x for x in fcarray]
                        
                        cp_tensor = np.hstack((doc['image'], fcarray, np.array([doc['label']])))
            
                        weights,factors = perform_cp_decomposition(cp_tensor, k)
            
                        label_factors.append(factors[2])
            
                label_factors = []
                
                for i, factor in enumerate(factors):
                    weights = factor[:, 0]  # Extract the weights
                    latent_semantic = [(label, weight) for label, weight in zip(labels_array.flatten(), weights)]
                    latent_semantic.sort(key=lambda x: x[1], reverse=True)
                    label_factors.append(latent_semantic)"""

    label_factors = np.array(label_factors)

    top_k_indices = get_top_k_latent_semantics(label_factors, k)

    print(top_k_indices)

    pickle.dump((top_k_indices, label_factors), open(output_file, 'wb+'))

    imageID_weight_pairs = list_imageID_weight_pairs(top_k_indices, label_factors)

    with st.container():
        rank = 1
        for imageID, weight in imageID_weight_pairs:
            st.markdown("Rank: "+str(rank))
            with st.expander("Image ID: "+str(imageID)+" weights:"):
                st.write(weight.tolist())
            rank+=1

def ls4(feature_model,k,dimred,similarity_collection):

    mod_path = Path(__file__).parent.parent
    output_file = str(mod_path)+"/LatentSemantics/"

    similarity_matrix = [[0 for col in range(4339)] for row in range(4339)]

    if feature_model == "Color Moments":
        output_file += "latent_semantics_4_color_moments_"+dimred+"_"+str(k)+"_output.pkl"
        for idx in tqdm(range(0,dataset_size,2)):
            scores = similarity_collection.find_one({'_id': idx})
            for cmpidx in range(0,dataset_size,2):
                similarity_matrix[int(idx/2)][int(cmpidx/2)] = 1 - scores['color_moments'][str(cmpidx)]


    elif feature_model == "Histograms of Oriented Gradients(HOG)":
        output_file += "latent_semantics_4_hog_descriptor_"+dimred+"_"+str(k)+"_output.pkl"
        for idx in tqdm(range(0,dataset_size,2)):
            scores = similarity_collection.find_one({'_id': idx})
            for cmpidx in range(0,dataset_size,2):
                similarity_matrix[int(idx/2)][int(cmpidx/2)] = scores['hog_descriptor'][str(cmpidx)]

    elif feature_model == "ResNet-AvgPool-1024":
        output_file += "latent_semantics_4_avgpool_descriptor_"+dimred+"_"+str(k)+"_output.pkl"
        for idx in tqdm(range(0,dataset_size,2)):
            scores = similarity_collection.find_one({'_id': idx})
            for cmpidx in range(0,dataset_size,2):
                similarity_matrix[int(idx/2)][int(cmpidx/2)] = scores['avgpool_descriptor'][str(cmpidx)]

    elif feature_model == "ResNet-Layer3-1024":
        output_file += "latent_semantics_4_layer3_descriptor_"+dimred+"_"+str(k)+"_output.pkl"
        for idx in tqdm(range(0,dataset_size,2)):
            scores = similarity_collection.find_one({'_id': idx})
            for cmpidx in range(0,dataset_size,2):
                similarity_matrix[int(idx/2)][int(cmpidx/2)] = 1 - scores['layer3_descriptor'][str(cmpidx)]

    elif feature_model == "ResNet-FC-1000":
        output_file += "latent_semantics_4_fc_descriptor_"+dimred+"_"+str(k)+"_output.pkl"
        for idx in tqdm(range(0,dataset_size,2)):
            scores = similarity_collection.find_one({'_id': idx})
            for cmpidx in range(0,dataset_size,2):
                similarity_matrix[int(idx/2)][int(cmpidx/2)] = scores['fc_descriptor'][str(cmpidx)]

    similarity_vector = np.array(similarity_matrix).reshape(-1,1)
    #print(similarity_vector.shape)
    latent_semantics = reduce_dimensionality(similarity_matrix, k, dimred)
    top_k_indices = get_top_k_latent_semantics(latent_semantics, k)

    pickle.dump((top_k_indices, latent_semantics), open(output_file, 'wb+'))

    imageID_weight_pairs = list_imageID_weight_pairs(top_k_indices, latent_semantics)

    with st.container():
        rank = 1
        for imageID, weight in imageID_weight_pairs:
            st.markdown("Rank: "+str(rank))
            with st.expander("Image ID: "+str(imageID)+" weights:"):
                st.write(weight.tolist())
            rank+=1

    return similarity_matrix

dataset_size = 8677
dataset_mean_values = [0.5021372281891864, 0.5287581550675707, 0.5458470856851454]
dataset_std_dev_values = [0.24773670511666424, 0.24607509728422117, 0.24912913964278197]

data = {
	0: "Faces", 1: "Faces_easy", 2: "Leopards", 3: "Motorbikes", 4: "accordion", 5: "airplanes", 6: "anchor", 7: "ant", 8: "barrel", 9: "bass",
	10: "beaver", 11: "binocular", 12: "bonsai", 13: "brain", 14: "brontosaurus", 15: "buddha", 16: "butterfly", 17: "camera", 18: "cannon", 19: "car_side",
	20: "ceiling_fan", 21: "cellphone", 22: "chair", 23: "chandelier", 24: "cougar_body", 25: "cougar_face", 26: "crab", 27: "crayfish", 28: "crocodile", 29: "crocodile_head",
	30: "cup", 31: "dalmatian", 32: "dollar_bill", 33: "dolphin", 34: "dragonfly", 35: "electric_guitar", 36: "elephant", 37: "emu", 38: "euphonium", 39: "ewer",
	40: "ferry", 41: "flamingo", 42: "flamingo_head", 43: "garfield", 44: "gerenuk", 45: "gramophone", 46: "grand_piano", 47: "hawksbill", 48: "headphone", 49: "hedgehog",
	50: "helicopter", 51: "ibis", 52: "inline_skate", 53: "joshua_tree", 54: "kangaroo", 55: "ketch", 56: "lamp", 57: "laptop", 58: "llama", 59: "lobster",
	60: "lotus", 61: "mandolin", 62: "mayfly", 63: "menorah", 64: "metronome", 65: "minaret", 66: "nautilus", 67: "octopus", 68: "okapi", 69: "pagoda",
	70: "panda", 71: "pigeon", 72: "pizza", 73: "platypus", 74: "pyramid", 75: "revolver", 76: "rhino", 77: "rooster", 78: "saxophone", 79: "schooner",
	80: "scissors", 81: "scorpion", 82: "seahorse", 83: "snoopy", 84: "soccer_ball", 85: "stapler", 86: "starfish", 87: "stegosaurus", 88: "stop_sign", 89: "strawberry",
	90: "sunflower", 91: "tick", 92: "trilobite", 93: "umbrella", 94: "watch", 95: "water_lilly", 96: "wheelchair", 97: "wild_cat", 98: "windsor_chair", 99: "wrench", 100: "yin_yang"}