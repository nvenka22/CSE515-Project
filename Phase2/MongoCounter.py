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
import pickle
import json
from MongoDB.MongoDBUtils import *
from multiprocessing import Process
from multiprocessing import set_start_method

import pymongo
from pymongo import MongoClient


mod_path = Path(__file__).parent.parent
caltech101 = Caltech101(str(mod_path) + "/caltech101",download=True)

final_similarity_scores = {}

dbName = "CSE515-MWD-Vaishnavi-ProjectPhase2"
similarity_collection = connect_to_db(dbName,'image_similarities')

transferclient = MongoClient('192.168.0.5',27017)
transferdb = transferclient["CSE515-MWD-Nikhil_V_Ramanan-ProjectPhase2"]
transfercollection = transferdb['image_similarities']

oddcounter = 0
evencounter = 0
transferoddcounter = 0
transferevencounter = 0

for index in tqdm(range(0,8677)):
	similarities = similarity_collection.find_one({'_id': index})
	transfersimilarities = transfercollection.find_one({'_id': index})
	if similarities!=None:
		if index%2 == 0: evencounter+=1
		else: oddcounter+=1
	if transfersimilarities!=None:
		if index%2 == 0:transferevencounter+=1
		else:transferoddcounter+=1

print("Local DB: "+str(evencounter)+" "+str(oddcounter))
print("Transfer DB: "+str(transferevencounter)+" "+str(transferoddcounter))