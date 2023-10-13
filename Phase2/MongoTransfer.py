from numba import njit 
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
set_start_method("fork")

def transfer(transfercollection,idx,similarities):
	transfercollection.update_one({'_id':idx},{'$set':similarities},upsert = True)


def transfer_similarities(similarity_collection,transfercollection):
	for idx in tqdm(range(0,8677)):
		similarities = similarity_collection.find_one({'_id': idx})
		if similarities!=None:
			transfer_sim_scores = transfercollection.find_one({'_id': idx})
			if transfer_sim_scores==None:
				print("Updating entry for ImageID: "+str(idx))
				#print(similarities,transfer_sim_scores)
				transfer(transfercollection,idx,similarities)
			else:
				print('Present in DB for'+str(idx))

if __name__=="__main__": 
    mod_path = Path(__file__).parent.parent
    caltech101 = Caltech101(str(mod_path) + "/caltech101",download=True)

    final_similarity_scores = {}

    dbName = "CSE515-MWD-Vaishnavi-ProjectPhase2"
    similarity_collection = connect_to_db(dbName,'image_similarities')

    transferclient = MongoClient('192.168.0.5',27017)
    transferdb = transferclient["CSE515-MWD-Nikhil_V_Ramanan-ProjectPhase2"]
    transfercollection = transferdb['image_similarities']

    transfer_similarities(transfercollection,similarity_collection)
