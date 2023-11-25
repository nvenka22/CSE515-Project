#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 04:53:14 2023

@author: nikhilvr
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 18:56:48 2023

@author: nikhilvr
"""

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

from FeatureDescriptors.FeatureDescriptorUtils import *
from FeatureDescriptors.SimilarityScoreUtils import *
from MongoDB.MongoDBUtils import *



def push_resnet_to_mongo(similarity_collection):
    for index in range(1,8677,2):
        fileName = "/Users/nikhilvr/Documents/MS/Course-Work/Multimedia/CSE515-Project/Phase2/odd_results/image_"+str(index)+".json"
        try:
            with open(fileName) as file:
                sim_score = json.load(file)
                similarity_collection.update_one({'_id':index},{'$set':sim_score},upsert = True)
                print('DB Updated for image '+str(index))
        except FileNotFoundError as e:
            print('File open error for image '+str(index))
            
if __name__=="__main__": 
    mod_path = Path(__file__).parent.parent
    caltech101 = Caltech101(str(mod_path) + "/caltech101",download=True)


    dbName = "CSE515-MWD-ProjectPhase2-Final"
    odd_feature_collection = connect_to_db(dbName,'image_features_odd')
    feature_collection = connect_to_db(dbName,'image_features')
    similarity_collection = connect_to_db(dbName,'image_similarities')
    
    push_resnet_to_mongo(similarity_collection)



    # processes = [Process(target = calc_sim_score_softmax,args = (idx,idx+500,feature_collection,similarity_collection)) for idx in range(0,3000,250)]
    
    
    # print("Threads Running")
    # for p in processes:

    #     p.start()


    # for p in processes:

    #     p.join()
        
        
    # print("Threads Done")
