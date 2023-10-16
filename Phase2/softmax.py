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
set_start_method("fork")


def calc_feature_desc_softmax(startIndex,endIndex,dataset,feature_collection):
    for cmpidx in tqdm(range(startIndex,endIndex,2)):
        print("Calculating for image "+str(cmpidx))
        imagedata = feature_collection.find_one({"_id": cmpidx})
        print('Image fetched')
        imagedata['fc_softmax_descriptor'] = fc_calculator_2(np.array(imagedata['image'], dtype=np.uint8)).tolist()
        feature_collection.update_one({'_id':cmpidx},{'$set':imagedata},upsert = True)
        print('DB Updated for '+str(cmpidx))
        
        
        
def calc_sim_score_softmax(startIndex,endIndex,feature_collection,similarity_collection):
    print('Entry calc_sim_score_softmax')
    for idx in tqdm(range(startIndex,endIndex,2)):
        imagedata1 = feature_collection.find_one({"_id": idx})
        softmax_score = {}
        for cmpidx in range(startIndex,endIndex,2):
            imagedata2 = feature_collection.find_one({"_id": cmpidx})
            sim_score = get_similarity_score_resnet(imagedata1['fc_softmax_descriptor'],imagedata2['fc_softmax_descriptor'])
            softmax_score[str(cmpidx)] = sim_score
        
        
        sim_scores = similarity_collection.find_one({"_id": idx})
        sim_scores['fc_softmax_descriptor'] = softmax_score
        similarity_collection.update_one({'_id':idx},{'$set':sim_scores},upsert = True)
        print('Calculation and DB updated done for image'+str(idx))
    
    print('Exit calc_sim_score_softmax')
    

def check_and_calc_softmax(startIndex,endIndex,feature_collection,similarity_collection):
    for idx in tqdm(range(startIndex,endIndex,2)):
        similarity_scores = feature_collection.find_one({"_id": idx})
        if('fc_softmax_descriptor' not in similarity_scores):
            print('Not present in DB, calculating for '+str(idx))
            calc_sim_score_softmax(idx, idx+1, feature_collection, similarity_collection)
        
        else:
            
    
    

if __name__=="__main__": 
    mod_path = Path(__file__).parent.parent
    caltech101 = Caltech101(str(mod_path) + "/caltech101",download=True)


    dbName = "CSE515-MWD-Nikhil_V_Ramanan-ProjectPhase2"
    odd_feature_collection = connect_to_db(dbName,'image_features_odd')
    feature_collection = connect_to_db(dbName,'image_features')
    similarity_collection = connect_to_db(dbName,'image_similarities')



    processes = [Process(target = calc_sim_score_softmax,args = (idx,idx+500,feature_collection,similarity_collection)) for idx in range(0,3000,250)]
    
    
    print("Threads Running")
    for p in processes:

        p.start()


    for p in processes:

        p.join()
        
        
    print("Threads Done")
