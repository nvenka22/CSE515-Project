### Implement a program which computes and prints the “inherent dimensionality” 
##         associated with the even numbered Caltec101 images.

import argparse
import scipy
import os

parser = argparse.ArgumentParser()
ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) 

def calculate_inherent_dim(features, method):
    pass

def get_features_by_descriptor(fd):
    data = scipy.io.loadmat(ROOT_DIR+'/Store/arrays.mat')
    features = data[fd]
    


if __name__ == "__main__":
    
    parser.add_argument('--fd', type=str,
                    help='Enter a Feature Descriptor to calculate Latent Semantics')
    args = parser.parse_args()

    featur_descriptor = args.fd
    features = get_features_by_descriptor(featur_descriptor)