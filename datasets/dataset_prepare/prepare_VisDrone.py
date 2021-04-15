from PIL import  Image
import os
import cv2 as cv
import matplotlib.pyplot as plt
from pylab import plot
import numpy as np
import json
import math
import torch
from functions import euclidean_dist,  generate_cycle_mask, average_del_min
mode = 'train'
import glob

img_path = '/home/ubuntu/src/IIM/datasets/ProcessedData/VisDrone/images/'
json_path = '/home/ubuntu/src/IIM/datasets/ProcessedData/VisDrone/json'
mask_path = '/home/ubuntu/src/IIM/datasets/ProcessedData/VisDrone/mask_50_60'
cycle = False

def calc_mean_std():
    count = 0
    mean = 0
    delta = 0
    delta2 = 0
    M2 = 0
    for filename in glob.glob(os.path.join(img_path, '*.jpg')): #assuming gif
        img = np.asarray(Image.open(filename))


if __name__ == '__main__':
    calc_mean_std()
