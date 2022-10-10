"""
@author: Chun Hei Michael Chan
@copyright: Private Copyright
@credits: [Chun Hei Michael Chan]
@maintainer: Chun Hei Michael Chan
@email: miki998chan@gmail.com
"""


import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm


###################################################### 
################### GLOB VARIABLES ###################
######################################################

plt.style.use('fivethirtyeight')


PALETTE      = ['b','r','g','k','c','m','y']
ROOT         = "/Users/michaelc.c.h/Desktop/EPFL/"
TO_DATA_PATH = ""


###################################################### 
################### VISUALISATION ####################
######################################################



def plot_eigenvector_importance(gradient):
    """
    Information:
    ------------
    Plot decomposed components' importance after connectivity gradient fitting

    Parameters
    ----------
    gradient::[GradientMaps]
        GradientMaps class from brain space object that already had been fitted a FC

    Returns
    -------
    None: [None]
    """

    fig, ax = plt.subplots(1, figsize=(5, 4))
    ratio   = gradient.lambdas_/np.sum(gradient.lambdas_)
    ax.scatter(range(gradient.lambdas_.size), ratio, label="distribution")
    ax.set_xlabel('components')
    ax.set_ylabel('eigenvalue')
    ax.set_title("distribution of explained variance")
    ax.legend()

    plt.show()



###################################################### 
###################  DATA - FUNC  ####################
######################################################


def df_to_timeseries(df, filename):
    """
    Information:
    ------------
    Read our formatted dataframes to obtain timeseries 
    in (time,voxels) format of a specific acquisition

    Parameters
    ----------
    df      ::[DataFrame]
        DataFrame containing mri info
    
    filename::[string]
        Name of the acquisition we want to single out


    Returns
    -------
    series  ::[2darray<float>]
        timeseries that we are interested in 
    cur_file::[DatFrame]
        DataFrame of info about the filename we gave as input
    """
    
    cur_file = df[df.filename == filename]
    nbv, nbt = int(cur_file.vindex.max()) + 1, len(cur_file[cur_file.vindex==0])
    series   = np.zeros((nbt,nbv))

    for v in range(nbv):
        series[:,v] = np.array(cur_file[cur_file.vindex==v]['score'])
    return series, cur_file




###################################################### 
################### OS-LEVEL FUNC ####################
######################################################

import pickle
### saving and loading made-easy
def save(pickle_file, array):
    with open(pickle_file, 'wb') as handle:
        pickle.dump(array, handle, protocol=pickle.HIGHEST_PROTOCOL)
def load(pickle_file):
    with open(pickle_file, 'rb') as handle:
        b = pickle.load(handle)
    return b


def loadimg_in_order(unordered_img):
    """
    Information:
    ------------
    Specifically working for our format of TYPE_FRAMENUMBER.jpg files
    we extract the numbers to reorder them in increasing order.

    Parameters
    ----------
    unordered_img::[list<string>]
        list of unordered filenames (of images) in the format shown above
        very specific to our use case

    Returns
    -------
    ordered_img::[list<string>]
        ordered filenames (of images)
    """


    numbers      = [int(r.strip('.jpg').split('_')[1]) for r in unordered_img]
    sorted_index = np.argsort(numbers)
    ordered_img  = np.array(unordered_img)[sorted_index]

    return ordered_img
    


def img2video(img_array, fps, outpath_name="out.mp4"):
    """
    Information:
    ------------
    Read our formatted dataframes to obtain timeseries 
    in (time,voxels) format of a specific acquisition

    Parameters
    ----------
    img_array   ::[4darray<uint8>]
        Stream of images (most of times RGB) that we want to link as a video
    
    fps         ::[int]
        Encoding/Displaying fps
    
    outpath_name::[string]
        Path and name of the video file to output

    Returns
    -------
    None::[None]
    """    
    height, width, layers = img_array[0].shape
    size = (width,height)

    out = cv2.VideoWriter(outpath_name,cv2.VideoWriter_fourcc(*'MP4V'), fps, size)
    
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()