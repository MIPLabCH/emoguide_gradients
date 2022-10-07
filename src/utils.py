"""
@author: Chun Hei Michael Chan
@copyright: Private Copyright
@credits: [Chun Hei Michael Chan]
@maintainer: Chun Hei Michael Chan
@email: miki998chan@gmail.com
"""


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm


###################################################### 
################### GLOB VARIABLES ###################
######################################################

plt.style.use('fivethirtyeight')





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
################### LOADING FUNC  ####################
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