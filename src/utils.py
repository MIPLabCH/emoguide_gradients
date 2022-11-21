"""
@author: Chun Hei Michael Chan
@copyright: Private Copyright
@credits: [Chun Hei Michael Chan]
@maintainer: Chun Hei Michael Chan
@email: miki998chan@gmail.com
"""


import os
import cv2
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from copy import deepcopy

from tqdm import tqdm

# Brain Imaging specifics
from nilearn.connectome import ConnectivityMeasure
from brainspace.gradient import GradientMaps

from brainspace.plotting import plot_hemispheres
from brainspace.utils.parcellation import map_to_labels
from brainspace.datasets import load_group_fc, load_parcellation, load_conte69


###################################################### 
################### GLOB VARIABLES ###################
######################################################

# plt.style.use('fivethirtyeight')

# PALETTE       = ['b','r','g','k','c','m','y']
PALETTES = ['PuOr', 'hsv', 'hsv', 'Spectral']
TR            = 1.3 # seconds
FILM2DURATION = {'AfterTheRain': 496, 'BetweenViewing': 808, 'BigBuckBunny': 490, 'Chatter': 405, 'FirstBite': 599, 'LessonLearned': 667, 'Payload': 1008, 'Sintel': 722, 'Spaceman': 805, 'Superhero': 1028, 'TearsOfSteel': 588, 'TheSecretNumber': 784, 'ToClaireFromSonny': 402, 'YouAgain': 798}

# trim the washimg time for movies before and after
WASH  = 93.9/ TR # duration in seconds for wash is 93.9 sec
ONSET = 6 / TR # duration of onset is assumed to be 6 sec


###################################################### 
###################  SIG -- PROC  ####################
######################################################

def sscore(signal):
    """
    Information:
    ------------
    Shift and Scale signal

    Parameters
    ----------
    signal::[ndarray<float>]
        Signal shift and scale

    Returns
    -------
    score::[ndarray<float>]
    """    
    signal = (signal - signal.min())
    score  = signal / signal.max()
    return score

def zscore(signal, ret_param=False):
    """
    Information:
    ------------
    Remove mean and normalize by standard deviation on any array size

    Parameters
    ----------
    signal    ::[ndarray<float>]
        Signal remove mean and normalize

    ret_params::[Bool]
        Whether we return the mean and standard deviation of original signal
    Returns
    -------
    score::[ndarray<float>]
    """    
    m, s  = signal.mean(), signal.std()
    score = (signal - m) / s
    if ret_param:
        return score, m, s
    return score


def overlap_add(signal, wsize=3, pad=False):
    """
    Information:
    ------------
    Smoothen a signal by adding to part of itself to other intervals

    Parameters
    ----------
    signal::[1darray<float>]
        Signal to do rolling average on 

    ws    ::[int]
        Window size to do rolling average on

    pad   ::[Bool]
        If true then pad the boundaries to return an array of same size as input
        If false then leave boundaries not computed

    Returns
    -------
    overlapped::[1darray<float>]
    """

    if pad:
        overlapped = np.concatenate([np.convolve(signal, np.ones(wsize)/wsize, mode='valid'),signal[-(wsize-1):]])
    else:
        overlapped = np.convolve(signal, np.ones(wsize)/wsize, mode='valid')
    return overlapped

def low_pass(signal, ks=10):
    """
    Information:
    ------------
    Smoothen a signal by adding to part of itself to other intervals

    Parameters
    ----------
    signal::[1darray<float>]
        Signal to do local average / smooth on 
        
    ks    ::[int]
        Kernel size

    Returns
    -------
    convolved::[1darray<float>]
    """    
    convolved = np.convolve(signal, np.ones(ks)/ks, 'same')
    return convolved


def resize_like():
    pass

###################################################### 
################### VISUALISATION ####################
######################################################

def plot_spectrum(sig, sampling_rate=1/TR, ls=0, rs=1):
    """
    Information:
    ------------
    Plot power spectrum of a signal

    Parameters
    ----------
    sig           ::[1darray<float>]
    sampling_rate ::[int]

    Returns
    -------
    None::[None]
    """    
    fourier_transform = np.fft.rfft(sig)
    abs_fourier_transform = np.abs(fourier_transform)
    power_spectrum = np.square(abs_fourier_transform)

    frequency = np.linspace(0, sampling_rate/2, len(power_spectrum))

    plt.plot(frequency, power_spectrum, label='spectre')
    plt.legend()
    plt.xlabel('Freq')
    plt.title("Power spectrum")
    plt.xlim(ls,rs)
    plt.show()


def compare_videos(arr1, arr2):
    """
    Information:
    ------------
    Read our formatted dataframes to obtain timeseries 
    in (time,voxels) format of a specific acquisition

    Parameters
    ----------
    arr1::[4darray<uint8>]
        First Stream of images (to be concatenated in the left side)
    
    arr2::[4darray<uint8>]
        Second Stream of images (to be concatenated in the right side)

    Returns
    -------
    concat::[4darray<uint8>]
        Stream of images that were concatenated horizontally together
    """
        
    arr1 = np.asarray(arr1)
    arr2 = np.asarray(arr2)

    # same number of timepoints: we take the minimum
    t_min = min(arr1.shape[0], arr2.shape[0])
    # put a vertical separator of arbitrary color
    spacer = 255 * np.ones((t_min,arr1.shape[1], 50,3), dtype=np.uint8)
    concat = np.concatenate([arr1[:t_min],spacer,arr2[:t_min]],axis=2)

    return concat

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
    None::[None]
    """

    fig, ax = plt.subplots(1, figsize=(5, 4))
    ratio   = gradient.lambdas_/np.sum(gradient.lambdas_)
    ax.scatter(range(gradient.lambdas_.size), ratio, label="distribution")
    ax.set_xlabel('components')
    ax.set_ylabel('eigenvalue')
    ax.set_title("distribution of explained variance")
    ax.legend()

    plt.show()

def plot_gradient_cortical(gradients,label_text):
    """ temporary not sure if useful yet"""
    labeling         = load_parcellation('schaefer', scale=400, join=True)
    surf_lh, surf_rh = load_conte69()

    mask = labeling != 0

    # TODO: to fix, we are currently artificially fixing the value bar
    grad = [None] * len(gradients)
    for k in range(2):
        # map the gradient to the parcels
        grad[k] = map_to_labels(gradients[k], labeling, mask=mask, fill=np.nan)

    plot_hemispheres(surf_lh, surf_rh, array_name=grad, size=(1000, 600), cmap='hot',
                    color_bar=True, label_text=label_text, zoom=1.25, embed_nb=True, interactive=False,
                    transparent_bg=False)




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
    
def video2img(video_path, start_idx, end_idx):
    """
    Information:
    ------------
    Convert mp4 video into stream of numpy arrays

    Parameters
    ----------
    video_path::[string]
        Path to the video to read

    start_idx ::[int]
        frame of the video to start reading from

    end_idx   ::[int]
        frame of the video to stop reading at

    Returns
    -------
    frames::[4darray<uint8>]
        Stream of images (most of times RGB) that we obtain from reading the video
    """        
    frames = []
    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name
    cap = cv2.VideoCapture(video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("Display FPS is: {}".format(fps))
    # Check if camera opened successfully
    if (cap.isOpened()== False):
        print("Error opening video stream or file")

    for frame_id in range(start_idx,end_idx):
        # Capture frame-by-frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if ret == True:
            frames.append(frame)

    # When everything done, release the video capture object
    cap.release()

    frames = np.asarray(frames)
    return frames


def img2video(img_array, fps, outpath_name="out.mp4"):
    """
    Information:
    ------------
    Convert a stream of RGB images into mp4 video

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