"""
Copyright © 2023 Chun Hei Michael Chan, MIPLab EPFL
"""

import cv2
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt

from tqdm.auto import tqdm

from brainspace.plotting import plot_hemispheres
from brainspace.utils.parcellation import map_to_labels
from brainspace.datasets import load_parcellation, load_conte69


from src.constants import *

###################################################### 
###################  SIG -- PROC  ####################
######################################################


### NI-EDU - copied code "GLM inference"
def design_variance(X, which_predictor=1):
    ''' Returns the design variance of a predictor (or contrast) in X.
    
    Parameters
    ----------
    X : numpy array
        Array of shape (N, P)
    which_predictor : int or list/array
        The index of the predictor you want the design var from.
        Note that 0 refers to the intercept!
        Alternatively, "which_predictor" can be a contrast-vector
        (which will be discussed later this lab).
        
    Returns
    -------
    des_var : float
        Design variance of the specified predictor/contrast from X.
    '''
    
    is_single = isinstance(which_predictor, int)
    if is_single:
        idx = which_predictor
    else:
        idx = np.array(which_predictor) != 0
    
    c = np.zeros(X.shape[1])
    c[idx] = 1 if is_single == 1 else which_predictor[idx]
    des_var = c.dot(np.linalg.inv(X.T.dot(X))).dot(c.T)
    return des_var

def local_average(signal,ks):
    """
    Information:
    ------------
    Compute local average per timepoint

    Parameters
    ----------
    signal  ::[1darray<float>]
    
    ks      ::[int]
        Kernel size for averaging


    Returns
    -------
    res  ::[1darray<float>]
        locally averaged signal
    """

    size = len(signal)
    res  = np.zeros((size // ks+1))
    for idx,k in enumerate(range(0,len(signal), ks)):
        res[idx] = signal[k:k+ks].mean()
    return res


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

def visualize_jointplot_dc_grad(outname, dc_grad, wsub=False, framerate=10, ws=40):
    """
    Information:
    ------------
    Generate video mp4 visualizing dynamic gradients

    Parameters
    ----------
    outname  ::[String]
        Prefix Name of the file
    
    dc_grad  ::[3darray<float>]
        dynamic gradients
        dim:(# of windows, # of parcels, # of gradient dimension)

    wsub     ::[Bool]
        Is the dynamic gradients including subcortical regions? 
    
    framerate::[Int]
        Frame rate of outputed mp4

    ws       ::[Int]
        Window size of dynamic gradient

    Returns
    -------
    None::[None]
    """
    pal       = sns.color_palette('colorblind', 8)
    if wsub:
        partial_region = load('./resources/region414yeo7.pkl')
        coloring  = [partial_region[i] for i in range(414)]
    else:    
        partial_region = load('./resources/region400yeo7.pkl')
        coloring  = [partial_region[i] for i in range(400)]
    
    # figure size that is saved

    tosave = []
    for j in tqdm(range(len(dc_grad))):
        tmp_df = {"G1":dc_grad[j][:,0] , "G2": dc_grad[j][:,1], "region": coloring}
        tmp_df = pd.DataFrame.from_dict(tmp_df)
        ax     = sns.jointplot(data=tmp_df, x="G1", y="G2", 
                    hue="region", height=7, 
                    xlim=(dc_grad[:,:,0].min()-10,dc_grad[:,:,0].max()+10), 
                    ylim=(dc_grad[:,:,1].min()-10,dc_grad[:,:,1].max()+10), palette=pal)
        ax.fig.suptitle("Gradients's closenedness plot ({})".format(outname), fontsize=12)
        legend_properties = {'weight':'bold','size':5}
        ax.ax_joint.legend(prop=legend_properties,loc='upper right')
        ax.ax_joint.set_xlabel('G1',fontsize=10)
        ax.ax_joint.set_ylabel('G2',fontsize=10)
        w,h = ax.fig.canvas.get_width_height()

        ax.fig.canvas.draw()
        img_arr = np.fromstring(ax.fig.canvas.tostring_rgb(), 
                        dtype=np.uint8,
                        sep='')
        img_arr = img_arr.reshape(w,h,3)[:,:,::-1]
        tosave.append(img_arr)
        
    plt.close("all")
    img2video(tosave, framerate, outpath_name='./media/{}_plots_ws{}.mp4'.format(outname,ws))


###################################################### 
################### OS-LEVEL FUNC ####################
######################################################

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
    cap            = cv2.VideoCapture(video_path)
    total_nb_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))    
    fps            = cap.get(cv2.CAP_PROP_FPS)

    print("Display FPS is: {}".format(fps))
    # Check if camera opened successfully
    if (cap.isOpened()== False):
        print("Error opening video stream or file")

    # Set to last frames if -1 is encodeds
    if end_idx == -1:
        end_idx = total_nb_frame

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


###################################################### 
################### TEXT RENDERING ###################
######################################################


# used only in this notebook functional
def latexify_significancy_table(statistics, pvalues, corrflag=True, columnsname=[], rowsname=[]):
    """
    Information:
    ------------
    Read our formatted dataframes to obtain timeseries 
    in (time,voxels) format of a specific acquisition

    Parameters
    ----------
    ref_gradient::[2darray<float>]
        reference gradients with dimension (nb of region, nb of features) in our case most of times
        number of features would be the number of eigenvectors
    
    aligned     ::[2darray<float>]
        gradients that we aligned to reference with same dimension as reference gradients

    Returns
    -------
    error::[float]
        Score for the alignement process in comparison to the original reference
    """

    # statistics is 2d array with the scores for two conditions
    def color_positive(val):
        """
        Takes a scalar and returns a string with
        the css property `'color: green'` for positive
        strings, black otherwise.
        """
        if val[-1] == '*':
            if val[0] == '-':
                color = 'blue'
            else:
                color = 'red'
        else:
            color = 'black'
        return 'color: %s' % color
    
    def bold(val):
        if val[-1] == '*':
            return "font-weight: bold"
    
    df    = pd.DataFrame.from_records(np.round(statistics,3))
    sigdf = pd.DataFrame.from_records(pvalues).applymap(lambda x: ''.join(['*' for t in [.05, .01, .001] if x<=t]))
    signidf = (df.round(3).astype(str) + sigdf)

    if corrflag:
        signidf = signidf.style.applymap(color_positive)
    else:
        signidf = signidf.style.applymap(bold)
        
    if len(columnsname) != 0 and len(rowsname) != 0:
        signidf.data.columns = [s[:5] for s in columnsname]
        signidf.data.index = rowsname
    
    print(signidf.to_latex(
    caption="Selected stock correlation and simple statistics.",
    clines="skip-last;data",
    convert_css=True,
    position_float="centering",
    multicol_align="|c|",
    hrules=True))    