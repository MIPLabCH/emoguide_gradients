"""
@author: Chun Hei Michael Chan
@copyright: Private Copyright
@credits: [Chun Hei Michael Chan]
@maintainer: Chun Hei Michael Chan
@email: miki998chan@gmail.com
"""


from src.gradient_metrics import procrustes_score
from src.utils import *
from brainspace.gradient import procrustes_alignment

###################################################### 
################### ALIGNEMENT-GRAD ##################
######################################################

def procrustes_align(list_gradients, ref=None, score_flag=True, n_iter=100, tol=1e-5):
    """
    Information:
    ------------
    Procruste alignment Evaluations

    Parameters
    ----------
    list_gradients::[list<2darray<float>>]
        List of gradients each gradients have dimension (nb regions, nb features)
    ref           ::[2darray<float>]
        Reference gradients used in alignment (nb regions, nb features)
    score_flag    ::[bool]
        True if we want to evalute procrustes score as well
    n_iter        ::[int]
        Number of iteration for procrustes alignement

    Returns
    -------
    aligned_gradients::[list<2darray<float>>]
        List of aligned gradients each gradients have dimension (nb regions, nb features)
    ref              ::[2darray<float>]
        Reference gradients used in alignment (nb regions, nb features)
    score            ::[float]
        Average procrustes alignment score on across all alignments
    """    
    aligned_gradients, ref = procrustes_alignment(list_gradients, reference=ref, return_reference=True, n_iter=n_iter, tol=tol)
    score     = None
    if score_flag:
        # score = np.mean([procrustes_score(ref,aligned_gradients[i]) for i in range(len(aligned_gradients))])
        score = [procrustes_score(ref ,aligned_gradients[i]) for i in range(len(aligned_gradients))]
    
    return aligned_gradients, ref, score
    

###################################################### 
################### FUNC-CONNECTIV ###################
######################################################



def FC(series,verbose=False):
    """
    Information:
    ------------
    Compute the static functional connectivity matrix of a timeseries of fMRI

    Parameters
    ----------
    series::[2darray<float>]
        fMRI timeseries of dimension : (nb timepoints, nb regions)

    Returns
    -------
    fc::[2darray<float>]
        FC of interest of dimension : (nb regions, nb regions)
    """
    # Removal of NaN
    S = series[np.isnan(series).sum(axis=1) == 0]

    # Arbitrary cutoff for relevance of correlation
    if S.shape[0] < 10: 
        if verbose: print("Less than 10 timepoints for correlation")
        return 0 

    correlation_measure = ConnectivityMeasure(kind='correlation')
    fc = correlation_measure.fit_transform([S])[0]
    return fc

def sliding_window_FC(series, ws, step=1):
    """
    Information:
    ------------
    Compute the dynamic functional connectivity matrices of a timeseries of fMRI

    Parameters
    ----------
    series::[2darray<float>]
        fMRI timeseries of dimension : (nb timepoints, nb regions)
    ws    ::[int]
        Window size for sliding window
    step  ::[int]
        Step size to go from one window to the next one

    Returns
    -------
    dFC::[3darray<float>]
        Dynamic FC of interest of dimension : (nb windows, nb regions, nb regions)
    """
    nbr, nbt = series.T.shape

    dFC = np.zeros(((nbt - ws) // step + 1, nbr,nbr))

    for c, sidx in enumerate(range(0, nbt, step)):
        T = series[sidx:sidx+ws]
        if T.shape[0] != ws: 
            continue
        dFC[c] = FC(T)
    
    return dFC

def jackknife_FC(series):
    """
    Information:
    ------------
    Compute the dynamic functional connectivity matrices of a timeseries of fMRI

    Parameters
    ----------
    series::[2darray<float>]
        fMRI timeseries of dimension : (nb timepoints, nb regions)

    Returns
    -------
    dFC::[3darray<float>]
        Dynamic FC of interest of dimension : (nb timestamp, nb regions, nb regions)
    """
    nbr, nbt = series.T.shape

    dFC = np.zeros((nbt, nbr,nbr))
    for sidx in range(0, nbt):
        # Jackknife sampling
        T = np.concatenate([series[:sidx], series[sidx+1:]])
        dFC[sidx] = -FC(T)
    
    return dFC    



