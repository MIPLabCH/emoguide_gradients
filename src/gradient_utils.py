"""
Copyright © 2023 Chun Hei Michael Chan, MIPLab EPFL
"""


from src.utils import *
from brainspace.gradient import procrustes_alignment

###################################################### 
################### ALIGNEMENT-GRAD ##################
######################################################

def procrustes_score(ref_gradient, aligned):
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
        
    nbr, nbf = ref_gradient.shape
    error = np.square(ref_gradient - aligned).sum()
    error = error/nbr
    return error

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
    
    return np.asarray(aligned_gradients), ref, score
    

###################################################### 
################### FUNC-CONNECTIV ###################
######################################################



def pearson_correlation(arr1,arr2):
    """
    Information:
    ------------
    Direct pearson correlation

    Parameters
    ----------
    arr1   ::[1darray<float>]
        First signal

    arr2   ::[1darray<float>]
        Second signal

    Returns
    -------
    corr ::[float]
        Correlation value modulo shift with limited tolerance
        
    """
    
    corr = stats.pearsonr(arr1, arr2).statistic
    return corr

def spearman_correlation(arr1,arr2):
    """
    Information:
    ------------
    Direct spearman correlation

    Parameters
    ----------
    arr1   ::[1darray<float>]
        First signal

    arr2   ::[1darray<float>]
        Second signal

    Returns
    -------
    corr ::[float]
        Correlation value modulo shift with limited tolerance
        
    """
    
    corr = stats.spearmanr(arr1, arr2).correlation
    return corr

def correlation_search(arr1, arr2, tolshift, find=0):
    """
    Information:
    ------------
    On a large series 

    Parameters
    ----------
    arr1   ::[1darray<float>]
        First signal

    arr2   ::[1darray<float>]
        Second signal

    tolshift::[int]
        Number of timepoints we are allowed to shift

    find    ::[bool]
        Returning index of where the arrays have highest correlation

    Returns
    -------
    corr ::[float]
        Correlation value modulo shift with limited tolerance

    tup  ::[tuple<int, int>]
        Indexes for array1 and array2 to start with for the shift with best correlation

    p-val::[float]
        Significance of correlation

        
    """
    
    assert arr1.shape == arr2.shape
    # NOTE: we do not allow both to be shifted
    # shift 1st array then select max
    scores1 = [stats.pearsonr(arr1, arr2).statistic] + [stats.pearsonr(arr1[i:], arr2[:-i]).statistic for i in range(1,tolshift)]

    # shift 2nd array then select max
    scores2 = [stats.pearsonr(arr1[:-i], arr2[i:]).statistic for i in range(1,tolshift)]
    
    corr_pos = np.max(scores1 + scores2)
    corr_neg = np.min(scores1 + scores2)
    corr     = corr_pos * (np.abs(corr_pos) >= np.abs(corr_neg)) + corr_neg * (np.abs(corr_pos) < np.abs(corr_neg))
    if find:
        if corr in scores1: return corr, (scores1.index(corr), 0), stats.pearsonr(arr1[scores1.index(corr):], arr2).pvalue
        elif corr in scores2: return corr, (0, scores2.index(corr) + 1), stats.pearsonr(arr1, arr2[scores1.index(corr):]).pvalue

    return corr

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



