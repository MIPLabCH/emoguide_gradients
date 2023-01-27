"""
@author: Chun Hei Michael Chan
@copyright: Private Copyright
@credits: [Chun Hei Michael Chan]
@maintainer: Chun Hei Michael Chan
@email: miki998chan@gmail.com
"""


import numpy as np
from scipy import stats
from scipy.spatial import ConvexHull

from src.utils import *

index2region = load('./resources/yeo7region414.pkl')
region2index = load('./resources/region400yeo7.pkl')

# Selected Item Representing emotions
select = ['Anger','Guilt','WarmHeartedness', \
             'Disgust','Happiness','Fear','Regard','Anxiety', \
             'Satisfaction','Pride','Surprise','Love','Sad']

###################################################### 
################### METRICS - EVAL ###################
######################################################

def points_distance(pts1, pts2, pmethod="L1"):
    """
    Information:
    ------------
    Compute distance between two points

    Parameters
    ----------    
    pts1   ::[1darray<float>]
        Point would be of length the number of features
    pts2   ::[2darray<float>]

    pmethod::[string]
        The type of distance to implement for two points

    Returns
    -------
    dist::[float]
        Distance between the two points
    """

    assert pts1.shape == pts2.shape

    if pmethod == "L0":
        dist = max(np.abs(pts1-pts2))

    elif pmethod == "L1":
        dist = np.abs(pts1 - pts2).sum()
    
    elif pmethod == "L2":
        dist = np.linalg.norm(pts1-pts2)
    
    return dist 

def cluster_distance(clst1, clst2, method="closest", pmethod="L1"):
    """
    Information:
    ------------
    Compute distance between two clusters

    Parameters
    ----------
    clst1  ::[2darray<float>]
        Array of points, point would be of length the number of features (nb_points, nb_features)
    clst2  ::[2darray<float>]

    method ::[string]
        The type of distance to implement for two clusters
    pmethod::[string]
        The type of distance to implement for two points

    Returns
    -------
    dist::[float]
        Distance between the two clusters
    """

    if method == "closest":
        dist = min([points_distance(pts1, pts2, pmethod=pmethod) for pts1 in clst1 for pts2 in clst2])
            
                
    elif method == "centroid":
        centroid1 = clst1.mean(axis=0)
        centroid2 = clst2.mean(axis=0)
        dist = points_distance(centroid1, centroid2, pmethod=pmethod)
    
    return dist

def gradients_distance(g1, g2, pmethod="L1"):
    """
    Information:
    ------------
    Compute distance between two clusters

    Parameters
    ----------
    g1  ::[2darray<float>]
        gradient of a given FC matrix (dimension (nb_region, nb_features))
    g2  ::[2darray<float>]
        gradient of a given FC matrix (dimension (nb_region, nb_features))
    pmethod::[string]
        The type of distance to implement for two points

    Returns
    -------
    dist::[float]
        Distance between the two gradients (also a distance used for procrustes value)
    """
            
    dist = np.sum([points_distance(g1[n],g2[n], pmethod=pmethod) for n in range(len(g1))])
    return dist    

def network_position(grad, network, grad_idx=0):
    """
    Information:
    ------------
    Remove mean and normalize by standard deviation on any array size

    Parameters
    ----------
    grad      ::[2darray<float>]
        Gradients with dimension (nb regions, nb features)
    
    network   ::[str]
        Network name: that points to indices in the 400 parcellations that belong to a certain network (e.g "Vis")
    
    grad_idx  ::[int]
        Which gradient we are estimating the variance of

    Returns
    -------
    dist::[float]
    """        
    dist = np.abs(grad[:,index2region[network]]).mean(axis=1)[:,grad_idx]
    return dist

def network_variance(grad, network, grad_idx=0):
    """
    Information:
    ------------
    Remove mean and normalize by standard deviation on any array size

    Parameters
    ----------
    grad      ::[2darray<float>]
        Gradients with dimension (nb regions, nb features)
    
    network   ::[str]
        Network name: that points to indices in the 400 parcellations that belong to a certain network (e.g "Vis")
    
    grad_idx  ::[int]
        Which gradient we are estimating the variance of

    Returns
    -------
    dist::[float]
    """        
    dist = np.abs(grad[:,index2region[network]]).std(axis=1)[:,grad_idx]
    return dist


def parcel_to_network(full_grad, type='grad_centroid', pmethod='L2'):
    """
    Information:
    ------------
    Compute the distance of each parcels to its respective network or to the general gradient

    Parameters
    ----------
    full_grad ::[2darray<float>]
        Gradients with dimension (nb regions, nb features)
    
    type      ::[str]
        Name of the reference to compute distance to parcels either "grad_centroid" or "net_centroid"

    pmethod   ::[string]
        The type of distance to implement for two points

    Returns
    -------
    dist::[1darray<float>]
    """

    if type == "grad_centroid":
        centroid = np.asarray([np.mean(full_grad[:,i]) for i in range(full_grad.shape[1])])
        dist     = np.array([points_distance(pts, centroid, pmethod=pmethod) for pts in full_grad])
    elif type == "net_centroid":
        centroid_dict = {}
        for network in index2region.keys():
            net_grad = full_grad[index2region[network]]
            centroid_dict[network] = np.asarray([np.mean(net_grad[:,i]) for i in range(net_grad.shape[1])])

        dist = []
        for idx,pts in enumerate(full_grad):
            centroid = centroid_dict[region2index[idx]]
            dist.append(points_distance(pts, centroid, pmethod=pmethod))

        dist = np.array(dist)
    
    return dist




def network_volume(grad, network, method='distance', pmethod='L2'):
    """
    Information:
    ------------
    Compute the dispersion / volume generated by the parcels belonging to the same network

    Parameters
    ----------
    grad      ::[2darray<float>]
        Gradients with dimension (nb regions, nb features)
    
    network   ::[str]
        Network name: that points to indices in the 400 parcellations that belong to a certain network (e.g "Vis")
    
    method    ::[string]
        The type of distance to implement for volume computation

    pmethod   ::[string]
        The type of distance to implement for two points

    Returns
    -------
    dist::[float]
    """
    if network == "all":
        net_grad = grad
    else:
        net_grad = grad[index2region[network]]
        
    if method == 'distance':
        centroid = np.asarray([np.mean(net_grad[:,i]) for i in range(net_grad.shape[1])])
        dist     = np.mean([points_distance(pts, centroid, pmethod=pmethod) for pts in net_grad])

    elif method == 'hull':
        hull = ConvexHull(net_grad)
        dist = hull.volume

    return dist

def mean_region_motion(grads, pmethod="L2"):
    """
    Information:
    ------------
    Compute average movement of each parcel (400) from dynamic gradients

    Parameters
    ----------
    grads      ::[list<2darray<float>>]
        Gradients with dimension (nb regions, nb features)

    pmethod   ::[string]
        The type of distance to implement for two points

    Returns
    -------
    region_motion::[1darray<float>]
        array of parcels' average movements
    """    
    # For each point we look at its movement over time and compute its mean 
    # the mean acts as a way of knowing how much in average that point moves
    t,nbr,nbf = grads.shape
    region_motion = np.zeros((nbr))
    for r in range(nbr):
        tmpA = grads[:, r][:-1]
        tmpB = grads[:, r][1:]

        D = np.asarray([ points_distance(tmpA[pidx], tmpB[pidx], pmethod=pmethod) 
                        for pidx in range(len(tmpA))])
        region_motion[r] = D.mean()
    
    return region_motion

def networks_distance(G1, G2, N1, N2, method="centroid"):
    """
    Information:
    ------------
    Compute the distance between to networks (potentially from same gradients or different gradients)

    Parameters
    ----------
    grad1     ::[2darray<float>]
        First set of Gradients with dimension (nb regions, nb features) that we take network1 from
    grad2     ::[2darray<float>]
        Second set of Gradients with dimension (nb regions, nb features) that we take network2 from
    network1  ::[str]
        Network name: that points to indices in the 400 parcellations that belong to a certain network (e.g "Vis")
    network2  ::[str]
        Network name: that points to indices in the 400 parcellations that belong to a certain network (e.g "Vis")        
    
    method    ::[string]
        The type of distance to implement for volume computation
    Returns
    -------
    dist::[float]
    """        
    c1 = G1[index2region[N1]]
    c2 = G2[index2region[N2]]
    dist = cluster_distance(c1,c2, method=method)
    return dist


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