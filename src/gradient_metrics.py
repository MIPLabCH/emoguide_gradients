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
from src.constants import *
from src.gradient_utils import *

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
    G1    ::[2darray<float>]
        First set of Gradients with dimension (nb regions, nb features) that we take network1 from
    G2    ::[2darray<float>]
        Second set of Gradients with dimension (nb regions, nb features) that we take network2 from
    N1    ::[str]
        Network name: that points to indices in the 400 parcellations that belong to a certain network (e.g "Vis")
    N2    ::[str]
        Network name: that points to indices in the 400 parcellations that belong to a certain network (e.g "Vis")
    method::[string]
        The type of distance to implement for volume computation

    Returns
    -------
    dist::[float]
    """

    c1 = G1[index2region[N1]]
    c2 = G2[index2region[N2]]
    dist = cluster_distance(c1,c2, method=method)
    return dist

# gradients distance with scores being the distance to centroid of each gradient
def gradients_distance2(G1,G2,similarity=False, pmethod='L2'):
    """
    Information:
    ------------
    Compute the distance between to networks (potentially from same gradients or different gradients)

    Parameters
    ----------
    G1        ::[2darray<float>]
        First set of Gradients with dimension (nb regions, nb features) that we take network1 from
    G2        ::[2darray<float>]
        Second set of Gradients with dimension (nb regions, nb features) that we take network2 from
    similarity::[Bool]
        similarity metric between two gradients' extracted features from 
    pmethod   ::[string]
        The type of distance between points
        
    Returns
    -------
    dist::[float]
    """

    c1 = G1.mean(axis=0)
    c2 = G2.mean(axis=0)
    
    V1 = np.asarray([points_distance(G1[n],c1, pmethod=pmethod) for n in range(len(G1))])
    V2 = np.asarray([points_distance(G2[n],c2, pmethod=pmethod) for n in range(len(G2))])
    
    if similarity:
        dist = pearson_correlation(V1,V2)
    else:
        dist = np.sum((V1 - V2)**2)

    return dist