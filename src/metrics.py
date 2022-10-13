"""
@author: Chun Hei Michael Chan
@copyright: Private Copyright
@credits: [Chun Hei Michael Chan]
@maintainer: Chun Hei Michael Chan
@email: miki998chan@gmail.com
"""


import numpy as np






###################################################### 
################### METRICS - EVAL ###################
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