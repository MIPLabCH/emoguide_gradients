"""
@author: Chun Hei Michael Chan
@copyright: Private Copyright
@credits: [Chun Hei Michael Chan]
@maintainer: Chun Hei Michael Chan
@email: miki998chan@gmail.com
"""


from src.utils import *

###################################################### 
################### FUNC-CONNECTIV ###################
######################################################



def FC(series):
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

    correlation_measure = ConnectivityMeasure(kind='correlation')
    fc = correlation_measure.fit_transform([series])[0]
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



