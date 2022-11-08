"""
@author: Chun Hei Michael Chan
@copyright: Private Copyright
@credits: [Chun Hei Michael Chan]
@maintainer: Chun Hei Michael Chan
@email: miki998chan@gmail.com
"""

from src.utils import *
from src.metrics import *

###################################################### 
################### FUNC-CONNECTIV ###################
######################################################

def null_score(null_distrib, sample):
    """
    Information:
    ------------
    Give the p-value of estimated value against a null distribution

    Parameters
    ----------
    null_distrib::[1darray<float>]
    sample      ::[float]


    Returns
    -------
    score::[float]
        p-value
    """

    # single tail
    if np.sign(sample) == -1:
        score = np.mean(null_distrib < sample)
    else:
        score = np.mean(null_distrib > sample)
    return score

def plot_nulldistribution(totest, mainseries, mixseries):
    ndistrib,nscore = moviemix_stat_test(totest, mainseries, mixseries)
    
    plt.hist(ndistrib, bins=100, alpha=0.7, label="null distribution")
    plt.axvline(x=nscore, c='r', label='obtained correlation', linestyle='dotted', linewidth=3)
    plt.title("Null distirbution plot")
    plt.legend(loc=1, prop={'size':8})
    plt.show()


def moviemix_stat_test(totest, mainseries, mixseries):
    """
    Information:
    ------------
    Compute null distribution and main correlation
    (null distrib is defined as set of correlations obtained from
    all continuous timecourses of same size as the main one applied on all 
    the other timecourses)

    Parameters
    ----------
    totest    ::[1darray<float>]
        Most of times in our case would be metric series
    mainseries::[1darray<float>]
        Most of times in our case would be emotion series of a interest movie
    mixseries ::[1darray<float>]
        Most of times in our case would be emotion series of all but the interest movie

    Returns
    -------
    ndistrib::[1darray<float>]
    nscore  ::[float]
    cur     ::[float]s
        p-value
    """    

    # Obtained sample correlation
    mainseries = overlap_add(mainseries, 1)
    m          = min(len(totest), len(mainseries))
    cur,_,_    = correlation_search(zscore(totest[:m]), zscore(mainseries[:m]), 0, 1)

    # generate null distribution
    ndistrib = []
    for i in range(len(mixseries) - len(totest)):
        b = mixseries[i:i+len(totest)]
        b = overlap_add(b, 1)
        a, b = zscore(totest), zscore(b)
        corr, _,_ = correlation_search(a,b, 0, 1)
        ndistrib.append(corr)

    ndistrib = np.asarray(ndistrib)
    nscore   = null_score(ndistrib, cur)
    return ndistrib, nscore, cur    

