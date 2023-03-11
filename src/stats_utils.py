"""
@author: Chun Hei Michael Chan
@copyright: Private Copyright
@credits: [Chun Hei Michael Chan]
@maintainer: Chun Hei Michael Chan
@email: miki998chan@gmail.com
"""

from src.utils import *
from src.gradient_metrics import *

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
    cur     ::[float]
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


def shift_series_nulldistribution(Y, othermoviesdf, emotion_df, smfactor=1):
    """
    Information:
    ------------
    (Wrap moviemixstat)
    Compute null distribution and main correlation
    (null distrib is defined as set of correlations obtained from
    all continuous timecourses of same size as the main one applied on all 
    the other timecourses)

    Parameters
    ----------
    Y            ::[1darray<float>]
        Metric series
    othermoviesdf::[DataFrame]
        Dataframe of other movies 
    emotion_df   ::[DataFrame]
        Dataframe of current movie with which we computed Y. All emotions are considered for the current movie.
    

    Returns
    -------
    ret_score  ::[1darray<float>]
        correlations for each emotion
    ret_nscore ::[1darray<float>]
        p-value for each emotion
    """    

    ret_nscore = np.zeros((len(select)))
    ret_score  = np.zeros((len(select)))

    z1 = zscore(Y)
    for jdx, emotion in enumerate(select):
        concat_other = np.array(othermoviesdf[othermoviesdf.item == emotion]['score'])
        emo_series   = np.array(emotion_df[emotion_df.item==emotion]['score'])
        smoothened   = overlap_add(emo_series, smfactor)
        z2           = zscore(smoothened[:z1.shape[0]])

        _, nscore, corr = moviemix_stat_test(z1, z2, concat_other)
        ret_nscore[jdx] = nscore
        ret_score[jdx]  = corr
    
    return ret_score, ret_nscore
