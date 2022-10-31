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

def null_score(null_distrib, sample):
    # single tail
    if np.sign(sample) == -1:
        score = np.mean(null_distrib < sample)
    else:
        score = np.mean(null_distrib > sample)
    return score

