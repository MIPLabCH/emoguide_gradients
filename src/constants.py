"""
@author: Chun Hei Michael Chan
@copyright: Private Copyright
@credits: [Chun Hei Michael Chan]
@maintainer: Chun Hei Michael Chan
@email: miki998chan@gmail.com
"""                 

import os
import pickle
from copy import deepcopy

# Brain Imaging specifics
from nilearn.connectome import ConnectivityMeasure
from brainspace.gradient import GradientMaps



PALETTES      = ['PuOr', 'hsv', 'hsv', 'Spectral']
TR            = 1.3 # seconds
FILM2DURATION = {'AfterTheRain': 496, 
                 'BetweenViewing': 808,
                 'BigBuckBunny': 490, 
                 'Chatter': 405, 
                 'FirstBite': 599, 
                 'LessonLearned': 667, 
                 'Payload': 1008, 
                 'Sintel': 722, 
                 'Spaceman': 805, 
                 'Superhero': 1028, 
                 'TearsOfSteel': 588, 
                 'TheSecretNumber': 784, 
                 'ToClaireFromSonny': 402, 
                 'YouAgain': 798}

# trim the washimg time for movies before and after
WASH  = 93.9/ TR # duration in seconds for wash is 93.9 sec
ONSET = 6 / TR # duration of onset is assumed to be 6 sec



# Selected Item Representing emotions
select = ['Anger','Guilt','WarmHeartedness', \
             'Disgust','Happiness','Fear','Regard','Anxiety', \
             'Satisfaction','Pride','Surprise','Love','Sad']

### saving and loading made-easy
def save(pickle_file, array):
    """
    Pickle array
    """
    with open(pickle_file, 'wb') as handle:
        pickle.dump(array, handle, protocol=pickle.HIGHEST_PROTOCOL)
def load(pickle_file):
    """
    Loading pickled array
    """
    with open(pickle_file, 'rb') as handle:
        b = pickle.load(handle)
    return b

index2region = load('../resources/yeo7region414.pkl')
region2index = load('../resources/region414yeo7.pkl')