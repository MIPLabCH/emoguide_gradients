"""
@author: Chun Hei Michael Chan
@copyright: Private Copyright
@credits: [Chun Hei Michael Chan]
@maintainer: Chun Hei Michael Chan
@email: miki998chan@gmail.com
"""                 

import os
from copy import deepcopy

# Brain Imaging specifics
from nilearn.connectome import ConnectivityMeasure
from brainspace.gradient import GradientMaps


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