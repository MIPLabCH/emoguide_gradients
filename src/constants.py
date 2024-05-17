"""
Copyright © 2023 Chun Hei Michael Chan, MIPLab EPFL
"""

from src.utils import *

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

index2region = load('./resources/yeo7region414.pkl')
region2index = load('./resources/region414yeo7.pkl')

index2region17 = load('./resources/yeo17region414.pkl')
region2index17 = load('./resources/region414yeo17.pkl')

onset_dur = load('./data/run_onsets.pkl')
