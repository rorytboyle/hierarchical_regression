# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 15:26:45 2019

@author: BOYLER1
"""

import pandas as pd
import statsmodels.api as sm
import scipy.stats
import numpy as np

#import os
#curr_dir = os.getcwd()
#os.chdir(r'C:\Users\boyler1\Documents\PhD\Code\cognitive_reserve')
#import hierarchical_regression as h_r
#os.chdir(curr_dir)

# download this data file
# https://courses.edx.org/c4x/MITx/15.071x_2/asset/NBA_train.csv

# read in data
nba = pd.read_csv(r'C:\Users\boyler1\Documents\PhD\Miscellaneous\datasets\NBA_train.csv')

# prep data
#X = [nba['PTS'], nba[['PTS', 'ORB']], nba[['PTS', 'ORB', 'DRB']]]
nba['interaction'] = nba['PTS'] * nba['ORB']
X = [nba['PTS'], nba[['PTS', 'ORB']], nba[['PTS', 'ORB', 'interaction']]]
y = nba['W']
names = [['points'], ['points', 'offensive_rebounds'], 
         ['points', 'offensive_rebounds', 'interaction']]

results, models = hierarchical_regression(y, X, names, 
                                          r'B:\cognitive_reserve\prelim_analysis_280919')

runfile('C:/Users/boyler1/Documents/PhD/Code/cognitive_reserve/regression_diagnostics.py',
        wdir='C:/Users/boyler1/Documents/PhD/Code/cognitive_reserve')
runfile('C:/Users/boyler1/Documents/PhD/Code/cognitive_reserve/hierarchical_regression.py',
        wdir='C:/Users/boyler1/Documents/PhD/Code/cognitive_reserve')

model=models[1][1]
result = results.iloc[1]
model.summary()

X = X[1]

saveto = r'C:\Users\boyler1\Documents\PhD\CognitiveReserve\assumptionTest'

saveFolder = r'C:\Users\boyler1\Documents\PhD\CognitiveReserve\assumptionTest'


X = predictors
y = cogFunction
saveFolder = proxyDir