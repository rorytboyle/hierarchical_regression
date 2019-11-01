"""
@author: Rory Boyle rorytboyle@gmail.com github.com/rorytboyle
@date: 31/10/2019 
"""

import pandas as pd
import statsmodels.api as sm
import scipy.stats
import numpy as np
from sklearn.metrics import mean_absolute_error

def robust_reg(y, X, names):
    """
    Runs a linear regression using statsmodels, extracts results from the
    model. Returns a list of results and the OLS results object.

    :param y: outcome variable
    :param X: predictor variable(s)
    :param names: names of predictor variables
    :return list: list of extracted stats/results from statsmodels OLS object
    :return model: OLS results object
    """
    # run robust regression - add column of 1s to X to serve as intercept
    # runs an iteratively reweighted least squares with Tukey's
    # bisquare/biweight function and median absolute deviation scaling (default)
    # This is equivalent to MATLABs robustfit() defaults
    model = sm.RLM(y, sm.add_constant(X),
                   M = sm.robust.norms.TukeyBiweight()).fit()
    
    # calculate MAE of model
    mae = mean_absolute_error(y, model.fittedvalues)

    # extract results from statsmodel OLS object
    results = [names, model.nobs, model.df_resid, model.df_model, mae]
             
    # deep copy names and add constant - otherwise results list will contain
    # multiple repetitions of constant (due to below loop)
    namesCopy = names[:]
    namesCopy.insert(0, 'constant')

    # create dicts with name of each parameter in model (i.e. predictor
    # variables) and the beta coefficient andp-value
    coeffs = {}
    p_values = {}
    for ix, coeff in enumerate(model.params):
        coeffs[namesCopy[ix]] = coeff
        p_values[namesCopy[ix]] = model.pvalues[ix]

    results.append(coeffs)
    results.append(p_values)
    
    return results, model

def calculate_change_stats_robust(model_stats):
    """
    Calculates change in MAE between robust regression models in a hierarchical
    regression.

    :param model_stats: description of parameter x
    
    """
    # get number of steps
    num_steps = model_stats['step'].max()

    # calculate change stats
    MAE_change = [model_stats.iloc[step+1]['mae'] - 
                  model_stats.iloc[step]['mae'] for step in range(0, num_steps-1)]

    return MAE_change

def robust_hierarchical_regression(y, X, names, saveFolder):
    """
    Runs hierarchical linear regressions predicting y from X. Uses statsmodels
    OLS to run linear regression for each step. Returns results of regression
    in each step as well as r-squared change, f change, and p-value of f change
    for the change from step 1 to step 2, step 2 to step 3, and so on.

    The number of lists contained within names specifies the number of steps of
    hierarchical regressions. If names contains two nested lists of strings,
    e.g. if names = [[variable 1], [variable 1, variable 2]], then a two-step
    hierarchical regression will be conducted.

    :param y: outcome variable (dataframe with 1 column)
    :param X: nested lists with each list containing predictor variables for
              each step - if running a two step regression, X should contain
              two lists, each containing a dataframe.
              If Step 1 contains a variable "height", and Step 2 contains
              "height" and "weight", then X should be:
              [[height], [height, weight]]
    :param names: nested lists with each list containing names of predictor
              variables for each step. names should be structured as above.
    :param saveFolder: full path for folder in which to save model results and/
              diagnostics info

    :return: model_stats - a df (rows = number of steps * cols = 18)
    with following info for each step:
        step = step number
        x = predictor names
        num_obs = number of observations in model
        df_resid = df of residuals
        df_mod = df of model
        mae = mean absolute error of model 
        beta_coeff = coefficient values for intercept and predictors
        p_values = p-values for intercept and predictors
        mae_change = change in MAE for model (Step 2 MAE - Step 1 MAE)
    :return reg_models: - a nested list containing the step name of each model
    and the OLS model object 
    """
    # Loop through steps and run regressions for each step
    results = []
    reg_models = []
    for ix, currentX in enumerate(X):

        # run regression - ### CHANGED 19/08 TO ADD IN DIAGNOSTICS
        currentStepResults, currentStepModel = robust_reg(y,
                                                          currentX, names[ix])
        currentStepResults.insert(0, ix+1)  # add step number to results

        saveto = saveFolder + r'\step' + str(ix+1)
        modelSave = saveto + "model.pickle"
        currentStepModel.save(modelSave)

#        # run regression diagnostics - NOT EDITED FOR ROBUST REGRESSION YET
#        assumptionsToCheck = regression_diagnostics(
#                currentStepModel, currentStepResults, y, currentX, saveto)
#        currentStepResults.append(assumptionsToCheck)

        results.append(currentStepResults)

        # add model to list of models along with step number
        reg_models.append(['Step ' + str(ix+1), currentStepModel])
        
    # add results to model_stats dataframe
    model_stats = pd.DataFrame(results)
    model_stats.columns = ['step', 'predictors', 'num_obs', 'df_resid',
                           'df_mod', 'mae', 'beta_coeff',
                           'p_values']

    # calculate r-sq change, f change, p-value of f change
    change_results = calculate_change_stats_robust(model_stats)

    # append step number to change results
    for ix, change_stat in enumerate(change_results):
        change_results[ix] = [ix+2, change_stat]

    # add change results to change_stats dataframe
    change_stats = pd.DataFrame(change_results)
    change_stats.columns = ['step', 'MAE_change']

    # merge model_stats and change_stats
    model_stats = pd.merge(model_stats, change_stats, on='step', how='outer')

    return model_stats, reg_models