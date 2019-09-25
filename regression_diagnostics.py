import statsmodels.api as sm
import statsmodels.stats.outliers_influence as sm_diagnostics
import statsmodels.stats as sm_stats
import matplotlib.pyplot as plt
import scipy as scipy
import pandas as pd
import seaborn as sns
import os

# https://www.statsmodels.org/dev/examples/notebooks/generated/regression_plots.html
# https://towardsdatascience.com/verifying-the-assumptions-of-linear-regression-in-python-and-r-f4cd2907d4c0

def regression_diagnostics(model, result, y, X, saveto):
    """High level description of function COMPLETE THIS COMMENT ###############

    Additional details on function COMPLETE THIS COMMENT ######################

    :param model: regression.linear_model.RegressionResultsWrapper
                  from statsmodels.OLS
    :param result: Series containing extracted results from
                  linear regression. One row of results df
                  returned by hierarchical_regression()
    :param y: COMPLETE THIS COMMENT #########################
    :param X: COMPLETE THIS COMMENT #########################
    :param saveto: folder specifying dir to save results and
                  plots
    :return: COMPLETE THIS COMMENT ##########################
    """
    # get resid vals
    influence_df = sm_diagnostics.OLSInfluence(model).summary_frame()
    # create dict to store diagnostics test info
    diagnostics = {}
    # create dict to link diagnostic tests to assumptions
    assumptionTests = {}
    # create dict with formal names of diagnostic tests - for printing warnings
    formalNames = {}
    # create folder
    os.makedirs(saveto)
    # get step number 
    step = saveto.split("\\")[-1]

# ASSUMPTION 1 - INDEPENDENCE OF RESIDUALS
    # Durbin-Watson stat (no autocorrelation)
    diagnostics['durbin_watson_stat'] = sm_stats.stattools.durbin_watson(
            model.resid, axis=0)
    # Acceptable Durbin-Watson values = 1.5 to 2.5
    if diagnostics['durbin_watson_stat'] >= 1.5 and diagnostics[
            'durbin_watson_stat'] <= 2.5:
        diagnostics['durbin_watson_passed'] = 'Yes'
    else:
        diagnostics['durbin_watson_passed'] = 'No'
    # link test to assumption
    assumptionTests['durbin_watson_passed'] = 'Independence of Residuals'
    formalNames['durbin_watson_passed'] = 'Durbin-Watson Test'

# ASSUMPTION 2 - LINEARITY
    # a) linearity between DV and each IV - Pearson's r
    if len(X.shape) > 1:  # run code if there are multiple predictors
        correlations = [scipy.stats.pearsonr(X[var], y)
                        for var in X.columns]
        for ix, corr in enumerate(correlations):
            xName = 'IV_' + X.columns[ix] + '_pearson_'
            diagnostics[xName + 'r'] = corr[0]
            diagnostics[xName + 'p'] = corr[1]
    else:  # run code if only 1 predictor
        correlations = scipy.stats.pearsonr(X, y)

    # search through dict for keys with pearson_p and assign yes to passed var
    # if all p's < 0.05
    nonSigLinearIV_toDV = 0  # flag
    nonSigLinearVars = []
    for key in diagnostics:
        if key[-9:] == 'pearson_p':
            if diagnostics[key] > 0.05:
                nonSigLinearIV_toDV += 1
                nonSigLinearVars.append(key)

    if nonSigLinearIV_toDV == 0:
        diagnostics['linear_DVandIVs_passed'] = 'Yes'
    else:
        diagnostics['linear_DVandIVs_passed'] = 'No:' + ', '.join(
                nonSigLinearVars)
    # link test to assumption
    assumptionTests['linear_DVandIVs_passed'] = 'Linearity'
    formalNames['linear_DVandIVs_passed'] = 'Non-sig. linear relationship between DV and each IV'

    # b) linearity between DV and IVs collectively
    # Harvey-Collier multiplier test for linearity -
    # null hypo = residuals (and thus the true model) are linear
    diagnostics[
            'harvey_collier_linearity'] = sm_stats.api.linear_harvey_collier(
            model)[1]

    if diagnostics['harvey_collier_linearity'] < 0.05:
        diagnostics['harvey_collier_linearity_passed'] = 'Yes'
    else:
        diagnostics['harvey_collier_linearity_passed'] = 'No'
    # link test to assumption
    assumptionTests['harvey_collier_linearity_passed'] = 'Linearity'
    formalNames['harvey_collier_linearity_passed'] = 'Harvey-Collier Multiplier Test'

    # rainbow test for linearity - null hypo = model has adequate linear fit
    diagnostics['rainbow_linearity'] = sm_stats.diagnostic.linear_rainbow(
            model)[1]

    if diagnostics['rainbow_linearity'] > 0.05:
        diagnostics['rainbow_linearity_passed'] = 'Yes'
    else:
        diagnostics['rainbow_linearity_passed'] = 'No'
    # link test to assumption
    assumptionTests['rainbow_linearity_passed'] = 'Linearity'
    formalNames['rainbow_linearity_passed'] = 'Rainbow Test'

# ASSUMPTION 3 - HOMOSCEDASTICITY
    # https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.het_breuschpagan.html
    breusch_pagan_test = sm_stats.diagnostic.het_breuschpagan(
            model.resid, model.model.exog)
    diagnostics['breusch_pagan_p'] = breusch_pagan_test[1]
    diagnostics['f_test_p'] = breusch_pagan_test[3]

    # if breusch pagan test is sig, then reject null hypo of homoscedasticity
    if diagnostics['breusch_pagan_p'] < .05:
        diagnostics['breusch_pagan_passed'] = 'No'
    else:
        diagnostics['breusch_pagan_passed'] = 'Yes'
    assumptionTests['breusch_pagan_passed'] = 'Homoscedasticity'
    formalNames['breusch_pagan_passed'] = 'Bruesch Pagan Test'

    # if f test is sig, then reject null hypo of homoscedasticity
    # f test more appropriate for for small or moderately large samples
    if diagnostics['f_test_p'] < .05:
        diagnostics['f_test_passed'] = 'No'
    else:
        diagnostics['f_test_passed'] = 'Yes'
    assumptionTests['f_test_passed'] = 'Homoscedasticity'
    formalNames['f_test_passed'] = 'F-test for residual variance'

    # Goldfeld Quandt test
    goldfeld_quandt_test = sm_stats.api.het_goldfeldquandt(
            model.resid, model.model.exog)
    diagnostics['goldfeld_quandt_p'] = goldfeld_quandt_test[1]
    # if goldfeld quandt test is sig, then reject null hypo of homoscedasticity
    if diagnostics['goldfeld_quandt_p'] < .05:
        diagnostics['goldfeld_quandt_passed'] = 'No'
    else:
        diagnostics['goldfeld_quandt_passed'] = 'Yes'
    assumptionTests['goldfeld_quandt_passed'] = 'Homoscedasticity'
    formalNames['goldfeld_quandt_passed'] = 'Goldfeld Quandt Test'

# ASSUMPTION 4 - MULTICOLLINEARITY
    # a) check pairwise correlations < 0.8
    if len(X.shape) > 1:  # run code if there are multiple predictors
        pairwise_corr = X.corr()
        pairwise_corr = pairwise_corr[pairwise_corr != 1]  # make diagonals=nan

        high_pairwise_corr = pairwise_corr[pairwise_corr >= 0.3]
        if high_pairwise_corr.isnull().all().all():
            diagnostics['high_pairwise_correlations_passed'] = 'Yes'
        else:
            diagnostics['high_pairwise_correlations_passed'] = 'No'
    else:  # run code if only 1 predictor
        diagnostics['high_pairwise_correlations_passed'] = 'Yes'

    # link test to assumption
    assumptionTests['high_pairwise_correlations_passed'] = 'Multicollinearity'
    formalNames['high_pairwise_correlations_passed'] = 'High Pairwise correlations'

    # b) Variance Inflation Factors < 10
    if len(X.shape) > 1:  # run code if there are multiple predictors
        vif = pd.DataFrame()
        vif['VIF'] = [sm_stats.outliers_influence.variance_inflation_factor(
                X.values, i) for i in range(X.shape[1])]
        vif['features'] = X.columns

        # if no predictors have vif > 5
        if ((vif['VIF'] < 5).all()):
            diagnostics['VIF_passed'] = 'Yes'
            diagnostics['VIF_predictorsFailed'] = []
        else:
            diagnostics['VIF_passed'] = 'No'
            # add predictor names to diagnostics
            diagnostics['VIF_predictorsFailed'] = vif[vif > 5].to_string(
                    index=False, header=False)       

    else:  # run code if only 1 predictor
        diagnostics['VIF_passed'] = 'Yes'

    # link test to assumption
    assumptionTests['VIF_passed'] = 'Multicollinearity'
    formalNames['VIF_passed'] = 'High Variance Inflation Factor'

# ASSUMPTION 5 - OUTLIERS
    #  no outliers, high leverage pts, or highly influential pts
    # get index of outliers w/ std. resids above/below 3/-3
    highOutliers = influence_df[
            influence_df['standard_resid'] < -3].index.tolist()
    lowOutliers = influence_df[
            influence_df['standard_resid'] > 3].index.tolist()
    diagnostics['outlier_index'] = highOutliers + lowOutliers

    if not diagnostics['outlier_index']:
        diagnostics['outliers_passed'] = 'Yes'
    else:
        diagnostics['outliers_passed'] = 'No'
    # link test to assumption
    assumptionTests['outliers_passed'] = 'Outliers/Leverage/Influence'
    formalNames['outliers_passed'] = 'Extreme Standardised Residuals'

    # influence = Cook's Distance
    # https://www.researchgate.net/publication/2526564_A_Teaching_Note_on_Cook's_Distance_-_A_Guideline
    if len(X.shape) == 1:
        cooks_cutOff = 0.7  # cut off for 1 predictor = Cooks > 0.7 (n>15)
    elif X.shape[1] == 2:
        cooks_cutOff = 0.8  # cut off for 2 predictors = Cooks > 0.8 (n>15)
    elif X.shape[1] > 2:
        cooks_cutOff = 0.85  # cut off for >2 predictors = Cooks > 0.85 (n>15)

    diagnostics['influence_largeCooksD_index'] = influence_df[
            influence_df['cooks_d'] > cooks_cutOff].index.tolist()

    if not diagnostics['influence_largeCooksD_index']:
        diagnostics['influence_passed'] = 'Yes'
    else:
        diagnostics['influence_passed'] = 'No'
    # link test to assumption
    assumptionTests['influence_passed'] = 'Outliers/Leverage/Influence'
    formalNames['influence_passed'] = "Large Cook's Distance"

# ASSUMPTION 6 - NORMALITY
    #  normal distribution of residuals
    # check mean is 0 ( < 0.1 & > -0.1) and errors approx normally distributed
    diagnostics['meanOfResiduals'] = model.resid.mean()
    if diagnostics['meanOfResiduals'] < .1 and diagnostics[
            'meanOfResiduals'] > -.1:
        diagnostics['meanOfResiduals_passed'] = 'Yes'
    else:
        diagnostics['meanOfResiduals_passed'] = 'No'
    # link test to assumption
    assumptionTests['meanOfResiduals_passed'] = 'Normality'
    formalNames['meanOfResiduals_passed'] = "Mean of residuals not approx = 0"

    # Shapiro-Wilk test on residuals
    diagnostics['shapiroWilks_p'] = scipy.stats.shapiro(model.resid)[1]
    if diagnostics['shapiroWilks_p'] > 0.05:
        diagnostics['shapiroWilks_passed'] = 'Yes'
    else:
        diagnostics['shapiroWilks_passed'] = 'No'
    # link test to assumption
    assumptionTests['shapiroWilks_passed'] = 'Normality'
    formalNames['shapiroWilks_passed'] = 'Shapiro-Wilk Test'

# SUMMARISE DIAGNOSTIC TEST INFO
    # check whether diagnostic tests are passed. If all tests passed, then
    # print message telling user that model is ok. If any test failed, print
    # message telling user that model may not satisfy assumptions, check plots,
    # and investigate further.
    diagnostic_tests = 0
    diagnosticsPassed = 0
    violated = []

    print('\n\n\nDiagnostic summary for: ' + step)
    for key in diagnostics:
        if key[-6:] == 'passed':
            diagnostic_tests += 1
            if diagnostics[key] == 'Yes':
                diagnosticsPassed += 1
            else:
                # find which assumption diagnostic test referred to AND print
                # message telling user to investigate assumption further
                print('Diagnostic test (' + formalNames[key] +
                      ') failed for ' + assumptionTests[key])
                # add assumption to possible violations
                violated.append(assumptionTests[key])

    # summarise how many tests passed/failed for each assumption
    assumptionList = [i for i in assumptionTests.values()]
    assumptions = list(set(assumptionList))

    summaryTextList = []
    summarySentence = ' diagnostic tests passed for assumption - '

    for assumption in assumptions:
        testsFailed = violated.count(assumption)
        testsPerformed = assumptionList.count(assumption)
        testsPassed = testsPerformed - testsFailed

        sentence = str(testsPassed) + '/' + str(
                testsPerformed) + summarySentence + assumption
        if testsFailed > 0:
            print("\nFURTHER INSPECTION REQUIRED - CHECK PLOTS + DATA \n" +
                  sentence)
        summaryTextList.append(sentence)

# SAVE DIAGNOSTICS INFO AND SUMMARY OF DIAGNOSTIC TESTS
    # write out text file with summary of tests
    summaryFile = saveto + '\\' + step + '_testSummary.txt'
    with open(summaryFile, 'w') as f:
        for item in summaryTextList:
            f.write("%s\n" % item)

    csvName = saveto + '\\' + step + '_diagnostic_results.csv'
    # saves a csv with 28 rows and two columns (long but easily readable)
    pd.Series(diagnostics).to_csv(csvName)
    # Alternative code
    # saves a csv with 1 row and 28 columns (wide but not easily readable)
#    pd.DataFrame.from_dict(
#            diagnostics, orient='index').transpose().to_csv(csvName)
    # requires a roundabout way of creating the dataframe as regular method of
    # pd.DataFrame.from_dict(diagnostics) creates an empty df

    # save csv of pairwise correlations - only if there are multiple predictors
    if len(X.shape) > 1:
        pairwiseCorrName = saveto + '\\' + step + '_pairwise_correlations.csv'
        high_pairwise_corr.to_csv(pairwiseCorrName)

# MAKE AND SAVE PLOTS

# PLOT 1 - STUDENTISED RESIDUALS VS FITTED VALUES
# Used to inspect linearity and homoscedasticity
    # get values
    student_resid = influence_df['student_resid']
    fitted_vals = model.fittedvalues
    # plot with a LOWESS (Locally Weighted Scatterplot Smoothing) line
    # a relativelty straight LOWESS line indicates a linear model is reasonable
    residsVsFittedVals_plot = plt.figure()
    residsVsFittedVals_plot.axes[0] = sns.residplot(
            fitted_vals, student_resid, lowess=True,
            scatter_kws={'alpha': 0.5},
            line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
    residsVsFittedVals_plot.axes[0].set(ylim=(-3.5, 3.5))
    residsVsFittedVals_plot.axes[0].set_title('Residuals vs Fitted')
    residsVsFittedVals_plot.axes[0].set_xlabel('Fitted values')
    residsVsFittedVals_plot.axes[0].set_ylabel('Studentised Residuals')
    # name + save plot
    figName = saveto + '\\' + step + '_residualsVSfittedValuesPlot.png'
    residsVsFittedVals_plot.savefig(figName)
    plt.clf()

# PLOT 2 - NORMAL QQ PLOT OF RESIDUALS
# Used to inspect normality
    qq_fig = sm.qqplot(model.resid, fit=True, line='45')
    qq_fig.axes[0].set_title('Normal QQ Plot of Residuals')
    # name + save plot
    figName = saveto + '\\' + step + '_NormalQQPlot.png'
    qq_fig.savefig(figName)
    plt.clf()

# PLOT 3 - INFLUENCE PLOT WITH COOK'S DISTANCE
# Used to inspect influence
    # Outliers/Leverage/Influence - Influence plot w/ Cook's Distance & boxplot
    # https://www.statsmodels.org/stable/generated/statsmodels.stats.outliers_influence.OLSInfluence.plot_influence.html#statsmodels.stats.outliers_influence.OLSInfluence.plot_influence
    fig_influence, ax_influence = plt.subplots(figsize=(12, 8))
    fig_influence = sm.graphics.influence_plot(model, ax=ax_influence,
                                               criterion="cooks")
    # name + save plot
    figName = saveto + '\\' + step + '_InfluencePlot_CooksD.png'
    fig_influence.savefig(figName)
    plt.clf()

# PLOT 4 - BOX PLOT OF STANDARDISED RESIDUALS
# Used to inspect outliers (residuals)
    outlier_fig = sns.boxplot(y=influence_df['standard_resid'])
    outlier_fig = sns.swarmplot(y=influence_df['standard_resid'], color="red")
    outlier_fig.axes.set(ylim=(-3.5, 3.5))
    outlier_fig.axes.set_title('Boxplot of Standardised Residuals')
    residBoxplot = outlier_fig.get_figure()  # get figure to save
    # name + save plot
    figName = saveto + '\\' + step + '_ResidualsBoxplot.png'
    residBoxplot.savefig(figName)
    plt.clf()

# PLOT 5 - PARTIAL REGRESSION PLOTS
# Used to inspect linearity
    # Partial regression plots
    fig_partRegress = plt.figure(figsize=(12, 8))
    fig_partRegress = sm.graphics.plot_partregress_grid(model,
                                                        fig=fig_partRegress)
    # name + save plot
    figName = saveto + '\\' + step + '_PartialRegressionPlots.png'
    fig_partRegress.savefig(figName)
    plt.clf()
