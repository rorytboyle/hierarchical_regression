import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def make_interaction_plot(y, x, z, title, saveFolder):
    """ Creates interaction plots to visualise moderation effect of a variable
        Z on the X-Y relationship. Creates two groups in both X and Y based on
        a median split of Z. Plots each X on x-axis and Y on y-axis with
        separate colours and regression lines for the two groups created after
        splitting Z.
        Note: y, x, and z should all be pandas dataframes. This function labels
        the axes with the column name in the dataframe.
        @author: Rory Boyle rorytboyle@gmail.com
        @date: 02/10/2019

    :param y: outcome variable (dataframe)
    :param x: predictor variable (dataframe)
    :param z: interaction variable - the variable to median split
    :param title: title for plot
    :param saveFolder: full path for folder in which to save plots
    """
    # Perform median split of z
    z_high = z[z >= z.median()].dropna()
    z_low = z[z < z.median()].dropna()

    # Split y based on z and convert to series
    y_high = y[y.index.isin(z_high.index)].iloc[:,0]
    y_low = y[y.index.isin(z_low.index)].iloc[:,0]

    # Split x based on z and convert to series
    x_high = x[x.index.isin(z_high.index)].iloc[:,0]
    x_low = x[x.index.isin(z_low.index)].iloc[:,0]
    
    # set seaborn to default
    sns.set()
    
    # plot values
    fig, ax = plt.subplots(1,1)
    sns.regplot(x_low, y_low, x_ci ='ci', color='purple', marker='^',
                label='Low CR', ax=ax)
    sns.regplot(x_high, y_high, x_ci ='ci', color='green', marker='o',
                label='High CR', ax=ax).set_title(col)
    ax.legend()
    
    # save figure
    filePath = saveFolder + '\\' + 'interactionPlot_'+col+'.png'
    ax.get_figure().savefig(filePath)    
    plt.clf()